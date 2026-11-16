"""Protocolo: búsqueda en el pliegue 2018, finalistas en 2019-2021 y prueba retrospectiva.

Cada entrenamiento escribe su carpeta con ``resultado.json`` al terminar; si la carpeta ya lo
contiene, no se repite. Por eso el proceso se puede interrumpir y reanudar.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from demanda import metricas as MET

from . import motor
from .datos import DatosNumericos, escalas_a_json
from . import redes


SIN_MEMORIA = getattr(torch, "OutOfMemoryError", None) or torch.cuda.OutOfMemoryError


def guardar_json(ruta: Path, obj) -> None:
    ruta = Path(ruta)
    tmp = ruta.with_suffix(ruta.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    os.replace(tmp, ruta)


def leer_json(ruta: Path):
    return json.loads(Path(ruta).read_text(encoding="utf-8"))


class Contexto:
    """Estado compartido: datos, texto, configuración, dispositivo, bitácora y barra de avance."""

    def __init__(self, raiz: Path, salida: Path, cfg: dict, datos: DatosNumericos, dispositivo: str, log, rapido: dict | None = None):
        self.raiz, self.salida, self.cfg, self.datos, self.dispositivo, self.log = Path(raiz), Path(salida), cfg, datos, dispositivo, log
        self.rapido = rapido
        self.texto = None        # dict(vectores, lista) en CPU; cada red los copia a su dispositivo
        self.descripciones = None  # (n_canales, 768) solo en TGForecaster
        self.barra = None
        self._cache = (None, None)
        self.opciones = dict(cfg["entrenamiento"])
        if rapido:
            self.opciones.update(max_epocas=rapido["max_epocas"], min_epocas=1)

    # ------------------------------------------------------------------ datos por pliegue
    def recortar(self, pos: dict) -> dict:
        """En la prueba rápida se usan pocos orígenes; en la corrida real no cambia nada."""
        if not self.rapido:
            return pos
        r = self.rapido
        return dict(tr=pos["tr"][-r["n_tr"]:], es=pos["es"][-r["n_es"]:], ev=pos["ev"][:r["n_ev"]])

    def conjuntos(self, clave: str, pos: dict):
        if self._cache[0] != clave:
            esc = self.datos.ajustar_escalas(pos["tr"])
            self._cache = (clave, (esc, {k: self.datos.construir(pos[k], esc) for k in ("tr", "es", "ev")}))
        return self._cache[1]


def _evaluar(datos: DatosNumericos, red, conj, t) -> tuple[np.ndarray, dict]:
    yhat = motor.predecir(red, t).astype("float64") * conj.nivel[:, None]
    return yhat, MET.evaluar(conj.y, yhat, datos.q168)


def corrida(ctx: Contexto, variante: str, p: dict, semilla: int, clave_datos: str, pos: dict, carpeta: Path, nombre_ev: str,
            guardar_modelo: bool = False) -> dict:
    """Entrena una configuración con una semilla y evalúa en ``pos['ev']``. Reanudable."""
    carpeta = Path(carpeta)
    rr = carpeta / "resultado.json"
    etiqueta = f"{variante} cfg {p['id']:02d} semilla {semilla} {nombre_ev}"
    if rr.exists():
        if ctx.barra is not None:
            ctx.barra.update(1)
        return leer_json(rr)
    carpeta.mkdir(parents=True, exist_ok=True)
    datos = ctx.datos
    esc, conj = ctx.conjuntos(clave_datos, pos)
    texto = None
    if variante == "N+O":
        texto = dict(vectores=torch.from_numpy(ctx.texto["vectores"]), lista=torch.from_numpy(ctx.texto["lista"]))
    desc = torch.from_numpy(ctx.descripciones) if ctx.descripciones is not None else None
    dispositivos = [ctx.dispositivo] + (["cpu"] if ctx.dispositivo != "cpu" else [])
    for intento, dev in enumerate(dispositivos):
        try:
            motor.fijar_semilla(semilla)
            red = redes.construir(ctx.cfg["paquete"]["red"], p, conj["tr"].x.shape[2], conj["tr"].cal_p.shape[2], datos.L, datos.H, texto, desc).to(dev)
            tens = {k: motor.a_tensores(v, dev) for k, v in conj.items()}

            def observador(h, _e=etiqueta):
                if ctx.barra is not None:
                    ctx.barra.set_postfix_str(f"{_e} | época {h['epoca']} | parada {h['perdida_parada']:.5f}", refresh=True)

            t0 = time.time()
            info = motor.entrenar(red, tens["tr"], tens["es"], p, semilla, ctx.opciones, observador)
            yhat, met = _evaluar(datos, red, conj["ev"], tens["ev"])
            mase_tr = _evaluar(datos, red, conj["tr"], tens["tr"])[1]["MASE"]
            mase_es = _evaluar(datos, red, conj["es"], tens["es"])[1]["MASE"]
            break
        except SIN_MEMORIA:
            if intento == len(dispositivos) - 1:
                raise
            ctx.log(f"Memoria de GPU insuficiente en {etiqueta}; esta corrida se repite en CPU.")
            red = tens = None
            torch.cuda.empty_cache()
    res = dict(variante=variante, configuracion=p, semilla=int(semilla), evaluacion=nombre_ev, dispositivo=str(dev),
               n_parametros=red.n_parametros(), epocas=info["epocas"], mejor_epoca=info["mejor_epoca"], mejor_perdida_parada=info["mejor_perdida_parada"],
               segundos=round(time.time() - t0, 1), n_ajuste=len(conj["tr"]), n_parada=len(conj["es"]), n_evaluacion=len(conj["ev"]),
               origen_ajuste=[str(conj["tr"].origen[0]), str(conj["tr"].origen[-1])], origen_parada=[str(conj["es"].origen[0]), str(conj["es"].origen[-1])],
               origen_evaluacion=[str(conj["ev"].origen[0]), str(conj["ev"].origen[-1])],
               filas_escalas=[esc["inicio_ajuste"], esc["fin_ajuste"]], MASE_ajuste=mase_tr, MASE_parada=mase_es, metricas=met,
               mase_por_horizonte=[float(v) for v in MET.mase_por_horizonte(conj["ev"].y, yhat, datos.q168)], diagnostico=red.diagnostico())
    pd.DataFrame(info["historia"]).to_csv(carpeta / "historia.csv", index=False)
    np.savez_compressed(carpeta / f"pred_{nombre_ev}.npz", origen=np.asarray(conj["ev"].origen.strftime("%Y-%m-%d")), yhat_kwh=yhat.astype("float32"),
                        y_kwh=conj["ev"].y.astype("float32"), mase_por_origen=datos.mase_por_origen(conj["ev"].y, yhat))
    if guardar_modelo:
        torch.save({k: v.cpu() for k, v in red.state_dict().items()}, carpeta / "modelo.pt")
        guardar_json(carpeta / "escaladores.json", escalas_a_json(esc, datos))
    guardar_json(rr, res)
    del red, tens
    if ctx.barra is not None:
        ctx.barra.update(1)
    ctx.log(f"{etiqueta}: MASE {met['MASE']:.4f} | épocas {info['epocas']} | {res['segundos']:.0f} s | {dev}", consola=False)
    return res


# ------------------------------------------------------------------------------- etapas
def _fila(r: dict) -> dict:
    p = r["configuracion"]
    return dict(variante=r["variante"], **p, semilla=r["semilla"], evaluacion=r["evaluacion"],
                MASE=r["metricas"]["MASE"], MAE_kwh=r["metricas"]["MAE"], RMSE_kwh=r["metricas"]["RMSE"], MAPE=r["metricas"]["MAPE"],
                MASE_ajuste=r["MASE_ajuste"], MASE_parada=r["MASE_parada"], n_parametros=r["n_parametros"], epocas=r["epocas"],
                mejor_epoca=r["mejor_epoca"], segundos=r["segundos"], dispositivo=r["dispositivo"], **{"diag_" + k: v for k, v in r.get("diagnostico", {}).items()})


def _pos_pliegue(ctx: Contexto, f: dict) -> dict:
    return ctx.recortar(dict(tr=f["tr"], es=f["es"], ev=f["val"]))


def etapa_busqueda(ctx: Contexto, configs: list[dict]) -> dict:
    """Pliegue inicial: todas las configuraciones con una semilla; pasan las mejores de cada variante."""
    b = ctx.cfg["busqueda"]
    f = next(x for x in ctx.datos.pliegues() if x["anio"] == b["pliegue_inicial"])
    pos, filas = _pos_pliegue(ctx, f), []
    for variante in ctx.cfg["variantes"]:
        for p in configs:
            carpeta = ctx.salida / "busqueda" / variante.replace("+", "_") / f"pliegue_{f['anio']}" / f"cfg_{p['id']:02d}"
            filas.append(_fila(corrida(ctx, variante, p, b["semilla"], f"pliegue_{f['anio']}", pos, carpeta, f"val_{f['anio']}")))
    t = pd.DataFrame(filas).sort_values(["variante", "MASE", "n_parametros", "id"]).reset_index(drop=True)
    t["puesto"] = t.groupby("variante").cumcount() + 1
    t["finalista"] = t["puesto"] <= b["finalistas"]
    t.to_csv(ctx.salida / "busqueda" / "tabla_pliegue_inicial.csv", index=False)
    finalistas = {v: t.loc[(t.variante == v) & t.finalista, "id"].astype(int).tolist() for v in ctx.cfg["variantes"]}
    guardar_json(ctx.salida / "busqueda" / "finalistas.json", finalistas)
    return finalistas


def etapa_pliegues(ctx: Contexto, configs: list[dict], finalistas: dict) -> dict:
    """Pliegues restantes para las finalistas; se elige la de menor MASE medio en los cuatro pliegues."""
    b = ctx.cfg["busqueda"]
    por_id = {p["id"]: p for p in configs}
    filas = pd.read_csv(ctx.salida / "busqueda" / "tabla_pliegue_inicial.csv")
    filas = filas.loc[filas.finalista].drop(columns=["puesto", "finalista"]).to_dict("records")
    for f in ctx.datos.pliegues():
        if f["anio"] not in b["pliegues_restantes"]:
            continue
        pos = _pos_pliegue(ctx, f)
        for variante in ctx.cfg["variantes"]:
            for i in finalistas[variante]:
                carpeta = ctx.salida / "busqueda" / variante.replace("+", "_") / f"pliegue_{f['anio']}" / f"cfg_{i:02d}"
                filas.append(_fila(corrida(ctx, variante, por_id[i], b["semilla"], f"pliegue_{f['anio']}", pos, carpeta, f"val_{f['anio']}")))
    t = pd.DataFrame(filas)
    t.to_csv(ctx.salida / "busqueda" / "tabla_finalistas_por_pliegue.csv", index=False)
    r = (t.groupby(["variante", "id"]).agg(MASE_medio=("MASE", "mean"), MASE_desv_pliegues=("MASE", "std"), MASE_min=("MASE", "min"), MASE_max=("MASE", "max"),
                                             pliegues=("MASE", "size"), n_parametros=("n_parametros", "first")).reset_index()
         .sort_values(["variante", "MASE_medio", "n_parametros", "id"]).reset_index(drop=True))
    r["puesto"] = r.groupby("variante").cumcount() + 1
    r.to_csv(ctx.salida / "busqueda" / "resumen_finalistas.csv", index=False)
    elegidas = {v: int(r.loc[(r.variante == v) & (r.puesto == 1), "id"].iloc[0]) for v in ctx.cfg["variantes"]}
    guardar_json(ctx.salida / "busqueda" / "configuracion_elegida.json",
                 {v: dict(configuracion=por_id[i], MASE_medio_validacion=float(r.loc[(r.variante == v) & (r.id == i), "MASE_medio"].iloc[0])) for v, i in elegidas.items()})
    return elegidas


def bootstrap_bloques(dif: np.ndarray, base: np.ndarray, bloque: int, remuestreos: int, semilla: int) -> dict:
    """Bootstrap circular por bloques de días para la diferencia pareada por origen (A - B) y su valor relativo a B."""
    n = len(dif)
    g = np.random.default_rng(semilla)
    nb = int(np.ceil(n / bloque))
    d_obs, b_obs = float(dif.mean()), float(base.mean())
    md, mb = np.empty(remuestreos), np.empty(remuestreos)
    for a in range(0, remuestreos, 500):
        k = min(500, remuestreos - a)
        ix = ((g.integers(0, n, size=(k, nb))[:, :, None] + np.arange(bloque)[None, None, :]) % n).reshape(k, -1)[:, :n]
        md[a:a + k], mb[a:a + k] = dif[ix].mean(axis=1), base[ix].mean(axis=1)
    rel = 100 * md / mb
    return dict(bloque_dias=int(bloque), remuestreos=int(remuestreos), diferencia_MASE=d_obs, ic95_diferencia_MASE=[float(v) for v in np.percentile(md, [2.5, 97.5])],
                diferencia_relativa_pct=100 * d_obs / b_obs, ic95_diferencia_relativa_pct=[float(v) for v in np.percentile(rel, [2.5, 97.5])],
                p_bilateral_bootstrap=float((np.abs(md - d_obs) >= abs(d_obs)).mean()))


def etapa_final(ctx: Contexto, configs: list[dict], elegidas: dict) -> dict:
    """Ajuste final con los datos anteriores a 2022 y evaluación retrospectiva 2022-2025."""
    datos, por_id = ctx.datos, {p["id"]: p for p in configs}
    tr, es = datos.ajuste_final()
    pos = ctx.recortar(dict(tr=tr, es=es, ev=datos.prueba()))
    semillas = ctx.rapido["semillas_finales"] if ctx.rapido else ctx.cfg["final"]["semillas"]
    filas, perdidas = [], {}
    for variante in ctx.cfg["variantes"]:
        p = por_id[elegidas[variante]]
        for s in semillas:
            carpeta = ctx.salida / "final" / variante.replace("+", "_") / f"semilla_{s}"
            r = corrida(ctx, variante, p, s, "final", pos, carpeta, "prueba", guardar_modelo=True)
            filas.append(_fila(r))
            perdidas.setdefault(variante, []).append(np.load(carpeta / "pred_prueba.npz")["mase_por_origen"])
    t = pd.DataFrame(filas)
    t.to_csv(ctx.salida / "final" / "metricas_por_semilla.csv", index=False)
    y = datos.Y[pos["ev"][:, None] + np.arange(datos.H)]
    origen = datos.indice[pos["ev"]].strftime("%Y-%m-%d")
    ing = {k: datos.mase_por_origen(y, v) for k, v in datos.ingenuos(pos["ev"]).items()}
    media = {v: np.mean(perdidas[v], axis=0) for v in perdidas}          # pérdida por origen promediada sobre semillas
    pd.DataFrame(dict(origen=origen, **{f"MASE_{v}": media[v] for v in media}, **{f"MASE_{k}": ing[k] for k in ing})).to_csv(ctx.salida / "final" / "perdida_por_origen.csv", index=False)
    por_h = {}
    for variante in ctx.cfg["variantes"]:
        hs = [leer_json(ctx.salida / "final" / variante.replace("+", "_") / f"semilla_{s}" / "resultado.json")["mase_por_horizonte"] for s in semillas]
        por_h[variante] = np.mean(hs, axis=0)
    pd.DataFrame(dict(horizonte_h=np.arange(1, datos.H + 1), **{f"MASE_{v}": por_h[v] for v in por_h})).to_csv(ctx.salida / "final" / "mase_por_horizonte.csv", index=False)

    resumen = dict(n_origenes_prueba=int(len(pos["ev"])), periodo_prueba=[origen[0], origen[-1]], semillas=list(map(int, semillas)), q168_kwh=datos.q168,
                   referencias={k: float(v.mean()) for k, v in ing.items()}, variantes={})
    for variante in ctx.cfg["variantes"]:
        m = t.loc[t.variante == variante, "MASE"].to_numpy()
        resumen["variantes"][variante] = dict(configuracion=por_id[elegidas[variante]], MASE_media_semillas=float(m.mean()),
                                              MASE_desv_semillas=float(m.std(ddof=1)) if len(m) > 1 else None, MASE_por_semilla=[float(v) for v in m],
                                              MAE_kwh_media=float(t.loc[t.variante == variante, "MAE_kwh"].mean()), MAPE_media=float(t.loc[t.variante == variante, "MAPE"].mean()))
    if {"N", "N+O"} <= set(media):
        inf = ctx.cfg["inferencia"]
        rem = ctx.rapido["remuestreos"] if ctx.rapido else inf["remuestreos"]
        dif = media["N+O"] - media["N"]
        resumen["contraste_N+O_menos_N"] = dict(
            definicion="diferencia pareada por origen de la pérdida MASE, promediada sobre semillas; relativa = 100 (MASE_N+O - MASE_N) / MASE_N; negativo favorece al texto",
            margen_practico_pct=100 * inf["margen_relativo"], primario=bootstrap_bloques(dif, media["N"], inf["bloque_dias"], rem, inf["semilla_bootstrap"]),
            sensibilidad=[bootstrap_bloques(dif, media["N"], b, rem, inf["semilla_bootstrap"]) for b in inf["sensibilidad_bloque_dias"]],
            nota="Resultado descriptivo de este modelo. La inferencia del proyecto aplica Holm a la familia preregistrada de contrastes; el valor p de aquí no está ajustado.")
    guardar_json(ctx.salida / "final" / "resumen_prueba.json", resumen)
    return resumen

"""Exploración de diseño solo numérica, en la partición de desarrollo (pliegues 2018 y 2019). No lee la prueba.

Etapas (reanudables: una carpeta con ``resultado.json`` no se repite):
  busqueda_periodica el codificador periódico (idea I) recibe la búsqueda corta de la familia (pliegue 2018 y finalistas en
                     2019-2021, semilla 43), como la tuvieron las otras tres bases en el Entregable 2 y en la parte 1
  exportabilidad     cada base, con y sin mecanismos, se exporta a ONNX (pesos perturbados, sin entrenar) y se compara con
                     una entrada distinta de la de trazado
  rejilla            4 bases x 6 variantes x 2 pliegues x 5 semillas (101-105); guarda las predicciones de validación y de
                     la parada interna
  agregacion         idea J sobre las predicciones guardadas (sin entrenar): dos conjuntos de expertos, cinco referencias
  decision           regla preregistrada (plan_exploracion.json) -> decision.json, design_log.md, RESUMEN_EXPLORACION.md
  sensibilidad       hiperparámetros del mecanismo ganador, uno a la vez (3 semillas, los dos pliegues); descriptiva
"""
from __future__ import annotations

import datetime as dt
import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats

from demanda import metricas as MET
from entrenamiento import motor
from entrenamiento import protocolo as PR
from entrenamiento import protocolo_e2 as P2
from entrenamiento import redes_e3
from entrenamiento.protocolo import guardar_json, leer_json

from . import agregacion as AG
from . import mecanismos as EX


def _hoy() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


class Exploracion:
    def __init__(self, raiz: Path, salida: Path, plan: dict, datos, dispositivo: str, log, rapido: dict | None, args):
        self.raiz, self.salida, self.plan, self.datos = Path(raiz), Path(salida), plan, datos
        self.dispositivo, self.log, self.rapido, self.args = dispositivo, log, rapido, args
        self.barra = None

    # ================================================================== utilidades
    def semillas(self) -> list[int]:
        return list(self.rapido["semillas_diseno"]) if self.rapido else list(self.plan["semillas"])

    def bases(self) -> list[str]:
        sel = getattr(self.args, "bases", None)
        return [b for b in self.plan["bases"] if not sel or b in sel]

    def ctx(self, carpeta: Path, clave: str, red: str, p: dict) -> PR.Contexto:
        c = self.plan["comun"]
        busq = dict(c["busqueda"], finalistas=self.plan["bases"].get("periodica", {}).get("finalistas", 3))
        if self.rapido:
            busq["finalistas"] = self.rapido["finalistas"]
        cfg = dict(version=self.plan["version"], paquete=dict(clave=clave, nombre=clave, red="x_" + red, carpeta_resultados=Path(carpeta).name),
                   variantes=["N"], busqueda=busq, final=dict(semillas=[]), entrenamiento=dict(c["entrenamiento"]), inferencia={})
        x = PR.Contexto(self.raiz, carpeta, cfg, self.datos, self.dispositivo, self.log, self.rapido)
        if self.rapido:
            x.opciones["lote_evaluacion"] = self.rapido.get("lote_evaluacion", 16)
        x.h_cal = EX.horizonte_calendario(p, self.datos.H)
        x.hilos, x.mlflow, x.mlflow_raiz, x.barra = self.args.hilos, not self.args.sin_mlflow, self.salida, self.barra
        return x

    def config_base(self, base: str) -> dict:
        b = self.plan["bases"][base]
        if "config" in b:
            return dict(b["config"])
        r = self.salida / "busqueda_periodica" / "busqueda" / "configuracion_elegida.json"
        if not r.exists():
            raise RuntimeError("El codificador periódico no tiene configuración: ejecute antes la etapa busqueda_periodica")
        return dict(leer_json(r)["N"]["configuracion"])

    def config_celda(self, base: str, mec: str, extra: dict | None = None) -> dict:
        p = self.config_base(base)
        p.update(self.plan["mecanismos"][mec]["parametros"])
        p.update(extra or {})
        if self.rapido and "rec_inicio" in p:          # la prueba rápida tiene 2 épocas: el SWA debe activarse igual
            p["rec_inicio"] = 1
        return p

    def _fila(self, r: dict, **k) -> dict:
        return dict(**k, MASE=r["metricas"]["MASE"], MASE_ajuste=r["MASE_ajuste"], MASE_parada=r["MASE_parada"], n_parametros=r["n_parametros"],
                    epocas=r["epocas"], mejor_epoca=r["mejor_epoca"], segundos=r["segundos"], **{"diag_" + a: v for a, v in r.get("diagnostico", {}).items()})

    # ================================================================== etapas
    def exportabilidad(self) -> dict:
        """ONNX sin entrenar: se traza con una entrada y se compara con otra distinta (detecta periodos fijados al trazar)."""
        import onnxruntime as ort
        out = {}
        d = self.datos
        n_x, n_cal = d.X.shape[1] + d.M.shape[1], d.K.shape[1]
        g = torch.Generator().manual_seed(7)
        for base in self.bases():
            for mec in [m for m in self.plan.get("exportabilidad", ["control", "coh", "dro"]) if m in self.plan["mecanismos"]]:
                clave = f"{base}·{mec}"
                ruta = self.salida / "exportabilidad" / f"{base}_{mec}.onnx"
                try:
                    p = self.config_celda(base, mec)
                except RuntimeError as e:
                    out[clave] = dict(exportable=None, motivo=str(e))
                    continue
                hc = EX.horizonte_calendario(p, d.H)
                torch.manual_seed(0)
                red = redes_e3.construir("x_" + self.plan["bases"][base]["red"], p, n_x, n_cal, d.L, d.H, n_mascaras=len(d.nombres_mascara)).eval()
                with torch.no_grad():                       # pesos perturbados: una cabeza iniciada en cero haría trivial la comparación
                    for w in red.parameters():
                        w.add_(0.05 * torch.randn(w.shape, generator=g))

                def entrada(escala):
                    return (escala * torch.randn(4, d.L, n_x, generator=g), torch.randn(4, d.L, n_cal, generator=g), torch.randn(4, hc, n_cal, generator=g),
                            torch.arange(4, dtype=torch.int64) + 100)
                a, b = entrada(1.0), entrada(3.0)
                try:
                    ruta.parent.mkdir(parents=True, exist_ok=True)
                    torch.onnx.export(red, a, str(ruta), input_names=["x", "cal_p", "cal_f", "dia"], output_names=["y"], opset_version=17,
                                      dynamic_axes={k: {0: "lote"} for k in ("x", "cal_p", "cal_f", "dia", "y")}, dynamo=False)
                    s = ort.InferenceSession(str(ruta), providers=["CPUExecutionProvider"])
                    nombres = {i.name for i in s.get_inputs()}
                    errs = []
                    for e in (a, b):
                        feed = {k: v.numpy() for k, v in zip(["x", "cal_p", "cal_f", "dia"], e) if k in nombres}
                        with torch.no_grad():
                            errs.append(float(np.abs(s.run(None, feed)[0] - red(*e).numpy()).max()))
                    out[clave] = dict(exportable=bool(max(errs) < 1e-4), error_max_entrada_de_trazado=errs[0], error_max_otra_entrada=errs[1])
                except Exception as e:                      # se documenta, no detiene la exploración
                    out[clave] = dict(exportable=False, motivo=f"{type(e).__name__}: {str(e)[:300]}")
                ruta.unlink(missing_ok=True)
                self.log(f"EXPORTABILIDAD {clave}: {out[clave]}", consola=False)
        guardar_json(self.salida / "exportabilidad" / "resultado.json", out)
        return out

    def busqueda_periodica(self) -> dict:
        b = self.plan["bases"]["periodica"]
        cfgs = [c for c in b["configuraciones"] if not self.rapido or c["id"] in self.rapido["configuraciones"]]
        ctx = self.ctx(self.salida / "busqueda_periodica", "periodica", "periodica", cfgs[0])
        fin = PR.etapa_busqueda(ctx, cfgs)
        eleg = PR.etapa_pliegues(ctx, cfgs, fin)
        self.log(f"BÚSQUEDA PERIÓDICA: configuración {eleg['N']}.")
        return leer_json(self.salida / "busqueda_periodica" / "busqueda" / "configuracion_elegida.json")["N"]

    def _celda(self, base: str, mec: str, anio: int, s: int, raiz: str = "rejilla", extra: dict | None = None, nombre: str | None = None) -> dict:
        p = self.config_celda(base, mec, extra)
        car = self.salida / raiz / base / (nombre or mec) / f"pliegue_{anio}" / f"semilla_{s}"
        ctx = self.ctx(car.parent, f"{base}_{nombre or mec}", self.plan["bases"][base]["red"], p)
        pos = P2._pos(ctx, anio)
        r = PR.corrida(ctx, "N", p, s, f"pliegue_{anio}", pos, car, f"val_{anio}", guardar_parada=True)
        return self._fila(r, base=base, mecanismo=nombre or mec, celda=f"{base}·{nombre or mec}", pliegue=anio, semilla=s)

    def rejilla(self) -> pd.DataFrame:
        filas = []
        for anio in self.plan["pliegues"]:
            for base in self.bases():
                for mec in self.plan["mecanismos"]:
                    for s in self.semillas():
                        filas.append(self._celda(base, mec, anio, s))
                        pd.DataFrame(filas).to_csv(self.salida / "rejilla" / "desarrollo_por_semilla_parcial.csv", index=False)
        t = pd.DataFrame(filas)
        t.to_csv(self.salida / "rejilla" / "desarrollo_por_semilla.csv", index=False)
        return t

    def _pred(self, base: str, mec: str, anio: int, s: int, cual: str):
        car = self.salida / "rejilla" / base / mec / f"pliegue_{anio}" / f"semilla_{s}"
        z = np.load(car / (f"pred_val_{anio}.npz" if cual == "val" else "pred_parada.npz"))
        return z["origen"], z["yhat_kwh"].astype("float64"), z["y_kwh"].astype("float64")

    def agregacion(self) -> pd.DataFrame:
        a = self.plan["agregacion"]
        conjuntos = {n: [(b, m) for b in c["bases"] for m in c["mecanismos"] if b in self.bases()] for n, c in a["conjuntos"].items()}
        conjuntos = {n: v for n, v in conjuntos.items() if len(v) >= 2}
        filas, info = [], {}
        for nombre, expertos in conjuntos.items():
            for anio in self.plan["pliegues"]:
                for s in self.semillas():
                    Pv, Pe = [], []
                    for b, m in expertos:
                        ov, pv, yv = self._pred(b, m, anio, s, "val")
                        oe, pe, ye = self._pred(b, m, anio, s, "parada")
                        if Pv and (not np.array_equal(ov, ov0) or not np.array_equal(oe, oe0)):
                            raise RuntimeError("Los expertos no comparten los mismos orígenes")
                        ov0, oe0 = ov, oe
                        Pv.append(pv); Pe.append(pe)
                    pron, inf = AG.reglas(np.stack(Pe), ye, np.stack(Pv), yv, self.datos.q168, a["retraso_origenes"], tuple(a["betas"]), a["cota_perdida"])
                    inf["expertos"] = [f"{b}·{m}" for b, m in expertos]
                    inf["experto_ancla"] = inf["expertos"][inf["experto_ancla"]]
                    info[f"{nombre}_{anio}_{s}"] = inf
                    for regla, yhat in pron.items():
                        filas.append(dict(base="agregacion", mecanismo=f"{nombre}_{regla}", celda=f"agregacion·{nombre}_{regla}", pliegue=anio, semilla=s,
                                          MASE=MET.evaluar(yv, yhat, self.datos.q168)["MASE"], n_parametros=0))
        t = pd.DataFrame(filas)
        (self.salida / "agregacion").mkdir(exist_ok=True)
        t.to_csv(self.salida / "agregacion" / "desarrollo_por_semilla.csv", index=False)
        guardar_json(self.salida / "agregacion" / "informacion.json", info)
        malas = [k for k, v in info.items() for r, ok in v["cota_verificada"].items() if not ok]
        if malas:
            raise RuntimeError(f"La desigualdad de la Proposición J no se cumplió en {malas}: revise el código")
        return t

    # ------------------------------------------------------------------ regla de decisión
    def _tabla(self) -> pd.DataFrame:
        partes = [pd.read_csv(self.salida / "rejilla" / "desarrollo_por_semilla.csv")]
        if (self.salida / "agregacion" / "desarrollo_por_semilla.csv").exists():
            partes.append(pd.read_csv(self.salida / "agregacion" / "desarrollo_por_semilla.csv"))
        return pd.concat(partes, ignore_index=True)

    def _partir_agregacion(self, mec: str) -> tuple[str, str]:
        for n in sorted(self.plan["agregacion"]["conjuntos"], key=len, reverse=True):
            if mec.startswith(n + "_"):
                return n, mec[len(n) + 1:]
        raise ValueError(mec)

    @staticmethod
    def _contraste(t: pd.DataFrame, a: str, b: str) -> dict:
        """a frente a b, pareado por (pliegue, semilla). Mejora > 0 favorece a a."""
        x = t[t.celda == a].set_index(["pliegue", "semilla"]).MASE
        y = t[t.celda == b].set_index(["pliegue", "semilla"]).MASE
        j = x.index.intersection(y.index)
        x, y = x.loc[j].to_numpy(), y.loc[j].to_numpy()
        d = y - x
        out = dict(candidata=a, referencia=b, pares=int(len(j)), MASE_candidata=float(x.mean()), MASE_referencia=float(y.mean()),
                   mejora_pct=float(100 * (y.mean() - x.mean()) / y.mean()))
        if len(j) >= 3 and np.std(d) > 0:
            out["p_t_pareada_bilateral"] = float(stats.ttest_rel(y, x).pvalue)
            out["p_wilcoxon_bilateral"] = float(stats.wilcoxon(y, x).pvalue) if len(j) >= 6 and np.any(d != 0) else None
        return out

    def decision(self) -> dict:
        """Regla preregistrada y genérica: cada variante declara en el plan si es propuesta y frente a qué variantes de la
        misma base debe ganar (``referencias``); además, una propuesta sobre E debe ganar a la misma receta sobre PatchTST
        (``frente_a_otra_base``), para que la ventaja no sea solo de la receta."""
        pr = self.plan["regla"]
        t = self._tabla()
        esperadas = {f"{b}·{m}" for b in self.plan["bases"] for m in self.plan["mecanismos"]}
        faltan = sorted(esperadas - set(t.celda))
        n_esperado = len(self.plan["pliegues"]) * len(self.semillas())
        cortas = t.groupby("celda").size()
        cortas = cortas[cortas < n_esperado]
        if (faltan or len(cortas)) and not self.rapido:
            raise RuntimeError(f"La rejilla está incompleta: faltan {faltan[:4]}, cortas {cortas.to_dict()}")
        r = (t.groupby(["celda", "base", "mecanismo"]).agg(MASE_medio=("MASE", "mean"), MASE_desv=("MASE", "std"), n_parametros=("n_parametros", "max"),
                                                          corridas=("MASE", "size")).reset_index())
        m = dict(zip(r.celda, r.MASE_medio))
        controles = r[(r.mecanismo == "control") & r.base.isin(pr["bases_de_referencia"])].sort_values("MASE_medio")
        mejor_control = str(controles.celda.iloc[0])
        r["mejora_frente_al_mejor_control_pct"] = 100 * (m[mejor_control] - r.MASE_medio) / m[mejor_control]
        r["mejora_frente_a_su_control_pct"] = [100 * (m[f"{b}·control"] - v) / m[f"{b}·control"] for b, v in zip(r.base, r.MASE_medio)]
        otra = pr.get("frente_a_otra_base", {})
        r["mejora_frente_a_la_misma_receta_en_otra_base_pct"] = [
            100 * (m[f"{otra[b]}·{mec}"] - v) / m[f"{otra[b]}·{mec}"] if b in otra and f"{otra[b]}·{mec}" in m else np.nan
            for b, mec, v in zip(r.base, r.mecanismo, r.MASE_medio)]
        exp_r = self.salida / "exportabilidad" / "resultado.json"
        exp = leer_json(exp_r) if exp_r.exists() else {}
        exp_base = {b: bool(exp) and all(v.get("exportable") for k, v in exp.items() if k.split("·")[0] == b) and any(k.split("·")[0] == b for k in exp)
                    for b in self.plan["bases"]}
        r["exportable"] = r.base.map(exp_base).fillna(False).astype(bool)
        eleg, motivos = {}, {}
        for x in r.itertuples():
            d = self.plan["mecanismos"][x.mecanismo]
            if not d.get("propuesta") or x.base not in pr["bases_propuesta"]:
                continue
            refs = [f"{x.base}·{q}" for q in d["referencias"]]
            if x.base in otra:
                refs.append(f"{otra[x.base]}·{x.mecanismo}")
            perd = [q for q in refs if q in m and not m[x.celda] < m[q]]
            eleg[x.celda] = not perd
            motivos[x.celda] = "elegible" if not perd else "no mejora a " + ", ".join(perd)
        r["propuesta"] = r.celda.isin(eleg)
        r["elegible"] = r.celda.map(eleg).fillna(False).astype(bool)
        r["motivo"] = r.celda.map(motivos).fillna("referencia")
        r = r.sort_values("MASE_medio").reset_index(drop=True)
        (self.salida / "decision").mkdir(exist_ok=True)
        r.to_csv(self.salida / "decision" / "desarrollo_resumen.csv", index=False)
        t.to_csv(self.salida / "decision" / "desarrollo_por_semilla.csv", index=False)
        texto, gan = [], None
        cand = r[r.elegible & r.exportable].sort_values(["MASE_medio", "n_parametros"]).reset_index(drop=True)
        if len(cand):
            gan = str(cand.celda.iloc[0])
            if not m[gan] < m[mejor_control]:
                texto.append(f"La mejor propuesta elegible ({gan}, {m[gan]:.4f}) no mejora al mejor control ({mejor_control}, {m[mejor_control]:.4f}).")
                gan = None
        else:
            texto.append("Ninguna propuesta exportable cumple su condición de elegibilidad.")
        contrastes = []
        if gan:
            b = gan.split("·")[0]
            refs = [mejor_control] + [q for q in (f"{b}·control", f"{otra.get(b)}·{gan.split('·')[1]}") if q in m and q != mejor_control]
            contrastes = [self._contraste(t, gan, q) for q in refs]
            fuerte = contrastes[0]["mejora_pct"] >= 100 * pr["margen_fuerte"]
            texto.append(f"Se elige {gan} (MASE de desarrollo {m[gan]:.4f}); mejora frente al mejor control {mejor_control}: "
                         f"{contrastes[0]['mejora_pct']:.2f} % (p t pareada {contrastes[0].get('p_t_pareada_bilateral', float('nan')):.3g}). "
                         + ("Alcanza el margen de 3 %." if fuerte else "No alcanza el margen de 3 % del Entregable 3."))
        else:
            texto.append("Decisión: ninguna variante nueva de esta exploración; se conserva lo que decida Explorar_E3 (o E).")
        pd.DataFrame(contrastes).to_csv(self.salida / "decision" / "contrastes_ganadora.csv", index=False)
        n_prop = int(r.propuesta.sum())
        dec = dict(ganadora=gan, motivo=" ".join(texto), regla=pr["texto"], mejor_control=mejor_control, contrastes=contrastes, exportable_por_base=exp_base,
                   N_d_esta_exploracion=n_prop, N_d_previo=self.plan["diseno_previo"]["N_d"], N_d_acumulado=n_prop + self.plan["diseno_previo"]["N_d"],
                   pliegues_desarrollo=self.plan["pliegues"], semillas=self.semillas(), fecha=_hoy(), prueba_rapida=bool(self.rapido),
                   transparencia=self.plan["transparencia"])
        ruta = self.salida / "decision" / "decision.json"
        if ruta.exists():
            previa = leer_json(ruta)
            if previa["ganadora"] != gan:
                raise RuntimeError("La decisión cambió al reanudar; no se sobrescribe una decisión congelada")
            dec = previa
        else:
            guardar_json(ruta, dec)
        self._bitacora(r, dec)
        self.log(f"DECISIÓN: {dec['motivo']}")
        return dec

    def _bitacora(self, r: pd.DataFrame, dec: dict):
        f = ["| N.° | Celda | Propuesta | MASE de desarrollo (media ± DE) | Frente a su control | Frente al mejor control | ONNX | Decisión | Motivo |",
             "|---|---|---|---|---|---|---|---|---|"]
        for i, x in enumerate(r.itertuples(), start=1):
            d = "Congelar" if x.celda == dec["ganadora"] else ("Descartar" if x.propuesta else "Referencia")
            sc = "" if pd.isna(x.mejora_frente_a_su_control_pct) else f"{x.mejora_frente_a_su_control_pct:.2f} %"
            f.append(f"| {i} | {x.celda} | {'sí' if x.propuesta else 'no'} | {x.MASE_medio:.4f} ± {x.MASE_desv:.4f} | {sc} | "
                     f"{x.mejora_frente_al_mejor_control_pct:.2f} % | {'sí' if x.exportable else 'no'} | {d} | {x.motivo} |")
        (self.salida / "decision" / "design_log.md").write_text(
            "# Bitácora de diseño: exploración numérica (generada por la etapa decision)\n\n"
            f"Partición de desarrollo: pliegues {dec['pliegues_desarrollo']} con semillas {dec['semillas']}; la prueba 2022-2025 no se lee. "
            f"Propuestas evaluadas aquí: {dec['N_d_esta_exploracion']}; antes (parte 1: A, AB, AC, E, F): {dec['N_d_previo']}; "
            f"N_d acumulado = {dec['N_d_acumulado']}.\n\n" + "\n".join(f) +
            f"\n\nDecisión ({dec['fecha'][:10]}): **{dec['ganadora'] or 'ninguna'}**. {dec['motivo']}\n\nRegla preregistrada: {dec['regla']}\n\n"
            f"Transparencia: {dec['transparencia']}\n", encoding="utf-8")

    # ------------------------------------------------------------------ sensibilidad del ganador
    def sensibilidad(self) -> pd.DataFrame:
        dec = leer_json(self.salida / "decision" / "decision.json")
        gan = dec["ganadora"]
        if not gan or gan.startswith("agregacion") or gan.endswith("·control") or not any(
                k in self.plan["sensibilidad"]["valores"] for k in self.plan["mecanismos"][gan.split("·")[1]]["parametros"]):
            self.log("SENSIBILIDAD: el ganador no tiene hiperparámetros de mecanismo que variar (se omite).")
            return pd.DataFrame()
        base, mec = gan.split("·")
        sp = self.plan["sensibilidad"]
        puntos = []
        for clave, valores in sp["valores"].items():
            if clave in self.plan["mecanismos"][mec]["parametros"]:
                puntos += [(clave, v) for v in valores]
        sem = self.semillas()[:sp["semillas"]] if not self.rapido else self.semillas()[:1]
        filas = []
        for (clave, v), anio, s in itertools.product(puntos, self.plan["pliegues"], sem):
            nombre = f"{mec}_{clave}_{v:g}"
            fila = self._celda(base, mec, anio, s, raiz="sensibilidad", extra={clave: v}, nombre=nombre)
            filas.append(dict(fila, parametro=clave, valor=v))
        for anio, s in itertools.product(self.plan["pliegues"], sem):          # el punto preregistrado, de la rejilla
            r = leer_json(self.salida / "rejilla" / base / mec / f"pliegue_{anio}" / f"semilla_{s}" / "resultado.json")
            for clave in {c for c, _ in puntos}:
                filas.append(self._fila(r, base=base, mecanismo=mec, celda=gan, pliegue=anio, semilla=s, parametro=clave,
                                        valor=self.plan["mecanismos"][mec]["parametros"][clave]))
        t = pd.DataFrame(filas)
        t.to_csv(self.salida / "sensibilidad" / "tabla.csv", index=False)
        t.groupby(["parametro", "valor"]).MASE.agg(["mean", "std", "size"]).reset_index().to_csv(self.salida / "sensibilidad" / "resumen.csv", index=False)
        return t

    # ------------------------------------------------------------------ resumen legible
    def resumen(self) -> str:
        out = ["# Resumen de la exploración numérica", ""]
        dec_r = self.salida / "decision" / "decision.json"
        if dec_r.exists():
            dec = leer_json(dec_r)
            r = pd.read_csv(self.salida / "decision" / "desarrollo_resumen.csv")
            out += [f"Decisión: **{dec['ganadora'] or 'ninguna'}**. {dec['motivo']}", "",
                    "| Celda | MASE de desarrollo | Frente al mejor control | Elegible | ONNX |", "|---|---|---|---|---|"]
            out += [f"| {x.celda} | {x.MASE_medio:.4f} ± {x.MASE_desv:.4f} | {x.mejora_frente_al_mejor_control_pct:.2f} % | "
                    f"{'sí' if x.elegible else ('—' if not x.propuesta else 'no')} | {'sí' if x.exportable else 'no'} |" for x in r.itertuples()]
        ex = self.salida / "exportabilidad" / "resultado.json"
        if ex.exists():
            out += ["", "Exportabilidad a ONNX (sin entrenar; trazado con una entrada y verificación con otra):", ""]
            out += [f"- {k}: {'sí' if v.get('exportable') else 'no'}" + (f" ({v['motivo'][:120]})" if v.get("motivo") else
                    f" (error máximo con otra entrada {v.get('error_max_otra_entrada', float('nan')):.2e})") for k, v in leer_json(ex).items()]
        s = self.salida / "sensibilidad" / "resumen.csv"
        if s.exists():
            t = pd.read_csv(s)
            out += ["", "Sensibilidad del ganador (MASE medio):", ""] + [f"- {x.parametro} = {x.valor:g}: {x.mean:.4f} ({int(x.size)} corridas)" for x in t.itertuples()]
        txt = "\n".join(out) + "\n"
        (self.salida / "RESUMEN_EXPLORACION.md").write_text(txt, encoding="utf-8")
        return txt

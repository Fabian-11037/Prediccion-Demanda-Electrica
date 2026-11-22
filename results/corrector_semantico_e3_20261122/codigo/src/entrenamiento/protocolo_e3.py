"""Etapas de entrenamiento del Entregable 3 sobre el protocolo común (``protocolo.py``).

Orden (cada etapa es reanudable: una carpeta con ``resultado.json`` no se repite):
  diseno        las ideas A, AB, AC, E y F y el control PatchTST N en la partición de desarrollo (pliegues 2018 y 2019,
                semillas 101-105); regla de decisión preregistrada; la idea elegida se congela (diseno/decision.json)
  ajuste        hiperparámetros de la idea elegida con el presupuesto de la familia (20 configuraciones en 2018, 5 en 2019-2021)
  potencia      piloto de 5 semillas en el pliegue 2021 del original y de cada rival; σ_d, R por contraste y R final (≥ 10)
  familia       el original y los 17 rivales con R semillas en la prueba 2022-2025 (sellada hasta aquí), con costes
  adicionales   los tres conjuntos adicionales: búsqueda con el mismo presupuesto y R semillas para los 18 modelos
  ablacion      factorial 2^3 de los componentes propios + base sin rama propia + control de capacidad (≥ 10 semillas)
  sensibilidad  hiperparámetros propios, uno a la vez, en el pliegue 2021 (3 semillas)
  curva         curva de aprendizaje (10, 25, 50 y 100 % de los orígenes de ajuste) del original y del mejor rival
  escalabilidad tiempo, memoria y FLOPs al variar la longitud de la historia (sin entrenar)
  robustez      perturbaciones de la prueba en 3 severidades con los pesos finales de todos los modelos
  sanidad       el original con etiquetas permutadas
  exportar      ONNX, prueba de ida y vuelta, metadata.json, cuantización si hace falta y latencia en CPU
  interpretabilidad  gradientes integrados por grupo de variables, borrado y estabilidad entre semillas
"""
from __future__ import annotations

import datetime as dt
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats

from demanda import metricas as MET

from . import clasicos, motor
from . import protocolo as PR
from . import protocolo_e2 as P2
from . import redes_e3
from .protocolo import guardar_json, leer_json


def _hoy() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


class SinResultadosE2(RuntimeError):
    pass


class E3:
    def __init__(self, raiz: Path, salida: Path, plan: dict, datos, texto: dict | None, descripciones, dispositivo: str, log, rapido: dict | None, args,
                 adicionales: dict | None = None, descripciones_adicionales: dict | None = None, parte1: Path | None = None):
        self.raiz, self.salida, self.plan, self.datos = Path(raiz), Path(salida), plan, datos
        self.p1 = Path(parte1) if parte1 is not None else self.salida      # resultados de la parte 1 (solo lectura en la parte 2)
        self.texto, self.desc, self.dispositivo, self.log, self.rapido, self.args = texto, descripciones, dispositivo, log, rapido, args
        self.adic = adicionales or {}
        self.desc_adic = descripciones_adicionales or {}
        self.barra = None
        self._fam = None
        self._ctx_cache = (None, None)
        self._esc_final = None

    # ================================================================== utilidades
    def ctx(self, carpeta: Path, clave: str, red: str, variante: str, datos=None, adicional: bool = False) -> PR.Contexto:
        c = self.plan["comun"]
        busq = dict(c["busqueda"])
        if adicional:
            busq.update(pliegue_inicial=1, pliegues_restantes=[2, 3, 4])
        if self.rapido:
            busq["finalistas"] = self.rapido["finalistas"]
        datos = datos if datos is not None else self.datos
        cfg = dict(version=self.plan["version"], paquete=dict(clave=clave, nombre=clave, red=red, carpeta_resultados=Path(carpeta).name), variantes=[variante],
                   busqueda=busq, final=dict(semillas=self.semillas()), entrenamiento=dict(c["entrenamiento"]), inferencia=dict(c["inferencia"]))
        x = PR.Contexto(self.raiz, carpeta, cfg, datos, self.dispositivo, self.log, self.rapido)
        if self.rapido:
            x.opciones["lote_evaluacion"] = self.rapido.get("lote_evaluacion", 16)
        x.hilos, x.mlflow, x.mlflow_raiz, x.barra = self.args.hilos, not self.args.sin_mlflow, self.salida, self.barra
        if variante == "N+O" and datos is self.datos:
            x.texto = self.texto
        if red.startswith("tgforecaster"):
            x.descripciones = self.desc if datos is self.datos else self.desc_adic.get(getattr(datos, "nombre", ""))
        return x

    def semillas(self) -> list[int]:
        if self.rapido:
            return list(self.rapido["semillas_finales"])
        r = self.salida / "potencia" / "R.json"
        R = leer_json(r)["R"] if r.exists() else self.plan["potencia"]["R_minimo"]
        return list(range(self.plan["comun"]["semilla_base_final"], self.plan["comun"]["semilla_base_final"] + int(R)))

    def _pos_pliegue(self, ctx, anio):
        return P2._pos(ctx, anio)

    def _pos_final(self, ctx):
        return P2._pos_final(ctx)

    # ------------------------------------------------------------------ resultados del Entregable 2
    def _buscar(self, patron: str) -> Path | None:
        for d in self.args.resultados_e2 or []:
            for r in sorted(Path(d).glob("**/" + patron)):
                return r
        return None

    def familia_resuelta(self) -> dict:
        """Configuración principal de cada rival: fija (paquetes del E2) o la elegida por validación en Completar_E2."""
        if self._fam is not None:
            return self._fam
        out, faltan = {}, []
        for k, d in self.plan["familia"].items():
            src = d["config_principal"]
            e = dict(d)
            if "fija" in src:
                e["config"], e["origen_config"] = dict(src["fija"]), "configuración elegida en el Entregable 2 (paquete del modelo)"
            else:
                s = src["e2"]
                if s["tipo"] == "busqueda14":
                    r = self._buscar(f"busqueda14_{s['modelo']}/resumen_busqueda14.json")
                    if r:
                        j = leer_json(r)
                        sel = j["seleccion"]
                        e["config"] = dict(j["hiperparametros_fijos"], id=3000, arq=sel["ganadora"], anchos=sel["anchos"], patron_dropout=sel["patron_dropout"])
                        e["MASE_validacion_e2"], e["origen_config"] = sel.get("MASE_medio_fase2"), str(r)
                elif s["tipo"] == "nuevo":
                    r = self._buscar(f"nuevo_{s['modelo']}/busqueda/configuracion_elegida.json")
                    if r:
                        j = leer_json(r)["N"]
                        e["config"], e["MASE_validacion_e2"], e["origen_config"] = j["configuracion"], j.get("MASE_medio_validacion"), str(r)
                elif s["tipo"] == "lightgbm":
                    r = self._buscar("clasico_lightgbm/busqueda/configuracion_elegida.json")
                    if r:
                        j = leer_json(r)
                        e["config"], e["MASE_validacion_e2"], e["origen_config"] = j["configuracion"], j.get("MASE_medio_validacion"), str(r)
                elif s["tipo"] == "mstl":
                    r = self._buscar("clasico_mstl/resumen_mstl.json")
                    if r:
                        e["config"], e["origen_config"] = dict(id=1, semanas=int(leer_json(r)[s["modelo"]]["semanas_elegidas"])), str(r)
                if "config" not in e:
                    if self.rapido:          # prueba rápida sin el Entregable 2: configuración de prueba, nunca válida para el informe
                        e["config"] = self._config_de_prueba(k, d)
                        e["origen_config"] = "PRUEBA RÁPIDA: configuración provisional"
                    else:
                        faltan.append(f"{k} ({s})")
            out[k] = e
        if faltan:
            raise SinResultadosE2("Faltan resultados del Entregable 2 para: " + ", ".join(faltan) +
                                  ". Indique las carpetas con --resultados-e2 (resultados_e2 de Completar_E2 y resultados_e2_nbeats_timesnet).")
        self._fam = out
        guardar_json(self.salida / "familia_configuraciones.json", out)
        return out

    def _config_de_prueba(self, k, d):
        if d["tipo"] == "mstl":
            return dict(id=1, semanas=4)
        p = dict(d["espacio_adicionales"][0])
        if d.get("arquitectura_de"):
            p.update(id=3000, anchos=[16, 16, 16], patron_dropout=[0, 1, 0])
        if d.get("red") == "nbeats":
            p.update(pilas=3, ancho=64)
        if d.get("red") == "timesnet":
            p.update(d_model=16, d_ff=16, num_kernels=2, top_k=2, e_layers=1)
        return p

    def arquitectura(self, clave: str) -> dict:
        c = self.familia_resuelta()[clave]["config"]
        return dict(anchos=c["anchos"], patron_dropout=c["patron_dropout"])

    # ------------------------------------------------------------------ original
    def decision(self) -> dict:
        r = self.p1 / "diseno" / "decision.json"
        if not r.exists():
            raise RuntimeError("La etapa diseno no ha elegido idea: ejecútela antes")
        return leer_json(r)

    def original(self) -> tuple[str, dict, dict]:
        """(idea, definición, configuración elegida en la etapa ajuste)."""
        idea = self.decision()["idea"]
        r = self.p1 / "ajuste" / "busqueda" / "configuracion_elegida.json"
        if not r.exists():
            raise RuntimeError("La etapa ajuste no ha elegido configuración: ejecútela antes")
        d = self.plan["original"][idea]
        return idea, d, leer_json(r)[d["variante"]]["configuracion"]

    def miembros(self, con_original: bool = True, con_rivales: bool = True) -> list[dict]:
        """Lista uniforme de procedimientos de la familia confirmatoria (el original primero). Los rivales necesitan los
        resultados del Entregable 2; la parte 1 (``con_rivales=False``) no los pide."""
        out = []
        if con_original:
            idea, d, p = self.original()
            out.append(dict(clave="original", nombre=f"Original ({d['nombre']})", tipo="red", red=d["red"], variante=d["variante"], config=p, grupo="original",
                            variante_adicionales="N", espacio_adicionales=d["configuraciones"]))
        if con_rivales:
            for k, e in self.familia_resuelta().items():
                out.append(dict(e, clave=k))
        return out

    # ================================================================== etapas
    def diseno(self) -> dict:
        pd_ = self.plan["diseno"]
        sem = self.rapido["semillas_diseno"] if self.rapido else pd_["semillas"]
        pliegues = pd_["pliegues"]
        filas = []
        for var in pd_["variantes"]:
            if var == "control_patchtst":
                red, variante, p = "tats", "N", dict(self.plan["familia"]["patchtst"]["config_principal"]["fija"])
            else:
                d = self.plan["original"][var]
                red, variante, p = d["red"], d["variante"], dict(d["configuraciones"][pd_["configuracion"] - 1])
            for anio in pliegues:
                for s in sem:
                    car = self.salida / "diseno" / var / f"pliegue_{anio}" / f"semilla_{s}"
                    ctx = self.ctx(car.parent, f"diseno_{var}", red, variante)
                    pos = self._pos_pliegue(ctx, anio)
                    texto = variante == "N+O"
                    r = PR.corrida(ctx, variante, p, s, f"pliegue_{anio}", pos, car, f"val_{anio}", guardar_modelo=texto)
                    fila = dict(variante=var, pliegue=anio, semilla=s, MASE=r["metricas"]["MASE"], MASE_ajuste=r["MASE_ajuste"], MASE_parada=r["MASE_parada"],
                                n_parametros=r["n_parametros"], epocas=r["epocas"])
                    if texto:
                        fila["MASE_placebo"] = self._mase_placebo(ctx, red, p, car, pos, anio)
                    filas.append(fila)
        t = pd.DataFrame(filas)
        t.to_csv(self.salida / "diseno" / "desarrollo_por_semilla.csv", index=False)
        agg = dict(MASE_medio=("MASE", "mean"), MASE_desv=("MASE", "std"), n_parametros=("n_parametros", "first"), corridas=("MASE", "size"))
        if "MASE_placebo" in t:
            agg["MASE_placebo_medio"] = ("MASE_placebo", "mean")
        r = t.groupby("variante").agg(**agg).reset_index()
        ref = float(r.loc[r.variante == "control_patchtst", "MASE_medio"].iloc[0])
        r["mejora_frente_al_control_pct"] = 100 * (ref - r.MASE_medio) / ref
        r.to_csv(self.salida / "diseno" / "desarrollo_resumen.csv", index=False)
        # regla preregistrada
        tx = r[r.variante.isin(pd_["ideas_texto"])].sort_values("MASE_medio")
        motivo = []
        idea = None
        if len(tx):
            mejor = tx.iloc[0]
            pasa_margen = mejor.mejora_frente_al_control_pct >= 100 * pd_["margen_texto"]
            pasa_placebo = bool(mejor.MASE_medio < mejor.get("MASE_placebo_medio", np.inf))
            motivo.append(f"Mejor idea de texto: {mejor.variante} (MASE {mejor.MASE_medio:.4f}; mejora frente al control {mejor.mejora_frente_al_control_pct:.2f} %; "
                          f"placebo {mejor.get('MASE_placebo_medio', float('nan')):.4f}). Margen de 3 %: {'sí' if pasa_margen else 'no'}; mejor que placebo: {'sí' if pasa_placebo else 'no'}.")
            if pasa_margen and pasa_placebo:
                idea = str(mejor.variante)
                motivo.append("Se adopta la línea de texto.")
        if idea is None:
            nu = r[r.variante.isin(pd_["ideas_numericas"])].sort_values(["MASE_medio", "n_parametros"]).reset_index(drop=True)
            idea = str(nu.variante.iloc[0])
            if len(nu) > 1 and (nu.MASE_medio.iloc[1] - nu.MASE_medio.iloc[0]) / nu.MASE_medio.iloc[0] < pd_["empate_relativo"]:
                idea = str(nu.sort_values(["n_parametros", "MASE_medio"]).variante.iloc[0])
                motivo.append("Empate técnico entre E y F (menos de 1 %): se elige la de menos parámetros.")
            motivo.append(f"Línea numérica: se elige {idea} (MASE de desarrollo {float(r.loc[r.variante == idea, 'MASE_medio'].iloc[0]):.4f}).")
        if self.rapido and self.rapido.get("idea_forzada"):        # solo para verificar el código en la prueba rápida
            idea = self.rapido["idea_forzada"]
            motivo.append(f"PRUEBA RÁPIDA: idea forzada {idea} para recorrer su código.")
        dec = dict(idea=idea, motivo=" ".join(motivo), regla=pd_["regla"], N_d=int(len(pd_["variantes"]) - 1), variantes_evaluadas=pd_["variantes"],
                   pliegues_desarrollo=pliegues, semillas=sem, fecha=_hoy(), resumen=r.to_dict("records"), prueba_rapida=bool(self.rapido))
        if not (self.salida / "diseno" / "decision.json").exists():
            guardar_json(self.salida / "diseno" / "decision.json", dec)
        else:
            previa = leer_json(self.salida / "diseno" / "decision.json")
            if previa["idea"] != idea:
                raise RuntimeError("La decisión de diseño cambió al reanudar; revise la carpeta (no se sobrescribe una decisión congelada)")
            dec = previa
        self._bitacora_diseno(r, dec)
        self.log(f"DISEÑO: idea elegida {idea}. {dec['motivo']}")
        return dec

    def _mase_placebo(self, ctx, red, p, car: Path, pos, anio) -> float:
        """MASE del mismo modelo entrenado cuando el texto se reemplaza por el de un día donante anterior (fijo por origen)."""
        rr = car / "placebo.json"
        if rr.exists():
            return leer_json(rr)["MASE_placebo"]
        esc, conj = ctx.conjuntos(f"pliegue_{anio}", pos)
        tx = {k: torch.from_numpy(v) for k, v in self.texto.items()}
        m = redes_e3.construir(red, p, conj["ev"].x.shape[2], conj["ev"].cal_p.shape[2], self.datos.L, self.datos.H, tx, None,
                               n_mascaras=len(self.datos.nombres_mascara))
        m.load_state_dict(torch.load(car / "modelo.pt", map_location="cpu", weights_only=True))
        m.placebo_fijo = True
        m = m.to(self.dispositivo).eval()
        t = motor.a_tensores(conj["ev"], self.dispositivo)
        with torch.no_grad():
            o = torch.cat([m(t["x"][i:i + 256], t["cal_p"][i:i + 256], t["cal_f"][i:i + 256], t["dia"][i:i + 256], placebo=True).cpu()
                           for i in range(0, len(t["y"]), 256)]).numpy()
        mase = MET.evaluar(conj["ev"].y, o * conj["ev"].nivel[:, None], self.datos.q168)["MASE"]
        guardar_json(rr, dict(MASE_placebo=mase, regla="donante 28 + (día · 7919 mod 337) días antes del origen"))
        return mase

    def _bitacora_diseno(self, r: pd.DataFrame, dec: dict):
        ruta = self.salida / "diseno" / "design_log.md"
        filas = ["| N.° | Variante evaluada | MASE de desarrollo (media ± DE) | Mejora frente al control | Decisión | Motivo y fecha |", "|---|---|---|---|---|---|"]
        for i, x in enumerate(r.sort_values("MASE_medio").itertuples(), start=1):
            decision = "Congelar" if x.variante == dec["idea"] else ("Referencia" if x.variante == "control_patchtst" else "Descartar")
            filas.append(f"| {i} | {x.variante} | {x.MASE_medio:.4f} ± {x.MASE_desv:.4f} | {x.mejora_frente_al_control_pct:.2f} % | {decision} | {dec['fecha'][:10]} |")
        ruta.write_text("# Bitácora de diseño (generada por la etapa diseno)\n\n"
                        f"Partición de desarrollo: pliegues {dec['pliegues_desarrollo']} con semillas {dec['semillas']}. N_d = {dec['N_d']} variantes propias "
                        "más el control.\n\n" + "\n".join(filas) + f"\n\nDecisión: **{dec['idea']}**. {dec['motivo']}\n\nRegla preregistrada: {dec['regla']}\n",
                        encoding="utf-8")

    # ------------------------------------------------------------------ ajuste del original
    def ajuste(self) -> dict:
        idea = self.decision()["idea"]
        d = self.plan["original"][idea]
        ctx = self.ctx(self.salida / "ajuste", f"original_{idea}", d["red"], d["variante"])
        cfgs = [c for c in d["configuraciones"] if not self.rapido or c["id"] in self.rapido["configuraciones"]]
        fin = PR.etapa_busqueda(ctx, cfgs)
        eleg = PR.etapa_pliegues(ctx, cfgs, fin)
        self.log(f"AJUSTE: configuración {eleg[d['variante']]} de la idea {idea}.")
        return leer_json(self.salida / "ajuste" / "busqueda" / "configuracion_elegida.json")

    # ------------------------------------------------------------------ corridas por tipo de procedimiento
    def correr(self, m: dict, carpeta: Path, semilla: int, pos: dict, clave_datos: str, nombre_ev: str, datos=None, adicional=False, config=None,
               variante=None, guardar=False, medir=False) -> dict:
        import gc
        gc.collect()
        datos = datos if datos is not None else self.datos
        p = dict(config if config is not None else m["config"])
        variante = variante or m["variante"]
        if m["tipo"] in ("red", "lightgbm"):
            red = m["red"] if m["tipo"] == "red" else "lightgbm"
            llave = (m["clave"], red, variante, getattr(datos, "nombre", "principal"), adicional, clave_datos)
            if self._ctx_cache[0] != llave:
                self._ctx_cache = (llave, self.ctx(Path(carpeta).parent, m["clave"], red, variante, datos, adicional))
            ctx = self._ctx_cache[1]
            if m["tipo"] == "red":
                return PR.corrida(ctx, variante, p, semilla, clave_datos, pos, carpeta, nombre_ev, guardar_modelo=guardar, medir=medir)
            return clasicos.corrida_lgbm(ctx, p, semilla, clave_datos, pos, carpeta, nombre_ev, guardar=bool(guardar))
        return self._mstl(m, carpeta, pos, nombre_ev, datos, int(p["semanas"]))

    def _mstl(self, m: dict, carpeta: Path, pos: dict, nombre_ev: str, datos, semanas: int) -> dict:
        carpeta = Path(carpeta)
        rr = carpeta / "resultado.json"
        if rr.exists():
            return leer_json(rr)
        carpeta.mkdir(parents=True, exist_ok=True)
        import time
        ev = pos["ev"]
        t0 = time.time()
        yhat = clasicos._pronosticar(datos, ev, m["modelo"], semanas, self.args.hilos)
        seg = time.time() - t0
        y = datos.Y[ev[:, None] + np.arange(datos.H)]
        np.savez_compressed(carpeta / f"pred_{nombre_ev}.npz", origen=np.asarray(datos.indice[ev].strftime("%Y-%m-%d"), dtype="U10"), yhat_kwh=yhat.astype("float32"),
                            y_kwh=y.astype("float32"), mase_por_origen=datos.mase_por_origen(y, yhat))
        met = MET.evaluar(y, yhat, datos.q168)
        res = dict(variante="N", configuracion=dict(id=1, semanas=semanas, modelo=m["modelo"]), semilla=None, evaluacion=nombre_ev, dispositivo="cpu", n_parametros=0,
                   metricas=met, mase_por_horizonte=[float(v) for v in MET.mase_por_horizonte(y, yhat, datos.q168)], determinista=True,
                   costes=dict(segundos_entrenamiento=0.0, latencia_ms_por_origen_lote=1000 * seg / len(ev), dispositivo="cpu"), segundos=round(seg, 1),
                   MASE_ajuste=None, MASE_parada=None, epocas=0)
        guardar_json(rr, res)
        return res

    # ------------------------------------------------------------------ potencia
    def _de_parte1(self, m: dict, carpeta_rel: str, s) -> Path:
        """Carpeta de una corrida del original hecha en la parte 1; en la parte 2 debe existir (no se reentrena ni se escribe)."""
        car = self.p1 / carpeta_rel
        if self.p1 != self.salida and not (car / "resultado.json").exists():
            raise RuntimeError(f"Falta {car / 'resultado.json'} de la parte 1: termine la parte 1 antes de la parte 2")
        return car

    def piloto_original(self) -> pd.DataFrame:
        """Parte 1: las 5 semillas piloto del original en el pliegue 2021 (las de los rivales se agregan en la etapa potencia)."""
        t = self._piloto(self.miembros(con_rivales=False))
        t.to_csv(self.salida / "potencia" / "piloto_original.csv", index=False)
        self.log(f"PILOTO del original: MASE medio {t.MASE.mean():.4f} en el pliegue {self.plan['potencia']['pliegue_piloto']} ({len(t)} semillas).")
        return t

    def _piloto(self, miembros: list[dict]) -> pd.DataFrame:
        pp = self.plan["potencia"]
        sem = self.rapido["semillas_finales"] if self.rapido else pp["semillas_piloto"]
        filas = []
        for m in miembros:
            for s in (sem if not m.get("determinista") else [None]):
                rel = f"potencia/{m['clave']}/" + (f"semilla_{s}" if s is not None else "determinista")
                car = self._de_parte1(m, rel, s) if m["clave"] == "original" else self.salida / rel
                ctx = self.ctx(car.parent, m["clave"], m.get("red", "lightgbm"), m["variante"])
                pos = self._pos_pliegue(ctx, pp["pliegue_piloto"])
                r = self.correr(m, car, s if s is not None else 0, pos, f"pliegue_{pp['pliegue_piloto']}", f"val_{pp['pliegue_piloto']}")
                filas.append(dict(clave=m["clave"], semilla=s, MASE=r["metricas"]["MASE"]))
        return pd.DataFrame(filas)

    def potencia(self) -> dict:
        """Parte 2: piloto de los 17 rivales (el del original se reanuda), σ_d por contraste y R."""
        pp = self.plan["potencia"]
        t = self._piloto(self.miembros())
        t.to_csv(self.salida / "potencia" / "piloto_por_semilla.csv", index=False)
        ori = t[t.clave == "original"].set_index("semilla").MASE
        rivales = [k for k in t.clave.unique() if k != "original"]
        mm = len(rivales)
        za, zb = stats.norm.ppf(1 - pp["alfa"] / (2 * mm)), stats.norm.ppf(pp["potencia_objetivo"])
        out = []
        for k in rivales:
            v = t[t.clave == k]
            mase_k = float(v.MASE.mean())
            if v.semilla.isna().all():
                dif = ori.to_numpy() - float(v.MASE.iloc[0])
            else:
                v = v.set_index("semilla").MASE.reindex(ori.index)
                dif = (ori - v).to_numpy()
            sd = float(np.std(dif, ddof=1)) if len(dif) > 1 else float("nan")
            delta = pp["margen_relativo"] * mase_k
            Rk = math.ceil((za + zb) ** 2 * sd ** 2 / delta ** 2) if np.isfinite(sd) and delta > 0 else None
            out.append(dict(rival=k, MASE_rival_piloto=mase_k, MASE_original_piloto=float(ori.mean()), diferencia_media=float(np.mean(dif)),
                            sd_diferencias=sd, delta=delta, R_necesario=Rk))
        a = pd.DataFrame(out)
        Rnec = int(np.nanmax(a.R_necesario.to_numpy(dtype=float))) if a.R_necesario.notna().any() else pp["R_minimo"]
        R = int(min(pp["R_maximo"], max(pp["R_minimo"], Rnec)))
        a["potencia_con_R"] = [float(stats.norm.cdf(r.delta * math.sqrt(R) / r.sd_diferencias - za)) if r.sd_diferencias and r.sd_diferencias > 0 else 1.0 for r in a.itertuples()]
        a["diferencia_minima_detectable_con_R"] = [(za + zb) * r.sd_diferencias / math.sqrt(R) for r in a.itertuples()]
        a.to_csv(self.salida / "potencia" / "analisis_potencia.csv", index=False)
        res = dict(R=R, R_necesario_maximo=Rnec, limitado_por_R_maximo=bool(Rnec > pp["R_maximo"]), m=mm, alfa_ajustado=pp["alfa"] / mm,
                   potencia_conjunta_aprox=float(np.prod(a.potencia_con_R)), regla=pp["formula"], fecha=_hoy(), prueba_rapida=bool(self.rapido))
        if not (self.salida / "potencia" / "R.json").exists():
            guardar_json(self.salida / "potencia" / "R.json", res)
        self.log(f"POTENCIA: R = {R} semillas (máximo necesario {Rnec}); potencia conjunta aproximada {res['potencia_conjunta_aprox']:.3f}.")
        return res

    # ------------------------------------------------------------------ familia en la prueba
    def familia(self) -> pd.DataFrame:
        if not self.rapido and not (self.salida / "potencia" / "R.json").exists():
            raise RuntimeError("La etapa potencia no ha fijado R: la prueba no se abre antes del análisis de potencia")
        filas = []
        for m in self.miembros():
            for i, s in enumerate(self.semillas() if not m.get("determinista") else [None]):
                car = self.salida / "familia" / m["clave"] / (f"semilla_{s}" if s is not None else "determinista")
                ctx = self.ctx(car.parent, m["clave"], m.get("red", "lightgbm"), m["variante"])
                pos = self._pos_final(ctx)
                r = self.correr(m, car, s if s is not None else 0, pos, "final", "prueba", guardar="completo" if i == 0 else True, medir=True)
                filas.append(dict(clave=m["clave"], grupo=m["grupo"], semilla=s, MASE=r["metricas"]["MASE"], n_parametros=r["n_parametros"]))
        t = pd.DataFrame(filas)
        t.to_csv(self.salida / "familia" / "metricas_por_semilla.csv", index=False)
        return t

    # ------------------------------------------------------------------ conjuntos adicionales
    def adicionales(self, solo: list[str] | None = None) -> pd.DataFrame:
        filas = []
        for nombre, datos in self.adic.items():
            if solo and nombre not in solo:
                continue
            base = self.salida / "adicionales" / nombre
            base.mkdir(parents=True, exist_ok=True)
            guardar_json(base / "conjunto.json", datos.resumen())
            for m in self.miembros():
                filas += self._adicional(nombre, datos, m, base / m["clave"])
        t = pd.DataFrame(filas)
        if len(t):
            prev = self.salida / "adicionales" / "metricas_por_semilla.csv"
            if prev.exists():
                t = pd.concat([pd.read_csv(prev), t]).drop_duplicates(["conjunto", "clave", "semilla"], keep="last")
            t.to_csv(prev, index=False)
        return t

    def adicionales_original(self, solo: list[str] | None = None) -> dict:
        """Parte 1: búsqueda de hiperparámetros del original en cada conjunto adicional (solo validación; sin la prueba)."""
        out = {}
        m = self.miembros(con_rivales=False)[0]
        for nombre, datos in self.adic.items():
            if solo and nombre not in solo:
                continue
            base = self.salida / "adicionales" / nombre
            guardar_json(base / "conjunto.json", datos.resumen())
            out[nombre] = self._adicional(nombre, datos, m, base / m["clave"], solo_busqueda=True)
        return out

    def _adicional(self, nombre, datos, m, carpeta: Path, solo_busqueda: bool = False) -> list[dict]:
        sem = self.semillas()
        if m["tipo"] == "mstl":
            ctx = self.ctx(carpeta, m["clave"], "mstl", "N", datos, True)
            plan = dict(modelos=[m["modelo"]], ventanas_semanas=m["ventanas_semanas"][:1] if self.rapido else m["ventanas_semanas"], pliegues_validacion=[1, 2, 3, 4])
            r = clasicos.mstl(ctx, plan)[m["modelo"]]
            return [dict(conjunto=nombre, clave=m["clave"], semilla=None, MASE=r["MASE_prueba"])]
        if m["tipo"] == "lightgbm":
            ctx = self.ctx(carpeta, m["clave"], "lightgbm", "N", datos, True)
            ctx.cfg["final"]["semillas"] = sem
            r = clasicos.lightgbm(ctx, m["espacio_adicionales"])
            return [dict(conjunto=nombre, clave=m["clave"], semilla=s, MASE=v) for s, v in zip(sem, r["MASE_por_semilla"])]
        variante = m.get("variante_adicionales", "N")
        espacio = [dict(c) for c in m["espacio_adicionales"]]
        if m.get("arquitectura_de"):
            arq = self.arquitectura(m["clave"])
            espacio = [dict(c, **arq) for c in espacio]
        if self.rapido:
            espacio = [c for c in espacio if c["id"] in self.rapido["configuraciones"]]
            if m["red"] == "nbeats":
                espacio = [dict(c, pilas=3, ancho=64) for c in espacio]
            if m["red"] == "timesnet":
                espacio = [dict(c, d_model=16, d_ff=16, num_kernels=2, top_k=2, e_layers=1) for c in espacio]
        ctx = self.ctx(carpeta, m["clave"], m["red"], variante, datos, True)
        previa = self.p1 / "adicionales" / nombre / m["clave"] / "busqueda" / "configuracion_elegida.json"
        if m["clave"] == "original" and self.p1 != self.salida:          # búsqueda hecha en la parte 1 (solo lectura)
            if not previa.exists():
                raise RuntimeError(f"Falta {previa} de la parte 1 (etapa adicionales_ajuste)")
            p = leer_json(previa)[variante]["configuracion"]
        else:
            fin = PR.etapa_busqueda(ctx, espacio)
            eleg = PR.etapa_pliegues(ctx, espacio, fin)
            p = next(c for c in espacio if c["id"] == eleg[variante])
        if solo_busqueda:
            return dict(configuracion=p)
        out = []
        for i, s in enumerate(sem):
            car = carpeta / "final" / f"semilla_{s}"
            pos = self._pos_final(ctx)
            r = PR.corrida(ctx, variante, p, s, "final", pos, car, "prueba", guardar_modelo=True, medir=True)
            out.append(dict(conjunto=nombre, clave=m["clave"], semilla=s, MASE=r["metricas"]["MASE"]))
        return out

    # ------------------------------------------------------------------ ablación
    def variantes_ablacion(self) -> dict:
        idea, d, p = self.original()
        comp = d["componentes"]
        nombres = list(comp)
        out = {}
        for bits in itertools.product((1, 0), repeat=len(nombres)):
            q = dict(p)
            for n, b in zip(nombres, bits):
                v = comp[n][0] if b else comp[n][1]
                q[n] = p.get(n, comp[n][0]) if v == "config" else v
            on = [n for n, b in zip(nombres, bits) if b]
            clave = ("completo" if len(on) == len(nombres) else "ninguno" if not on else
                     ("sin_" + next(n for n, b in zip(nombres, bits) if not b)) if len(on) == len(nombres) - 1 else "solo_" + on[0])
            out[clave] = dict(red=d["red"], variante=d["variante"], config=q, bits=dict(zip(nombres, bits)))
        base = {k: p[k] for k in ("nucleo", "d_model", "capas", "cabezas", "dropout", "lr", "weight_decay", "lote") if k in p}
        if idea in ("A", "AB", "AC"):
            out["base_sin_rama"] = dict(red="tats", variante="N", config=dict(base, nucleo="patchtst", id=p["id"]), bits={})
        elif idea == "E":
            out["base_sin_rama"] = dict(red=d["red"], variante="N", config=dict(p, sin_rama=True), bits={})
        n_obj = self._n_parametros(d["red"], d["variante"], p)
        base_cap = dict(nucleo="patchtst", d_model=64, capas=2, cabezas=4, dropout=0.1, lr=p["lr"], weight_decay=p["weight_decay"], lote=p["lote"])
        base_cap.update(base, nucleo="patchtst", id=p["id"])
        cap = self._capacidad_igualada(base_cap, n_obj)
        out["capacidad_igualada"] = dict(red="tats", variante="N", config=cap, bits={}, parametros_objetivo=n_obj)
        return out

    def _n_parametros(self, red, variante, p) -> int:
        n_x, n_cal = len(self.datos.hist) + len(self.datos.nombres_mascara), len(self.datos.cal)
        tx = {k: torch.from_numpy(v) for k, v in self.texto.items()} if variante == "N+O" and self.texto is not None else None
        return redes_e3.construir(red, p, n_x, n_cal, self.datos.L, self.datos.H, tx, None, n_mascaras=len(self.datos.nombres_mascara)).n_parametros()

    def _capacidad_igualada(self, base: dict, objetivo: int) -> dict:
        mejor = None
        for dm in range(16, 513, 16):
            q = dict(base, d_model=dm, cabezas=4)
            n = self._n_parametros("tats", "N", q)
            if mejor is None or abs(n - objetivo) < abs(mejor[1] - objetivo):
                mejor = (q, n)
        return mejor[0]

    def ablacion(self) -> pd.DataFrame:
        sem = self.semillas() if self.rapido else self.semillas()[: self.plan["ablacion"]["semillas_minimas"]]
        var = self.variantes_ablacion()
        guardar_json(self.salida / "ablacion" / "variantes.json", var)
        filas = []
        for nombre, v in var.items():
            m = dict(clave=f"ablacion_{nombre}", tipo="red", red=v["red"], variante=v["variante"], config=v["config"])
            for s in sem:
                car = self.salida / "ablacion" / nombre / f"semilla_{s}"
                ctx = self.ctx(car.parent, m["clave"], m["red"], m["variante"])
                r = self.correr(m, car, s, self._pos_final(ctx), "final", "prueba", guardar=True, medir=True)
                filas.append(dict(variante=nombre, semilla=s, MASE=r["metricas"]["MASE"], n_parametros=r["n_parametros"], **{f"c_{k}": b for k, b in v["bits"].items()}))
        t = pd.DataFrame(filas)
        t.to_csv(self.salida / "ablacion" / "metricas_por_semilla.csv", index=False)
        return t

    # ------------------------------------------------------------------ sensibilidad
    def sensibilidad(self) -> pd.DataFrame:
        idea, d, p = self.original()
        ps = self.plan["sensibilidad"]
        rej = dict(ps["rejillas"]["perfiles"] if idea == "F" else ps["rejillas"]["residual"])
        if idea == "AC":
            rej.update(ps["rejillas"]["placebo"])
        sem = self.rapido["semillas_finales"] if self.rapido else ps["semillas"]
        puntos = [("centro", None, dict(p))]
        for k, vals in rej.items():
            for v in vals:
                if k == "factor_lr":
                    puntos.append((f"lr_x{v}", v, dict(p, lr=p["lr"] * v)))
                elif p.get(k) != v:
                    puntos.append((f"{k}_{v}", v, dict(p, **{k: v})))
        if self.rapido:
            puntos = puntos[:3]
        filas = []
        for nombre, valor, q in puntos:
            m = dict(clave="sensibilidad", tipo="red", red=d["red"], variante=d["variante"], config=q)
            for s in sem:
                car = self.salida / "sensibilidad" / nombre / f"semilla_{s}"
                ctx = self.ctx(car.parent, "sensibilidad", d["red"], d["variante"])
                r = self.correr(m, car, s, self._pos_pliegue(ctx, ps["pliegue"]), f"pliegue_{ps['pliegue']}", f"val_{ps['pliegue']}")
                filas.append(dict(punto=nombre, parametro=nombre.rsplit("_", 1)[0] if valor is not None else "centro", valor=valor, semilla=s, MASE=r["metricas"]["MASE"]))
        t = pd.DataFrame(filas)
        t.to_csv(self.salida / "sensibilidad" / "sensibilidad_por_semilla.csv", index=False)
        return t

    # ------------------------------------------------------------------ curva de aprendizaje
    def mejor_rival(self) -> str:
        """Rival neuronal con menor MASE en el piloto (pliegue 2021): referencia de la curva, la escalabilidad y la interpretabilidad."""
        r = self.salida / "potencia" / "analisis_potencia.csv"
        if r.exists():
            a = pd.read_csv(r)
            redes = [k for k, d in self.plan["familia"].items() if d["tipo"] == "red"]
            a = a[a.rival.isin(redes)]
            if len(a):
                return str(a.sort_values("MASE_rival_piloto").rival.iloc[0])
        return None

    def _original_y_rival(self) -> tuple[list[dict], str | None]:
        """Parte 1: solo el original (el mejor rival aún no se conoce). Parte 2: original y mejor rival del piloto."""
        rival = self.mejor_rival()
        if rival is None:
            self.log("El mejor rival se conoce después de la etapa potencia: por ahora solo el original (la parte 2 agrega el rival).")
            return self.miembros(con_rivales=False), None
        return [m for m in self.miembros() if m["clave"] in ("original", rival)], rival

    def curva(self) -> pd.DataFrame:
        pc = self.plan["curva_aprendizaje"]
        sem = self.rapido["semillas_finales"] if self.rapido else pc["semillas"]
        ms, rival = self._original_y_rival()
        filas = []
        for m in ms:
            if m["tipo"] == "mstl":
                continue
            for f in pc["fracciones"]:
                for s in sem:
                    rel = f"curva/{m['clave']}/fraccion_{f:.2f}/semilla_{s}"
                    car = self._de_parte1(m, rel, s) if m["clave"] == "original" else self.salida / rel
                    ctx = self.ctx(car.parent, m["clave"], m.get("red", "lightgbm"), m["variante"])
                    pos = dict(self._pos_pliegue(ctx, pc["pliegue"]))
                    pos["tr"] = pos["tr"][-max(32, int(round(f * len(pos["tr"])))):]
                    r = self.correr(m, car, s, pos, f"curva_{f:.2f}", f"val_{pc['pliegue']}")
                    filas.append(dict(clave=m["clave"], fraccion=f, n_ajuste=len(pos["tr"]), semilla=s, MASE=r["metricas"]["MASE"],
                                      MASE_ajuste=r.get("MASE_ajuste"), MASE_parada=r.get("MASE_parada")))
        t = pd.DataFrame(filas)
        t.to_csv(self.salida / "curva" / "curva_aprendizaje.csv", index=False)
        return t

    # ------------------------------------------------------------------ escalabilidad
    def escalabilidad(self) -> pd.DataFrame:
        import time
        from . import costes
        pe = self.plan["escalabilidad"]
        ms, rival = self._original_y_rival()
        filas = []
        n_x, n_cal = len(self.datos.hist) + len(self.datos.nombres_mascara), len(self.datos.cal)
        for m in [m for m in ms if m["tipo"] == "red"]:
            for L in pe["longitudes"]:
                tx = {k: torch.from_numpy(v) for k, v in self.texto.items()} if m["variante"] == "N+O" and self.texto is not None else None
                desc = torch.from_numpy(self.desc) if m["red"].startswith("tgforecaster") and self.desc is not None else None
                try:
                    red = redes_e3.construir(m["red"], m["config"], n_x, n_cal, L, self.datos.H, tx, desc, n_mascaras=len(self.datos.nombres_mascara)).to(self.dispositivo)
                except Exception as e:
                    filas.append(dict(clave=m["clave"], L=L, error=str(e)[:160]))
                    continue
                for B in pe["lotes"]:
                    t = dict(x=torch.randn(B, L, n_x, device=self.dispositivo), cal_p=torch.randn(B, L, n_cal, device=self.dispositivo),
                             cal_f=torch.randn(B, self.datos.H, n_cal, device=self.dispositivo), dia=torch.full((B,), 5000, dtype=torch.long, device=self.dispositivo))
                    reps = 3 if self.rapido else pe["repeticiones"]
                    red.eval()
                    with torch.no_grad():
                        red(t["x"], t["cal_p"], t["cal_f"], t["dia"])
                        if self.dispositivo == "cuda":
                            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                        t0 = time.perf_counter()
                        for _ in range(reps):
                            red(t["x"], t["cal_p"], t["cal_f"], t["dia"])
                        if self.dispositivo == "cuda":
                            torch.cuda.synchronize()
                        inf = (time.perf_counter() - t0) / reps
                    red.train()
                    t0 = time.perf_counter()
                    for _ in range(reps):
                        red.zero_grad(set_to_none=True)
                        red(t["x"], t["cal_p"], t["cal_f"], t["dia"]).abs().mean().backward()
                    if self.dispositivo == "cuda":
                        torch.cuda.synchronize()
                    ent = (time.perf_counter() - t0) / reps
                    fl = costes.flops_por_muestra(red, t).get("flops_por_origen")
                    filas.append(dict(clave=m["clave"], L=L, lote=B, parametros=red.n_parametros(), flops_por_origen=fl, segundos_inferencia_lote=inf,
                                      segundos_entrenamiento_paso=ent, memoria_pico_gpu_mb=torch.cuda.max_memory_allocated() / 2**20 if self.dispositivo == "cuda" else None))
        t = pd.DataFrame(filas)
        (self.salida / "escalabilidad").mkdir(parents=True, exist_ok=True)
        t.to_csv(self.salida / "escalabilidad" / "escalabilidad.csv", index=False)
        return t

    # ------------------------------------------------------------------ robustez
    def robustez(self) -> pd.DataFrame:
        """Perturbaciones de P2 (mismas semillas que el E2) y, en los modelos con texto, texto ausente y texto placebo."""
        datos = self.datos
        pos = datos.prueba()
        if self.rapido:
            pos = pos[:self.rapido["n_ev"]]
        escenarios = [("original", 0, 0.0)] + [(t, i + 1, s) for t in P2.SEVERIDADES for i, s in enumerate(P2.SEVERIDADES[t])]
        filas = []
        X0, M0 = datos.X, datos.M
        ms = self.miembros()
        self._esc_final = datos.ajustar_escalas(datos.ajuste_final()[0])     # LightGBM: escalas del ajuste sin perturbar
        try:
            for tipo, nivel, s in escenarios:
                datos.X, datos.M = (X0, M0) if tipo == "original" else P2._perturbar(X0, M0, tipo, s, nivel)
                for m in ms:
                    filas += self._robustez_modelo(m, pos, tipo, nivel, s)
                self.log(f"Robustez {tipo} nivel {nivel}: {len(ms)} procedimientos evaluados.")
        finally:
            datos.X, datos.M = X0, M0
        for m in [m for m in ms if m.get("variante") == "N+O"]:
            for tipo in ("texto_ausente", "texto_placebo"):
                filas += self._robustez_modelo(m, pos, tipo, 1, 1.0)
        t = pd.DataFrame(filas)
        (self.salida / "robustez").mkdir(parents=True, exist_ok=True)
        t.to_csv(self.salida / "robustez" / "robustez_por_semilla.csv", index=False)
        return t

    def cargar_red(self, m: dict, carpeta: Path, conj, texto_modo: str | None = None):
        datos = self.datos
        r = leer_json(carpeta / "resultado.json")
        tx = None
        if m["variante"] == "N+O":
            tx = {k: torch.from_numpy(v) for k, v in self.texto.items()}
            if texto_modo == "texto_ausente":
                tx = {k: (torch.full_like(v, -1) if k.startswith("lista") else v) for k, v in tx.items()}
        desc = torch.from_numpy(self.desc) if m["red"].startswith("tgforecaster") and self.desc is not None else None
        red = redes_e3.construir(m["red"], r["configuracion"], conj.x.shape[2], conj.cal_p.shape[2], datos.L, datos.H, tx, desc, n_mascaras=len(datos.nombres_mascara))
        red.load_state_dict(torch.load(carpeta / "modelo.pt", map_location="cpu", weights_only=True), strict=True)
        return red.to(self.dispositivo).eval(), r

    def _robustez_modelo(self, m, pos, tipo, nivel, s) -> list[dict]:
        datos, out = self.datos, []
        base = self.salida / "familia" / m["clave"]
        if m["tipo"] == "mstl":
            if tipo.startswith("texto"):
                return []
            r = leer_json(base / "determinista" / "resultado.json")
            yhat = clasicos._pronosticar(datos, pos, m["modelo"], int(r["configuracion"]["semanas"]), self.args.hilos)
            y = datos.Y[pos[:, None] + np.arange(datos.H)]
            return [dict(clave=m["clave"], semilla=None, tipo=tipo, nivel=nivel, severidad=s, MASE=float(np.abs(yhat - y).mean() / datos.q168))]
        for car in sorted(base.glob("semilla_*")):
            if not (car / "resultado.json").exists():
                continue
            esc = P2._escalas(leer_json(car / "escaladores.json"), datos) if (car / "escaladores.json").exists() else None
            if m["tipo"] == "lightgbm":
                import lightgbm as lgb
                if tipo.startswith("texto") or not (car / "modelo_lightgbm.txt").exists():
                    continue
                conj = datos.construir(pos, self._esc_final)
                X, _ = clasicos._tabla(conj)
                b = lgb.Booster(model_file=str(car / "modelo_lightgbm.txt"))
                yhat = (1.0 + b.predict(X).reshape(len(conj), datos.H)) * conj.nivel[:, None]
                sem = leer_json(car / "resultado.json")["semilla"]
            else:
                conj = datos.construir(pos, esc)
                red, r = self.cargar_red(m, car, conj, tipo if tipo.startswith("texto") else None)
                if tipo == "texto_placebo":
                    if not hasattr(red, "placebo_fijo"):
                        continue
                    red.placebo_fijo = True
                    t = motor.a_tensores(conj, self.dispositivo)
                    with torch.no_grad():
                        o = torch.cat([red(t["x"][i:i + 256], t["cal_p"][i:i + 256], t["cal_f"][i:i + 256], t["dia"][i:i + 256], placebo=True).cpu()
                                       for i in range(0, len(t["y"]), 256)]).numpy()
                else:
                    o = motor.predecir(red, motor.a_tensores(conj, self.dispositivo))
                yhat = o.astype("float64") * conj.nivel[:, None]
                sem = r["semilla"]
            out.append(dict(clave=m["clave"], semilla=sem, tipo=tipo, nivel=nivel, severidad=s, MASE=float(np.abs(yhat - conj.y).mean() / datos.q168)))
        return out

    # ------------------------------------------------------------------ sanidad
    def sanidad(self) -> dict:
        idea, d, p = self.original()
        ps = self.plan["sanidad"]
        m = dict(clave="sanidad_original", tipo="red", red=d["red"], variante=d["variante"], config=p)
        car = self._de_parte1(m, "sanidad/original_etiquetas_permutadas", None)
        ctx = self.ctx(car.parent, m["clave"], d["red"], d["variante"])
        r = PR.corrida(ctx, d["variante"], p, ps["semilla"], "final", self._pos_final(ctx), car, "prueba", permutar=ps["semilla_permutacion"])
        ref = self.salida / "familia" / "original" / f"semilla_{self.semillas()[0]}" / "resultado.json"
        real = leer_json(ref)["metricas"]["MASE"] if ref.exists() else None
        res = dict(MASE_etiquetas_permutadas=r["metricas"]["MASE"], MASE_modelo_real=real, razon=(r["metricas"]["MASE"] / real) if real else None,
                   criterio_razon_minima=ps["criterio_razon_minima"])
        res["supera_criterio"] = bool(res["razon"] is not None and res["razon"] >= ps["criterio_razon_minima"])
        guardar_json(self.salida / "sanidad" / "sanidad.json", res)
        return res

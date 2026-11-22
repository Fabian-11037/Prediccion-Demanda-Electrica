"""Etapas del corrector: busqueda (2018), variantes y placebo (2019-2021), estricto y evaluacion. Reanudables."""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from entrenamiento.protocolo import guardar_json, leer_json

from . import evaluacion as EV
from . import modelo as MO
from . import residuos as RS


class Corrida:
    def __init__(self, salida: Path, plan: dict, datos_num, dispositivo: str, log, rapido: dict | None, barra=None):
        self.salida, self.plan, self.dn, self.dev, self.log, self.rapido, self.barra = Path(salida), plan, datos_num, dispositivo, log, rapido, barra
        self.C = RS.calendario_diario(datos_num)
        self._cand, self._emb = {}, {}

    # ------------------------------------------------------------------ insumos
    def semillas(self):
        return self.rapido["semillas"] if self.rapido else self.plan["semillas"]

    def cr(self, s: int) -> dict:
        z = np.load(self.salida / "cr" / f"semilla_{s}.npz")
        return {k: z[k] for k in z.files}

    def candidatos(self, col: str) -> tuple[dict, torch.Tensor]:
        if col not in self._cand:
            car = self.salida / ("texto" if col == self.plan["texto"]["disponibilidad_principal"] else f"texto_{col}")
            z = np.load(car / "candidatos.npz")
            res = json.loads((car / "resumen.json").read_text(encoding="utf-8"))
            cand = dict(dias=z["dias"], cand=z["cand"], feat=z["feat"], fuente=z["fuente"], reservas={k: tuple(v) for k, v in res["reservas"].items()},
                        n_fuentes=len(res["fuentes"]))
            emb = torch.from_numpy(np.load((car / "vectores.txt").read_text(encoding="utf-8").strip()).astype("float16"))
            if len(emb) == 0:
                emb = torch.zeros(1, 768, dtype=torch.float16)
            if self.dev != "cpu" and emb.numel() * 2 < 2.5 * 2**30:
                emb = emb.to(self.dev)
            self._cand[col], self._emb[col] = cand, emb
        return self._cand[col], self._emb[col]

    def tramos(self, datos: MO.Datos, anio: int) -> dict:
        ini = self.plan["corrector"]["primer_anio_ajuste"] if not self.rapido else self.rapido["primer_anio_ajuste"]
        return dict(ajuste=datos.tramo(range(ini, anio - 1)), parada=datos.tramo([anio - 1]), evaluacion=datos.tramo([anio]))

    def _una(self, etapa: str, variante: str, p: dict, s: int, anio: int, col: str, desplazamiento: int = 0, modelo: str | None = None) -> dict:
        car = self.salida / "corrector" / etapa / variante / f"anio_{anio}" / f"semilla_{s}"
        if (car / "resultado.json").exists():
            if self.barra is not None:
                self.barra.update(1)
            return leer_json(car / "resultado.json")
        cand, emb = self.candidatos(col)
        datos = MO.Datos(self.cr(s), self.C, cand, self.dn.q168, desplazamiento)
        cfg = self.plan["corrector"]
        r = MO.entrenar(datos, cand, emb, modelo or variante, p, s, self.tramos(datos, anio), self.dev,
                        cfg["max_epocas"] if not self.rapido else 2, cfg["paciencia"])
        car.mkdir(parents=True, exist_ok=True)
        np.save(car / "delta.npy", r.pop("delta").astype("float32"))
        r.update(etapa=etapa, variante=variante, semilla=s, anio=anio, disponibilidad=col, desplazamiento=desplazamiento, configuracion=p)
        guardar_json(car / "resultado.json", r)
        if self.barra is not None:
            self.barra.update(1)
            self.barra.set_postfix_str(f"{etapa} {variante} {anio} s{s} MASE {r['mase_evaluacion']:.4f}", refresh=True)
        return r

    # ------------------------------------------------------------------ etapas
    def busqueda(self) -> dict:
        g = self.plan["corrector"]["rejilla"]
        claves = sorted(g)
        cfgs = [dict(zip(claves, v), id=i + 1) for i, v in enumerate(itertools.product(*(g[k] for k in claves)))]
        if self.rapido:
            cfgs = cfgs[:2]
        col = self.plan["texto"]["disponibilidad_principal"]
        anio = self.plan["corrector"]["anio_seleccion"] if not self.rapido else self.rapido["anio_seleccion"]
        filas = []
        for p in cfgs:
            for s in self.semillas():
                r = self._una("busqueda", f"oficial_prensa_cfg{p['id']:02d}", p, s, anio, col, modelo="oficial_prensa")
                filas.append(dict(cfg=p["id"], semilla=s, mase=r["mase_evaluacion"], **{k: p[k] for k in claves}))
        t = pd.DataFrame(filas)
        t.to_csv(self.salida / "corrector" / "busqueda" / "tabla.csv", index=False)
        mejor = int(t.groupby("cfg").mase.mean().idxmin())
        elegida = next(p for p in cfgs if p["id"] == mejor)
        ruta = self.salida / "corrector" / "busqueda" / "configuracion_elegida.json"
        if ruta.exists() and leer_json(ruta)["id"] != mejor:
            raise RuntimeError("La configuración elegida cambió al reanudar")
        guardar_json(ruta, elegida)
        self.log(f"BÚSQUEDA DEL CORRECTOR: configuración {mejor} {elegida} (MASE 2018 {t.groupby('cfg').mase.mean().min():.4f}).")
        return elegida

    def _elegida(self) -> dict:
        return leer_json(self.salida / "corrector" / "busqueda" / "configuracion_elegida.json")

    def anios_eval(self):
        return self.plan["corrector"]["anios_evaluacion"] if not self.rapido else self.rapido["anios_evaluacion"]

    def variantes(self):
        p, col = self._elegida(), self.plan["texto"]["disponibilidad_principal"]
        for v in MO.VARIANTES:
            for anio in self.anios_eval():
                for s in self.semillas():
                    self._una("variantes", v, p, s, anio, col)

    def placebo(self):
        p, col = self._elegida(), self.plan["texto"]["disponibilidad_principal"]
        for d in (self.plan["corrector"]["desplazamientos_placebo"] if not self.rapido else self.rapido["desplazamientos"]):
            for anio in self.anios_eval():
                for s in self.semillas():
                    self._una("placebo", f"desalineado_{d}", p, s, anio, col, d)

    def estricto(self):
        p, col = self._elegida(), self.plan["texto"]["disponibilidad_sensibilidad"]
        for anio in self.anios_eval():
            for s in self.semillas():
                self._una("estricto", "oficial_prensa", p, s, anio, col)

    # ------------------------------------------------------------------ evaluación
    def evaluacion(self) -> dict:
        q = self.dn.q168
        col = self.plan["texto"]["disponibilidad_principal"]
        cand, _ = self.candidatos(col)
        anios = self.anios_eval()
        err_h, base = {}, None
        desp = self.plan["corrector"]["desplazamientos_placebo"] if not self.rapido else self.rapido["desplazamientos"]
        nombres = list(MO.VARIANTES) + [f"desalineado_{d}" for d in desp]
        hay_estricto = (self.salida / "corrector" / "estricto").exists()
        if hay_estricto:
            nombres.append("estricto")
        acum = {k: [] for k in ["E", "C+R"] + nombres}
        for s in self.semillas():
            cr = self.cr(s)
            m = np.isin(cr["anio"], anios)
            if base is None:
                base = dict(dias=cr["dia"][m], anio=cr["anio"][m])
            elif not np.array_equal(base["dias"], cr["dia"][m]):
                raise RuntimeError("Las semillas no comparten los mismos orígenes")
            y, ycr = cr["y"][m], cr["yhat_cr"][m]
            acum["E"].append(np.abs(y - cr["yhat_e"][m]) / q)
            acum["C+R"].append(np.abs(y - ycr) / q)
            for v in nombres:
                etapa = "placebo" if v.startswith("desalineado") else ("estricto" if v == "estricto" else "variantes")
                carp = "oficial_prensa" if v == "estricto" else v
                d = np.concatenate([np.load(self.salida / "corrector" / etapa / carp / f"anio_{a}" / f"semilla_{s}" / "delta.npy") for a in anios])
                acum[v].append(np.abs(y - ycr * (1 + d)) / q)
        err_h = {k: np.mean(v, axis=0) for k, v in acum.items()}
        err_h["desalineado"] = np.mean([err_h[f"desalineado_{d}"] for d in desp], axis=0)
        err = {k: v.mean(1) for k, v in err_h.items()}
        an = np.asarray(base["anio"])
        dias = np.asarray(base["dias"]).astype("datetime64[D]")
        cd = np.asarray(cand["dias"]).astype("datetime64[D]")
        pos = {d: i for i, d in enumerate(cd)}
        fila = np.array([pos[d] for d in dias])
        a0, a1 = cand["reservas"]["alineados"]
        oficial_al = ((cand["cand"][fila, a0:a1] >= 0) & (cand["feat"][fila, a0:a1, 0] > 0.5)).any(1)
        subs = dict(global_=np.ones(len(an), dtype=bool), con_anuncio_oficial_alineado=oficial_al)
        B = self.plan["evaluacion"]["remuestreos"] if not self.rapido else 199
        t = EV.evaluar(err, an, {("global" if k == "global_" else k): v for k, v in subs.items() if v.sum() >= 20}, B, self.plan["evaluacion"]["bloque_dias"])
        if hay_estricto:                                       # sensibilidad: misma variante con la disponibilidad estricta (fuera de la familia)
            idx = EV.indices_bootstrap(an, B, self.plan["evaluacion"]["bloque_dias"], 20261121)
            extra = [dict(subconjunto="sensibilidad_estricta", A="oficial_prensa_estricto", B=b, n_origenes=int(len(an)),
                          **EV.contraste(err["estricto"], err[b], idx), p_holm=float("nan")) for b in ("C+R", "num_horario", "oficial_prensa")]
            t = pd.concat([t, pd.DataFrame(extra)], ignore_index=True)
        out = self.salida / "evaluacion"
        out.mkdir(exist_ok=True)
        t.to_csv(out / "contrastes.csv", index=False)
        filas = [dict(variante=k, anio=int(a), MASE=float(err[k][an == a].mean())) for k in err for a in np.unique(an)]
        pd.DataFrame(filas).to_csv(out / "mase_por_anio.csv", index=False)
        pd.concat([EV.por_hora(err_h, a, b) for a, b in [("oficial_prensa", "C+R"), ("oficial", "C+R"), ("oficial_prensa", "num_horario"),
                                                          ("oficial", "num_horario"), ("num_horario", "C+R")]]).to_csv(out / "por_hora.csv", index=False)
        ver = EV.veredicto(t, self.plan["evaluacion"]["margen_pct"])
        res = dict(veredicto=ver, n_origenes=int(len(an)), dias_con_anuncio_oficial_alineado=int(oficial_al.sum()),
                   MASE={k: float(v.mean()) for k, v in err.items()}, regla=self.plan["evaluacion"]["regla"], prueba_rapida=bool(self.rapido))
        guardar_json(out / "resultado.json", res)
        self._informe(t, res)
        return res

    def _informe(self, t: pd.DataFrame, res: dict):
        g = t[t.subconjunto == "global"]
        lin = ["# Corrector semántico por hora sobre E + C+R (desarrollo 2019-2021)", "",
               f"Orígenes evaluados: {res['n_origenes']}; con anuncio oficial alineado: {res['dias_con_anuncio_oficial_alineado']}.", "",
               "| Contraste | MASE A | MASE B | Reducción | IC 95 % | p Holm |", "|---|---|---|---|---|---|"]
        for x in t.itertuples():
            lin.append(f"| {x.subconjunto}: {x.A} frente a {x.B} | {x.MASE_A:.4f} | {x.MASE_B:.4f} | {x.reduccion_pct:.2f} % | "
                       f"[{x.ic95_inf:.2f}; {x.ic95_sup:.2f}] | {x.p_holm:.4f} |")
        lin += ["", "Veredicto preregistrado:", ""]
        for v, d in res["veredicto"].items():
            lin.append(f"- {v}: aporte del texto demostrado = {'sí' if d['aporte_del_texto_demostrado'] else 'no'}; reducción relevante (≥ 3 %) "
                       f"frente a C+R = {'sí' if d['reduccion_relevante_frente_a_CR'] else 'no'}; se descarta un aporte del texto de 3 % = "
                       f"{'sí' if d['se_descarta_aporte_de_texto_de_3pct'] else 'no'} (cota superior {d['cota_superior_ic_aporte_texto']}).")
        lin += ["", f"Regla: {res['regla']}"]
        del g
        (self.salida / "RESUMEN_CORRECTOR.md").write_text("\n".join(lin) + "\n", encoding="utf-8")

"""Etapas del paquete de cierre del Entregable 2 sobre el protocolo común (``protocolo.py``).

* ``modelo_estandar``: 20 configuraciones en el pliegue 2018, 5 finalistas en 2019-2021, 5 semillas finales.
* ``busqueda14``: búsqueda de arquitectura de §14 en dos fases con estudio piloto del ruido por semilla.
* ``ablacion``: componentes del mejor modelo numérico con las mismas cinco semillas finales.
* ``sanidad``: etiquetas permutadas en ajuste y parada (§10.1); la prueba no se toca para nada más.
Todas las corridas son reanudables: si una carpeta ya tiene ``resultado.json`` no se repite.
"""
from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from . import protocolo as PR
from .protocolo import corrida, guardar_json

ANCHOS = (16, 32, 64, 128, 256)


# ------------------------------------------------------------------------------------ utilidades
def arquitecturas() -> list[dict]:
    """A1 de §14.4: las 1000 arquitecturas, producto cartesiano de W^3 y {0,1}^3, con identificador estable."""
    out = []
    for i, (w, b) in enumerate(itertools.product(itertools.product(ANCHOS, repeat=3), itertools.product((0, 1), repeat=3))):
        out.append(dict(arq=f"A{i:04d}", anchos=list(w), patron_dropout=list(b)))
    assert len(out) == 1000 and sum(1 for a in out if sum(a["patron_dropout"]) == 0) == 125
    return out


def _pos(ctx, anio: int) -> dict:
    f = next(x for x in ctx.datos.pliegues() if x["anio"] == anio)
    return ctx.recortar(dict(tr=f["tr"], es=f["es"], ev=f["val"]))


def _pos_final(ctx) -> dict:
    tr, es = ctx.datos.ajuste_final()
    return ctx.recortar(dict(tr=tr, es=es, ev=ctx.datos.prueba()))


def _fila(r: dict, extra: dict | None = None) -> dict:
    f = PR._fila(r)
    f.update(extra or {})
    for k in ("anchos", "patron_dropout"):
        if k in f and isinstance(f[k], list):
            f[k] = "-".join(map(str, f[k]))
    return f


# ------------------------------------------------------------------------------------ modelo estándar
def modelo_estandar(ctx, configs: list[dict]) -> dict:
    finalistas = PR.etapa_busqueda(ctx, configs)
    elegidas = PR.etapa_pliegues(ctx, configs, finalistas)
    return finales_con_costes(ctx, configs, elegidas)


def finales_con_costes(ctx, configs: list[dict], elegidas: dict) -> dict:
    """Etapa final del protocolo común; además mide costes (FLOPs, memoria, energía, latencia) en cada corrida."""
    original = PR.corrida

    def con_medicion(*a, **k):
        k.setdefault("medir", True)
        return original(*a, **k)

    PR.corrida = con_medicion
    try:
        return PR.etapa_final(ctx, configs, elegidas)
    finally:
        PR.corrida = original


# ------------------------------------------------------------------------------------ búsqueda de §14
class Busqueda14:
    def __init__(self, ctx, plan: dict, modelo: dict):
        self.ctx, self.plan, self.m = ctx, plan, modelo
        self.raiz = ctx.salida
        self.vb = modelo["variante_busqueda"]
        self.control = dict(anchos=list(plan["control"]["anchos"]), patron_dropout=list(plan["control"]["patron_dropout"]))
        self.f1 = plan["fase1"]
        self.rapido = ctx.rapido

    # fase 0: hiperparámetros con la arquitectura de control (§14.3)
    def fase0(self) -> dict:
        r0 = self.raiz / "fase0_hiperparametros"
        if "fija" in self.m["fase0"]:
            p = dict(self.m["fase0"]["fija"])
            r0.mkdir(parents=True, exist_ok=True)
            guardar_json(r0 / "elegida.json", dict(configuracion=p, origen=self.m["fase0"]["origen"]))
            return p
        configs = [dict(c, **self.control) for c in self.m["fase0"]["configuraciones"]]
        if self.rapido:
            configs = [c for c in configs if c["id"] in self.rapido["configuraciones"]]
        sub = PR.Contexto(self.ctx.raiz, r0, dict(self.ctx.cfg, variantes=[self.vb]), self.ctx.datos, self.ctx.dispositivo, self.ctx.log, self.rapido)
        self._heredar(sub)
        finalistas = PR.etapa_busqueda(sub, configs)
        elegidas = PR.etapa_pliegues(sub, configs, finalistas)
        p = next(c for c in configs if c["id"] == elegidas[self.vb])
        guardar_json(r0 / "elegida.json", dict(configuracion=p, regla="protocolo común con la arquitectura de control"))
        return {k: v for k, v in p.items() if k not in ("anchos", "patron_dropout")}

    def _heredar(self, sub):
        sub.texto, sub.descripciones, sub.barra = self.ctx.texto, self.ctx.descripciones, self.ctx.barra
        sub.mlflow, sub.mlflow_raiz = getattr(self.ctx, "mlflow", False), getattr(self.ctx, "mlflow_raiz", self.ctx.salida)

    def _p(self, base: dict, a: dict, ident: int) -> dict:
        return dict(base, id=ident, arq=a["arq"], anchos=a["anchos"], patron_dropout=a["patron_dropout"])

    def _opciones_f1(self) -> dict:
        if self.rapido:
            return {}
        return dict(max_epocas=self.f1["max_epocas"], min_epocas=self.f1["min_epocas"], paciencia=self.f1["paciencia"])

    def fase1(self, base: dict) -> pd.DataFrame:
        arqs = arquitecturas()
        if self.rapido:
            arqs = [a for a in arqs if a["arq"] in self.rapido["arquitecturas"]]
        pos = _pos(self.ctx, self.f1["pliegue"])
        filas = []
        for i, a in enumerate(arqs):
            p = self._p(base, a, 1000 + i)
            c = self.raiz / "fase1" / a["arq"]
            r = corrida(self.ctx, self.vb, p, self.f1["semilla"], f"pliegue_{self.f1['pliegue']}", pos, c, f"val_{self.f1['pliegue']}", opciones=self._opciones_f1())
            filas.append(_fila(r, dict(arq=a["arq"])))
        t = pd.DataFrame(filas).sort_values(["MASE", "n_parametros", "arq"]).reset_index(drop=True)
        t["puesto"] = np.arange(1, len(t) + 1)
        t.to_csv(self.raiz / "fase1" / "resultados_fase1.csv", index=False)
        return t

    def piloto(self, base: dict, t1: pd.DataFrame) -> dict:
        """Ruido por semilla con el presupuesto de la fase 1 sobre arquitecturas fijadas a priori."""
        pl = self.plan["piloto"]
        arqs = {a["arq"]: a for a in arquitecturas()}
        ids = [a for a in pl["arquitecturas"] if a in arqs]
        if self.rapido:
            ids = [a for a in ids if a in self.rapido["arquitecturas"]][:2]
        semillas = pl["semillas"][:2] if self.rapido else pl["semillas"]
        pos = _pos(self.ctx, self.f1["pliegue"])
        filas = []
        for a in ids:
            for s in semillas:
                r = corrida(self.ctx, self.vb, self._p(base, arqs[a], 900), s, f"pliegue_{self.f1['pliegue']}", pos, self.raiz / "piloto" / a / f"semilla_{s}",
                            f"val_{self.f1['pliegue']}", opciones=self._opciones_f1())
                filas.append(dict(arq=a, semilla=s, MASE=r["metricas"]["MASE"]))
        t = pd.DataFrame(filas)
        t.to_csv(self.raiz / "piloto" / "piloto_ruido.csv", index=False)
        ruido = float(t.groupby("arq").MASE.std(ddof=1).mean())
        dispersion = float(t1.MASE.std(ddof=1))
        razon = ruido / dispersion if dispersion > 0 else float("inf")
        regla = self.plan["fase2"]["k_regla"]
        k = regla["k_bajo"] if razon <= regla["umbral_razon"] else regla["k_alto"]
        if self.rapido:
            k = self.rapido["k"]
        d = dict(ruido_semilla_desv_media=ruido, dispersion_entre_arquitecturas=dispersion, razon=razon, k=int(k), regla=regla)
        guardar_json(self.raiz / "piloto" / "eleccion_k.json", d)
        return d

    def fase2(self, base: dict, t1: pd.DataFrame, k: int) -> pd.DataFrame:
        arqs = {a["arq"]: a for a in arquitecturas()}
        ctrl = next(a["arq"] for a in arqs.values() if a["anchos"] == self.control["anchos"] and a["patron_dropout"] == self.control["patron_dropout"])
        finalistas = list(t1.arq.head(k))
        if ctrl not in finalistas:
            finalistas.append(ctrl)
        semillas = self.rapido["semillas_finales"] if self.rapido else self.plan["fase2"]["semillas"]
        pos = _pos(self.ctx, self.f1["pliegue"])
        filas = []
        for a in finalistas:
            for s in semillas:
                r = corrida(self.ctx, self.vb, self._p(base, arqs[a], 2000), s, f"pliegue_{self.f1['pliegue']}", pos, self.raiz / "fase2" / a / f"semilla_{s}",
                            f"val_{self.f1['pliegue']}")
                filas.append(_fila(r, dict(arq=a, control=a == ctrl)))
        t = pd.DataFrame(filas)
        t.to_csv(self.raiz / "fase2" / "resultados_fase2_por_semilla.csv", index=False)
        return t

    def seleccionar(self, t1: pd.DataFrame, t2: pd.DataFrame) -> dict:
        """Regla de §14.5 declarada a priori: menor MASE medio; empate técnico (diferencia relativa menor que el
        margen) -> menos parámetros; después menor identificador."""
        margen = self.plan["seleccion"]["margen_relativo"]
        r = (t2.groupby("arq").agg(MASE_medio=("MASE", "mean"), MASE_desv=("MASE", "std"), semillas=("MASE", "size"), n_parametros=("n_parametros", "first"),
                                   brecha_ajuste_parada=("MASE_parada", "mean"), MASE_ajuste=("MASE_ajuste", "mean"), segundos=("segundos", "mean"),
                                   control=("control", "first")).reset_index().sort_values(["MASE_medio", "n_parametros", "arq"]).reset_index(drop=True))
        r["brecha_ajuste_parada"] = r["brecha_ajuste_parada"] - r["MASE_ajuste"]
        mejor = float(r.MASE_medio.iloc[0])
        r["empate_tecnico"] = r.MASE_medio <= mejor * (1 + margen)
        # Conjunto indistinguible por TOST (Welch) con margen relativo +-3 % respecto del mejor
        ref = t2.loc[t2.arq == r.arq.iloc[0], "MASE"].to_numpy()
        tost = []
        for a in r.arq:
            v = t2.loc[t2.arq == a, "MASE"].to_numpy()
            if a == r.arq.iloc[0] or len(v) < 2:
                tost.append(np.nan)
                continue
            m = margen * mejor
            p1 = stats.ttest_ind(v, ref - m, equal_var=False, alternative="greater").pvalue     # H0: media(v) - media(ref) <= -m
            p2 = stats.ttest_ind(v, ref + m, equal_var=False, alternative="less").pvalue        # H0: media(v) - media(ref) >= +m
            tost.append(float(max(p1, p2)))
        r["p_TOST_frente_al_mejor"] = tost
        r["indistinguible_TOST"] = [True if a == r.arq.iloc[0] else (p < 0.05) for a, p in zip(r.arq, tost)]
        cand = r.loc[r.empate_tecnico].sort_values(["n_parametros", "MASE_medio", "arq"])
        gan = cand.iloc[0]
        r["ganadora"] = r.arq == gan.arq
        rf1 = t1.set_index("arq").puesto
        rho = stats.spearmanr(rf1.reindex(r.arq).to_numpy(), r.MASE_medio.rank().to_numpy()).statistic if len(r) > 2 else np.nan
        r.to_csv(self.raiz / "fase2" / "resumen_finalistas.csv", index=False)
        arqs = {a["arq"]: a for a in arquitecturas()}
        a = arqs[gan.arq]
        borde = any(w in (ANCHOS[0], ANCHOS[-1]) for w in a["anchos"]) and len(set(a["anchos"])) == 1
        d = dict(ganadora=gan.arq, anchos=a["anchos"], patron_dropout=a["patron_dropout"], MASE_medio_fase2=float(gan.MASE_medio),
                 mejor_media=r.arq.iloc[0], tamano_empate_tecnico=int(r.empate_tecnico.sum()), tamano_indistinguible_TOST=int(r.indistinguible_TOST.sum()),
                 correlacion_rangos_fase1_fase2_spearman=None if np.isnan(rho) else float(rho), en_el_borde_del_espacio=bool(borde),
                 regla=self.plan["seleccion"])
        guardar_json(self.raiz / "fase2" / "seleccion.json", d)
        return d

    def final(self, base: dict, sel: dict) -> dict:
        p = dict(base, id=3000, arq=sel["ganadora"], anchos=sel["anchos"], patron_dropout=sel["patron_dropout"])
        sub = PR.Contexto(self.ctx.raiz, self.raiz, dict(self.ctx.cfg, variantes=self.m["variantes"]), self.ctx.datos, self.ctx.dispositivo, self.ctx.log, self.rapido)
        self._heredar(sub)
        return finales_con_costes(sub, [p], {v: 3000 for v in self.m["variantes"]})

    def analisis(self, t1: pd.DataFrame, t2: pd.DataFrame) -> dict:
        """§14.6: superficie, efectos principales e interacciones, pruebas por factor, Pareto y sensibilidad."""
        import statsmodels.formula.api as smf
        from statsmodels.stats.anova import anova_lm
        from statsmodels.stats.multitest import multipletests
        out_dir = self.raiz / "analisis_espacio"
        out_dir.mkdir(parents=True, exist_ok=True)
        d = t1.copy()
        w = d.anchos.str.split("-", expand=True).astype(int)
        b = d.patron_dropout.str.split("-", expand=True).astype(int)
        for i in range(3):
            d[f"w{i + 1}"], d[f"b{i + 1}"] = w[i].astype(str), b[i]
        d["log2_ancho_total"] = np.log2(w.sum(axis=1))
        res = {}
        if len(d) >= 30:
            mod = smf.ols("MASE ~ C(w1) + C(w2) + C(w3) + b1 + b2 + b3 + b1:b2 + b1:b3 + b2:b3", data=d).fit()
            tab = anova_lm(mod, typ=2)
            tab["eta2_parcial"] = tab["sum_sq"] / (tab["sum_sq"] + tab.loc["Residual", "sum_sq"])
            pv = tab["PR(>F)"].drop("Residual")
            tab.loc[pv.index, "p_Holm"] = multipletests(pv.to_numpy(), method="holm")[1]
            tab.to_csv(out_dir / "anova_factores.csv")
            coef = pd.DataFrame(dict(coef=mod.params, ic95_inf=mod.conf_int()[0], ic95_sup=mod.conf_int()[1], p=mod.pvalues))
            coef.to_csv(out_dir / "efectos_principales_e_interacciones.csv")
            res["R2_modelo_factores"] = float(mod.rsquared)
            tend = smf.ols("MASE ~ log2_ancho_total", data=d).fit()
            res["tendencia_log2_ancho"] = dict(pendiente=float(tend.params["log2_ancho_total"]), p=float(tend.pvalues["log2_ancho_total"]))
            # sin dropout frente a cada patrón con dropout, pareado por la combinación de anchos (125 bloques)
            filas = []
            d["combo"] = d.anchos
            base = d[(d.b1 == 0) & (d.b2 == 0) & (d.b3 == 0)].set_index("combo").MASE
            for patron, g in d.groupby(d.patron_dropout):
                if patron == "0-0-0":
                    continue
                v = g.set_index("combo").MASE.reindex(base.index)
                dif = (v - base).dropna()
                if len(dif) > 5:
                    wt = stats.wilcoxon(dif)
                    delta = float(np.sign(dif.to_numpy()).mean())     # > 0: el dropout empeora más veces de las que mejora
                    filas.append(dict(patron=patron, n=len(dif), mediana_diferencia=float(dif.median()), p_wilcoxon=float(wt.pvalue), delta_signo=delta))
            if filas:
                tw = pd.DataFrame(filas)
                tw["p_Holm"] = multipletests(tw.p_wilcoxon, method="holm")[1]
                tw.to_csv(out_dir / "dropout_frente_a_sin_dropout.csv", index=False)
        # superficie: un mapa por patrón (w1 x (w2,w3)) como tabla
        d.pivot_table(index=["patron_dropout", "w1"], columns=["w2", "w3"], values="MASE").to_csv(out_dir / "superficie_fase1.csv")
        # Pareto (validación frente a parámetros y tiempo)
        def pareto(df, x):
            df = df.sort_values([x, "MASE"])
            return df[df.MASE <= df.MASE.cummin()].arq.tolist()
        res["pareto_parametros"] = pareto(d, "n_parametros")
        res["pareto_segundos"] = pareto(d, "segundos")
        if len(t2):
            res["sensibilidad"] = dict(desv_entre_arquitecturas_fase1=float(t1.MASE.std(ddof=1)),
                                       desv_entre_semillas_fase2=float(t2.groupby("arq").MASE.std(ddof=1).mean()))
            res["sensibilidad"]["conclusion_arquitectura_irrelevante"] = res["sensibilidad"]["desv_entre_arquitecturas_fase1"] <= res["sensibilidad"]["desv_entre_semillas_fase2"]
        guardar_json(out_dir / "resumen_analisis.json", res)
        self._figuras(d, out_dir)
        return res

    def _figuras(self, d: pd.DataFrame, out_dir: Path):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception:
            return
        pats = sorted(d.patron_dropout.unique())
        fig, axes = plt.subplots(2, 4, figsize=(18, 8), squeeze=False)
        for ax, pt in zip(axes.flat, pats):
            g = d[d.patron_dropout == pt].copy()
            g["col"] = g.w2 + "-" + g.w3
            m = g.pivot_table(index="w1", columns="col", values="MASE")
            m = m.reindex(index=[str(a) for a in ANCHOS], columns=[f"{a}-{c}" for a in ANCHOS for c in ANCHOS])
            im = ax.imshow(m.to_numpy(), aspect="auto", cmap="viridis_r")
            ax.set_title(f"dropout {pt}", fontsize=9)
            ax.set_yticks(range(5), [str(a) for a in ANCHOS], fontsize=7)
            ax.set_xticks(range(0, 25, 5), [f"w2={a}" for a in ANCHOS], fontsize=7)
            fig.colorbar(im, ax=ax, fraction=0.04)
        fig.suptitle("Fase 1: MASE de validación 2018 por anchos (filas w1; columnas w2-w3) y patrón de dropout")
        fig.tight_layout()
        fig.savefig(out_dir / "superficie_fase1.png", dpi=110)
        plt.close(fig)

    def ejecutar(self) -> dict:
        self.raiz.mkdir(parents=True, exist_ok=True)
        base = self.fase0()
        t1 = self.fase1(base)
        k = self.piloto(base, t1)["k"]
        t2 = self.fase2(base, t1, k)
        sel = self.seleccionar(t1, t2)
        self.analisis(t1, t2)
        fin = self.final(base, sel)
        guardar_json(self.raiz / "resumen_busqueda14.json", dict(hiperparametros_fijos=base, seleccion=sel, prueba=fin["variantes"]))
        return fin


# ------------------------------------------------------------------------------------ ablación y sanidad
def ablacion(ctx, plan: dict) -> pd.DataFrame:
    pos = _pos_final(ctx)
    semillas = ctx.rapido["semillas_finales"] if ctx.rapido else plan["semillas"]
    filas, perdidas = [], {}
    for j, ab in enumerate(plan["ablaciones"]):
        p = dict(plan["configuracion_base"], id=4000 + j, ablacion=ab)
        for s in semillas:
            c = ctx.salida / ab / f"semilla_{s}"
            r = corrida(ctx, "N", p, s, "final", pos, c, "prueba", guardar_modelo=True, medir=True)
            filas.append(_fila(r, dict(ablacion=ab)))
            perdidas.setdefault(ab, []).append(np.load(c / "pred_prueba.npz")["mase_por_origen"])
    t = pd.DataFrame(filas)
    t.to_csv(ctx.salida / "metricas_por_semilla.csv", index=False)
    inf = ctx.cfg["inferencia"]
    rem = ctx.rapido["remuestreos"] if ctx.rapido else inf["remuestreos"]
    media = {a: np.mean(v, axis=0) for a, v in perdidas.items()}
    contrastes = []
    for ab in plan["ablaciones"]:
        if ab == "completo":
            continue
        dif = media[ab] - media["completo"]
        bb = PR.bootstrap_bloques(dif, media["completo"], inf["bloque_dias"], rem, inf["semilla_bootstrap"])
        sa, sc = t.loc[t.ablacion == ab, "MASE"].to_numpy(), t.loc[t.ablacion == "completo", "MASE"].to_numpy()
        contrastes.append(dict(ablacion=ab, MASE_medio=float(sa.mean()), MASE_completo=float(sc.mean()), **{k: v for k, v in bb.items() if k != "remuestreos"},
                               t_pareado_semillas_p=float(stats.ttest_rel(sa, sc).pvalue) if len(sa) == len(sc) and len(sa) > 1 else None))
    tc = pd.DataFrame(contrastes)
    if len(tc):
        from statsmodels.stats.multitest import multipletests
        tc["p_bootstrap_Holm"] = multipletests(tc.p_bilateral_bootstrap, method="holm")[1]
    tc.to_csv(ctx.salida / "contrastes_ablacion.csv", index=False)
    pd.DataFrame(dict(origen=pd.DatetimeIndex(ctx.datos.indice[pos["ev"]]).strftime("%Y-%m-%d"), **{f"MASE_{a}": v for a, v in media.items()})).to_csv(
        ctx.salida / "perdida_por_origen.csv", index=False)
    return tc


def sanidad(ctx, plan: dict) -> pd.DataFrame:
    """Entrena con etiquetas permutadas y evalúa en la prueba; el error debe quedar lejos del modelo real."""
    pos = _pos_final(ctx)
    filas = []
    for caso in plan["casos"]:
        sub = PR.Contexto(ctx.raiz, ctx.salida, dict(ctx.cfg, paquete=dict(ctx.cfg["paquete"], red=caso["red"])), ctx.datos, ctx.dispositivo, ctx.log, ctx.rapido)
        sub.barra, sub.mlflow, sub.mlflow_raiz = ctx.barra, getattr(ctx, "mlflow", False), getattr(ctx, "mlflow_raiz", ctx.salida)
        r = corrida(sub, "N", dict(caso["configuracion"]), plan["semilla"], "final", pos, ctx.salida / caso["nombre"], "prueba", permutar=plan["semilla_permutacion"])
        filas.append(dict(caso=caso["nombre"], MASE_prueba_etiquetas_permutadas=r["metricas"]["MASE"], MASE_modelo_real_referencia=caso["MASE_real"],
                          razon=r["metricas"]["MASE"] / caso["MASE_real"], epocas=r["epocas"]))
    t = pd.DataFrame(filas)
    t["supera_criterio"] = t.razon >= plan["criterio_razon_minima"]
    t.to_csv(ctx.salida / "sanidad_etiquetas_permutadas.csv", index=False)
    return t


# ------------------------------------------------------------------------------------ robustez de los modelos nuevos
SEVERIDADES = {"ruido_demanda": [0.01, 0.03, 0.05], "atipicos": [1, 3, 6], "faltantes": [0.10, 0.30, 0.50]}
SEMILLA_ROBUSTEZ = 20261118


def _perturbar(X: np.ndarray, M: np.ndarray, tipo: str, s: float, nivel: int):
    """Mismas perturbaciones y semillas que scripts/e2c_robustez.py del repositorio (misma longitud del panel)."""
    g = np.random.default_rng(SEMILLA_ROBUSTEZ + nivel + 10 * len(tipo))
    X, M = X.copy(), M.copy()
    if tipo == "ruido_demanda":
        X[:, 0] = X[:, 0] * (1 + s * g.standard_normal(len(X)))
    elif tipo == "atipicos":
        m = g.random(len(X)) < s / 168
        X[m, 0] = X[m, 0] * 1.2
    elif tipo == "faltantes":
        f = g.random(len(X)) < s
        X[np.ix_(f, np.arange(1, X.shape[1]))] = np.nan
        M[f] = 1.0
    return X, M


def _escalas(js: dict, datos) -> dict:
    h, c = js["columnas_historicas"], js["columnas_calendario"]
    assert h == datos.hist and c == datos.cal
    return dict(mediana=np.array([js["mediana"][k] for k in h]), media=np.array([js["media"][k] for k in h]), std=np.array([js["std"][k] for k in h]),
                media_cal=np.array([js["media_calendario"][k] for k in c]), std_cal=np.array([js["std_calendario"][k] for k in c]), sigma_demanda=js["sigma_demanda"])


def robustez(raiz_res: Path, datos, red_de: dict, texto, descripciones, dispositivo: str, log, rapido: dict | None = None) -> pd.DataFrame:
    """Inferencia con los pesos finales guardados bajo tres tipos de perturbación y tres severidades."""
    import torch
    from . import motor, redes_e2
    pos = datos.prueba()
    if rapido:
        pos = pos[:rapido["n_ev"]]
    filas = []
    corridas = [p for p in sorted(Path(raiz_res).glob("*/final/*/semilla_*/modelo.pt")) if p.parts[-5] in red_de]
    escenarios = [("original", 0, 0.0)] + [(t, i + 1, s) for t in SEVERIDADES for i, s in enumerate(SEVERIDADES[t])]
    X0, M0 = datos.X, datos.M
    try:
        for tipo, nivel, s in escenarios:
            datos.X, datos.M = (X0, M0) if tipo == "original" else _perturbar(X0, M0, tipo, s, nivel)
            for ck in corridas:
                exp, var = ck.parts[-5], ck.parts[-3].replace("_", "+")
                r = PR.leer_json(ck.parent / "resultado.json")
                esc = _escalas(PR.leer_json(ck.parent / "escaladores.json"), datos)
                conj = datos.construir(pos, esc)
                tx = dict(vectores=torch.from_numpy(texto["vectores"]), lista=torch.from_numpy(texto["lista"])) if var == "N+O" else None
                desc = torch.from_numpy(descripciones) if descripciones is not None else None
                red = redes_e2.construir(red_de[exp], r["configuracion"], conj.x.shape[2], conj.cal_p.shape[2], datos.L, datos.H, tx, desc)
                red.load_state_dict(torch.load(ck, map_location="cpu", weights_only=True), strict=True)
                red = red.to(dispositivo)
                yhat = motor.predecir(red, motor.a_tensores(conj, dispositivo)).astype("float64") * conj.nivel[:, None]
                filas.append(dict(clave=f"{exp}__{var}", semilla=r["semilla"], tipo=tipo, nivel=nivel, severidad=s,
                                  MASE=float(np.abs(yhat - conj.y).mean() / datos.q168)))
            log(f"Robustez {tipo} nivel {nivel}: {len(corridas)} modelos evaluados.")
    finally:
        datos.X, datos.M = X0, M0
    t = pd.DataFrame(filas)
    t.to_csv(Path(raiz_res) / "robustez_nuevos_por_semilla.csv", index=False)
    return t


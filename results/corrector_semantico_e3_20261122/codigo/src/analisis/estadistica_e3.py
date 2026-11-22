"""Bloque D del Entregable 3: verificación estadística del modelo original frente a la familia completa.

Solo lee las predicciones guardadas por las etapas de entrenamiento; no entrena ni elige nada con la prueba.
Tablas (carpeta ``estadistica``):
  tabla_maestra.csv           una fila por conjunto, procedimiento y semilla: métricas, épocas, parámetros y costes
  resumen_por_modelo.csv      media ± DE, IC 95 % (bootstrap de semillas y de bloques de días), CV, costes
  contrastes_r3.csv           plantilla A.4: original frente a cada rival con DM-HLN, bootstrap de bloques y t pareada por
                              semillas (las tres exigidas) más Wilcoxon, t de Nadeau-Bengio y bootstrap bayesiano con ROPE;
                              p crudo y Holm sobre los m rivales; diferencia e IC 95 %; δ de Cliff y d de Cohen; decisión
  veredicto_r3.json           las seis condiciones de §8.4 y la decisión de unión-intersección
  mcs.csv, friedman_nemenyi.json, diagrama_diferencia_critica.png
  potencia_final.json         σ_d con las R semillas, diferencia mínima detectable, potencia por contraste y conjunta
  sensibilidad_conclusion.csv la decisión R3 con pruebas, métrica secundaria (RMSE) y subconjuntos de semillas alternativos
  efectos_mixtos.json         y = µ + α_modelo + u_bloque + v_semilla (MixedLM)
  adicionales_*.csv/json      tabla por conjunto, contrastes con Holm, consistencia (≥ 2 de 3) y prueba entre conjuntos
                              (Friedman con Iman-Davenport, Wilcoxon-Holm y prueba bayesiana de rangos con signo con ROPE)
  adecuacion_original.json    Tablas A y B del original: residuos (Ljung-Box, ARCH-LM, BDS, Jarque-Bera), Mincer-Zarnowitz,
                              brecha ajuste-parada, consistencia validación-prueba, ingenuos, sanidad, Giacomini-Rossi
  incertidumbre.csv           ensamble de semillas, conformal adaptativo (ACI) y ACI condicionado por texto (idea D):
                              cobertura total y por subgrupo, anchura, Winkler, Kupiec, Christoffersen, CRPS y PIT
  ablacion_*.csv, robustez_resumen.csv, pareto.csv, errores_*.csv, sensibilidad_resumen.csv, curva_resumen.csv,
  escalabilidad_ajuste.csv y RESUMEN_E3.md
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ALFA = 0.05


# ====================================================================================== utilidades
def _holm(p) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    if ok.any():
        from statsmodels.stats.multitest import multipletests
        out[ok] = multipletests(p[ok], method="holm")[1]
    return out


def _guardar(sal: Path, nombre: str, obj):
    sal.mkdir(parents=True, exist_ok=True)
    if isinstance(obj, pd.DataFrame):
        obj.to_csv(sal / nombre, index=False)
    else:
        (sal / nombre).write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=lambda o: float(o) if isinstance(o, np.floating) else str(o)), encoding="utf-8")


def indices_bloques(n: int, largo: int, b: int, semilla: int) -> np.ndarray:
    g = np.random.default_rng(semilla)
    nb = int(np.ceil(n / largo))
    return ((g.integers(0, n, size=(b, nb))[:, :, None] + np.arange(largo)[None, None, :]) % n).reshape(b, -1)[:, :n]


def dm_hln(d: np.ndarray, rezagos: int = 13) -> tuple[float, float, float]:
    """Diebold-Mariano con varianza HAC de Bartlett y corrección HLN (h = 1). Devuelve (estadístico, p bilateral, p unilateral H1: media < 0)."""
    T = len(d)
    u = d - d.mean()
    lrv = u @ u / T
    for k in range(1, rezagos + 1):
        lrv += 2 * (1 - k / (rezagos + 1)) * (u[k:] @ u[:-k]) / T
    if lrv <= 0:
        return float("nan"), float("nan"), float("nan")
    est = d.mean() / math.sqrt(lrv / T) * math.sqrt((T - 1) / T)
    return float(est), float(2 * stats.t.sf(abs(est), T - 1)), float(stats.t.cdf(est, T - 1))


def bootstrap(d: np.ndarray, base: np.ndarray, bloque: int, b: int, semilla: int) -> dict:
    """Bootstrap circular por bloques de la diferencia media; p unilateral con el estadístico centrado."""
    ix = indices_bloques(len(d), bloque, b, semilla)
    md, mb = d[ix].mean(1), base[ix].mean(1)
    rel = 100 * md / mb
    d0 = d.mean()
    return dict(diferencia=float(d0), ic95=[float(x) for x in np.percentile(md, [2.5, 97.5])], diferencia_relativa_pct=float(100 * d0 / base.mean()),
                ic95_relativa_pct=[float(x) for x in np.percentile(rel, [2.5, 97.5])], ic90_relativa_pct=[float(x) for x in np.percentile(rel, [5, 95])],
                p_unilateral=float(((md - d0) <= d0).mean()), p_bilateral=float((np.abs(md - d0) >= abs(d0)).mean()))


def bayes_bloques(d: np.ndarray, ref: float, rope: float, bloque: int, b: int = 4000, semilla: int = 20261118) -> dict:
    nb = int(np.ceil(len(d) / bloque))
    medias = np.array([d[i * bloque:(i + 1) * bloque].mean() for i in range(nb)])
    n_i = np.array([len(d[i * bloque:(i + 1) * bloque]) for i in range(nb)])
    w = np.random.default_rng(semilla).dirichlet(np.ones(nb), size=b) * n_i
    m = (w * medias).sum(1) / w.sum(1)
    r = rope * ref
    return dict(P_mejor=float((m < -r).mean()), P_equivalente=float((np.abs(m) <= r).mean()), P_peor=float((m > r).mean()))


def cliff(a: np.ndarray, b: np.ndarray) -> float:
    """δ de Cliff con signo: positivo si a tiende a ser menor (mejor) que b."""
    bs = np.sort(b)
    menores = np.searchsorted(bs, a, side="left")
    mayores = len(b) - np.searchsorted(bs, a, side="right")
    return float((mayores.sum() - menores.sum()) / (len(a) * len(b)))


def mcs(L: pd.DataFrame, alfa: float, b: int, bloque: int, semilla: int) -> pd.DataFrame:
    """Model Confidence Set (Hansen, Lunde y Nason, 2011) con T_max y bootstrap circular por bloques."""
    vivos, salida, p_acum = list(L.columns), [], 0.0
    idx = indices_bloques(len(L), bloque, b, semilla + 7)
    X = L.to_numpy()
    while len(vivos) > 1:
        x = X[:, [L.columns.get_loc(c) for c in vivos]]
        dd = x - x.mean(1, keepdims=True)
        dbar = dd.mean(0)
        boot = dd[idx].mean(1)
        var = ((boot - dbar) ** 2).mean(0)
        t = dbar / np.sqrt(var)
        p = float((((boot - dbar) / np.sqrt(var)).max(1) >= t.max()).mean())
        p_acum = max(p_acum, p)
        peor = vivos[int(np.argmax(t))]
        salida.append(dict(procedimiento=peor, p_MCS=p_acum))
        if p_acum >= alfa:
            break
        vivos.remove(peor)
    salida += [dict(procedimiento=c, p_MCS=1.0) for c in vivos if c not in [s["procedimiento"] for s in salida]]
    t = pd.DataFrame(salida)
    t["en_el_conjunto"] = t.p_MCS >= alfa
    return t


def friedman(M: np.ndarray, nombres: list[str]) -> dict:
    """M: (bloques, modelos) de pérdidas. Friedman, corrección de Iman-Davenport y diferencia crítica de Nemenyi."""
    N, k = M.shape
    R = np.apply_along_axis(stats.rankdata, 1, M)
    rm = R.mean(0)
    chi = 12 * N / (k * (k + 1)) * (np.sum(rm ** 2) - k * (k + 1) ** 2 / 4)
    ff = (N - 1) * chi / (N * (k - 1) - chi) if N * (k - 1) - chi > 0 else float("inf")
    q = stats.studentized_range.ppf(1 - ALFA, k, np.inf) / math.sqrt(2)
    cd = q * math.sqrt(k * (k + 1) / (6 * N))
    return dict(N=int(N), k=int(k), chi2=float(chi), p_friedman=float(stats.chi2.sf(chi, k - 1)), F_iman_davenport=float(ff),
                p_iman_davenport=float(stats.f.sf(ff, k - 1, (k - 1) * (N - 1))), diferencia_critica_nemenyi=float(cd),
                rango_medio=dict(zip(nombres, map(float, rm))))


def bayes_rangos_signo(z: np.ndarray, rope: float, b: int = 20000, semilla: int = 3) -> dict:
    """Prueba bayesiana de rangos con signo (Benavoli et al., 2017) sobre diferencias relativas entre conjuntos (z < 0 favorece al original)."""
    z = np.r_[0.0, np.asarray(z, dtype=float)]           # pseudo-observación del proceso de Dirichlet
    w = np.random.default_rng(semilla).dirichlet(np.ones(len(z)), size=b)
    s = (z[:, None] + z[None, :]) / 2
    izq, med, der = (s < -rope), (np.abs(s) <= rope), (s > rope)
    pl = np.einsum("bi,ij,bj->b", w, izq, w)
    pr = np.einsum("bi,ij,bj->b", w, med, w)
    pd_ = np.einsum("bi,ij,bj->b", w, der, w)
    gana = np.argmax(np.stack([pl, pr, pd_], 1), 1)
    return dict(P_original_mejor=float((gana == 0).mean()), P_equivalentes=float((gana == 1).mean()), P_rival_mejor=float((gana == 2).mean()))


# ====================================================================================== carga de predicciones
class Pred:
    def __init__(self, clave: str, Yhat: np.ndarray, Y: np.ndarray, origen: np.ndarray, semillas: list, resultados: list[dict], q: float):
        self.clave, self.Yhat, self.Y, self.origen, self.semillas, self.resultados, self.q = clave, Yhat, Y, origen, semillas, resultados, q
        self.L = np.abs(Yhat - Y[None]).mean(2) / q                    # (R, N) MASE por origen
        self.Lsq = np.sqrt(((Yhat - Y[None]) ** 2).mean(2)) / q         # RMSE por origen (métrica secundaria), misma escala

    @property
    def media_origen(self) -> np.ndarray:
        return self.L.mean(0)

    @property
    def por_semilla(self) -> np.ndarray:
        return self.L.mean(1)


def _leer(rutas: list[Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list, list]:
    Yh, sem, res, Y, o = [], [], [], None, None
    for r in rutas:
        with np.load(r, allow_pickle=False) as z:
            if o is None:
                o, Y = z["origen"].astype(str), z["y_kwh"].astype("float64")
            elif not np.array_equal(z["origen"].astype(str), o):
                raise ValueError(f"{r}: orígenes distintos a los del resto del procedimiento")
            Yh.append(z["yhat_kwh"].astype("float64"))
        jr = r.parent / "resultado.json"
        rr = json.loads(jr.read_text(encoding="utf-8")) if jr.exists() else {}
        res.append(rr)
        sem.append(rr.get("semilla"))
    return np.stack(Yh), Y, o, sem, res


def cargar_familia(salida: Path, q: float) -> dict[str, Pred]:
    out = {}
    for d in sorted((salida / "familia").glob("*")):
        if not d.is_dir():
            continue
        rutas = sorted(d.glob("semilla_*/pred_prueba.npz"), key=lambda p: int(p.parent.name.split("_")[1])) or sorted(d.glob("determinista/pred_prueba.npz"))
        if rutas:
            out[d.name] = Pred(d.name, *_leer(rutas), q)
    return out


def cargar_adicional(salida: Path, nombre: str, q: float) -> dict[str, Pred]:
    out = {}
    base = salida / "adicionales" / nombre
    for d in sorted(base.glob("*")):
        if not d.is_dir():
            continue
        rutas = sorted(d.glob("final/semilla_*/pred_prueba.npz"), key=lambda p: int(p.parent.name.split("_")[1]))
        if not rutas:
            rutas = sorted(d.glob("*/prueba_W*.npz"))
        if rutas:
            if rutas[0].name.startswith("prueba_W"):
                with np.load(rutas[0], allow_pickle=False) as z:
                    out[d.name] = Pred(d.name, z["yhat_kwh"][None].astype("float64"), z["y_kwh"].astype("float64"), z["origen"].astype(str), [None], [{}], q)
            else:
                out[d.name] = Pred(d.name, *_leer(rutas), q)
    return out


# ====================================================================================== contrastes
def contrastar(o: Pred, k: Pred, est: dict, metrica: str = "MASE", rezagos: int | None = None, bloque: int | None = None,
               semillas: list | None = None, n_ajuste: int | None = None) -> dict:
    """Original frente a un rival: las pruebas exigidas y las de sensibilidad, sin ajustar."""
    rez = rezagos or est["rezagos_hac"]
    blo = bloque or est["bloque_dias"]
    Lo, Lk = (o.L, k.L) if metrica == "MASE" else (o.Lsq, k.Lsq)
    if semillas is not None:
        io = [i for i, s in enumerate(o.semillas) if s in semillas]
        Lo = Lo[io]
        if k.semillas[0] is not None:
            Lk = Lk[[i for i, s in enumerate(k.semillas) if s in semillas]]
    lo, lk = Lo.mean(0), Lk.mean(0)
    d = lo - lk
    e_dm, p_dm2, p_dm1 = dm_hln(d, rez)
    bb = bootstrap(d, lk, blo, est["remuestreos"], est["semilla"])
    so, sk = Lo.mean(1), Lk.mean(1)
    if len(sk) == 1:                                           # rival determinista: t de una muestra contra su valor
        ds = so - sk[0]
    else:
        n = min(len(so), len(sk))
        ds = so[:n] - sk[:n]
    R = len(ds)
    sd = float(np.std(ds, ddof=1)) if R > 1 else float("nan")
    t_s = float(ds.mean() / (sd / math.sqrt(R))) if R > 1 and sd > 0 else float("nan")
    p_t = float(stats.t.cdf(t_s, R - 1)) if np.isfinite(t_s) else float("nan")
    p_sw = float(stats.shapiro(ds).pvalue) if R >= 3 and sd > 0 else float("nan")
    try:
        p_w = float(stats.wilcoxon(ds, alternative="less").pvalue) if R >= 6 and np.any(ds != 0) else float("nan")
    except ValueError:
        p_w = float("nan")
    if n_ajuste:
        sdnb = math.sqrt((1 / R + len(d) / n_ajuste) * sd ** 2) if R > 1 else float("nan")
        t_nb = float(ds.mean() / sdnb) if sdnb and sdnb > 0 else float("nan")
        p_nb = float(stats.t.cdf(t_nb, R - 1)) if np.isfinite(t_nb) else float("nan")
    else:
        t_nb = p_nb = float("nan")
    by = bayes_bloques(d, lk.mean(), est["rope_relativo"], blo)
    return dict(rival=k.clave, metrica=metrica, MASE_original=float(lo.mean()), MASE_rival=float(lk.mean()), diferencia=bb["diferencia"],
                ic95_diferencia=bb["ic95"], diferencia_relativa_pct=bb["diferencia_relativa_pct"], ic95_relativa_pct=bb["ic95_relativa_pct"],
                ic90_relativa_pct=bb["ic90_relativa_pct"], DM_estadistico=e_dm, p_DM=p_dm1, p_DM_bilateral=p_dm2, p_bootstrap=bb["p_unilateral"],
                t_semillas=t_s, p_t_semillas=p_t, shapiro_p=p_sw, p_wilcoxon=p_w, t_nadeau_bengio=t_nb, p_nadeau_bengio=p_nb, semillas_pareadas=R,
                sd_diferencias_semillas=sd, cliff_delta=cliff(lo, lk), cohen_dz=float(-ds.mean() / sd) if R > 1 and sd > 0 else float("nan"),
                P_original_mejor_por_origen=float((lo < lk).mean() + 0.5 * (lo == lk).mean()), **{f"bayes_{a}": b for a, b in by.items()})


def decidir(t: pd.DataFrame, pruebas: list[str], est: dict) -> pd.DataFrame:
    t = t.copy()
    for p in pruebas:
        t[p + "_Holm"] = _holm(t[p])
    t["p_max_Holm"] = t[[p + "_Holm" for p in pruebas]].max(axis=1)
    t["rechazan_todas"] = t.p_max_Holm < ALFA
    m = 100 * est["rope_relativo"]
    sup = t.ic95_relativa_pct.map(lambda x: x[1])
    inf = t.ic95_relativa_pct.map(lambda x: x[0])
    t["efecto_no_despreciable"] = (t.cliff_delta.abs() >= est["umbral_cliff"]) | (t.cohen_dz.abs() >= est["umbral_cohen"])
    t["ic_mejora_mayor_que_cero"] = sup < 0
    t["ic_mejora_mayor_que_delta"] = sup < -m
    eq = t.ic90_relativa_pct.map(lambda x: x[0] > -m and x[1] < m)
    t["decision"] = np.select([t.rechazan_todas & t.ic_mejora_mayor_que_delta & t.efecto_no_despreciable, t.rechazan_todas, eq],
                              ["significativa y relevante", "significativa sin relevancia práctica", "equivalente (TOST ±3 %)"], "no concluyente")
    t["original_mejor_en_media"] = t.MASE_original < t.MASE_rival
    return t


def veredicto(t: pd.DataFrame) -> dict:
    return dict(m=int(len(t)), mejor_en_media_que_todos=bool(t.original_mejor_en_media.all()), todas_las_pruebas_rechazan_en_todos=bool(t.rechazan_todas.all()),
                diferencia_supera_delta_en_todos=bool(t.ic_mejora_mayor_que_delta.all()), efecto_no_despreciable_en_todos=bool(t.efecto_no_despreciable.all()),
                ic_mejora_positiva_en_todos=bool(t.ic_mejora_mayor_que_cero.all()), contrastes_superados=int((t.decision == "significativa y relevante").sum()),
                rival_mas_dificil=str(t.sort_values("p_max_Holm", ascending=False).rival.iloc[0]) if len(t) else None,
                prueba_que_menos_rechaza=str(t[[c for c in t.columns if c.endswith("_Holm") and c != "p_max_Holm"]].max().idxmax()) if len(t) else None)


# ====================================================================================== análisis principal
def ejecutar(e3) -> dict:
    sal = e3.salida / "estadistica"
    sal.mkdir(parents=True, exist_ok=True)
    est = e3.plan["estadistica"]
    if e3.rapido:
        est = dict(est, remuestreos=300, mcs_remuestreos=200)
    datos = e3.datos
    P = cargar_familia(e3.salida, datos.q168)
    if "original" not in P:
        raise RuntimeError("No hay predicciones del original en familia/: ejecute la etapa familia")
    o = P["original"]
    rivales = [k for k in P if k != "original"]
    resumen = dict(rivales=rivales, R=len(o.semillas))
    n_aj = int(o.resultados[0].get("n_ajuste") or 0) or None

    # ---------- tabla maestra y resumen
    tm = _tabla_maestra(e3, P)
    _guardar(sal, "tabla_maestra.csv", tm)
    rs = _resumen_modelos(P, tm, est)
    _guardar(sal, "resumen_por_modelo.csv", rs)

    # ---------- contrastes confirmatorios
    filas = [contrastar(o, P[k], est, n_ajuste=n_aj) for k in rivales]
    t = decidir(pd.DataFrame(filas), ["p_DM", "p_bootstrap", "p_t_semillas"], est)
    t["p_wilcoxon_Holm"], t["p_nadeau_bengio_Holm"] = _holm(t.p_wilcoxon), _holm(t.p_nadeau_bengio)
    _guardar(sal, "contrastes_r3.csv", _formato_a4(t))
    _guardar(sal, "contrastes_r3_detalle.csv", t)
    v = veredicto(t)

    # ---------- globales: MCS y Friedman-Nemenyi
    L = pd.DataFrame({k: P[k].media_origen for k in P})
    ing = datos.ingenuos(datos.prueba()[: len(o.origen)])
    for nm, yh in ing.items():
        L[nm] = np.abs(yh - o.Y).mean(1) / datos.q168
    tmcs = mcs(L[[c for c in L.columns if not c.startswith("ingenuo")]], est["mcs_alfa"], est["mcs_remuestreos"], est["bloque_dias"], est["semilla"])
    _guardar(sal, "mcs.csv", tmcs)
    fr = friedman(L[list(P)].to_numpy(), list(P))
    _guardar(sal, "friedman_nemenyi.json", fr)
    _diagrama_cd(fr, sal)
    v["original_en_MCS"] = bool(tmcs.set_index("procedimiento").loc["original", "en_el_conjunto"])
    v["original_unico_en_MCS"] = bool(tmcs.en_el_conjunto.sum() == 1 and v["original_en_MCS"])

    # ---------- potencia final
    za = stats.norm.ppf(1 - ALFA / (2 * len(rivales)))
    pw = []
    for r in t.itertuples():
        sd, R = r.sd_diferencias_semillas, r.semillas_pareadas
        delta = 0.03 * r.MASE_rival
        pw.append(dict(rival=r.rival, R=R, sd_diferencias=sd, delta=delta, potencia=float(stats.norm.cdf(delta * math.sqrt(R) / sd - za)) if sd and sd > 0 else 1.0,
                       diferencia_minima_detectable=float((za + stats.norm.ppf(0.9)) * sd / math.sqrt(R)) if sd and sd > 0 else 0.0))
    pw = pd.DataFrame(pw)
    pot = dict(por_contraste=pw.to_dict("records"), potencia_conjunta=float(np.prod(pw.potencia)), alfa_ajustado=ALFA / len(rivales),
               minima_potencia_por_contraste=float(pw.potencia.min()), cumple_090_en_todos=bool((pw.potencia >= 0.90).all()))
    _guardar(sal, "potencia_final.json", pot)

    # ---------- sensibilidad de la conclusión
    sens = [dict(escenario="principal (DM + bootstrap + t por semillas)", r3=_r3(v))]
    alternativas = {"Wilcoxon en lugar de t": ["p_DM", "p_bootstrap", "p_wilcoxon"], "t de Nadeau-Bengio en lugar de t": ["p_DM", "p_bootstrap", "p_nadeau_bengio"]}
    for nombre, pr in alternativas.items():
        sens.append(dict(escenario=nombre, r3=_r3(veredicto(decidir(t.drop(columns=[c for c in t.columns if c.endswith("_Holm")]), pr, est)))))
    for rz in est["sensibilidad_rezagos_hac"]:
        tt = decidir(pd.DataFrame([contrastar(o, P[k], est, rezagos=rz, n_ajuste=n_aj) for k in rivales]), ["p_DM", "p_bootstrap", "p_t_semillas"], est)
        sens.append(dict(escenario=f"DM con {rz} rezagos HAC", r3=_r3(veredicto(tt))))
    for bl in est["sensibilidad_bloque_dias"]:
        tt = decidir(pd.DataFrame([contrastar(o, P[k], est, bloque=bl, n_ajuste=n_aj) for k in rivales]), ["p_DM", "p_bootstrap", "p_t_semillas"], est)
        sens.append(dict(escenario=f"bootstrap con bloques de {bl} días", r3=_r3(veredicto(tt))))
    tt = decidir(pd.DataFrame([contrastar(o, P[k], est, metrica="RMSE", n_ajuste=n_aj) for k in rivales]), ["p_DM", "p_bootstrap", "p_t_semillas"], est)
    _guardar(sal, "contrastes_rmse.csv", tt)
    sens.append(dict(escenario="métrica secundaria: RMSE por origen", r3=_r3(veredicto(tt))))
    sem = [s for s in o.semillas]
    subconjuntos = {"primeras 10 semillas": sem[:10], "semillas pares": sem[0::2], "semillas impares": sem[1::2]}
    for nombre, ss in subconjuntos.items():
        if len(ss) >= 3:
            tt = decidir(pd.DataFrame([contrastar(o, P[k], est, semillas=ss, n_ajuste=n_aj) for k in rivales]), ["p_DM", "p_bootstrap", "p_t_semillas"], est)
            sens.append(dict(escenario=nombre, r3=_r3(veredicto(tt))))
    ts = pd.DataFrame(sens)
    v["conclusion_robusta_a_la_prueba"] = bool(ts.r3.nunique() == 1)
    _guardar(sal, "sensibilidad_conclusion.csv", ts)

    # ---------- efectos mixtos
    _guardar(sal, "efectos_mixtos.json", _mixto(P, est))

    # ---------- adecuación del original (Tablas A y B)
    ad = _adecuacion(e3, o, P, t, est)
    _guardar(sal, "adecuacion_original.json", ad)
    v["adecuacion_original"] = ad.get("cumple")

    # ---------- conjuntos adicionales
    adic = _adicionales(e3, est, sal)
    v.update(adic.get("veredicto", {}))

    # ---------- ablación, incertidumbre, robustez, pareto, errores, sensibilidad, curva, escalabilidad
    for nombre, fn in (("ablacion", _ablacion), ("incertidumbre", _incertidumbre), ("robustez", _robustez), ("pareto", _pareto), ("errores", _errores),
                       ("sensibilidad", _sensibilidad), ("curva", _curva), ("escalabilidad", _escalabilidad)):
        try:
            resumen[nombre] = fn(e3, P, est, sal)
        except FileNotFoundError as e:
            resumen[nombre] = f"sin datos ({e})"
        except Exception as e:                       # un análisis secundario que falla no impide el resto; queda registrado
            resumen[nombre] = f"error: {type(e).__name__}: {str(e)[:200]}"
            e3.log(f"Análisis {nombre}: {resumen[nombre]}")
    v["R3_cumplida"] = bool(v["mejor_en_media_que_todos"] and v["todas_las_pruebas_rechazan_en_todos"] and v["diferencia_supera_delta_en_todos"]
                            and v["efecto_no_despreciable_en_todos"] and v.get("consistencia_adicionales", False) and bool(v.get("adecuacion_original"))
                            and v["conclusion_robusta_a_la_prueba"])
    v["tope_por_R3"] = 5.0 if v["R3_cumplida"] else 4.0
    v["nota"] = "El tope depende también de R1 y R2, que no se verifican con datos. Un resultado no concluyente con potencia declarada se reporta como tal."
    _guardar(sal, "veredicto_r3.json", v)
    resumen["veredicto"] = v
    _resumen_md(e3, sal, t, v, pot, tmcs, ts)
    return resumen


def _r3(v: dict) -> str:
    ok = v["mejor_en_media_que_todos"] and v["todas_las_pruebas_rechazan_en_todos"] and v["diferencia_supera_delta_en_todos"] and v["efecto_no_despreciable_en_todos"]
    return "superior a toda la familia" if ok else f"no superior ({v['contrastes_superados']}/{v['m']} contrastes superados)"


def _formato_a4(t: pd.DataFrame) -> pd.DataFrame:
    filas = []
    for r in t.itertuples():
        for nombre, p, ph, h0, sup in (("Diebold-Mariano (HLN, HAC 13)", r.p_DM, r.p_DM_Holm, "E[L_orig − L_rival] ≥ 0", "estacionariedad de d_t; HAC por autocorrelación"),
                                       ("Bootstrap circular de bloques de 14 días", r.p_bootstrap, r.p_bootstrap_Holm, "E[d] ≥ 0", "dependencia de corto plazo dentro de bloques"),
                                       ("t pareada por semillas", r.p_t_semillas, r.p_t_semillas_Holm, "E[MASE_orig − MASE_rival] ≥ 0 entre semillas",
                                        f"normalidad de las diferencias (Shapiro p = {r.shapiro_p:.3f})" if np.isfinite(r.shapiro_p) else "normalidad no evaluable")):
            filas.append(dict(contraste=f"Original frente a {r.rival}", prueba=nombre, hipotesis_nula=h0, supuestos=sup, p_crudo=p, p_Holm=ph,
                              diferencia_MASE=r.diferencia, ic95_diferencia=r.ic95_diferencia, diferencia_relativa_pct=r.diferencia_relativa_pct,
                              ic95_relativa_pct=r.ic95_relativa_pct, delta_cliff=r.cliff_delta, d_cohen=r.cohen_dz, decision=r.decision))
    return pd.DataFrame(filas)


def _tabla_maestra(e3, P: dict) -> pd.DataFrame:
    filas = []

    def agregar(conjunto, clave, res):
        c = res.get("costes") or {}
        m = res.get("metricas") or {}
        filas.append(dict(conjunto=conjunto, clave=clave, semilla=res.get("semilla"), MASE=m.get("MASE"), MAE=m.get("MAE"), RMSE=m.get("RMSE"), MAPE=m.get("MAPE"),
                          MASE_ajuste=res.get("MASE_ajuste"), MASE_parada=res.get("MASE_parada"), epocas=res.get("epocas"), n_parametros=res.get("n_parametros"),
                          flops_por_origen=c.get("flops_por_origen"), segundos_entrenamiento=c.get("segundos_entrenamiento", res.get("segundos")),
                          memoria_pico_proceso_mb=c.get("memoria_pico_proceso_mb"), memoria_pico_gpu_mb=c.get("memoria_pico_gpu_mb"), energia_gpu_wh=c.get("energia_gpu_wh"),
                          latencia_ms_por_origen_lote=c.get("latencia_ms_por_origen_lote"), dispositivo=res.get("dispositivo")))
    for k, p in P.items():
        for res in p.resultados:
            agregar("principal", k, res)
    for nombre in e3.adic:
        for r in sorted((e3.salida / "adicionales" / nombre).glob("*/final/semilla_*/resultado.json")):
            agregar(nombre, r.parts[-4], json.loads(r.read_text(encoding="utf-8")))
    return pd.DataFrame(filas)


def _resumen_modelos(P: dict, tm: pd.DataFrame, est: dict) -> pd.DataFrame:
    g = np.random.default_rng(est["semilla"])
    filas = []
    for k, p in P.items():
        s = p.por_semilla
        bs = [g.choice(s, len(s)).mean() for _ in range(2000)] if len(s) > 1 else [s[0]]
        ix = indices_bloques(p.L.shape[1], est["bloque_dias"], 2000, est["semilla"])
        bb = p.media_origen[ix].mean(1)
        c = tm[(tm.conjunto == "principal") & (tm.clave == k)]
        filas.append(dict(clave=k, semillas=len(s), MASE_media=float(s.mean()), MASE_desv=float(s.std(ddof=1)) if len(s) > 1 else 0.0,
                          ic95_semillas=[float(x) for x in np.percentile(bs, [2.5, 97.5])], ic95_bloques=[float(x) for x in np.percentile(bb, [2.5, 97.5])],
                          cv_pct=float(100 * s.std(ddof=1) / s.mean()) if len(s) > 1 else 0.0, n_parametros=c.n_parametros.median(), flops_por_origen=c.flops_por_origen.median(),
                          segundos_entrenamiento=c.segundos_entrenamiento.median(), latencia_ms=c.latencia_ms_por_origen_lote.median(),
                          memoria_pico_mb=c.memoria_pico_gpu_mb.median() if c.memoria_pico_gpu_mb.notna().any() else c.memoria_pico_proceso_mb.median(),
                          energia_gpu_wh=c.energia_gpu_wh.median(),
                          kruskal_wallis_semillas_p=float(stats.kruskal(*p.L).pvalue) if p.L.shape[0] > 1 and np.ptp(p.L) > 0 else None))
    t = pd.DataFrame(filas).sort_values("MASE_media").reset_index(drop=True)
    t["puesto"] = np.arange(1, len(t) + 1)
    return t


def _mixto(P: dict, est: dict) -> dict:
    try:
        import statsmodels.formula.api as smf
        filas = []
        for k, p in P.items():
            nb = int(np.ceil(p.L.shape[1] / est["bloque_dias"]))
            for i, s in enumerate(p.semillas):
                for b in range(nb):
                    filas.append(dict(modelo=k, semilla=str(s), bloque=b, perdida=float(p.L[i, b * est["bloque_dias"]:(b + 1) * est["bloque_dias"]].mean())))
        d = pd.DataFrame(filas)
        m = smf.mixedlm("perdida ~ C(modelo, Treatment('original'))", d, groups=d["bloque"], vc_formula={"semilla": "0 + C(semilla)"}).fit(reml=True, method="lbfgs")
        coef = pd.DataFrame(dict(coef=m.params, ee=m.bse, p=m.pvalues)).reset_index().rename(columns={"index": "termino"})
        coef = coef[coef.termino.str.startswith("C(modelo")]
        coef["p_Holm"] = _holm(coef.p)
        return dict(modelo="y_ijr = µ + α_j + u_bloque + v_semilla + ε (pérdida MASE media por bloque de 14 días)", n=int(len(d)),
                    coeficientes_frente_al_original=coef.to_dict("records"), var_bloque=float(m.cov_re.iloc[0, 0]) if m.cov_re.size else None,
                    var_semilla=[float(x) for x in np.atleast_1d(m.vcomp)] if np.size(m.vcomp) else None,
                    var_residual=float(m.scale), interpretacion="coeficiente > 0: el rival tiene mayor pérdida que el original")
    except Exception as e:
        return dict(error=f"{type(e).__name__}: {str(e)[:300]}")


def _tipo_dia(datos, pos: np.ndarray) -> np.ndarray:
    cal = pd.DataFrame(datos.K[pos], columns=datos.cal)
    esp = np.zeros(len(pos), bool)
    for c in ("cal_es_festivo", "cal_es_fin_semana_largo", "cal_es_semana_santa", "cal_es_temporada_navidena", "festivo"):
        if c in cal:
            esp |= cal[c].to_numpy() > 0.5
    fds = np.zeros(len(pos), bool)
    for c in ("cal_es_sabado", "cal_es_domingo", "dia_semana_5", "dia_semana_6"):
        if c in cal:
            fds |= cal[c].to_numpy() > 0.5
    return np.where(esp, "especial", np.where(fds, "fin_de_semana", "laboral"))


def _adecuacion(e3, o: Pred, P: dict, t: pd.DataFrame, est: dict) -> dict:
    from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
    from statsmodels.tsa.stattools import bds
    import statsmodels.api as sm
    datos = e3.datos
    yhat = o.Yhat.mean(0)
    r = (o.Y - yhat).mean(1) / 1e6
    lb = acorr_ljungbox(r, lags=[7, 14], return_df=True)
    arch = het_arch(r, nlags=7)
    try:
        b = bds(r, max_dim=3)
        bds_p = [float(x) for x in np.atleast_1d(b[1])]
    except Exception:
        bds_p = None
    jb = stats.jarque_bera(r)
    mz = sm.OLS(o.Y.mean(1) / 1e6, sm.add_constant(yhat.mean(1) / 1e6)).fit(cov_type="HAC", cov_kwds=dict(maxlags=est["rezagos_hac"]))
    brecha = [(x["MASE_parada"] - x["MASE_ajuste"]) / x["MASE_ajuste"] for x in o.resultados if x.get("MASE_ajuste")]
    val = None
    cj = e3.p1 / "ajuste" / "busqueda" / "configuracion_elegida.json"
    if cj.exists():
        val = list(json.loads(cj.read_text(encoding="utf-8")).values())[0].get("MASE_medio_validacion")
    pos = datos.prueba()[: len(o.origen)]
    ing = datos.ingenuos(pos)
    triv = {}
    for nm, yh in ing.items():
        lk = np.abs(yh - o.Y).mean(1) / datos.q168
        triv[nm] = dict(MASE=float(lk.mean()), p_DM=dm_hln(o.media_origen - lk, est["rezagos_hac"])[2])
    san = e3.salida / "sanidad" / "sanidad.json"
    gr = None
    if len(t):
        mejor = str(t.sort_values("MASE_rival").rival.iloc[0])
        gr = _giacomini_rossi(o.media_origen - P[mejor].media_origen, est["rezagos_hac"])
        gr["rival"] = mejor
    d = dict(residuos=dict(ljung_box_p=dict(zip(["7", "14"], map(float, lb.lb_pvalue))), arch_lm_p=float(arch[1]), bds_p=bds_p, jarque_bera_p=float(jb.pvalue),
                           nota="residuo medio diario en GWh del pronóstico promedio de semillas"),
             mincer_zarnowitz=dict(intercepto=float(mz.params[0]), pendiente=float(mz.params[1]), p_conjunta=float(mz.f_test("const = 0, x1 = 1").pvalue)),
             brecha_ajuste_parada=dict(media=float(np.mean(brecha)) if brecha else None, margen=0.10, cumple=bool(brecha and np.mean(brecha) < 0.10)),
             validacion_frente_a_prueba=dict(MASE_validacion=val, MASE_prueba=float(o.por_semilla.mean()), razon=(float(o.por_semilla.mean()) / val) if val else None),
             supera_ingenuos=triv, sanidad=json.loads(san.read_text(encoding="utf-8")) if san.exists() else None, giacomini_rossi=gr)
    d["cumple"] = bool(all(v["MASE"] > o.por_semilla.mean() and v["p_DM"] < ALFA for v in triv.values()) and d["brecha_ajuste_parada"]["cumple"]
                       and (d["sanidad"] is None or d["sanidad"].get("supera_criterio")))
    return d


def _giacomini_rossi(d: np.ndarray, rezagos: int, mu: float = 0.3, sims: int = 5000, semilla: int = 11) -> dict:
    T = len(d)
    m = int(mu * T)
    u = d - d.mean()
    lrv = u @ u / T + 2 * sum((1 - k / (rezagos + 1)) * (u[k:] @ u[:-k]) / T for k in range(1, rezagos + 1))
    c = np.cumsum(np.r_[0.0, d])
    med = (c[m:] - c[:-m]) / m
    est = np.abs((med - d.mean()) / np.sqrt(lrv / m))
    g = np.random.default_rng(semilla)
    pasos = 400
    W = np.concatenate([np.zeros((sims, 1)), np.cumsum(g.standard_normal((sims, pasos)) / math.sqrt(pasos), axis=1)], axis=1)
    k = int(mu * pasos)
    puente = W - np.linspace(0, 1, pasos + 1)[None, :] * W[:, -1:]
    sup = (np.abs(puente[:, k:] - puente[:, :-k]) / math.sqrt(mu)).max(1)
    return dict(estadistico_centrado_max=float(est.max()), valor_critico_5pct=float(np.quantile(sup, 0.95)), p_simulado=float((sup >= est.max()).mean()), ventana=m)


def _adicionales(e3, est: dict, sal: Path) -> dict:
    tab, contr, matriz = [], [], {}
    for nombre, datos in e3.adic.items():
        P = cargar_adicional(e3.salida, nombre, datos.q168)
        if "original" not in P:
            continue
        o = P["original"]
        matriz[nombre] = {k: float(p.por_semilla.mean()) for k, p in P.items()}
        for k, p in P.items():
            s = p.por_semilla
            tab.append(dict(conjunto=nombre, clave=k, semillas=len(s), MASE_media=float(s.mean()), MASE_desv=float(s.std(ddof=1)) if len(s) > 1 else 0.0))
        for k in [k for k in P if k != "original"]:
            c = contrastar(o, P[k], est)
            c["conjunto"] = nombre
            contr.append(c)
    if not contr:
        return {}
    t = pd.DataFrame(contr)
    t = decidir(t, ["p_DM", "p_bootstrap", "p_t_semillas"], est)
    t["peor_significativo"] = (t.p_DM_bilateral < ALFA) & (t.diferencia > 0)
    _guardar(sal, "adicionales_por_conjunto.csv", pd.DataFrame(tab))
    _guardar(sal, "adicionales_contrastes.csv", t)
    cons = {}
    for nombre, g in t.groupby("conjunto"):
        cons[nombre] = dict(mejor_en_media_que_todos=bool(g.original_mejor_en_media.all()), algun_rival_significativamente_mejor=bool(g.peor_significativo.any()),
                            superados=int((g.decision == "significativa y relevante").sum()), m=int(len(g)))
    n_mejor = sum(v["mejor_en_media_que_todos"] for v in cons.values())
    ver = dict(consistencia_adicionales=bool(n_mejor >= 2 and not any(v["algun_rival_significativamente_mejor"] for v in cons.values())),
               adicionales_mejor_en=int(n_mejor), adicionales_evaluados=len(cons))
    # prueba entre conjuntos (principal + adicionales)
    try:
        Pp = cargar_familia(e3.salida, e3.datos.q168)
        matriz["principal"] = {k: float(p.por_semilla.mean()) for k, p in Pp.items()}
    except Exception:
        pass
    M = pd.DataFrame(matriz).T.dropna(axis=1)
    entre = {}
    if len(M) >= 3 and "original" in M:
        entre["friedman_iman_davenport"] = friedman(M.to_numpy(), list(M.columns))
        filas = []
        for k in [c for c in M.columns if c != "original"]:
            z = ((M["original"] - M[k]) / M[k]).to_numpy()
            try:
                pw = float(stats.wilcoxon(M["original"], M[k], alternative="less").pvalue)
            except ValueError:
                pw = float("nan")
            filas.append(dict(rival=k, p_wilcoxon=pw, diferencias_relativas=list(map(float, z)), **bayes_rangos_signo(z, est["rope_relativo"])))
        w = pd.DataFrame(filas)
        w["p_wilcoxon_Holm"] = _holm(w.p_wilcoxon)
        entre["por_rival"] = w.to_dict("records")
        entre["nota"] = f"{len(M)} conjuntos: el p exacto mínimo de Wilcoxon es {2 ** -len(M):.4f} (unilateral); la conclusión se apoya en la consistencia por conjunto"
    _guardar(sal, "adicionales_entre_conjuntos.json", dict(matriz_MASE=M.to_dict(), **entre, consistencia=cons))
    return dict(veredicto=ver)


def _ablacion(e3, P, est, sal) -> dict:
    base = e3.salida / "ablacion"
    var = {}
    for d in sorted(base.glob("*")):
        rutas = sorted(d.glob("semilla_*/pred_prueba.npz"), key=lambda p: int(p.parent.name.split("_")[1]))
        if rutas:
            var[d.name] = Pred(d.name, *_leer(rutas), e3.datos.q168)
    if "completo" not in var:
        raise FileNotFoundError("ablacion/completo")
    c = var["completo"]
    t = decidir(pd.DataFrame([contrastar(var[k], c, est) for k in var if k != "completo"]), ["p_DM", "p_bootstrap", "p_t_semillas"], est)
    t = t.rename(columns={"rival": "variante"})
    t["variante"] = [k for k in var if k != "completo"]
    t["nota"] = "diferencia = variante − completo; positiva: retirar o cambiar el componente empeora"
    _guardar(sal, "ablacion_contrastes.csv", t)
    vj = json.loads((base / "variantes.json").read_text(encoding="utf-8"))
    filas = []
    for k, p in var.items():
        bits = vj.get(k, {}).get("bits", {})
        if bits:
            for i, s in enumerate(p.semillas):
                filas.append(dict(variante=k, semilla=s, MASE=float(p.L[i].mean()), **bits))
    f = pd.DataFrame(filas)
    efectos = {}
    if len(f) and f.variante.nunique() == 8:
        import statsmodels.formula.api as smf
        comps = [c for c in f.columns if c not in ("variante", "semilla", "MASE")]
        mod = smf.ols("MASE ~ " + " * ".join(f"Q('{c}')" for c in comps), f).fit(cov_type="HC3")
        ic = mod.conf_int()
        efectos = [dict(termino=k, efecto=float(mod.params[k]), ic95=[float(ic.loc[k, 0]), float(ic.loc[k, 1])], p=float(mod.pvalues[k])) for k in mod.params.index if k != "Intercept"]
        _guardar(sal, "ablacion_factorial.json", dict(modelo="MASE por semilla ~ efectos principales e interacciones de los tres componentes (HC3)", efectos=efectos))
    return dict(variantes=list(var), contrastes=int(len(t)))


def _incertidumbre(e3, P, est, sal) -> dict:
    """Intervalos del 90 %: ensamble de semillas, ACI y ACI condicionado por texto (idea D) en el original y los tres mejores rivales."""
    datos = e3.datos
    nivel, alfa = 0.90, 0.10
    orden = sorted([k for k in P if k != "original"], key=lambda k: P[k].por_semilla.mean())[:3]
    pos = datos.prueba()[: len(P["original"].origen)]
    tipo = _tipo_dia(datos, pos)
    anio = pd.DatetimeIndex(pd.to_datetime(P["original"].origen)).year.to_numpy()
    ndocs = None
    if e3.texto is not None:
        dia = datos.dia_de(pos)
        ndocs = (e3.texto["lista"][dia] >= 0).sum(1)
    filas = []
    for k in ["original"] + orden:
        p = P[k]
        y, yh = p.Y, p.Yhat.mean(0)
        if p.Yhat.shape[0] >= 5:
            lo, hi = np.quantile(p.Yhat, alfa / 2, axis=0), np.quantile(p.Yhat, 1 - alfa / 2, axis=0)
            filas += _evaluar_intervalos(k, "ensamble_semillas", y, lo, hi, nivel, tipo, anio, ndocs)
            crps = float((np.abs(p.Yhat - y[None]).mean(0) - 0.5 * np.abs(p.Yhat[:, None] - p.Yhat[None, :]).mean((0, 1))).mean())
            pit = (p.Yhat < y[None]).mean(0).ravel()
            filas.append(dict(clave=k, metodo="ensamble_semillas", subgrupo="calibracion", CRPS_kwh=crps, PIT_KS_p=float(stats.kstest(pit, "uniform").pvalue)))
        cal = _calibracion(e3, k, datos)
        if cal is None:
            continue
        rc, ndc = cal
        res = np.abs(np.concatenate([rc, y - yh]))
        rad = _aci(res, alfa, est)[len(rc):]
        filas += _evaluar_intervalos(k, "ACI", y, yh - rad, yh + rad, nivel, tipo, anio, ndocs)
        if ndocs is not None and ndc is not None:
            esc_c, esc_t = _escala_texto(np.abs(rc), ndc, ndocs)
            sc = res / np.concatenate([esc_c, esc_t])
            rad = _aci(sc, alfa, est)[len(rc):] * esc_t
            filas += _evaluar_intervalos(k, "ACI_condicionado_por_texto (idea D)", y, yh - rad, yh + rad, nivel, tipo, anio, ndocs)
    t = pd.DataFrame(filas)
    _guardar(sal, "incertidumbre.csv", t)
    return dict(modelos=["original"] + orden)


def _calibracion(e3, clave, datos):
    """Residuos de calibración: predicciones del pliegue 2021 (etapa potencia, mismas configuraciones) promediadas entre semillas."""
    rutas = sorted(((e3.p1 if clave == "original" else e3.salida) / "potencia" / clave).glob("*/pred_val_*.npz"))
    if not rutas:
        return None
    Yh, Y, o, _, _ = _leer(rutas)
    pos = np.searchsorted(datos.indice, pd.to_datetime(o).tz_localize(datos.indice.tz) if datos.indice.tz is not None else pd.to_datetime(o))
    ndc = (e3.texto["lista"][datos.dia_de(pos)] >= 0).sum(1) if e3.texto is not None else None
    return Y - Yh.mean(0), ndc


def _escala_texto(rc: np.ndarray, ndc: np.ndarray, ndt: np.ndarray):
    """log|r| medio por origen = β0 + β1 1[hay texto] + β2 log(1 + documentos), ajustado solo con la calibración; escala = exp(·)."""
    Xc = np.c_[np.ones(len(ndc)), ndc > 0, np.log1p(ndc)]
    yc = np.log(rc.mean(1) + 1e-9)
    beta = np.linalg.lstsq(Xc, yc, rcond=None)[0]
    Xt = np.c_[np.ones(len(ndt)), ndt > 0, np.log1p(ndt)]
    ec, et = np.exp(Xc @ beta), np.exp(Xt @ beta)
    return np.repeat((ec / ec.mean())[:, None], rc.shape[1], 1), np.repeat((et / ec.mean())[:, None], rc.shape[1], 1)


def _aci(res_abs: np.ndarray, alfa: float, est: dict, gamma: float = 0.005, retraso: int = 2, ventana: int = 365) -> np.ndarray:
    n, h = res_abs.shape
    radio = np.full((n, h), np.nan)
    a_t = np.full(h, alfa)
    for t in range(n):
        hist = res_abs[max(0, t - retraso - ventana + 1):max(0, t - retraso + 1)]
        if len(hist) >= 60:
            for j in range(h):
                radio[t, j] = np.quantile(hist[:, j], min(1.0, max(0.0, 1 - a_t[j])))
        s = t - retraso
        if s >= 0 and not np.isnan(radio[s]).any():
            a_t = a_t + gamma * (alfa - (res_abs[s] > radio[s]).astype(float))
    return radio


def _evaluar_intervalos(clave, metodo, y, lo, hi, nivel, tipo, anio, ndocs) -> list[dict]:
    a = 1 - nivel
    ok = ~np.isnan(lo).any(1)
    out = []
    grupos = {"total": ok}
    for t in np.unique(tipo):
        grupos[f"dia_{t}"] = ok & (tipo == t)
    for an in np.unique(anio):
        grupos[f"anio_{an}"] = ok & (anio == an)
    if ndocs is not None:
        grupos["con_texto"], grupos["sin_texto"] = ok & (ndocs > 0), ok & (ndocs == 0)
    for g, m in grupos.items():
        if m.sum() < 5:
            continue
        yy, l, h = y[m], lo[m], hi[m]
        dentro = (yy >= l) & (yy <= h)
        w = h - l
        wink = w + (2 / a) * (l - yy) * (yy < l) + (2 / a) * (yy - h) * (yy > h)
        fallo = (~dentro).mean(1) > 0.5                     # origen con mayoría de horas fuera del intervalo (independencia diaria)
        f_h = (~dentro).ravel()
        out.append(dict(clave=clave, metodo=metodo, subgrupo=g, n_origenes=int(m.sum()), nominal=nivel, cobertura=float(dentro.mean()),
                        anchura_media_kwh=float(w.mean()), winkler_kwh=float(wink.mean()), kupiec_p=_kupiec(f_h, a) if g == "total" else None,
                        christoffersen_p=_christoffersen(fallo) if g == "total" else None))
    return out


def _kupiec(f: np.ndarray, a: float) -> float:
    n, x = len(f), int(f.sum())
    if x in (0, n):
        return float(stats.chi2.sf(-2 * n * np.log(1 - a) if x == 0 else -2 * n * np.log(a), 1))
    pi = x / n
    return float(stats.chi2.sf(-2 * ((n - x) * np.log(1 - a) + x * np.log(a) - (n - x) * np.log(1 - pi) - x * np.log(pi)), 1))


def _christoffersen(f: np.ndarray) -> float:
    f = f.astype(int)
    n00, n01 = int(((f[:-1] == 0) & (f[1:] == 0)).sum()), int(((f[:-1] == 0) & (f[1:] == 1)).sum())
    n10, n11 = int(((f[:-1] == 1) & (f[1:] == 0)).sum()), int(((f[:-1] == 1) & (f[1:] == 1)).sum())
    if min(n00 + n01, n10 + n11) == 0:
        return float("nan")
    p01, p11, p = n01 / (n00 + n01), n11 / (n10 + n11), (n01 + n11) / (n00 + n01 + n10 + n11)
    ll = lambda a, b, q: (a * np.log(1 - q) if a else 0) + (b * np.log(q) if b else 0)   # noqa: E731
    lr = -2 * (ll(n00 + n10, n01 + n11, p) - ll(n00, n01, p01) - ll(n10, n11, p11)) if 0 < p < 1 else 0.0
    return float(stats.chi2.sf(lr, 1))


def _robustez(e3, P, est, sal) -> dict:
    t = pd.read_csv(e3.salida / "robustez" / "robustez_por_semilla.csv")
    m = t.groupby(["clave", "tipo", "nivel"]).MASE.mean().reset_index()
    filas = []
    for k, g in m.groupby("clave"):
        b = g[g.tipo == "original"].MASE
        if not len(b):
            continue
        b = float(b.iloc[0])
        fila = dict(clave=k, MASE_sin_perturbar=b)
        for tp in [x for x in g.tipo.unique() if x != "original"]:
            c = g[g.tipo == tp].sort_values("nivel")
            curva = np.r_[1.0, c.MASE.to_numpy() / b]
            fila[f"indice_{tp}"] = float(np.trapezoid(curva, dx=1.0) / max(1, len(curva) - 1)) if hasattr(np, "trapezoid") else float(np.trapz(curva, dx=1.0) / max(1, len(curva) - 1))
            fila[f"MASE_max_{tp}"] = float(c.MASE.max())
        filas.append(fila)
    r = pd.DataFrame(filas)
    if "original" in set(r.clave):
        for c in [c for c in r.columns if c.startswith("indice_")]:
            r[f"original_degrada_menos_{c[7:]}"] = r[c] >= float(r.loc[r.clave == "original", c].iloc[0])
    _guardar(sal, "robustez_resumen.csv", r)
    return dict(modelos=int(len(r)))


def _pareto(e3, P, est, sal) -> dict:
    rs = pd.read_csv(sal / "resumen_por_modelo.csv")
    ex = e3.salida / "artifacts" / "resumen_exportacion.csv"
    if ex.exists():
        rs = rs.merge(pd.read_csv(ex)[["clave", "latencia_onnx_mediana_ms", "latencia_torch_mediana_ms", "tamano_mb"]], on="clave", how="left")
    out = {}
    for c in ("n_parametros", "flops_por_origen", "latencia_ms", "segundos_entrenamiento", "energia_gpu_wh", "latencia_torch_mediana_ms"):
        if c not in rs or rs[c].isna().all():
            continue
        d = rs.dropna(subset=[c]).sort_values([c, "MASE_media"])
        frente, mejor = [], np.inf
        for r in d.itertuples():
            if r.MASE_media < mejor:
                frente.append(r.clave)
                mejor = r.MASE_media
        rs[f"pareto_{c}"] = rs.clave.isin(frente)
        out[c] = frente
    _guardar(sal, "pareto.csv", rs)
    return out


def _errores(e3, P, est, sal) -> dict:
    datos = e3.datos
    o = P["original"]
    pos = datos.prueba()[: len(o.origen)]
    tipo = _tipo_dia(datos, pos)
    anio = pd.DatetimeIndex(pd.to_datetime(o.origen)).year.to_numpy()
    filas, comp = [], []
    for k, p in P.items():
        lo = p.media_origen
        for g in np.unique(tipo):
            filas.append(dict(clave=k, eje="tipo_de_dia", valor=g, MASE=float(lo[tipo == g].mean()), n=int((tipo == g).sum())))
        for a in np.unique(anio):
            filas.append(dict(clave=k, eje="anio", valor=int(a), MASE=float(lo[anio == a].mean()), n=int((anio == a).sum())))
        mh = np.abs(p.Yhat.mean(0) - p.Y).mean(0) / datos.q168
        filas += [dict(clave=k, eje="hora_del_horizonte", valor=h + 1, MASE=float(v), n=len(lo)) for h, v in enumerate(mh)]
        if k != "original":
            d = o.media_origen - lo
            comp.append(dict(rival=k, fraccion_origenes_original_peor=float((d > 0).mean()), correlacion_perdidas=float(np.corrcoef(o.media_origen, lo)[0, 1]),
                             **{f"diferencia_rel_{g}": float(100 * d[tipo == g].mean() / lo[tipo == g].mean()) for g in np.unique(tipo)}))
    _guardar(sal, "errores_por_subgrupo.csv", pd.DataFrame(filas))
    _guardar(sal, "errores_compartidos.csv", pd.DataFrame(comp))
    peores = np.argsort(o.media_origen)[-max(5, len(o.media_origen) // 20):]
    _guardar(sal, "errores_peores_dias.csv", pd.DataFrame(dict(origen=o.origen[peores], MASE=o.media_origen[peores], tipo=tipo[peores])).sort_values("MASE", ascending=False))
    return dict(subgrupos=list(np.unique(tipo)))


def _sensibilidad(e3, P, est, sal) -> dict:
    t = pd.read_csv(e3.p1 / "sensibilidad" / "sensibilidad_por_semilla.csv")
    r = t.groupby(["punto", "parametro", "valor"], dropna=False).MASE.agg(["mean", "std", "size"]).reset_index()
    pil = e3.salida / "potencia" / "analisis_potencia.csv"
    if pil.exists():
        a = pd.read_csv(pil)
        mejor = float(a.MASE_rival_piloto.min())
        r["MASE_mejor_rival_pliegue_2021"] = mejor
        r["conserva_ventaja_frente_al_mejor_rival"] = r["mean"] < mejor
        r["rivales_superados"] = [int((a.MASE_rival_piloto > x).sum()) for x in r["mean"]]
    _guardar(sal, "sensibilidad_resumen.csv", r)
    return dict(puntos=int(len(r)))


def _curva(e3, P, est, sal) -> dict:
    t = pd.read_csv(e3.salida / "curva" / "curva_aprendizaje.csv")
    r = t.groupby(["clave", "fraccion"]).agg(n_ajuste=("n_ajuste", "first"), MASE=("MASE", "mean"), MASE_desv=("MASE", "std"), MASE_ajuste=("MASE_ajuste", "mean"),
                                             MASE_parada=("MASE_parada", "mean")).reset_index()
    _guardar(sal, "curva_resumen.csv", r)
    return dict(filas=int(len(r)))


def _escalabilidad(e3, P, est, sal) -> dict:
    t = pd.read_csv(e3.salida / "escalabilidad" / "escalabilidad.csv")
    filas = []
    for (k, b), g in t.dropna(subset=["segundos_inferencia_lote"]).groupby(["clave", "lote"]):
        if len(g) >= 2:
            pend = np.polyfit(np.log(g.L), np.log(g.segundos_inferencia_lote), 1)[0]
            pend_f = np.polyfit(np.log(g.L), np.log(g.flops_por_origen.astype(float)), 1)[0] if g.flops_por_origen.notna().all() else None
            filas.append(dict(clave=k, lote=b, exponente_tiempo_frente_a_L=float(pend), exponente_flops_frente_a_L=pend_f))
    _guardar(sal, "escalabilidad_ajuste.csv", pd.DataFrame(filas))
    return dict(filas=len(filas))


def _diagrama_cd(fr: dict, sal: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    r = pd.Series(fr["rango_medio"]).sort_values()
    fig, ax = plt.subplots(figsize=(10, 0.32 * len(r) + 1.6))
    ax.hlines(range(len(r)), 1, r.to_numpy(), color="#c9d3d9", lw=1)
    col = ["#be5146" if k == "original" else "#247ba0" for k in r.index]
    ax.scatter(r.to_numpy(), range(len(r)), c=col, zorder=3)
    ax.set_yticks(range(len(r)), list(r.index), fontsize=8)
    cd = fr["diferencia_critica_nemenyi"]
    ax.errorbar(r.iloc[0], -0.8, xerr=[[0], [cd]], fmt="none", ecolor="#444", capsize=4)
    ax.text(r.iloc[0] + cd / 2, -1.3, f"DC de Nemenyi = {cd:.2f}", ha="center", fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Rango medio por origen de la prueba (1 = menor error)")
    ax.set_title("Diagrama de diferencia crítica con el modelo original (rojo)")
    fig.tight_layout()
    fig.savefig(sal / "diagrama_diferencia_critica.png", dpi=130)
    plt.close(fig)


def _resumen_md(e3, sal, t, v, pot, tmcs, ts):
    idea = e3.decision()["idea"]
    lin = [f"# Resumen estadístico del Entregable 3 (generado {pd.Timestamp.now():%Y-%m-%d %H:%M})", "",
           f"Idea elegida en desarrollo: **{idea}**. R = {v.get('m') and len(pot['por_contraste']) and pot['por_contraste'][0]['R']} semillas; m = {v['m']} rivales.", "",
           f"**R3 cumplida: {'sí' if v['R3_cumplida'] else 'no'}** (tope por R3: {v['tope_por_R3']}).", "",
           "| Condición (§8.4) | Resultado |", "|---|---|",
           f"| Mejor en media que todos | {v['mejor_en_media_que_todos']} |", f"| Las tres pruebas rechazan en todos (Holm) | {v['todas_las_pruebas_rechazan_en_todos']} |",
           f"| IC de la diferencia más allá de δ en todos | {v['diferencia_supera_delta_en_todos']} |", f"| Efecto no despreciable en todos | {v['efecto_no_despreciable_en_todos']} |",
           f"| Consistencia en adicionales (≥ 2 de 3, ninguno peor) | {v.get('consistencia_adicionales')} |", f"| Adecuación del original | {v.get('adecuacion_original')} |",
           f"| Conclusión robusta a la prueba | {v['conclusion_robusta_a_la_prueba']} |", "",
           f"Contrastes superados: {v['contrastes_superados']} de {v['m']}. Rival más difícil: {v['rival_mas_dificil']}. Prueba que menos rechaza: {v['prueba_que_menos_rechaza']}.",
           f"Potencia conjunta: {pot['potencia_conjunta']:.3f}; mínima por contraste: {pot['minima_potencia_por_contraste']:.3f}.",
           f"Original en el MCS al 90 %: {v['original_en_MCS']} (único: {v['original_unico_en_MCS']}).", "", "## Contrastes", "",
           "| Rival | MASE original | MASE rival | Diferencia relativa (IC 95 %) | p máx. Holm | δ Cliff | Decisión |", "|---|---|---|---|---|---|---|"]
    for r in t.sort_values("MASE_rival").itertuples():
        lin.append(f"| {r.rival} | {r.MASE_original:.4f} | {r.MASE_rival:.4f} | {r.diferencia_relativa_pct:.2f} % ({r.ic95_relativa_pct[0]:.2f}; {r.ic95_relativa_pct[1]:.2f}) | "
                   f"{r.p_max_Holm:.4f} | {r.cliff_delta:.3f} | {r.decision} |")
    lin += ["", "## Sensibilidad de la conclusión", ""] + [f"* {r.escenario}: {r.r3}" for r in ts.itertuples()]
    (sal / "RESUMEN_E3.md").write_text("\n".join(lin) + "\n", encoding="utf-8")

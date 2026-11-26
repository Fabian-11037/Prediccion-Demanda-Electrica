"""Diagnóstico de información del texto: ¿los documentos oficiales predicen el error del modelo numérico?

Residuos fuera de muestra: predicciones de validación 2018-2021 (cada año, modelo entrenado solo con años anteriores) de
dos bases: la idea E (configuración 7, semilla 43) y TimesNet (configuración 19, semilla 43). Para cada origen d:
    r_d = media_h (y - ŷ) / ŷ   (residuo relativo con signo)      y      |r|_d = media_h |y - ŷ| / ŷ

Rasgos (solo documentos con disponible_desde <= origen, la regla del proyecto):
  C   calendario del día objetivo (día de la semana, mes, festivos y vecindad)
  R   residuos ya observables en d: r_{d-2} y media de r_{d-8..d-2} (la demanda horaria llega con 1 h de rezago)
  KW  ejes por palabras clave (10 temas), documentos en 72 h y en 7 días, y conteos por fuente
  LSA TF-IDF + SVD de 50 dimensiones ajustados solo con documentos disponibles antes de 2018; media de 72 h y presencia
  AN  días análogos: residuo medio de los 10 días pasados (<= d-2, misma serie) más parecidos en LSA de 7 días
Modelo: ridge (alfa por validación cruzada interna) con forward chaining: se prueba 2019, 2020 y 2021 entrenando con los
años anteriores. Placebo: los rasgos de texto del día d se reemplazan por los del día d - s (20 desplazamientos s en
35-330 días); conserva la densidad y el estilo del texto, pero rompe la alineación con el día.
Métrica operativa: MASE del pronóstico corregido ŷ (1 + r̂) frente al de la base, con q168 del contrato.
"""
from __future__ import annotations

import json
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

PAQ = Path(sys.argv[1])            # paquete con data/ y config/
RES = Path(sys.argv[2])            # carpeta de resultados (e3p1 y nbt)
SAL = Path(sys.argv[3])
SAL.mkdir(parents=True, exist_ok=True)
TZ = "America/Bogota"
ANIOS = [2018, 2019, 2020, 2021]
PRUEBA = [2019, 2020, 2021]
BASES = {"E": RES / "e3p1/resultados_e3_parte1/ajuste/busqueda/N/pliegue_{a}/cfg_07/pred_val_{a}.npz",
         "TimesNet": RES / "nbt/resultados_e2_nbeats_timesnet/nuevo_timesnet/busqueda/N/pliegue_{a}/cfg_19/pred_val_{a}.npz"}
EJES = {"ahorro_racionamiento": r"racionamiento|ahorro de energia|ahorrar energia|uso eficiente|apagon|deficit de energia",
        "calor": r"ola de calor|altas temperaturas|temperaturas? (alta|maxima)|calor",
        "nino_sequia": r"fenomeno (de )?el nino|el nino|sequia|deficit hidrico|tiempo seco|niveles? (bajos?|criticos?) de (los )?embalses",
        "nina_lluvia": r"la nina|lluvias|inundacion|temporada de lluvias|creciente",
        "festivo_vacaciones": r"festivo|puente|vacaciones|semana santa|dia sin carro",
        "social": r"\bparo\b|protesta|bloqueo|movilizacion|huelga",
        "covid": r"covid|coronavirus|cuarentena|aislamiento (preventivo|obligatorio)|confinamiento|pandemia",
        "falla_red": r"atentado|indisponibilidad|falla|desconexion|mantenimiento",
        "demanda_alza": r"(aumento|crecimiento|incremento) de la demanda|demanda (record|maxima|historica)",
        "demanda_baja": r"(disminucion|caida|reduccion|descenso) de la demanda"}


def plano(s: str) -> str:
    s = unicodedata.normalize("NFKD", (s or "").lower())
    return "".join(c for c in s if not unicodedata.combining(c))


# ------------------------------------------------------------------ residuos
filas = []
for base, patron in BASES.items():
    for a in ANIOS:
        z = np.load(str(patron).format(a=a))
        y, p = z["y_kwh"].astype(float), z["yhat_kwh"].astype(float)
        filas.append(pd.DataFrame(dict(base=base, dia=pd.to_datetime(z["origen"]), r=((y - p) / p).mean(1), rabs=(np.abs(y - p) / p).mean(1),
                                       ae=np.abs(y - p).mean(1))).assign(anio=a, y=list(y), p=list(p)))
D = pd.concat(filas, ignore_index=True)
contrato = json.loads((PAQ / "config/diseno_entrenamiento_texto.json").read_text(encoding="utf-8"))
q168 = float(contrato["metrica"]["q168_kwh"])

# ------------------------------------------------------------------ calendario del día objetivo (12:00, columnas diarias)
panel = pd.read_parquet(PAQ / "data/processed/panel_horario.parquet", columns=[c for c in contrato["numericas"]["calendario"] if c != "cal_hora"])
cal = panel[panel.index.hour == 12].copy()
cal.index = cal.index.tz_localize(None).normalize() if cal.index.tz is None else cal.index.tz_convert(TZ).tz_localize(None).normalize()
cols_dia = ["cal_es_festivo", "cal_es_no_laboral", "cal_es_vispera_festivo", "cal_es_dia_post_festivo", "cal_es_fin_semana_largo",
            "cal_es_semana_santa", "cal_es_temporada_navidena", "cal_es_fin_de_anio", "cal_dias_al_siguiente_festivo", "cal_dias_desde_festivo_previo"]
C = pd.concat([cal[cols_dia].astype(float), pd.get_dummies(cal.cal_dow.astype(int), prefix="dow", dtype=float),
               pd.get_dummies(cal.cal_mes.astype(int), prefix="mes", dtype=float)], axis=1)
C[["cal_dias_al_siguiente_festivo", "cal_dias_desde_festivo_previo"]] = np.log1p(C[["cal_dias_al_siguiente_festivo", "cal_dias_desde_festivo_previo"]].clip(0, 60))

# ------------------------------------------------------------------ corpus
corp = pd.read_parquet(PAQ / "data/processed/corpus_oficial.parquet").sort_values("disponible_desde").reset_index(drop=True)
corp["txt"] = [plano(f"{t} {x[:20000]}") for t, x in zip(corp.titulo, corp.texto.fillna(""))]
disp = pd.DatetimeIndex(corp.disponible_desde).tz_convert(TZ).tz_localize(None).as_unit("ns").asi8
dias = pd.DatetimeIndex(sorted(D.dia.unique()))
todos_dias = pd.date_range(dias.min() - pd.Timedelta(days=400), dias.max(), freq="D")


def ventana(d_ns: np.ndarray, horas: int) -> tuple[np.ndarray, np.ndarray]:
    hi = np.searchsorted(disp, d_ns, side="right")                 # disponible <= origen (00:00 del día objetivo)
    lo = np.searchsorted(disp, d_ns - horas * 3600 * 10**9, side="left")
    return lo, hi


ns = todos_dias.as_unit("ns").asi8
lo3, hi3 = ventana(ns, 72)
lo7, hi7 = ventana(ns, 168)
marca = {k: corp.txt.str.contains(v, regex=True).to_numpy().astype(float) for k, v in EJES.items()}
fuentes = sorted(corp.fuente.unique())
cum = lambda v: np.r_[0.0, np.cumsum(v)]                            # noqa: E731
KW = {}
for k, v in marca.items():
    c = cum(v)
    KW[f"kw72_{k}"], KW[f"kw7d_{k}"] = np.log1p(c[hi3] - c[lo3]), np.log1p(c[hi7] - c[lo7])
for f in fuentes:
    c = cum((corp.fuente == f).to_numpy().astype(float))
    KW[f"n72_{f}"] = np.log1p(c[hi3] - c[lo3])
KW["n72_total"], KW["n7d_total"] = np.log1p(hi3 - lo3), np.log1p(hi7 - lo7)
KW = pd.DataFrame(KW, index=todos_dias)

ajuste = disp < pd.Timestamp("2018-01-01").value                   # vocabulario y SVD solo con documentos anteriores a 2018
vec = TfidfVectorizer(min_df=5, max_df=0.5, sublinear_tf=True, max_features=50000)
Xa = vec.fit_transform(corp.txt[ajuste])
svd = TruncatedSVD(50, random_state=0).fit(Xa)
Z = svd.transform(vec.transform(corp.txt))
Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12
cz = np.vstack([np.zeros((1, Z.shape[1])), np.cumsum(Z, axis=0)])
n3 = (hi3 - lo3)[:, None]
LSA = pd.DataFrame(np.where(n3 > 0, (cz[hi3] - cz[lo3]) / np.maximum(n3, 1), 0.0), index=todos_dias, columns=[f"lsa{i}" for i in range(Z.shape[1])])
LSA["presencia72"] = (n3[:, 0] > 0).astype(float)
n7 = (hi7 - lo7)[:, None]
L7 = np.where(n7 > 0, (cz[hi7] - cz[lo7]) / np.maximum(n7, 1), 0.0)
L7 = pd.DataFrame(L7 / (np.linalg.norm(L7, axis=1, keepdims=True) + 1e-12), index=todos_dias)


def rasgos_texto(d: pd.DatetimeIndex, desplazamiento: int = 0) -> dict[str, pd.DataFrame]:
    fuente = d - pd.Timedelta(days=desplazamiento)
    return dict(KW=KW.loc[fuente].set_axis(d), LSA=LSA.loc[fuente].set_axis(d), L7=L7.loc[fuente].set_axis(d))


def analogos(serie: pd.DataFrame, L7d: pd.DataFrame, k: int = 10) -> np.ndarray:
    """Residuo medio de los k días pasados (<= d-2, misma base) más parecidos por texto de 7 días; 0 si no hay texto."""
    V, r = L7d.to_numpy(), serie.r.to_numpy()
    out = np.zeros(len(serie))
    for i in range(len(serie)):
        j = np.arange(0, max(0, i - 1))
        if len(j) < k or not V[i].any():
            continue
        s = V[j] @ V[i]
        top = j[np.argsort(-s)[:k]]
        out[i] = r[top].mean()
    return out


def matriz(serie: pd.DataFrame, conj: str, desplazamiento: int = 0) -> pd.DataFrame:
    d = pd.DatetimeIndex(serie.dia)
    r = serie.r.to_numpy()
    lag2 = np.r_[np.full(2, np.nan), r[:-2]]
    m7 = pd.Series(r).shift(2).rolling(7, min_periods=1).mean().to_numpy()
    X = [C.loc[d].reset_index(drop=True), pd.DataFrame(dict(r_lag2=np.nan_to_num(lag2), r_m7=np.nan_to_num(m7)))]
    if conj != "C+R":
        T = rasgos_texto(d, desplazamiento)
        if "KW" in conj:
            X.append(T["KW"].reset_index(drop=True))
        if "LSA" in conj:
            X.append(T["LSA"].reset_index(drop=True))
        if "AN" in conj:
            X.append(pd.DataFrame(dict(analogo=analogos(serie, T["L7"]))))
    return pd.concat(X, axis=1).astype(float)


def evaluar(serie: pd.DataFrame, conj: str, objetivo: str, desplazamiento: int = 0) -> dict:
    X = matriz(serie, conj, desplazamiento)
    y = serie[objetivo].to_numpy()
    pred = np.full(len(serie), np.nan)
    for a in PRUEBA:
        tr, te = (serie.anio < a).to_numpy(), (serie.anio == a).to_numpy()
        sc = StandardScaler().fit(X[tr])
        m = RidgeCV(alphas=np.logspace(-1, 5, 25)).fit(sc.transform(X[tr]), y[tr])
        pred[te] = m.predict(sc.transform(X[te]))
    te = serie.anio.isin(PRUEBA).to_numpy()
    yt, pt = y[te], pred[te]
    out = dict(R2_fuera_de_muestra=float(1 - ((yt - pt) ** 2).sum() / ((yt - yt.mean()) ** 2).sum()))
    if objetivo == "r":
        Y, P = np.stack(serie.y.to_numpy()[te]), np.stack(serie.p.to_numpy()[te])
        base = np.abs(Y - P).mean() / q168
        corr = np.abs(Y - P * (1 + pt[:, None])).mean() / q168
        out.update(MASE_base=float(base), MASE_corregido=float(corr), mejora_pct=float(100 * (base - corr) / base))
        for a in PRUEBA:
            s = (serie.anio[te] == a).to_numpy()
            b_a = np.abs(Y[s] - P[s]).mean() / q168
            out[f"mejora_pct_{a}"] = float(100 * (b_a - np.abs(Y[s] - P[s] * (1 + pt[s, None])).mean() / q168) / b_a)
    return out


rng = np.random.default_rng(20261120)
desplaz = sorted(rng.choice(np.arange(35, 331), 20, replace=False).tolist())
CONJ = ["C+R", "C+R+KW", "C+R+LSA", "C+R+AN", "C+R+KW+LSA+AN"]
res, plac = [], []
for base in BASES:
    serie = D[D.base == base].sort_values("dia").reset_index(drop=True)
    for objetivo in ("r", "rabs"):
        ref = evaluar(serie, "C+R", objetivo)
        for conj in CONJ:
            e = evaluar(serie, conj, objetivo)
            fila = dict(base=base, objetivo=objetivo, conjunto=conj, **e)
            if conj != "C+R":
                clave = "mejora_pct" if objetivo == "r" else "R2_fuera_de_muestra"
                fila["ganancia_sobre_C+R"] = e[clave] - ref[clave]
                pl = []
                for s in desplaz:
                    ep = evaluar(serie, conj, objetivo, s)
                    pl.append(ep[clave] - ref[clave])
                    plac.append(dict(base=base, objetivo=objetivo, conjunto=conj, desplazamiento=s, ganancia_sobre_C_R=pl[-1]))
                pl = np.array(pl)
                fila.update(placebo_media=float(pl.mean()), placebo_p95=float(np.percentile(pl, 95)),
                            p_placebo=float((1 + (pl >= fila["ganancia_sobre_C+R"]).sum()) / (1 + len(pl))))
            res.append(fila)
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in fila.items()}, flush=True)
R = pd.DataFrame(res)
R.to_csv(SAL / "diagnostico_texto.csv", index=False)
pd.DataFrame(plac).to_csv(SAL / "diagnostico_texto_placebos.csv", index=False)

# ------------------------------------------------------------------ descriptivos útiles
desc = {}
cob = pd.DataFrame(dict(n72=np.expm1(KW["n72_total"]), kw=np.expm1(KW[[c for c in KW if c.startswith("kw72_")]]).sum(1)), index=todos_dias)
desc["documentos_por_dia_72h_media"] = {str(a): float(cob.n72[cob.index.year == a].mean()) for a in ANIOS}
desc["dias_sin_documentos_72h_pct"] = {str(a): float(100 * (cob.n72[cob.index.year == a] == 0).mean()) for a in ANIOS}
for base in BASES:
    s = D[D.base == base].set_index("dia")
    t = s.join(KW[["kw7d_covid", "kw7d_social", "kw7d_ahorro_racionamiento"]])
    desc[f"{base}_corr_residuo"] = {k: float(np.corrcoef(t.r, t[k])[0, 1]) for k in ["kw7d_covid", "kw7d_social", "kw7d_ahorro_racionamiento"]}
    desc[f"{base}_residuo_medio_pct_por_mes_2020"] = {str(m): float(100 * s.r[(s.index.year == 2020) & (s.index.month == m)].mean()) for m in range(1, 13)}
    desc[f"{base}_residuo_medio_pct_por_anio"] = {str(a): float(100 * s.r[s.index.year == a].mean()) for a in ANIOS}
docs_covid = corp[(marca["covid"] > 0)]
desc["documentos_covid_por_anio"] = pd.DatetimeIndex(docs_covid.disponible_desde).year.value_counts().sort_index().to_dict()
desc["documentos_covid_primeros"] = [(str(t)[:10], f, x[:90]) for t, f, x in zip(docs_covid.disponible_desde[:6], docs_covid.fuente[:6], docs_covid.titulo[:6])]
desc["desplazamientos_placebo"] = desplaz
(SAL / "diagnostico_texto_descriptivos.json").write_text(json.dumps(desc, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
print(json.dumps(desc, ensure_ascii=False, indent=1, default=str))

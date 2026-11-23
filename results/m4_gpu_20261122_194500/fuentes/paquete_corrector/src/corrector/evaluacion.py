"""Evaluación preregistrada del corrector (años 2019-2021 de la partición de desarrollo).

Unidad: el origen. Para cada variante, el error escalado de un origen es la media sobre semillas de
mean_h |y - ŷ| / q168; el MASE es su media sobre orígenes. Reducción relativa de A frente a B:
G = 100 (MASE_B - MASE_A) / MASE_B. IC del 95 %: bootstrap pareado por origen con bloques circulares de 14 días dentro
de cada año (9.999 remuestreos). p unilateral de mejora: proporción de estadísticos bootstrap centrados que superan al
observado. Holm dentro de cada familia.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CONTRASTES = [("num_horario", "C+R"), ("oficial_reciente", "C+R"), ("oficial", "C+R"), ("oficial_prensa", "C+R"), ("metadatos", "C+R"),
              ("oficial", "num_horario"), ("oficial_prensa", "num_horario"), ("oficial_prensa", "metadatos"), ("oficial", "oficial_reciente"),
              ("oficial_prensa", "desalineado")]


def indices_bootstrap(anios: np.ndarray, B: int, bloque: int, semilla: int) -> np.ndarray:
    """(B, n) índices de remuestreo por bloques circulares dentro de cada año."""
    rng = np.random.default_rng(semilla)
    n = len(anios)
    out = np.empty((B, n), dtype="int64")
    for a in np.unique(anios):
        pos = np.flatnonzero(anios == a)
        m = len(pos)
        k = int(np.ceil(m / bloque))
        ini = rng.integers(0, m, size=(B, k))
        rel = (ini[:, :, None] + np.arange(bloque)[None, None, :]).reshape(B, -1)[:, :m] % m
        out[:, pos] = pos[rel]
    return out


def holm(p: list[float]) -> list[float]:
    p = np.asarray(p, dtype=float)
    orden = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    acum = 0.0
    for r, i in enumerate(orden):
        acum = max(acum, min(1.0, (m - r) * p[i]))
        adj[i] = acum
    return adj.tolist()


def contraste(err_a: np.ndarray, err_b: np.ndarray, idx: np.ndarray) -> dict:
    ma, mb = err_a.mean(), err_b.mean()
    g = 100 * (mb - ma) / mb
    gb = 100 * (err_b[idx].mean(1) - err_a[idx].mean(1)) / err_b[idx].mean(1)
    p = (1 + np.sum(gb - g >= g)) / (len(gb) + 1)
    lo, hi = np.percentile(gb, [2.5, 97.5])
    return dict(MASE_A=float(ma), MASE_B=float(mb), reduccion_pct=float(g), ic95_inf=float(lo), ic95_sup=float(hi), p_unilateral=float(p))


def evaluar(errores: dict, anios: np.ndarray, subconjuntos: dict, B: int = 9999, bloque: int = 14, semilla: int = 20261121) -> pd.DataFrame:
    """``errores``: variante -> (n_origenes,) error escalado medio sobre semillas; incluye 'C+R', 'E' y 'desalineado'
    (media de los desplazamientos). ``subconjuntos``: nombre -> máscara booleana de orígenes."""
    filas = []
    for nombre, msk in subconjuntos.items():
        idx = indices_bootstrap(anios[msk], B, bloque, semilla)
        fam = []
        for a, b in CONTRASTES:
            if a in errores and b in errores:
                r = contraste(errores[a][msk], errores[b][msk], idx)
                fam.append(dict(subconjunto=nombre, A=a, B=b, n_origenes=int(msk.sum()), **r))
        ph = holm([f["p_unilateral"] for f in fam])
        for f, q in zip(fam, ph):
            f["p_holm"] = q
        filas += fam
    return pd.DataFrame(filas)


def por_hora(err_h: dict, a: str, b: str) -> pd.DataFrame:
    """Reducción por hora del horizonte (descriptiva). ``err_h``: variante -> (n, 24)."""
    ma, mb = err_h[a].mean(0), err_h[b].mean(0)
    return pd.DataFrame(dict(hora=np.arange(24), A=a, B=b, MASE_A=ma, MASE_B=mb, reduccion_pct=100 * (mb - ma) / mb))


def veredicto(t: pd.DataFrame, margen: float = 3.0) -> dict:
    g = t[t.subconjunto == "global"].set_index(["A", "B"])
    out = {}
    for v in ("oficial", "oficial_prensa"):
        controles = [(v, "num_horario")] + ([(v, "metadatos"), (v, "desalineado")] if v == "oficial_prensa" else [])
        texto = all(k in g.index and g.loc[k, "p_holm"] < 0.05 and g.loc[k, "reduccion_pct"] > 0 for k in controles)
        rel = (v, "C+R") in g.index and g.loc[(v, "C+R"), "reduccion_pct"] >= margen and g.loc[(v, "C+R"), "ic95_inf"] > 0
        cota_sup = max(float(g.loc[k, "ic95_sup"]) for k in controles if k in g.index) if any(k in g.index for k in controles) else None
        out[v] = dict(aporte_del_texto_demostrado=bool(texto), reduccion_relevante_frente_a_CR=bool(rel),
                      se_descarta_aporte_de_texto_de_3pct=bool(cota_sup is not None and cota_sup < margen), cota_superior_ic_aporte_texto=cota_sup)
    return out

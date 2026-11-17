"""Construcción de ventanas por origen sin fuga.

Garantía estructural: para el origen ``t0 = indice[p]`` solo se leen
  * ``x_*`` (as-of) y ``cal_*`` en filas ``p-L .. p-1``;
  * ``cal_*`` (y, si se pasa, pronósticos con emisión previa) en filas ``p .. p+H-1``;
  * ``y_*`` solo en las filas ``p .. p+H-1`` como etiqueta.
Las columnas ``x_*`` jamás se leen en filas ``>= p`` (prueba en ``tests/``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .particion import AjustadorEscala


def columnas_por_clase(registro: dict, clase: str) -> list[str]:
    return [c for c, v in registro.items() if isinstance(v, dict) and v.get("clase") == clase]


def construir_ventanas(panel: pd.DataFrame, posiciones: np.ndarray, columnas_x: list[str], columnas_cal: list[str],
                       ventana: int = C.VENTANA, horizonte: int = C.HORIZONTE, escala: AjustadorEscala | None = None,
                       pronostico: np.ndarray | None = None, permitir_nan: bool = False) -> dict:
    """Devuelve arrays alineados por origen.

    ``pronostico``: tensor (n_filas_panel, H, k) de ``pronosticos.pronostico_futuro`` (opcional).
    Los orígenes con NaN en la entrada se descartan salvo ``permitir_nan=True``.
    """
    for c in columnas_x:
        if not c.startswith("x_"):
            raise ValueError(f"{c} no es una entrada past_only (prefijo x_)")
    for c in columnas_cal:
        if not c.startswith("cal_"):
            raise ValueError(f"{c} no es known_future (prefijo cal_)")
    X = panel[columnas_x].to_numpy(dtype="float64")
    K = panel[columnas_cal].to_numpy(dtype="float64")
    Y = panel["y_" + C.OBJETIVO].to_numpy(dtype="float64")

    xs, cs, cf, ys, ps, orig = [], [], [], [], [], []
    for p in posiciones:
        xw = X[p - ventana: p]
        if not permitir_nan and np.isnan(xw).any():
            continue
        xs.append(xw)
        cs.append(K[p - ventana: p])
        cf.append(K[p: p + horizonte])
        ys.append(Y[p: p + horizonte])
        if pronostico is not None:
            ps.append(pronostico[p])
        orig.append(panel.index[p])
    out = {
        "x_pasado": np.stack(xs) if xs else np.empty((0, ventana, len(columnas_x))),
        "cal_pasado": np.stack(cs) if cs else np.empty((0, ventana, len(columnas_cal))),
        "cal_futuro": np.stack(cf) if cf else np.empty((0, horizonte, len(columnas_cal))),
        "y": np.stack(ys) if ys else np.empty((0, horizonte)),
        "origen": pd.DatetimeIndex(orig),
        "columnas_x": columnas_x,
        "columnas_cal": columnas_cal,
    }
    if pronostico is not None:
        out["pronostico_futuro"] = np.stack(ps) if ps else np.empty((0, horizonte, pronostico.shape[-1]))
    if escala is not None:
        out["x_pasado"] = escala.transformar(out["x_pasado"].reshape(-1, len(columnas_x)), columnas_x).reshape(out["x_pasado"].shape)
        ycol = ["y_" + C.OBJETIVO]
        out["y_escalado"] = escala.transformar(out["y"].reshape(-1, 1), ycol).reshape(out["y"].shape)
    return out

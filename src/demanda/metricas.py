"""Métricas de pronóstico (primaria: MASE medio sobre el horizonte)."""
from __future__ import annotations

import numpy as np


def mase_por_horizonte(y: np.ndarray, yhat: np.ndarray, escala: float) -> np.ndarray:
    """MAE_h / escala para cada horizonte h (columnas)."""
    return np.abs(y - yhat).mean(axis=0) / escala


def evaluar(y: np.ndarray, yhat: np.ndarray, escala: float, horizontes=(1, 12, 24)) -> dict:
    """Métricas agregadas. ``y`` y ``yhat`` tienen forma (n_origenes, H) en kWh.

    MASE = media sobre h de MAE_h / escala (escala = MAE del ingenuo estacional en entrenamiento).
    """
    err = yhat - y
    mh = mase_por_horizonte(y, yhat, escala)
    out = {
        "MASE": float(mh.mean()),
        "RMSE": float(np.sqrt((err ** 2).mean())),
        "MAE": float(np.abs(err).mean()),
        "sMAPE": float((200 * np.abs(err) / (np.abs(y) + np.abs(yhat))).mean()),
        "MAPE": float((100 * np.abs(err) / np.abs(y)).mean()),
    }
    for h in horizontes:
        if 1 <= h <= y.shape[1]:
            out[f"MASE_h{h}"] = float(mh[h - 1])
    return out


def n_semillas_requerido(sd: float, media: float, delta_rel: float, alfa: float = 0.05, potencia: float = 0.80) -> int:
    """Número de ejecuciones por modelo para detectar una diferencia relativa ``delta_rel``.

    Aproximación normal para dos muestras independientes con igual desviación ``sd``:
    n = 2 * ((z_{1-alfa/2} + z_{potencia}) * sd / (delta_rel * media))**2. Es una estimación
    de piloto (no definitiva); con sd = 0 (modelo determinista) devuelve 1.
    """
    from scipy.stats import norm

    if sd <= 0 or media <= 0 or delta_rel <= 0:
        return 1
    z = norm.ppf(1 - alfa / 2) + norm.ppf(potencia)
    return int(np.ceil(2 * (z * sd / (delta_rel * media)) ** 2))

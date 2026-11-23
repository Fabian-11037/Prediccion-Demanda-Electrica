"""Idea J: agregación de expertos con ancla acotada, y las reglas de referencia que exige el Entregable 3 para una fusión.

Notación: K expertos, pronósticos P_k (n, H) en kWh; la parada interna de cada experto (orígenes anteriores a la
validación) se usa como historia inicial; en validación las pérdidas llegan con ``retraso`` orígenes (la demanda horaria
se publica con 1 h de rezago, así que el día del origen d − 1 no está completo en d; con retraso 2 se usan solo días
completos).

Reglas (todas con pesos por hora del horizonte):
  mejor        el experto de menor MAE en la parada (k*), fijo
  media        promedio simple
  inversa      pesos ∝ 1 / MAE de la parada, fijos
  apilamiento  mínimos cuadrados no negativos por hora en la parada, fijos
  ewa          pesos exponenciales en línea, w_k ∝ exp(−η Σ ℓ_k) con ℓ = |y − ŷ| / q168 recortada a [0, B], η = √(8 ln K / T) / B
  anclada_β    (1 − β) P_{k*} + β P_ewa                                                       (propuesta)

Proposición J. Para toda hora y todo origen, por convexidad del valor absoluto,
    |y − ŷ_anclada| ≤ (1 − β) |y − ŷ_{k*}| + β |y − ŷ_ewa|,
así que MAE(anclada) ≤ MAE(k*) + β (MAE(ewa) − MAE(k*)); y como la pérdida de EWA está a lo sumo a
B √(T ln K / 2) / T del mejor experto en retrospectiva (cota clásica de pesos exponenciales con pérdidas acotadas),
MAE(anclada) ≤ MAE(k*) + β (min_k MAE_k − MAE(k*) + B √(ln K / (2T)) + retraso · B / T).
La propuesta nunca empeora al experto ancla en más de β veces el arrepentimiento de EWA. La desigualdad de convexidad se
verifica numéricamente en cada aplicación (``cota_verificada``).
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import nnls


def _mae(y, p):
    return float(np.abs(y - p).mean())


def ewa_en_linea(P: np.ndarray, y: np.ndarray, q168: float, retraso: int, cota: float) -> tuple[np.ndarray, np.ndarray]:
    """P (K, T, H), y (T, H) en orden temporal. Devuelve (pronóstico (T, H), pesos (T, K, H)). La fila t solo usa
    pérdidas de las filas ≤ t − retraso."""
    K, T, H = P.shape
    ell = np.clip(np.abs(y[None] - P) / q168, 0.0, cota)                      # (K, T, H)
    eta = math.sqrt(8.0 * math.log(max(K, 2)) / T) / cota
    acum = np.concatenate([np.zeros((K, 1, H)), np.cumsum(ell, axis=1)], axis=1)   # acum[:, t] = Σ_{s<t}
    idx = np.clip(np.arange(T) - retraso + 1, 0, T)                             # pérdidas de s ≤ t − retraso
    C = acum[:, idx, :]                                                         # (K, T, H)
    logw = -eta * (C - C.min(axis=0, keepdims=True))
    w = np.exp(logw)
    w /= w.sum(axis=0, keepdims=True)
    return (w * P).sum(axis=0), w.transpose(1, 0, 2)


def reglas(P_es: np.ndarray, y_es: np.ndarray, P_v: np.ndarray, y_v: np.ndarray, q168: float, retraso: int = 2,
           betas=(0.25, 0.5), cota: float = 5.0) -> tuple[dict, dict]:
    """P_es (K, n_es, H) y P_v (K, n_v, H). Devuelve ({regla: pronóstico de validación}, información)."""
    K = P_es.shape[0]
    mae_es = np.array([_mae(y_es, P_es[k]) for k in range(K)])
    k_ast = int(np.argmin(mae_es))
    out = dict(mejor=P_v[k_ast], media=P_v.mean(axis=0))
    w_inv = (1.0 / mae_es) / (1.0 / mae_es).sum()
    out["inversa"] = np.tensordot(w_inv, P_v, axes=1)
    H = P_v.shape[2]
    w_ap = np.zeros((K, H))
    for h in range(H):
        w_ap[:, h] = nnls(P_es[:, :, h].T.astype("float64"), y_es[:, h].astype("float64"))[0]
    out["apilamiento"] = np.einsum("kh,knh->nh", w_ap, P_v)
    P = np.concatenate([P_es, P_v], axis=1)
    y = np.concatenate([y_es, y_v], axis=0)
    comb, w = ewa_en_linea(P, y, q168, retraso, cota)
    n_es = P_es.shape[1]
    out["ewa"] = comb[n_es:]
    verif = {}
    for b in betas:
        a = (1 - b) * P_v[k_ast] + b * out["ewa"]
        out[f"anclada_{b:g}"] = a
        lhs = np.abs(y_v - a)
        rhs = (1 - b) * np.abs(y_v - P_v[k_ast]) + b * np.abs(y_v - out["ewa"])
        verif[f"anclada_{b:g}"] = bool((lhs <= rhs + 1e-6 * (1 + rhs)).all())
    info = dict(experto_ancla=k_ast, MAE_parada=mae_es.tolist(), pesos_inversa=w_inv.tolist(), pesos_apilamiento_media_h=w_ap.mean(axis=1).tolist(),
                pesos_ewa_final_media_h=w[-1].mean(axis=1).tolist(), eta=math.sqrt(8.0 * math.log(max(K, 2)) / P.shape[1]) / cota,
                cota_verificada=verif, retraso=retraso)
    return out, info

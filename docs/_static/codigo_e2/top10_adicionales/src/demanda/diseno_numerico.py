"""Contrato numérico ampliado; no ajusta transformaciones ni entrena modelos."""
from __future__ import annotations

import pandas as pd


def preparar_numericas_comunes(panel: pd.DataFrame, registro: dict, contrato: dict):
    """Selecciona entradas y aplica solo el rezago adicional al clima ya desplazado.

    El índice de cada fila representa el comienzo de la hora; su información
    está habilitada al final. Las ventanas futuras solo pueden leer calendario.
    Las versiones históricas de ERA5 siguen sin estar certificadas.
    """
    if not isinstance(panel.index, pd.DatetimeIndex) or panel.index.tz is None:
        raise ValueError("Se requiere índice temporal con zona horaria")
    if not panel.index.is_unique or not panel.index.is_monotonic_increasing:
        raise ValueError("Índice duplicado o desordenado")
    if len(panel) > 1 and not panel.index.to_series().diff().iloc[1:].eq(pd.Timedelta(hours=1)).all():
        raise ValueError("El panel debe tener una malla horaria completa")
    x = contrato["numericas"]["historicas"]
    cal = contrato["numericas"]["calendario"]
    if len(x) != len(set(x)) or len(cal) != len(set(cal)) or set(x) & set(cal):
        raise ValueError("Columnas duplicadas o roles incompatibles")
    for c in x:
        if registro[c]["clase"] != "past_only":
            raise ValueError(f"Entrada histórica inválida: {c}")
    for c in cal:
        if registro[c]["clase"] != "known_future":
            raise ValueError(f"Entrada futura inválida: {c}")
    out = panel.loc[:, x + cal].copy()
    lag = contrato["numericas"]["rezago_clima_h"]
    for c in x:
        if c.startswith("x_obs_"):
            original = registro[c]["rezago_h"]
            if not isinstance(original, int) or not isinstance(lag, int) or lag < original:
                raise ValueError("No se puede adelantar la disponibilidad climática")
            out[c] = out[c].shift(lag - original)
    # Ninguna imputación o escala se estima usando este panel completo.
    mascara = out.isna().astype("uint8").add_prefix("ausente_")
    return out, mascara

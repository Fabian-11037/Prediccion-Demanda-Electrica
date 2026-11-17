"""Pronósticos meteorológicos de emisión previa (``known_future`` condicionada).

El clima observado de la hora objetivo (ERA5) NO es una entrada válida: en
producción solo existe un pronóstico emitido antes del origen. Esta capa solo
admite pronósticos con emisión conocida.

Archivo esperado: ``data/external/pronosticos_temperatura/pronosticos_temperatura.parquet``
(lo genera ``scripts/adquisicion/descargar_pronosticos_temperatura.py``) con columnas
``valido`` (hora Colombia, inicio del intervalo), ``ciudad``, ``lead_dias`` (N >= 1)
y ``temperature_2m_c``. El pronóstico N días antes de ``valido`` se emitió, de forma
aproximada, en ``valido - N*24h``.

Regla de disponibilidad en el origen ``t0``::

    valido - N*24h + MARGEN + LATENCIA <= t0

Con ``valido = t0 + h`` solo depende del horizonte ``h``: ``N_h = ceil((h + MARGEN + LATENCIA)/24)``.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C

RUTA = C.EXTERNO / "pronosticos_temperatura" / "pronosticos_temperatura.parquet"


def lead_requerido(h: int, latencia_h: int = C.PRONOSTICO_LATENCIA_H, margen_h: int = C.PRONOSTICO_MARGEN_H) -> int:
    """Menor N (días) cuyo pronóstico está disponible en t0 para el intervalo t0+h."""
    return max(1, math.ceil((h + latencia_h + margen_h) / 24))


def cargar_pronosticos(ruta: Path | str = RUTA) -> pd.DataFrame | None:
    ruta = Path(ruta)
    if not ruta.exists():
        return None
    d = pd.read_parquet(ruta)
    faltan = {"valido", "ciudad", "lead_dias", "temperature_2m_c"} - set(d.columns)
    if faltan:
        raise ValueError(f"Faltan columnas: {sorted(faltan)}")
    if (d["lead_dias"] < 1).any():
        raise ValueError("lead_dias < 1 no es un pronóstico previo")
    return d


def matriz_por_lead(indice: pd.DatetimeIndex, d: pd.DataFrame, ciudades: list[str], lead: int, modelo: str | None = None) -> np.ndarray:
    """Pronóstico a ``lead`` días por (hora de ``indice``, ciudad); NaN si falta.

    Si el archivo trae varios modelos hay que elegir uno con ``modelo`` (no se promedian en silencio)."""
    if "modelo" in d.columns:
        if modelo is None and d["modelo"].nunique() > 1:
            raise ValueError("El archivo contiene varios modelos: indicar modelo=")
        if modelo is not None:
            d = d[d["modelo"] == modelo]
    s = d[d["lead_dias"] == lead].pivot_table(index="valido", columns="ciudad", values="temperature_2m_c", aggfunc="last")
    return s.reindex(index=indice, columns=ciudades).to_numpy(dtype="float32")


def pronostico_futuro(indice: pd.DatetimeIndex, d: pd.DataFrame, ciudades: list[str], horizonte: int, modelo: str | None = None) -> np.ndarray:
    """Tensor (n_filas, H, n_ciudades) con el pronóstico utilizable para t0 = fila p.

    ``out[p, h]`` es el pronóstico para el intervalo ``p+h`` emitido con el lead
    mínimo disponible en t0 (``lead_requerido(h)``). No contiene nada posterior al origen.
    """
    n = len(indice)
    leads = sorted({lead_requerido(h) for h in range(horizonte)})
    mats = {n_: matriz_por_lead(indice, d, ciudades, n_, modelo) for n_ in leads}
    out = np.full((n, horizonte, len(ciudades)), np.nan, dtype="float32")
    for h in range(horizonte):
        m = mats[lead_requerido(h)]
        out[: n - h, h, :] = m[h:, :]
    return out

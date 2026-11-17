"""Climatología esperada de temperatura, ajustada solo con la partición de entrenamiento.

Es un sustituto sin fuga del pronóstico de temperatura: depende únicamente de
(mes, hora) y de estadísticas de entrenamiento, por lo que es ``known_future``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


class ClimatologiaTemperatura:
    def __init__(self, ciudades=None):
        self.ciudades = list(ciudades or C.CIUDADES)
        self.tabla_: pd.DataFrame | None = None

    def ajustar(self, panel: pd.DataFrame, fin_train: str) -> "ClimatologiaTemperatura":
        fin = pd.Timestamp(fin_train, tz=C.TZ) + pd.Timedelta(days=1)  # exclusivo
        tr = panel.loc[panel.index < fin]
        cols = [f"x_obs_temperature_2m_c_{c}" for c in self.ciudades]
        g = tr[cols].groupby([tr.index.month, tr.index.hour]).mean()
        g.columns = self.ciudades
        g.index.names = ["mes", "hora"]
        self.tabla_ = g
        return self

    def transformar(self, indice: pd.DatetimeIndex) -> pd.DataFrame:
        if self.tabla_ is None:
            raise RuntimeError("Llamar a ajustar() primero")
        clave = pd.MultiIndex.from_arrays([indice.month, indice.hour], names=["mes", "hora"])
        out = self.tabla_.reindex(clave)
        out.index = indice
        return out.add_prefix("cal_clima_esperado_")

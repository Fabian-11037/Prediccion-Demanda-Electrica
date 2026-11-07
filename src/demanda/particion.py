"""Partición cronológica por origen, con purga y guarda del conjunto de prueba."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def limites(split: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """(primer instante, último instante-hora) de la partición, hora de Colombia."""
    a, b = C.PARTICION[split]
    return pd.Timestamp(a, tz=C.TZ), pd.Timestamp(b, tz=C.TZ) + pd.Timedelta(hours=23)


def origenes(indice: pd.DatetimeIndex, split: str, horizonte: int = C.HORIZONTE, ventana: int = C.VENTANA,
             paso_h: int = 24, hora_origen: int = 0, allow_test: bool = False) -> np.ndarray:
    """Posiciones (enteros) del panel que son orígenes válidos de ``split``.

    Reglas:
      * el origen es ``t0 = indice[p]`` con ``t0.hour == hora_origen`` y paso ``paso_h``;
      * todos los intervalos-objetivo ``t0..t0+(H-1)h`` están dentro de la partición
        (esto aplica la purga de ``H`` horas antes de la frontera siguiente);
      * ``t0 >= inicio`` de la partición (la entrada puede tomar datos previos: son pasado);
      * hay ``ventana`` filas de historia y la fila ``p-1`` existe.
    """
    if split == "test" and C.PRUEBA_CONGELADA and not allow_test:
        raise PermissionError("El conjunto de prueba está congelado: usar allow_test=True solo en la corrida final")
    ini, fin = limites(split)
    pos = np.arange(len(indice))
    ok = (indice >= ini) & (indice + pd.Timedelta(hours=horizonte - 1) <= fin)
    ok &= pos >= ventana
    ok &= pos + horizonte <= len(indice)
    if paso_h <= 0 or (paso_h < 24 and 24 % paso_h != 0) or (paso_h > 24 and paso_h % 24 != 0):
        raise ValueError("paso_h debe dividir a 24 o ser múltiplo de 24")
    if paso_h < 24:
        ok &= (indice.hour % paso_h) == (hora_origen % paso_h)
    else:
        ok &= indice.hour == hora_origen
    sel = pos[ok]
    if paso_h > 24:  # submuestreo adicional (múltiplos de 24 h)
        sel = sel[:: paso_h // 24]
    return sel


class AjustadorEscala:
    """Media y desviación por columna calculadas solo con filas de entrenamiento."""

    def __init__(self):
        self.media_: pd.Series | None = None
        self.std_: pd.Series | None = None

    def ajustar(self, panel: pd.DataFrame, columnas: list[str], fin_train: str | None = None) -> "AjustadorEscala":
        fin = pd.Timestamp(fin_train or C.PARTICION["train"][1], tz=C.TZ) + pd.Timedelta(days=1)
        tr = panel.loc[panel.index < fin, columnas]
        self.media_ = tr.mean()
        self.std_ = tr.std().replace(0, 1.0)
        return self

    def transformar(self, a: np.ndarray, columnas: list[str]) -> np.ndarray:
        m = self.media_[columnas].to_numpy(dtype="float64")
        s = self.std_[columnas].to_numpy(dtype="float64")
        return ((a - m) / s).astype("float32")


def escala_mase(panel: pd.DataFrame, rezago_h: int = 168, fin_train: str | None = None) -> float:
    """MAE del ingenuo estacional (rezago ``rezago_h``) sobre entrenamiento.

    Usar rezago 168 h (semanal) o 24 h según el comparador elegido en el preregistro.
    Nota: es una escala en unidades de demanda; con tendencia creciente conviene
    una escala relativa al nivel (p. ej. MAPE/sMAPE o MASE sobre la razón).
    """
    fin = pd.Timestamp(fin_train or C.PARTICION["train"][1], tz=C.TZ) + pd.Timedelta(days=1)
    y = panel.loc[panel.index < fin, "y_" + C.OBJETIVO]
    return float((y - y.shift(rezago_h)).abs().dropna().mean())

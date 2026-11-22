"""Partición cronológica por origen, con purga y guarda del conjunto de prueba."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def limites(split: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """(primer instante, último instante-hora) de la partición, hora de Colombia."""
    a, b = C.PARTICION[split]
    return pd.Timestamp(a, tz=C.TZ), pd.Timestamp(b, tz=C.TZ) + pd.Timedelta(hours=23)


def _candidatos(indice: pd.DatetimeIndex, horizonte: int, ventana: int, paso_h: int, hora_origen: int) -> np.ndarray:
    """Posiciones con historia suficiente, horizonte completo y la regla de paso/hora de origen."""
    if paso_h <= 0 or (paso_h < 24 and 24 % paso_h != 0) or (paso_h > 24 and paso_h % 24 != 0):
        raise ValueError("paso_h debe dividir a 24 o ser múltiplo de 24")
    pos = np.arange(len(indice))
    ok = (pos >= ventana) & (pos + horizonte <= len(indice))
    if paso_h < 24:
        ok &= (indice.hour % paso_h) == (hora_origen % paso_h)
    else:
        ok &= indice.hour == hora_origen
    return pos[ok]


def _submuestrear(sel: np.ndarray, paso_h: int) -> np.ndarray:
    return sel[:: paso_h // 24] if paso_h > 24 else sel


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
    cand = _candidatos(indice, horizonte, ventana, paso_h, hora_origen)
    t = indice[cand]
    sel = cand[(t >= ini) & (t + pd.Timedelta(hours=horizonte - 1) <= fin)]
    return _submuestrear(sel, paso_h)


def _dividir_es(pos: np.ndarray, indice: pd.DatetimeIndex, horizonte: int, frac_es: float) -> tuple[np.ndarray, np.ndarray]:
    """Separa los últimos ``frac_es`` orígenes para parada temprana, con purga de ``horizonte`` horas."""
    n_es = max(1, int(round(frac_es * len(pos))))
    es = pos[-n_es:]
    corte = indice[es[0]] - pd.Timedelta(hours=horizonte)  # el objetivo de entrenamiento debe terminar antes del inicio de es
    tr = pos[indice[pos] + pd.Timedelta(hours=horizonte - 1) < indice[es[0]]]
    assert len(tr) and indice[tr].max() + pd.Timedelta(hours=horizonte - 1) < indice[es].min() and corte <= indice[es[0]]
    return tr, es


def folds_forward_chaining(indice: pd.DatetimeIndex, anios_val=None, horizonte: int = C.HORIZONTE, ventana: int = C.VENTANA,
                           paso_h: int = 24, hora_origen: int = 0, frac_es: float = C.FRACCION_PARADA_TEMPRANA) -> list[dict]:
    """Validación cruzada de ventana expansiva (forward chaining) con bloques anuales.

    Para cada año ``y`` de validación: ``val`` = orígenes cuyo objetivo cae en ``y``; el resto son orígenes
    anteriores cuyo objetivo termina antes del 1-ene de ``y`` (purga). De esos, los últimos ``frac_es`` se
    usan solo para parada temprana (``es``) y el resto para entrenar (``tr``). Los pliegues solo pueden
    caer dentro de la partición de validación propuesta: la prueba no se toca.
    """
    anios_val = list(anios_val or C.CV_ANIOS)
    cand = _candidatos(indice, horizonte, ventana, paso_h, hora_origen)
    t = indice[cand]
    _, fin_val = limites("val")
    folds = []
    for y in sorted(anios_val):
        ini = pd.Timestamp(f"{y}-01-01", tz=C.TZ)
        fin = pd.Timestamp(f"{y}-12-31", tz=C.TZ) + pd.Timedelta(hours=23)
        if fin > fin_val:
            raise PermissionError(f"El pliegue {y} sale de la partición de validación y tocaría la prueba")
        val = cand[(t >= ini) & (t + pd.Timedelta(hours=horizonte - 1) <= fin)]
        pre = cand[t + pd.Timedelta(hours=horizonte - 1) < ini]
        val, pre = _submuestrear(val, paso_h), _submuestrear(pre, paso_h)
        tr, es = _dividir_es(pre, indice, horizonte, frac_es)
        folds.append({"anio": y, "tr": tr, "es": es, "val": val})
    return folds


def pretest(indice: pd.DatetimeIndex, horizonte: int = C.HORIZONTE, ventana: int = C.VENTANA, paso_h: int = 24,
            hora_origen: int = 0, frac_es: float = C.FRACCION_PARADA_TEMPRANA) -> tuple[np.ndarray, np.ndarray]:
    """Entrenamiento y parada temprana para el ajuste final: todos los orígenes cuyo objetivo cae antes de la prueba."""
    ini_test, _ = limites("test")
    cand = _candidatos(indice, horizonte, ventana, paso_h, hora_origen)
    pre = _submuestrear(cand[indice[cand] + pd.Timedelta(hours=horizonte - 1) < ini_test], paso_h)
    return _dividir_es(pre, indice, horizonte, frac_es)


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

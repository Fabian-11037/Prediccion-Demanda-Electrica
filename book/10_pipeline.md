# Código completo del pipeline de entrenamiento y prueba

Las fuentes corresponden al commit 02ff447d94fa78e1cde323fa2382da5bce24fb6e. Los bloques siguientes conservan el código de la ejecución; el análisis del libro no invoca sus funciones de entrenamiento. El controlador GPU registra caché, pesos y predicciones por ajuste; la interfaz central admite CPU y GPU.

## Reproducción desde una copia independiente

Utilizar una copia del commit indicado con el mismo panel y registro de variables en data/processed. Instalar las dependencias del entrenamiento y la variante de PyTorch adecuada al dispositivo. Una reproducción debe escribir en un directorio nuevo y conservar las adendas y el límite LSTM.

```python
from pathlib import Path
import shutil
salida=Path("results/reproduccion_e1")
salida.mkdir(parents=True,exist_ok=True)
shutil.copyfile("results/base_v2/limites_busqueda.json",salida/"limites_busqueda.json")
```

```bash
python scripts/entrenar_modelos_base.py buscar --salida results/reproduccion_e1 --n-config 20 --semillas-por-config 3 --max-epocas 200 --dispositivo auto
python scripts/entrenar_modelos_base.py piloto --salida results/reproduccion_e1 --max-epocas 200 --dispositivo auto
python scripts/entrenar_modelos_base.py final --salida results/reproduccion_e1 --max-epocas 200 --n-semillas 195 --dispositivo auto --allow-test --prereg PREREGISTRO.md
```

Los comandos son una receta de reproducción y no se ejecutan al construir este libro. El controlador original de la GPU utiliza fcntl y es específico de Linux; el pipeline central anterior puede invocarse desde Windows o Linux con las dependencias apropiadas. No se afirma identidad bit a bit entre dispositivos.

## src/demanda/config.py

```python
"""Constantes del problema y reglas de disponibilidad (a priori).

Los valores marcados como ``supuesto`` no están certificados por la fuente: son
decisiones conservadoras que deben declararse en el preregistro. Cámbialos aquí
(y solo aquí) para que panel, ventanas y pruebas permanezcan coherentes.
"""
from __future__ import annotations

from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
DATOS = RAIZ / "data"
RAW = DATOS / "raw"
EXTERNO = DATOS / "external"
PROCESADO = DATOS / "processed"

TZ = "America/Bogota"  # Colombia no usa horario de verano

# ---- Problema ---------------------------------------------------------------
OBJETIVO = "demanda_real_kwh"  # demanda real horaria del SIN
FRECUENCIA = "h"
HORIZONTE = 24  # intervalos horarios por pronóstico (propuesta; ver CONTEXTO_PROYECTO.md)
VENTANA = 168  # horas de historia en la entrada (una semana; propuesta)

# El origen t0 es un instante en hora entera. El pronóstico cubre los intervalos
# que empiezan en t0, t0+1h, ..., t0+(H-1)h. La entrada solo puede contener
# información cuya disponibilidad sea <= t0.

# ---- Partición cronológica (PROPUESTA, no preregistrada) --------------------
# Cada intervalo-objetivo debe quedar completo dentro de una sola partición
# (purga = HORIZONTE). Las fechas son inclusivas, hora de Colombia.
PARTICION = {
    "train": ("2000-01-01", "2017-12-31"),
    "val": ("2018-01-01", "2021-12-31"),
    "test": ("2022-01-01", "2025-12-31"),
}
# Validación cruzada de selección: forward chaining (ventana expansiva). Cada pliegue valida en un año
# calendario y entrena con todo lo anterior (con purga = HORIZONTE). Nunca toca la prueba.
CV_ANIOS = [2018, 2019, 2020, 2021]
FRACCION_PARADA_TEMPRANA = 0.10  # últimos orígenes (cronológicos) del entrenamiento reservados para la parada temprana
PRUEBA_CONGELADA = True  # `test` solo con allow_test=True (una sola corrida final)

# ---- Rezagos de disponibilidad ---------------------------------------------
# Horas entre el fin del intervalo observado y el momento en que el valor puede
# usarse. Todos son supuestos conservadores (la fuente no certifica horas de
# publicación históricas).
REZAGO_H = {
    "demanda_real": 1,  # supuesto: el dato horario puede usarse 1 h después de cerrar la hora
    "precio_bolsa": 24,  # supuesto: el precio ex post se publica al día siguiente
    "clima_obs": 1,  # supuesto: observación disponible 1 h después (ERA5 es reanálisis)
    "hidrologia": 24,  # supuesto: dato diario disponible 24 h después de terminar el día
}
# ISE (DANE): se publica ~45-49 días después de terminar el mes de referencia;
# se usa 60 días como cota conservadora. Las cifras se revisan: ver docs.
ISE_REZAGO_DIAS = 60
# Pronósticos meteorológicos: emisión aproximada + latencia de disponibilidad.
PRONOSTICO_LATENCIA_H = 6
# Margen por la semántica de ``temperature_2m_previous_dayN`` (la corrida usada puede
# no ser exactamente la de ``valido - N*24h``): se suma a la emisión aproximada.
PRONOSTICO_MARGEN_H = 6

CIUDADES = ["bogota", "medellin", "cali", "barranquilla", "cartagena", "bucaramanga", "pereira", "cucuta"]
VARIABLES_CLIMA = ["temperature_2m_c", "relative_humidity_2m_pct", "precipitation_preceding_hour_mm"]

# Habitantes por ciudad (censo 2018 del DANE ajustado por omisión censal, cifras citadas por Portafolio,
# 17-oct-2019; fuente secundaria: sustituir por el archivo oficial del DANE cuando se descargue).
# Son pesos estáticos: el censo se publicó después de 2018, pero no depende de la demanda; declararlo.
RUTA_CIUDADES = RAW / "numeric" / "ciudades.csv"
COLUMNA_HABITANTES = "habitantes_censo_2018_ajustado"


def pesos_ciudades() -> "dict[str, float]":
    """Pesos normalizados (suman 1) proporcionales a los habitantes de cada ciudad de ``CIUDADES``."""
    import pandas as pd

    t = pd.read_csv(RUTA_CIUDADES).set_index("ciudad")[COLUMNA_HABITANTES]
    t = t.loc[CIUDADES].astype("float64")
    return (t / t.sum()).to_dict()
```

## src/demanda/particion.py

```python
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
```

## src/demanda/ventanas.py

```python
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
```

## src/demanda/metricas.py

```python
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
```

## src/demanda/modelos_base.py

```python
"""Tres modelos base en PyTorch (ARX lineal con penalización L2/L1, MLP y LSTM) con un pipeline común.

LightGBM queda como modelo opcional en CPU (no usa PyTorch).

Todos usan las mismas ventanas, la misma partición y la misma normalización
(ajustada con entrenamiento). Interfaz común::

    modelo = CLASES[nombre]()
    info = modelo.ajustar(train, val, params, semilla, opciones)   # val solo para parada temprana
    yhat_norm = modelo.predecir(conjunto)                          # (n, H) en escala normalizada
    yhat = datos.invertir(yhat_norm, conjunto)                     # kWh

``Conjunto`` es un contenedor de arrays por partición. Las covariables se eligen con
``ConfigDatos.covariables``: ``serie`` (solo demanda), ``calendario`` (demanda + festivos y
calendario) o ``completo`` (además precio, hidrología y clima ponderado por habitantes).
"""
from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Necesario para el determinismo estricto en CUDA (debe definirse antes de iniciar CUDA)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from . import config as C  # noqa: E402
from .particion import AjustadorEscala, escala_mase, origenes
from .ventanas import columnas_por_clase, construir_ventanas  # noqa: E402

COVARIABLES = ("serie", "calendario", "completo")
NORMALIZACIONES = ("nivel", "global")
PERDIDAS = ("mse", "mae", "huber")


# --------------------------------------------------------------------------- datos
@dataclass
class ConfigDatos:
    covariables: str = "calendario"
    ventana: int = C.VENTANA
    horizonte: int = C.HORIZONTE
    paso_h: int = 24
    hora_origen: int = 0
    normalizacion: str = "nivel"
    clima_por_ciudad: bool = False  # solo aplica con covariables='completo'
    max_origenes_train: int | None = None  # submuestreo uniforme para pruebas rápidas
    escala_mase_lag: int = 168


@dataclass
class Conjunto:
    x: np.ndarray  # (n, L, Fx) entradas as-of; columna 0 = demanda
    cal_p: np.ndarray  # (n, L, Kc)
    cal_f: np.ndarray  # (n, H, Kc)
    y: np.ndarray  # (n, H) kWh
    origen: pd.DatetimeIndex
    nivel: np.ndarray  # (n,) media de la demanda en la ventana (modo 'nivel')

    def __len__(self):
        return len(self.y)


def _columnas(registro: dict, cfg: ConfigDatos) -> tuple[list[str], list[str]]:
    past = columnas_por_clase(registro, "past_only")
    cal = columnas_por_clase(registro, "known_future")
    x = ["x_" + C.OBJETIVO]
    if cfg.covariables == "serie":
        return x, []
    if cfg.covariables == "calendario":
        return x, cal
    extra = ["x_precio_bolsa_cop_kwh"] + [c for c in past if c.startswith("x_hid_")]
    clima = [c for c in past if c.startswith("x_obs_") and (not c.endswith("_ponderada")) == cfg.clima_por_ciudad]
    return x + extra + clima, cal


class Datos:
    """Prepara train/val(/test) con normalización ajustada solo con entrenamiento."""

    def __init__(self, panel: pd.DataFrame, registro: dict, cfg: ConfigDatos):
        if cfg.covariables not in COVARIABLES or cfg.normalizacion not in NORMALIZACIONES:
            raise ValueError("covariables o normalizacion inválidas")
        self.panel, self.registro, self.cfg = panel, registro, cfg
        self.cols_x, self.cols_cal = _columnas(registro, cfg)
        ajuste = self.cols_x + self.cols_cal + ["y_" + C.OBJETIVO]
        self.escala = AjustadorEscala().ajustar(panel, ajuste)
        self.escala_mase = escala_mase(panel, cfg.escala_mase_lag)
        self._cache: dict[str, Conjunto] = {}

    def conjunto(self, split: str, allow_test: bool = False) -> Conjunto:
        """Conjunto de una partición fija (train, val o test)."""
        if split in self._cache:
            return self._cache[split]
        cfg = self.cfg
        pos = origenes(self.panel.index, split, cfg.horizonte, cfg.ventana, cfg.paso_h, cfg.hora_origen, allow_test=allow_test)
        if split == "train" and cfg.max_origenes_train and len(pos) > cfg.max_origenes_train:
            pos = pos[np.linspace(0, len(pos) - 1, cfg.max_origenes_train).astype(int)]
        out = self.conjunto_pos(pos)
        self._cache[split] = out
        return out

    def conjunto_pos(self, pos: np.ndarray, clave: str | None = None) -> Conjunto:
        """Ventanas normalizadas para posiciones de origen arbitrarias (p. ej. pliegues de validación cruzada)."""
        if clave is not None and clave in self._cache:
            return self._cache[clave]
        cfg = self.cfg
        w = construir_ventanas(self.panel, pos, self.cols_x, self.cols_cal, cfg.ventana, cfg.horizonte)
        x = w["x_pasado"].astype("float64")
        nivel = x[:, :, 0].mean(axis=1)
        d = self.escala
        # normalización: la demanda según el modo; el resto con estadísticas de entrenamiento
        for j, c in enumerate(self.cols_x):
            if j == 0 and cfg.normalizacion == "nivel":
                x[:, :, 0] = x[:, :, 0] / nivel[:, None]
            else:
                x[:, :, j] = (x[:, :, j] - d.media_[c]) / d.std_[c]
        cs = (w["cal_pasado"] - d.media_[self.cols_cal].to_numpy()) / d.std_[self.cols_cal].to_numpy() if self.cols_cal else w["cal_pasado"]
        cf = (w["cal_futuro"] - d.media_[self.cols_cal].to_numpy()) / d.std_[self.cols_cal].to_numpy() if self.cols_cal else w["cal_futuro"]
        out = Conjunto(x.astype("float32"), cs.astype("float32"), cf.astype("float32"), w["y"], w["origen"], nivel)
        if clave is not None:
            self._cache[clave] = out
        return out

    def objetivo_norm(self, c: Conjunto) -> np.ndarray:
        if self.cfg.normalizacion == "nivel":
            return (c.y / c.nivel[:, None]).astype("float32")
        k = "y_" + C.OBJETIVO
        return ((c.y - self.escala.media_[k]) / self.escala.std_[k]).astype("float32")

    def invertir(self, yhat_norm: np.ndarray, c: Conjunto) -> np.ndarray:
        if self.cfg.normalizacion == "nivel":
            return yhat_norm * c.nivel[:, None]
        k = "y_" + C.OBJETIVO
        return yhat_norm * self.escala.std_[k] + self.escala.media_[k]

    def referencias(self, split: str, allow_test: bool = False) -> dict[str, np.ndarray]:
        """Ingenuos de una partición fija (kWh); ver ``referencias_pos``."""
        cfg = self.cfg
        pos = origenes(self.panel.index, split, cfg.horizonte, cfg.ventana, cfg.paso_h, cfg.hora_origen, allow_test=allow_test)
        return self.referencias_pos(pos, self.conjunto(split, allow_test))

    def referencias_pos(self, pos: np.ndarray, c: Conjunto) -> dict[str, np.ndarray]:
        """Ingenuos con la misma disponibilidad que los modelos (kWh).

        ``y[p+h-24k]`` con k mínimo tal que el dato ya exista en t0 (rezago de demanda incluido).
        """
        cfg = self.cfg
        pos = pos[np.isin(self.panel.index[pos], c.origen)]  # alinear con orígenes no descartados
        y = self.panel["y_" + C.OBJETIVO].to_numpy()
        lag = C.REZAGO_H["demanda_real"]
        out = {}
        for nombre, base in (("ingenuo_24h", 24), ("ingenuo_168h", 168)):
            pred = np.zeros((len(pos), cfg.horizonte))
            for h in range(cfg.horizonte):
                k = int(np.ceil((h + 1 + lag) / base)) if base == 24 else 1
                pred[:, h] = y[pos + h - base * max(k, 1)]
            out[nombre] = pred
        return out


# ------------------------------------------------------- características tabulares
def _tabular_base(c: Conjunto, lags: int) -> np.ndarray:
    """Historia de demanda (últimos ``lags``) + resumen de las demás entradas (último, media 24 h, media ventana)."""
    n = len(c)
    partes = [c.x[:, -lags:, 0]]
    if c.x.shape[2] > 1:
        e = c.x[:, :, 1:]
        partes += [e[:, -1, :], e[:, -24:, :].mean(axis=1), e.mean(axis=1)]
    return np.concatenate(partes, axis=1) if partes else np.empty((n, 0))


def tabular_lineal(c: Conjunto, lags: int) -> np.ndarray:
    base = _tabular_base(c, lags)
    return np.concatenate([base, c.cal_f.reshape(len(c), -1)], axis=1).astype("float32")


def tabular_arboles(c: Conjunto, lags: int) -> np.ndarray:
    """Una fila por (origen, h): historia + calendario del intervalo objetivo + h."""
    n, H = len(c), c.y.shape[1]
    base = _tabular_base(c, lags)
    filas = np.empty((n, H, base.shape[1] + c.cal_f.shape[2] + 1), dtype="float32")
    filas[:, :, : base.shape[1]] = base[:, None, :]
    filas[:, :, base.shape[1]: -1] = c.cal_f
    filas[:, :, -1] = np.arange(H)[None, :]
    return filas.reshape(n * H, -1)


# --------------------------------------------------------------- espacios de búsqueda
def _cat(*v):
    return {"tipo": "cat", "valores": list(v)}


def _fl(a, b, log=False):
    return {"tipo": "float", "min": a, "max": b, "log": log}


def _int(a, b, log=False):
    return {"tipo": "int", "min": a, "max": b, "log": log}


# Hiperparámetros de optimización comunes a los tres modelos PyTorch (versión 2 del preregistro)
_OPT = {
    "lr": _fl(1e-4, 1e-2, True),
    "scheduler": _cat("none", "cosine", "plateau"),
    "perdida": _cat("mse", "mae", "huber"),
    "paciencia": _cat(8, 15, 25),
}

ESPACIOS = {
    "arx_lineal": {  # 8 hiperparámetros (4 propios + 4 de optimización): regresión lineal con penalización L2 (ridge) o L1 (lasso) sobre rezagos
        "penalizacion": _cat("ridge", "lasso"),
        "lambda": {"condicional_en": "penalizacion", "ridge": _fl(1e-6, 1e-1, True), "lasso": _fl(1e-7, 1e-2, True)},
        "lags": _cat(24, 48, 96, 168),
        "batch_size": _cat(32, 64, 128, 256),
        **_OPT,
    },
    "mlp": {  # 9 hiperparámetros (5 propios + 4 de optimización)
        "capas": _cat(1, 2, 3),
        "unidades": _cat(64, 128, 256, 512),
        "dropout": _fl(0.0, 0.5),
        "weight_decay": _fl(1e-6, 1e-2, True),
        "batch_size": _cat(32, 64, 128, 256),
        **_OPT,
    },
    "lstm": {  # 9 hiperparámetros (5 propios + 4 de optimización)
        "capas": _cat(1, 2, 3),
        "unidades": _cat(32, 64, 128, 256),
        "dropout": _fl(0.0, 0.5),
        "weight_decay": _fl(1e-6, 1e-2, True),
        "batch_size": _cat(32, 64, 128, 256),
        **_OPT,
    },
    "cnn1d": {
        "capas": _cat(1, 2, 3),
        "filtros": _cat(32, 64, 128),
        "kernel": _cat(3, 5, 7),
        "dropout": _fl(0.0, 0.5),
        "weight_decay": _fl(1e-6, 1e-2, True),
        "batch_size": _cat(32, 64, 128, 256),
        **_OPT,
    },
    "lightgbm": {  # opcional (CPU, no usa PyTorch); n_estimators máximo y paciencia: opciones de línea de comandos
        "num_leaves": _int(8, 128, True),
        "max_depth": _cat(-1, 4, 6, 8),
        "min_child_samples": _int(10, 200, True),
        "reg_lambda": _fl(1e-3, 100.0, True),
    },
}
MODELOS_PYTORCH = ("arx_lineal", "mlp", "lstm", "cnn1d")


def _muestra(spec: dict, rng: np.random.Generator):
    t = spec["tipo"]
    if t == "cat":
        return spec["valores"][int(rng.integers(len(spec["valores"])))]
    lo, hi, log = spec["min"], spec["max"], spec.get("log", False)
    if t == "float":
        return float(np.exp(rng.uniform(np.log(lo), np.log(hi)))) if log else float(rng.uniform(lo, hi))
    return int(round(float(np.exp(rng.uniform(np.log(lo), np.log(hi)))))) if log else int(rng.integers(lo, hi + 1))


def muestrear(espacio: dict, rng: np.random.Generator, fijos: dict | None = None) -> dict:
    """Una configuración aleatoria; ``fijos`` sustituye parámetros (no se buscan)."""
    fijos = fijos or {}
    p = {}
    for k, spec in espacio.items():
        if k in fijos:
            p[k] = fijos[k]
        elif "condicional_en" in spec:
            continue
        else:
            p[k] = _muestra(spec, rng)
    for k, spec in espacio.items():
        if "condicional_en" in spec and k not in fijos:
            p[k] = _muestra(spec[p[spec["condicional_en"]]], rng)
    return p


# ----------------------------------------------------------------- entrenamiento PyTorch
@dataclass
class Opciones:
    """Opciones de entrenamiento que NO se buscan (iguales para todos los modelos que las usan)."""
    max_epocas: int = 200  # modelos PyTorch (tope alto: la parada temprana decide cuándo termina)
    paciencia: int = 8  # épocas (PyTorch) o rondas (LightGBM) sin mejora en validación
    max_arboles: int = 1500  # LightGBM
    lr: float = 1e-3  # modelos PyTorch (AdamW); valor por defecto si 'lr' no está en los parámetros
    min_delta: float = 1e-6  # mejora mínima de la pérdida de validación para reiniciar la paciencia
    huber_delta: float | None = None  # None: 0.05 con normalización 'nivel' y 0.5 con 'global' (unidades de la pérdida normalizada)
    lr_arboles: float = 0.05  # LightGBM
    perdida: str = "mse"  # valor por defecto si 'perdida' no está en los parámetros
    clip: float = 1.0  # recorte de norma del gradiente
    hilos: int = 1  # hilos de CPU
    dispositivo: str = "auto"  # auto | cpu | cuda | cuda:N
    determinismo: str = "estricto"  # estricto | rapido
    amp: bool = False  # precisión mixta bfloat16 (solo CUDA)
    max_gb_gpu: float = 2.0  # si los datos de entrenamiento superan esto, quedan en CPU y se envían por lotes


def resolver_dispositivo(txt: str):
    import torch

    if txt == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = torch.device(txt)
    if d.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Se pidió CUDA pero torch.cuda.is_available() es False: revisar controlador y versión de PyTorch con CUDA")
    return d


def fijar_semillas(semilla: int, op: Opciones):
    import torch

    random.seed(semilla)
    np.random.seed(semilla)
    torch.manual_seed(semilla)
    torch.cuda.manual_seed_all(semilla)
    torch.set_num_threads(op.hilos)
    if op.determinismo == "estricto":
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def _definir_redes():
    """Define las redes al primer uso para no importar PyTorch al cargar el módulo."""
    import torch
    from torch import nn

    class Tabular(nn.Module):
        """Entrada tabular calculada en el propio dispositivo: historia de demanda, resumen de las
        demás entradas y calendario del horizonte, estandarizados con estadísticas de entrenamiento."""

        def __init__(s, n_x, lags, H, kc):
            super().__init__()
            s.n_x, s.lags = n_x, lags
            s.d = lags + 3 * (n_x - 1) + H * kc
            s.register_buffer("mu", torch.zeros(s.d))
            s.register_buffer("sd", torch.ones(s.d))

        def feats(s, xp, cf):
            partes = [xp[:, -s.lags:, 0]]
            if s.n_x > 1:
                e = xp[:, :, 1: s.n_x]
                partes += [e[:, -1], e[:, -24:].mean(1), e.mean(1)]
            partes.append(cf.flatten(1))
            return torch.cat(partes, 1)

        @torch.no_grad()
        def preparar(s, xp, cf):  # solo con entrenamiento
            f = torch.cat([s.feats(xp[i:i + 2048], cf[i:i + 2048]) for i in range(0, len(xp), 2048)])
            s.mu.copy_(f.mean(0))
            s.sd.copy_(f.std(0).clamp_min(1e-6))

        def entrada(s, xp, cf):
            return (s.feats(xp, cf) - s.mu) / s.sd

        def penalizacion(s):
            return 0.0

    class ARXLineal(Tabular):
        def __init__(s, n_x, lags, H, kc, penalizacion, lam):
            super().__init__(n_x, lags, H, kc)
            s.lin = nn.Linear(s.d, H)
            s.pen, s.lam = penalizacion, lam

        def forward(s, xp, cf):
            return s.lin(s.entrada(xp, cf))

        def penalizacion(s):
            w = s.lin.weight
            return s.lam * ((w ** 2).sum() if s.pen == "ridge" else w.abs().sum())

    class MLP(Tabular):
        def __init__(s, n_x, lags, H, kc, capas, unidades, dropout):
            super().__init__(n_x, lags, H, kc)
            capa, d = [], s.d
            for _ in range(capas):
                capa += [nn.Linear(d, unidades), nn.ReLU(), nn.Dropout(dropout)]
                d = unidades
            s.red = nn.Sequential(*capa, nn.Linear(d, H))

        def forward(s, xp, cf):
            return s.red(s.entrada(xp, cf))

    class LSTMNet(nn.Module):
        def __init__(s, f_in, H, kc, capas, unidades, dropout):
            super().__init__()
            s.lstm = nn.LSTM(f_in, unidades, capas, batch_first=True, dropout=dropout if capas > 1 else 0.0)
            s.drop = nn.Dropout(dropout)
            s.cal = nn.Linear(H * kc, unidades) if kc else None
            s.head = nn.Sequential(nn.Linear(unidades * (2 if kc else 1), unidades), nn.ReLU(), nn.Dropout(dropout), nn.Linear(unidades, H))

        def forward(s, xp, cf):
            _, (h, _) = s.lstm(xp)
            z = s.drop(h[-1])
            if s.cal is not None:
                z = torch.cat([z, torch.relu(s.cal(cf.flatten(1)))], 1)
            return s.head(z)

        def preparar(s, xp, cf):
            pass

        def penalizacion(s):
            return 0.0

    return Tabular, ARXLineal, MLP, LSTMNet


def _red_cnn1d(f_in, H, kc, capas, filtros, kernel, dropout):
    """Convoluciones sobre historia y rama separada para calendario futuro conocido."""
    import torch
    from torch import nn

    class PromedioSiete(nn.Module):
        def forward(self, x):
            # L=168 es divisible entre siete: bloques disjuntos de 24 horas.
            # Equivale a AdaptiveAvgPool1d(7), con backward determinista.
            if x.shape[-1] % 7:
                raise ValueError("La historia CNN debe ser divisible entre siete")
            return x.reshape(x.shape[0], x.shape[1], 7, x.shape[-1] // 7).mean(dim=-1)

    class CNN1DNet(nn.Module):
        def __init__(self):
            super().__init__()
            blocks = []
            channels = f_in
            for _ in range(capas):
                blocks.extend([nn.Conv1d(channels, filtros, kernel, padding=kernel // 2),
                               nn.ReLU(), nn.Dropout1d(dropout)])
                channels = filtros
            self.convs = nn.Sequential(*blocks)
            self.historia = nn.Sequential(PromedioSiete(), nn.Flatten(),
                                          nn.Linear(7 * filtros, filtros), nn.ReLU(), nn.Dropout(dropout))
            self.cal = nn.Linear(H * kc, filtros) if kc else None
            self.head = nn.Sequential(nn.Linear(filtros * (2 if kc else 1), filtros), nn.ReLU(),
                                      nn.Dropout(dropout), nn.Linear(filtros, H))

        def forward(self, xp, cf):
            z = self.historia(self.convs(xp.transpose(1, 2)))
            if self.cal is not None:
                z = torch.cat([z, torch.relu(self.cal(cf.flatten(1)))], dim=1)
            return self.head(z)

        def preparar(self, xp, cf):
            pass

        def penalizacion(self):
            return 0.0

    return CNN1DNet()


def construir_red(nombre: str, p: dict, tr: Conjunto):
    _, ARX, MLP, LSTMNet = _definir_redes()
    n_x, L = tr.x.shape[2], tr.x.shape[1]
    H, kc = tr.cal_f.shape[1], tr.cal_f.shape[2]
    if nombre == "cnn1d":
        return _red_cnn1d(n_x + tr.cal_p.shape[2], H, kc, int(p["capas"]), int(p["filtros"]),
                          int(p["kernel"]), float(p["dropout"]))
    if nombre == "arx_lineal":
        return ARX(n_x, min(int(p["lags"]), L), H, kc, p["penalizacion"], float(p["lambda"]))
    if nombre == "mlp":
        return MLP(n_x, L, H, kc, int(p["capas"]), int(p["unidades"]), float(p["dropout"]))
    return LSTMNet(n_x + tr.cal_p.shape[2], H, kc, int(p["capas"]), int(p["unidades"]), float(p["dropout"]))


class ModeloTorch:
    """Entrenador común (AdamW, parada temprana sobre la pérdida de validación, restauración del mejor estado).

    Los datos completos se mueven una sola vez al dispositivo cuando caben (``max_gb_gpu``);
    si no, se envían por lotes."""

    determinista = False

    def __init__(self, nombre: str):
        self.nombre = nombre

    def _tensores(self, c: Conjunto, y: np.ndarray | None, dev, en_dev: bool):
        import torch

        xp = torch.from_numpy(np.concatenate([c.x, c.cal_p], axis=2))
        cf = torch.from_numpy(c.cal_f)
        yy = torch.from_numpy(y) if y is not None else None
        if en_dev:
            xp, cf = xp.to(dev), cf.to(dev)
            yy = yy.to(dev) if yy is not None else None
        elif dev.type == "cuda":
            xp, cf = xp.pin_memory(), cf.pin_memory()
        return xp, cf, yy

    def ajustar(self, tr: Conjunto, va: Conjunto, datos: Datos, p: dict, semilla: int, op: Opciones) -> dict:
        import torch
        from torch import nn

        fijar_semillas(semilla, op)
        self.dev = dev = resolver_dispositivo(op.dispositivo)
        self.red = red = construir_red(self.nombre, p, tr).to(dev)
        f_in = tr.x.shape[2] + tr.cal_p.shape[2]
        gb = (tr.x.nbytes + tr.cal_p.nbytes + tr.cal_f.nbytes) / 1e9
        en_dev = dev.type != "cuda" or gb <= op.max_gb_gpu
        xt, ct, yt = self._tensores(tr, datos.objetivo_norm(tr), dev, en_dev)
        xv, cv, yv = self._tensores(va, datos.objetivo_norm(va), dev, True)
        gen = torch.Generator().manual_seed(semilla)
        if en_dev:
            red.preparar(xt, ct)
        else:  # datos grandes en CPU: estadísticas con una submuestra uniforme de entrenamiento
            sel = torch.randperm(len(xt), generator=gen)[:16384]
            red.preparar(xt[sel].to(dev), ct[sel].to(dev))
        delta_h = op.huber_delta if op.huber_delta is not None else (0.05 if datos.cfg.normalizacion == "nivel" else 0.5)
        perdida = p.get("perdida", op.perdida)
        lossf = {"mse": nn.MSELoss(), "mae": nn.L1Loss(), "huber": nn.HuberLoss(delta=delta_h)}[perdida]
        opt = torch.optim.AdamW(red.parameters(), lr=float(p.get("lr", op.lr)), weight_decay=float(p.get("weight_decay", 0.0)))
        paciencia = int(p.get("paciencia", op.paciencia))
        sched_nombre = p.get("scheduler", "none")
        if sched_nombre == "cosine":
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=op.max_epocas)
        elif sched_nombre == "plateau":
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=max(2, paciencia // 3))
        else:
            sched = None
        bs = int(p["batch_size"])
        usar_amp = op.amp and dev.type == "cuda"
        mejor, estado, malas, hist = np.inf, None, 0, []

        def ac():
            return torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=usar_amp)

        for ep in range(1, op.max_epocas + 1):
            red.train()
            perm = torch.randperm(len(xt), generator=gen)
            tot = 0.0
            for i in range(0, len(perm), bs):
                idx = perm[i: i + bs]
                xb, cb, yb = xt[idx.to(xt.device)].to(dev, non_blocking=True), ct[idx.to(ct.device)].to(dev, non_blocking=True), yt[idx.to(yt.device)].to(dev, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with ac():
                    loss = lossf(red(xb, cb).float(), yb)
                (loss + red.penalizacion()).backward()
                nn.utils.clip_grad_norm_(red.parameters(), op.clip)
                opt.step()
                tot += float(loss.detach()) * len(idx)
            red.eval()
            with torch.no_grad(), ac():
                vl = float(sum(lossf(red(xv[i:i + 4096], cv[i:i + 4096]).float(), yv[i:i + 4096]) * len(yv[i:i + 4096]) for i in range(0, len(xv), 4096)) / len(xv))
            hist.append({"epoca": ep, "train_loss": tot / len(xt), "val_loss": vl})
            observador = globals().get("PROGRESO_ENTRENAMIENTO")
            if observador is not None:
                observador(hist[-1])
            if sched is not None:
                sched.step(vl) if sched_nombre == "plateau" else sched.step()
            if vl < mejor - op.min_delta:
                mejor, malas = vl, 0
                estado = {k: v.detach().clone() for k, v in red.state_dict().items()}
            else:
                malas += 1
                if malas >= paciencia:
                    break
        red.load_state_dict(estado)
        n_par = int(sum(q.numel() for q in red.parameters()))
        out = {"n_parametros": n_par, "epocas": len(hist), "mejor_epoca": int(np.argmin([h["val_loss"] for h in hist]) + 1),
               "dispositivo": str(dev), "historia": hist}
        if self.nombre == "arx_lineal":
            out["n_no_nulos"] = int((red.lin.weight.detach().abs() > 1e-6).sum())
        return out

    def predecir(self, c: Conjunto) -> np.ndarray:
        import torch

        self.red.eval()
        xp = torch.from_numpy(np.concatenate([c.x, c.cal_p], axis=2))
        cf = torch.from_numpy(c.cal_f)
        sal = []
        with torch.no_grad():
            for i in range(0, len(xp), 4096):
                sal.append(self.red(xp[i:i + 4096].to(self.dev), cf[i:i + 4096].to(self.dev)).float().cpu().numpy())
        return np.concatenate(sal).astype("float32")


class LightGBM:
    """Opcional (CPU): no usa PyTorch ni GPU. Se activa con --modelos lightgbm."""
    nombre = "lightgbm"
    determinista = False

    def ajustar(self, tr, va, datos, p, semilla, op):
        import lightgbm as lgb

        self.lags = tr.x.shape[1]
        Xtr, Xva = tabular_arboles(tr, self.lags), tabular_arboles(va, self.lags)
        ytr, yva = datos.objetivo_norm(tr).reshape(-1), datos.objetivo_norm(va).reshape(-1)
        self.m = lgb.LGBMRegressor(
            n_estimators=op.max_arboles, learning_rate=op.lr_arboles, num_leaves=int(p["num_leaves"]),
            max_depth=int(p["max_depth"]), min_child_samples=int(p["min_child_samples"]), reg_lambda=float(p["reg_lambda"]),
            colsample_bytree=0.5, subsample=0.8, subsample_freq=1, random_state=semilla, n_jobs=op.hilos,
            deterministic=True, force_row_wise=True, verbose=-1)
        res = {}
        self.m.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="l2",
                   callbacks=[lgb.early_stopping(op.paciencia, verbose=False), lgb.record_evaluation(res)])
        curva = res["valid_0"]["l2"]
        hist = [{"epoca": i + 1, "val_loss": float(v)} for i, v in enumerate(curva)]
        n_arb = int(self.m.best_iteration_ or len(curva))
        hojas = int(self.m.booster_.trees_to_dataframe().query("tree_index < @n_arb")["split_feature"].isna().sum())
        return {"n_parametros": hojas, "n_arboles": n_arb, "dispositivo": "cpu", "historia": hist}

    def predecir(self, c):
        n, H = len(c), c.y.shape[1]
        return self.m.predict(tabular_arboles(c, self.lags), num_iteration=self.m.best_iteration_).reshape(n, H).astype("float32")


CLASES = {"arx_lineal": lambda: ModeloTorch("arx_lineal"), "mlp": lambda: ModeloTorch("mlp"),
          "lstm": lambda: ModeloTorch("lstm"), "cnn1d": lambda: ModeloTorch("cnn1d"), "lightgbm": LightGBM}


def entrenar_y_evaluar(nombre: str, params: dict, semilla: int, datos: Datos, op: Opciones, tr: Conjunto, es: Conjunto,
                       ev: Conjunto, guardar_pred: bool = False) -> dict:
    """Entrena con ``tr`` (parada temprana con ``es``) y evalúa en ``ev``."""
    from .metricas import evaluar

    m = CLASES[nombre]()
    t0 = time.perf_counter()
    info = m.ajustar(tr, es, datos, params, semilla, op)
    seg = time.perf_counter() - t0
    yhat = datos.invertir(m.predecir(ev), ev)
    out = {"modelo": nombre, "semilla": semilla, "segundos": round(seg, 2), "params": params,
           **{k: v for k, v in info.items() if k != "historia"}, **evaluar(ev.y, yhat, datos.escala_mase)}
    if guardar_pred:
        out["_yhat"] = yhat
    out["_historia"] = info["historia"]
    return out


def validacion_cruzada_semillas(nombre: str, params: dict, semillas: list[int], datos: Datos, op: Opciones, folds: list[dict]) -> dict:
    """Promedia la validación cruzada sobre varias semillas (reduce el ruido de la selección)."""
    rs = [validacion_cruzada(nombre, params, s, datos, op, folds) for s in semillas]
    if len(rs) == 1:
        rs[0]["MASE_sd_semillas"] = 0.0
        return rs[0]
    out = {"modelo": nombre, "semilla": ",".join(str(s) for s in semillas), "params": params, "n_parametros": rs[0]["n_parametros"],
           "segundos": round(sum(r["segundos"] for r in rs), 2)}
    for k in rs[0]:
        if k.startswith("_") or k in out or k in ("modelo", "semilla", "params"):
            continue
        out[k] = float(np.mean([r[k] for r in rs]))
    out["MASE_sd_semillas"] = float(np.std([r["MASE"] for r in rs], ddof=1))
    out["_historia"] = rs[0]["_historia"]
    return out


def validacion_cruzada(nombre: str, params: dict, semilla: int, datos: Datos, op: Opciones, folds: list[dict]) -> dict:
    """Forward chaining: entrena en cada pliegue y promedia las métricas de validación.

    Devuelve las métricas medias (criterio de selección: ``MASE``), por pliegue (``MASE_<año>``) y las curvas."""
    filas, hist = [], {}
    for f in folds:
        tr, es, va = (datos.conjunto_pos(f[k]) for k in ("tr", "es", "val"))
        r = entrenar_y_evaluar(nombre, params, semilla, datos, op, tr, es, va)
        filas.append(r)
        hist[f["anio"]] = r["_historia"]
    out = {"modelo": nombre, "semilla": semilla, "params": params,
           "segundos": round(sum(r["segundos"] for r in filas), 2), "n_parametros": filas[0]["n_parametros"]}
    for k in ("MASE", "RMSE", "MAE", "sMAPE", "MAPE"):
        out[k] = float(np.mean([r[k] for r in filas]))
    out["MASE_sd_pliegues"] = float(np.std([r["MASE"] for r in filas], ddof=1)) if len(filas) > 1 else 0.0
    for f, r in zip(folds, filas):
        out[f"MASE_{f['anio']}"] = r["MASE"]
    out["_historia"] = hist
    return out
```

## scripts/entrenar_modelos_base.py

```python
"""Entrena los tres modelos base en PyTorch (ARX lineal ridge/lasso, MLP, LSTM) con el mismo pipeline.

Usa GPU si está disponible (--dispositivo auto). LightGBM es opcional (--modelos lightgbm) y corre en CPU.

Etapas (en este orden, cada una con su subcomando):

  listar    muestra los espacios de búsqueda y las opciones
  buscar    búsqueda aleatoria, misma estrategia y mismo presupuesto (--n-config) para los tres
            modelos, 1 semilla. Cada configuración se evalúa con VALIDACIÓN CRUZADA FORWARD
            CHAINING (ventana expansiva; un pliegue por año de --cv-anios; purga = horizonte;
            parada temprana con los últimos --frac-es del entrenamiento de cada pliegue).
            Criterio de selección: MASE medio entre pliegues.
  piloto    mejor configuración de cada modelo x semillas del piloto (mismos pliegues); estima
            el número de semillas del benchmark (potencia, delta relativo)
  final     mejor configuración x n semillas: reajusta con todo lo anterior a la prueba (parada
            temprana con el último tramo) y evalúa en PRUEBA. Exige --allow-test y que exista el
            preregistro. Úsala una sola vez.

Ejemplos:
  python scripts/entrenar_modelos_base.py listar
  python scripts/entrenar_modelos_base.py buscar --n-config 20 --semillas-por-config 3 --salida results/base_v2
  python scripts/entrenar_modelos_base.py buscar --modelos lstm --fijar lstm.unidades=64 --max-epocas 40
  python scripts/entrenar_modelos_base.py buscar --dispositivo cuda --amp --determinismo rapido
  python scripts/entrenar_modelos_base.py piloto
  python scripts/entrenar_modelos_base.py final --allow-test --prereg PREREGISTRO.md

Prueba rápida de humo:
  python scripts/entrenar_modelos_base.py buscar --n-config 2 --max-origenes-train 300 --max-epocas 3 --max-arboles 50 --salida /tmp/humo
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from demanda import config as C  # noqa: E402
from demanda import modelos_base as M  # noqa: E402
from demanda.metricas import evaluar, n_semillas_requerido  # noqa: E402

SEMILLAS_PILOTO = [100, 101, 102, 103, 104]
SEMILLA_FINAL_BASE = 1000


# ------------------------------------------------------------------------ utilidades
def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("etapa", choices=["listar", "buscar", "piloto", "final"])
    g = ap.add_argument_group("datos y partición")
    g.add_argument("--covariables", choices=M.COVARIABLES, default="calendario",
                   help="serie: solo demanda | calendario: + festivos/calendario | completo: + precio, hidrología y clima ponderado")
    g.add_argument("--clima-por-ciudad", action="store_true", help="con 'completo': 24 columnas de clima en vez del clima ponderado por habitantes")
    g.add_argument("--ventana", type=int, default=C.VENTANA, help="horas de historia L (por defecto %(default)s)")
    g.add_argument("--horizonte", type=int, default=C.HORIZONTE, help="intervalos pronosticados H (por defecto %(default)s)")
    g.add_argument("--paso-h", type=int, default=24, help="separación entre orígenes en horas; divisor o múltiplo de 24 (por defecto %(default)s)")
    g.add_argument("--hora-origen", type=int, default=0, help="hora del día del origen (por defecto %(default)s)")
    g.add_argument("--normalizacion", choices=M.NORMALIZACIONES, default="nivel",
                   help="nivel: demanda dividida por su media en la ventana | global: estandarizada con entrenamiento")
    g.add_argument("--escala-mase-lag", type=int, choices=[24, 168], default=168, help="rezago del ingenuo para la escala de MASE")
    g.add_argument("--max-origenes-train", type=int, default=None, help="(sin efecto en validación cruzada) submuestreo de la partición fija de entrenamiento")
    g.add_argument("--cv-anios", type=int, nargs="+", default=C.CV_ANIOS,
                   help="años de validación del forward chaining (entre 2018 y 2021); por defecto %(default)s")
    g.add_argument("--frac-es", type=float, default=C.FRACCION_PARADA_TEMPRANA,
                   help="fracción final (cronológica) del entrenamiento reservada para la parada temprana")

    b = ap.add_argument_group("búsqueda y modelos")
    b.add_argument("--modelos", nargs="+", choices=list(M.CLASES), default=list(M.MODELOS_PYTORCH))
    b.add_argument("--n-config", type=int, default=20, help="configuraciones por modelo (mismo presupuesto para todos; 20 desde la v2 por el espacio ampliado)")
    b.add_argument("--semillas-por-config", type=int, default=3, help="semillas por configuración en la búsqueda (se promedian; 3 desde la v2, 1 en la v1)")
    b.add_argument("--semilla-busqueda", type=int, default=1, help="semilla del muestreo de configuraciones y del entrenamiento en la búsqueda")
    b.add_argument("--fijar", action="append", default=[], metavar="modelo.param=valor",
                   help="fija un hiperparámetro (sale de la búsqueda); repetible. Ej.: lstm.unidades=64")
    b.add_argument("--espacio", type=Path, default=None, help="JSON que reemplaza espacios de búsqueda: {modelo: {param: spec}}")

    t = ap.add_argument_group("entrenamiento (no se buscan)")
    t.add_argument("--max-epocas", type=int, default=200, help="modelos PyTorch: épocas máximas (la parada temprana con --paciencia decide cuándo termina)")
    t.add_argument("--paciencia", type=int, default=8, help="épocas (PyTorch, si no se busca) o rondas (LightGBM) sin mejora en validación; en la v2 la paciencia de los modelos PyTorch es hiperparámetro")
    t.add_argument("--max-arboles", type=int, default=1500, help="LightGBM: árboles máximos (parada temprana)")
    t.add_argument("--lr", type=float, default=1e-3, help="tasa de aprendizaje por defecto si no se busca (en la v2 es hiperparámetro)")
    t.add_argument("--lr-arboles", type=float, default=0.05, help="LightGBM: tasa de aprendizaje")
    t.add_argument("--perdida", choices=M.PERDIDAS, default="mse", help="pérdida por defecto si no se busca (en la v2 es hiperparámetro: mse/mae/huber)")
    t.add_argument("--huber-delta", type=float, default=None, help="delta de Huber en unidades normalizadas (por defecto 0.05 con normalización nivel, 0.5 con global)")
    t.add_argument("--min-delta", type=float, default=1e-6, help="mejora mínima de la pérdida de validación para reiniciar la paciencia")
    t.add_argument("--clip", type=float, default=1.0, help="recorte de la norma del gradiente")
    t.add_argument("--hilos", type=int, default=1, help="hilos de CPU (1 = determinismo más estricto)")
    d = ap.add_argument_group("dispositivo")
    d.add_argument("--dispositivo", default="auto", help="auto (GPU si hay) | cpu | cuda | cuda:N")
    d.add_argument("--determinismo", choices=["estricto", "rapido"], default="estricto",
                   help="estricto: torch.use_deterministic_algorithms(True) (reproducible, más lento en GPU) | rapido: sin garantías bit a bit")
    d.add_argument("--amp", action="store_true", help="precisión mixta bfloat16 en GPU (más rápido; cambia levemente los resultados)")
    d.add_argument("--max-gb-gpu", type=float, default=2.0, help="si los datos de entrenamiento superan este tamaño se mantienen en CPU y se envían por lotes")

    s = ap.add_argument_group("piloto y final")
    s.add_argument("--piloto-ultimo-pliegue", action="store_true", help="el piloto usa solo el último pliegue (más rápido)")
    s.add_argument("--semillas-piloto", type=int, nargs="+", default=SEMILLAS_PILOTO)
    s.add_argument("--delta", type=float, default=0.03, help="diferencia relativa mínima de interés para la potencia (definir en el preregistro)")
    s.add_argument("--alfa", type=float, default=0.05)
    s.add_argument("--potencia", type=float, default=0.80)
    s.add_argument("--n-semillas", type=int, default=None, help="semillas de la corrida final (por defecto, el valor del piloto; mínimo 5)")
    s.add_argument("--allow-test", action="store_true", help="autoriza leer la PRUEBA (solo etapa final)")
    s.add_argument("--prereg", type=Path, default=C.RAIZ / "PREREGISTRO.md", help="archivo de preregistro requerido para la etapa final")
    s.add_argument("--salida", type=Path, default=C.RAIZ / "results" / "base")
    return ap


def _valor(txt: str):
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        return txt


def espacios(a) -> dict:
    esp = {k: dict(v) for k, v in M.ESPACIOS.items()}
    if a.espacio:
        for mod, d in json.loads(a.espacio.read_text()).items():
            esp[mod] = d
    return esp


def fijos(a) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for f in a.fijar:
        clave, _, val = f.partition("=")
        mod, _, par = clave.partition(".")
        if mod not in M.CLASES or not par or not _:
            sys.exit(f"--fijar inválido: {f} (formato modelo.param=valor)")
        out.setdefault(mod, {})[par] = _valor(val)
    return out


def opciones(a) -> M.Opciones:
    return M.Opciones(a.max_epocas, a.paciencia, a.max_arboles, a.lr, a.min_delta, a.huber_delta, a.lr_arboles, a.perdida, a.clip,
                      a.hilos, a.dispositivo, a.determinismo, a.amp, a.max_gb_gpu)


def cfg_datos(a) -> M.ConfigDatos:
    return M.ConfigDatos(a.covariables, a.ventana, a.horizonte, a.paso_h, a.hora_origen, a.normalizacion,
                         a.clima_por_ciudad, a.max_origenes_train, a.escala_mase_lag)


def cargar_datos(a) -> M.Datos:
    panel = pd.read_parquet(C.PROCESADO / "panel_horario.parquet")
    reg = json.loads((C.PROCESADO / "feature_registry.json").read_text())
    return M.Datos(panel, reg, cfg_datos(a))


def hacer_folds(a, datos: M.Datos):
    from demanda.particion import folds_forward_chaining

    c = datos.cfg
    return folds_forward_chaining(datos.panel.index, a.cv_anios, c.horizonte, c.ventana, c.paso_h, c.hora_origen, a.frac_es)


def registrar_entorno(a, datos: M.Datos, etapa: str):
    a.salida.mkdir(parents=True, exist_ok=True)
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=C.RAIZ).stdout.strip()
    except Exception:
        commit = ""
    import torch
    try:
        import lightgbm
        lgb_v = lightgbm.__version__
    except ImportError:
        lgb_v = None
    dev = M.resolver_dispositivo(a.dispositivo)
    gpu = {"dispositivo": str(dev), "nombre": torch.cuda.get_device_name(dev) if dev.type == "cuda" else None,
           "cuda": torch.version.cuda, "cuda_disponible": torch.cuda.is_available()}
    env = {"etapa": etapa, "fecha": pd.Timestamp.now(tz=C.TZ).isoformat(), "python": sys.version, "plataforma": platform.platform(),
           "cpu": platform.processor(), "nucleos": __import__("os").cpu_count(), "hilos": a.hilos, "gpu": gpu,
           "versiones": {"numpy": np.__version__, "pandas": pd.__version__, "torch": torch.__version__,
                         "lightgbm": lgb_v},
           "commit": commit, "datos": asdict(datos.cfg), "opciones": asdict(opciones(a)), "n_config": a.n_config,
           "semilla_busqueda": a.semilla_busqueda, "semillas_por_config": a.semillas_por_config, "cv": {"tipo": "forward chaining (ventana expansiva)", "anios_validacion": a.cv_anios, "purga_h": datos.cfg.horizonte, "frac_parada_temprana": a.frac_es}, "columnas_x": datos.cols_x, "columnas_cal": datos.cols_cal,
           "escala_mase": datos.escala_mase, "determinismo": a.determinismo,
           "particion": C.PARTICION}
    (a.salida / f"entorno_{etapa}.json").write_text(json.dumps(env, indent=2, ensure_ascii=False, default=str))


def guardar_hist(a, etapa, nombre, semilla, hist):
    d = a.salida / "hist"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{etapa}_{nombre}_{semilla}.json").write_text(json.dumps(hist))


def fila(r: dict) -> dict:
    return {k: (json.dumps(v) if k == "params" else v) for k, v in r.items() if not k.startswith("_")}


# ---------------------------------------------------------------------------- etapas
def etapa_listar(a):
    esp = espacios(a)
    print(json.dumps(esp, indent=1, ensure_ascii=False))
    print("\nOpciones de entrenamiento:", asdict(opciones(a)))
    print("Fijados:", fijos(a))


def etapa_buscar(a):
    datos = cargar_datos(a)
    registrar_entorno(a, datos, "buscar")
    esp, fj, op = espacios(a), fijos(a), opciones(a)
    folds = hacer_folds(a, datos)
    print("Pliegues (forward chaining):", [(f["anio"], len(f["tr"]), len(f["es"]), len(f["val"])) for f in folds], flush=True)
    ruta = a.salida / "busqueda.csv"
    filas = pd.read_csv(ruta).to_dict("records") if ruta.exists() else []
    for nombre in a.modelos:
        rng = np.random.default_rng(a.semilla_busqueda)
        configs = [M.muestrear(esp[nombre], rng, fj.get(nombre)) for _ in range(a.n_config)]
        limites_path = a.salida / "limites_busqueda.json"
        limites = json.loads(limites_path.read_text()) if limites_path.exists() else {}
        limite = limites.get(nombre, a.n_config)
        if not isinstance(limite, int) or not 1 <= limite <= a.n_config:
            raise ValueError(f"Límite inválido para {nombre}: {limite}")
        for i, p in enumerate(configs[:limite]):
            if any(f["modelo"] == nombre and f["config"] == i and json.loads(f["params"]) == p for f in filas):
                continue
            semillas = [a.semilla_busqueda + k for k in range(a.semillas_por_config)]
            r = M.validacion_cruzada_semillas(nombre, p, semillas, datos, op, folds)
            filas.append({"config": i, **fila(r)})
            guardar_hist(a, "busqueda", f"{nombre}_c{i}", a.semilla_busqueda, {str(k): v for k, v in r["_historia"].items()})
            pd.DataFrame(filas).to_csv(ruta, index=False)
            por_fold = {k: round(v, 3) for k, v in r.items() if k.startswith("MASE_2")}
            print(f"[{nombre} {i + 1}/{a.n_config}] MASE_cv={r['MASE']:.4f} {por_fold} ({r['segundos']}s) {p}", flush=True)
    df = pd.DataFrame(filas)
    mejores = {}
    for nombre in a.modelos:
        d = df[df.modelo == nombre].sort_values("MASE")
        mejores[nombre] = {"params": json.loads(d.iloc[0]["params"]), "MASE_cv": float(d.iloc[0]["MASE"]), "n_config_evaluadas": int(len(d))}
    (a.salida / "mejor_config.json").write_text(json.dumps(mejores, indent=2))
    ref = {}
    for f in folds:
        ev = datos.conjunto_pos(f["val"])
        for k, v in datos.referencias_pos(f["val"], ev).items():
            ref.setdefault(k, []).append(evaluar(ev.y, v, datos.escala_mase)["MASE"])
    ref = {k: {"MASE_cv": float(np.mean(v)), "por_pliegue": dict(zip([f["anio"] for f in folds], map(float, v)))} for k, v in ref.items()}
    (a.salida / "referencias_cv.json").write_text(json.dumps(ref, indent=2))
    print("Mejores:", json.dumps(mejores, indent=1))
    print("Referencias (MASE medio en los pliegues):", {k: round(v["MASE_cv"], 4) for k, v in ref.items()})


def _mejores(a) -> dict:
    p = a.salida / "mejor_config.json"
    if not p.exists():
        sys.exit("Falta mejor_config.json: ejecutar primero la etapa 'buscar'")
    return json.loads(p.read_text())


def etapa_piloto(a):
    datos = cargar_datos(a)
    registrar_entorno(a, datos, "piloto")
    mej, op = _mejores(a), opciones(a)
    folds = hacer_folds(a, datos)
    if a.piloto_ultimo_pliegue:
        folds = folds[-1:]
    filas = []
    for nombre in a.modelos:
        for s in a.semillas_piloto:
            r = M.validacion_cruzada(nombre, mej[nombre]["params"], s, datos, op, folds)
            filas.append({"etapa": "piloto", **fila(r)})
            guardar_hist(a, "piloto", nombre, s, {str(k): v for k, v in r["_historia"].items()})
            print(f"[piloto {nombre} semilla={s}] MASE_cv={r['MASE']:.4f}", flush=True)
            pd.DataFrame(filas).to_csv(a.salida / "piloto.csv", index=False)
    df = pd.DataFrame(filas)
    res = {}
    for nombre in a.modelos:
        v = df[df.modelo == nombre]["MASE"].to_numpy()
        res[nombre] = {"media": float(v.mean()), "sd": float(v.std(ddof=1)), "n_piloto": int(len(v)),
                       "n_requerido": n_semillas_requerido(float(v.std(ddof=1)), float(v.mean()), a.delta, a.alfa, a.potencia),
                       "determinista": False}
    n_final = max(5, max(r["n_requerido"] for r in res.values()))
    out = {"delta_rel": a.delta, "alfa": a.alfa, "potencia": a.potencia, "n_final": n_final, "pliegues": [f["anio"] for f in folds],
           "por_modelo": res, "nota": "Estimación del piloto (normal, dos muestras); no es definitiva. Mínimo 5; 6 si se usan pruebas por rangos."}
    (a.salida / "piloto_resumen.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))


def etapa_final(a):
    from demanda.particion import pretest

    if not a.allow_test:
        sys.exit("La etapa final lee la PRUEBA: requiere --allow-test (úsala una sola vez, tras el preregistro)")
    if not a.prereg.exists():
        sys.exit(f"No existe el preregistro {a.prereg}: escribirlo y versionarlo antes de tocar la prueba")
    datos = cargar_datos(a)
    registrar_entorno(a, datos, "final")
    mej, op = _mejores(a), opciones(a)
    n = a.n_semillas
    if n is None:
        p = a.salida / "piloto_resumen.json"
        n = json.loads(p.read_text())["n_final"] if p.exists() else 5
    n = max(n, 5)
    c = datos.cfg
    pos_tr, pos_es = pretest(datos.panel.index, c.horizonte, c.ventana, c.paso_h, c.hora_origen, a.frac_es)
    tr, es, te = datos.conjunto_pos(pos_tr), datos.conjunto_pos(pos_es), datos.conjunto("test", True)
    print(f"Ajuste final: {len(tr)} orígenes de entrenamiento, {len(es)} de parada temprana, {len(te)} de prueba; n semillas = {n}", flush=True)
    filas, preds = [], {}
    for nombre in a.modelos:
        for i in range(n):
            s = SEMILLA_FINAL_BASE + i
            r = M.entrenar_y_evaluar(nombre, mej[nombre]["params"], s, datos, op, tr, es, te, guardar_pred=True)
            filas.append({"etapa": "final", **fila(r)})
            preds[f"{nombre}_{s}"] = r["_yhat"].astype("float32")
            guardar_hist(a, "final", nombre, s, r["_historia"])
            print(f"[final {nombre} semilla={s}] MASE_test={r['MASE']:.4f}", flush=True)
            pd.DataFrame(filas).to_csv(a.salida / "final_runs.csv", index=False)
    refs = {k: evaluar(te.y, v, datos.escala_mase) for k, v in datos.referencias("test", True).items()}
    (a.salida / "referencias_test.json").write_text(json.dumps(refs, indent=2))
    pdir = a.salida / "preds"
    pdir.mkdir(exist_ok=True)
    np.savez_compressed(pdir / "test.npz", y=te.y, origen=te.origen.astype("int64"), **preds)
    from scipy import stats

    df = pd.DataFrame(filas)
    resumen = []
    for nombre in a.modelos:
        d = df[df.modelo == nombre]
        v = d["MASE"].to_numpy()
        h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else float("nan")
        fila_r = {"modelo": nombre, "n": len(v), "MASE_media": v.mean(), "MASE_sd": v.std(ddof=1) if len(v) > 1 else 0.0,
                  "IC95_inf": v.mean() - h, "IC95_sup": v.mean() + h}
        for k in ("RMSE", "MAE", "sMAPE", "MAPE", "MASE_h1", "MASE_h12", "MASE_h24"):
            fila_r[k + "_media"] = d[k].mean()
        resumen.append(fila_r)
    for k, v in refs.items():
        resumen.append({"modelo": k, "n": 1, "MASE_media": v["MASE"], **{kk + "_media": v[kk] for kk in ("RMSE", "MAE", "sMAPE", "MAPE", "MASE_h1", "MASE_h12", "MASE_h24")}})
    pd.DataFrame(resumen).to_csv(a.salida / "resumen_final.csv", index=False)
    print(pd.DataFrame(resumen).to_string(index=False))


if __name__ == "__main__":
    args = parser().parse_args()
    t0 = time.time()
    {"listar": etapa_listar, "buscar": etapa_buscar, "piloto": etapa_piloto, "final": etapa_final}[args.etapa](args)
    print(f"Listo en {time.time() - t0:.0f}s")
```

## scripts/ejecutar_gpu.py

```python
"""Ejecuta búsqueda, piloto y final v2 con observación y persistencia por ajuste."""
import fcntl,hashlib,html,importlib.util,json,os,sys,time,traceback
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
spec=importlib.util.spec_from_file_location('entrenador',ROOT/'scripts/entrenar_modelos_base.py')
E=importlib.util.module_from_spec(spec);spec.loader.exec_module(E)
M=E.M
OUT=ROOT/'results/base_v2'
STATE={}
ACTIVE={}

def guardar(path,values):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(values,ensure_ascii=False,indent=2,default=str));temp.replace(path)

def publicar(**values):
    STATE.update(values);STATE['actualizado']=E.pd.Timestamp.now(tz='America/Bogota').isoformat()
    guardar(OUT/'estado_gpu.json',STATE)
    total=STATE.get('total_etapa',1);done=STATE.get('completados_etapa',0)
    current=STATE.get('ajuste',{})
    description=' · '.join(f'{k}: {v}' for k,v in current.items())
    body=f'''<!doctype html><html lang="es"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><meta http-equiv="refresh" content="5"><title>Entrenamiento de demanda eléctrica</title><style>body{{font:18px system-ui;max-width:940px;margin:48px auto;padding:24px;background:#14202c;color:#eff5fa}}progress{{width:100%;height:32px;accent-color:#60c4ed}}p{{line-height:1.65}}small{{color:#b2c5d4}}</style><h1>Pronóstico de demanda eléctrica · GPU</h1><p>{html.escape(STATE.get('estado','Preparación'))} · etapa {html.escape(STATE.get('etapa','preparación'))}</p><progress value="{done}" max="{total}"></progress><p><strong>{done} / {total} ajustes de la etapa · {100*done/max(total,1):.2f}%</strong></p><p>{html.escape(description)}</p><p>Búsqueda: 20 configuraciones de ARX, MLP y CNN1D; 5 de LSTM × 3 semillas × 4 pliegues = 780 ajustes. Búsqueda LSTM reducida por costo computacional.<br>Piloto: 5 semillas × 4 modelos × 4 pliegues = 80 ajustes.<br>Etapa final: número de semillas determinado por el piloto, mínimo 5 por modelo.</p><p>Máximo 200 épocas · parada temprana en tramo interno · criterio MASE.<br>Demanda histórica y calendario · prueba sólo en etapa final preregistrada.</p><small>Actualización: {html.escape(STATE['actualizado'])} · refresco cada 5 segundos</small></html>'''
    temp=OUT/'progreso.tmp';temp.write_text(body);temp.replace(OUT/'progreso.html')

def epoca(info):
    ACTIVE['historia'].append(dict(info))
    publicar(ajuste=dict(ACTIVE['descripcion'],epoca=info['epoca'],perdida_entrenamiento=round(info['train_loss'],8),
        perdida_parada=round(info['val_loss'],8),mejor_perdida=round(min(v['val_loss'] for v in ACTIVE['historia']),8)))
    guardar(ACTIVE['carpeta']/'historia_en_curso.json',ACTIVE['historia'])

ORIGINAL_AJUSTAR=M.ModeloTorch.ajustar
ORIGINAL_EVALUAR=M.entrenar_y_evaluar

def ajustar(self,tr,va,datos,p,semilla,op):
    result=ORIGINAL_AJUSTAR(self,tr,va,datos,p,semilla,op)
    torch.save(dict(state_dict={k:v.detach().cpu() for k,v in self.red.state_dict().items()},modelo=self.nombre,
        params=p,semilla=semilla,datos=asdict(datos.cfg),columnas_x=datos.cols_x,columnas_cal=datos.cols_cal,
        media=datos.escala.media_.to_dict(),std=datos.escala.std_.to_dict(),escala_mase=datos.escala_mase,
        informacion={k:v for k,v in result.items() if k!='historia'}),ACTIVE['carpeta']/'modelo.pt')
    return result

def evaluar(nombre,params,semilla,datos,op,tr,es,ev,guardar_pred=False):
    identity=dict(etapa=STATE['etapa'],modelo=nombre,params=params,semilla=semilla,
        inicio_evaluacion=str(ev.origen[0]),fin_evaluacion=str(ev.origen[-1]),n_train=len(tr),n_es=len(es),n_eval=len(ev))
    key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:20]
    folder=OUT/'ajustes'/STATE['etapa']/key;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'resultado.json').exists():
        result=json.loads((folder/'resultado.json').read_text())
        result['_historia']=json.loads((folder/'historia.json').read_text())
        if guardar_pred:
            with np.load(folder/'predicciones.npz') as p:result['_yhat']=p['yhat']
    else:
        ACTIVE.clear();ACTIVE.update(carpeta=folder,historia=[],descripcion=dict(modelo=nombre,semilla=semilla,
            pliegue=str(ev.origen[0])[:4],id=key))
        guardar(folder/'identidad.json',identity)
        publicar(ajuste=dict(ACTIVE['descripcion'],epoca=0))
        result=ORIGINAL_EVALUAR(nombre,params,semilla,datos,op,tr,es,ev,guardar_pred=True)
        np.savez_compressed(folder/'predicciones.npz',y=ev.y,yhat=result['_yhat'],origen=ev.origen.astype('int64'))
        guardar(folder/'historia.json',result['_historia'])
        guardar(folder/'resultado.json',{k:v for k,v in result.items() if not k.startswith('_')})
        if not guardar_pred:result.pop('_yhat',None)
    publicar(completados_etapa=STATE['completados_etapa']+1)
    return result

def main():
    OUT.mkdir(exist_ok=True,parents=True)
    lock=(OUT/'gpu.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (OUT/'ejecucion_completa.json').exists():print('La ejecución v2 ya está completa; no se repite prueba.');return
    if not torch.cuda.is_available():raise RuntimeError('La GPU es obligatoria para esta ejecución')
    source_paths=['PREREGISTRO.md','src/demanda/modelos_base.py','src/demanda/particion.py','src/demanda/ventanas.py',
        'scripts/entrenar_modelos_base.py','scripts/ejecutar_gpu.py','PREREGISTRO_CNN1D.md','data/processed/panel_horario.parquet','data/processed/feature_registry.json']
    frozen=dict(sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in source_paths},
        dispositivo='cuda',determinismo='estricto',n_config=20,semillas_por_config=3,max_epocas=200,
        modelos=list(M.MODELOS_PYTORCH),delta=.03,amp=False)
    if (OUT/'plan_gpu.json').exists() and json.loads((OUT/'plan_gpu.json').read_text())!=frozen:
        anterior=json.loads((OUT/'plan_gpu.json').read_text())
        if '--incorporar-cnn' not in sys.argv:
            raise RuntimeError('Las fuentes o el plan difieren de la ejecución existente')
        assert anterior['modelos']==['arx_lineal','mlp','lstm']
        assert frozen['modelos']==['arx_lineal','mlp','lstm','cnn1d']
        assert not (OUT/'final_completo.json').exists()
        for key,value in anterior.items():
            if key not in ['modelos','sha256']:assert frozen[key]==value,key
        cambiables={'src/demanda/modelos_base.py','scripts/ejecutar_gpu.py'}
        for key,value in anterior['sha256'].items():
            if key not in cambiables:assert frozen['sha256'][key]==value,key
        guardar(OUT/'incorporacion_cnn.json',dict(plan_anterior=anterior,plan_ampliado=frozen,
            motivo='CNN1D y espacio confirmados antes de su entrenamiento',
            fecha=E.pd.Timestamp.now(tz='America/Bogota').isoformat()))
    guardar(OUT/'plan_gpu.json',frozen)
    guardar(OUT/'hardware_gpu.json',dict(torch=torch.__version__,hip=torch.version.hip,gpu=torch.cuda.get_device_name(0),
        memoria=torch.cuda.get_device_properties(0).total_memory,compatibilidad=os.environ.get('HSA_OVERRIDE_GFX_VERSION')))
    M.PROGRESO_ENTRENAMIENTO=epoca;M.ModeloTorch.ajustar=ajustar;M.entrenar_y_evaluar=evaluar
    common=['--salida',str(OUT),'--dispositivo','cuda','--determinismo','estricto','--max-epocas','200',
        '--n-config','20','--semillas-por-config','3','--delta','0.03','--prereg',str(ROOT/'PREREGISTRO.md')]
    started=time.time()
    for stage,call in [('buscar',E.etapa_buscar),('piloto',E.etapa_piloto),('final',E.etapa_final)]:
        if (OUT/f'{stage}_completo.json').exists():continue
        total=12*sum(json.loads((OUT/'limites_busqueda.json').read_text()).get(m,20) for m in M.MODELOS_PYTORCH) if stage=='buscar' and (OUT/'limites_busqueda.json').exists() else (960 if stage=='buscar' else 80)
        done=0
        if stage=='buscar' and (OUT/'busqueda.csv').exists():done=len(E.pd.read_csv(OUT/'busqueda.csv'))*12
        if stage=='final':total=4*max(5,json.loads((OUT/'piloto_resumen.json').read_text())['n_final'])
        publicar(estado='Entrenando',etapa=stage,total_etapa=total,completados_etapa=done,pid=os.getpid(),ajuste={})
        args=E.parser().parse_args([stage,*common,*(['--allow-test'] if stage=='final' else [])])
        call(args)
        guardar(OUT/f'{stage}_completo.json',dict(ajustes=total,fecha=STATE['actualizado']))
    publicar(estado='Completado',ajuste={})
    guardar(OUT/'ejecucion_completa.json',dict(segundos=time.time()-started,final_preregistrado=True))

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        if OUT.exists():publicar(estado='Error',error=str(exc))
        traceback.print_exc();raise
```

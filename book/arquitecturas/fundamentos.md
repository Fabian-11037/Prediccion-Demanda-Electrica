(fundamentos-arquitecturas)=
# 10.1. Fundamentos y preprocesamiento compartido

## Qué recibe y qué produce una red

En el origen diario $t$, la historia $X_t$ tiene $L=168$ filas horarias y una columna por variable. El calendario futuro $F_t$ tiene $H=24$ filas. Un lote de $B$ orígenes se representa con tensores $X\in\mathbb R^{B\times L\times C}$ y $F\in\mathbb R^{B\times H\times23}$. La salida tiene forma $B\times24$: cada fila contiene las 24 predicciones emitidas simultáneamente para un día. La letra $B$ denota aquí el tamaño del lote, no el número de remuestreos estadísticos.

El nivel $\ell_t=L^{-1}\sum_{s=t-L}^{t-1}x^{dem}_s$ procede exclusivamente de la demanda histórica habilitada. La etiqueta normalizada es $r_{t,h}=y_{t,h}/\ell_t$ y la inversión devuelve $\widehat y_{t,h}=\ell_t\widehat r_{t,h}$. Las redes que predicen una corrección utilizan $\widehat r=1+g_\theta(X,F)$; el MLP y ARX producen directamente la razón normalizada. La etiqueta y el nivel cumplen funciones diferentes: el nivel puede calcularse en el origen; la etiqueta se conoce después.

Las covariables se estandarizan con estadísticas del tramo de ajuste. Una observación ausente se sustituye por una mediana de ese mismo tramo y conserva su indicador de ausencia. Esas estadísticas se mantienen al transformar parada interna, validación y prueba. RevIN, cuando se utiliza dentro de PatchTST, añade una normalización reversible por ventana y canal; no reemplaza las reglas temporales del escalado externo.

## Proyecciones, activaciones y compuertas

Una capa lineal calcula $z=Wx+b$: combina atributos y cambia su dimensión. ReLU aplica $\max(0,z)$ y permite que una composición de capas represente relaciones no lineales. La sigmoide $\sigma(z)=1/(1+e^{-z})$ produce pesos entre cero y uno; $\tanh(z)$ produce valores entre menos uno y uno. Una compuerta multiplica una representación por un peso aprendido: puede atenuarla de forma distinta para cada observación.

El dropout utiliza durante entrenamiento una máscara $m_i\sim\operatorname{Bernoulli}(1-p)$ y calcula $D_p(z_i)=m_i z_i/(1-p)$. En evaluación actúa como la identidad. La normalización de capa, LayerNorm, centra y escala los atributos de cada representación; BatchNorm utiliza estadísticas por canal del lote y conserva estadísticas para evaluación. Las implementaciones de las fichas especifican cuál se utiliza y dónde.

## Qué significan consulta, clave y valor

La atención compara una consulta con las claves de los elementos disponibles y mezcla sus valores. La consulta expresa qué información busca una representación; la clave permite medir afinidad; el valor contiene la información que se transmite. Son proyecciones aprendidas:

$$Q=ZW_Q,\qquad K=MW_K,\qquad V=MW_V,\qquad
A=\operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}+\mathcal M\right),\qquad O=AV.$$

El softmax se aplica por consulta, $\alpha_j=\exp(s_j)/\sum_k\exp(s_k)$, por lo que los pesos de los elementos permitidos suman uno. La máscara $\mathcal M$ vale cero donde se permite consultar y menos infinito donde se prohíbe. En atención propia, $Z=M$; en atención cruzada, las consultas y la memoria proceden de representaciones distintas.

La atención multicabeza repite el cálculo con proyecciones distintas. Habitualmente concatena las salidas y aplica otra proyección: $\operatorname{MultiHead}=\operatorname{Concat}(O_1,\ldots,O_a)W_O$. TFT utiliza la variante interpretable descrita en su ficha, con valores compartidos y promedio entre cabezas. Un bloque Transformer añade conexiones residuales, normalización y una red por posición, $\operatorname{FFN}(z)=W_2\phi(W_1z+b_1)+b_2$. La posición y el tipo de token determinan qué relación modela: entre parches en PatchTST, entre variables en iTransformer y entre consultas del horizonte y memoria en TGForecaster.

Los pesos de atención son componentes del cálculo, no estimaciones de efectos causales. Tampoco identifican por sí solos la importancia global de una variable.

## Pérdida, optimización y parada

Para un lote, el MAE normalizado es $\mathcal L=(BH)^{-1}\sum_{b,h}|r_{b,h}-\widehat r_{b,h}|$. Algunas familias base utilizan Huber con transición $\eta=0,05$:

$$\ell_\eta(e)=\begin{cases}\frac12e^2,& |e|\le\eta,\\
\eta\left(|e|-\frac12\eta\right),& |e|>\eta.
\end{cases}$$

AdamW actualiza los pesos utilizando promedios de gradientes y de sus cuadrados, con decaimiento de pesos separado. El recorte limita la norma global del gradiente a uno. La reducción de tasa en meseta y la paciencia de parada son reglas distintas: la primera modifica la tasa de aprendizaje y la segunda termina el ajuste. Las fichas conservan la paciencia efectiva del motor correspondiente. El estado restaurado es el mínimo absoluto observado en la reserva de parada interna, aunque el contador de paciencia exija una mejora mínima para reiniciarse.

El MASE usado para comparar se calcula después de invertir a kWh y divide el MAE por la escala fija q168. No se confunden pérdida de optimización normalizada y métrica de evaluación. La dispersión entre cinco semillas caracteriza inicialización; los intervalos por bloques caracterizan variación temporal condicional a esas semillas.

## Cómo interpretar el coste

Una proyección densa requiere aproximadamente $O(d_{in}d_{out})$ operaciones por observación. Una LSTM de ancho $d$ sobre $L$ horas requiere $O(Ld(C+d))$; una convolución requiere $O(LkC_{in}C_{out})$. La atención propia sobre $T$ tokens incluye un término $O(T^2d)$ y la atención cruzada, $O(T_qT_md)$. Estos términos explican por qué cambia el coste al usar horas, parches o variables como tokens. No sustituyen tiempos medidos ni incluyen toda la memoria del ajuste, redes internas, preprocesamiento o codificación de texto.

Las fichas contienen el código de las redes y los pipelines completos. Los programas de entrenamiento se presentan como fuentes reproducibles; la construcción documental ejecuta el análisis de artefactos conservados. La compatibilidad del adaptador portable se mantiene separada de los orquestadores originales dependientes de un sistema operativo.


## Código compartido de preparación

Las fichas muestran cada red y su ajuste. Los siguientes bloques reúnen las operaciones reutilizadas de partición, construcción de ventanas, escalado y evaluación.

## Partición temporal y parada interna

Fuente: `src/demanda/particion.py`. Se muestran las clases y funciones que intervienen en el pipeline.

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

## Datos, ventanas, escalas y referencias

Fuente: `src/demanda/modelos_base.py`. Se muestran las clases y funciones que intervienen en el pipeline.

```python
from __future__ import annotations

import os

import random

import time

from dataclasses import dataclass, field

import numpy as np

import pandas as pd

from . import config as C  # noqa: E402

from .particion import AjustadorEscala, escala_mase, origenes

from .ventanas import columnas_por_clase, construir_ventanas  # noqa: E402

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
```

## Métricas

Fuente: `src/demanda/metricas.py`. Se muestran las clases y funciones que intervienen en el pipeline.

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

## Opciones de ejecución compartidas

Las clases específicas están en sus fichas; este bloque conserva la selección de dispositivo, la fijación de semillas y las opciones reutilizadas por los pipelines.

```python
from __future__ import annotations

import os

import random

import time

from dataclasses import dataclass, field

import numpy as np

import pandas as pd

from . import config as C

from .particion import AjustadorEscala, escala_mase, origenes

from .ventanas import columnas_por_clase, construir_ventanas

@dataclass
class Opciones:
    """Opciones de entrenamiento que NO se buscan (iguales para todos los modelos que las usan)."""
    max_epocas: int = 200
    paciencia: int = 8
    max_arboles: int = 1500
    lr: float = 0.001
    min_delta: float = 1e-06
    huber_delta: float | None = None
    lr_arboles: float = 0.05
    perdida: str = 'mse'
    clip: float = 1.0
    hilos: int = 1
    dispositivo: str = 'auto'
    determinismo: str = 'estricto'
    amp: bool = False
    max_gb_gpu: float = 2.0

def resolver_dispositivo(txt: str):
    import torch
    if txt == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    d = torch.device(txt)
    if d.type == 'cuda' and (not torch.cuda.is_available()):
        raise RuntimeError('Se pidió CUDA pero torch.cuda.is_available() es False: revisar controlador y versión de PyTorch con CUDA')
    return d

def fijar_semillas(semilla: int, op: Opciones):
    import torch
    random.seed(semilla)
    np.random.seed(semilla)
    torch.manual_seed(semilla)
    torch.cuda.manual_seed_all(semilla)
    torch.set_num_threads(op.hilos)
    if op.determinismo == 'estricto':
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
```

(exportacion-esquemas)=
## Exportación de los esquemas

Los diagramas globales y de detalle se conservan como imágenes de alta resolución; al pulsarlos se abre el archivo completo. Las dimensiones y las operaciones corresponden a las configuraciones evaluadas. Los diagramas de PatchTST, TCN, iTransformer, TaTS, fusión tardía y TGForecaster se construyen con objetos editables, siguiendo el formato gráfico de la presentación.

El siguiente código genera sus bloques, conexiones, ecuaciones y archivos editables a partir de las dimensiones documentadas.

```{literalinclude} ../_static/codigo_e2/crear_esquemas_ampliados.mjs
:language: javascript
:linenos:
```

La extracción utiliza PowerPoint en Windows. La construcción del JBook usa las imágenes incluidas y no necesita PowerPoint ni ejecutar modelos.

```{literalinclude} ../_static/codigo_e2/exportar_esquemas_presentacion.ps1
:language: powershell
:linenos:
```

# 10.10. Pipeline de entrenamiento y evaluación

El pipeline utiliza historia anterior al origen, calendario conocido del horizonte, escalado por pliegue y parada interna. El código corresponde a los modelos identificados en el registro consolidado. Los bloques documentan el procedimiento y no se ejecutan al construir el libro.

Las rutas se expresan con pathlib. El dispositivo se selecciona entre CPU y GPU compatible con PyTorch; el entorno y las dependencias se registran junto con los resultados. Las etiquetas de prueba requieren acceso explícito en una evaluación final.

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

## Modelos densos y opciones de ejecución

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

class ModeloTorch:
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
```

## Preprocesamiento por pliegue y bucle de optimización

Fuente: `src/demanda/reentrenamiento_v3.py`. Se muestran las clases y funciones que intervienen en el pipeline.

```python
from dataclasses import asdict

import numpy as np

import pandas as pd

import torch

from torch import nn

from . import modelos_base as M

class DatosFold(M.Datos):
    def __init__(self, panel, registro, cfg, posiciones_train, secuencial=False):
        super().__init__(panel, registro, cfg)
        pos=np.asarray(posiciones_train)
        inicio=panel.index[pos.min()-cfg.ventana]
        fin=panel.index[pos.max()-1]
        filas=panel.loc[(panel.index>=inicio)&(panel.index<=fin)]
        cols=self.cols_x+self.cols_cal
        self.escala.media_=filas[cols].mean()
        self.escala.std_=filas[cols].std().replace(0,1.)
        # Estadísticas del objetivo exclusivamente en etiquetas de entrenamiento.
        target='y_demanda_real_kwh'
        etiquetas=panel.loc[(panel.index>=panel.index[pos.min()])&(panel.index<=panel.index[pos.max()]+pd.Timedelta(hours=cfg.horizonte-1)),target]
        self.escala.media_[target]=etiquetas.mean();self.escala.std_[target]=etiquetas.std()
        self.secuencial=False
        train=super().conjunto_pos(pos)
        self.sigma_demanda=max(float(train.x[:,:,0].std()),1e-6)
        self.secuencial=secuencial
        self.auditoria=dict(inicio_ajuste=str(inicio),fin_ajuste=str(fin),filas=len(filas),sigma_demanda=self.sigma_demanda,secuencial=secuencial,
            descripcion_escala_mase='Fija 2000–2017, como denominador de referencia; no participa en optimización')
    def conjunto_pos(self,pos,clave=None):
        c=super().conjunto_pos(pos,clave)
        if self.secuencial:
            c.x[:,:,0]=(c.x[:,:,0]-1.)/self.sigma_demanda
        return c
    def escaladores(self):
        return dict(media=self.escala.media_.to_dict(),std=self.escala.std_.to_dict(),columnas_x=self.cols_x,columnas_cal=self.cols_cal,
            escala_mase=self.escala_mase,datos=asdict(self.cfg),**self.auditoria)

class Secuencial(nn.Module):
    def __init__(self,nombre,p,tr):
        super().__init__();self.nombre=nombre
        f=tr.x.shape[2]+tr.cal_p.shape[2];L=tr.x.shape[1];H,k=tr.cal_f.shape[1:]
        if nombre=='lstm':
            w=int(p['unidades']);self.encoder=nn.LSTM(f,w,num_layers=1,batch_first=True);d=w
        else:
            w=int(p['filtros']);kernel=int(p['kernel'])
            self.encoder=nn.Sequential(nn.Conv1d(f,w,kernel,padding=kernel//2),nn.ReLU(),nn.Dropout1d(p['dropout']),nn.AvgPool1d(2),nn.Flatten());d=w*(L//2)
        self.drop=nn.Dropout(p['dropout']);self.salida=nn.Linear(d+H*k,H)
        # Predicción inicial próxima al nivel; aprende desviaciones horarias.
        nn.init.zeros_(self.salida.bias)
    def forward(self,x,c):
        if self.nombre=='lstm':_,(h,_)=self.encoder(x);z=h[-1]
        else:z=self.encoder(x.transpose(1,2))
        return 1.+self.salida(torch.cat([self.drop(z),c.flatten(1)],1))
    def preparar(self,x,c):pass
    def penalizacion(self):return 0.

class Entrenador(M.ModeloTorch):
    def ajustar(self,tr,va,datos,p,semilla,op):
        M.fijar_semillas(semilla,op);self.dev=dev=M.resolver_dispositivo(op.dispositivo)
        self.red=red=(Secuencial(self.nombre,p,tr) if self.nombre in ('lstm','cnn1d') else M.construir_red(self.nombre,p,tr)).to(dev)
        xt,ct,yt=self._tensores(tr,datos.objetivo_norm(tr),dev,True)
        xv,cv,yv=self._tensores(va,datos.objetivo_norm(va),dev,True)
        red.preparar(xt,ct)
        lossf={'mse':nn.MSELoss(),'mae':nn.L1Loss(),'huber':nn.HuberLoss(delta=.05)}[p['perdida']]
        opt=torch.optim.AdamW(red.parameters(),lr=p['lr'],weight_decay=p.get('weight_decay',0.))
        sched=(torch.optim.lr_scheduler.ReduceLROnPlateau(opt,factor=.5,patience=2) if p['scheduler']=='plateau' else
            torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=op.max_epocas) if p['scheduler']=='cosine' else None)
        best=reference=float('inf');state=None;stale=0;hist=[];gen=torch.Generator().manual_seed(semilla)
        bs=int(p['batch_size'])
        for ep in range(1,op.max_epocas+1):
            red.train();tot=0.
            for ix in torch.randperm(len(xt),generator=gen).split(bs):
                ix=ix.to(dev);opt.zero_grad(set_to_none=True)
                loss=lossf(red(xt[ix],ct[ix]),yt[ix]);objective=loss+red.penalizacion()
                if not torch.isfinite(objective):raise FloatingPointError('Pérdida no finita')
                objective.backward();nn.utils.clip_grad_norm_(red.parameters(),op.clip);opt.step();tot+=float(loss.detach())*len(ix)
            red.eval()
            with torch.no_grad():
                vl=sum(float(lossf(red(xv[i:i+512],cv[i:i+512]),yv[i:i+512]))*len(xv[i:i+512]) for i in range(0,len(xv),512))/len(xv)
            hist.append(dict(epoca=ep,train_loss=tot/len(xt),val_loss=vl,lr=opt.param_groups[0]['lr']))
            if vl<best:
                best=vl;state={k:v.detach().clone() for k,v in red.state_dict().items()}
            if not np.isfinite(reference) or vl<reference-max(1e-6,.002*abs(reference)):
                reference=vl;stale=0
            else:stale+=1
            if sched is not None:sched.step(vl) if p['scheduler']=='plateau' else sched.step()
            callback=getattr(self,'observador',None)
            if callback:callback(hist[-1])
            if ep>=10 and stale>=6:break
        red.load_state_dict(state)
        return dict(n_parametros=sum(v.numel() for v in red.parameters()),epocas=len(hist),mejor_epoca=int(np.argmin([v['val_loss'] for v in hist])+1),dispositivo=str(dev),historia=hist)
```

## Codificador recurrente seleccionado

Fuente: `src/demanda/variantes_v4.py`. Se muestran las clases y funciones que intervienen en el pipeline.

```python
import torch

from torch import nn

class Secuencial(nn.Module):
    def __init__(self,nombre,p,tr):
        super().__init__();self.nombre=nombre
        f=tr.x.shape[2]+tr.cal_p.shape[2];L=tr.x.shape[1];H,k=tr.cal_f.shape[1:]
        if nombre=='lstm':
            w=int(p['unidades']);n=int(p['capas'])
            self.encoder=nn.LSTM(f,w,num_layers=n,batch_first=True,dropout=p['dropout'] if n>1 else 0.);d=w
        else:
            w=int(p['filtros']);kernel=int(p['kernel']);blocks=[]
            for dilation in p['dilataciones']:
                blocks.extend([nn.Conv1d(f,w,kernel,dilation=dilation,padding=dilation*(kernel-1)//2),nn.ReLU(),nn.Dropout1d(p['dropout'])]);f=w
            blocks.extend([nn.AvgPool1d(2),nn.Flatten()]);self.encoder=nn.Sequential(*blocks);d=w*(L//2)
        self.drop=nn.Dropout(p['dropout']);self.salida=nn.Linear(d+H*k,H);nn.init.zeros_(self.salida.bias)
    def forward(self,x,c):
        if self.nombre=='lstm':_,(h,_)=self.encoder(x);z=h[-1]
        else:z=self.encoder(x.transpose(1,2))
        return 1.+self.salida(torch.cat([self.drop(z),c.flatten(1)],1))
    def preparar(self,x,c):pass
    def penalizacion(self):return 0.
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

## Pipeline integrado

El siguiente bloque vincula las implementaciones mostradas y permite seleccionar CPU o GPU sin funciones exclusivas del sistema operativo. La inicialización del dispositivo se conserva en resolver_dispositivo.

```python
from pathlib import Path
import json,sys
import numpy as np
import pandas as pd

def ejecutar_configuracion(raiz, familia, semilla=1000, dispositivo='auto',
                          fase='validacion', anio=2021, permitir_prueba=False,
                          salida=None):
    raiz=Path(raiz).resolve()
    procedencia=json.loads((raiz/'results/jbook/modelos_actualizados.json').read_text())
    fuentes=raiz/'results/repositorio'/procedencia['commit']
    sys.path.insert(0,str(fuentes/'src'))
    from demanda import modelos_base as M, particion as P
    from demanda import reentrenamiento_v3 as motor
    from demanda.variantes_v4 import Secuencial as Recurrente
    from demanda.metricas import evaluar
    import torch
    registro=json.loads((fuentes/'results/consolidado/modelos.json').read_text())
    config=registro[familia]['configuracion']
    panel=pd.read_parquet(raiz/'data/processed/panel_horario.parquet')
    variables=json.loads((fuentes/'data/processed/feature_registry.json').read_text())
    cfg=M.ConfigDatos(ventana=config['ventana'])
    if fase=='prueba':
        if not permitir_prueba:raise PermissionError('La prueba requiere acceso explícito')
        tr,es=P.pretest(panel.index,ventana=cfg.ventana)
        ev=P.origenes(panel.index,'test',ventana=cfg.ventana,allow_test=True)
    else:
        fold=P.folds_forward_chaining(panel.index,[anio],ventana=cfg.ventana)[0]
        tr,es,ev=fold['tr'],fold['es'],fold['val']
    datos=motor.DatosFold(panel,variables,cfg,tr,secuencial=familia in ('lstm','cnn1d'))
    train,stop,evaluacion=[datos.conjunto_pos(p) for p in [tr,es,ev]]
    assert train.origen.max()+pd.Timedelta(hours=23)<stop.origen.min()
    assert stop.origen.max()+pd.Timedelta(hours=23)<evaluacion.origen.min()
    assert pd.Timestamp(datos.auditoria['fin_ajuste'])<stop.origen.min()
    fabrica=motor.Secuencial
    try:
        if familia=='lstm':motor.Secuencial=Recurrente
        modelo=motor.Entrenador(familia)
        info=modelo.ajustar(train,stop,datos,config['params'],semilla,
                           M.Opciones(max_epocas=60,dispositivo=dispositivo,hilos=2))
    finally:
        motor.Secuencial=fabrica
    pred=datos.invertir(modelo.predecir(evaluacion),evaluacion)
    metricas=evaluar(evaluacion.y,pred,datos.escala_mase)
    if salida is not None:
        destino=Path(salida)/familia/str(semilla)/str(anio if fase=='validacion' else fase)
        destino.mkdir(parents=True,exist_ok=False)
        escala=datos.escaladores()
        torch.save({'state_dict':{k:v.detach().cpu() for k,v in modelo.red.state_dict().items()},
                    'modelo':familia,'params':config['params'],'escaladores':escala},
                    destino/'modelo.pt')
        np.savez_compressed(destino/'predicciones.npz',y=evaluacion.y,yhat=pred,
                            origen=evaluacion.origen.as_unit('ns').asi8)
        for nombre,objeto in [('escaladores',escala),('historia',info['historia']),
                              ('metricas',metricas)]:
            (destino/(nombre+'.json')).write_text(json.dumps(objeto,ensure_ascii=False,
                                                indent=2,default=str),encoding='utf-8')
    return modelo,datos,evaluacion,pred,info,metricas

# Selección: comparar el MASE medio de confirmación, sin leer prueba.
# Piloto: repetir cada configuración seleccionada con semillas 100–104
# en los cuatro años externos. Final: semillas 1000–1004 con acceso explícito.
# El argumento salida conserva pesos, escaladores, curvas, métricas y predicciones.

```

## Reconstrucción del benchmark

Las celdas de los capítulos de modelos y evaluación cargan exclusivamente predicciones, historiales y metadatos guardados. Verifican igualdad de etiquetas y orígenes, reconstruyen métricas y calculan los diagnósticos. El registro identifica el commit d1ae9a4e09dd1dead4af1329f16aa3e8d0f2c3d3.

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

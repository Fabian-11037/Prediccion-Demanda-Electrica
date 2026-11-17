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

COVARIABLES = ("serie", "calendario", "completo")

NORMALIZACIONES = ("nivel", "global")

PERDIDAS = ("mse", "mae", "huber")

MODELOS_PYTORCH = ("arx_lineal", "mlp", "lstm", "cnn1d")

CLASES = {"arx_lineal": lambda: ModeloTorch("arx_lineal"), "mlp": lambda: ModeloTorch("mlp"),
          "lstm": lambda: ModeloTorch("lstm"), "cnn1d": lambda: ModeloTorch("cnn1d"), "lightgbm": LightGBM}

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

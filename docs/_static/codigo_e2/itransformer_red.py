from __future__ import annotations

import math

import torch

from torch import nn

D_REL = 64                 # dimensión de la relevancia aprendida (fija, fuera del presupuesto)

PARCHE, PASO = 16, 8       # parches de la historia

PARCHE_H, PASO_H = 8, 4    # parches del horizonte en TGForecaster

D_CTX = 128                # ancho de la vía de contexto de PatchTST (fijo)

DILATACIONES_TCN = (1, 4, 16)

class NucleoITransformer(nn.Module):
    def __init__(self, n_hist: int, n_cal: int, n_txt: int, L: int, H: int, p: dict):
        super().__init__()
        d, capas, cabezas, dropout = int(p["d_model"]), int(p["capas"]), int(p["cabezas"]), float(p["dropout"])
        self.emb_hist, self.emb_cal = nn.Linear(L, d), nn.Linear(L + H, d)
        self.emb_txt = nn.Linear(L, d) if n_txt else None
        self.identidad = nn.Parameter(torch.zeros(n_hist + n_cal + n_txt, d))
        nn.init.normal_(self.identidad, std=0.02)
        capa = nn.TransformerEncoderLayer(d, cabezas, 2 * d, dropout, activation="gelu", batch_first=True, norm_first=False)
        self.encoder = nn.TransformerEncoder(capa, capas, norm=nn.LayerNorm(d), enable_nested_tensor=False)
        self.drop = nn.Dropout(dropout)
        self.salida = nn.Linear(d, H)
        nn.init.zeros_(self.salida.bias)

    def forward(self, x, cal_p, cal_f, txt=None):
        fichas = [self.emb_hist(x.transpose(1, 2)), self.emb_cal(torch.cat([cal_p, cal_f], dim=1).transpose(1, 2))]
        if self.emb_txt is not None:
            fichas.append(self.emb_txt(txt.transpose(1, 2)))
        h = self.encoder(self.drop(torch.cat(fichas, dim=1) + self.identidad))
        return 1.0 + self.salida(h[:, 0])           # ficha 0 = demanda

class NumericoITransformer(nn.Module):
    """iTransformer numérico (Liu et al., ICLR 2024): un token por variable y atención entre variables."""

    def __init__(self, p: dict, n_x: int, n_cal: int, L: int, H: int):
        super().__init__()
        self.nucleo = NucleoITransformer(n_x, n_cal, 0, L, H, p)

    def forward(self, x, cal_p, cal_f, dia):
        return self.nucleo(x, cal_p, cal_f, None)

    def diagnostico(self) -> dict:
        return {}

    def n_parametros(self) -> int:
        return sum(v.numel() for v in self.parameters() if v.requires_grad)

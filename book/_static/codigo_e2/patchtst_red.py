from __future__ import annotations

import math

import torch

from torch import nn

D_REL = 64      # dimensión de la relevancia aprendida (fija, fuera del presupuesto)

PARCHE, PASO = 16, 8

D_CTX = 128     # ancho de la vía de contexto de PatchTST (fijo, fuera del presupuesto)

class _CapaPatch(nn.Module):
    """Capa de codificador con normalización por lotes, como en PatchTST."""

    def __init__(self, d, cabezas, dropout):
        super().__init__()
        self.atencion = nn.MultiheadAttention(d, cabezas, dropout=dropout, batch_first=True)
        self.n1, self.n2 = nn.BatchNorm1d(d), nn.BatchNorm1d(d)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * d, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.drop(self.atencion(x, x, x, need_weights=False)[0])
        x = self.n1(x.transpose(1, 2)).transpose(1, 2)
        x = x + self.drop(self.ff(x))
        return self.n2(x.transpose(1, 2)).transpose(1, 2)

class NucleoPatchTST(nn.Module):
    """Codificador independiente por canal (históricas, indicadores de ausencia y texto) y cabeza de dos vías.

    Vía lineal: la cabeza aplanada del artículo sobre los parches de la demanda.
    Vía de contexto: un MLP sobre el resumen de la demanda, el resumen de cada uno de los demás canales,
    el calendario futuro y el promedio diario del calendario de la historia. Sin esta vía las covariables
    y el texto no llegarían a la demanda, y el efecto del calendario quedaría limitado a una función lineal.
    El calendario es determinista y no pasa por el codificador.
    """

    def __init__(self, n_hist: int, n_cal: int, n_txt: int, L: int, H: int, p: dict):
        super().__init__()
        d, capas, cabezas, dropout = int(p["d_model"]), int(p["capas"]), int(p["cabezas"]), float(p["dropout"])
        self.n_parches = (L + PASO - PARCHE) // PASO + 1          # 21 con L = 168: se rellena el final con PASO copias
        self.canales = n_hist + n_txt
        self.dias = L // 24
        self.emb = nn.Linear(PARCHE, d)
        self.pos = nn.Parameter(torch.zeros(self.n_parches, d))
        nn.init.normal_(self.pos, std=0.02)
        self.capas = nn.ModuleList([_CapaPatch(d, cabezas, dropout) for _ in range(capas)])
        self.drop = nn.Dropout(dropout)
        self.lineal = nn.Linear(self.n_parches * d, H)
        self.contexto = nn.Sequential(nn.Linear(self.canales * d + H * n_cal + self.dias * n_cal, D_CTX), nn.GELU(), nn.Dropout(dropout), nn.Linear(D_CTX, H))
        nn.init.zeros_(self.lineal.bias)
        nn.init.zeros_(self.contexto[-1].bias)

    def forward(self, x, cal_p, cal_f, txt=None):
        s = torch.cat([x, txt], dim=2) if txt is not None else x                    # (B, L, C)
        B, L, Cn = s.shape
        s = s.transpose(1, 2).reshape(B * Cn, L)
        s = torch.cat([s, s[:, -1:].expand(-1, PASO)], dim=1)                       # relleno con el último valor
        z = self.drop(self.emb(s.unfold(1, PARCHE, PASO)) + self.pos)               # (B*C, 21, d)
        for capa in self.capas:
            z = capa(z)
        z = z.reshape(B, Cn, self.n_parches, -1)
        cal_dia = cal_p.reshape(B, self.dias, L // self.dias, -1).mean(2)           # (B, 7, n_cal)
        ctx = torch.cat([z.mean(2).flatten(1), cal_f.flatten(1), cal_dia.flatten(1)], dim=1)   # canal 0 = demanda
        return 1.0 + self.lineal(z[:, 0].flatten(1)) + self.contexto(ctx)

class Modelo6(nn.Module):
    """Variante N (sin texto) o N+O (con canal oficial) sobre el núcleo elegido."""

    def __init__(self, p: dict, n_hist: int, n_cal: int, L: int, H: int, texto: dict | None = None):
        super().__init__()
        self.agregador = None
        n_txt = 0
        if texto is not None:
            self.agregador = AgregadorTexto(texto["vectores"], texto["lista"], n_hist, int(p["d_texto"]))
            n_txt = int(p["d_texto"]) + 1
        clase = {"itransformer": NucleoITransformer, "patchtst": NucleoPatchTST}[p["nucleo"]]
        self.nucleo = clase(n_hist, n_cal, n_txt, L, H, p)

    def forward(self, x, cal_p, cal_f, dia):
        txt = self.agregador(x, dia)[0] if self.agregador is not None else None
        return self.nucleo(x, cal_p, cal_f, txt)

    def n_parametros(self) -> int:
        return sum(v.numel() for v in self.parameters() if v.requires_grad)

import torch
from torch import nn


import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "top10_adicionales/src/vendor/timesnet"))
import TimesNet as _TN
from layers.Embed import TokenEmbedding, PositionalEmbedding


class _Base(nn.Module):
    def diagnostico(self) -> dict:
        return {}

    def n_parametros(self) -> int:
        return sum(v.numel() for v in self.parameters() if v.requires_grad)

class TimesNetE2(_Base):
    """TimesNet (Wu et al., ICLR 2023) con los TimesBlock e Inception_Block_V1 oficiales.

    Entrada: las variables históricas y los indicadores de ausencia (C canales), normalización no estacionaria por
    instancia como en el código oficial, embedding de valor (convolución de token), posicional y del calendario de
    la historia (lineal, en lugar de las marcas de tiempo de 4 rasgos), extensión lineal L -> L + H, ``e_layers``
    TimesBlock y proyección a C canales. Adaptación declarada: la demanda pronosticada de cada hora se combina con
    el calendario conocido de esa hora mediante una lectura lineal compartida (1 + n_cal -> 1)."""

    def __init__(self, p: dict, n_x: int, n_cal: int, L: int, H: int):
        super().__init__()
        from types import SimpleNamespace
        cfg = SimpleNamespace(seq_len=L, pred_len=H, top_k=int(p["top_k"]), d_model=int(p["d_model"]), d_ff=int(p["d_ff"]),
                              num_kernels=int(p["num_kernels"]))
        self.L, self.H = L, H
        self.valor = TokenEmbedding(n_x, cfg.d_model)
        self.posicion = PositionalEmbedding(cfg.d_model)
        self.marca = nn.Linear(n_cal, cfg.d_model, bias=False)
        self.drop = nn.Dropout(float(p["dropout"]))
        self.predict_linear = nn.Linear(L, L + H)
        self.bloques = nn.ModuleList([_TN.TimesBlock(cfg) for _ in range(int(p["e_layers"]))])
        self.norma = nn.LayerNorm(cfg.d_model)
        self.proyeccion = nn.Linear(cfg.d_model, n_x)
        self.lectura = nn.Linear(1 + n_cal, 1)
        with torch.no_grad():
            self.lectura.weight.zero_()
            self.lectura.weight[0, 0] = 0.1
            self.lectura.bias.zero_()

    def forward(self, x, cal_p, cal_f, dia):
        media = x.mean(1, keepdim=True).detach()
        z = x - media
        desv = torch.sqrt(torch.var(z, dim=1, keepdim=True, unbiased=False) + 1e-5)
        z = z / desv
        e = self.drop(self.valor(z) + self.posicion(z) + self.marca(cal_p))
        e = self.predict_linear(e.permute(0, 2, 1)).permute(0, 2, 1)
        for bloque in self.bloques:
            e = self.norma(bloque(e))
        out = self.proyeccion(e) * desv[:, 0, :].unsqueeze(1) + media[:, 0, :].unsqueeze(1)
        demanda = out[:, -self.H:, :1]                                         # (B, H, 1): canal 0 = demanda
        return 1.0 + self.lectura(torch.cat([demanda, cal_f], dim=2)).squeeze(-1)
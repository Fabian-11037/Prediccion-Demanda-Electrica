from __future__ import annotations

import math

import torch

from torch import nn

D_REL = 64                 # dimensión de la relevancia aprendida (fija, fuera del presupuesto)

PARCHE, PASO = 16, 8       # parches de la historia

PARCHE_H, PASO_H = 8, 4    # parches del horizonte en TGForecaster

D_CTX = 128                # ancho de la vía de contexto de PatchTST (fijo)

DILATACIONES_TCN = (1, 4, 16)

def campo_receptivo_tcn(kernel: int) -> int:
    """Dos convoluciones causales por bloque: R = 1 + 2 (k - 1) (1 + 4 + 16)."""
    return 1 + 2 * (int(kernel) - 1) * sum(DILATACIONES_TCN)

class _BloqueTCN(nn.Module):
    """Bloque residual de Bai et al. (2018): dos convoluciones causales dilatadas con normalización de pesos."""

    def __init__(self, c_in: int, c_out: int, kernel: int, dilatacion: int, dropout: float):
        super().__init__()
        self.recorte = (kernel - 1) * dilatacion
        convs = [nn.Conv1d(c_in, c_out, kernel, padding=self.recorte, dilation=dilatacion), nn.Conv1d(c_out, c_out, kernel, padding=self.recorte, dilation=dilatacion)]
        for c in convs:
            nn.init.normal_(c.weight, 0.0, 0.01)
        self.c1, self.c2 = (nn.utils.parametrizations.weight_norm(c) for c in convs)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()       # al final del bloque
        self.atajo = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else None

    def forward(self, x):
        y = torch.relu(self.c1(x)[..., :-self.recorte])       # el recorte quita el relleno derecho: no se mira adelante
        y = self.drop(torch.relu(self.c2(y)[..., :-self.recorte]))
        return torch.relu(y + (x if self.atajo is None else self.atajo(x)))

class TCN(nn.Module):
    """TCN de tres bloques con dilataciones (1, 4, 16) y cabeza con el calendario futuro conocido.

    Entrada de las convoluciones: variables históricas, indicadores de ausencia y calendario de la historia.
    Se toma la representación de la última hora de la historia, se une con una proyección del calendario
    futuro y una cabeza densa produce las 24 horas (misma convención que las redes base del proyecto).
    """

    def __init__(self, p: dict, n_x: int, n_cal: int, L: int, H: int):
        super().__init__()
        anchos, patron, k, tasa = [int(a) for a in p["anchos"]], [int(b) for b in p["patron_dropout"]], int(p["kernel"]), float(p["dropout"])
        if len(anchos) != 3 or len(patron) != 3:
            raise ValueError("TCN usa tres bloques")
        if campo_receptivo_tcn(k) < L:
            raise ValueError(f"El campo receptivo ({campo_receptivo_tcn(k)}) no cubre la historia ({L})")
        canales = [n_x + n_cal] + anchos
        self.bloques = nn.ModuleList([_BloqueTCN(canales[i], canales[i + 1], k, DILATACIONES_TCN[i], tasa if patron[i] else 0.0) for i in range(3)])
        w = anchos[-1]
        self.cal = nn.Linear(H * n_cal, w)
        self.cabeza = nn.Sequential(nn.Linear(2 * w, w), nn.ReLU(), nn.Linear(w, H))
        nn.init.zeros_(self.cabeza[-1].bias)

    def forward(self, x, cal_p, cal_f, dia):
        s = torch.cat([x, cal_p], dim=2).transpose(1, 2)        # (B, canales, L)
        for bloque in self.bloques:
            s = bloque(s)
        z = torch.cat([s[:, :, -1], torch.relu(self.cal(cal_f.flatten(1)))], dim=1)
        return 1.0 + self.cabeza(z)

    def diagnostico(self) -> dict:
        return {}

    def n_parametros(self) -> int:
        return sum(v.numel() for v in self.parameters() if v.requires_grad)

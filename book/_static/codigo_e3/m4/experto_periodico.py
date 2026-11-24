from __future__ import annotations
import torch
from torch import nn

def _n_parametros(m: nn.Module) -> int:
    return sum(v.numel() for v in m.parameters() if v.requires_grad)

class Periodica7x24(nn.Module):
    """Idea I: codificador periódico fijo. Entrada (B, L, C) y calendario pasado -> imagen (B, C', L/24, 24)."""

    def __init__(self, p: dict, n_x: int, n_cal: int, L: int, H: int):
        super().__init__()
        if L % 24 or H % 24:
            raise ValueError("La historia y el horizonte deben ser múltiplos de 24 h")
        d, nb, dp = int(p["d_model"]), int(p["bloques"]), float(p["dropout"])
        self.dias, self.H = L // 24, H
        self.entrada = nn.Conv2d(n_x + n_cal, d, 1)
        self.bloques = nn.ModuleList([nn.Sequential(nn.Conv2d(d, d, (3, 3), padding=1), nn.GELU(), nn.Dropout(dp),
                                                    nn.Conv2d(d, d, (1, 5), padding=(0, 2))) for _ in range(nb)])
        self.normas = nn.ModuleList([nn.GroupNorm(1, d) for _ in range(nb)])
        self.lectura = nn.Sequential(nn.Linear(2 * d + n_cal, d), nn.GELU(), nn.Dropout(dp), nn.Linear(d, 1))
        nn.init.zeros_(self.lectura[-1].weight)
        nn.init.zeros_(self.lectura[-1].bias)
        self.register_buffer("hora", torch.arange(H) % 24, persistent=False)

    def forward(self, x, cal_p, cal_f, dia):
        B, L, _ = x.shape
        u = torch.cat([x, cal_p], dim=2).transpose(1, 2).reshape(B, -1, self.dias, 24)
        z = self.entrada(u)
        for b, n in zip(self.bloques, self.normas):
            z = n(z + b(z))
        ultimo, medio = z[:, :, -1, :], z.mean(2)                              # (B, d, 24)
        f = torch.cat([ultimo.index_select(2, self.hora), medio.index_select(2, self.hora)], dim=1).transpose(1, 2)   # (B, H, 2d)
        return 1.0 + self.lectura(torch.cat([f, cal_f], dim=2)).squeeze(-1)

    def diagnostico(self) -> dict:
        return {}

    def n_parametros(self) -> int:
        return _n_parametros(self)

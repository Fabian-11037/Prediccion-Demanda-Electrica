import torch
from torch import nn


class _Base(nn.Module):
    def diagnostico(self) -> dict:
        return {}

    def n_parametros(self) -> int:
        return sum(v.numel() for v in self.parameters() if v.requires_grad)

class _BloqueNBEATS(nn.Module):
    def __init__(self, d_in: int, L: int, H: int, ancho: int, capas: int, dropout: float):
        super().__init__()
        m, d = [], d_in
        for _ in range(capas):
            m += [nn.Linear(d, ancho), nn.ReLU()] + ([nn.Dropout(dropout)] if dropout > 0 else [])
            d = ancho
        self.fc = nn.Sequential(*m)
        self.theta = nn.Linear(ancho, L + H)        # base genérica: identidad sobre theta
        self.L = L

    def forward(self, entrada):
        t = self.theta(self.fc(entrada))
        return t[:, :self.L], t[:, self.L:]

class NBEATS(_Base):
    """N-BEATS-G (Oreshkin et al., ICLR 2020): 30 pilas de 1 bloque, 4 capas de 512, base genérica, residuo doble.

    Adaptación declarada: además del residuo de la demanda (168), cada bloque recibe las últimas 24 h de las demás
    entradas históricas y sus indicadores de ausencia y el calendario de las 24 h futuras. La salida es la suma de
    los pronósticos parciales de los bloques (en unidades de demanda / nivel - 1)."""

    def __init__(self, p: dict, n_x: int, n_cal: int, L: int, H: int):
        super().__init__()
        self.L, self.H = L, H
        pilas, ancho, capas = int(p.get("pilas", 30)), int(p.get("ancho", 512)), int(p.get("capas_fc", 4))
        d_in = L + 24 * (n_x - 1) + H * n_cal
        self.bloques = nn.ModuleList([_BloqueNBEATS(d_in, L, H, ancho, capas, float(p["dropout"])) for _ in range(pilas)])

    def forward(self, x, cal_p, cal_f, dia):
        r = x[:, :, 0].flip(1)                                  # residuo de la demanda (orden inverso como en el artículo)
        exo = x[:, -24:, 1:].flatten(1)
        fut = cal_f.flatten(1)
        y = torch.zeros(x.shape[0], self.H, device=x.device, dtype=x.dtype)
        for b in self.bloques:
            back, f = b(torch.cat([r, exo, fut], dim=1))
            r = r - back
            y = y + f
        return 1.0 + y
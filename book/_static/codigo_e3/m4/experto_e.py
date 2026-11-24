from __future__ import annotations
import math
import torch
from torch import nn
from entrenamiento import redes as R

class MLP2(nn.Module):
    """Perceptrón de dos capas con GELU; la capa de salida puede iniciar en cero."""

    def __init__(self, d_in: int, ancho: int, d_out: int, dropout: float = 0.0, salida_cero: bool = False):
        super().__init__()
        self.red = nn.Sequential(nn.Linear(d_in, ancho), nn.GELU(), nn.Dropout(dropout), nn.Linear(ancho, d_out))
        if salida_cero:
            nn.init.zeros_(self.red[-1].weight)
            nn.init.zeros_(self.red[-1].bias)

    def forward(self, u):
        return self.red(u)

class CorreccionAcotada(nn.Module):
    """Δ(u) = ρ σ(g(u)) ⊙ tanh(v(u)); ``cota`` y ``compuerta`` se pueden retirar para la ablación."""

    def __init__(self, d_in: int, H: int, ancho: int, rho: float, cota: bool = True, compuerta: bool = True, dropout: float = 0.0):
        super().__init__()
        if rho <= 0:
            raise ValueError("ρ debe ser positivo")
        self.rho, self.cota, self.compuerta = float(rho), bool(cota), bool(compuerta)
        self.v = MLP2(d_in, ancho, H, dropout, salida_cero=True)
        self.g = MLP2(d_in, ancho, H, dropout) if compuerta else None

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        v = self.v(u)
        s = torch.sigmoid(self.g(u)) if self.g is not None else 1.0
        if self.cota:
            return self.rho * s * torch.tanh(v)
        return s * v

    def cota_teorica(self) -> float:
        return self.rho if self.cota else math.inf

class _BaseOriginal(nn.Module):
    """Validación de entradas común: forma (B, L, C) con el número de canales y la longitud con los que se construyó."""

    def _forma(self, n_x: int, L: int, H: int, n_cal: int):
        self._esperado = (int(n_x), int(L), int(H), int(n_cal))

    def _validar(self, x, cal_p, cal_f):
        n_x, L, H, n_cal = self._esperado
        if x.dim() != 3 or x.shape[1] != L or x.shape[2] != n_x:
            raise RuntimeError(f"x debe tener forma (B, {L}, {n_x}); se recibió {tuple(x.shape)}")
        if cal_p.shape[1:] != (L, n_cal) or cal_f.shape[1:] != (H, n_cal):
            raise RuntimeError(f"calendario con forma inválida: {tuple(cal_p.shape)}, {tuple(cal_f.shape)}")

    def diagnostico(self) -> dict:
        return {}

    def n_parametros(self) -> int:
        return sum(v.numel() for v in self.parameters() if v.requires_grad)

class ModeloE(_BaseOriginal):
    """o = o_base(demanda, calendario) + a · Δ(u_exo),  a = fracción de covariables disponibles en las últimas 24 h.

    La base es el codificador por parches del proyecto con un solo canal (la demanda) y el calendario. La rama exógena
    recibe, por covariable, las 24 últimas horas y las 7 medias diarias de la historia, la ausencia media por bloque y el
    calendario del horizonte. Proposición 1: cada hora difiere de la base de solo demanda en a lo sumo ρ · nivel,
    cualesquiera que sean los valores de las covariables (perturbadas, faltantes o desplazadas).
    Componentes: cota, compuerta y bandera (ponderación por disponibilidad). ``sin_rama`` deja solo la base (ablación)."""

    def __init__(self, p: dict, n_x: int, n_cal: int, L: int, H: int, n_mascaras: int = 0):
        super().__init__()
        self._forma(n_x, L, H, n_cal)
        self.base = R.NucleoPatchTST(1, n_cal, 0, L, H, p)
        self.n_mascaras = int(n_mascaras)
        self.n_exo = n_x - 1 - self.n_mascaras
        self.bandera = bool(p.get("bandera", True))
        self.dias = L // 24
        d_in = (n_x - 1) * (24 + self.dias) + H * n_cal
        self.correccion = CorreccionAcotada(d_in, H, int(p["ancho_correccion"]), float(p["rho"]), bool(p.get("cota", True)),
                                            bool(p.get("compuerta", True)), float(p["dropout"])) if n_x > 1 and not p.get("sin_rama") else None

    def entrada_exogena(self, x, cal_f):
        B, L, C = x.shape
        e = x[:, :, 1:]
        diario = e.reshape(B, self.dias, L // self.dias, C - 1).mean(2)
        return torch.cat([e[:, -24:].flatten(1), diario.flatten(1), cal_f.flatten(1)], dim=1)

    def disponibilidad(self, x):
        if self.n_mascaras == 0:
            return torch.ones(x.shape[0], 1, device=x.device, dtype=x.dtype)
        return 1.0 - x[:, -24:, -self.n_mascaras:].mean(dim=(1, 2))[:, None]

    def aporte(self, x, cal_f):
        a = self.disponibilidad(x) if self.bandera else 1.0
        return a * self.correccion(self.entrada_exogena(x, cal_f))

    def forward(self, x, cal_p, cal_f, dia):
        self._validar(x, cal_p, cal_f)
        o = self.base(x[:, :, :1], cal_p, cal_f, None)
        return o if self.correccion is None else o + self.aporte(x, cal_f)

    def cota(self) -> float:
        return 0.0 if self.correccion is None else self.correccion.cota_teorica()

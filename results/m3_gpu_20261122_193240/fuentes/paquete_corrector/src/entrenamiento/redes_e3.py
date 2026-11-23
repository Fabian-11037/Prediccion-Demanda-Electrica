"""Fábrica de redes del Entregable 3: el modelo original, la familia de contrastes y las variantes de ablación.

Familia (todos con el motor común y la misma política de reentrenamiento, §5.2 del Entregable 3):
  * TaTS adaptado (modelo 6; Li et al., arXiv:2502.08942): texto como variables auxiliares sobre PatchTST. La
    variante N (sin texto) es el control PatchTST de la fusión y de TaTS. Código del paquete del modelo 6.
  * N-HiTS (Challu et al., AAAI 2023): bloques oficiales de Nixtla/neuralforecast (Apache-2.0, en src/vendor/nhits)
    con la arquitectura del repositorio (3 bloques, MLP 3 x 512, pooling 2-2-1, interpolación lineal 4-2-1).
    Adaptación declarada: se ejecuta en el motor común (demanda normalizada como residuo, sin la reconstrucción con
    sigma del motor del repositorio).
  * ARX lineal: los mismos rasgos del modelo base (48 rezagos de demanda; último valor, media de 24 h y de 168 h de
    cada otra entrada; calendario del horizonte) y una capa lineal, entrenada con el motor común (MAE); la penalización
    lasso del Entregable 1 (λ = 1,4e-7) se omite por ser despreciable.
  * El resto (PatchTST, iTransformer, TCN, TFT, TGForecaster, fusión, MLP/LSTM/CNN1D de §14, N-BEATS, TimesNet) viene
    de ``redes_e2`` y ``redes`` sin cambios.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
from torch import nn

from . import redes as R
from . import redes_e2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from original import mecanismos as OM                                # noqa: E402
from vendor.nhits.core import NHITSBlock                             # noqa: E402  (Nixtla, Apache-2.0)


# ------------------------------------------------------------------------------------ TaTS (modelo 6)
class AgregadorTexto(nn.Module):
    """z_d = Σ_j α_j V e_j, α = softmax_j ⟨K e_j, q⟩ / √d sobre los documentos de cada uno de los 7 días de la historia;
    MLP de tres capas -> d_texto canales más el indicador de presencia, repetidos en las 24 h de su día."""

    def __init__(self, vectores: torch.Tensor, lista: torch.Tensor, n_x: int, d_texto: int, dropout: float = 0.1):
        super().__init__()
        self.register_buffer("vectores", vectores, persistent=False)
        self.register_buffer("lista", lista, persistent=False)
        d_e = vectores.shape[1]
        self.consulta = nn.Sequential(nn.Linear(2 * n_x, R.D_REL), nn.GELU(), nn.Linear(R.D_REL, R.D_REL))
        self.clave = nn.Linear(d_e, R.D_REL, bias=False)
        self.valor = nn.Linear(d_e, R.D_REL, bias=False)
        self.mlp = nn.Sequential(nn.Linear(R.D_REL, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 32), nn.GELU(), nn.Linear(32, d_texto))

    def forward(self, x: torch.Tensor, dia: torch.Tensor, dias_hist: int = 7):
        B, L, _ = x.shape
        q = self.consulta(torch.cat([x[:, -24:].mean(1), x.mean(1)], dim=1))
        dias = dia[:, None] + torch.arange(-(dias_hist - 1), 1, device=dia.device)[None, :]
        idx = self.lista[dias.clamp(min=0)]
        presente = (idx >= 0) & (dias >= 0)[:, :, None]
        e = self.vectores[idx.clamp(min=0)]
        puntaje = torch.einsum("bkme,be->bkm", e, q @ self.clave.weight) / math.sqrt(R.D_REL)
        puntaje = puntaje.masked_fill(~presente, float("-inf"))
        hay = presente.any(-1)
        alfa = torch.softmax(puntaje.masked_fill(~hay[:, :, None], 0.0), dim=-1) * presente
        z = self.valor(torch.einsum("bkm,bkme->bke", alfa, e))
        canales = self.mlp(z) * hay.unsqueeze(-1)
        salida = torch.cat([canales, hay.unsqueeze(-1).to(canales.dtype)], dim=-1)
        return salida.repeat_interleave(L // dias_hist, dim=1), alfa


class TaTS(redes_e2._Base):
    def __init__(self, p: dict, n_x: int, n_cal: int, L: int, H: int, texto: dict | None = None):
        super().__init__()
        self.agregador = None
        n_txt = 0
        if texto is not None:
            self.agregador = AgregadorTexto(texto["vectores"], texto["lista"], n_x, int(p["d_texto"]))
            n_txt = int(p["d_texto"]) + 1
        clase = {"itransformer": R.NucleoITransformer, "patchtst": R.NucleoPatchTST}[p["nucleo"]]
        self.nucleo = clase(n_x, n_cal, n_txt, L, H, p)

    def forward(self, x, cal_p, cal_f, dia):
        txt = self.agregador(x, dia)[0] if self.agregador is not None else None
        return self.nucleo(x, cal_p, cal_f, txt)


# ------------------------------------------------------------------------------------ N-HiTS
class _BaseLineal(nn.Module):
    """Interpolación lineal de los nodos materializada como matriz fija (equivale a interpolate, align_corners=False)."""

    def __init__(self, L, H, nodos):
        super().__init__()
        self.L = L
        m = nn.functional.interpolate(torch.eye(nodos)[:, None, :], size=H, mode="linear", align_corners=False)[:, 0, :]
        self.register_buffer("interpolacion", m)

    def forward(self, theta):
        return theta[:, :self.L], theta[:, self.L:] @ self.interpolacion


class NHITS14(redes_e2._Base):
    def __init__(self, p: dict, n_x: int, n_cal: int, L: int, H: int):
        super().__init__()
        self.bloques = nn.ModuleList([
            NHITSBlock(input_size=L, h=H, n_theta=L + H // f, mlp_units=[[512, 512]] * 3, basis=_BaseLineal(L, H, H // f),
                       futr_input_size=n_cal, hist_input_size=n_x - 1, stat_input_size=0, n_pool_kernel_size=pool, pooling_mode="MaxPool1d",
                       dropout_prob=float(p["dropout"]), activation="ReLU")
            for pool, f in zip([2, 2, 1], [4, 2, 1])])

    def forward(self, x, cal_p, cal_f, dia):
        residuo = x[:, :, 0].flip(-1)
        hist, fut = x[:, :, 1:], torch.cat([cal_p, cal_f], dim=1)
        y = torch.zeros(x.shape[0], cal_f.shape[1], device=x.device, dtype=x.dtype)
        for b in self.bloques:
            back, f = b(residuo, fut, hist, None)
            residuo = residuo - back
            y = y + f
        return 1.0 + y


# ------------------------------------------------------------------------------------ ARX lineal
class ARX(redes_e2._Base):
    def __init__(self, p: dict, n_x: int, n_cal: int, L: int, H: int):
        super().__init__()
        self.rezagos = int(p.get("rezagos", 48))
        self.lin = nn.Linear(self.rezagos + 3 * (n_x - 1) + H * n_cal, H)
        nn.init.zeros_(self.lin.bias)

    def forward(self, x, cal_p, cal_f, dia):
        partes = [x[:, -self.rezagos:, 0]]
        if x.shape[2] > 1:
            e = x[:, :, 1:]
            partes += [e[:, -1], e[:, -24:].mean(1), e.mean(1)]
        partes.append(cal_f.flatten(1))
        return 1.0 + self.lin(torch.cat(partes, dim=1))


# ------------------------------------------------------------------------------------ fábrica
def construir(red: str, p: dict, n_x: int, n_cal: int, L: int, H: int, texto: dict | None = None, descripciones: torch.Tensor | None = None,
              n_mascaras: int = 0):
    if red.startswith("x_"):                                   # exploración de diseño: mecanismos G/H y codificador periódico I
        from exploracion import mecanismos as EX
        return EX.construir(red[2:], p, n_x, n_cal, L, H, n_mascaras, construir)
    if red.startswith("original_"):
        return OM.construir(red.split("_", 1)[1], p, n_x, n_cal, L, H, texto, n_mascaras)
    if red == "tats":
        return TaTS(p, n_x, n_cal, L, H, texto)
    if red == "nhits":
        return NHITS14(p, n_x, n_cal, L, H)
    if red == "arx":
        return ARX(p, n_x, n_cal, L, H)
    return redes_e2.construir(red, p, n_x, n_cal, L, H, texto, descripciones)

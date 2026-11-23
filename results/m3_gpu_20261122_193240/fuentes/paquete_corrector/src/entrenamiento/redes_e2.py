"""Redes del paquete de cierre del Entregable 2.

Todas siguen la interfaz del motor: ``red(x, cal_p, cal_f, dia) -> (B, H)`` con la etiqueta
``demanda / nivel de la ventana`` y devuelven ``1 + corrección``.

* Búsqueda de arquitectura (§14): MLP, LSTM y CNN1D (entradas de los modelos base: demanda y
  calendario), TCN, TGForecaster y la rama de texto de la fusión tardía, con tres bloques de anchos
  ``anchos = (w1, w2, w3)`` en {16, 32, 64, 128, 256} y dropout 0,2 según ``patron_dropout = (b1, b2, b3)``.
* Modelos nuevos del Top 10 con arquitectura publicada: N-BEATS genérico (Oreshkin et al., 2020) y
  TimesNet (Wu et al., 2023; bloques oficiales de Time-Series-Library, licencia MIT).
* Ablaciones del mejor modelo numérico (PatchTST adaptado, configuración 18 del modelo 6).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

from . import redes as R

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vendor" / "timesnet"))
import TimesNet as _TN                                           # noqa: E402  (oficial, MIT)
from layers.Embed import PositionalEmbedding, TokenEmbedding     # noqa: E402  (oficial, MIT)

TASA_14 = 0.2          # §14.1: tasa fija de dropout
ANCHOS_14 = (16, 32, 64, 128, 256)


class _Base(nn.Module):
    def diagnostico(self) -> dict:
        return {}

    def n_parametros(self) -> int:
        return sum(v.numel() for v in self.parameters() if v.requires_grad)


def _patron(p: dict) -> tuple[list[int], list[int]]:
    w, b = [int(a) for a in p["anchos"]], [int(a) for a in p["patron_dropout"]]
    if len(w) != 3 or len(b) != 3:
        raise ValueError("La búsqueda de §14 usa exactamente tres bloques")
    return w, b


def _drop(activo: int) -> nn.Module:
    return nn.Dropout(TASA_14) if activo else nn.Identity()


# ------------------------------------------------------------------ modelos base (demanda y calendario)
class MLP14(_Base):
    """Entrada: 168 rezagos de demanda normalizada y calendario de las 24 h del horizonte (720 valores).
    Tres capas densas con ReLU; dropout tras la activación de la capa i si b_i = 1; salida lineal sin dropout."""

    def __init__(self, p: dict, n_x: int, n_cal: int, L: int, H: int):
        super().__init__()
        w, b = _patron(p)
        dims, capas = [L + H * n_cal] + w, []
        for i in range(3):
            capas += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU(), _drop(b[i])]
        self.red = nn.Sequential(*capas, nn.Linear(w[-1], H))
        nn.init.zeros_(self.red[-1].bias)

    def forward(self, x, cal_p, cal_f, dia):
        return 1.0 + self.red(torch.cat([x[:, :, 0], cal_f.flatten(1)], dim=1))


class LSTM14(_Base):
    """Por hora: demanda normalizada y calendario de la historia. Tres capas LSTM apiladas con anchos propios;
    el dropout se aplica a la secuencia de salida de la capa i (en la capa 3, al último estado, antes de la cabeza).
    Dropout estándar (no variacional). Cabeza: último estado + calendario futuro -> 24 h."""

    def __init__(self, p: dict, n_x: int, n_cal: int, L: int, H: int):
        super().__init__()
        w, b = _patron(p)
        entradas = [1 + n_cal] + w
        self.lstm = nn.ModuleList([nn.LSTM(entradas[i], entradas[i + 1], batch_first=True) for i in range(3)])
        self.drops = nn.ModuleList([_drop(b[i]) for i in range(3)])
        self.cabeza = nn.Linear(w[-1] + H * n_cal, H)
        nn.init.zeros_(self.cabeza.bias)

    def forward(self, x, cal_p, cal_f, dia):
        s = torch.cat([x[:, :, :1], cal_p], dim=2)
        for i, capa in enumerate(self.lstm):
            s = capa(s)[0]
            if i < 2:
                s = self.drops[i](s)
        h = self.drops[2](s[:, -1])
        return 1.0 + self.cabeza(torch.cat([h, cal_f.flatten(1)], dim=1))


class CNN14(_Base):
    """Canales: demanda normalizada y calendario de la historia. Tres bloques Conv1D (relleno simétrico, misma
    longitud) + ReLU; dropout tras el bloque i si b_i = 1. Después AvgPool 2, aplanado, calendario futuro y lectura lineal."""

    def __init__(self, p: dict, n_x: int, n_cal: int, L: int, H: int):
        super().__init__()
        w, b = _patron(p)
        k = int(p["kernel"])
        canales, capas = [1 + n_cal] + w, []
        for i in range(3):
            capas += [nn.Conv1d(canales[i], canales[i + 1], k, padding=k // 2), nn.ReLU(), _drop(b[i])]
        self.conv = nn.Sequential(*capas)
        self.pool = nn.AvgPool1d(2)
        self.lectura = nn.Linear(w[-1] * (L // 2) + H * n_cal, H)
        nn.init.zeros_(self.lectura.bias)

    def forward(self, x, cal_p, cal_f, dia):
        z = self.pool(self.conv(torch.cat([x[:, :, :1], cal_p], dim=2).transpose(1, 2)))
        return 1.0 + self.lectura(torch.cat([z.flatten(1), cal_f.flatten(1)], dim=1))


# ------------------------------------------------------------------ TGForecaster con tres bloques de anchos libres
class _CapaPatchAncho(nn.Module):
    """Capa del codificador de parches con ancho w y cabezas w/16 (§14.2); dropout 0,2 tras la atención y tras
    la red feed-forward si b = 1. Si el ancho cambia, una proyección lineal precede al bloque."""

    def __init__(self, d_in: int, d: int, activo: int):
        super().__init__()
        self.proy = nn.Linear(d_in, d) if d_in != d else nn.Identity()
        self.capa = R._CapaPatch(d, max(1, d // 16), TASA_14 if activo else 0.0)

    def forward(self, z):
        return self.capa(self.proy(z))


class TGForecaster14(R.TGForecaster):
    """TGForecaster del modelo 7 con el codificador de parches sustituido por tres bloques (w_i, b_i).
    El resto (descripciones, consultas del horizonte, mezcla, decodificador) usa el ancho w3 y las capas de
    mezcla y el dropout fijados en la fase 0."""

    def __init__(self, p: dict, n_x: int, n_cal: int, L: int, H: int, descripciones: torch.Tensor, texto: dict | None = None):
        w, b = _patron(p)
        q = dict(p, d_model=w[-1], cabezas=max(1, w[-1] // 16), capas=3)
        super().__init__(q, n_x, n_cal, L, H, descripciones, texto)
        self.emb = nn.Linear(R.PARCHE, w[0])
        self.pos = nn.Parameter(torch.zeros(self.n_parches, w[0]))
        nn.init.normal_(self.pos, std=0.02)
        anchos = [w[0]] + w
        self.capas = nn.ModuleList([_CapaPatchAncho(anchos[i], anchos[i + 1], b[i]) for i in range(3)])


# ------------------------------------------------------------------ fusión tardía con rama de texto de tres capas
class FusionTardia14(R.FusionTardia):
    """Núcleo numérico fijado en la fase 0; rama de texto 64 -> w1 -> w2 -> w3 -> 24 con GELU y dropout según el
    patrón (la capa de salida no lleva dropout y empieza en cero)."""

    def __init__(self, p: dict, n_x: int, n_cal: int, L: int, H: int, texto: dict | None = None):
        super().__init__(dict(p, d_fusion=16), n_x, n_cal, L, H, texto)
        if texto is not None:
            w, b = _patron(p)
            dims, capas = [R.D_REL] + w, []
            for i in range(3):
                capas += [nn.Linear(dims[i], dims[i + 1]), nn.GELU(), _drop(b[i])]
            self.rama = nn.Sequential(*capas, nn.Linear(w[-1], H))
            nn.init.zeros_(self.rama[-1].weight)
            nn.init.zeros_(self.rama[-1].bias)


# ------------------------------------------------------------------ N-BEATS genérico
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


# ------------------------------------------------------------------ TimesNet
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


# ------------------------------------------------------------------ ablaciones de PatchTST
ABLACIONES = ("completo", "sin_via_contexto", "sin_atencion", "sin_exogenas", "sin_calendario")


class PatchTSTAblacion(_Base):
    """Núcleo PatchTST del modelo 6 (variante N). ``p['ablacion']``:
    completo          sin cambios;
    sin_via_contexto  solo la cabeza lineal aplanada del artículo sobre la demanda (sin MLP de contexto);
    sin_atencion      el codificador de parches se reemplaza por la identidad (sin capas de atención);
    sin_exogenas      solo el canal de demanda (sin precio, hidrología, clima ni indicadores de ausencia);
    sin_calendario    el calendario de la historia y del horizonte se anula (ceros)."""

    def __init__(self, p: dict, n_x: int, n_cal: int, L: int, H: int):
        super().__init__()
        self.ab = p["ablacion"]
        if self.ab not in ABLACIONES:
            raise ValueError(f"Ablación desconocida: {self.ab}")
        self.n = R.NucleoPatchTST(1 if self.ab == "sin_exogenas" else n_x, n_cal, 0, L, H, dict(p, capas=0 if self.ab == "sin_atencion" else p["capas"]))
        if self.ab == "sin_via_contexto":
            self.n.contexto = None

    def forward(self, x, cal_p, cal_f, dia):
        if self.ab == "sin_exogenas":
            x = x[:, :, :1]
        if self.ab == "sin_calendario":
            cal_p, cal_f = torch.zeros_like(cal_p), torch.zeros_like(cal_f)
        if self.ab != "sin_via_contexto":
            return self.n(x, cal_p, cal_f, None)
        n = self.n
        B, L, C = x.shape
        z = n.drop(n.emb(R._parches(x.transpose(1, 2).reshape(B * C, L))) + n.pos)
        for capa in n.capas:
            z = capa(z)
        z = z.reshape(B, C, n.n_parches, -1)
        return 1.0 + n.lineal(z[:, 0].flatten(1))


# ------------------------------------------------------------------ fábrica
def construir(red: str, p: dict, n_x: int, n_cal: int, L: int, H: int, texto: dict | None = None, descripciones: torch.Tensor | None = None):
    if red == "mlp14":
        return MLP14(p, n_x, n_cal, L, H)
    if red == "lstm14":
        return LSTM14(p, n_x, n_cal, L, H)
    if red == "cnn14":
        return CNN14(p, n_x, n_cal, L, H)
    if red == "tcn14":
        return R.TCN(dict(p, dropout=TASA_14), n_x, n_cal, L, H)
    if red == "tgforecaster14":
        return TGForecaster14(p, n_x, n_cal, L, H, descripciones, texto)
    if red == "fusion14":
        return FusionTardia14(p, n_x, n_cal, L, H, texto)
    if red == "nbeats":
        return NBEATS(p, n_x, n_cal, L, H)
    if red == "timesnet":
        return TimesNetE2(p, n_x, n_cal, L, H)
    if red == "patchtst_ablacion":
        return PatchTSTAblacion(p, n_x, n_cal, L, H)
    return R.construir(red, p, n_x, n_cal, L, H, texto, descripciones)

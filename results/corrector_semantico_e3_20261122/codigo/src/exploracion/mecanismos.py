"""Mecanismos solo numéricos de la exploración de diseño (ideas G, H e I; la idea J, agregación, está en ``agregacion.py``).

G. Coherencia entre orígenes. La base pronostica 2H horas (calendario futuro de 2H, determinista). La pérdida de cada lote
   suma, para los pares de orígenes consecutivos d, d+1 que están ambos en el conjunto de ajuste,

       μ · MAE(ŷ_d[H:2H], y_{d+1})  +  λ · RMS(ŷ_d[H:2H] − sg(ŷ_{d+1}[0:H]))

   en la escala del nivel de d (sg: sin gradiente). Proposición G: si ŷ es la esperanza condicional (pronóstico óptimo con
   pérdida cuadrática), la revisión ŷ_{d+1}[0:H] − ŷ_d[H:2H] es una diferencia de martingala con media cero dada la
   información de d; el término de coherencia en L2 es cero en esperanza exactamente cuando el pronóstico del día 2 es la
   esperanza condicional del pronóstico del día siguiente, así que no desplaza al pronóstico óptimo y solo actúa como
   regularizador en muestra finita. Supuesto declarado: la pérdida principal es MAE (mediana), por eso la propiedad se usa
   solo en el término L2. μ = 0 da la coherencia "pura" (el día 2 solo se ancla al pronóstico del día siguiente);
   λ = 0 con μ > 0 es el control de supervisión a 48 h, que no es una contribución propia.
   En inferencia solo se usan las primeras H horas: la salida tiene la misma forma que la de la base.

H. DRO temporal por grupos de años. Grupos g = año del origen de ajuste. La pérdida del lote es Σ_g q_g L_g / Σ_g q_g sobre
   los grupos presentes, con q = softmax(log π + (L̄_g / media(L̄) − 1) / τ), L̄ media exponencial (sin gradiente) de la
   pérdida de cada grupo y π_g ∝ 2^{−(G−1−g)/vida} la prior de recencia. q es el maximizador de Σ_g q_g L̄_g − τ' KL(q‖π)
   (dual de Donsker-Varadhan, con τ' = τ · media(L̄)); por eso el objetivo acota el riesgo de cualquier mezcla de años
   con KL(q‖π) ≤ r: Σ q_g L_g ≤ objetivo + τ' r (Proposición H, desigualdad de Fenchel-Young).

I. Codificador periódico 7 × 24. La historia de 168 h se reorganiza como imagen (días × horas) y se procesa con
   convoluciones 2D residuales; la lectura de cada hora del horizonte usa la fila del último día, la media de los 7 días en
   esa hora y el calendario de esa hora. Periodos fijos (24 y 168 h), sin FFT: determinista y exportable a ONNX.
"""
from __future__ import annotations

import math

import torch
from torch import nn


def _n_parametros(m: nn.Module) -> int:
    return sum(v.numel() for v in m.parameters() if v.requires_grad)


class Envoltura(nn.Module):
    """Aplica G (coherencia) y/o H (DRO temporal) a cualquier red del motor. ``base`` debe construirse con H_salida horas:
    2H con coherencia, H sin ella. Sin mecanismos activos es exactamente la base."""

    def __init__(self, base: nn.Module, H: int, coherencia: dict | None = None, dro: dict | None = None):
        super().__init__()
        self.base, self.H = base, int(H)
        self.coh = dict(coherencia) if coherencia else None
        self.dro = dict(dro) if dro else None
        self._ult = None
        self._ema = None
        self._q = None
        self.register_buffer("_dummy", torch.zeros(0), persistent=False)

    # ------------------------------------------------------------------ inferencia
    def forward(self, x, cal_p, cal_f, dia):
        o = self.base(x, cal_p, cal_f, dia)
        if self.coh is not None:
            if self.training:
                self._ult = o
            return o[:, :self.H]
        return o

    # ------------------------------------------------------------------ pérdidas de entrenamiento
    def perdida_principal(self, salida, t: dict, ix):
        y = t["y"][ix]
        if self.dro is None:
            return (salida - y).abs().mean()
        g = t["grupo"][ix]
        G = int(t["grupo"].max()) + 1
        if "recencia" in self.dro:                     # control de H: pesos fijos 2^{-(G-1-g)/vida}, sin adversario
            w = torch.pow(2.0, -(G - 1 - g).float() / float(self.dro["recencia"]))
            return (w * (salida - y).abs().mean(1)).sum() / w.sum()
        if self._ema is None or self._ema.numel() != G:
            self._ema = torch.full((G,), float("nan"), device=salida.device)
            vida = float(self.dro.get("vida", 3.0))
            k = torch.arange(G, device=salida.device, dtype=torch.float32)
            self._logpi = -(G - 1 - k) * math.log(2.0) / vida if math.isfinite(vida) else torch.zeros(G, device=salida.device)
            self._logpi = self._logpi - torch.logsumexp(self._logpi, 0)
        por = (salida - y).abs().mean(1)
        presentes = torch.unique(g)
        L = torch.stack([por[g == k].mean() for k in presentes])
        beta = float(self.dro.get("beta", 0.9))
        with torch.no_grad():
            e = self._ema[presentes]
            self._ema[presentes] = torch.where(torch.isnan(e), L.detach(), beta * e + (1 - beta) * L.detach())
            ema = torch.where(torch.isnan(self._ema), torch.nanmean(self._ema), self._ema)
            q = torch.softmax(self._logpi + (ema / ema.mean().clamp_min(1e-12) - 1.0) / float(self.dro.get("tau", 0.1)), 0)
            self._q = q
        w = q[presentes]
        return (w * L).sum() / w.sum()

    def perdida_adicional(self, t: dict, ix):
        if self.coh is None or self._ult is None:
            return torch.zeros((), device=t["y"].device)
        o = self._ult
        self._ult = None
        sig = t["sig"][ix]
        m = sig >= 0
        if not bool(m.any()):
            return o.new_zeros(())
        j = sig[m]
        r = (t["nivel"][j] / t["nivel"][ix[m]])[:, None]           # escala del nivel de d+1 -> escala del nivel de d
        d2 = o[m, self.H:2 * self.H]
        frac = m.float().mean()
        total = o.new_zeros(())
        mu, lam = float(self.coh.get("mu", 1.0)), float(self.coh.get("lam", 0.0))
        if mu > 0:
            total = total + mu * frac * (d2 - t["y"][j] * r).abs().mean()
        if lam > 0:
            self.base.eval()                         # objetivo sin gradiente y sin ruido de dropout (el estado de ajuste se restaura)
            with torch.no_grad():
                sig_pred = self.base(t["x"][j], t["cal_p"][j], t["cal_f"][j], t["dia"][j])[:, :self.H] * r
            self.base.train()
            total = total + lam * frac * torch.sqrt(((d2 - sig_pred) ** 2).mean() + 1e-12)
        return total

    # ------------------------------------------------------------------ interfaz del motor
    def diagnostico(self) -> dict:
        d = dict(self.base.diagnostico()) if hasattr(self.base, "diagnostico") else {}
        if self._q is not None:
            q = self._q.detach().cpu()
            d.update(dro_peso_ultimo_anio=float(q[-1]), dro_peso_maximo=float(q.max()), dro_entropia=float(-(q * q.clamp_min(1e-12).log()).sum()))
        return d

    def n_parametros(self) -> int:
        return _n_parametros(self)


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


# ------------------------------------------------------------------------------------ fábrica
CLAVES_MECANISMO = ("mec_mu", "mec_lam", "mec_tau", "mec_vida", "mec_beta", "mec_rec_vida", "rec_promedio", "rec_epocas_ema", "rec_inicio")


def mecanismos_de(p: dict) -> tuple[dict | None, dict | None]:
    mu, lam = float(p.get("mec_mu", 0.0)), float(p.get("mec_lam", 0.0))
    coh = dict(mu=mu, lam=lam) if (mu > 0 or lam > 0) else None
    dro = dict(tau=float(p["mec_tau"]), vida=float(p.get("mec_vida", 3.0)), beta=float(p.get("mec_beta", 0.9))) if "mec_tau" in p else None
    if "mec_rec_vida" in p:
        if dro is not None:
            raise ValueError("La ponderación por recencia y el DRO temporal son excluyentes")
        dro = dict(recencia=float(p["mec_rec_vida"]))
    return coh, dro


def construir(red_base: str, p: dict, n_x: int, n_cal: int, L: int, H: int, n_mascaras: int, fabrica) -> nn.Module:
    """``red_base``: 'periodica' o cualquier red de ``fabrica`` (redes_e3.construir). Los hiperparámetros del mecanismo van en
    ``p`` con prefijo ``mec_`` (quedan registrados en la configuración de cada corrida)."""
    coh, dro = mecanismos_de(p)
    h_salida = 2 * H if coh is not None else H
    q = {k: v for k, v in p.items() if k not in CLAVES_MECANISMO}
    if red_base == "periodica":
        base = Periodica7x24(q, n_x, n_cal, L, h_salida)
    else:
        base = fabrica(red_base, q, n_x, n_cal, L, h_salida, None, None, n_mascaras=n_mascaras)
    if coh is None and dro is None:
        return base                      # la receta de promedio de pesos (rec_*) la aplica el motor, no la arquitectura
    return Envoltura(base, H, coh, dro)


def horizonte_calendario(p: dict, H: int) -> int:
    return 2 * H if mecanismos_de(p)[0] is not None else H

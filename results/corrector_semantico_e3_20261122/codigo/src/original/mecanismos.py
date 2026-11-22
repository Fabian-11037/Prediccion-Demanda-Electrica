"""Modelo original del Entregable 3: las seis ideas del documento de diseño, programadas desde su formulación.

Notación (todas las redes siguen la interfaz del motor del proyecto):
  entrada  x (B, L, C) historia normalizada (canal 0: demanda / nivel - 1 escalada), cal_p (B, L, K), cal_f (B, H, K),
           dia (B,) índice del día del origen en la tabla de documentos;
  salida   o (B, H) = demanda / nivel de la ventana; la predicción en kWh es o * nivel.

Componente común — corrección residual acotada (ideas A, B, C y E):
        o = o_base + a ⊙ Δ(u),    Δ(u) = ρ · σ(g(u)) ⊙ tanh(v(u))
  * cota:      |Δ_h(u)| ≤ ρ para todo u, porque 0 < σ < 1 y |tanh| < 1;
  * compuerta: σ(g(u)) decide en qué horas se corrige; sin compuerta Δ = ρ · tanh(v(u));
  * bandera:   a ∈ [0, 1] es la presencia de la fuente (1 si hay documentos o covariables; 0 si faltan);
  * sin cota:  Δ = σ(g) ⊙ v (no acotado), la variante estándar de la ablación.
  Proposición 1 (no degradación acotada). Para cualquier entrada auxiliar, |ŷ_h − ŷ_base,h| ≤ ρ · nivel y, por la
  desigualdad triangular, MAE(ŷ) ≤ MAE(ŷ_base) + ρ · media(nivel). Si a = 0 la salida es idéntica a la base.
  La última capa de v empieza en cero (ReZero, Bachlechner et al., 2020): el entrenamiento parte de o = o_base.

Ideas:
  A   ModeloA    base PatchTST adaptado (todas las variables numéricas) + texto oficial resumido por relevancia aprendida.
  AB  ModeloAB   A con el resumen de texto del núcleo de respuesta al impulso (idea B).
  AC  ModeloAC   A entrenado con la pérdida de texto placebo (idea C, ``perdida_adicional``).
  E   ModeloE    base de solo demanda y calendario + corrección acotada de las covariables exógenas.
  F   ModeloF    nivel separado por una mezcla convexa de K perfiles diarios aprendidos.
  D   no es una red: intervalos conformes condicionados por texto (``analisis/incertidumbre_e3.py``).
"""
from __future__ import annotations

import math

import torch
from torch import nn

from entrenamiento import redes as R

D_REL = R.D_REL


# ============================================================================ componentes propios
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


class NucleoImpulso(nn.Module):
    """Idea B: resumen de documentos con un núcleo de respuesta al impulso aprendido (adaptado de los núcleos de
    decaimiento de Mei y Eisner, 2017, para eventos que afectan a una serie continua).

        w_i = κ(k_i) · σ(⟨K e_i, q⟩ / √d),   κ(k) = Σ_j softplus(α_j) exp(−k / τ_j),   τ_j = 0,5 + softplus(β_j) días
        z   = Σ_i w_i V e_i / (1 + Σ_i w_i)

    k_i ∈ {0, …, D−1} es la antigüedad en días del documento i (disponible en (d−k−1, d−k] días); solo entran los
    documentos con disponibilidad no posterior al origen. Propiedades: (i) invarianza a cualquier permutación de los
    documentos (suma); (ii) un documento disponible después del origen no puede entrar (k ≥ 0 por construcción de
    ``lista_dia``); (iii) ‖z‖ ≤ max_i ‖V e_i‖, porque z es una combinación con pesos no negativos de suma < 1.
    """

    def __init__(self, vectores: torch.Tensor, lista_dia: torch.Tensor, n_x: int, dias: int = 14, nucleos: int = 3):
        super().__init__()
        self.register_buffer("vectores", vectores, persistent=False)
        self.register_buffer("lista_dia", lista_dia, persistent=False)
        d_e = vectores.shape[1]
        self.dias = int(dias)
        self.consulta = nn.Sequential(nn.Linear(2 * n_x, D_REL), nn.GELU(), nn.Linear(D_REL, D_REL))
        self.clave = nn.Linear(d_e, D_REL, bias=False)
        self.valor = nn.Linear(d_e, D_REL, bias=False)
        self.alfa = nn.Parameter(torch.zeros(nucleos))
        self.beta = nn.Parameter(torch.linspace(-1.0, 2.0, nucleos))

    def kappa(self, k: torch.Tensor) -> torch.Tensor:
        tau = 0.5 + nn.functional.softplus(self.beta)
        return (nn.functional.softplus(self.alfa)[None, :] * torch.exp(-k[..., None] / tau)).sum(-1)

    def forward(self, x: torch.Tensor, dia: torch.Tensor, desplazamiento: torch.Tensor | None = None):
        d = dia if desplazamiento is None else dia - desplazamiento
        q = self.consulta(torch.cat([x[:, -24:].mean(1), x.mean(1)], dim=1))
        k = torch.arange(self.dias, device=x.device)
        dias = d[:, None] - k[None, :]                                           # (B, D)
        idx = self.lista_dia[dias.clamp(min=0)]                                  # (B, D, M)
        presente = (idx >= 0) & (dias >= 0)[:, :, None]
        e = self.vectores[idx.clamp(min=0)]                                      # (B, D, M, 768)
        afinidad = torch.sigmoid(torch.einsum("bkme,be->bkm", e, q @ self.clave.weight) / math.sqrt(D_REL))
        w = self.kappa(k.to(x.dtype))[None, :, None] * afinidad * presente        # (B, D, M), w ≥ 0
        s = w.sum(dim=(1, 2))
        z = self.valor(torch.einsum("bkm,bkme->be", w, e)) / (1.0 + s)[:, None]
        hay = presente.flatten(1).any(-1).to(x.dtype)
        return z * hay[:, None], hay, w


def _contexto_numerico(x: torch.Tensor, cal_f: torch.Tensor) -> torch.Tensor:
    """Resumen numérico del origen para las ramas: medias de las últimas 24 h y de las 168 h y calendario medio futuro."""
    return torch.cat([x[:, -24:].mean(1), x.mean(1), cal_f.mean(1)], dim=1)


def _desplazamiento_placebo(dia: torch.Tensor, generador: torch.Generator | None = None, fijo: bool = False) -> torch.Tensor:
    """Donante estrictamente anterior al origen: entre 28 y 364 días antes (nunca un día posterior)."""
    if fijo:
        return 28 + (dia * 7919) % 337
    u = torch.randint(28, 365, tuple(dia.shape), generator=generador, device="cpu").to(dia.device)
    return u


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


# ============================================================================ ideas A, AB y AC (texto)
class ModeloA(_BaseOriginal):
    """o = o_base(x, cal) + a · Δ([z, c]),  a = presencia de documentos (bandera), z resumen de texto, c contexto numérico.

    ``p``: hiperparámetros de la base (d_model, capas, cabezas, dropout) y propios: rho, ancho_correccion; componentes
    cota, compuerta, bandera (True por omisión) y resumen ('relevancia' en A; 'impulso' en AB). Sin texto (variante N)
    el modelo es exactamente su base."""

    resumen_texto = "relevancia"

    def __init__(self, p: dict, n_x: int, n_cal: int, L: int, H: int, texto: dict | None = None):
        super().__init__()
        self._forma(n_x, L, H, n_cal)
        self.base = R.NucleoPatchTST(n_x, n_cal, 0, L, H, p)
        self.bandera = bool(p.get("bandera", True))
        self.texto = None
        self.placebo_fijo = False
        if texto is not None:
            if p.get("resumen", self.resumen_texto) == "impulso":
                self.texto = NucleoImpulso(texto["vectores"], texto["lista_dia"], n_x, int(p.get("dias_impulso", 14)))
            else:
                self.texto = R.Relevancia(texto["vectores"], texto["lista"], n_x)
            self.correccion = CorreccionAcotada(D_REL + 2 * n_x + n_cal, H, int(p["ancho_correccion"]), float(p["rho"]),
                                                bool(p.get("cota", True)), bool(p.get("compuerta", True)), float(p["dropout"]))

    def _resumen(self, x, dia, placebo: bool = False, generador=None):
        if not placebo:
            return self.texto(x, dia)[:2]
        delta = _desplazamiento_placebo(dia, generador, fijo=self.placebo_fijo)
        if isinstance(self.texto, NucleoImpulso):
            return self.texto(x, dia, delta)[:2]
        return self.texto(x, (dia - delta).clamp(min=0))[:2]

    def aporte(self, x, cal_f, dia, placebo: bool = False, generador=None) -> torch.Tensor:
        z, hay = self._resumen(x, dia, placebo, generador)
        a = hay[:, None] if self.bandera else torch.ones_like(hay)[:, None]
        return a * self.correccion(torch.cat([z, _contexto_numerico(x, cal_f)], dim=1))

    def forward(self, x, cal_p, cal_f, dia, placebo: bool = False):
        self._validar(x, cal_p, cal_f)
        o = self.base(x, cal_p, cal_f, None)
        return o if self.texto is None else o + self.aporte(x, cal_f, dia, placebo)

    def cota(self) -> float:
        return 0.0 if self.texto is None else self.correccion.cota_teorica()


class ModeloAB(ModeloA):
    resumen_texto = "impulso"


class ModeloAC(ModeloA):
    """Idea C: pérdida con texto placebo. Además del MAE, el motor suma

        λ · mean | a_p ⊙ Δ([z_p, c]) |

    donde z_p es el resumen del texto de un día donante anterior (entre 28 y 364 días antes) y c el contexto numérico
    del origen real. Proposición 3: si el texto es independiente de la historia y del objetivo, el texto del donante
    tiene la misma distribución conjunta con (x, y) que el real, el término MAE no puede mejorar en esperanza con el
    texto y el término placebo es mínimo con Δ ≡ 0; luego el minimizador poblacional no usa el texto."""

    def __init__(self, p: dict, n_x: int, n_cal: int, L: int, H: int, texto: dict | None = None):
        super().__init__(p, n_x, n_cal, L, H, texto)
        self.lam = float(p.get("lambda_placebo", 0.3))
        self._gen = torch.Generator().manual_seed(int(p.get("semilla_placebo", 20261118)))

    def perdida_adicional(self, t: dict, ix) -> torch.Tensor:
        if self.texto is None or self.lam == 0:
            return torch.zeros((), device=t["x"].device)
        x, cal_f, dia = t["x"][ix], t["cal_f"][ix], t["dia"][ix]
        return self.lam * self.aporte(x, cal_f, dia, placebo=True, generador=self._gen).abs().mean()


# ============================================================================ idea E (covariables exógenas)
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


# ============================================================================ idea F (perfiles diarios)
class ModeloF(_BaseOriginal):
    """o_h = ℓ · Σ_k π_k P_{k,h},  π = softmax(W_π φ) ∈ Δ^{K−1},  P_k = softplus(θ_k) / media_h softplus(θ_k),
    ℓ = 1 + ρ_ℓ tanh(w_ℓ · φ),  φ = MLP([demanda normalizada de 168 h, calendario futuro, calendario diario pasado]).

    Proposición 2. (i) Equivarianza de escala: si la historia de demanda se multiplica por c > 0 (las demás entradas
    fijas), la entrada normalizada no cambia y ŷ = o · nivel se multiplica por c. (ii) Envolvente: para cualquier
    entrada, (1 − ρ_ℓ) min_k P_{k,h} ≤ o_h ≤ (1 + ρ_ℓ) max_k P_{k,h}; la forma diaria está en la envolvente convexa
    de los perfiles y el pronóstico es positivo. Componentes: simplex (softmax frente a combinación lineal libre),
    nivel separado (frente a ℓ = 1) y calendario en los pesos (frente a solo historia)."""

    def __init__(self, p: dict, n_x: int, n_cal: int, L: int, H: int):
        super().__init__()
        self._forma(n_x, L, H, n_cal)
        self.K, self.H, self.dias = int(p["perfiles"]), H, L // 24
        self.simplex, self.nivel, self.con_calendario = bool(p.get("simplex", True)), bool(p.get("nivel_separado", True)), bool(p.get("calendario", True))
        self.rho_l = float(p.get("rho_nivel", 0.2))
        d_in = L + ((H + self.dias) * n_cal if self.con_calendario else 0)
        ancho, capas, dr = int(p["ancho"]), int(p.get("capas_mlp", 2)), float(p["dropout"])
        m, d = [], d_in
        for _ in range(capas):
            m += [nn.Linear(d, ancho), nn.GELU(), nn.Dropout(dr)]
            d = ancho
        self.tronco = nn.Sequential(*m)
        self.pesos = nn.Linear(ancho, self.K)
        self.theta = nn.Parameter(torch.randn(self.K, H) * 0.1)
        self.w_nivel = nn.Linear(ancho, 1)
        nn.init.zeros_(self.w_nivel.weight)
        nn.init.zeros_(self.w_nivel.bias)

    def perfiles(self) -> torch.Tensor:
        s = nn.functional.softplus(self.theta) + 1e-4
        return s / s.mean(dim=1, keepdim=True)

    def mezcla(self, phi):
        a = self.pesos(phi)
        return torch.softmax(a, dim=-1) if self.simplex else a / self.K + 1.0 / self.K

    def nivel_relativo(self, phi):
        return 1.0 + self.rho_l * torch.tanh(self.w_nivel(phi)) if self.nivel else torch.ones(phi.shape[0], 1, device=phi.device, dtype=phi.dtype)

    def caracteristicas(self, x, cal_p, cal_f):
        B, L = x.shape[:2]
        partes = [x[:, :, 0]]
        if self.con_calendario:
            partes += [cal_f.flatten(1), cal_p.reshape(B, self.dias, L // self.dias, -1).mean(2).flatten(1)]
        return self.tronco(torch.cat(partes, dim=1))

    def forward(self, x, cal_p, cal_f, dia):
        self._validar(x, cal_p, cal_f)
        phi = self.caracteristicas(x, cal_p, cal_f)
        return self.nivel_relativo(phi) * (self.mezcla(phi) @ self.perfiles())

    def envolvente(self) -> tuple[torch.Tensor, torch.Tensor]:
        P = self.perfiles().detach()
        lo, hi = (1 - self.rho_l if self.nivel else 1.0), (1 + self.rho_l if self.nivel else 1.0)
        return lo * P.min(0).values, hi * P.max(0).values


# ============================================================================ fábrica
IDEAS = {"A": ModeloA, "AB": ModeloAB, "AC": ModeloAC, "E": ModeloE, "F": ModeloF}
TEXTO = {"A", "AB", "AC"}


def construir(idea: str, p: dict, n_x: int, n_cal: int, L: int, H: int, texto: dict | None = None, n_mascaras: int = 0) -> nn.Module:
    if idea not in IDEAS:
        raise ValueError(f"Idea desconocida: {idea}")
    if idea in TEXTO:
        return IDEAS[idea](p, n_x, n_cal, L, H, texto)
    if idea == "E":
        return ModeloE(p, n_x, n_cal, L, H, n_mascaras)
    return ModeloF(p, n_x, n_cal, L, H)

from __future__ import annotations

import math

import torch

from torch import nn

D_REL = 64                 # dimensión de la relevancia aprendida (fija, fuera del presupuesto)

PARCHE, PASO = 16, 8       # parches de la historia

PARCHE_H, PASO_H = 8, 4    # parches del horizonte en TGForecaster

D_CTX = 128                # ancho de la vía de contexto de PatchTST (fijo)

DILATACIONES_TCN = (1, 4, 16)

class _GLU(nn.Module):
    """Compuerta lineal: sigmoide(W4 g + b4) * (W5 g + b5)."""

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.lineal = nn.Linear(d_in, 2 * d_out)

    def forward(self, g):
        a, b = self.lineal(g).chunk(2, dim=-1)
        return torch.sigmoid(a) * b

class _CompuertaSuma(nn.Module):
    """LayerNorm(atajo + GLU(dropout(x)))."""

    def __init__(self, d: int, dropout: float):
        super().__init__()
        self.drop, self.glu, self.norma = nn.Dropout(dropout), _GLU(d, d), nn.LayerNorm(d)

    def forward(self, x, atajo):
        return self.norma(atajo + self.glu(self.drop(x)))

class _GRN(nn.Module):
    """Red residual con compuerta del TFT: LayerNorm(a + GLU(W1 ELU(W2 a + b2) + b1))."""

    def __init__(self, d_in: int, d: int, d_out: int, dropout: float):
        super().__init__()
        self.fc2, self.fc1 = nn.Linear(d_in, d), nn.Linear(d, d)
        self.atajo = nn.Linear(d_in, d_out) if d_in != d_out else None
        self.drop, self.glu, self.norma = nn.Dropout(dropout), _GLU(d, d_out), nn.LayerNorm(d_out)

    def forward(self, a):
        h = self.fc1(torch.nn.functional.elu(self.fc2(a)))
        return self.norma((a if self.atajo is None else self.atajo(a)) + self.glu(self.drop(h)))

class _SeleccionVariables(nn.Module):
    """Red de selección de variables: una GRN por variable y pesos softmax por paso de tiempo.

    Las GRN por variable se calculan en paralelo con pesos apilados (V, d, d); equivale a V redes separadas.
    """

    def __init__(self, n_var: int, d: int, dropout: float):
        super().__init__()
        self.n_var, self.d = n_var, d
        self.w_emb, self.b_emb = nn.Parameter(torch.empty(n_var, d)), nn.Parameter(torch.zeros(n_var, d))     # embedding lineal de cada variable continua
        nn.init.normal_(self.w_emb, std=1.0 / math.sqrt(d))
        self.w2, self.b2 = nn.Parameter(torch.empty(n_var, d, d)), nn.Parameter(torch.zeros(n_var, d))
        self.w1, self.b1 = nn.Parameter(torch.empty(n_var, d, d)), nn.Parameter(torch.zeros(n_var, d))
        self.wg, self.bg = nn.Parameter(torch.empty(n_var, d, 2 * d)), nn.Parameter(torch.zeros(n_var, 2 * d))
        for w in (self.w2, self.w1, self.wg):
            nn.init.uniform_(w, -1.0 / math.sqrt(d), 1.0 / math.sqrt(d))
        self.ganancia, self.sesgo = nn.Parameter(torch.ones(n_var, d)), nn.Parameter(torch.zeros(n_var, d))
        self.drop = nn.Dropout(dropout)
        self.pesos = _GRN(n_var * d, d, n_var, dropout)

    def forward(self, x):
        """x: (B, T, V) -> (B, T, d) y pesos de selección (B, T, V)."""
        xi = x.unsqueeze(-1) * self.w_emb + self.b_emb                                   # (B, T, V, d)
        v = torch.softmax(self.pesos(xi.flatten(2)), dim=-1)                             # (B, T, V)
        h = torch.nn.functional.elu(torch.einsum("btvd,vde->btve", xi, self.w2) + self.b2)
        h = self.drop(torch.einsum("btvd,vde->btve", h, self.w1) + self.b1)
        a, b = (torch.einsum("btvd,vde->btve", h, self.wg) + self.bg).chunk(2, dim=-1)
        z = torch.nn.functional.layer_norm(xi + torch.sigmoid(a) * b, (self.d,)) * self.ganancia + self.sesgo
        return (v.unsqueeze(-1) * z).sum(2), v

class _AtencionInterpretable(nn.Module):
    """Atención multicabeza interpretable: las cabezas comparten los valores y se promedian sus salidas."""

    def __init__(self, d: int, cabezas: int, dropout: float):
        super().__init__()
        if d % cabezas:
            raise ValueError("El tamaño oculto debe ser divisible entre el número de cabezas")
        self.h, self.dk = cabezas, d // cabezas
        self.q, self.k = nn.Linear(d, d, bias=False), nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, self.dk, bias=False)                  # valores compartidos por todas las cabezas
        self.salida = nn.Linear(self.dk, d, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, consulta, memoria, mascara):
        B, Tq, _ = consulta.shape
        q = self.q(consulta).view(B, Tq, self.h, self.dk).transpose(1, 2)               # (B, h, Tq, dk)
        k = self.k(memoria).view(B, -1, self.h, self.dk).transpose(1, 2)                # (B, h, Tk, dk)
        a = (q @ k.transpose(-1, -2)) / math.sqrt(self.dk)
        a = self.drop(torch.softmax(a.masked_fill(~mascara, float("-inf")), dim=-1))    # (B, h, Tq, Tk)
        return self.salida(a.mean(1) @ self.v(memoria))

class TFT(nn.Module):
    """Temporal Fusion Transformer (Lim et al., 2021, arXiv:1912.09363) para una serie y pronóstico puntual.

    Entradas pasadas: variables históricas, indicadores de ausencia y calendario de la historia (168 h).
    Entradas futuras conocidas: calendario de las 24 h del horizonte.
    Componentes del artículo: selección de variables, codificador y decodificador LSTM, compuertas con
    suma y normalización, enriquecimiento con GRN, atención multicabeza interpretable con máscara causal
    y GRN por posición. No hay covariables estáticas (una sola serie), así que se omiten sus codificadores.
    """

    def __init__(self, p: dict, n_x: int, n_cal: int, L: int, H: int):
        super().__init__()
        d, cabezas, dropout = int(p["d_model"]), int(p["cabezas"]), float(p["dropout"])
        self.L, self.H = L, H
        self.sel_pasado = _SeleccionVariables(n_x + n_cal, d, dropout)
        self.sel_futuro = _SeleccionVariables(n_cal, d, dropout)
        self.codificador, self.decodificador = nn.LSTM(d, d, batch_first=True), nn.LSTM(d, d, batch_first=True)
        self.puerta_lstm = _CompuertaSuma(d, dropout)
        self.enriquecimiento = _GRN(d, d, d, dropout)
        self.atencion = _AtencionInterpretable(d, cabezas, dropout)
        self.puerta_atencion = _CompuertaSuma(d, dropout)
        self.posicion = _GRN(d, d, d, dropout)
        self.puerta_final = _CompuertaSuma(d, dropout)
        self.salida = nn.Linear(d, 1)
        nn.init.zeros_(self.salida.bias)
        pos = torch.arange(L + H)
        self.register_buffer("mascara", pos[None, :] <= pos[L:, None], persistent=False)    # (H, L + H): cada hora futura ve hasta sí misma

    def forward(self, x, cal_p, cal_f, dia):
        pasado, self.pesos_pasado = self.sel_pasado(torch.cat([x, cal_p], dim=2))            # (B, L, d)
        futuro, self.pesos_futuro = self.sel_futuro(cal_f)                                  # (B, H, d)
        enc, estado = self.codificador(pasado)
        dec, _ = self.decodificador(futuro, estado)
        phi = self.puerta_lstm(torch.cat([enc, dec], dim=1), torch.cat([pasado, futuro], dim=1))   # (B, L + H, d)
        theta = self.enriquecimiento(phi)
        delta = self.puerta_atencion(self.atencion(theta[:, self.L:], theta, self.mascara), theta[:, self.L:])
        psi = self.puerta_final(self.posicion(delta), phi[:, self.L:])
        return 1.0 + self.salida(psi).squeeze(-1)

    def diagnostico(self) -> dict:
        return {}

    def n_parametros(self) -> int:
        return sum(v.numel() for v in self.parameters() if v.requires_grad)

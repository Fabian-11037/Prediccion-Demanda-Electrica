from __future__ import annotations

import math

import torch

from torch import nn

D_REL = 64                 # dimensión de la relevancia aprendida (fija, fuera del presupuesto)

PARCHE, PASO = 16, 8       # parches de la historia

PARCHE_H, PASO_H = 8, 4    # parches del horizonte en TGForecaster

D_CTX = 128                # ancho de la vía de contexto de PatchTST (fijo)

class Relevancia(nn.Module):
    """z = V sum_j alfa_j e_j, alfa = softmax_j <K e_j, q> / sqrt(d), sobre los documentos de [origen - 72 h, origen]."""

    def __init__(self, vectores: torch.Tensor, lista: torch.Tensor, n_x: int, descripcion: torch.Tensor | None = None):
        super().__init__()
        self.register_buffer("vectores", vectores, persistent=False)   # (n_docs, 768), congelados
        self.register_buffer("lista", lista, persistent=False)         # (n_dias, max_docs), -1 = relleno
        d_e = vectores.shape[1]
        self.consulta = nn.Sequential(nn.Linear(2 * n_x, D_REL), nn.GELU(), nn.Linear(D_REL, D_REL))
        self.clave = nn.Linear(d_e, D_REL, bias=False)
        self.valor = nn.Linear(d_e, D_REL, bias=False)
        self.desc = None
        if descripcion is not None:                                     # descripción fija del canal objetivo (TGForecaster)
            self.register_buffer("descripcion", descripcion, persistent=False)
            self.desc = nn.Linear(d_e, D_REL, bias=False)

    def forward(self, x: torch.Tensor, dia: torch.Tensor):
        """x: (B, L, n_x) historia numérica; dia: (B,) día del origen. Devuelve z (B, D_REL), presencia (B,) y pesos."""
        q = self.consulta(torch.cat([x[:, -24:].mean(1), x.mean(1)], dim=1))    # solo historia numérica
        if self.desc is not None:
            q = q + self.desc(self.descripcion)[None, :]
        idx = self.lista[dia]                                                    # (B, M): documentos del origen
        presente = idx >= 0
        e = self.vectores[idx.clamp(min=0)]                                      # (B, M, 768); sin gradiente
        puntaje = torch.einsum("bme,be->bm", e, q @ self.clave.weight) / math.sqrt(D_REL)
        puntaje = puntaje.masked_fill(~presente, float("-inf"))
        hay = presente.any(-1)
        alfa = torch.softmax(puntaje.masked_fill(~hay[:, None], 0.0), dim=-1) * presente
        hay = hay.to(e.dtype)
        z = self.valor(torch.einsum("bm,bme->be", alfa, e)) * hay[:, None]       # cero exacto sin documentos
        return z, hay, alfa

def _parches(s: torch.Tensor) -> torch.Tensor:
    """(N, L) -> (N, 21, PARCHE): se rellena el final con PASO copias del último valor."""
    s = torch.cat([s, s[:, -1:].expand(-1, PASO)], dim=1)
    return s.unfold(1, PARCHE, PASO)

class _CapaPatch(nn.Module):
    """Capa de codificador con normalización por lotes, como en PatchTST."""

    def __init__(self, d, cabezas, dropout):
        super().__init__()
        self.atencion = nn.MultiheadAttention(d, cabezas, dropout=dropout, batch_first=True)
        self.n1, self.n2 = nn.BatchNorm1d(d), nn.BatchNorm1d(d)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * d, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.drop(self.atencion(x, x, x, need_weights=False)[0])
        x = self.n1(x.transpose(1, 2)).transpose(1, 2)
        x = x + self.drop(self.ff(x))
        return self.n2(x.transpose(1, 2)).transpose(1, 2)

class _CapaMezcla(nn.Module):
    """Atención cruzada (consulta: tokens del horizonte; claves y valores: memoria de la serie) y red feed-forward."""

    def __init__(self, d, cabezas, dropout):
        super().__init__()
        self.atencion = nn.MultiheadAttention(d, cabezas, dropout=dropout, batch_first=True)
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * d, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, q, memoria):
        q = self.n1(q + self.drop(self.atencion(q, memoria, memoria, need_weights=False)[0]))
        return self.n2(q + self.drop(self.ff(q)))

class TGForecaster(nn.Module):
    """TGForecaster adaptado a un solo objetivo y a texto disponible en el origen.

    Del artículo: codificador de parches tipo PatchTST sin normalización reversible, descripciones de canal
    codificadas como texto, relevancia de los documentos respecto de la descripción del canal, mezcla por
    atención cruzada con el texto como consulta y la serie como claves y valores, consultas alineadas con
    el horizonte y decodificador por tokens con suma ponderada.
    Adaptaciones: (1) los documentos son los disponibles en [origen - 72 h, origen], no texto del horizonte;
    (2) la consulta de cada parche del horizonte suma la descripción de la demanda, el resumen de documentos,
    el calendario futuro de ese parche y su posición; (3) la memoria contiene los parches de todas las
    variables, identificadas con su descripción, y siete tokens con el calendario diario de la historia,
    porque solo se pronostica la demanda y las covariables deben poder informarla.
    """

    def __init__(self, p: dict, n_x: int, n_cal: int, L: int, H: int, descripciones: torch.Tensor, texto: dict | None = None):
        super().__init__()
        d, capas, cabezas, dropout = int(p["d_model"]), int(p["capas"]), int(p["cabezas"]), float(p["dropout"])
        if descripciones.shape[0] != n_x:
            raise ValueError("Debe haber una descripción por canal de la historia")
        self.register_buffer("descripciones", descripciones, persistent=False)     # (n_x, 768), congeladas; canal 0 = demanda
        self.n_parches = (L + PASO - PARCHE) // PASO + 1
        self.dias, self.H = L // 24, H
        self.emb = nn.Linear(PARCHE, d)
        self.pos = nn.Parameter(torch.zeros(self.n_parches, d))
        nn.init.normal_(self.pos, std=0.02)
        self.capas = nn.ModuleList([_CapaPatch(d, cabezas, dropout) for _ in range(capas)])
        self.drop = nn.Dropout(dropout)
        self.proy_desc = nn.Linear(descripciones.shape[1], d)
        self.cal_dia = nn.Linear(n_cal, d)
        inicios = torch.arange(0, H - PARCHE_H + 1, PASO_H)
        self.n_consultas = len(inicios)
        self.register_buffer("indice_h", inicios[:, None] + torch.arange(PARCHE_H)[None, :], persistent=False)   # (K, PARCHE_H)
        cubre = torch.zeros(self.n_consultas, H, dtype=torch.bool)
        cubre.scatter_(1, self.indice_h, True)
        self.register_buffer("cubre", cubre, persistent=False)
        self.cal_q = nn.Linear(PARCHE_H * n_cal, d)
        self.pos_q = nn.Parameter(torch.zeros(self.n_consultas, d))
        nn.init.normal_(self.pos_q, std=0.02)
        self.relevancia = None
        if texto is not None:
            self.relevancia = Relevancia(texto["vectores"], texto["lista"], n_x, descripciones[0])
            self.proy_txt = nn.Linear(D_REL, d)
        self.mezcla = nn.ModuleList([_CapaMezcla(d, cabezas, dropout) for _ in range(int(p["capas_mezcla"]))])
        self.salida = nn.Linear(d, PARCHE_H)
        nn.init.zeros_(self.salida.bias)
        self.peso = nn.Parameter(torch.zeros(self.n_consultas, H))               # suma ponderada de los parches solapados

    def texto_del_origen(self, x, dia):
        return self.relevancia(x, dia)

    def pesos_horizonte(self) -> torch.Tensor:
        """(K, H): para cada hora, pesos que suman 1 entre los parches que la cubren."""
        return torch.softmax(self.peso.masked_fill(~self.cubre, float("-inf")), dim=0)

    def forward(self, x, cal_p, cal_f, dia):
        B, L, C = x.shape
        z = self.drop(self.emb(_parches(x.transpose(1, 2).reshape(B * C, L))) + self.pos)
        for capa in self.capas:
            z = capa(z)
        desc = self.proy_desc(self.descripciones)                                              # (C, d)
        memoria = (z.reshape(B, C, self.n_parches, -1) + desc[None, :, None, :]).reshape(B, C * self.n_parches, -1)
        dias = self.cal_dia(cal_p.reshape(B, self.dias, L // self.dias, -1).mean(2))           # (B, 7, d)
        memoria = torch.cat([memoria, dias], dim=1)
        q = desc[0][None, None, :] + self.cal_q(cal_f[:, self.indice_h].flatten(2)) + self.pos_q   # (B, K, d)
        if self.relevancia is not None:
            zt, hay, _ = self.relevancia(x, dia)
            q = q + (self.proy_txt(zt) * hay[:, None])[:, None, :]                             # cero exacto sin documentos; el sesgo marca presencia
        q = self.drop(q)
        for capa in self.mezcla:
            q = capa(q, memoria)
        partes = self.salida(q)                                                                # (B, K, PARCHE_H)
        lienzo = torch.zeros(B, self.n_consultas, self.H, dtype=partes.dtype, device=partes.device)
        lienzo = lienzo.scatter(2, self.indice_h[None].expand(B, -1, -1), partes)
        return 1.0 + (lienzo * self.pesos_horizonte()[None]).sum(1)

    def diagnostico(self) -> dict:
        return {}

    def n_parametros(self) -> int:
        return sum(v.numel() for v in self.parameters() if v.requires_grad)

from __future__ import annotations

import math

import torch

from torch import nn

D_REL = 64      # dimensión de la relevancia aprendida (fija, fuera del presupuesto)

PARCHE, PASO = 16, 8

D_CTX = 128     # ancho de la vía de contexto de PatchTST (fijo, fuera del presupuesto)

class AgregadorTexto(nn.Module):
    """z_d = sum_j alfa_j V e_j, alfa = softmax_j <K e_j, q> / sqrt(d); después MLP de tres capas -> d_texto."""

    def __init__(self, vectores: torch.Tensor, lista: torch.Tensor, n_x: int, d_texto: int, dropout: float = 0.1):
        super().__init__()
        self.register_buffer("vectores", vectores, persistent=False)   # (n_docs, 768), congelados
        self.register_buffer("lista", lista, persistent=False)         # (n_dias, max_docs), -1 = relleno
        d_e = vectores.shape[1]
        self.consulta = nn.Sequential(nn.Linear(2 * n_x, D_REL), nn.GELU(), nn.Linear(D_REL, D_REL))
        self.clave = nn.Linear(d_e, D_REL, bias=False)
        self.valor = nn.Linear(d_e, D_REL, bias=False)
        self.mlp = nn.Sequential(nn.Linear(D_REL, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 32), nn.GELU(), nn.Linear(32, d_texto))
        self.d_texto = d_texto
        self.placebo = None  # control opcional: sustituye los vectores por ruido con igual máscara

    def forward(self, x: torch.Tensor, dia: torch.Tensor, dias_hist: int = 7):
        """x: (B, L, n_x) historia numérica; dia: (B,) día del origen. Devuelve (B, L, d_texto + 1) y los pesos."""
        B, L, _ = x.shape
        q = self.consulta(torch.cat([x[:, -24:].mean(1), x.mean(1)], dim=1))                 # (B, D_REL): solo historia numérica
        dias = dia[:, None] + torch.arange(-(dias_hist - 1), 1, device=dia.device)[None, :]  # día que cierra cada bloque de 24 h
        idx = self.lista[dias.clamp(min=0)]                                                   # (B, 7, M)
        presente = (idx >= 0) & (dias >= 0)[:, :, None]
        fuente = self.vectores if self.placebo is None else self.placebo
        e = fuente[idx.clamp(min=0)]                                                          # (B, 7, M, 768); sin gradiente
        # <K e_j, q> = e_j . (K^T q): se proyecta la consulta una vez en lugar de proyectar cada documento.
        puntaje = torch.einsum("bkme,be->bkm", e, q @ self.clave.weight) / math.sqrt(D_REL)
        puntaje = puntaje.masked_fill(~presente, float("-inf"))
        hay = presente.any(-1)                                                                # (B, 7)
        alfa = torch.softmax(puntaje.masked_fill(~hay[:, :, None], 0.0), dim=-1) * presente
        z = self.valor(torch.einsum("bkm,bkme->bke", alfa, e))                                # V lineal: sum_j alfa_j V e_j = V sum_j alfa_j e_j
        canales = self.mlp(z) * hay.unsqueeze(-1)                                             # cero exacto sin documentos
        salida = torch.cat([canales, hay.unsqueeze(-1).to(canales.dtype)], dim=-1)            # + indicador de presencia
        return salida.repeat_interleave(L // dias_hist, dim=1), alfa

class NucleoITransformer(nn.Module):
    def __init__(self, n_hist: int, n_cal: int, n_txt: int, L: int, H: int, p: dict):
        super().__init__()
        d, capas, cabezas, dropout = int(p["d_model"]), int(p["capas"]), int(p["cabezas"]), float(p["dropout"])
        self.emb_hist, self.emb_cal = nn.Linear(L, d), nn.Linear(L + H, d)
        self.emb_txt = nn.Linear(L, d) if n_txt else None
        self.identidad = nn.Parameter(torch.zeros(n_hist + n_cal + n_txt, d))
        nn.init.normal_(self.identidad, std=0.02)
        capa = nn.TransformerEncoderLayer(d, cabezas, 2 * d, dropout, activation="gelu", batch_first=True, norm_first=False)
        self.encoder = nn.TransformerEncoder(capa, capas, norm=nn.LayerNorm(d), enable_nested_tensor=False)
        self.drop = nn.Dropout(dropout)
        self.salida = nn.Linear(d, H)
        nn.init.zeros_(self.salida.bias)

    def forward(self, x, cal_p, cal_f, txt=None):
        fichas = [self.emb_hist(x.transpose(1, 2)), self.emb_cal(torch.cat([cal_p, cal_f], dim=1).transpose(1, 2))]
        if self.emb_txt is not None:
            fichas.append(self.emb_txt(txt.transpose(1, 2)))
        h = self.encoder(self.drop(torch.cat(fichas, dim=1) + self.identidad))
        return 1.0 + self.salida(h[:, 0])           # ficha 0 = demanda

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

class NucleoPatchTST(nn.Module):
    """Codificador independiente por canal (históricas, indicadores de ausencia y texto) y cabeza de dos vías.

    Vía lineal: la cabeza aplanada del artículo sobre los parches de la demanda.
    Vía de contexto: un MLP sobre el resumen de la demanda, el resumen de cada uno de los demás canales,
    el calendario futuro y el promedio diario del calendario de la historia. Sin esta vía las covariables
    y el texto no llegarían a la demanda, y el efecto del calendario quedaría limitado a una función lineal.
    El calendario es determinista y no pasa por el codificador.
    """

    def __init__(self, n_hist: int, n_cal: int, n_txt: int, L: int, H: int, p: dict):
        super().__init__()
        d, capas, cabezas, dropout = int(p["d_model"]), int(p["capas"]), int(p["cabezas"]), float(p["dropout"])
        self.n_parches = (L + PASO - PARCHE) // PASO + 1          # 21 con L = 168: se rellena el final con PASO copias
        self.canales = n_hist + n_txt
        self.dias = L // 24
        self.emb = nn.Linear(PARCHE, d)
        self.pos = nn.Parameter(torch.zeros(self.n_parches, d))
        nn.init.normal_(self.pos, std=0.02)
        self.capas = nn.ModuleList([_CapaPatch(d, cabezas, dropout) for _ in range(capas)])
        self.drop = nn.Dropout(dropout)
        self.lineal = nn.Linear(self.n_parches * d, H)
        self.contexto = nn.Sequential(nn.Linear(self.canales * d + H * n_cal + self.dias * n_cal, D_CTX), nn.GELU(), nn.Dropout(dropout), nn.Linear(D_CTX, H))
        nn.init.zeros_(self.lineal.bias)
        nn.init.zeros_(self.contexto[-1].bias)

    def forward(self, x, cal_p, cal_f, txt=None):
        s = torch.cat([x, txt], dim=2) if txt is not None else x                    # (B, L, C)
        B, L, Cn = s.shape
        s = s.transpose(1, 2).reshape(B * Cn, L)
        s = torch.cat([s, s[:, -1:].expand(-1, PASO)], dim=1)                       # relleno con el último valor
        z = self.drop(self.emb(s.unfold(1, PARCHE, PASO)) + self.pos)               # (B*C, 21, d)
        for capa in self.capas:
            z = capa(z)
        z = z.reshape(B, Cn, self.n_parches, -1)
        cal_dia = cal_p.reshape(B, self.dias, L // self.dias, -1).mean(2)           # (B, 7, n_cal)
        ctx = torch.cat([z.mean(2).flatten(1), cal_f.flatten(1), cal_dia.flatten(1)], dim=1)   # canal 0 = demanda
        return 1.0 + self.lineal(z[:, 0].flatten(1)) + self.contexto(ctx)

class Modelo6(nn.Module):
    """Variante N (sin texto) o N+O (con canal oficial) sobre el núcleo elegido."""

    def __init__(self, p: dict, n_hist: int, n_cal: int, L: int, H: int, texto: dict | None = None):
        super().__init__()
        self.agregador = None
        n_txt = 0
        if texto is not None:
            self.agregador = AgregadorTexto(texto["vectores"], texto["lista"], n_hist, int(p["d_texto"]))
            n_txt = int(p["d_texto"]) + 1
        clase = {"itransformer": NucleoITransformer, "patchtst": NucleoPatchTST}[p["nucleo"]]
        self.nucleo = clase(n_hist, n_cal, n_txt, L, H, p)

    def forward(self, x, cal_p, cal_f, dia):
        txt = self.agregador(x, dia)[0] if self.agregador is not None else None
        return self.nucleo(x, cal_p, cal_f, txt)

    def n_parametros(self) -> int:
        return sum(v.numel() for v in self.parameters() if v.requires_grad)

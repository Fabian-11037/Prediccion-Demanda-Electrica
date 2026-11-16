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
    """Codificador independiente por canal y cabeza de dos vías (lineal sobre la demanda + MLP de contexto)."""

    def __init__(self, n_hist: int, n_cal: int, n_txt: int, L: int, H: int, p: dict):
        super().__init__()
        d, capas, cabezas, dropout = int(p["d_model"]), int(p["capas"]), int(p["cabezas"]), float(p["dropout"])
        self.n_parches = (L + PASO - PARCHE) // PASO + 1
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
        s = torch.cat([x, txt], dim=2) if txt is not None else x
        B, L, Cn = s.shape
        z = self.drop(self.emb(_parches(s.transpose(1, 2).reshape(B * Cn, L))) + self.pos)
        for capa in self.capas:
            z = capa(z)
        z = z.reshape(B, Cn, self.n_parches, -1)
        cal_dia = cal_p.reshape(B, self.dias, L // self.dias, -1).mean(2)
        ctx = torch.cat([z.mean(2).flatten(1), cal_f.flatten(1), cal_dia.flatten(1)], dim=1)
        return 1.0 + self.lineal(z[:, 0].flatten(1)) + self.contexto(ctx)

class FusionTardia(nn.Module):
    """y = y_num + alfa * m * y_txt.

    y_num: pronóstico del núcleo numérico (iTransformer o PatchTST), que no recibe texto.
    y_txt: pronóstico de 24 h de la rama de texto, un MLP sobre el resumen de documentos del origen.
    alfa:  peso escalar aprendido (inicio 0,5); m vale 0 cuando no hay documentos.
    Las dos ramas se entrenan juntas. La última capa de la rama de texto empieza en cero, de modo que
    el modelo parte exactamente del pronóstico numérico. Sin texto (variante N) es el núcleo solo.
    """

    def __init__(self, p: dict, n_x: int, n_cal: int, L: int, H: int, texto: dict | None = None):
        super().__init__()
        clase = {"itransformer": NucleoITransformer, "patchtst": NucleoPatchTST}[p["nucleo"]]
        self.nucleo = clase(n_x, n_cal, 0, L, H, p)
        self.relevancia = None
        if texto is not None:
            ancho = int(p["d_fusion"])
            self.relevancia = Relevancia(texto["vectores"], texto["lista"], n_x)
            self.rama = nn.Sequential(nn.Linear(D_REL, ancho), nn.GELU(), nn.Dropout(float(p["dropout"])), nn.Linear(ancho, H))
            nn.init.zeros_(self.rama[-1].weight)
            nn.init.zeros_(self.rama[-1].bias)
            self.alfa = nn.Parameter(torch.tensor(0.5))

    def texto_del_origen(self, x, dia):
        return self.relevancia(x, dia)

    def aporte_texto(self, x, dia):
        z, hay, _ = self.relevancia(x, dia)
        return self.alfa * self.rama(z) * hay[:, None]

    def forward(self, x, cal_p, cal_f, dia):
        y = self.nucleo(x, cal_p, cal_f, None)
        return y if self.relevancia is None else y + self.aporte_texto(x, dia)

    def diagnostico(self) -> dict:
        return {} if self.relevancia is None else {"alfa": float(self.alfa.detach())}

    def n_parametros(self) -> int:
        return sum(v.numel() for v in self.parameters() if v.requires_grad)

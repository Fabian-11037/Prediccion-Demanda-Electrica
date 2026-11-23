"""Redes de los paquetes de entrenamiento (numéricas y con texto).

Piezas comunes:
  * ``Relevancia``: resume los documentos oficiales elegibles en el origen con pesos aprendidos; la
    consulta depende solo de la historia numérica (y, en TGForecaster, de la descripción fija del canal).
    Sin documentos devuelve cero y el indicador de presencia en cero.
  * ``NucleoITransformer`` y ``NucleoPatchTST``: los mismos núcleos numéricos del modelo 6.

Modelos:
  * ``FusionTardia``: pronóstico numérico + peso aprendido x pronóstico de la rama de texto.
  * ``TGForecaster``: codificador por parches, consultas del horizonte guiadas por texto, mezcla por
    atención cruzada y decodificador por tokens con suma ponderada (Xu et al., arXiv:2405.13522),
    adaptado a documentos disponibles en el origen.
Todas las redes devuelven 1 + corrección: la etiqueta es demanda / nivel de la ventana.
"""
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


def _parches(s: torch.Tensor) -> torch.Tensor:
    """(N, L) -> (N, 21, PARCHE): se rellena el final con PASO copias del último valor.

    Equivale exactamente a ``s.unfold(1, PARCHE, PASO)`` (lo verifica tests/test_original_model.py), pero con un índice
    fijo, para que la exportación a ONNX admita un número variable de orígenes (Entregable 3, §5.2)."""
    s = torch.cat([s, s[:, -1:].repeat(1, PASO)], dim=1)
    n = (s.shape[1] - PARCHE) // PASO + 1
    idx = (torch.arange(n, device=s.device)[:, None] * PASO + torch.arange(PARCHE, device=s.device)[None, :]).reshape(-1)
    return s.index_select(1, idx).reshape(s.shape[0], n, PARCHE)


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


class NumericoITransformer(nn.Module):
    """iTransformer numérico (Liu et al., ICLR 2024): un token por variable y atención entre variables."""

    def __init__(self, p: dict, n_x: int, n_cal: int, L: int, H: int):
        super().__init__()
        self.nucleo = NucleoITransformer(n_x, n_cal, 0, L, H, p)

    def forward(self, x, cal_p, cal_f, dia):
        return self.nucleo(x, cal_p, cal_f, None)

    def diagnostico(self) -> dict:
        return {}

    def n_parametros(self) -> int:
        return sum(v.numel() for v in self.parameters() if v.requires_grad)


DILATACIONES_TCN = (1, 4, 16)


def campo_receptivo_tcn(kernel: int) -> int:
    """Dos convoluciones causales por bloque: R = 1 + 2 (k - 1) (1 + 4 + 16)."""
    return 1 + 2 * (int(kernel) - 1) * sum(DILATACIONES_TCN)


class _BloqueTCN(nn.Module):
    """Bloque residual de Bai et al. (2018): dos convoluciones causales dilatadas con normalización de pesos."""

    def __init__(self, c_in: int, c_out: int, kernel: int, dilatacion: int, dropout: float):
        super().__init__()
        self.recorte = (kernel - 1) * dilatacion
        convs = [nn.Conv1d(c_in, c_out, kernel, padding=self.recorte, dilation=dilatacion), nn.Conv1d(c_out, c_out, kernel, padding=self.recorte, dilation=dilatacion)]
        for c in convs:
            nn.init.normal_(c.weight, 0.0, 0.01)
        self.c1, self.c2 = (nn.utils.parametrizations.weight_norm(c) for c in convs)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()       # al final del bloque
        self.atajo = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else None

    def forward(self, x):
        y = torch.relu(self.c1(x)[..., :-self.recorte])       # el recorte quita el relleno derecho: no se mira adelante
        y = self.drop(torch.relu(self.c2(y)[..., :-self.recorte]))
        return torch.relu(y + (x if self.atajo is None else self.atajo(x)))


class TCN(nn.Module):
    """TCN de tres bloques con dilataciones (1, 4, 16) y cabeza con el calendario futuro conocido.

    Entrada de las convoluciones: variables históricas, indicadores de ausencia y calendario de la historia.
    Se toma la representación de la última hora de la historia, se une con una proyección del calendario
    futuro y una cabeza densa produce las 24 horas (misma convención que las redes base del proyecto).
    """

    def __init__(self, p: dict, n_x: int, n_cal: int, L: int, H: int):
        super().__init__()
        anchos, patron, k, tasa = [int(a) for a in p["anchos"]], [int(b) for b in p["patron_dropout"]], int(p["kernel"]), float(p["dropout"])
        if len(anchos) != 3 or len(patron) != 3:
            raise ValueError("TCN usa tres bloques")
        if campo_receptivo_tcn(k) < L:
            raise ValueError(f"El campo receptivo ({campo_receptivo_tcn(k)}) no cubre la historia ({L})")
        canales = [n_x + n_cal] + anchos
        self.bloques = nn.ModuleList([_BloqueTCN(canales[i], canales[i + 1], k, DILATACIONES_TCN[i], tasa if patron[i] else 0.0) for i in range(3)])
        w = anchos[-1]
        self.cal = nn.Linear(H * n_cal, w)
        self.cabeza = nn.Sequential(nn.Linear(2 * w, w), nn.ReLU(), nn.Linear(w, H))
        nn.init.zeros_(self.cabeza[-1].bias)

    def forward(self, x, cal_p, cal_f, dia):
        s = torch.cat([x, cal_p], dim=2).transpose(1, 2)        # (B, canales, L)
        for bloque in self.bloques:
            s = bloque(s)
        z = torch.cat([s[:, :, -1], torch.relu(self.cal(cal_f.flatten(1)))], dim=1)
        return 1.0 + self.cabeza(z)

    def diagnostico(self) -> dict:
        return {}

    def n_parametros(self) -> int:
        return sum(v.numel() for v in self.parameters() if v.requires_grad)


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


def construir(red: str, p: dict, n_x: int, n_cal: int, L: int, H: int, texto: dict | None = None, descripciones: torch.Tensor | None = None):
    if red == "fusion_tardia":
        return FusionTardia(p, n_x, n_cal, L, H, texto)
    if red == "tgforecaster":
        return TGForecaster(p, n_x, n_cal, L, H, descripciones, texto)
    if red == "itransformer":
        return NumericoITransformer(p, n_x, n_cal, L, H)
    if red == "tcn":
        return TCN(p, n_x, n_cal, L, H)
    if red == "tft":
        return TFT(p, n_x, n_cal, L, H)
    raise ValueError(f"Red desconocida: {red}")

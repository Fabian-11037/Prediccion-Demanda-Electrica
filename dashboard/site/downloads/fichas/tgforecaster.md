# 10.12. TGForecaster: consultas del horizonte guiadas por texto

La notación y las operaciones compartidas definen los tensores, el nivel de demanda, la atención, las activaciones y la optimización.

## Arquitectura y fundamentos

TGForecaster utiliza descripciones textuales de los canales para orientar la relación entre información documental y señales numéricas. La atención cruzada distingue qué se pregunta —tokens del horizonte— de dónde se busca la respuesta —la memoria de la historia—. La variante N conserva las descripciones estáticas de canales y no incorpora documentos que cambien por origen; N+O añade el contexto documental oficial disponible.

Cada uno de los 33 canales históricos se divide en 21 parches de 16 horas con paso ocho. Sus embeddings se codifican y se suman a una proyección de la descripción E5 del canal; siete tokens adicionales representan el calendario diario pasado. La memoria reúne así $33\times21+7=700$ tokens. No se aplica RevIN dentro de este codificador; la normalización externa del contrato permanece.

El horizonte de 24 horas se divide en cinco parches de ocho con paso cuatro. La consulta de cada parche combina descripción de demanda, calendario futuro conocido, posición y resumen de documentos oficiales disponibles en el origen:

$$
q_k=W_DE_{dem}+W_FF_k+p_k+m_tW_Tz_t,\quad
H=\operatorname{softmax}\!\left(\frac{QK_M^\top}{\sqrt{d_k}}\right)V_M.
$$ (eq-arquitecturas-tgforecaster-001)

El texto participa como parte de la consulta; las claves y valores proceden de la memoria numérica. Cada token decodifica ocho correcciones. Para las horas cubiertas por varios parches se aprende una combinación convexa: $\omega_{k,h}=\operatorname{softmax}_{k:h\in k}(a_{k,h})$, de modo que $\sum_k\omega_{k,h}=1$. La salida es $\ell_t[1+\sum_k\omega_{k,h}o_{k,h}]$.

El paquete adapta el artículo a una sola serie objetivo y a documentos disponibles al emitir el pronóstico; no incorpora texto realizado durante el horizonte. Las descripciones y los embeddings son extractores congelados; la selección documental y la fusión son entrenables. La variante N selecciona dimensión256, tres capas temporales, dos de mezcla y ocho cabezas; N+O selecciona dimensión64, dos capas temporales, una de mezcla y cuatro cabezas. Sus configuraciones distintas hacen que el contraste estime el procedimiento completo de selección, además de la incorporación de documentos.

### Esquema global

[![Arquitectura global de tgforecaster](../_static/arquitecturas_presentacion/tgforecaster_global.png)](../_static/arquitecturas_presentacion/tgforecaster_global.png)

La historia se divide en parches y se combina con descripciones estáticas de las 33 variables. El calendario pasado completa una memoria de 700 tokens. Cinco consultas incorporan el calendario del horizonte, la descripción de demanda y el resumen documental; la atención cruzada genera parches de pronóstico superpuestos.

### Detalle de las operaciones

[![Detalle de la arquitectura de tgforecaster](../_static/arquitecturas_presentacion/tgforecaster_detalle.png)](../_static/arquitecturas_presentacion/tgforecaster_detalle.png)

La identidad de cada variable se proyecta sobre sus parches históricos. Las consultas del horizonte atienden esa memoria y producen cinco parches de ocho horas, desplazados cuatro horas. Pesos aprendidos, normalizados entre los parches que cubren cada hora, reconstruyen las 24 salidas. La variante numérica conserva las descripciones estáticas y omite los documentos.

El código de exportación de los esquemas forma parte del preprocesamiento compartido.

### Código de la arquitectura

```python
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
```

<!-- diseno-entrenamiento:inicio -->

## Diseño de entrenamiento y prueba


### TGForecaster · N

Se evaluaron **20 configuraciones** y 5 llegaron a la comparación de selección. La **configuración 19** obtuvo el menor MASE de validación (0,7047) sobre 2018–2021. La configuración fijada para prueba es **19**, con 5 semillas.

#### Configuración conservada
| Parámetro | Valor |
|---|---|
| Dimensión interna | 256 |
| Capas | 3 |
| Capas de mezcla | 2 |
| Cabezas | 8 |
| Dropout | 0,1 |
| Tasa de aprendizaje | 0,0001 |
| Decaimiento de pesos | 0,0001 |
| Lote de ajuste | 32 |

#### Etapas del entrenamiento y prueba

| Etapa | Especificación |
|---|---|
| Filtro inicial | 20 configuraciones; semilla 43; validación 2018 |
| Confirmación y criterio | Cinco finalistas; semilla 43; evaluación adicional 2019–2021; media de los cuatro años 2018–2021. Menor MASE medio de validación; cada año tiene el mismo peso. |
| Ajuste final | 2000-01-09 00:00:00-05:00 a 2019-10-20 00:00:00-05:00 |
| Parada interna final | 2019-10-21 00:00:00-05:00 a 2021-12-31 00:00:00-05:00 |
| Prueba | 2022-01-01 a 2025-12-31; 1.461 orígenes diarios |
| Semillas finales | 1000, 1001, 1002, 1003, 1004 |
| Historia y horizonte | 168 y 24 horas, respectivamente |
| Pérdida y optimizador | MAE de demanda dividida por el nivel histórico; AdamW |
| Épocas y parada | Máximo 200; mínimo 10; paciencia efectiva 6; restauración del menor valor interno |
| Tasa de aprendizaje | ReduceLROnPlateau: factor 0,5; paciencia 2 |
| Lotes | Ajuste: 32; evaluación de prueba: hasta 256 orígenes por lote |
| Épocas ejecutadas | 36–48; épocas de los checkpoints: 40, 30, 41, 36, 42 |


#### Configuraciones evaluadas en desarrollo
El filtro y la selección corresponden a periodos y cantidades de repeticiones distintos. El guion identifica las configuraciones que no avanzaron a confirmación; no representa un resultado nulo.
| Configuración | Hiperparámetros | MASE del filtro | MASE de validación | Estado |
|---|---|---:|---:|---|
| **19** | Dimensión interna: 256; Capas: 3; Capas de mezcla: 2; Cabezas: 8; Dropout: 0,1; Tasa de aprendizaje: 0,0001; Decaimiento de pesos: 0,0001; Lote de ajuste: 32 | 0,4628 | 0,7047 | **Seleccionada para prueba** |
| 1 | Dimensión interna: 64; Capas: 2; Capas de mezcla: 1; Cabezas: 4; Dropout: 0,1; Tasa de aprendizaje: 0,001; Decaimiento de pesos: 0,0001; Lote de ajuste: 64 | 0,5293 | 0,7261 | Finalista |
| 12 | Dimensión interna: 128; Capas: 3; Capas de mezcla: 3; Cabezas: 8; Dropout: 0,3; Tasa de aprendizaje: 0,001; Decaimiento de pesos: 0,01; Lote de ajuste: 128 | 0,7403 | 0,9152 | Finalista |
| 15 | Dimensión interna: 32; Capas: 3; Capas de mezcla: 1; Cabezas: 4; Dropout: 0,1; Tasa de aprendizaje: 0,001; Decaimiento de pesos: 0,0001; Lote de ajuste: 64 | 0,7349 | 0,9257 | Finalista |
| 7 | Dimensión interna: 128; Capas: 2; Capas de mezcla: 1; Cabezas: 8; Dropout: 0,1; Tasa de aprendizaje: 0,0003; Decaimiento de pesos: 0,0001; Lote de ajuste: 64 | 0,6672 | 0,9331 | Finalista |
| 2 | Dimensión interna: 64; Capas: 2; Capas de mezcla: 2; Cabezas: 4; Dropout: 0,2; Tasa de aprendizaje: 0,001; Decaimiento de pesos: 0,0001; Lote de ajuste: 64 | 0,7680 | — | Filtro inicial |
| 10 | Dimensión interna: 128; Capas: 3; Capas de mezcla: 2; Cabezas: 8; Dropout: 0,1; Tasa de aprendizaje: 0,0001; Decaimiento de pesos: 0,0001; Lote de ajuste: 32 | 0,8162 | — | Filtro inicial |
| 5 | Dimensión interna: 64; Capas: 3; Capas de mezcla: 3; Cabezas: 4; Dropout: 0,3; Tasa de aprendizaje: 0,001; Decaimiento de pesos: 0,001; Lote de ajuste: 128 | 0,8367 | — | Filtro inicial |
| 9 | Dimensión interna: 128; Capas: 3; Capas de mezcla: 1; Cabezas: 8; Dropout: 0,2; Tasa de aprendizaje: 0,001; Decaimiento de pesos: 0,001; Lote de ajuste: 128 | 0,8498 | — | Filtro inicial |
| 16 | Dimensión interna: 32; Capas: 3; Capas de mezcla: 2; Cabezas: 4; Dropout: 0,2; Tasa de aprendizaje: 0,001; Decaimiento de pesos: 0,001; Lote de ajuste: 64 | 0,8522 | — | Filtro inicial |
| 6 | Dimensión interna: 64; Capas: 2; Capas de mezcla: 3; Cabezas: 4; Dropout: 0,1; Tasa de aprendizaje: 0,0001; Decaimiento de pesos: 0,0001; Lote de ajuste: 32 | 0,8524 | — | Filtro inicial |
| 11 | Dimensión interna: 128; Capas: 3; Capas de mezcla: 3; Cabezas: 8; Dropout: 0,2; Tasa de aprendizaje: 0,0003; Decaimiento de pesos: 0,001; Lote de ajuste: 64 | 0,8664 | — | Filtro inicial |
| 18 | Dimensión interna: 256; Capas: 2; Capas de mezcla: 1; Cabezas: 8; Dropout: 0,2; Tasa de aprendizaje: 0,0001; Decaimiento de pesos: 0,001; Lote de ajuste: 32 | 0,9454 | — | Filtro inicial |
| 8 | Dimensión interna: 128; Capas: 2; Capas de mezcla: 2; Cabezas: 8; Dropout: 0,2; Tasa de aprendizaje: 0,0003; Decaimiento de pesos: 0,0001; Lote de ajuste: 64 | 0,9699 | — | Filtro inicial |
| 4 | Dimensión interna: 64; Capas: 3; Capas de mezcla: 2; Cabezas: 4; Dropout: 0,1; Tasa de aprendizaje: 0,0003; Decaimiento de pesos: 0,001; Lote de ajuste: 128 | 1,0103 | — | Filtro inicial |
| 20 | Dimensión interna: 256; Capas: 2; Capas de mezcla: 2; Cabezas: 8; Dropout: 0,3; Tasa de aprendizaje: 0,0003; Decaimiento de pesos: 0,01; Lote de ajuste: 128 | 1,2870 | — | Filtro inicial |
| 13 | Dimensión interna: 128; Capas: 2; Capas de mezcla: 3; Cabezas: 16; Dropout: 0,2; Tasa de aprendizaje: 0,0001; Decaimiento de pesos: 0,0001; Lote de ajuste: 128 | 1,5243 | — | Filtro inicial |
| 3 | Dimensión interna: 64; Capas: 3; Capas de mezcla: 1; Cabezas: 4; Dropout: 0,2; Tasa de aprendizaje: 0,0003; Decaimiento de pesos: 0,0001; Lote de ajuste: 64 | 1,5583 | — | Filtro inicial |
| 14 | Dimensión interna: 128; Capas: 3; Capas de mezcla: 2; Cabezas: 16; Dropout: 0,3; Tasa de aprendizaje: 0,0003; Decaimiento de pesos: 0,0001; Lote de ajuste: 64 | 1,6427 | — | Filtro inicial |
| 17 | Dimensión interna: 32; Capas: 2; Capas de mezcla: 3; Cabezas: 4; Dropout: 0,3; Tasa de aprendizaje: 0,0003; Decaimiento de pesos: 0,0001; Lote de ajuste: 128 | 1,6490 | — | Filtro inicial |

[Resultados de desarrollo por año y semilla](../_static/resultados/diseno_entrenamiento/adj_modelo7_20261114_0430__N.csv) · [Configuraciones completas](../_static/resultados/diseno_entrenamiento/adj_modelo7_20261114_0430__N_configuraciones.csv).



### TGForecaster · N+O

Se evaluaron **20 configuraciones** y 5 llegaron a la comparación de selección. La **configuración 1** obtuvo el menor MASE de validación (0,9126) sobre 2018–2021. La configuración fijada para prueba es **1**, con 5 semillas.

#### Configuración conservada
| Parámetro | Valor |
|---|---|
| Dimensión interna | 64 |
| Capas | 2 |
| Capas de mezcla | 1 |
| Cabezas | 4 |
| Dropout | 0,1 |
| Tasa de aprendizaje | 0,001 |
| Decaimiento de pesos | 0,0001 |
| Lote de ajuste | 64 |

#### Etapas del entrenamiento y prueba

| Etapa | Especificación |
|---|---|
| Filtro inicial | 20 configuraciones; semilla 43; validación 2018 |
| Confirmación y criterio | Cinco finalistas; semilla 43; evaluación adicional 2019–2021; media de los cuatro años 2018–2021. Menor MASE medio de validación; cada año tiene el mismo peso. |
| Ajuste final | 2000-01-09 00:00:00-05:00 a 2019-10-20 00:00:00-05:00 |
| Parada interna final | 2019-10-21 00:00:00-05:00 a 2021-12-31 00:00:00-05:00 |
| Prueba | 2022-01-01 a 2025-12-31; 1.461 orígenes diarios |
| Semillas finales | 1000, 1001, 1002, 1003, 1004 |
| Historia y horizonte | 168 y 24 horas, respectivamente |
| Pérdida y optimizador | MAE de demanda dividida por el nivel histórico; AdamW |
| Épocas y parada | Máximo 200; mínimo 10; paciencia efectiva 6; restauración del menor valor interno |
| Tasa de aprendizaje | ReduceLROnPlateau: factor 0,5; paciencia 2 |
| Lotes | Ajuste: 64; evaluación de prueba: hasta 256 orígenes por lote |
| Épocas ejecutadas | 23–45; épocas de los checkpoints: 29, 20, 25, 39, 17 |


#### Configuraciones evaluadas en desarrollo
El filtro y la selección corresponden a periodos y cantidades de repeticiones distintos. El guion identifica las configuraciones que no avanzaron a confirmación; no representa un resultado nulo.
| Configuración | Hiperparámetros | MASE del filtro | MASE de validación | Estado |
|---|---|---:|---:|---|
| **1** | Dimensión interna: 64; Capas: 2; Capas de mezcla: 1; Cabezas: 4; Dropout: 0,1; Tasa de aprendizaje: 0,001; Decaimiento de pesos: 0,0001; Lote de ajuste: 64 | 0,7176 | 0,9126 | **Seleccionada para prueba** |
| 19 | Dimensión interna: 256; Capas: 3; Capas de mezcla: 2; Cabezas: 8; Dropout: 0,1; Tasa de aprendizaje: 0,0001; Decaimiento de pesos: 0,0001; Lote de ajuste: 32 | 0,7240 | 0,9179 | Finalista |
| 12 | Dimensión interna: 128; Capas: 3; Capas de mezcla: 3; Cabezas: 8; Dropout: 0,3; Tasa de aprendizaje: 0,001; Decaimiento de pesos: 0,01; Lote de ajuste: 128 | 0,7703 | 1,0030 | Finalista |
| 2 | Dimensión interna: 64; Capas: 2; Capas de mezcla: 2; Cabezas: 4; Dropout: 0,2; Tasa de aprendizaje: 0,001; Decaimiento de pesos: 0,0001; Lote de ajuste: 64 | 0,7562 | 1,0456 | Finalista |
| 4 | Dimensión interna: 64; Capas: 3; Capas de mezcla: 2; Cabezas: 4; Dropout: 0,1; Tasa de aprendizaje: 0,0003; Decaimiento de pesos: 0,001; Lote de ajuste: 128 | 0,7652 | 1,0566 | Finalista |
| 10 | Dimensión interna: 128; Capas: 3; Capas de mezcla: 2; Cabezas: 8; Dropout: 0,1; Tasa de aprendizaje: 0,0001; Decaimiento de pesos: 0,0001; Lote de ajuste: 32 | 0,7753 | — | Filtro inicial |
| 6 | Dimensión interna: 64; Capas: 2; Capas de mezcla: 3; Cabezas: 4; Dropout: 0,1; Tasa de aprendizaje: 0,0001; Decaimiento de pesos: 0,0001; Lote de ajuste: 32 | 0,7833 | — | Filtro inicial |
| 7 | Dimensión interna: 128; Capas: 2; Capas de mezcla: 1; Cabezas: 8; Dropout: 0,1; Tasa de aprendizaje: 0,0003; Decaimiento de pesos: 0,0001; Lote de ajuste: 64 | 0,7873 | — | Filtro inicial |
| 11 | Dimensión interna: 128; Capas: 3; Capas de mezcla: 3; Cabezas: 8; Dropout: 0,2; Tasa de aprendizaje: 0,0003; Decaimiento de pesos: 0,001; Lote de ajuste: 64 | 0,8069 | — | Filtro inicial |
| 15 | Dimensión interna: 32; Capas: 3; Capas de mezcla: 1; Cabezas: 4; Dropout: 0,1; Tasa de aprendizaje: 0,001; Decaimiento de pesos: 0,0001; Lote de ajuste: 64 | 0,8106 | — | Filtro inicial |
| 16 | Dimensión interna: 32; Capas: 3; Capas de mezcla: 2; Cabezas: 4; Dropout: 0,2; Tasa de aprendizaje: 0,001; Decaimiento de pesos: 0,001; Lote de ajuste: 64 | 0,8597 | — | Filtro inicial |
| 8 | Dimensión interna: 128; Capas: 2; Capas de mezcla: 2; Cabezas: 8; Dropout: 0,2; Tasa de aprendizaje: 0,0003; Decaimiento de pesos: 0,0001; Lote de ajuste: 64 | 0,8765 | — | Filtro inicial |
| 9 | Dimensión interna: 128; Capas: 3; Capas de mezcla: 1; Cabezas: 8; Dropout: 0,2; Tasa de aprendizaje: 0,001; Decaimiento de pesos: 0,001; Lote de ajuste: 128 | 0,8978 | — | Filtro inicial |
| 18 | Dimensión interna: 256; Capas: 2; Capas de mezcla: 1; Cabezas: 8; Dropout: 0,2; Tasa de aprendizaje: 0,0001; Decaimiento de pesos: 0,001; Lote de ajuste: 32 | 0,9014 | — | Filtro inicial |
| 3 | Dimensión interna: 64; Capas: 3; Capas de mezcla: 1; Cabezas: 4; Dropout: 0,2; Tasa de aprendizaje: 0,0003; Decaimiento de pesos: 0,0001; Lote de ajuste: 64 | 0,9160 | — | Filtro inicial |
| 5 | Dimensión interna: 64; Capas: 3; Capas de mezcla: 3; Cabezas: 4; Dropout: 0,3; Tasa de aprendizaje: 0,001; Decaimiento de pesos: 0,001; Lote de ajuste: 128 | 1,0757 | — | Filtro inicial |
| 14 | Dimensión interna: 128; Capas: 3; Capas de mezcla: 2; Cabezas: 16; Dropout: 0,3; Tasa de aprendizaje: 0,0003; Decaimiento de pesos: 0,0001; Lote de ajuste: 64 | 1,0889 | — | Filtro inicial |
| 13 | Dimensión interna: 128; Capas: 2; Capas de mezcla: 3; Cabezas: 16; Dropout: 0,2; Tasa de aprendizaje: 0,0001; Decaimiento de pesos: 0,0001; Lote de ajuste: 128 | 1,2778 | — | Filtro inicial |
| 17 | Dimensión interna: 32; Capas: 2; Capas de mezcla: 3; Cabezas: 4; Dropout: 0,3; Tasa de aprendizaje: 0,0003; Decaimiento de pesos: 0,0001; Lote de ajuste: 128 | 1,4525 | — | Filtro inicial |
| 20 | Dimensión interna: 256; Capas: 2; Capas de mezcla: 2; Cabezas: 8; Dropout: 0,3; Tasa de aprendizaje: 0,0003; Decaimiento de pesos: 0,01; Lote de ajuste: 128 | 1,4644 | — | Filtro inicial |

[Resultados de desarrollo por año y semilla](../_static/resultados/diseno_entrenamiento/adj_modelo7_20261114_0430__N+O.csv) · [Configuraciones completas](../_static/resultados/diseno_entrenamiento/adj_modelo7_20261114_0430__N+O_configuraciones.csv).

<!-- diseno-entrenamiento:fin -->

## Pipeline de entrenamiento y prueba

<!-- entrenamiento-portable:inicio -->
### Comandos de entrenamiento y evaluación

Validación temporal de la configuración seleccionada:

```bash
python scripts/entrenar_arquitectura.py --modelo tgforecaster --anio 2021 --semilla 1000 --dispositivo auto --salida salidas/tgforecaster_validacion_1000
```

Comprobación funcional breve:

```bash
python scripts/entrenar_arquitectura.py --modelo tgforecaster --smoke --dispositivo cpu --salida salidas/tgforecaster_funcional
```

Evaluación final, con ajuste y parada anteriores a 2022:

```bash
python scripts/entrenar_arquitectura.py --modelo tgforecaster --fase prueba --allow-test --semilla 1000 --dispositivo auto --salida salidas/tgforecaster_prueba_1000
```

El código completo de preparación, optimización, guardado y restauración está en el pipeline ejecutable compartido. Cada salida debe ser una carpeta nueva.

El control numérico con descripciones estáticas se ejecuta con `--modelo tgforecaster_n`; la variante `tgforecaster` incorpora los documentos oficiales.
<!-- entrenamiento-portable:fin -->

El pipeline fija el plan y las huellas de entrada, genera ventanas cronológicas, ajusta las transformaciones por pliegue y evalúa 20 candidatos en 2018. Confirma cinco finalistas en 2019–2021, selecciona por MASE medio y reajusta con cinco semillas finales usando la reserva interna anterior a 2022. La lectura de la prueba solo ocurre después de cerrar la selección. El libro muestra las implementaciones y no llama al entrenamiento.

### Datos, referencias y escaladores por pliegue

```python
"""Datos numéricos del contrato común (30 históricas + 23 de calendario) por pliegue.

Reglas que se cumplen aquí:
  * el panel y sus rezagos de disponibilidad no se modifican; el clima recibe el rezago
    de diseño (168 h) mediante ``demanda.diseno_numerico.preparar_numericas_comunes``;
  * medianas de imputación, medias y desviaciones se estiman solo con las filas de
    ajuste de cada pliegue (sin parada interna, validación ni prueba);
  * la demanda se divide por el nivel de su ventana y se centra con la dispersión de
    ajuste, igual que el motor v3; la etiqueta es ``y / nivel``;
  * las entradas ``x_*`` solo se leen en filas anteriores al origen.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from demanda import config as C
from demanda import particion as P
from demanda.diseno_numerico import preparar_numericas_comunes

BLOQUES = {"precio": "x_precio_", "hidrologia": "x_hid_", "clima": "x_obs_"}


def sha256(ruta: Path) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


@dataclass
class Conjunto:
    x: np.ndarray        # (n, L, n_hist + n_mascaras) normalizado; columna 0 = demanda
    cal_p: np.ndarray    # (n, L, n_cal)
    cal_f: np.ndarray    # (n, H, n_cal)
    y: np.ndarray        # (n, H) kWh
    nivel: np.ndarray    # (n,) media de la demanda en la ventana
    origen: pd.DatetimeIndex
    dia: np.ndarray      # (n,) índice del día del origen en la tabla de días de texto

    def __len__(self):
        return len(self.y)

    @property
    def y_norm(self) -> np.ndarray:
        return (self.y / self.nivel[:, None]).astype("float32")


class DatosNumericos:
    """Carga el panel congelado y prepara ventanas bajo el contrato numérico común."""

    def __init__(self, raiz: Path, verificar_hash: bool = True):
        self.raiz = Path(raiz)
        self.contrato = json.loads((self.raiz / "config" / "diseno_entrenamiento_texto.json").read_text(encoding="utf-8"))
        ruta_panel = self.raiz / "data" / "processed" / "panel_horario.parquet"
        self.hash_panel = sha256(ruta_panel)
        if verificar_hash and self.hash_panel != self.contrato["panel_sha256"]:
            raise RuntimeError("El panel no coincide con el SHA-256 del contrato; no se entrena con otro archivo")
        panel = pd.read_parquet(ruta_panel)
        registro = json.loads((self.raiz / "data" / "processed" / "feature_registry.json").read_text(encoding="utf-8"))
        out, _ = preparar_numericas_comunes(panel, registro, self.contrato)
        self.hist = list(self.contrato["numericas"]["historicas"])
        self.cal = list(self.contrato["numericas"]["calendario"])
        if self.hist[0] != "x_" + C.OBJETIVO:
            raise RuntimeError("La demanda debe ser la primera variable histórica")
        self.indice = panel.index
        self.X = out[self.hist].to_numpy(dtype="float64")
        self.K = out[self.cal].to_numpy(dtype="float64")
        self.Y = panel["y_" + C.OBJETIVO].to_numpy(dtype="float64")
        self.L, self.H = int(self.contrato["ventana_h"]), int(self.contrato["horizonte_h"])
        self.q168 = float(self.contrato["metrica"]["q168_kwh"])
        if np.isnan(self.K).any():
            raise RuntimeError("El calendario no puede tener ausencias")
        # Máscaras de ausencia: una por bloque cuando todas sus columnas comparten patrón; si no, una por columna.
        nan = np.isnan(self.X)
        self.nombres_mascara, cols = [], []
        for nombre, prefijo in BLOQUES.items():
            js = [j for j, c in enumerate(self.hist) if c.startswith(prefijo)]
            if not js:
                continue
            if all((nan[:, j] == nan[:, js[0]]).all() for j in js):
                cols.append(nan[:, js[0]]); self.nombres_mascara.append("ausente_" + nombre)
            else:
                for j in js:
                    cols.append(nan[:, j]); self.nombres_mascara.append("ausente_" + self.hist[j])
        self.M = np.stack(cols, axis=1).astype("float32")
        self.dia_base = self.indice[0].normalize()
        self._off_l = np.arange(-self.L, 0)
        self._off_h = np.arange(0, self.H)

    # ------------------------------------------------------------------ orígenes
    def elegibles(self, pos: np.ndarray) -> np.ndarray:
        """Orígenes con demanda completa en la ventana y en la etiqueta. Regla común a todas las variantes."""
        pos = np.asarray(pos)
        ok = ~np.isnan(self.X[pos[:, None] + self._off_l, 0]).any(axis=1) & ~np.isnan(self.Y[pos[:, None] + self._off_h]).any(axis=1)
        return pos[ok]

    def pliegues(self) -> list[dict]:
        v = self.contrato["validacion"]
        fs = P.folds_forward_chaining(self.indice, v["anios"], self.H, self.L, 24, 0, v["fraccion_parada"])
        return [{k: (self.elegibles(f[k]) if k != "anio" else f[k]) for k in f} for f in fs]

    def ajuste_final(self) -> tuple[np.ndarray, np.ndarray]:
        tr, es = P.pretest(self.indice, self.H, self.L, 24, 0, self.contrato["validacion"]["fraccion_parada"])
        return self.elegibles(tr), self.elegibles(es)

    def prueba(self) -> np.ndarray:
        """Orígenes de la prueba retrospectiva 2022-2025. Solo se llama en la etapa final."""
        return self.elegibles(P.origenes(self.indice, "test", self.H, self.L, 24, 0, allow_test=True))

    def dia_de(self, pos: np.ndarray) -> np.ndarray:
        return ((self.indice[pos].normalize() - self.dia_base) / pd.Timedelta(days=1)).round().astype("int64").to_numpy()

    # ------------------------------------------------------------------- escalas
    def ajustar_escalas(self, pos_tr: np.ndarray) -> dict:
        """Estadísticos estimados solo con las filas de historia de los orígenes de ajuste."""
        a, b = int(pos_tr.min()) - self.L, int(pos_tr.max())  # filas [a, b): historia de todos los orígenes de ajuste
        Xa, Ka = self.X[a:b], self.K[a:b]
        sin_datos = np.isnan(Xa).all(axis=0)
        with np.errstate(all="ignore"):
            mediana = np.where(sin_datos, 0.0, np.nanmedian(np.where(sin_datos, 0.0, Xa), axis=0))
            media = np.where(sin_datos, 0.0, np.nanmean(np.where(sin_datos, 0.0, Xa), axis=0))
            std = np.where(sin_datos, 1.0, np.nanstd(np.where(sin_datos, 0.0, Xa), axis=0, ddof=1))
        std = np.where(std > 0, std, 1.0)
        mc, sc = Ka.mean(axis=0), Ka.std(axis=0, ddof=1)
        sc = np.where(sc > 0, sc, 1.0)
        dem = self.X[pos_tr[:, None] + self._off_l, 0]
        sigma = max(float((dem / dem.mean(axis=1, keepdims=True)).std()), 1e-6)
        return dict(mediana=mediana, media=media, std=std, media_cal=mc, std_cal=sc, sigma_demanda=sigma,
                    no_estimables=[c for c, s in zip(self.hist, sin_datos) if s],
                    inicio_ajuste=str(self.indice[a]), fin_ajuste=str(self.indice[b - 1]), filas_ajuste=int(b - a))

    def construir(self, pos: np.ndarray, esc: dict) -> Conjunto:
        pos = np.asarray(pos)
        il, ih = pos[:, None] + self._off_l, pos[:, None] + self._off_h
        xw = self.X[il]                                    # (n, L, n_hist); solo filas anteriores al origen
        mascara = self.M[il]
        xw = np.where(np.isnan(xw), esc["mediana"], xw)
        nivel = xw[:, :, 0].mean(axis=1)
        x = (xw - esc["media"]) / esc["std"]
        x[:, :, 0] = (xw[:, :, 0] / nivel[:, None] - 1.0) / esc["sigma_demanda"]
        cal_p = (self.K[il] - esc["media_cal"]) / esc["std_cal"]
        cal_f = (self.K[ih] - esc["media_cal"]) / esc["std_cal"]
        return Conjunto(np.concatenate([x, mascara], axis=2).astype("float32"), cal_p.astype("float32"), cal_f.astype("float32"),
                        self.Y[ih], nivel, self.indice[pos], self.dia_de(pos))

    # ---------------------------------------------------------------- referencias
    def ingenuos(self, pos: np.ndarray) -> dict[str, np.ndarray]:
        """Ingenuos diario y semanal con la misma disponibilidad que los modelos (kWh)."""
        lag = C.REZAGO_H["demanda_real"]
        out = {}
        for nombre, base in (("ingenuo_24h", 24), ("ingenuo_168h", 168)):
            pred = np.zeros((len(pos), self.H))
            for h in range(self.H):
                k = int(np.ceil((h + 1 + lag) / base)) if base == 24 else 1
                pred[:, h] = self.Y[pos + h - base * max(k, 1)]
            out[nombre] = pred
        return out

    def mase_por_origen(self, y: np.ndarray, yhat: np.ndarray) -> np.ndarray:
        return np.abs(y - yhat).mean(axis=1) / self.q168


def escalas_a_json(esc: dict, datos: DatosNumericos) -> dict:
    return dict(columnas_historicas=datos.hist, columnas_calendario=datos.cal, mascaras=datos.nombres_mascara,
                mediana=dict(zip(datos.hist, map(float, esc["mediana"]))), media=dict(zip(datos.hist, map(float, esc["media"]))),
                std=dict(zip(datos.hist, map(float, esc["std"]))), media_calendario=dict(zip(datos.cal, map(float, esc["media_cal"]))),
                std_calendario=dict(zip(datos.cal, map(float, esc["std_cal"]))), sigma_demanda=esc["sigma_demanda"],
                no_estimables=esc["no_estimables"], inicio_ajuste=esc["inicio_ajuste"], fin_ajuste=esc["fin_ajuste"],
                filas_ajuste=esc["filas_ajuste"], q168_kwh=datos.q168)
```

### Bucle de ajuste y predicción

```python
"""Entrenamiento de una red con las reglas del motor v3 del proyecto.

AdamW, pérdida MAE sobre demanda / nivel, ReduceLROnPlateau(factor 0.5, paciencia 2), recorte
de gradiente, parada cuando la época es >= 10 y hay 6 épocas sin mejora relativa de 0.2 %,
y restauración del mejor estado absoluto en la parada interna.
"""
from __future__ import annotations

import random
import time

import numpy as np
import torch
from torch import nn


def fijar_semilla(semilla: int) -> None:
    random.seed(semilla)
    np.random.seed(semilla)
    torch.manual_seed(semilla)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(semilla)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def a_tensores(conj, dev) -> dict:
    return dict(x=torch.from_numpy(conj.x).to(dev), cal_p=torch.from_numpy(conj.cal_p).to(dev), cal_f=torch.from_numpy(conj.cal_f).to(dev),
                dia=torch.from_numpy(np.array(conj.dia, dtype="int64")).to(dev), y=torch.from_numpy(conj.y_norm).to(dev))


def _salida(red, t, ix):
    return red(t["x"][ix], t["cal_p"][ix], t["cal_f"][ix], t["dia"][ix])


@torch.no_grad()
def predecir(red, t: dict, lote: int = 256) -> np.ndarray:
    """Predicción normalizada (demanda / nivel) en modo evaluación."""
    red.eval()
    n = len(t["y"])
    partes = [_salida(red, t, slice(i, i + lote)).float().cpu().numpy() for i in range(0, n, lote)]
    return np.concatenate(partes, axis=0)


@torch.no_grad()
def _perdida(red, t: dict, lote: int = 256) -> float:
    red.eval()
    n, tot = len(t["y"]), 0.0
    for i in range(0, n, lote):
        s = slice(i, i + lote)
        tot += float((_salida(red, t, s) - t["y"][s]).abs().mean()) * len(t["y"][s])
    return tot / n


def entrenar(red, tr: dict, es: dict, p: dict, semilla: int, opciones: dict, observador=None) -> dict:
    """Ajusta ``red`` con ``tr`` y detiene con ``es``. Devuelve la historia y deja en ``red`` el mejor estado."""
    dev = tr["x"].device
    perdida = nn.L1Loss()
    opt = torch.optim.AdamW(red.parameters(), lr=float(p["lr"]), weight_decay=float(p["weight_decay"]))
    plan = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
    mejor = referencia = float("inf")
    estado, sin_mejora, historia = None, 0, []
    gen = torch.Generator().manual_seed(int(semilla))
    lote, n = int(p["lote"]), len(tr["y"])
    t0 = time.time()
    for ep in range(1, int(opciones["max_epocas"]) + 1):
        red.train()
        tot = 0.0
        for ix in torch.randperm(n, generator=gen).split(lote):
            if len(ix) < 2:      # la normalización por lotes necesita más de una muestra
                continue
            ix = ix.to(dev)
            opt.zero_grad(set_to_none=True)
            loss = perdida(_salida(red, tr, ix), tr["y"][ix])
            if not torch.isfinite(loss):
                raise FloatingPointError("Pérdida no finita")
            loss.backward()
            nn.utils.clip_grad_norm_(red.parameters(), float(opciones["recorte_gradiente"]))
            opt.step()
            tot += float(loss.detach()) * len(ix)
        vl = _perdida(red, es)
        historia.append(dict(epoca=ep, perdida_ajuste=tot / n, perdida_parada=vl, lr=opt.param_groups[0]["lr"], segundos=round(time.time() - t0, 1)))
        if vl < mejor:
            mejor = vl
            estado = {k: v.detach().clone() for k, v in red.state_dict().items()}
        if not np.isfinite(referencia) or vl < referencia - max(1e-6, 0.002 * abs(referencia)):
            referencia, sin_mejora = vl, 0
        else:
            sin_mejora += 1
        plan.step(vl)
        if observador is not None:
            observador(historia[-1])
        if ep >= int(opciones["min_epocas"]) and sin_mejora >= int(opciones["paciencia"]):
            break
    red.load_state_dict(estado)
    return dict(epocas=len(historia), mejor_epoca=int(np.argmin([h["perdida_parada"] for h in historia]) + 1),
                mejor_perdida_parada=float(mejor), historia=historia, segundos=round(time.time() - t0, 1))
```

### Búsqueda, confirmación y evaluación final

```python
"""Protocolo: búsqueda en el pliegue 2018, finalistas en 2019-2021 y prueba retrospectiva.

Cada entrenamiento escribe su carpeta con ``resultado.json`` al terminar; si la carpeta ya lo
contiene, no se repite. Por eso el proceso se puede interrumpir y reanudar.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from demanda import metricas as MET

from . import motor
from .datos import DatosNumericos, escalas_a_json
from . import redes


SIN_MEMORIA = getattr(torch, "OutOfMemoryError", None) or torch.cuda.OutOfMemoryError


def guardar_json(ruta: Path, obj) -> None:
    ruta = Path(ruta)
    tmp = ruta.with_suffix(ruta.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    os.replace(tmp, ruta)


def leer_json(ruta: Path):
    return json.loads(Path(ruta).read_text(encoding="utf-8"))


class Contexto:
    """Estado compartido: datos, texto, configuración, dispositivo, bitácora y barra de avance."""

    def __init__(self, raiz: Path, salida: Path, cfg: dict, datos: DatosNumericos, dispositivo: str, log, rapido: dict | None = None):
        self.raiz, self.salida, self.cfg, self.datos, self.dispositivo, self.log = Path(raiz), Path(salida), cfg, datos, dispositivo, log
        self.rapido = rapido
        self.texto = None        # dict(vectores, lista) en CPU; cada red los copia a su dispositivo
        self.descripciones = None  # (n_canales, 768) solo en TGForecaster
        self.barra = None
        self._cache = (None, None)
        self.opciones = dict(cfg["entrenamiento"])
        if rapido:
            self.opciones.update(max_epocas=rapido["max_epocas"], min_epocas=1)

    # ------------------------------------------------------------------ datos por pliegue
    def recortar(self, pos: dict) -> dict:
        """En la prueba rápida se usan pocos orígenes; en la corrida real no cambia nada."""
        if not self.rapido:
            return pos
        r = self.rapido
        return dict(tr=pos["tr"][-r["n_tr"]:], es=pos["es"][-r["n_es"]:], ev=pos["ev"][:r["n_ev"]])

    def conjuntos(self, clave: str, pos: dict):
        if self._cache[0] != clave:
            esc = self.datos.ajustar_escalas(pos["tr"])
            self._cache = (clave, (esc, {k: self.datos.construir(pos[k], esc) for k in ("tr", "es", "ev")}))
        return self._cache[1]


def _evaluar(datos: DatosNumericos, red, conj, t) -> tuple[np.ndarray, dict]:
    yhat = motor.predecir(red, t).astype("float64") * conj.nivel[:, None]
    return yhat, MET.evaluar(conj.y, yhat, datos.q168)


def corrida(ctx: Contexto, variante: str, p: dict, semilla: int, clave_datos: str, pos: dict, carpeta: Path, nombre_ev: str,
            guardar_modelo: bool = False) -> dict:
    """Entrena una configuración con una semilla y evalúa en ``pos['ev']``. Reanudable."""
    carpeta = Path(carpeta)
    rr = carpeta / "resultado.json"
    etiqueta = f"{variante} cfg {p['id']:02d} semilla {semilla} {nombre_ev}"
    if rr.exists():
        if ctx.barra is not None:
            ctx.barra.update(1)
        return leer_json(rr)
    carpeta.mkdir(parents=True, exist_ok=True)
    datos = ctx.datos
    esc, conj = ctx.conjuntos(clave_datos, pos)
    texto = None
    if variante == "N+O":
        texto = dict(vectores=torch.from_numpy(ctx.texto["vectores"]), lista=torch.from_numpy(ctx.texto["lista"]))
    desc = torch.from_numpy(ctx.descripciones) if ctx.descripciones is not None else None
    dispositivos = [ctx.dispositivo] + (["cpu"] if ctx.dispositivo != "cpu" else [])
    for intento, dev in enumerate(dispositivos):
        try:
            motor.fijar_semilla(semilla)
            red = redes.construir(ctx.cfg["paquete"]["red"], p, conj["tr"].x.shape[2], conj["tr"].cal_p.shape[2], datos.L, datos.H, texto, desc).to(dev)
            tens = {k: motor.a_tensores(v, dev) for k, v in conj.items()}

            def observador(h, _e=etiqueta):
                if ctx.barra is not None:
                    ctx.barra.set_postfix_str(f"{_e} | época {h['epoca']} | parada {h['perdida_parada']:.5f}", refresh=True)

            t0 = time.time()
            info = motor.entrenar(red, tens["tr"], tens["es"], p, semilla, ctx.opciones, observador)
            yhat, met = _evaluar(datos, red, conj["ev"], tens["ev"])
            mase_tr = _evaluar(datos, red, conj["tr"], tens["tr"])[1]["MASE"]
            mase_es = _evaluar(datos, red, conj["es"], tens["es"])[1]["MASE"]
            break
        except SIN_MEMORIA:
            if intento == len(dispositivos) - 1:
                raise
            ctx.log(f"Memoria de GPU insuficiente en {etiqueta}; esta corrida se repite en CPU.")
            red = tens = None
            torch.cuda.empty_cache()
    res = dict(variante=variante, configuracion=p, semilla=int(semilla), evaluacion=nombre_ev, dispositivo=str(dev),
               n_parametros=red.n_parametros(), epocas=info["epocas"], mejor_epoca=info["mejor_epoca"], mejor_perdida_parada=info["mejor_perdida_parada"],
               segundos=round(time.time() - t0, 1), n_ajuste=len(conj["tr"]), n_parada=len(conj["es"]), n_evaluacion=len(conj["ev"]),
               origen_ajuste=[str(conj["tr"].origen[0]), str(conj["tr"].origen[-1])], origen_parada=[str(conj["es"].origen[0]), str(conj["es"].origen[-1])],
               origen_evaluacion=[str(conj["ev"].origen[0]), str(conj["ev"].origen[-1])],
               filas_escalas=[esc["inicio_ajuste"], esc["fin_ajuste"]], MASE_ajuste=mase_tr, MASE_parada=mase_es, metricas=met,
               mase_por_horizonte=[float(v) for v in MET.mase_por_horizonte(conj["ev"].y, yhat, datos.q168)], diagnostico=red.diagnostico())
    pd.DataFrame(info["historia"]).to_csv(carpeta / "historia.csv", index=False)
    np.savez_compressed(carpeta / f"pred_{nombre_ev}.npz", origen=np.asarray(conj["ev"].origen.strftime("%Y-%m-%d")), yhat_kwh=yhat.astype("float32"),
                        y_kwh=conj["ev"].y.astype("float32"), mase_por_origen=datos.mase_por_origen(conj["ev"].y, yhat))
    if guardar_modelo:
        torch.save({k: v.cpu() for k, v in red.state_dict().items()}, carpeta / "modelo.pt")
        guardar_json(carpeta / "escaladores.json", escalas_a_json(esc, datos))
    guardar_json(rr, res)
    del red, tens
    if ctx.barra is not None:
        ctx.barra.update(1)
    ctx.log(f"{etiqueta}: MASE {met['MASE']:.4f} | épocas {info['epocas']} | {res['segundos']:.0f} s | {dev}", consola=False)
    return res


# ------------------------------------------------------------------------------- etapas
def _fila(r: dict) -> dict:
    p = r["configuracion"]
    return dict(variante=r["variante"], **p, semilla=r["semilla"], evaluacion=r["evaluacion"],
                MASE=r["metricas"]["MASE"], MAE_kwh=r["metricas"]["MAE"], RMSE_kwh=r["metricas"]["RMSE"], MAPE=r["metricas"]["MAPE"],
                MASE_ajuste=r["MASE_ajuste"], MASE_parada=r["MASE_parada"], n_parametros=r["n_parametros"], epocas=r["epocas"],
                mejor_epoca=r["mejor_epoca"], segundos=r["segundos"], dispositivo=r["dispositivo"], **{"diag_" + k: v for k, v in r.get("diagnostico", {}).items()})


def _pos_pliegue(ctx: Contexto, f: dict) -> dict:
    return ctx.recortar(dict(tr=f["tr"], es=f["es"], ev=f["val"]))


def etapa_busqueda(ctx: Contexto, configs: list[dict]) -> dict:
    """Pliegue inicial: todas las configuraciones con una semilla; pasan las mejores de cada variante."""
    b = ctx.cfg["busqueda"]
    f = next(x for x in ctx.datos.pliegues() if x["anio"] == b["pliegue_inicial"])
    pos, filas = _pos_pliegue(ctx, f), []
    for variante in ctx.cfg["variantes"]:
        for p in configs:
            carpeta = ctx.salida / "busqueda" / variante.replace("+", "_") / f"pliegue_{f['anio']}" / f"cfg_{p['id']:02d}"
            filas.append(_fila(corrida(ctx, variante, p, b["semilla"], f"pliegue_{f['anio']}", pos, carpeta, f"val_{f['anio']}")))
    t = pd.DataFrame(filas).sort_values(["variante", "MASE", "n_parametros", "id"]).reset_index(drop=True)
    t["puesto"] = t.groupby("variante").cumcount() + 1
    t["finalista"] = t["puesto"] <= b["finalistas"]
    t.to_csv(ctx.salida / "busqueda" / "tabla_pliegue_inicial.csv", index=False)
    finalistas = {v: t.loc[(t.variante == v) & t.finalista, "id"].astype(int).tolist() for v in ctx.cfg["variantes"]}
    guardar_json(ctx.salida / "busqueda" / "finalistas.json", finalistas)
    return finalistas


def etapa_pliegues(ctx: Contexto, configs: list[dict], finalistas: dict) -> dict:
    """Pliegues restantes para las finalistas; se elige la de menor MASE medio en los cuatro pliegues."""
    b = ctx.cfg["busqueda"]
    por_id = {p["id"]: p for p in configs}
    filas = pd.read_csv(ctx.salida / "busqueda" / "tabla_pliegue_inicial.csv")
    filas = filas.loc[filas.finalista].drop(columns=["puesto", "finalista"]).to_dict("records")
    for f in ctx.datos.pliegues():
        if f["anio"] not in b["pliegues_restantes"]:
            continue
        pos = _pos_pliegue(ctx, f)
        for variante in ctx.cfg["variantes"]:
            for i in finalistas[variante]:
                carpeta = ctx.salida / "busqueda" / variante.replace("+", "_") / f"pliegue_{f['anio']}" / f"cfg_{i:02d}"
                filas.append(_fila(corrida(ctx, variante, por_id[i], b["semilla"], f"pliegue_{f['anio']}", pos, carpeta, f"val_{f['anio']}")))
    t = pd.DataFrame(filas)
    t.to_csv(ctx.salida / "busqueda" / "tabla_finalistas_por_pliegue.csv", index=False)
    r = (t.groupby(["variante", "id"]).agg(MASE_medio=("MASE", "mean"), MASE_desv_pliegues=("MASE", "std"), MASE_min=("MASE", "min"), MASE_max=("MASE", "max"),
                                             pliegues=("MASE", "size"), n_parametros=("n_parametros", "first")).reset_index()
         .sort_values(["variante", "MASE_medio", "n_parametros", "id"]).reset_index(drop=True))
    r["puesto"] = r.groupby("variante").cumcount() + 1
    r.to_csv(ctx.salida / "busqueda" / "resumen_finalistas.csv", index=False)
    elegidas = {v: int(r.loc[(r.variante == v) & (r.puesto == 1), "id"].iloc[0]) for v in ctx.cfg["variantes"]}
    guardar_json(ctx.salida / "busqueda" / "configuracion_elegida.json",
                 {v: dict(configuracion=por_id[i], MASE_medio_validacion=float(r.loc[(r.variante == v) & (r.id == i), "MASE_medio"].iloc[0])) for v, i in elegidas.items()})
    return elegidas


def bootstrap_bloques(dif: np.ndarray, base: np.ndarray, bloque: int, remuestreos: int, semilla: int) -> dict:
    """Bootstrap circular por bloques de días para la diferencia pareada por origen (A - B) y su valor relativo a B."""
    n = len(dif)
    g = np.random.default_rng(semilla)
    nb = int(np.ceil(n / bloque))
    d_obs, b_obs = float(dif.mean()), float(base.mean())
    md, mb = np.empty(remuestreos), np.empty(remuestreos)
    for a in range(0, remuestreos, 500):
        k = min(500, remuestreos - a)
        ix = ((g.integers(0, n, size=(k, nb))[:, :, None] + np.arange(bloque)[None, None, :]) % n).reshape(k, -1)[:, :n]
        md[a:a + k], mb[a:a + k] = dif[ix].mean(axis=1), base[ix].mean(axis=1)
    rel = 100 * md / mb
    return dict(bloque_dias=int(bloque), remuestreos=int(remuestreos), diferencia_MASE=d_obs, ic95_diferencia_MASE=[float(v) for v in np.percentile(md, [2.5, 97.5])],
                diferencia_relativa_pct=100 * d_obs / b_obs, ic95_diferencia_relativa_pct=[float(v) for v in np.percentile(rel, [2.5, 97.5])],
                p_bilateral_bootstrap=float((np.abs(md - d_obs) >= abs(d_obs)).mean()))


def etapa_final(ctx: Contexto, configs: list[dict], elegidas: dict) -> dict:
    """Ajuste final con los datos anteriores a 2022 y evaluación retrospectiva 2022-2025."""
    datos, por_id = ctx.datos, {p["id"]: p for p in configs}
    tr, es = datos.ajuste_final()
    pos = ctx.recortar(dict(tr=tr, es=es, ev=datos.prueba()))
    semillas = ctx.rapido["semillas_finales"] if ctx.rapido else ctx.cfg["final"]["semillas"]
    filas, perdidas = [], {}
    for variante in ctx.cfg["variantes"]:
        p = por_id[elegidas[variante]]
        for s in semillas:
            carpeta = ctx.salida / "final" / variante.replace("+", "_") / f"semilla_{s}"
            r = corrida(ctx, variante, p, s, "final", pos, carpeta, "prueba", guardar_modelo=True)
            filas.append(_fila(r))
            perdidas.setdefault(variante, []).append(np.load(carpeta / "pred_prueba.npz")["mase_por_origen"])
    t = pd.DataFrame(filas)
    t.to_csv(ctx.salida / "final" / "metricas_por_semilla.csv", index=False)
    y = datos.Y[pos["ev"][:, None] + np.arange(datos.H)]
    origen = datos.indice[pos["ev"]].strftime("%Y-%m-%d")
    ing = {k: datos.mase_por_origen(y, v) for k, v in datos.ingenuos(pos["ev"]).items()}
    media = {v: np.mean(perdidas[v], axis=0) for v in perdidas}          # pérdida por origen promediada sobre semillas
    pd.DataFrame(dict(origen=origen, **{f"MASE_{v}": media[v] for v in media}, **{f"MASE_{k}": ing[k] for k in ing})).to_csv(ctx.salida / "final" / "perdida_por_origen.csv", index=False)
    por_h = {}
    for variante in ctx.cfg["variantes"]:
        hs = [leer_json(ctx.salida / "final" / variante.replace("+", "_") / f"semilla_{s}" / "resultado.json")["mase_por_horizonte"] for s in semillas]
        por_h[variante] = np.mean(hs, axis=0)
    pd.DataFrame(dict(horizonte_h=np.arange(1, datos.H + 1), **{f"MASE_{v}": por_h[v] for v in por_h})).to_csv(ctx.salida / "final" / "mase_por_horizonte.csv", index=False)

    resumen = dict(n_origenes_prueba=int(len(pos["ev"])), periodo_prueba=[origen[0], origen[-1]], semillas=list(map(int, semillas)), q168_kwh=datos.q168,
                   referencias={k: float(v.mean()) for k, v in ing.items()}, variantes={})
    for variante in ctx.cfg["variantes"]:
        m = t.loc[t.variante == variante, "MASE"].to_numpy()
        resumen["variantes"][variante] = dict(configuracion=por_id[elegidas[variante]], MASE_media_semillas=float(m.mean()),
                                              MASE_desv_semillas=float(m.std(ddof=1)) if len(m) > 1 else None, MASE_por_semilla=[float(v) for v in m],
                                              MAE_kwh_media=float(t.loc[t.variante == variante, "MAE_kwh"].mean()), MAPE_media=float(t.loc[t.variante == variante, "MAPE"].mean()))
    if {"N", "N+O"} <= set(media):
        inf = ctx.cfg["inferencia"]
        rem = ctx.rapido["remuestreos"] if ctx.rapido else inf["remuestreos"]
        dif = media["N+O"] - media["N"]
        resumen["contraste_N+O_menos_N"] = dict(
            definicion="diferencia pareada por origen de la pérdida MASE, promediada sobre semillas; relativa = 100 (MASE_N+O - MASE_N) / MASE_N; negativo favorece al texto",
            margen_practico_pct=100 * inf["margen_relativo"], primario=bootstrap_bloques(dif, media["N"], inf["bloque_dias"], rem, inf["semilla_bootstrap"]),
            sensibilidad=[bootstrap_bloques(dif, media["N"], b, rem, inf["semilla_bootstrap"]) for b in inf["sensibilidad_bloque_dias"]],
            nota="Resultado descriptivo de este modelo. La inferencia del proyecto aplica Holm a la familia preregistrada de contrastes; el valor p de aquí no está ajustado.")
    guardar_json(ctx.salida / "final" / "resumen_prueba.json", resumen)
    return resumen
```

### Disponibilidad, embeddings y preparación del texto

```python
"""Canal oficial de texto: vectores por documento y listas de documentos elegibles por día.

Contrato (config/diseno_entrenamiento_texto.json):
  * escenario declarado; un documento es elegible si su disponibilidad cae en
    [origen - 72 h, origen]; como máximo los 64 más recientes;
  * codificador congelado intfloat/multilingual-e5-base en la revisión fijada, prefijo
    ``query: ``, promedio de tokens con máscara y normalización L2;
  * máximo 512 tokens por fragmento (incluye prefijo y tokens especiales), solapamiento
    de 64 y hasta 8 fragmentos por documento, en orden de lectura;
  * no se concatenan URL, identificadores ni fechas de recuperación al contenido.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

DIM = 768


def cargar_corpus_oficial(raiz: Path) -> pd.DataFrame:
    t = pd.read_parquet(Path(raiz) / "data" / "processed" / "corpus_oficial.parquet")
    t["disponible_desde"] = pd.to_datetime(t["disponible_desde"], utc=True)
    if not t["disponible_desde"].is_monotonic_increasing:
        raise RuntimeError("El corpus debe estar ordenado por disponibilidad")
    return t.reset_index(drop=True)


def contenido(fila) -> str:
    cuerpo = (fila.texto or "").strip()
    return (fila.titulo.strip() + "\n" + cuerpo) if cuerpo else fila.titulo.strip()


def listas_por_dia(disponible: pd.Series, dia_base: pd.Timestamp, n_dias: int, ventana_h: int, max_n: int):
    """Para cada día d (00:00 de Bogotá) devuelve los documentos con disponibilidad en [d - ventana, d].

    Misma regla que ``demanda.corpus.documentos_en_origen``: extremos inclusivos y, si hay más de
    ``max_n``, se conservan los más recientes. Devuelve (lista, total_en_ventana): ``lista`` es
    (n_dias, max_n) con posiciones del corpus y -1 como relleno.
    """
    valores = pd.DatetimeIndex(disponible).as_unit("ns").asi8
    dias = pd.DatetimeIndex([dia_base + pd.Timedelta(days=i) for i in range(n_dias)]).tz_convert("UTC").as_unit("ns").asi8
    hi = np.searchsorted(valores, dias, side="right")
    lo = np.searchsorted(valores, dias - int(ventana_h) * 3600 * 10**9, side="left")
    total = hi - lo
    lo2 = np.maximum(lo, hi - max_n)
    lista = np.full((n_dias, max_n), -1, dtype="int64")
    for i in range(n_dias):
        k = hi[i] - lo2[i]
        if k:
            lista[i, :k] = np.arange(lo2[i], hi[i])
    return lista, total


# --------------------------------------------------------------------------- codificación
def fragmentar(ids: list[int], capacidad: int, solape: int, max_frag: int):
    """Fragmentos en orden de lectura. Devuelve (fragmentos conservados, número total de fragmentos)."""
    if len(ids) <= capacidad:
        return [ids], 1
    paso = capacidad - solape
    inicios = list(range(0, len(ids) - solape, paso))
    return [ids[i:i + capacidad] for i in inicios[:max_frag]], len(inicios)


def _ficha(cfg_texto: dict, corpus: pd.DataFrame, codificador: str) -> dict:
    h = hashlib.sha256()
    for t in corpus["sha_contenido"]:
        h.update(t.encode())
    return dict(encoder=codificador, revision=cfg_texto["revision"], prefijo=cfg_texto["prefijo"], pooling=cfg_texto["pooling"],
                normalizacion=cfg_texto["normalizacion"], tokens_max=cfg_texto["tokens_max"], solapamiento_tokens=cfg_texto["solapamiento_tokens"],
                max_fragmentos_documento=cfg_texto["max_fragmentos_documento"], n_documentos=int(len(corpus)), sha256_contenidos=h.hexdigest())


def vectores_de_prueba(corpus: pd.DataFrame) -> np.ndarray:
    """Vectores seudoaleatorios deterministas por documento. SOLO para verificar la instalación."""
    out = np.empty((len(corpus), DIM), dtype="float32")
    for i, s in enumerate(corpus["sha_contenido"]):
        g = np.random.default_rng(int(s[:16], 16))
        v = g.standard_normal(DIM).astype("float32")
        out[i] = v / np.linalg.norm(v)
    return out


def _copiar_de_vecino(vecinos: Path | None, carpeta: Path, ficha: dict, log) -> dict | None:
    """Si otro paquete extraído en la misma carpeta ya calculó estos mismos vectores, se copian."""
    if vecinos is None:
        return None
    for rf2 in sorted(Path(vecinos).glob("*/resultados_*/texto/ficha.json")):
        try:
            r = json.loads(rf2.read_text(encoding="utf-8"))
            origen = rf2.parent
            if r.get("ficha") != ficha or r.get("codificador_de_prueba") or not (origen / "vectores_documento.npy").exists() or origen.resolve() == carpeta.resolve():
                continue
            for nombre in ("vectores_documento.npy", "indice.parquet", "ficha.json"):
                if (origen / nombre).exists():
                    shutil.copyfile(origen / nombre, carpeta / nombre)
            log(f"Vectores de texto copiados del paquete vecino {origen.parent.parent.name}: no se recalculan.")
            return r
        except Exception:
            continue
    return None


def codificar(corpus: pd.DataFrame, cfg_texto: dict, carpeta: Path, dispositivo: str, log, codificador: str | None = None,
              limite: int | None = None, barra=None, vecinos: Path | None = None) -> dict:
    """Calcula (o reanuda) los vectores por documento y los guarda en ``carpeta``.

    ``codificador`` permite una ruta local; por defecto se descarga el modelo fijado en el contrato.
    ``limite`` codifica solo los primeros documentos (verificación de instalación).
    """
    import torch
    import transformers
    from transformers import AutoModel, AutoTokenizer

    transformers.logging.set_verbosity_error()   # el aviso de "más de 512 tokens" no aplica: aquí se fragmenta
    try:
        transformers.utils.logging.disable_progress_bar()   # la barra de carga de pesos se mezclaría con la de avance
    except Exception:
        pass

    carpeta = Path(carpeta); carpeta.mkdir(parents=True, exist_ok=True)
    nombre = codificador or cfg_texto["encoder"]
    es_local = Path(nombre).exists()
    ficha = _ficha(cfg_texto, corpus, nombre)
    rf, rv, ri, rp = carpeta / "ficha.json", carpeta / "vectores_documento.npy", carpeta / "indice.parquet", carpeta / "parcial.npz"
    if limite is None and rf.exists() and rv.exists() and json.loads(rf.read_text(encoding="utf-8")).get("ficha") == ficha:
        log("Vectores de texto ya calculados; se reutilizan.")
        return json.loads(rf.read_text(encoding="utf-8"))
    if limite is None and not rp.exists():
        r = _copiar_de_vecino(vecinos, carpeta, ficha, log)
        if r is not None:
            if barra is not None:
                barra.update(len(corpus))
            return r

    log(f"Cargando el codificador de texto {nombre} (la primera vez descarga cerca de 1 GB)...")
    kw = {} if es_local else {"revision": cfg_texto["revision"]}
    try:
        tok = AutoTokenizer.from_pretrained(nombre, **kw)
        modelo = AutoModel.from_pretrained(nombre, **kw).to(dispositivo).eval()
    except Exception as e:
        raise RuntimeError(f"No se pudo cargar el codificador de texto {nombre} (revisión {cfg_texto['revision']}). Revise la conexión a internet y vuelva a "
                           f"ejecutar. Si el mensaje original dice que la revisión o el modelo no existen, avise al autor del paquete. Mensaje original: {e}") from e
    dim = int(modelo.config.hidden_size)
    ids_prefijo = tok(cfg_texto["prefijo"], add_special_tokens=False)["input_ids"]
    # Tokens especiales de una secuencia simple, deducidos del propio tokenizador (válido en transformers 4 y 5).
    completo = tok(cfg_texto["prefijo"], add_special_tokens=True)["input_ids"]
    k = next(i for i in range(len(completo) - len(ids_prefijo) + 1) if completo[i:i + len(ids_prefijo)] == ids_prefijo)
    inicio, cierre = completo[:k], completo[k + len(ids_prefijo):]
    capacidad = int(cfg_texto["tokens_max"]) - len(inicio) - len(cierre) - len(ids_prefijo)
    solape, max_frag = int(cfg_texto["solapamiento_tokens"]), int(cfg_texto["max_fragmentos_documento"])
    n = len(corpus) if limite is None else min(limite, len(corpus))

    vec = np.zeros((n, dim), dtype="float32")
    meta = np.zeros((n, 3), dtype="int64")  # tokens, fragmentos totales, fragmentos usados
    hecho = 0
    if limite is None and rp.exists():
        z = np.load(rp, allow_pickle=False)
        if str(z["sha"]) == ficha["sha256_contenidos"] and z["vec"].shape[1] == dim:
            hecho = int(z["hecho"]); vec[:hecho] = z["vec"][:hecho]; meta[:hecho] = z["meta"][:hecho]
            log(f"Se reanuda la codificación desde el documento {hecho} de {n}.")
    lote = 32 if dispositivo != "cpu" else 8
    bloque = 256
    t0 = time.time()
    for a in range(hecho, n, bloque):
        b = min(a + bloque, n)
        frags, dueno = [], []
        for i in range(a, b):
            ids = tok(contenido(corpus.iloc[i]), add_special_tokens=False)["input_ids"]
            fr, total = fragmentar(ids, capacidad, solape, max_frag)
            meta[i] = (len(ids), total, len(fr))
            for f in fr:
                frags.append(inicio + ids_prefijo + f + cierre); dueno.append(i - a)
        orden = np.argsort([len(f) for f in frags])
        emb = np.zeros((len(frags), dim), dtype="float32")
        for j in range(0, len(orden), lote):
            sel = orden[j:j + lote]
            largo = max(len(frags[k]) for k in sel)
            ids_t = torch.full((len(sel), largo), tok.pad_token_id, dtype=torch.long)
            att = torch.zeros((len(sel), largo), dtype=torch.long)
            for r, k in enumerate(sel):
                ids_t[r, :len(frags[k])] = torch.tensor(frags[k]); att[r, :len(frags[k])] = 1
            ids_t, att = ids_t.to(dispositivo), att.to(dispositivo)
            with torch.no_grad():                                  # precisión simple, sin autocast
                h = modelo(input_ids=ids_t, attention_mask=att).last_hidden_state.float()
            m = att.unsqueeze(-1).float()
            v = (h * m).sum(1) / m.sum(1).clamp(min=1.0)          # promedio de tokens con máscara
            v = torch.nn.functional.normalize(v, dim=-1)           # L2 por fragmento
            emb[sel] = v.cpu().numpy()
        dueno = np.asarray(dueno)
        for i in range(a, b):
            d = emb[dueno == (i - a)].mean(axis=0)
            vec[i] = d / max(float(np.linalg.norm(d)), 1e-12)       # vector del documento: promedio de fragmentos, L2
        if barra is not None:
            barra.update(b - a)
        if limite is None and (b == n or (b // bloque) % 4 == 0):
            np.savez(rp, vec=vec, meta=meta, hecho=b, sha=ficha["sha256_contenidos"])
    del modelo
    if dispositivo.startswith("cuda"):
        torch.cuda.empty_cache()
    resumen = dict(ficha=ficha, dimension=dim, documentos_codificados=int(n), segundos=round(time.time() - t0, 1), dispositivo=dispositivo,
                   documentos_truncados=int((meta[:n, 1] > meta[:n, 2]).sum()),
                   porcentaje_documentos_truncados=round(100 * float((meta[:n, 1] > meta[:n, 2]).mean()), 2),
                   porcentaje_tokens_descartados=round(100 * float(1 - np.minimum(meta[:n, 0], meta[:n, 2] * capacidad).sum() / max(meta[:n, 0].sum(), 1)), 2),
                   capacidad_tokens_por_fragmento=int(capacidad), tokens_especiales=[len(inicio), len(cierre)], tokens_prefijo=len(ids_prefijo),
                   transformers=transformers.__version__, codificador_de_prueba=False)
    if limite is None:
        np.save(rv, vec.astype("float16"))
        pd.DataFrame(dict(id=corpus["id"].to_numpy()[:n], fuente=corpus["fuente"].to_numpy()[:n], sha_contenido=corpus["sha_contenido"].to_numpy()[:n],
                          tokens=meta[:n, 0], fragmentos_totales=meta[:n, 1], fragmentos_usados=meta[:n, 2])).to_parquet(ri, index=False)
        rf.write_text(json.dumps(resumen, ensure_ascii=False, indent=1), encoding="utf-8")
        if rp.exists():
            rp.unlink()
    return resumen


def preparar_descripciones(raiz: Path, carpeta: Path, cfg_texto: dict, canales: list[str], dispositivo: str, log, modo_prueba: bool = False,
                           codificador: str | None = None) -> np.ndarray:
    """Vectores de las descripciones de canal (TGForecaster), con el mismo codificador y prefijo que los documentos."""
    desc = json.loads((Path(raiz) / "config" / "descripciones_canal.json").read_text(encoding="utf-8"))["canales"]
    faltan = [c for c in canales if c not in desc]
    if faltan:
        raise RuntimeError(f"Faltan descripciones de canal: {faltan}")
    t = pd.DataFrame(dict(id=canales, fuente="descripcion_canal", titulo=[desc[c] for c in canales], texto=""))
    t["sha_contenido"] = [hashlib.sha256(contenido(f).encode("utf-8")).hexdigest() for f in t.itertuples()]
    carpeta = Path(carpeta); carpeta.mkdir(parents=True, exist_ok=True)
    if modo_prueba:
        vec = vectores_de_prueba(t)
        (carpeta / "ficha.json").write_text(json.dumps(dict(ficha=_ficha(cfg_texto, t, "VECTORES_DE_PRUEBA"), codificador_de_prueba=True), ensure_ascii=False, indent=1), encoding="utf-8")
    else:
        codificar(t, cfg_texto, carpeta, dispositivo, log, codificador)
        vec = np.load(carpeta / "vectores_documento.npy").astype("float32")
    t[["id", "titulo"]].rename(columns=dict(id="canal", titulo="descripcion")).to_csv(carpeta / "descripciones_usadas.csv", index=False)
    return vec


def preparar_canal(raiz: Path, carpeta: Path, cfg_texto: dict, dia_base, n_dias: int, dispositivo: str, log, modo_prueba: bool = False,
                   codificador: str | None = None, barra=None) -> dict:
    """Devuelve los vectores por documento, las listas por día y la cobertura del canal oficial."""
    corpus = cargar_corpus_oficial(raiz)
    corpus["sha_contenido"] = [hashlib.sha256(contenido(f).encode("utf-8")).hexdigest() for f in corpus.itertuples()]
    carpeta = Path(carpeta); carpeta.mkdir(parents=True, exist_ok=True)
    if modo_prueba:
        vec = vectores_de_prueba(corpus)
        resumen = dict(ficha=_ficha(cfg_texto, corpus, "VECTORES_DE_PRUEBA"), codificador_de_prueba=True, dimension=DIM)
        (carpeta / "ficha.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=1), encoding="utf-8")
    else:
        resumen = codificar(corpus, cfg_texto, carpeta, dispositivo, log, codificador, barra=barra, vecinos=Path(raiz).resolve().parent)
        vec = np.load(carpeta / "vectores_documento.npy").astype("float32")
    lista, total = listas_por_dia(corpus["disponible_desde"], dia_base, n_dias, cfg_texto["ventana_h"], cfg_texto["max_documentos_canal"])
    dias = pd.DatetimeIndex([dia_base + pd.Timedelta(days=i) for i in range(n_dias)])
    usados = (lista >= 0).sum(axis=1)
    cob = pd.DataFrame(dict(dia=dias.strftime("%Y-%m-%d"), documentos_en_ventana=total, documentos_usados=usados, descartados_por_limite=total - usados))
    cob.to_csv(carpeta / "cobertura_por_dia.csv", index=False)
    return dict(vectores=vec, lista=lista, resumen=resumen, corpus=corpus, cobertura=cob)
```

### Ejecución

El pipeline común utiliza las fuentes de esta ficha y la configuración seleccionada. Los siguientes comandos ejecutan el ajuste y la evaluación.

La configuración exacta del paquete se conserva a continuación. Identifica variantes, candidatos, semillas, parada y presupuesto; se mantiene separada de las métricas calculadas a partir de artefactos.

```json
{
 "version": "modelo7_tgforecaster_paquete_20261112",
 "modelo": "Modelo 7 (TGForecaster adaptado)",
 "paquete": {
  "clave": "modelo7",
  "nombre": "Modelo 7 (TGForecaster adaptado)",
  "red": "tgforecaster",
  "carpeta_resultados": "resultados_modelo7",
  "usa_descripciones": true,
  "carpeta": "Modelo7_TGForecaster"
 },
 "contrato": "config/diseno_entrenamiento_texto.json",
 "variantes": [
  "N",
  "N+O"
 ],
 "busqueda": {
  "pliegue_inicial": 2018,
  "pliegues_restantes": [
   2019,
   2020,
   2021
  ],
  "semilla": 43,
  "finalistas": 5,
  "criterio_finalistas": "menor MASE de validación en el pliegue 2018; empates: menos parámetros y luego menor id",
  "criterio_seleccion": "menor MASE medio de los cuatro pliegues (igual peso); empates: menos parámetros y luego menor id"
 },
 "final": {
  "semillas": [
   1000,
   1001,
   1002,
   1003,
   1004
  ],
  "ajuste": "orígenes con objetivo anterior a 2022-01-01; último 10 % como parada interna con purga de 24 h",
  "prueba": "2022-01-01 a 2025-12-31, retrospectiva"
 },
 "entrenamiento": {
  "optimizador": "AdamW",
  "perdida": "MAE sobre demanda / nivel de la ventana",
  "plan": "ReduceLROnPlateau(factor 0.5, paciencia 2)",
  "max_epocas": 200,
  "min_epocas": 10,
  "paciencia": 6,
  "mejora_relativa_minima": 0.002,
  "recorte_gradiente": 1.0
 },
 "inferencia": {
  "bloque_dias": 14,
  "sensibilidad_bloque_dias": [
   7,
   28
  ],
  "remuestreos": 10000,
  "semilla_bootstrap": 20261111,
  "margen_relativo": 0.03
 },
 "arquitectura_fija": {
  "parches_historia": {
   "parche": 16,
   "paso": 8,
   "relleno_final": 8,
   "parches": 21
  },
  "parches_horizonte": {
   "parche": 8,
   "paso": 4,
   "consultas": 5
  },
  "normalizacion": "nivel de la ventana del motor del proyecto; sin normalización reversible",
  "relevancia_texto": {
   "dimension": 64,
   "consulta": "descripción de la demanda + media de las últimas 24 h y de las 168 h de la historia numérica"
  },
  "memoria": "parches de todas las variables + descripción de cada canal + 7 tokens de calendario diario",
  "consulta_horizonte": "descripción de la demanda + resumen de documentos + calendario futuro del parche + posición",
  "decodificador": "lineal por token al largo del parche y suma ponderada de los parches solapados",
  "d_ff": "2 x d_model",
  "activacion": "GELU"
 },
 "prueba_rapida": {
  "configuraciones": [
   15,
   17
  ]
 },
 "configuraciones": [
  {
   "id": 1,
   "d_model": 64,
   "capas": 2,
   "capas_mezcla": 1,
   "cabezas": 4,
   "dropout": 0.1,
   "lr": 0.001,
   "weight_decay": 0.0001,
   "lote": 64
  },
  {
   "id": 2,
   "d_model": 64,
   "capas": 2,
   "capas_mezcla": 2,
   "cabezas": 4,
   "dropout": 0.2,
   "lr": 0.001,
   "weight_decay": 0.0001,
   "lote": 64
  },
  {
   "id": 3,
   "d_model": 64,
   "capas": 3,
   "capas_mezcla": 1,
   "cabezas": 4,
   "dropout": 0.2,
   "lr": 0.0003,
   "weight_decay": 0.0001,
   "lote": 64
  },
  {
   "id": 4,
   "d_model": 64,
   "capas": 3,
   "capas_mezcla": 2,
   "cabezas": 4,
   "dropout": 0.1,
   "lr": 0.0003,
   "weight_decay": 0.001,
   "lote": 128
  },
  {
   "id": 5,
   "d_model": 64,
   "capas": 3,
   "capas_mezcla": 3,
   "cabezas": 4,
   "dropout": 0.3,
   "lr": 0.001,
   "weight_decay": 0.001,
   "lote": 128
  },
  {
   "id": 6,
   "d_model": 64,
   "capas": 2,
   "capas_mezcla": 3,
   "cabezas": 4,
   "dropout": 0.1,
   "lr": 0.0001,
   "weight_decay": 0.0001,
   "lote": 32
  },
  {
   "id": 7,
   "d_model": 128,
   "capas": 2,
   "capas_mezcla": 1,
   "cabezas": 8,
   "dropout": 0.1,
   "lr": 0.0003,
   "weight_decay": 0.0001,
   "lote": 64
  },
  {
   "id": 8,
   "d_model": 128,
   "capas": 2,
   "capas_mezcla": 2,
   "cabezas": 8,
   "dropout": 0.2,
   "lr": 0.0003,
   "weight_decay": 0.0001,
   "lote": 64
  },
  {
   "id": 9,
   "d_model": 128,
   "capas": 3,
   "capas_mezcla": 1,
   "cabezas": 8,
   "dropout": 0.2,
   "lr": 0.001,
   "weight_decay": 0.001,
   "lote": 128
  },
  {
   "id": 10,
   "d_model": 128,
   "capas": 3,
   "capas_mezcla": 2,
   "cabezas": 8,
   "dropout": 0.1,
   "lr": 0.0001,
   "weight_decay": 0.0001,
   "lote": 32
  },
  {
   "id": 11,
   "d_model": 128,
   "capas": 3,
   "capas_mezcla": 3,
   "cabezas": 8,
   "dropout": 0.2,
   "lr": 0.0003,
   "weight_decay": 0.001,
   "lote": 64
  },
  {
   "id": 12,
   "d_model": 128,
   "capas": 3,
   "capas_mezcla": 3,
   "cabezas": 8,
   "dropout": 0.3,
   "lr": 0.001,
   "weight_decay": 0.01,
   "lote": 128
  },
  {
   "id": 13,
   "d_model": 128,
   "capas": 2,
   "capas_mezcla": 3,
   "cabezas": 16,
   "dropout": 0.2,
   "lr": 0.0001,
   "weight_decay": 0.0001,
   "lote": 128
  },
  {
   "id": 14,
   "d_model": 128,
   "capas": 3,
   "capas_mezcla": 2,
   "cabezas": 16,
   "dropout": 0.3,
   "lr": 0.0003,
   "weight_decay": 0.0001,
   "lote": 64
  },
  {
   "id": 15,
   "d_model": 32,
   "capas": 3,
   "capas_mezcla": 1,
   "cabezas": 4,
   "dropout": 0.1,
   "lr": 0.001,
   "weight_decay": 0.0001,
   "lote": 64
  },
  {
   "id": 16,
   "d_model": 32,
   "capas": 3,
   "capas_mezcla": 2,
   "cabezas": 4,
   "dropout": 0.2,
   "lr": 0.001,
   "weight_decay": 0.001,
   "lote": 64
  },
  {
   "id": 17,
   "d_model": 32,
   "capas": 2,
   "capas_mezcla": 3,
   "cabezas": 4,
   "dropout": 0.3,
   "lr": 0.0003,
   "weight_decay": 0.0001,
   "lote": 128
  },
  {
   "id": 18,
   "d_model": 256,
   "capas": 2,
   "capas_mezcla": 1,
   "cabezas": 8,
   "dropout": 0.2,
   "lr": 0.0001,
   "weight_decay": 0.001,
   "lote": 32
  },
  {
   "id": 19,
   "d_model": 256,
   "capas": 3,
   "capas_mezcla": 2,
   "cabezas": 8,
   "dropout": 0.1,
   "lr": 0.0001,
   "weight_decay": 0.0001,
   "lote": 32
  },
  {
   "id": 20,
   "d_model": 256,
   "capas": 2,
   "capas_mezcla": 2,
   "cabezas": 8,
   "dropout": 0.3,
   "lr": 0.0003,
   "weight_decay": 0.01,
   "lote": 128
  }
 ],
 "desviaciones_declaradas": [
  "Presupuesto: el contrato registra 20 configuraciones x 3 semillas (43, 44, 45) x 4 pliegues. Por tiempo y recursos, y antes de entrenar, el autor fijó 20 configuraciones x 1 semilla (43) en el pliegue 2018 y las 5 mejores en los pliegues 2019-2021. La comparación con otros modelos solo es equitativa si reciben el mismo esquema.",
  "La preselección en un solo pliegue y con una sola semilla está expuesta al ruido de inicialización; el orden de las 20 configuraciones no se interpreta como evidencia.",
  "Semillas finales: 5 (1000-1004). Es el mínimo de piloto del diseño; no acredita por sí mismo potencia 0,80.",
  "El bloque 2022-2025 ya fue examinado con otros modelos del proyecto: la prueba es retrospectiva y no constituye evidencia confirmatoria independiente.",
  "Texto: documentos oficiales con disponibilidad en [origen - 72 h, origen], como máximo los 64 más recientes; resumen por relevancia aprendida cuya consulta usa solo la historia numérica. Sin documentos el resumen vale cero.",
  "Se entrenan N y N+O. La variante N+O+P (prensa) no forma parte de este paquete.",
  "Ausencias numéricas: mediana del ajuste de cada pliegue y un indicador por bloque (precio, hidrología, clima), porque las columnas de cada bloque comparten el patrón de ausencia.",
  "El artículo de TGForecaster (arXiv:2405.13522) no publica número de capas, dimensión, cabezas ni hiperparámetros de entrenamiento; remite a su código. Solo fija parches de 6 con paso 3 (16 y 8 en su conjunto sintético). Este paquete no adopta, por tanto, una configuración publicada: el modelo no queda exento de la búsqueda de arquitecturas del E2 §14, y las 20 configuraciones son la búsqueda de hiperparámetros acordada, no esa búsqueda.",
  "El artículo usa texto del periodo que se pronostica. Aquí no existe texto futuro: se usan solo documentos disponibles en el origen o antes.",
  "Consultas del horizonte: 5 parches de 8 h con paso 4. Cada consulta suma la descripción de la demanda, el resumen de documentos, el calendario futuro de su parche y una posición aprendida. El decodificador proyecta cada token a su parche y combina los solapados con pesos aprendidos que suman 1 por hora.",
  "Memoria de la atención cruzada: parches de todas las variables de la historia (no solo de la demanda), identificadas con su descripción, más siete tokens con el calendario diario de la historia. Es una adaptación propia: el artículo trata cada canal por separado y pronostica todos; aquí solo se pronostica la demanda y las covariables deben poder informarla.",
  "Relevancia de documentos: una sola etapa con pesos aprendidos, cuya consulta combina la descripción de la demanda con la historia numérica (regla del contrato), en lugar de los bloques de atención cruzada del artículo.",
  "Parches de la historia de 16 h con paso 8, como en el modelo 6. Sin normalización reversible, igual que el artículo; la demanda se normaliza con el nivel de la ventana, como en el resto del proyecto.",
  "Descripciones de canal: tomadas del diccionario de datos del proyecto (sin la cláusula de desplazamiento, con su unidad); las tres de ausencia se redactaron para este paquete. Se codifican con el mismo codificador congelado que los documentos. La variante N las conserva: no son información externa.",
  "Durante la construcción del paquete se revisó la convergencia solo con el ajuste y la parada interna del pliegue 2018 (sin el año de validación ni la prueba). Las 20 configuraciones no se modificaron después."
 ],
 "manual": {
  "titulo": "Entrenamiento del modelo 7",
  "zip": "Modelo7_TGForecaster.zip",
  "que_se_entrena": [
   "Modelo 7 del Top 10: <b>TGForecaster adaptado</b>. Un codificador de parches procesa cada variable de la historia; cinco consultas, una por parche del horizonte, combinan la descripción de la demanda, el resumen de los documentos oficiales y el calendario futuro, y leen la historia por atención cruzada; un decodificador por tokens arma las 24 h.",
   "Variantes <b>N</b> (sin documentos; conserva las descripciones de canal) y <b>N+O</b> (con documentos oficiales), cada una con su propia búsqueda de 20 configuraciones."
  ],
  "fijo": "Parches de la historia 16/8; parches del horizonte 8/4. capas es la profundidad del codificador y capas_mezcla la de la atención cruzada.",
  "columnas": [
   [
    "id",
    "#",
    0.8
   ],
   [
    "d_model",
    "d_model",
    2.0
   ],
   [
    "capas",
    "Capas",
    1.7
   ],
   [
    "capas_mezcla",
    "Capas de mezcla",
    2.6
   ],
   [
    "cabezas",
    "Cabezas",
    2.0
   ],
   [
    "dropout",
    "Dropout",
    2.0
   ],
   [
    "lr",
    "lr",
    1.7
   ],
   [
    "weight_decay",
    "weight decay",
    2.6
   ],
   [
    "lote",
    "Lote",
    1.8
   ]
  ]
 }
}
```


## Evaluación

Las variantes N y N+O conservan cinco semillas finales cada una, con métricas y predicciones para 2022–2025. Su contraste documental y la inferencia conjunta se presentan en la evaluación integrada. Las descripciones estáticas están presentes en ambas variantes; la comparación aísla el procedimiento con y sin documentos oficiales variables en el tiempo, con hiperparámetros seleccionados por separado.

## Fuentes

[Xu et al., TGForecaster, versión 2](https://arxiv.org/html/2405.13522v2). La descripción corresponde a la implementación del experimento; las adaptaciones se identifican en el texto.

# 10.8. TFT: selección de variables y atención interpretable

La notación y las operaciones compartidas definen los tensores, el nivel de demanda, la atención, las activaciones y la optimización.

## Arquitectura y fundamentos

TFT combina selección de variables, memoria recurrente y atención. Primero decide qué variables ponderar en cada hora; después resume la secuencia y permite que cada hora del horizonte consulte los pasos relevantes de la historia. Las compuertas residuales regulan cuánto de cada transformación se incorpora.

La configuración seleccionada tiene ancho 160 y cuatro cabezas. Recibe 56 variables pasadas: 30 históricas, tres indicadores de ausencia por bloque y 23 de calendario. El futuro contiene únicamente 23 variables de calendario. Cada variable continua tiene un embedding lineal y una GRN propia. Una segunda GRN produce pesos softmax $v_{s,j}$; su suma es uno y la representación seleccionada es $z_s=\sum_jv_{s,j}\operatorname{GRN}_j(e_{s,j})$.

Una GRN utiliza una transformación no lineal, una compuerta y un atajo:

$$\operatorname{GRN}(a)=\operatorname{LayerNorm}\!\left(Sa+\operatorname{GLU}(W_1\operatorname{ELU}(W_2a+b_2)+b_1)\right),\quad
\operatorname{GLU}(u)=\sigma(W_gu+b_g)\odot(W_vu+b_v).$$

El codificador LSTM procesa las 168 horas y su estado inicializa al decodificador que recibe el calendario futuro. El enriquecimiento GRN precede a la atención interpretable: las cabezas comparten la proyección de valores y se promedian. La máscara impide que la consulta de la hora $h$ utilice representaciones del decodificador posteriores a ella. Las GRN por posición y las compuertas finales producen una salida escalar por hora. El total es de 11.715.596 parámetros.

Se omiten codificadores estáticos porque se estudia una sola serie. Se entrena un pronóstico puntual con MAE normalizado, no todos los cuantiles del artículo. Los pesos de selección describen el mecanismo del modelo; no son efectos causales de las variables. Esta documentación y los resultados usan la configuración TFT seleccionada del paquete, identificada por sus fuentes y checkpoints.

### Esquema global

[![Arquitectura global de tft](https://demanda-electrica-dashboard.onrender.com/architectures/tft_global.png)](https://demanda-electrica-dashboard.onrender.com/architectures/tft_global.png)

La selección de variables transforma las 56 entradas históricas y las 23 de calendario futuro a representaciones de dimensión 160. La recurrencia, las compuertas y la atención de cuatro cabezas producen una salida por hora.

### Detalle de las operaciones

[![Detalle de la arquitectura de tft](https://demanda-electrica-dashboard.onrender.com/architectures/tft_detalle.png)](https://demanda-electrica-dashboard.onrender.com/architectures/tft_detalle.png)

Las GRN combinan transformaciones no lineales con conexiones residuales y compuertas. La selección softmax pondera las representaciones de las variables en cada instante; esos pesos describen el mecanismo interno y no efectos causales.

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
```

## Pipeline de entrenamiento y prueba

<!-- entrenamiento-portable:inicio -->
### Comandos de entrenamiento y evaluación

Validación temporal de la configuración seleccionada:

```bash
python scripts/entrenar_arquitectura.py --modelo tft --anio 2021 --semilla 1000 --dispositivo auto --salida salidas/tft_validacion_1000
```

Comprobación funcional breve:

```bash
python scripts/entrenar_arquitectura.py --modelo tft --smoke --dispositivo cpu --salida salidas/tft_funcional
```

Evaluación final autorizada, con ajuste y parada anteriores a 2022:

```bash
python scripts/entrenar_arquitectura.py --modelo tft --fase prueba --allow-test --semilla 1000 --dispositivo auto --salida salidas/tft_prueba_1000
```

El código completo de preparación, optimización, guardado y restauración está en el pipeline ejecutable compartido. Cada salida debe ser una carpeta nueva.
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
        vl = _perdida(red, es, int(opciones.get("lote_evaluacion", 256)))
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


def _evaluar(datos: DatosNumericos, red, conj, t, lote: int = 256) -> tuple[np.ndarray, dict]:
    yhat = motor.predecir(red, t, lote).astype("float64") * conj.nivel[:, None]
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
            le = int(ctx.opciones.get("lote_evaluacion", 256))
            yhat, met = _evaluar(datos, red, conj["ev"], tens["ev"], le)
            mase_tr = _evaluar(datos, red, conj["tr"], tens["tr"], le)[1]["MASE"]
            mase_es = _evaluar(datos, red, conj["es"], tens["es"], le)[1]["MASE"]
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

### Ejecución

El pipeline común utiliza las fuentes de esta ficha y la configuración seleccionada. Los siguientes comandos ejecutan el ajuste y la evaluación.

La configuración exacta del paquete se conserva a continuación. Identifica variantes, candidatos, semillas, parada y presupuesto; se mantiene separada de las métricas calculadas a partir de artefactos.

```json
{
 "version": "modelo3_tft_paquete_20261114",
 "modelo": "Modelo 3 (TFT)",
 "paquete": {
  "clave": "modelo3",
  "nombre": "Modelo 3 (TFT)",
  "red": "tft",
  "carpeta_resultados": "resultados_modelo3",
  "usa_descripciones": false,
  "carpeta": "Modelo3_TFT"
 },
 "contrato": "config/diseno_entrenamiento_texto.json",
 "variantes": [
  "N"
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
  "recorte_gradiente": 1.0,
  "lote_evaluacion": 64
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
  "entradas_pasadas": "30 históricas + 3 indicadores de ausencia + 23 de calendario, 168 h",
  "entradas_futuras_conocidas": "23 de calendario, 24 h",
  "componentes": [
   "selección de variables (una GRN por variable)",
   "codificador y decodificador LSTM de una capa",
   "compuerta + suma + LayerNorm",
   "enriquecimiento con GRN",
   "atención multicabeza interpretable con máscara causal",
   "GRN por posición"
  ],
  "estaticas": "no hay (una sola serie)",
  "salida": "una proyección lineal por hora del horizonte; la red devuelve 1 + corrección"
 },
 "prueba_rapida": {
  "configuraciones": [
   19,
   20
  ]
 },
 "configuraciones": [
  {
   "id": 1,
   "d_model": 160,
   "cabezas": 4,
   "dropout": 0.1,
   "lr": 0.001,
   "weight_decay": 0.0,
   "lote": 32
  },
  {
   "id": 2,
   "d_model": 160,
   "cabezas": 4,
   "dropout": 0.1,
   "lr": 0.0003,
   "weight_decay": 0.0,
   "lote": 32
  },
  {
   "id": 3,
   "d_model": 160,
   "cabezas": 4,
   "dropout": 0.2,
   "lr": 0.001,
   "weight_decay": 0.0001,
   "lote": 32
  },
  {
   "id": 4,
   "d_model": 160,
   "cabezas": 4,
   "dropout": 0.3,
   "lr": 0.0003,
   "weight_decay": 0.001,
   "lote": 32
  },
  {
   "id": 5,
   "d_model": 160,
   "cabezas": 1,
   "dropout": 0.1,
   "lr": 0.001,
   "weight_decay": 0.0001,
   "lote": 32
  },
  {
   "id": 6,
   "d_model": 160,
   "cabezas": 4,
   "dropout": 0.1,
   "lr": 0.0001,
   "weight_decay": 0.0001,
   "lote": 32
  },
  {
   "id": 7,
   "d_model": 80,
   "cabezas": 4,
   "dropout": 0.1,
   "lr": 0.001,
   "weight_decay": 0.0,
   "lote": 64
  },
  {
   "id": 8,
   "d_model": 80,
   "cabezas": 4,
   "dropout": 0.1,
   "lr": 0.0003,
   "weight_decay": 0.0001,
   "lote": 64
  },
  {
   "id": 9,
   "d_model": 80,
   "cabezas": 4,
   "dropout": 0.2,
   "lr": 0.001,
   "weight_decay": 0.0001,
   "lote": 128
  },
  {
   "id": 10,
   "d_model": 80,
   "cabezas": 4,
   "dropout": 0.3,
   "lr": 0.001,
   "weight_decay": 0.001,
   "lote": 128
  },
  {
   "id": 11,
   "d_model": 80,
   "cabezas": 1,
   "dropout": 0.1,
   "lr": 0.001,
   "weight_decay": 0.0001,
   "lote": 64
  },
  {
   "id": 12,
   "d_model": 80,
   "cabezas": 4,
   "dropout": 0.2,
   "lr": 0.0003,
   "weight_decay": 0.001,
   "lote": 32
  },
  {
   "id": 13,
   "d_model": 80,
   "cabezas": 4,
   "dropout": 0.1,
   "lr": 0.0001,
   "weight_decay": 0.0001,
   "lote": 32
  },
  {
   "id": 14,
   "d_model": 80,
   "cabezas": 1,
   "dropout": 0.2,
   "lr": 0.0003,
   "weight_decay": 0.0001,
   "lote": 128
  },
  {
   "id": 15,
   "d_model": 40,
   "cabezas": 4,
   "dropout": 0.1,
   "lr": 0.001,
   "weight_decay": 0.0,
   "lote": 64
  },
  {
   "id": 16,
   "d_model": 40,
   "cabezas": 4,
   "dropout": 0.1,
   "lr": 0.0003,
   "weight_decay": 0.0001,
   "lote": 64
  },
  {
   "id": 17,
   "d_model": 40,
   "cabezas": 4,
   "dropout": 0.2,
   "lr": 0.001,
   "weight_decay": 0.0001,
   "lote": 128
  },
  {
   "id": 18,
   "d_model": 40,
   "cabezas": 4,
   "dropout": 0.3,
   "lr": 0.001,
   "weight_decay": 0.001,
   "lote": 128
  },
  {
   "id": 19,
   "d_model": 40,
   "cabezas": 1,
   "dropout": 0.1,
   "lr": 0.001,
   "weight_decay": 0.0001,
   "lote": 32
  },
  {
   "id": 20,
   "d_model": 40,
   "cabezas": 4,
   "dropout": 0.2,
   "lr": 0.0001,
   "weight_decay": 0.0001,
   "lote": 32
  }
 ],
 "desviaciones_declaradas": [
  "Presupuesto: el contrato registra 20 configuraciones x 3 semillas (43, 44, 45) x 4 pliegues. Por tiempo y recursos, y antes de entrenar, el autor fijó 20 configuraciones x 1 semilla (43) en el pliegue 2018 y las 5 mejores en los pliegues 2019-2021. La comparación con otros modelos solo es equitativa si reciben el mismo esquema.",
  "La preselección en un solo pliegue y con una sola semilla está expuesta al ruido de inicialización; el orden de las 20 configuraciones no se interpreta como evidencia.",
  "Semillas finales: 5 (1000-1004). Es el mínimo de piloto del diseño; no acredita por sí mismo potencia 0,80.",
  "El bloque 2022-2025 ya fue examinado con otros modelos del proyecto: la prueba es retrospectiva y no constituye evidencia confirmatoria independiente.",
  "Ausencias numéricas: mediana del ajuste de cada pliegue y un indicador por bloque (precio, hidrología, clima), porque las columnas de cada bloque comparten el patrón de ausencia.",
  "Modelo solo numérico: se entrena únicamente la variante N. No usa documentos ni codificador de texto.",
  "TFT (Lim et al., 2021, arXiv:1912.09363), implementación propia. Se omiten los codificadores de covariables estáticas porque hay una sola serie; todas las entradas son continuas, de modo que cada variable usa un embedding lineal.",
  "Pronóstico puntual con pérdida MAE normalizada (regla del proyecto) en lugar de la pérdida de cuantiles del artículo; equivale a entrenar solo el cuantil 0,5. La demanda se normaliza con el nivel de la ventana, como en el resto del proyecto.",
  "La configuración 1 usa tamaño oculto 160, 4 cabezas y dropout 0,1, los valores con que se replica el experimento de electricidad del artículo (historia de 168 h y horizonte de 24 h). Las 20 configuraciones incluyen tamaño oculto y cabezas además de tasa, lote y regularización; el diseño separa los hiperparámetros no arquitectónicos de la arquitectura publicada.",
  "El recorte de gradiente es 1 (regla del proyecto) y el optimizador AdamW con decaimiento de pesos como hiperparámetro.",
  "Por memoria de GPU, las configuraciones con tamaño oculto 160 usan lote 32 y la evaluación se hace en lotes de 64: la selección de variables guarda una representación por variable y por hora (56 variables x 168 h).",
  "Durante la construcción del paquete la red se revisó solo con el ajuste y la parada interna del pliegue 2018 (sin el año de validación ni la prueba)."
 ]
}
```


## Evaluación

Los resultados de esta familia se integran en la evaluación estadística conjunta, con métricas por semilla, horizonte y origen. La comparación distingue las entradas demanda/calendario de las entradas numéricas ampliadas y el aporte de texto oficial.

## Fuentes

[Lim et al., TFT](https://arxiv.org/abs/1912.09363). La descripción corresponde a la implementación del experimento; las adaptaciones se identifican en el texto.

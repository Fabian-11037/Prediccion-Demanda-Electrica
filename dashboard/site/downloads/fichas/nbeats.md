# 10.14. N-BEATS: reconstrucción y pronóstico residual

## Arquitectura y fundamentos

N-BEATS divide el pronóstico en contribuciones sucesivas. Cada bloque recibe el residuo de la demanda, reconstruye una parte de la historia (*backcast*) y produce un pronóstico parcial. La reconstrucción se resta de la entrada del bloque siguiente; los pronósticos se suman. Así, los bloques pueden representar estructuras temporales complementarias sin una descomposición estacional impuesta.

La configuración seleccionada usa 30 bloques independientes, cada uno con cuatro capas densas de 512 unidades, ReLU y dropout 0,1. El residuo contiene las 168 horas de demanda en orden inverso. Cada bloque incorpora además las últimas 24 horas de las 29 covariables y tres máscaras de ausencia, y las 23 variables de calendario para las 24 horas futuras: la entrada tiene dimensión $168+24\times32+24\times23=1488$.

$$u_b=[r_b,\operatorname{vec}(X_{exo,24}),\operatorname{vec}(C_f)],\quad
h_b=\operatorname{MLP}_b(u_b),\quad [\widehat b_b,\widehat f_b]=W_bh_b+a_b,$$
$$r_{b+1}=r_b-\widehat b_b,\qquad
\widehat{\mathbf y}=\ell_t\left(\mathbf1+\sum_{b=1}^{30}\widehat f_b\right).$$

La base genérica convierte directamente los coeficientes aprendidos en reconstrucción y pronóstico; no corresponde a las bases interpretables de tendencia y estacionalidad de otra variante de N-BEATS. Las covariables se mantienen en todos los bloques y solo se actualiza el residuo de demanda. El modelo contiene 49.464.960 parámetros. Su coste denso crece aproximadamente como $O(B[d_{in}w+3w^2+w(L+H)])$, con $B=30$ bloques y ancho $w=512$.

La arquitectura sigue el mecanismo de doble residuo de [Oreshkin et al., N-BEATS, ICLR 2020](https://arxiv.org/abs/1905.10437), ampliado con las covariables numéricas y el calendario del proyecto.

## Esquema global

[![Arquitectura global de nbeats](https://demanda-electrica-dashboard.onrender.com/architectures/nbeats_global.png)](https://demanda-electrica-dashboard.onrender.com/architectures/nbeats_global.png)

Los bloques reducen sucesivamente el residuo de la demanda y suman sus contribuciones a las siguientes 24 horas.

## Detalle de las operaciones

[![Operaciones de nbeats](https://demanda-electrica-dashboard.onrender.com/architectures/nbeats_detalle.png)](https://demanda-electrica-dashboard.onrender.com/architectures/nbeats_detalle.png)

La reconstrucción de 168 valores y el pronóstico de 24 valores son dos salidas del mismo bloque; solo la reconstrucción modifica la entrada del siguiente.

El código de generación de estos esquemas conserva sus dimensiones y conexiones.


## Código de la arquitectura

```python
import torch
from torch import nn


class _Base(nn.Module):
    def diagnostico(self) -> dict:
        return {}

    def n_parametros(self) -> int:
        return sum(v.numel() for v in self.parameters() if v.requires_grad)

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
```

## Pipeline de entrenamiento y prueba

<!-- entrenamiento-portable:inicio -->
### Comandos de entrenamiento y evaluación

Validación temporal de la configuración seleccionada:

```bash
python scripts/entrenar_arquitectura.py --modelo nbeats --anio 2021 --semilla 1000 --dispositivo auto --salida salidas/nbeats_validacion_1000
```

Comprobación funcional breve:

```bash
python scripts/entrenar_arquitectura.py --modelo nbeats --smoke --dispositivo cpu --salida salidas/nbeats_funcional
```

Evaluación final autorizada, con ajuste y parada anteriores a 2022:

```bash
python scripts/entrenar_arquitectura.py --modelo nbeats --fase prueba --allow-test --semilla 1000 --dispositivo auto --salida salidas/nbeats_prueba_1000
```

El código completo de preparación, optimización, guardado y restauración está en el pipeline ejecutable compartido. Cada salida debe ser una carpeta nueva.
<!-- entrenamiento-portable:fin -->

Se comparan 20 configuraciones en 2018; cinco finalistas se evalúan en 2019–2021 y se selecciona la menor media de los cuatro años. El ajuste final utiliza cinco semillas (1000–1004), máximo de 200 épocas, mínimo de 10 y paciencia de 6. El checkpoint se restaura en el mínimo de parada interna. La reserva contiene 803 orígenes de 2019-10-21 a 2021-12-31; la prueba comprende 1.461 orígenes de 2022–2025. Las imputaciones y escalas se ajustan con el tramo de entrenamiento. Precio, hidrología y clima se usan como historia con sus rezagos de disponibilidad.

### Preparación de datos

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

### Selección y ajuste con parada temprana

```python
"""Etapas del paquete de cierre del Entregable 2 sobre el protocolo común (``protocolo.py``).

* ``modelo_estandar``: 20 configuraciones en el pliegue 2018, 5 finalistas en 2019-2021, 5 semillas finales.
* ``busqueda14``: búsqueda de arquitectura de §14 en dos fases con estudio piloto del ruido por semilla.
* ``ablacion``: componentes del mejor modelo numérico con las mismas cinco semillas finales.
* ``sanidad``: etiquetas permutadas en ajuste y parada (§10.1); la prueba no se toca para nada más.
Todas las corridas son reanudables: si una carpeta ya tiene ``resultado.json`` no se repite.
"""
from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from . import protocolo as PR
from .protocolo import corrida, guardar_json

ANCHOS = (16, 32, 64, 128, 256)


# ------------------------------------------------------------------------------------ utilidades
def arquitecturas() -> list[dict]:
    """A1 de §14.4: las 1000 arquitecturas, producto cartesiano de W^3 y {0,1}^3, con identificador estable."""
    out = []
    for i, (w, b) in enumerate(itertools.product(itertools.product(ANCHOS, repeat=3), itertools.product((0, 1), repeat=3))):
        out.append(dict(arq=f"A{i:04d}", anchos=list(w), patron_dropout=list(b)))
    assert len(out) == 1000 and sum(1 for a in out if sum(a["patron_dropout"]) == 0) == 125
    return out


def _pos(ctx, anio: int) -> dict:
    f = next(x for x in ctx.datos.pliegues() if x["anio"] == anio)
    return ctx.recortar(dict(tr=f["tr"], es=f["es"], ev=f["val"]))


def _pos_final(ctx) -> dict:
    tr, es = ctx.datos.ajuste_final()
    return ctx.recortar(dict(tr=tr, es=es, ev=ctx.datos.prueba()))


def _fila(r: dict, extra: dict | None = None) -> dict:
    f = PR._fila(r)
    f.update(extra or {})
    for k in ("anchos", "patron_dropout"):
        if k in f and isinstance(f[k], list):
            f[k] = "-".join(map(str, f[k]))
    return f


# ------------------------------------------------------------------------------------ modelo estándar
def modelo_estandar(ctx, configs: list[dict]) -> dict:
    finalistas = PR.etapa_busqueda(ctx, configs)
    elegidas = PR.etapa_pliegues(ctx, configs, finalistas)
    return finales_con_costes(ctx, configs, elegidas)


def finales_con_costes(ctx, configs: list[dict], elegidas: dict) -> dict:
    """Etapa final del protocolo común; además mide costes (FLOPs, memoria, energía, latencia) en cada corrida."""
    original = PR.corrida

    def con_medicion(*a, **k):
        k.setdefault("medir", True)
        return original(*a, **k)

    PR.corrida = con_medicion
    try:
        return PR.etapa_final(ctx, configs, elegidas)
    finally:
        PR.corrida = original


# ------------------------------------------------------------------------------------ búsqueda de §14
class Busqueda14:
    def __init__(self, ctx, plan: dict, modelo: dict):
        self.ctx, self.plan, self.m = ctx, plan, modelo
        self.raiz = ctx.salida
        self.vb = modelo["variante_busqueda"]
        self.control = dict(anchos=list(plan["control"]["anchos"]), patron_dropout=list(plan["control"]["patron_dropout"]))
        self.f1 = plan["fase1"]
        self.rapido = ctx.rapido

    # fase 0: hiperparámetros con la arquitectura de control (§14.3)
    def fase0(self) -> dict:
        r0 = self.raiz / "fase0_hiperparametros"
        if "fija" in self.m["fase0"]:
            p = dict(self.m["fase0"]["fija"])
            r0.mkdir(parents=True, exist_ok=True)
            guardar_json(r0 / "elegida.json", dict(configuracion=p, origen=self.m["fase0"]["origen"]))
            return p
        configs = [dict(c, **self.control) for c in self.m["fase0"]["configuraciones"]]
        if self.rapido:
            configs = [c for c in configs if c["id"] in self.rapido["configuraciones"]]
        sub = PR.Contexto(self.ctx.raiz, r0, dict(self.ctx.cfg, variantes=[self.vb]), self.ctx.datos, self.ctx.dispositivo, self.ctx.log, self.rapido)
        self._heredar(sub)
        finalistas = PR.etapa_busqueda(sub, configs)
        elegidas = PR.etapa_pliegues(sub, configs, finalistas)
        p = next(c for c in configs if c["id"] == elegidas[self.vb])
        guardar_json(r0 / "elegida.json", dict(configuracion=p, regla="protocolo común con la arquitectura de control"))
        return {k: v for k, v in p.items() if k not in ("anchos", "patron_dropout")}

    def _heredar(self, sub):
        sub.texto, sub.descripciones, sub.barra = self.ctx.texto, self.ctx.descripciones, self.ctx.barra
        sub.mlflow, sub.mlflow_raiz = getattr(self.ctx, "mlflow", False), getattr(self.ctx, "mlflow_raiz", self.ctx.salida)

    def _p(self, base: dict, a: dict, ident: int) -> dict:
        return dict(base, id=ident, arq=a["arq"], anchos=a["anchos"], patron_dropout=a["patron_dropout"])

    def _opciones_f1(self) -> dict:
        if self.rapido:
            return {}
        return dict(max_epocas=self.f1["max_epocas"], min_epocas=self.f1["min_epocas"], paciencia=self.f1["paciencia"])

    def fase1(self, base: dict) -> pd.DataFrame:
        arqs = arquitecturas()
        if self.rapido:
            arqs = [a for a in arqs if a["arq"] in self.rapido["arquitecturas"]]
        pos = _pos(self.ctx, self.f1["pliegue"])
        filas = []
        for i, a in enumerate(arqs):
            p = self._p(base, a, 1000 + i)
            c = self.raiz / "fase1" / a["arq"]
            r = corrida(self.ctx, self.vb, p, self.f1["semilla"], f"pliegue_{self.f1['pliegue']}", pos, c, f"val_{self.f1['pliegue']}", opciones=self._opciones_f1())
            filas.append(_fila(r, dict(arq=a["arq"])))
        t = pd.DataFrame(filas).sort_values(["MASE", "n_parametros", "arq"]).reset_index(drop=True)
        t["puesto"] = np.arange(1, len(t) + 1)
        t.to_csv(self.raiz / "fase1" / "resultados_fase1.csv", index=False)
        return t

    def piloto(self, base: dict, t1: pd.DataFrame) -> dict:
        """Ruido por semilla con el presupuesto de la fase 1 sobre arquitecturas fijadas a priori."""
        pl = self.plan["piloto"]
        arqs = {a["arq"]: a for a in arquitecturas()}
        ids = [a for a in pl["arquitecturas"] if a in arqs]
        if self.rapido:
            ids = [a for a in ids if a in self.rapido["arquitecturas"]][:2]
        semillas = pl["semillas"][:2] if self.rapido else pl["semillas"]
        pos = _pos(self.ctx, self.f1["pliegue"])
        filas = []
        for a in ids:
            for s in semillas:
                r = corrida(self.ctx, self.vb, self._p(base, arqs[a], 900), s, f"pliegue_{self.f1['pliegue']}", pos, self.raiz / "piloto" / a / f"semilla_{s}",
                            f"val_{self.f1['pliegue']}", opciones=self._opciones_f1())
                filas.append(dict(arq=a, semilla=s, MASE=r["metricas"]["MASE"]))
        t = pd.DataFrame(filas)
        t.to_csv(self.raiz / "piloto" / "piloto_ruido.csv", index=False)
        ruido = float(t.groupby("arq").MASE.std(ddof=1).mean())
        dispersion = float(t1.MASE.std(ddof=1))
        razon = ruido / dispersion if dispersion > 0 else float("inf")
        regla = self.plan["fase2"]["k_regla"]
        k = regla["k_bajo"] if razon <= regla["umbral_razon"] else regla["k_alto"]
        if self.rapido:
            k = self.rapido["k"]
        d = dict(ruido_semilla_desv_media=ruido, dispersion_entre_arquitecturas=dispersion, razon=razon, k=int(k), regla=regla)
        guardar_json(self.raiz / "piloto" / "eleccion_k.json", d)
        return d

    def fase2(self, base: dict, t1: pd.DataFrame, k: int) -> pd.DataFrame:
        arqs = {a["arq"]: a for a in arquitecturas()}
        ctrl = next(a["arq"] for a in arqs.values() if a["anchos"] == self.control["anchos"] and a["patron_dropout"] == self.control["patron_dropout"])
        finalistas = list(t1.arq.head(k))
        if ctrl not in finalistas:
            finalistas.append(ctrl)
        semillas = self.rapido["semillas_finales"] if self.rapido else self.plan["fase2"]["semillas"]
        pos = _pos(self.ctx, self.f1["pliegue"])
        filas = []
        for a in finalistas:
            for s in semillas:
                r = corrida(self.ctx, self.vb, self._p(base, arqs[a], 2000), s, f"pliegue_{self.f1['pliegue']}", pos, self.raiz / "fase2" / a / f"semilla_{s}",
                            f"val_{self.f1['pliegue']}")
                filas.append(_fila(r, dict(arq=a, control=a == ctrl)))
        t = pd.DataFrame(filas)
        t.to_csv(self.raiz / "fase2" / "resultados_fase2_por_semilla.csv", index=False)
        return t

    def seleccionar(self, t1: pd.DataFrame, t2: pd.DataFrame) -> dict:
        """Regla de §14.5 declarada a priori: menor MASE medio; empate técnico (diferencia relativa menor que el
        margen) -> menos parámetros; después menor identificador."""
        margen = self.plan["seleccion"]["margen_relativo"]
        r = (t2.groupby("arq").agg(MASE_medio=("MASE", "mean"), MASE_desv=("MASE", "std"), semillas=("MASE", "size"), n_parametros=("n_parametros", "first"),
                                   brecha_ajuste_parada=("MASE_parada", "mean"), MASE_ajuste=("MASE_ajuste", "mean"), segundos=("segundos", "mean"),
                                   control=("control", "first")).reset_index().sort_values(["MASE_medio", "n_parametros", "arq"]).reset_index(drop=True))
        r["brecha_ajuste_parada"] = r["brecha_ajuste_parada"] - r["MASE_ajuste"]
        mejor = float(r.MASE_medio.iloc[0])
        r["empate_tecnico"] = r.MASE_medio <= mejor * (1 + margen)
        # Conjunto indistinguible por TOST (Welch) con margen relativo +-3 % respecto del mejor
        ref = t2.loc[t2.arq == r.arq.iloc[0], "MASE"].to_numpy()
        tost = []
        for a in r.arq:
            v = t2.loc[t2.arq == a, "MASE"].to_numpy()
            if a == r.arq.iloc[0] or len(v) < 2:
                tost.append(np.nan)
                continue
            m = margen * mejor
            p1 = stats.ttest_ind(v, ref - m, equal_var=False, alternative="greater").pvalue     # H0: media(v) - media(ref) <= -m
            p2 = stats.ttest_ind(v, ref + m, equal_var=False, alternative="less").pvalue        # H0: media(v) - media(ref) >= +m
            tost.append(float(max(p1, p2)))
        r["p_TOST_frente_al_mejor"] = tost
        r["indistinguible_TOST"] = [True if a == r.arq.iloc[0] else (p < 0.05) for a, p in zip(r.arq, tost)]
        cand = r.loc[r.empate_tecnico].sort_values(["n_parametros", "MASE_medio", "arq"])
        gan = cand.iloc[0]
        r["ganadora"] = r.arq == gan.arq
        rf1 = t1.set_index("arq").puesto
        rho = stats.spearmanr(rf1.reindex(r.arq).to_numpy(), r.MASE_medio.rank().to_numpy()).statistic if len(r) > 2 else np.nan
        r.to_csv(self.raiz / "fase2" / "resumen_finalistas.csv", index=False)
        arqs = {a["arq"]: a for a in arquitecturas()}
        a = arqs[gan.arq]
        borde = any(w in (ANCHOS[0], ANCHOS[-1]) for w in a["anchos"]) and len(set(a["anchos"])) == 1
        d = dict(ganadora=gan.arq, anchos=a["anchos"], patron_dropout=a["patron_dropout"], MASE_medio_fase2=float(gan.MASE_medio),
                 mejor_media=r.arq.iloc[0], tamano_empate_tecnico=int(r.empate_tecnico.sum()), tamano_indistinguible_TOST=int(r.indistinguible_TOST.sum()),
                 correlacion_rangos_fase1_fase2_spearman=None if np.isnan(rho) else float(rho), en_el_borde_del_espacio=bool(borde),
                 regla=self.plan["seleccion"])
        guardar_json(self.raiz / "fase2" / "seleccion.json", d)
        return d

    def final(self, base: dict, sel: dict) -> dict:
        p = dict(base, id=3000, arq=sel["ganadora"], anchos=sel["anchos"], patron_dropout=sel["patron_dropout"])
        sub = PR.Contexto(self.ctx.raiz, self.raiz, dict(self.ctx.cfg, variantes=self.m["variantes"]), self.ctx.datos, self.ctx.dispositivo, self.ctx.log, self.rapido)
        self._heredar(sub)
        return finales_con_costes(sub, [p], {v: 3000 for v in self.m["variantes"]})

    def analisis(self, t1: pd.DataFrame, t2: pd.DataFrame) -> dict:
        """§14.6: superficie, efectos principales e interacciones, pruebas por factor, Pareto y sensibilidad."""
        import statsmodels.formula.api as smf
        from statsmodels.stats.anova import anova_lm
        from statsmodels.stats.multitest import multipletests
        out_dir = self.raiz / "analisis_espacio"
        out_dir.mkdir(parents=True, exist_ok=True)
        d = t1.copy()
        w = d.anchos.str.split("-", expand=True).astype(int)
        b = d.patron_dropout.str.split("-", expand=True).astype(int)
        for i in range(3):
            d[f"w{i + 1}"], d[f"b{i + 1}"] = w[i].astype(str), b[i]
        d["log2_ancho_total"] = np.log2(w.sum(axis=1))
        res = {}
        if len(d) >= 30:
            mod = smf.ols("MASE ~ C(w1) + C(w2) + C(w3) + b1 + b2 + b3 + b1:b2 + b1:b3 + b2:b3", data=d).fit()
            tab = anova_lm(mod, typ=2)
            tab["eta2_parcial"] = tab["sum_sq"] / (tab["sum_sq"] + tab.loc["Residual", "sum_sq"])
            pv = tab["PR(>F)"].drop("Residual")
            tab.loc[pv.index, "p_Holm"] = multipletests(pv.to_numpy(), method="holm")[1]
            tab.to_csv(out_dir / "anova_factores.csv")
            coef = pd.DataFrame(dict(coef=mod.params, ic95_inf=mod.conf_int()[0], ic95_sup=mod.conf_int()[1], p=mod.pvalues))
            coef.to_csv(out_dir / "efectos_principales_e_interacciones.csv")
            res["R2_modelo_factores"] = float(mod.rsquared)
            tend = smf.ols("MASE ~ log2_ancho_total", data=d).fit()
            res["tendencia_log2_ancho"] = dict(pendiente=float(tend.params["log2_ancho_total"]), p=float(tend.pvalues["log2_ancho_total"]))
            # sin dropout frente a cada patrón con dropout, pareado por la combinación de anchos (125 bloques)
            filas = []
            d["combo"] = d.anchos
            base = d[(d.b1 == 0) & (d.b2 == 0) & (d.b3 == 0)].set_index("combo").MASE
            for patron, g in d.groupby(d.patron_dropout):
                if patron == "0-0-0":
                    continue
                v = g.set_index("combo").MASE.reindex(base.index)
                dif = (v - base).dropna()
                if len(dif) > 5:
                    wt = stats.wilcoxon(dif)
                    delta = float(np.sign(dif.to_numpy()).mean())     # > 0: el dropout empeora más veces de las que mejora
                    filas.append(dict(patron=patron, n=len(dif), mediana_diferencia=float(dif.median()), p_wilcoxon=float(wt.pvalue), delta_signo=delta))
            if filas:
                tw = pd.DataFrame(filas)
                tw["p_Holm"] = multipletests(tw.p_wilcoxon, method="holm")[1]
                tw.to_csv(out_dir / "dropout_frente_a_sin_dropout.csv", index=False)
        # superficie: un mapa por patrón (w1 x (w2,w3)) como tabla
        d.pivot_table(index=["patron_dropout", "w1"], columns=["w2", "w3"], values="MASE").to_csv(out_dir / "superficie_fase1.csv")
        # Pareto (validación frente a parámetros y tiempo)
        def pareto(df, x):
            df = df.sort_values([x, "MASE"])
            return df[df.MASE <= df.MASE.cummin()].arq.tolist()
        res["pareto_parametros"] = pareto(d, "n_parametros")
        res["pareto_segundos"] = pareto(d, "segundos")
        if len(t2):
            res["sensibilidad"] = dict(desv_entre_arquitecturas_fase1=float(t1.MASE.std(ddof=1)),
                                       desv_entre_semillas_fase2=float(t2.groupby("arq").MASE.std(ddof=1).mean()))
            res["sensibilidad"]["conclusion_arquitectura_irrelevante"] = res["sensibilidad"]["desv_entre_arquitecturas_fase1"] <= res["sensibilidad"]["desv_entre_semillas_fase2"]
        guardar_json(out_dir / "resumen_analisis.json", res)
        self._figuras(d, out_dir)
        return res

    def _figuras(self, d: pd.DataFrame, out_dir: Path):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception:
            return
        pats = sorted(d.patron_dropout.unique())
        fig, axes = plt.subplots(2, 4, figsize=(18, 8), squeeze=False)
        for ax, pt in zip(axes.flat, pats):
            g = d[d.patron_dropout == pt].copy()
            g["col"] = g.w2 + "-" + g.w3
            m = g.pivot_table(index="w1", columns="col", values="MASE")
            m = m.reindex(index=[str(a) for a in ANCHOS], columns=[f"{a}-{c}" for a in ANCHOS for c in ANCHOS])
            im = ax.imshow(m.to_numpy(), aspect="auto", cmap="viridis_r")
            ax.set_title(f"dropout {pt}", fontsize=9)
            ax.set_yticks(range(5), [str(a) for a in ANCHOS], fontsize=7)
            ax.set_xticks(range(0, 25, 5), [f"w2={a}" for a in ANCHOS], fontsize=7)
            fig.colorbar(im, ax=ax, fraction=0.04)
        fig.suptitle("Fase 1: MASE de validación 2018 por anchos (filas w1; columnas w2-w3) y patrón de dropout")
        fig.tight_layout()
        fig.savefig(out_dir / "superficie_fase1.png", dpi=110)
        plt.close(fig)

    def ejecutar(self) -> dict:
        self.raiz.mkdir(parents=True, exist_ok=True)
        base = self.fase0()
        t1 = self.fase1(base)
        k = self.piloto(base, t1)["k"]
        t2 = self.fase2(base, t1, k)
        sel = self.seleccionar(t1, t2)
        self.analisis(t1, t2)
        fin = self.final(base, sel)
        guardar_json(self.raiz / "resumen_busqueda14.json", dict(hiperparametros_fijos=base, seleccion=sel, prueba=fin["variantes"]))
        return fin


# ------------------------------------------------------------------------------------ ablación y sanidad
def ablacion(ctx, plan: dict) -> pd.DataFrame:
    pos = _pos_final(ctx)
    semillas = ctx.rapido["semillas_finales"] if ctx.rapido else plan["semillas"]
    filas, perdidas = [], {}
    for j, ab in enumerate(plan["ablaciones"]):
        p = dict(plan["configuracion_base"], id=4000 + j, ablacion=ab)
        for s in semillas:
            c = ctx.salida / ab / f"semilla_{s}"
            r = corrida(ctx, "N", p, s, "final", pos, c, "prueba", guardar_modelo=True, medir=True)
            filas.append(_fila(r, dict(ablacion=ab)))
            perdidas.setdefault(ab, []).append(np.load(c / "pred_prueba.npz")["mase_por_origen"])
    t = pd.DataFrame(filas)
    t.to_csv(ctx.salida / "metricas_por_semilla.csv", index=False)
    inf = ctx.cfg["inferencia"]
    rem = ctx.rapido["remuestreos"] if ctx.rapido else inf["remuestreos"]
    media = {a: np.mean(v, axis=0) for a, v in perdidas.items()}
    contrastes = []
    for ab in plan["ablaciones"]:
        if ab == "completo":
            continue
        dif = media[ab] - media["completo"]
        bb = PR.bootstrap_bloques(dif, media["completo"], inf["bloque_dias"], rem, inf["semilla_bootstrap"])
        sa, sc = t.loc[t.ablacion == ab, "MASE"].to_numpy(), t.loc[t.ablacion == "completo", "MASE"].to_numpy()
        contrastes.append(dict(ablacion=ab, MASE_medio=float(sa.mean()), MASE_completo=float(sc.mean()), **{k: v for k, v in bb.items() if k != "remuestreos"},
                               t_pareado_semillas_p=float(stats.ttest_rel(sa, sc).pvalue) if len(sa) == len(sc) and len(sa) > 1 else None))
    tc = pd.DataFrame(contrastes)
    if len(tc):
        from statsmodels.stats.multitest import multipletests
        tc["p_bootstrap_Holm"] = multipletests(tc.p_bilateral_bootstrap, method="holm")[1]
    tc.to_csv(ctx.salida / "contrastes_ablacion.csv", index=False)
    pd.DataFrame(dict(origen=pd.DatetimeIndex(ctx.datos.indice[pos["ev"]]).strftime("%Y-%m-%d"), **{f"MASE_{a}": v for a, v in media.items()})).to_csv(
        ctx.salida / "perdida_por_origen.csv", index=False)
    return tc


def sanidad(ctx, plan: dict) -> pd.DataFrame:
    """Entrena con etiquetas permutadas y evalúa en la prueba; el error debe quedar lejos del modelo real."""
    pos = _pos_final(ctx)
    filas = []
    for caso in plan["casos"]:
        sub = PR.Contexto(ctx.raiz, ctx.salida, dict(ctx.cfg, paquete=dict(ctx.cfg["paquete"], red=caso["red"])), ctx.datos, ctx.dispositivo, ctx.log, ctx.rapido)
        sub.barra, sub.mlflow, sub.mlflow_raiz = ctx.barra, getattr(ctx, "mlflow", False), getattr(ctx, "mlflow_raiz", ctx.salida)
        r = corrida(sub, "N", dict(caso["configuracion"]), plan["semilla"], "final", pos, ctx.salida / caso["nombre"], "prueba", permutar=plan["semilla_permutacion"])
        filas.append(dict(caso=caso["nombre"], MASE_prueba_etiquetas_permutadas=r["metricas"]["MASE"], MASE_modelo_real_referencia=caso["MASE_real"],
                          razon=r["metricas"]["MASE"] / caso["MASE_real"], epocas=r["epocas"]))
    t = pd.DataFrame(filas)
    t["supera_criterio"] = t.razon >= plan["criterio_razon_minima"]
    t.to_csv(ctx.salida / "sanidad_etiquetas_permutadas.csv", index=False)
    return t


# ------------------------------------------------------------------------------------ robustez de los modelos nuevos
SEVERIDADES = {"ruido_demanda": [0.01, 0.03, 0.05], "atipicos": [1, 3, 6], "faltantes": [0.10, 0.30, 0.50]}
SEMILLA_ROBUSTEZ = 20261118


def _perturbar(X: np.ndarray, M: np.ndarray, tipo: str, s: float, nivel: int):
    """Mismas perturbaciones y semillas que scripts/e2c_robustez.py del repositorio (misma longitud del panel)."""
    g = np.random.default_rng(SEMILLA_ROBUSTEZ + nivel + 10 * len(tipo))
    X, M = X.copy(), M.copy()
    if tipo == "ruido_demanda":
        X[:, 0] = X[:, 0] * (1 + s * g.standard_normal(len(X)))
    elif tipo == "atipicos":
        m = g.random(len(X)) < s / 168
        X[m, 0] = X[m, 0] * 1.2
    elif tipo == "faltantes":
        f = g.random(len(X)) < s
        X[np.ix_(f, np.arange(1, X.shape[1]))] = np.nan
        M[f] = 1.0
    return X, M


def _escalas(js: dict, datos) -> dict:
    h, c = js["columnas_historicas"], js["columnas_calendario"]
    assert h == datos.hist and c == datos.cal
    return dict(mediana=np.array([js["mediana"][k] for k in h]), media=np.array([js["media"][k] for k in h]), std=np.array([js["std"][k] for k in h]),
                media_cal=np.array([js["media_calendario"][k] for k in c]), std_cal=np.array([js["std_calendario"][k] for k in c]), sigma_demanda=js["sigma_demanda"])


def robustez(raiz_res: Path, datos, red_de: dict, texto, descripciones, dispositivo: str, log, rapido: dict | None = None) -> pd.DataFrame:
    """Inferencia con los pesos finales guardados bajo tres tipos de perturbación y tres severidades."""
    import torch
    from . import motor, redes_e2
    pos = datos.prueba()
    if rapido:
        pos = pos[:rapido["n_ev"]]
    filas = []
    corridas = [p for p in sorted(Path(raiz_res).glob("*/final/*/semilla_*/modelo.pt")) if p.parts[-5] in red_de]
    escenarios = [("original", 0, 0.0)] + [(t, i + 1, s) for t in SEVERIDADES for i, s in enumerate(SEVERIDADES[t])]
    X0, M0 = datos.X, datos.M
    try:
        for tipo, nivel, s in escenarios:
            datos.X, datos.M = (X0, M0) if tipo == "original" else _perturbar(X0, M0, tipo, s, nivel)
            for ck in corridas:
                exp, var = ck.parts[-5], ck.parts[-3].replace("_", "+")
                r = PR.leer_json(ck.parent / "resultado.json")
                esc = _escalas(PR.leer_json(ck.parent / "escaladores.json"), datos)
                conj = datos.construir(pos, esc)
                tx = dict(vectores=torch.from_numpy(texto["vectores"]), lista=torch.from_numpy(texto["lista"])) if var == "N+O" else None
                desc = torch.from_numpy(descripciones) if descripciones is not None else None
                red = redes_e2.construir(red_de[exp], r["configuracion"], conj.x.shape[2], conj.cal_p.shape[2], datos.L, datos.H, tx, desc)
                red.load_state_dict(torch.load(ck, map_location="cpu", weights_only=True), strict=True)
                red = red.to(dispositivo)
                yhat = motor.predecir(red, motor.a_tensores(conj, dispositivo)).astype("float64") * conj.nivel[:, None]
                filas.append(dict(clave=f"{exp}__{var}", semilla=r["semilla"], tipo=tipo, nivel=nivel, severidad=s,
                                  MASE=float(np.abs(yhat - conj.y).mean() / datos.q168)))
            log(f"Robustez {tipo} nivel {nivel}: {len(corridas)} modelos evaluados.")
    finally:
        datos.X, datos.M = X0, M0
    t = pd.DataFrame(filas)
    t.to_csv(Path(raiz_res) / "robustez_nuevos_por_semilla.csv", index=False)
    return t
```

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
from . import redes_e2 as redes
from . import costes


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


def _permutar_etiquetas(conj, semilla: int):
    """Copia del conjunto con las etiquetas (demanda / nivel) permutadas entre orígenes: rompe la relación x -> y."""
    from dataclasses import replace
    idx = np.random.default_rng(semilla).permutation(len(conj.y))
    return replace(conj, y=conj.y_norm[idx].astype("float64") * conj.nivel[:, None])


def corrida(ctx: Contexto, variante: str, p: dict, semilla: int, clave_datos: str, pos: dict, carpeta: Path, nombre_ev: str,
            guardar_modelo: bool = False, opciones: dict | None = None, permutar: int | None = None, medir: bool = False) -> dict:
    """Entrena una configuración con una semilla y evalúa en ``pos['ev']``. Reanudable.

    ``opciones`` reemplaza el presupuesto (fase 1 de §14); ``permutar`` permuta las etiquetas de ajuste y parada
    (prueba de sanidad de §10); ``medir`` registra FLOPs, memoria pico, energía y latencia (corridas finales)."""
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
    if permutar is not None:
        conj = dict(conj, tr=_permutar_etiquetas(conj["tr"], permutar), es=_permutar_etiquetas(conj["es"], permutar + 1))
    opc = dict(ctx.opciones, **(opciones or {}))
    texto = None
    if variante == "N+O":
        texto = dict(vectores=torch.from_numpy(ctx.texto["vectores"]), lista=torch.from_numpy(ctx.texto["lista"]))
    desc = torch.from_numpy(ctx.descripciones) if ctx.descripciones is not None else None
    dispositivos = [ctx.dispositivo] + (["cpu"] if ctx.dispositivo != "cpu" else [])
    for intento, dev in enumerate(dispositivos):
        try:
            motor.fijar_semilla(semilla)
            red = redes.construir(ctx.cfg["paquete"]["red"], p, conj["tr"].x.shape[2], conj["tr"].cal_p.shape[2], datos.L, datos.H, texto, desc).to(dev)
            medidor = costes.Medidor(dev) if medir else None
            tens = {k: motor.a_tensores(v, dev) for k, v in conj.items()}

            def observador(h, _e=etiqueta):
                if ctx.barra is not None:
                    ctx.barra.set_postfix_str(f"{_e} | época {h['epoca']} | parada {h['perdida_parada']:.5f}", refresh=True)

            t0 = time.time()
            if medidor:
                medidor.iniciar()
            info = motor.entrenar(red, tens["tr"], tens["es"], p, semilla, opc, observador)
            if medidor:
                medidor.detener()
            le = int(opc.get("lote_evaluacion", 256))
            t_inf = time.time()
            yhat, met = _evaluar(datos, red, conj["ev"], tens["ev"], le)
            t_inf = time.time() - t_inf
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
               mase_por_horizonte=[float(v) for v in MET.mase_por_horizonte(conj["ev"].y, yhat, datos.q168)], diagnostico=red.diagnostico(),
               presupuesto=dict(max_epocas=int(opc["max_epocas"]), min_epocas=int(opc["min_epocas"]), paciencia=int(opc["paciencia"])),
               etiquetas_permutadas=permutar is not None)
    if medidor:
        res["costes"] = dict(medidor.resumen(), **costes.flops_por_muestra(red, tens["ev"]), latencia_ms_por_origen_lote=1000 * t_inf / len(conj["ev"]))
    pd.DataFrame(info["historia"]).to_csv(carpeta / "historia.csv", index=False)
    np.savez_compressed(carpeta / f"pred_{nombre_ev}.npz", origen=np.asarray(conj["ev"].origen.strftime("%Y-%m-%d"), dtype="U10"), yhat_kwh=yhat.astype("float32"),
                        y_kwh=conj["ev"].y.astype("float32"), mase_por_origen=datos.mase_por_origen(conj["ev"].y, yhat))
    if guardar_modelo:
        torch.save({k: v.cpu() for k, v in red.state_dict().items()}, carpeta / "modelo.pt")
        guardar_json(carpeta / "escaladores.json", escalas_a_json(esc, datos))
    guardar_json(rr, res)
    costes.registrar_mlflow(ctx, carpeta, res)
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

### Optimización y restauración del checkpoint

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

### Prueba por origen con pesos congelados

El siguiente código restaura la configuración y los escaladores del checkpoint y obtiene las métricas sin modificar los pesos. TimesNet se evalúa con lote uno.

```python
"""Inferencia causal de checkpoints: un origen por llamada."""
from pathlib import Path
import json,numpy as np,torch
from entrenamiento.datos import DatosNumericos
from entrenamiento.protocolo_e2 import _escalas
from entrenamiento.redes_e2 import construir

def predecir_checkpoint(raiz_datos,carpeta_checkpoint,familia,origenes,dispositivo="cpu"):
    carpeta=Path(carpeta_checkpoint)
    meta=json.loads((carpeta/"resultado.json").read_text(encoding="utf8"))
    escalas=json.loads((carpeta/"escaladores.json").read_text(encoding="utf8"))
    datos=DatosNumericos(Path(raiz_datos))
    posiciones=datos.indice.get_indexer(origenes)
    assert (posiciones>=168).all()
    entradas=datos.construir(posiciones,_escalas(escalas,datos))
    red=construir(familia,meta["configuracion"],33,23,168,24,None,None).to(dispositivo)
    red.load_state_dict(torch.load(carpeta/"modelo.pt",map_location=dispositivo,weights_only=True),strict=True)
    red.eval();red.requires_grad_(False)
    resultados=[]
    with torch.inference_mode():
        for i in range(len(origenes)):
            args=[torch.from_numpy(a[i:i+1]).to(dispositivo) for a in
                  [entradas.x,entradas.cal_p,entradas.cal_f,entradas.dia]]
            pred=red(*args).cpu().numpy().astype(float)[0]*entradas.nivel[i]
            resultados.append(pred)
    pred=np.stack(resultados)
    error=pred-entradas.y
    metricas={"MASE":float(np.abs(error).mean()/escalas["q168_kwh"]),
              "MAE":float(np.abs(error).mean()),"RMSE":float(np.sqrt((error**2).mean()))}
    return pred,metricas
```

Los resultados, curvas de cinco semillas, métricas por horizonte y comparaciones están en la evaluación integrada.

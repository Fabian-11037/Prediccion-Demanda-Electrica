"""Los tres conjuntos de datos adicionales del Entregable 3 (§8.3), con la misma interfaz que ``DatosNumericos``.

Misma tarea que el problema principal: pronóstico directo de 24 h con origen diario a las 00:00 e historia de 168 h.
Cada conjunto se describe en ``config/plan_e3.json`` (``adicionales``): archivo, SHA-256, objetivo, covariables,
desplazamiento del objetivo, país de los festivos y fechas de la partición. Reglas comunes:

* el objetivo de la hora t es la columna 0 de la historia; todas las covariables se usan solo en el pasado
  (filas anteriores al origen); no hay pronósticos exógenos del futuro, solo calendario determinista;
* se agrega a frecuencia horaria con la media de la hora (si la fuente es más fina) y la serie se reindexa a
  todas las horas; las horas faltantes quedan como NaN;
* un origen es elegible si el objetivo está completo en las 168 h de la ventana y en las 24 h de la etiqueta
  (la misma regla que el problema principal); las covariables faltantes se imputan con la mediana de ajuste y
  llevan su indicador de ausencia;
* la partición es cronológica: referencia, cuatro pliegues de validación consecutivos (forward chaining, con
  purga de 24 h y el último 10 % como parada interna) y prueba; las fechas se fijan en el plan antes de ver resultados;
* el desplazamiento del objetivo (ETTh1: +10 °C) solo evita niveles de ventana cercanos a cero en la normalización
  por nivel; el MASE no cambia porque los errores absolutos y la escala q168 son invariantes a una constante.
"""
from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .datos import DatosNumericos, sha256

FRACCION_PARADA = 0.10


def _calendario(indice: pd.DatetimeIndex, pais: str) -> pd.DataFrame:
    import holidays
    años = sorted(set(indice.year))
    fest = holidays.country_holidays(pais, years=años)
    dias = indice.normalize()
    hora, dow, mes, doy = indice.hour.to_numpy(), indice.dayofweek.to_numpy(), indice.month.to_numpy(), indice.dayofyear.to_numpy()
    k = {"hora_sen": np.sin(2 * np.pi * hora / 24), "hora_cos": np.cos(2 * np.pi * hora / 24)}
    for d in range(7):
        k[f"dia_semana_{d}"] = (dow == d).astype(float)
    k.update(mes_sen=np.sin(2 * np.pi * (mes - 1) / 12), mes_cos=np.cos(2 * np.pi * (mes - 1) / 12),
             dia_anio_sen=np.sin(2 * np.pi * (doy - 1) / 365.25), dia_anio_cos=np.cos(2 * np.pi * (doy - 1) / 365.25),
             festivo=np.array([d.date() in fest for d in dias], dtype=float))
    return pd.DataFrame(k, index=indice)


def leer_fuente(raiz: Path, d: dict) -> pd.DataFrame:
    """Serie horaria (índice sin zona) con el objetivo en la primera columna y las covariables después."""
    ruta = Path(raiz) / "data" / "adicionales" / d["archivo"]
    if d["formato"] == "ett":
        t = pd.read_csv(ruta, parse_dates=["date"]).set_index("date")
    elif d["formato"] == "prsa":
        t = pd.read_csv(ruta)
        t.index = pd.to_datetime(t[["year", "month", "day", "hour"]])
        t = pd.concat([t, pd.get_dummies(t["cbwd"], prefix="viento", dtype=float)], axis=1)
    elif d["formato"] == "hogar":
        num = ["Global_active_power", "Global_reactive_power", "Voltage", "Global_intensity", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]
        with zipfile.ZipFile(ruta) as z:
            t = pd.read_csv(z.open(d["miembro"]), sep=";", na_values="?", dtype={c: "float32" for c in num} | {"Date": str, "Time": str})
        idx = pd.to_datetime(t.pop("Date") + " " + t.pop("Time"), format="%d/%m/%Y %H:%M:%S")
        t.index = idx
        t = t.astype("float64").resample("h").mean()
    else:
        raise ValueError(f"Formato desconocido: {d['formato']}")
    t = t[[d["objetivo"]] + list(d["covariables"])].astype(float)
    t = t[~t.index.duplicated(keep="first")].sort_index()
    t = t.reindex(pd.date_range(t.index.min().normalize(), t.index.max(), freq="h"))
    t[d["objetivo"]] = t[d["objetivo"]] + float(d.get("desplazamiento_objetivo", 0.0))
    return t


class DatosAdicionales(DatosNumericos):
    """Ventanas, escalas, pliegues y referencias de un conjunto adicional con la interfaz de ``DatosNumericos``."""

    def __init__(self, raiz: Path, nombre: str, d: dict, verificar_hash: bool = True, cache: Path | None = None):
        self.raiz, self.nombre, self.definicion = Path(raiz), nombre, d
        ruta = self.raiz / "data" / "adicionales" / d["archivo"]
        self.hash_panel = sha256(ruta)
        if verificar_hash and self.hash_panel != d["sha256"]:
            raise RuntimeError(f"{d['archivo']} no coincide con su SHA-256 preregistrado")
        rc = Path(cache) / f"{nombre}_{self.hash_panel[:12]}.parquet" if cache else None
        if rc is not None and rc.exists():
            t = pd.read_parquet(rc)
        else:
            t = leer_fuente(self.raiz, d)
            if rc is not None:
                rc.parent.mkdir(parents=True, exist_ok=True)
                t.to_parquet(rc)
        ini, fin = pd.Timestamp(d["particion"]["inicio"]), pd.Timestamp(d["particion"]["fin"]) + pd.Timedelta(hours=23)
        t = t.loc[(t.index >= ini - pd.Timedelta(hours=168)) & (t.index <= fin)]
        cal = _calendario(t.index, d["pais_festivos"])
        self.contrato = dict(texto=None, ventana_h=168, horizonte_h=24)
        self.hist = ["x_" + d["objetivo"]] + ["x_" + c for c in d["covariables"]]
        self.cal = list(cal.columns)
        self.indice = t.index
        self.X = t.to_numpy(dtype="float64")
        self.K = cal.to_numpy(dtype="float64")
        self.Y = t[d["objetivo"]].to_numpy(dtype="float64")
        self.L, self.H = 168, 24
        self.rezago_demanda = 0
        nan = np.isnan(self.X[:, 1:])
        self.nombres_mascara, cols = [], []
        if nan.any():
            grupos = {}
            for j in range(nan.shape[1]):
                if nan[:, j].any():
                    grupos.setdefault(nan[:, j].tobytes(), []).append(j)
            for k, js in enumerate(grupos.values()):
                cols.append(nan[:, js[0]]); self.nombres_mascara.append("ausente_" + ("_".join(d["covariables"][j] for j in js) if len(js) < 3 else f"bloque_{k}"))
        self.M = (np.stack(cols, axis=1) if cols else np.zeros((len(self.X), 0))).astype("float32")
        self.dia_base = self.indice[0].normalize()
        self._off_l, self._off_h = np.arange(-self.L, 0), np.arange(0, self.H)
        p = d["particion"]
        self._limites = dict(referencia=(p["inicio"], p["fin_referencia"]), prueba=(p["inicio_prueba"], p["fin"]),
                             pliegues=[tuple(x) for x in p["pliegues_validacion"]])
        ref = self._en(self._candidatos(), *self._limites["referencia"])
        y = self.Y
        self.q168 = float(np.nanmean(np.abs(y[168:] - y[:-168])[(self.indice[168:] >= pd.Timestamp(p["inicio"])) &
                                                             (self.indice[168:] <= pd.Timestamp(p["fin_referencia"]) + pd.Timedelta(hours=23))]))
        if not np.isfinite(self.q168) or self.q168 <= 0 or len(ref) < 50:
            raise RuntimeError(f"{nombre}: escala q168 o referencia inválidas")

    # ------------------------------------------------------------------ orígenes
    def _candidatos(self) -> np.ndarray:
        pos = np.arange(len(self.indice))
        ok = (pos >= self.L) & (pos + self.H <= len(self.indice)) & (self.indice.hour == 0)
        return pos[ok]

    def _en(self, cand: np.ndarray, ini: str, fin: str) -> np.ndarray:
        t = self.indice[cand]
        a, b = pd.Timestamp(ini), pd.Timestamp(fin) + pd.Timedelta(hours=23)
        return cand[(t >= a) & (t + pd.Timedelta(hours=self.H - 1) <= b)]

    def _antes(self, cand: np.ndarray, ini: str) -> np.ndarray:
        return cand[self.indice[cand] + pd.Timedelta(hours=self.H - 1) < pd.Timestamp(ini)]

    def _dividir(self, pre: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n_es = max(1, int(round(FRACCION_PARADA * len(pre))))
        es = pre[-n_es:]
        tr = pre[self.indice[pre] + pd.Timedelta(hours=self.H - 1) < self.indice[es[0]]]
        return tr, es

    def elegibles(self, pos: np.ndarray) -> np.ndarray:
        pos = np.asarray(pos)
        ok = ~np.isnan(self.Y[pos[:, None] + self._off_l]).any(axis=1) & ~np.isnan(self.Y[pos[:, None] + self._off_h]).any(axis=1)
        return pos[ok]

    def pliegues(self) -> list[dict]:
        cand, out = self._candidatos(), []
        for k, (ini, fin) in enumerate(self._limites["pliegues"], start=1):
            val = self.elegibles(self._en(cand, ini, fin))
            pre = self.elegibles(self._antes(cand[self.indice[cand] >= pd.Timestamp(self._limites["referencia"][0])], ini))
            tr, es = self._dividir(pre)
            out.append(dict(anio=k, tr=tr, es=es, val=val))
        return out

    def ajuste_final(self) -> tuple[np.ndarray, np.ndarray]:
        cand = self._candidatos()
        pre = self.elegibles(self._antes(cand[self.indice[cand] >= pd.Timestamp(self._limites["referencia"][0])], self._limites["prueba"][0]))
        return self._dividir(pre)

    def prueba(self) -> np.ndarray:
        return self.elegibles(self._en(self._candidatos(), *self._limites["prueba"]))

    # ---------------------------------------------------------------- referencias
    def ingenuos(self, pos: np.ndarray) -> dict[str, np.ndarray]:
        """Ingenuos diario y semanal: aquí el objetivo de la hora previa ya está disponible en el origen."""
        pos = np.asarray(pos)
        return {f"ingenuo_{b}h": np.stack([self.Y[pos + h - b] for h in range(self.H)], axis=1) for b in (24, 168)}

    def resumen(self) -> dict:
        f = self.pliegues()
        tr, es = self.ajuste_final()
        return dict(conjunto=self.nombre, sha256=self.hash_panel, horas=len(self.indice), q168=self.q168, covariables=self.hist[1:],
                    mascaras=self.nombres_mascara, calendario=self.cal, origenes_ajuste_final=len(tr), origenes_parada_final=len(es),
                    origenes_prueba=len(self.prueba()), pliegues=[dict(pliegue=x["anio"], tr=len(x["tr"]), es=len(x["es"]), val=len(x["val"])) for x in f])

"""Construye el panel horario «as-of» a partir de los datos crudos.

Convenciones del panel (índice = inicio del intervalo, hora de Colombia):

* ``y_demanda_real_kwh``  objetivo de la fila (intervalo [t, t+1h)). Solo se lee
  para las etiquetas del horizonte, nunca como entrada.
* ``x_*``  entradas *as-of*: la fila ``t`` contiene únicamente información
  disponible en el instante ``t+1h`` (fin del intervalo). Los rezagos de
  publicación ya están aplicados, de modo que cualquier ventana que termine en
  una fila ``<= t0-1h`` es usable en el origen ``t0``.
* ``cal_*``  variables de calendario (``known_future``): describen el propio
  intervalo y pueden leerse también para el horizonte.

No se incluyen: ``demanda_comercial_kwh`` (derivada de la demanda real más
pérdidas y liquidada después) ni el clima de la misma hora que el objetivo
(solo existe como reanálisis; clase ``oracle_only``).
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from .calendario import calendario_horario


def leer_horario() -> pd.DataFrame:
    fr = [pd.read_csv(p) for p in sorted(glob.glob(str(C.RAW / "numeric" / "hourly" / "*.csv.gz")))]
    h = pd.concat(fr, ignore_index=True)
    ts = pd.to_datetime(h["timestamp_start_colombia"], utc=True).dt.tz_convert(C.TZ)
    h.index = pd.DatetimeIndex(ts, name="inicio")
    if not h.index.is_monotonic_increasing or not h.index.is_unique:
        raise ValueError("Índice horario no monótono o con duplicados")
    paso = h.index.to_series().diff().dropna().unique()
    if len(paso) != 1 or paso[0] != pd.Timedelta(hours=1):
        raise ValueError(f"Hay huecos o pasos irregulares en la serie horaria: {paso}")
    return h


def leer_hidrologia() -> pd.DataFrame:
    fr = [pd.read_csv(p) for p in sorted(glob.glob(str(C.RAW / "numeric" / "hydrology" / "*.csv.gz")))]
    y = pd.concat(fr, ignore_index=True)
    y["date"] = pd.to_datetime(y["date"])
    y = y.sort_values("date").reset_index(drop=True)
    if not y.date.is_unique:
        raise ValueError("Fechas duplicadas en hidrología")
    return y


def _as_of_diario(indice_horario: pd.DatetimeIndex, diario: pd.DataFrame, rezago_h: int, prefijo: str) -> pd.DataFrame:
    """Asigna a cada fila horaria el último valor diario ya disponible.

    El valor del día ``d`` termina en ``d+1 00:00`` y está disponible en
    ``d+1 00:00 + rezago``. La fila ``t`` (información hasta ``t+1h``) recibe el
    valor más reciente cuya disponibilidad es ``<= t+1h``.
    """
    d = diario.copy()
    d["disp"] = (d["date"] + pd.Timedelta(days=1) + pd.Timedelta(hours=rezago_h)).dt.tz_localize(C.TZ)
    d = d.sort_values("disp")
    q = pd.DataFrame({"info_hasta": indice_horario + pd.Timedelta(hours=1)}, index=indice_horario)
    m = pd.merge_asof(q.reset_index(), d, left_on="info_hasta", right_on="disp", direction="backward")
    m = m.set_index("inicio")
    cols = [c for c in diario.columns if c != "date"]
    out = m[cols].add_prefix(prefijo)
    out[prefijo + "fecha_dato"] = m["date"]
    return out


def construir_panel() -> tuple[pd.DataFrame, dict]:
    h = leer_horario()
    idx = h.index
    panel = pd.DataFrame(index=idx)
    panel["y_" + C.OBJETIVO] = h[C.OBJETIVO].astype("float64")

    # --- entradas past_only con rezago de publicación -----------------------
    panel["x_" + C.OBJETIVO] = h[C.OBJETIVO].shift(C.REZAGO_H["demanda_real"])
    panel["x_precio_bolsa_cop_kwh"] = h["precio_bolsa_cop_kwh"].shift(C.REZAGO_H["precio_bolsa"])
    for v in C.VARIABLES_CLIMA:
        for c in C.CIUDADES:
            panel[f"x_obs_{v}_{c}"] = h[f"{v}_{c}"].astype("float32").shift(C.REZAGO_H["clima_obs"])

    # clima nacional ponderado por habitantes (pesos estáticos; ver config.pesos_ciudades)
    w = C.pesos_ciudades()
    for v in C.VARIABLES_CLIMA:
        panel[f"x_obs_{v}_ponderada"] = sum(w[c] * panel[f"x_obs_{v}_{c}"].astype("float64") for c in C.CIUDADES).astype("float32")

    hid = leer_hidrologia()
    hh = _as_of_diario(idx, hid, C.REZAGO_H["hidrologia"], "x_hid_")
    panel = panel.join(hh.drop(columns=["x_hid_fecha_dato"]))
    panel["aux_hid_fecha_dato"] = hh["x_hid_fecha_dato"]  # auditoría: día al que corresponde el dato
    # --- calendario (known_future) -------------------------------------------
    cal = calendario_horario(idx).add_prefix("cal_")
    panel = panel.join(cal)
    panel = panel.copy()

    registro = construir_registro(panel)
    return panel, registro


def construir_registro(panel: pd.DataFrame) -> dict:
    """Registro de variables: rol, clase de disponibilidad, rezago, fuente."""
    reg = {}
    for c in panel.columns:
        if c == "y_" + C.OBJETIVO:
            reg[c] = dict(clase="target", rol="objetivo", fuente="XM", rezago_h=None)
        elif c.startswith("cal_"):
            reg[c] = dict(clase="known_future", rol="entrada_y_futuro", fuente="calendario legal (Ley 51/1983 y posteriores)", rezago_h=0)
        elif c.startswith("aux_"):
            reg[c] = dict(clase="auditoria", rol="no_usar_como_entrada", fuente="XM", rezago_h=None)
        elif c == "x_" + C.OBJETIVO:
            reg[c] = dict(clase="past_only", rol="entrada_pasada", fuente="XM", rezago_h=C.REZAGO_H["demanda_real"], supuesto=True)
        elif c.startswith("x_precio"):
            reg[c] = dict(clase="past_only", rol="entrada_pasada", fuente="XM", rezago_h=C.REZAGO_H["precio_bolsa"], supuesto=True)
        elif c.startswith("x_obs_"):
            reg[c] = dict(clase="past_only", rol="entrada_pasada", fuente="ERA5 vía Open-Meteo (reanálisis)", rezago_h=C.REZAGO_H["clima_obs"], supuesto=True)
            if c.endswith("_ponderada"):
                reg[c]["nota"] = "promedio de las 8 ciudades ponderado por habitantes (censo 2018 ajustado)"
        elif c.startswith("x_hid_"):
            reg[c] = dict(clase="past_only", rol="entrada_pasada", fuente="XM (diario)", rezago_h=C.REZAGO_H["hidrologia"], supuesto=True)
        else:
            raise ValueError(f"Columna sin clasificar: {c}")
    reg["_excluidas"] = {
        "demanda_comercial_kwh": "excluded_input: derivada de la demanda real más pérdidas, liquidada con posterioridad; redundante y con riesgo de fuga",
        "clima_misma_hora_objetivo": "oracle_only: ERA5 es reanálisis observado; en producción solo existiría un pronóstico (ver pronosticos.py)",
        "precio_bolsa_hora_objetivo": "no disponible en el origen (solo rezagado)",
    }
    reg["_supuestos"] = {
        "origen": "t0 en hora entera; horizonte = intervalos que empiezan en t0..t0+(H-1)h",
        "as_of": "la fila t de x_* contiene información disponible en t+1h; rezagos ya aplicados",
        "rezagos_h": C.REZAGO_H,
        "ise_rezago_dias": C.ISE_REZAGO_DIAS,
        "zona_horaria": C.TZ,
        "pesos_ciudades": C.pesos_ciudades(),
    }
    return reg


def informe_calidad(panel: pd.DataFrame) -> dict:
    """Diagnóstico descriptivo (solo auditoría; no genera variables de entrada).

    Marca las horas de demanda inusualmente bajas respecto a la mediana de la
    misma hora de la semana en ±4 semanas. Usa información futura, por eso su
    uso queda restringido al informe y nunca como característica.
    """
    y = panel["y_" + C.OBJETIVO]
    ref = pd.concat([y.shift(168 * k) for k in (-4, -3, -2, -1, 1, 2, 3, 4)], axis=1).median(axis=1)
    ratio = y / ref
    bajos = ratio[ratio < 0.5]
    return {
        "filas": int(len(panel)),
        "inicio": str(panel.index.min()),
        "fin": str(panel.index.max()),
        "nan_por_columna_y": int(y.isna().sum()),
        "horas_demanda_menor_50pct_de_su_referencia_semanal": {str(k): round(float(v), 3) for k, v in bajos.items()},
        "nota": "Valores no corregidos ni imputados; decidir tratamiento en el preregistro.",
    }


def guardar(panel: pd.DataFrame, registro: dict, destino: Path | None = None) -> None:
    destino = destino or C.PROCESADO
    destino.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(destino / "panel_horario.parquet", compression="zstd")
    (destino / "feature_registry.json").write_text(json.dumps(registro, indent=2, ensure_ascii=False), encoding="utf-8")
    (destino / "calidad_datos.json").write_text(json.dumps(informe_calidad(panel), indent=2, ensure_ascii=False), encoding="utf-8")

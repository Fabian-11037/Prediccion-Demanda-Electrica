"""Constantes del problema y reglas de disponibilidad (a priori).

Los valores marcados como ``supuesto`` no están certificados por la fuente: son
decisiones conservadoras que deben declararse en el preregistro. Cámbialos aquí
(y solo aquí) para que panel, ventanas y pruebas permanezcan coherentes.
"""
from __future__ import annotations

from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
DATOS = RAIZ / "data"
RAW = DATOS / "raw"
EXTERNO = DATOS / "external"
PROCESADO = DATOS / "processed"

TZ = "America/Bogota"  # Colombia no usa horario de verano

# ---- Problema ---------------------------------------------------------------
OBJETIVO = "demanda_real_kwh"  # demanda real horaria del SIN
FRECUENCIA = "h"
HORIZONTE = 24  # intervalos horarios por pronóstico (propuesta; ver CONTEXTO_PROYECTO.md)
VENTANA = 168  # horas de historia en la entrada (una semana; propuesta)

# El origen t0 es un instante en hora entera. El pronóstico cubre los intervalos
# que empiezan en t0, t0+1h, ..., t0+(H-1)h. La entrada solo puede contener
# información cuya disponibilidad sea <= t0.

# ---- Partición cronológica (PROPUESTA, no preregistrada) --------------------
# Cada intervalo-objetivo debe quedar completo dentro de una sola partición
# (purga = HORIZONTE). Las fechas son inclusivas, hora de Colombia.
PARTICION = {
    "train": ("2000-01-01", "2017-12-31"),
    "val": ("2018-01-01", "2021-12-31"),
    "test": ("2022-01-01", "2025-12-31"),
}
PRUEBA_CONGELADA = True  # `test` solo con allow_test=True (una sola corrida final)

# ---- Rezagos de disponibilidad ---------------------------------------------
# Horas entre el fin del intervalo observado y el momento en que el valor puede
# usarse. Todos son supuestos conservadores (la fuente no certifica horas de
# publicación históricas).
REZAGO_H = {
    "demanda_real": 1,  # supuesto: el dato horario puede usarse 1 h después de cerrar la hora
    "precio_bolsa": 24,  # supuesto: el precio ex post se publica al día siguiente
    "clima_obs": 1,  # supuesto: observación disponible 1 h después (ERA5 es reanálisis)
    "hidrologia": 24,  # supuesto: dato diario disponible 24 h después de terminar el día
}
# ISE (DANE): se publica ~45-49 días después de terminar el mes de referencia;
# se usa 60 días como cota conservadora. Las cifras se revisan: ver docs.
ISE_REZAGO_DIAS = 60
# Pronósticos meteorológicos: emisión aproximada + latencia de disponibilidad.
PRONOSTICO_LATENCIA_H = 6
# Margen por la semántica de ``temperature_2m_previous_dayN`` (la corrida usada puede
# no ser exactamente la de ``valido - N*24h``): se suma a la emisión aproximada.
PRONOSTICO_MARGEN_H = 6

CIUDADES = ["bogota", "medellin", "cali", "barranquilla", "cartagena", "bucaramanga", "pereira", "cucuta"]
VARIABLES_CLIMA = ["temperature_2m_c", "relative_humidity_2m_pct", "precipitation_preceding_hour_mm"]

# Habitantes por ciudad (censo 2018 del DANE ajustado por omisión censal, cifras citadas por Portafolio,
# 17-oct-2019; fuente secundaria: sustituir por el archivo oficial del DANE cuando se descargue).
# Son pesos estáticos: el censo se publicó después de 2018, pero no depende de la demanda; declararlo.
RUTA_CIUDADES = RAW / "numeric" / "ciudades.csv"
COLUMNA_HABITANTES = "habitantes_censo_2018_ajustado"


def pesos_ciudades() -> "dict[str, float]":
    """Pesos normalizados (suman 1) proporcionales a los habitantes de cada ciudad de ``CIUDADES``."""
    import pandas as pd

    t = pd.read_csv(RUTA_CIUDADES).set_index("ciudad")[COLUMNA_HABITANTES]
    t = t.loc[CIUDADES].astype("float64")
    return (t / t.sum()).to_dict()

"""Descarga pronósticos de temperatura con emisión previa (Open-Meteo, Previous Runs API).

SIN PROBAR desde el entorno de desarrollo (la red del contenedor no alcanza Open-Meteo).
Ejecutar en un equipo con internet y revisar la salida con ``--dry-run`` primero.

Qué descarga: ``temperature_2m_previous_dayN`` (N = 1..LEADS) para las ocho ciudades.
El pronóstico ``N`` días antes de ``valido`` se emitió aproximadamente en
``valido - N*24h``; ``src/demanda/pronosticos.py`` aplica la regla de disponibilidad
(emisión + margen + latencia <= origen).

Cobertura (a verificar en la documentación vigente de Open-Meteo): GFS desde 2021-03;
la mayoría de modelos desde 2024-01. No son pronósticos «reales» de 24 h exactos
antes de 2021; no extrapolar la serie hacia atrás con ERA5.

Uso:
    python scripts/adquisicion/descargar_pronosticos_temperatura.py --inicio 2021-03-01 --fin 2025-12-31 --modelo gfs_seamless
    python scripts/adquisicion/descargar_pronosticos_temperatura.py --dry-run --inicio 2024-01-01 --fin 2024-01-31
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from demanda import config as C  # noqa: E402

URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
DESTINO = C.EXTERNO / "pronosticos_temperatura"
PATRON = re.compile(r"^temperature_2m_previous_day(\d+)(?:_(.+))?$")


def parsear_respuesta(js: dict, ciudad: str) -> pd.DataFrame:
    """Convierte la respuesta JSON en filas (valido, ciudad, lead_dias, temperature_2m_c, modelo)."""
    h = js["hourly"]
    t = pd.to_datetime(h["time"]).tz_localize(C.TZ)
    filas = []
    for k, vals in h.items():
        m = PATRON.match(k)
        if not m:
            continue
        lead = int(m.group(1))
        modelo = m.group(2) or "default"
        df = pd.DataFrame({"valido": t, "ciudad": ciudad, "lead_dias": lead, "temperature_2m_c": vals, "modelo": modelo})
        filas.append(df.dropna(subset=["temperature_2m_c"]))
    if not filas:
        raise ValueError(f"La respuesta no contiene temperature_2m_previous_dayN: claves {list(h)}")
    return pd.concat(filas, ignore_index=True)


def pedir(ciudad: str, lat: float, lon: float, ini: str, fin: str, modelo: str, leads: int, reintentos: int = 5) -> dict:
    hourly = ",".join(f"temperature_2m_previous_day{n}" for n in range(1, leads + 1))
    params = dict(latitude=lat, longitude=lon, hourly=hourly, models=modelo, start_date=ini, end_date=fin, timezone=C.TZ)
    for i in range(reintentos):
        r = requests.get(URL, params=params, timeout=120)
        if r.status_code == 200:
            return r.json()
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(30 * (i + 1))
            continue
        raise RuntimeError(f"{r.status_code}: {r.text[:300]}")
    raise RuntimeError("Demasiados reintentos")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inicio", default="2021-03-01")
    ap.add_argument("--fin", default="2025-12-31")
    ap.add_argument("--modelo", default="gfs_seamless")
    ap.add_argument("--leads", type=int, default=3)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    ciu = pd.read_csv(C.RAW / "numeric" / "ciudades.csv")
    (DESTINO / "crudo").mkdir(parents=True, exist_ok=True)
    trozos = []
    # un año por petición y ciudad: respeta los límites de la API gratuita
    rangos = [(max(pd.Timestamp(a.inicio), pd.Timestamp(f"{y}-01-01")), min(pd.Timestamp(a.fin), pd.Timestamp(f"{y}-12-31")))
              for y in range(pd.Timestamp(a.inicio).year, pd.Timestamp(a.fin).year + 1)]
    for _, c in ciu.iterrows():
        for ini, fin in rangos:
            cache = DESTINO / "crudo" / f"{c.ciudad}_{a.modelo}_{ini:%Y%m%d}_{fin:%Y%m%d}.json"
            if a.dry_run:
                print(c.ciudad, ini.date(), fin.date(), a.modelo)
                continue
            if cache.exists():
                js = json.loads(cache.read_text())
            else:
                js = pedir(c.ciudad, c.lat_solicitada, c.lon_solicitada, f"{ini:%Y-%m-%d}", f"{fin:%Y-%m-%d}", a.modelo, a.leads)
                cache.write_text(json.dumps(js))
                time.sleep(2)
            trozos.append(parsear_respuesta(js, c.ciudad))
    if a.dry_run:
        return
    d = pd.concat(trozos, ignore_index=True)
    d["temperature_2m_c"] = d["temperature_2m_c"].astype("float32")
    d.to_parquet(DESTINO / "pronosticos_temperatura.parquet", compression="zstd", index=False)
    cob = d.groupby(["lead_dias"]).valido.agg(["min", "max", "count"])
    print(cob)
    print("Revisar: huecos por ciudad/lead y que las series no sean idénticas entre leads (si lo son, la API no está entregando corridas distintas).")


if __name__ == "__main__":
    main()

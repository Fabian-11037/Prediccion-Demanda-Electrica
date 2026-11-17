"""Titulares con disponibilidad por origen (texto).

Solo se usa el titular. Dos escenarios, que se aplican igual en todas las
particiones:

* ``estricto``  solo registros con versión archivada verificada
  (``availability_status == verified_archived_version``) y ``available_at_verified``;
  la disponibilidad es la captura verificada, nunca la fecha declarada.
* ``declarado`` fecha declarada como disponibilidad (``available_at_assumed``):
  marca temporal con zona tal cual; fecha sin hora entra el día siguiente a las
  00:00 de Colombia. No certifica que la versión del titular sea la original.

Nunca se usan como característica: ``url``, fechas declaradas, ``availability_*``,
``quarantine_reason`` ni ``provenance_source``. Los registros en cuarentena y los
sin fecha quedan fuera.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

RUTA = C.RAW / "noticias" / "titulares.parquet"
ESCENARIOS = ("estricto", "declarado")


def cargar_titulares(escenario: str) -> pd.DataFrame:
    """Devuelve id, fuente, titular y ``disponible_desde`` (UTC) ordenado por disponibilidad."""
    if escenario not in ESCENARIOS:
        raise ValueError(f"escenario debe ser uno de {ESCENARIOS}")
    t = pd.read_parquet(RUTA)
    t = t[~t["quarantined"]]
    if escenario == "estricto":
        t = t[(t["availability_status"] == "verified_archived_version") & t["available_at_verified"].notna()
              & (t["availability_proof"] == "wayback_payload_digest_and_capture")]
        disp = pd.to_datetime(t["available_at_verified"], utc=True, format="ISO8601")
    else:
        t = t[t["available_at_assumed"].notna()]
        disp = pd.to_datetime(t["available_at_assumed"], utc=True, format="ISO8601")
    out = pd.DataFrame({"id": t["id"].values, "fuente": t["source"].values, "tipo": t["source_kind"].values,
                        "titular": t["titular"].values, "disponible_desde": disp.values})
    out = out[out["titular"].notna() & (out["titular"].str.strip() != "")]
    return out.sort_values("disponible_desde").reset_index(drop=True)


def titulares_en_origen(tit: pd.DataFrame, origen: pd.Timestamp, ventana_horas: int = 24, max_n: int | None = None) -> pd.DataFrame:
    """Titulares con ``origen - ventana <= disponible_desde <= origen`` (más recientes al final)."""
    o = pd.Timestamp(origen)
    o = (o.tz_localize(C.TZ) if o.tzinfo is None else o).tz_convert("UTC")
    v = tit["disponible_desde"].values
    hi = np.searchsorted(v, np.datetime64(o.tz_localize(None)), side="right")
    lo = np.searchsorted(v, np.datetime64((o - pd.Timedelta(hours=ventana_horas)).tz_localize(None)), side="left")
    sel = tit.iloc[lo:hi]
    return sel.tail(max_n) if max_n else sel


def cobertura(tit: pd.DataFrame) -> pd.DataFrame:
    """Titulares por fuente y año de disponibilidad (diagnóstico de cobertura)."""
    a = pd.DatetimeIndex(tit["disponible_desde"]).tz_localize("UTC").tz_convert(C.TZ).year
    return tit.groupby(["fuente", a]).size().unstack(0).fillna(0).astype(int)

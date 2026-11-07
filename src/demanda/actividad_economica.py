"""Indicador de Seguimiento a la Economía (ISE, DANE) con control de disponibilidad.

Archivo esperado: ``data/external/actividad_economica/ise_dane.csv`` con columnas
``mes`` (AAAA-MM), ``ise_original``, ``ise_desestacionalizado``, ``ise_tendencia_ciclo``
y, opcionalmente, ``vintage`` (fecha de descarga de la cifra) y ``fecha_publicacion``
(fecha real de publicación si se conoce; si falta se usa fin de mes + ``ISE_REZAGO_DIAS``).

Advertencia de vintage: el DANE revisa las cifras de meses anteriores en cada
publicación. Una serie descargada hoy contiene valores revisados que no existían
en la fecha de publicación original. Usar solo ``fecha_publicacion`` no corrige
ese sesgo; hay que declararlo como limitación o reconstruir vintages.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import config as C

RUTA_ISE = C.EXTERNO / "actividad_economica" / "ise_dane.csv"
COLUMNAS = ["ise_original", "ise_desestacionalizado", "ise_tendencia_ciclo"]


def cargar_ise(ruta: Path | str = RUTA_ISE) -> pd.DataFrame | None:
    """Lee y valida el ISE; devuelve None si el archivo no existe o está vacío."""
    ruta = Path(ruta)
    if not ruta.exists():
        return None
    d = pd.read_csv(ruta)
    if d.empty:
        return None
    faltan = {"mes", *COLUMNAS} - set(d.columns)
    if faltan:
        raise ValueError(f"Faltan columnas en {ruta}: {sorted(faltan)}")
    d["mes"] = pd.PeriodIndex(d["mes"], freq="M")
    if d["mes"].duplicated().any():
        raise ValueError("Meses duplicados en el ISE")
    d = d.sort_values("mes").reset_index(drop=True)
    fin_mes = d["mes"].dt.end_time.dt.normalize() + pd.Timedelta(days=1)  # 00:00 del mes siguiente
    disp = fin_mes + pd.Timedelta(days=C.ISE_REZAGO_DIAS)
    if "fecha_publicacion" in d.columns:
        real = pd.to_datetime(d["fecha_publicacion"], errors="coerce")
        # la fecha real solo puede adelantar la cota si es plausible (posterior al mes de referencia)
        ok = real.notna() & (real >= fin_mes)
        disp = disp.where(~ok, real + pd.Timedelta(days=1))  # entra al día siguiente de publicarse
    d["disponible_desde"] = pd.DatetimeIndex(disp).tz_localize(C.TZ)
    return d


def ise_as_of(indice_horario: pd.DatetimeIndex, ise: pd.DataFrame) -> pd.DataFrame:
    """Último ISE disponible para cada fila horaria (información hasta ``t+1h``)."""
    q = pd.DataFrame({"info_hasta": indice_horario + pd.Timedelta(hours=1)}, index=pd.DatetimeIndex(indice_horario, name="inicio"))
    d = ise.sort_values("disponible_desde")
    m = pd.merge_asof(q.reset_index(), d, left_on="info_hasta", right_on="disponible_desde", direction="backward").set_index("inicio")
    out = m[COLUMNAS].add_prefix("x_")
    ref_fin = m["mes"].map(lambda p: p.end_time if pd.notna(p) else pd.NaT)
    edad = (m["info_hasta"].dt.tz_localize(None) - pd.to_datetime(ref_fin)).dt.days / 30.44
    out["x_ise_edad_meses"] = edad.astype("float32")
    return out

"""Calendario de festivos de Colombia y variables de calendario.

Implementación propia de la Ley 51 de 1983 (Ley Emiliani), verificada en las
pruebas contra el paquete ``holidays`` (país CO). Todas las variables de este
módulo son conocidas con antelación (clase ``known_future``): dependen solo de
la fecha calendario y no de ninguna observación.

Reglas (años >= 1984):
  * Fijos: 1-ene, 1-may, 20-jul, 7-ago, 8-dic, 25-dic.
  * Trasladables al lunes siguiente (si la fecha no cae en lunes): 6-ene,
    19-mar, 29-jun, 15-ago, 12-oct, 1-nov, 11-nov.
  * Ligados a la Pascua: Jueves y Viernes Santo (sin traslado); Ascensión
    (Pascua + 39 d), Corpus Christi (+60 d) y Sagrado Corazón (+68 d),
    trasladados al lunes siguiente.
  * Desde 2026: Nuestra Señora del Rosario de Chiquinquirá (9-jul, trasladable).
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
from dateutil.easter import easter

FIJOS = {
    (1, 1): "Año Nuevo",
    (5, 1): "Día del Trabajo",
    (7, 20): "Día de la Independencia",
    (8, 7): "Batalla de Boyacá",
    (12, 8): "Inmaculada Concepción",
    (12, 25): "Navidad",
}
EMILIANI = {
    (1, 6): "Reyes Magos",
    (3, 19): "San José",
    (6, 29): "San Pedro y San Pablo",
    (8, 15): "Asunción de la Virgen",
    (10, 12): "Día de la Raza",
    (11, 1): "Todos los Santos",
    (11, 11): "Independencia de Cartagena",
}
# Festivos creados por leyes posteriores: nombre -> (mes, dia, anio_inicio, traslada_a_lunes)
# Ley 2578 de 2026 (1-jun-2026): Nuestra Señora del Rosario de Chiquinquirá, 9-jul,
# con traslado según Ley 51/1983 (verificado en prensa y en ``holidays``; no en el texto de la ley).
POSTERIORES = {
    "Nuestra Señora del Rosario de Chiquinquirá": (7, 9, 2026, True),
}
PASCUA = {  # nombre: (dias desde Pascua, traslada_a_lunes)
    "Jueves Santo": (-3, False),
    "Viernes Santo": (-2, False),
    "Ascensión del Señor": (39, True),
    "Corpus Christi": (60, True),
    "Sagrado Corazón": (68, True),
}


def _siguiente_lunes(d: date) -> date:
    """Mismo día si es lunes; si no, el lunes siguiente."""
    return d + timedelta(days=(7 - d.weekday()) % 7)


def festivos_colombia(anios) -> pd.DataFrame:
    """Festivos legales de Colombia para los años dados.

    Devuelve columnas: fecha, nombre, tipo (fijo | emiliani | pascua | ley_posterior).
    """
    filas = []
    for y in sorted(set(int(a) for a in anios)):
        if y < 1984:
            raise ValueError("La Ley 51/1983 aplica desde 1984")
        for (m, d), n in FIJOS.items():
            filas.append((date(y, m, d), n, "fijo"))
        for (m, d), n in EMILIANI.items():
            filas.append((_siguiente_lunes(date(y, m, d)), n, "emiliani"))
        for n, (m, d, y0, mover) in POSTERIORES.items():
            if y >= y0:
                f = date(y, m, d)
                filas.append((_siguiente_lunes(f) if mover else f, n, "ley_posterior"))
        p = easter(y)
        for n, (off, mover) in PASCUA.items():
            f = p + timedelta(days=off)
            filas.append((_siguiente_lunes(f) if mover else f, n, "pascua"))
    df = pd.DataFrame(filas, columns=["fecha", "nombre", "tipo"])
    df["fecha"] = pd.to_datetime(df["fecha"])
    return df.sort_values("fecha").reset_index(drop=True)


def semana_santa(anios) -> pd.DataFrame:
    """Domingo de Ramos .. Domingo de Pascua por año (inclusive)."""
    filas = []
    for y in sorted(set(int(a) for a in anios)):
        p = easter(y)
        filas.append((pd.Timestamp(p - timedelta(days=7)), pd.Timestamp(p)))
    return pd.DataFrame(filas, columns=["inicio", "fin"])


def calendario_diario(inicio: str, fin: str) -> pd.DataFrame:
    """Variables de calendario diarias (todas ``known_future``).

    ``dias_al_siguiente_festivo`` y ``dias_desde_festivo_previo`` se calculan con
    el calendario legal, que es público con años de antelación; los valores se
    truncan en 30 días para evitar colas largas.
    """
    f0, f1 = pd.Timestamp(inicio), pd.Timestamp(fin)
    anios = range(f0.year - 1, f1.year + 2)
    fest = festivos_colombia(anios)
    fechas = pd.date_range(f0, f1, freq="D")
    c = pd.DataFrame({"fecha": fechas})
    c["dow"] = c.fecha.dt.dayofweek  # 0 = lunes
    c["mes"] = c.fecha.dt.month
    c["dia_anio"] = c.fecha.dt.dayofyear
    c["es_festivo"] = c.fecha.isin(fest.fecha).astype("int8")
    c["es_domingo"] = (c.dow == 6).astype("int8")
    c["es_sabado"] = (c.dow == 5).astype("int8")
    c["es_no_laboral"] = ((c.es_festivo == 1) | (c.es_domingo == 1)).astype("int8")
    # tipo de día: 0 laboral, 1 sábado no festivo, 2 domingo o festivo
    c["tipo_dia"] = np.select([c.es_no_laboral == 1, c.es_sabado == 1], [2, 1], 0).astype("int8")

    fs = fest.fecha.values
    idx_prev = np.searchsorted(fs, c.fecha.values, side="left") - 1
    prev = np.where(idx_prev >= 0, fs[np.maximum(idx_prev, 0)], np.datetime64("NaT", "ns"))
    # festivo anterior estrictamente previo (arriba) y siguiente estrictamente posterior
    idx_s = np.searchsorted(fs, c.fecha.values, side="right")
    sig = np.where(idx_s < len(fs), fs[np.minimum(idx_s, len(fs) - 1)], np.datetime64("NaT", "ns"))
    c["dias_al_siguiente_festivo"] = ((pd.to_datetime(sig) - c.fecha).dt.days).clip(upper=30).astype("float32")
    c["dias_desde_festivo_previo"] = ((c.fecha - pd.to_datetime(prev)).dt.days).clip(upper=30).astype("float32")
    c["es_vispera_festivo"] = (c.dias_al_siguiente_festivo == 1).astype("int8")
    c["es_dia_post_festivo"] = (c.dias_desde_festivo_previo == 1).astype("int8")
    # puente: día laborable ubicado entre un festivo y un fin de semana
    # (festivo en viernes/lunes se resuelve por la propia ley; se marca el
    # sábado y domingo adyacentes a un festivo lunes o viernes como "fin de semana largo")
    fest_set = set(fest.fecha)
    fsl = np.zeros(len(c), dtype="int8")
    for i, d in enumerate(c.fecha):
        if d.dayofweek in (4, 5, 6, 0):
            lunes = d + pd.Timedelta(days=(7 - d.dayofweek) % 7) if d.dayofweek != 0 else d
            viernes = lunes - pd.Timedelta(days=3)
            if lunes in fest_set or viernes in fest_set:
                fsl[i] = 1
    c["es_fin_semana_largo"] = fsl

    ss = semana_santa(range(f0.year, f1.year + 1))
    c["es_semana_santa"] = 0
    for _, r in ss.iterrows():
        c.loc[(c.fecha >= r.inicio) & (c.fecha <= r.fin), "es_semana_santa"] = 1
    c["es_semana_santa"] = c.es_semana_santa.astype("int8")
    md = c.fecha.dt.month * 100 + c.fecha.dt.day
    c["es_temporada_navidena"] = ((md >= 1216) | (md <= 106)).astype("int8")
    c["es_fin_de_anio"] = ((md >= 1224) | (md <= 101)).astype("int8")
    return c


def calendario_horario(index_inicio: pd.DatetimeIndex) -> pd.DataFrame:
    """Variables de calendario por hora; ``index_inicio`` = inicio del intervalo (hora Colombia).

    Se adjunta ``hora`` 0..23 con codificación cíclica (hora y día del año)."""
    if index_inicio.tz is None:
        raise ValueError("El índice debe tener zona horaria America/Bogota")
    dias = calendario_diario(index_inicio.min().strftime("%Y-%m-%d"), index_inicio.max().strftime("%Y-%m-%d"))
    fecha = pd.DatetimeIndex(index_inicio.tz_localize(None).normalize())
    d = dias.set_index("fecha").loc[fecha].reset_index(drop=True)
    out = d.copy()
    out.index = index_inicio
    h = index_inicio.hour.values
    out["hora"] = h.astype("int8")
    out["hora_sin"] = np.sin(2 * np.pi * h / 24).astype("float32")
    out["hora_cos"] = np.cos(2 * np.pi * h / 24).astype("float32")
    out["dow_sin"] = np.sin(2 * np.pi * out.dow / 7).astype("float32")
    out["dow_cos"] = np.cos(2 * np.pi * out.dow / 7).astype("float32")
    out["doy_sin"] = np.sin(2 * np.pi * out.dia_anio / 366).astype("float32")
    out["doy_cos"] = np.cos(2 * np.pi * out.dia_anio / 366).astype("float32")
    return out

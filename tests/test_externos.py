import numpy as np
import pandas as pd

from demanda import config as C
from demanda import actividad_economica as AE
from demanda import noticias as N
from demanda import pronosticos as P
from demanda.climatologia import ClimatologiaTemperatura


def _ise_sintetico(tmp_path):
    d = pd.DataFrame({"mes": ["2024-01", "2024-02", "2024-03"], "ise_original": [100.0, 101.0, 102.0],
                      "ise_desestacionalizado": [100.5, 101.5, 102.5], "ise_tendencia_ciclo": [100.2, 101.2, 102.2]})
    f = tmp_path / "ise.csv"
    d.to_csv(f, index=False)
    return AE.cargar_ise(f)


def test_ise_disponibilidad_conservadora(tmp_path):
    ise = _ise_sintetico(tmp_path)
    idx = pd.date_range("2024-01-01", "2024-07-01", freq="h", tz=C.TZ)
    out = AE.ise_as_of(idx, ise)
    # enero (fin 1-feb) + 60 d = 1-abr: antes de esa fecha no hay dato
    assert out.loc[:"2024-03-31 22:00", "x_ise_original"].isna().all()
    assert out.loc["2024-04-01 00:00", "x_ise_original"] == 100.0
    # febrero (fin 1-mar) + 60 d = 30-abr
    assert out.loc["2024-04-29 22:00", "x_ise_original"] == 100.0
    assert out.loc["2024-04-30 00:00", "x_ise_original"] == 101.0


def test_pronostico_solo_emision_previa():
    ciudades = ["bogota"]
    idx = pd.date_range("2024-01-01", periods=24 * 10, freq="h", tz=C.TZ)
    filas = []
    for lead in (1, 2, 3):
        for v in idx:
            # el valor codifica la emisión aproximada: horas desde 2024-01-01 de (valido - lead*24h)
            emision_h = (v - pd.Timestamp("2024-01-01", tz=C.TZ)) / pd.Timedelta(hours=1) - 24 * lead
            filas.append((v, "bogota", lead, emision_h))
    d = pd.DataFrame(filas, columns=["valido", "ciudad", "lead_dias", "temperature_2m_c"])
    H = 24
    out = P.pronostico_futuro(idx, d, ciudades, H)
    for p in (80, 120, 200):
        for h in range(H):
            emision = out[p, h, 0]
            assert emision + C.PRONOSTICO_MARGEN_H + C.PRONOSTICO_LATENCIA_H <= p  # emisión + latencias <= t0
    assert P.lead_requerido(0) == 1 and P.lead_requerido(23) == 2


def test_climatologia_solo_train(panel):
    cl = ClimatologiaTemperatura().ajustar(panel, C.PARTICION["train"][1])
    alt = panel.copy()
    fin = pd.Timestamp(C.PARTICION["train"][1], tz=C.TZ) + pd.Timedelta(days=1)
    alt.loc[alt.index >= fin, [c for c in alt.columns if c.startswith("x_obs_temperature")]] = 99.0
    cl2 = ClimatologiaTemperatura().ajustar(alt, C.PARTICION["train"][1])
    pd.testing.assert_frame_equal(cl.tabla_, cl2.tabla_)


def test_titulares_disponibles_en_origen():
    for esc in N.ESCENARIOS:
        t = N.cargar_titulares(esc)
        assert t["disponible_desde"].is_monotonic_increasing
        o = pd.Timestamp("2024-06-03 06:00", tz=C.TZ)
        s = N.titulares_en_origen(t, o, 48)
        assert (pd.to_datetime(s["disponible_desde"]) <= o.tz_convert("UTC").tz_localize(None)).all()


def test_escenario_estricto_solo_verificado():
    raw = pd.read_parquet(N.RUTA)
    t = N.cargar_titulares("estricto")
    ids = set(t["id"])
    sub = raw[raw["id"].isin(ids)]
    assert (sub["availability_status"] == "verified_archived_version").all()
    assert not sub["quarantined"].any()
    assert sub["available_at_verified"].notna().all()


def test_titulares_declarados_excluyen_cuarentena_y_sin_fecha():
    raw = pd.read_parquet(N.RUTA)
    t = N.cargar_titulares("declarado")
    excl = raw[raw["quarantined"] | raw["available_at_assumed"].isna()]["id"]
    assert not set(excl) & set(t["id"])

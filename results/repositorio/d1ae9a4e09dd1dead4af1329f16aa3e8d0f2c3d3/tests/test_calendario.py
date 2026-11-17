import holidays
import pandas as pd

from demanda.calendario import calendario_diario, festivos_colombia


def test_coincide_con_paquete_holidays():
    for y in range(2000, 2031):
        mio = set(festivos_colombia([y]).fecha.dt.date)
        lib = set(holidays.CO(years=y).keys())
        assert mio == lib, (y, sorted(mio ^ lib))


def test_fechas_conocidas_2026():
    f = set(festivos_colombia([2026]).fecha.dt.strftime("%m-%d"))
    assert {"01-12", "03-23", "05-18", "06-08", "06-15", "07-13", "08-17", "11-02", "11-16"} <= f


def test_festivos_trasladables_caen_en_lunes():
    f = festivos_colombia(range(2000, 2031))
    mov = f[f.tipo.isin(["emiliani", "ley_posterior"]) | f.nombre.isin(["Ascensión del Señor", "Corpus Christi", "Sagrado Corazón"])]
    assert (mov.fecha.dt.dayofweek == 0).all()


def test_banderas_calendario():
    c = calendario_diario("2024-03-20", "2024-04-05").set_index("fecha")
    assert c.loc["2024-03-28", "es_festivo"] == 1  # Jueves Santo
    assert c.loc["2024-03-29", "es_festivo"] == 1  # Viernes Santo
    assert c.loc["2024-03-24", "es_semana_santa"] == 1  # Domingo de Ramos
    assert c.loc["2024-03-27", "es_vispera_festivo"] == 1
    assert c.loc["2024-03-30", "es_dia_post_festivo"] == 1


def test_calendario_solo_depende_de_la_fecha():
    # el mismo día calculado en dos rangos distintos da idéntico resultado (sin estado externo)
    a = calendario_diario("2023-12-01", "2024-01-31").set_index("fecha").loc["2024-01-06"]
    b = calendario_diario("2024-01-01", "2024-02-28").set_index("fecha").loc["2024-01-06"]
    pd.testing.assert_series_equal(a, b)

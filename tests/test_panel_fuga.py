import numpy as np
import pandas as pd
import pytest

from demanda import config as C
from demanda.particion import AjustadorEscala, origenes
from demanda.ventanas import columnas_por_clase, construir_ventanas


def test_registro_clasifica_todas_las_columnas(panel, registro):
    assert set(panel.columns) <= set(registro)
    assert "demanda_comercial_kwh" not in panel.columns
    assert not any(c.startswith("x_") and "comercial" in c for c in panel.columns)


def test_rezago_demanda_y_precio(panel):
    y = panel["y_" + C.OBJETIVO]
    k = C.REZAGO_H["demanda_real"]
    assert np.allclose(panel["x_" + C.OBJETIVO].iloc[k:], y.iloc[:-k])
    assert panel["x_" + C.OBJETIVO].iloc[:k].isna().all()
    assert panel["x_precio_bolsa_cop_kwh"].iloc[: C.REZAGO_H["precio_bolsa"]].isna().all()


def test_hidrologia_as_of(panel):
    muestra = panel.dropna(subset=["aux_hid_fecha_dato"]).sample(500, random_state=0)
    info_hasta = muestra.index + pd.Timedelta(hours=1)
    disp = (muestra["aux_hid_fecha_dato"] + pd.Timedelta(days=1, hours=C.REZAGO_H["hidrologia"])).dt.tz_localize(C.TZ)
    assert (disp <= info_hasta).all()
    # y el día siguiente aún no estaba disponible (se usa el más reciente posible)
    disp_sig = disp + pd.Timedelta(days=1)
    assert (disp_sig > info_hasta).all()


def test_ninguna_entrada_contiene_el_objetivo_de_la_misma_fila(panel, registro):
    xs = columnas_por_clase(registro, "past_only")
    y = panel["y_" + C.OBJETIVO]
    for c in xs:
        assert not np.allclose(panel[c].dropna().values[:1000], y.loc[panel[c].dropna().index].values[:1000]), c


def _ventana(panel, registro, p, pan=None):
    xc = columnas_por_clase(registro, "past_only")
    kc = columnas_por_clase(registro, "known_future")
    return construir_ventanas(pan if pan is not None else panel, np.array([p]), xc, kc), xc, kc


def test_ventana_no_lee_filas_futuras(panel, registro):
    p = int(np.where(panel.index == pd.Timestamp("2015-06-10 00:00", tz=C.TZ))[0][0])
    base, xc, kc = _ventana(panel, registro, p)
    alt = panel.copy()
    cols_x = [c for c in alt.columns if c.startswith("x_") or c.startswith("aux_")]
    # corromper todas las entradas x_* desde el origen en adelante, y las etiquetas más allá del horizonte
    alt.iloc[p:, [alt.columns.get_loc(c) for c in xc]] = 1e12
    alt.iloc[p + C.HORIZONTE:, alt.columns.get_loc("y_" + C.OBJETIVO)] = -1e12
    nuevo, _, _ = _ventana(panel, registro, p, alt)
    for k in ("x_pasado", "cal_pasado", "cal_futuro", "y"):
        assert np.array_equal(base[k], nuevo[k]), k
    # sanidad: corromper el pasado sí cambia la entrada
    alt2 = panel.copy()
    alt2.iloc[p - 1, [alt2.columns.get_loc(c) for c in xc]] = 1e12
    cambiado, _, _ = _ventana(panel, registro, p, alt2)
    assert not np.array_equal(base["x_pasado"], cambiado["x_pasado"])


def test_cal_futuro_solo_calendario(panel, registro):
    kc = columnas_por_clase(registro, "known_future")
    assert all(c.startswith("cal_") for c in kc)
    with pytest.raises(ValueError):
        construir_ventanas(panel, np.array([500]), ["y_" + C.OBJETIVO], kc)
    with pytest.raises(ValueError):
        construir_ventanas(panel, np.array([500]), ["x_precio_bolsa_cop_kwh"], ["x_demanda_real_kwh"])


def test_particiones_sin_solape_de_etiquetas(panel):
    idx = panel.index
    tr = origenes(idx, "train")
    va = origenes(idx, "val")
    fin_tr = idx[tr].max() + pd.Timedelta(hours=C.HORIZONTE - 1)
    assert fin_tr < idx[va].min()
    assert idx[tr].max() + pd.Timedelta(hours=C.HORIZONTE - 1) <= pd.Timestamp(C.PARTICION["train"][1], tz=C.TZ) + pd.Timedelta(hours=23)
    te = origenes(idx, "test", allow_test=True)
    assert idx[va].max() + pd.Timedelta(hours=C.HORIZONTE - 1) < idx[te].min()


def test_prueba_congelada(panel):
    with pytest.raises(PermissionError):
        origenes(panel.index, "test")


def test_escalas_solo_con_entrenamiento(panel, registro):
    xc = columnas_por_clase(registro, "past_only")
    a = AjustadorEscala().ajustar(panel, xc)
    alt = panel.copy()
    fin = pd.Timestamp(C.PARTICION["train"][1], tz=C.TZ) + pd.Timedelta(days=1)
    alt.loc[alt.index >= fin, xc] = 1e9  # alterar validación y prueba
    b = AjustadorEscala().ajustar(alt, xc)
    pd.testing.assert_series_equal(a.media_, b.media_)
    pd.testing.assert_series_equal(a.std_, b.std_)

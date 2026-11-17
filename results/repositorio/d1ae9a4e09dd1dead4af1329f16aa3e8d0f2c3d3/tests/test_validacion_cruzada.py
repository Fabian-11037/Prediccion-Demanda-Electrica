import numpy as np
import pandas as pd
import pytest

from demanda import config as C
from demanda.particion import folds_forward_chaining, limites, pretest

H = pd.Timedelta(hours=C.HORIZONTE - 1)


def test_pliegues_expansivos_sin_solape_ni_fuga(panel):
    idx = panel.index
    folds = folds_forward_chaining(idx)
    assert [f["anio"] for f in folds] == C.CV_ANIOS
    fin_ant = None
    for f in folds:
        ini_val = pd.Timestamp(f"{f['anio']}-01-01", tz=C.TZ)
        # los objetivos de entrenamiento y parada temprana terminan antes del primer instante de validación (purga)
        assert (idx[f["tr"]] + H < ini_val).all() and (idx[f["es"]] + H < ini_val).all()
        # parada temprana posterior al entrenamiento, con purga entre ambos
        assert idx[f["tr"]].max() + H < idx[f["es"]].min()
        # validación completa dentro del año y sin orígenes compartidos
        assert (idx[f["val"]] >= ini_val).all() and (idx[f["val"]] + H <= ini_val + pd.Timedelta(days=366) - pd.Timedelta(hours=1)).all()
        assert not (set(f["tr"]) & set(f["val"])) and not (set(f["es"]) & set(f["val"])) and not (set(f["tr"]) & set(f["es"]))
        # ventana expansiva: el conjunto previo (tr+es) solo crece
        fin = idx[np.concatenate([f["tr"], f["es"]])].max()
        assert fin_ant is None or fin > fin_ant
        fin_ant = fin


def test_los_pliegues_no_tocan_la_prueba(panel):
    with pytest.raises(PermissionError):
        folds_forward_chaining(panel.index, [2021, 2022])
    ini_test, _ = limites("test")
    for f in folds_forward_chaining(panel.index):
        assert (panel.index[f["val"]] + H < ini_test).all()


def test_ajuste_final_solo_usa_origenes_previos_a_la_prueba(panel):
    tr, es = pretest(panel.index)
    ini_test, _ = limites("test")
    assert (panel.index[tr] + H < ini_test).all() and (panel.index[es] + H < ini_test).all()
    assert panel.index[tr].max() + H < panel.index[es].min()

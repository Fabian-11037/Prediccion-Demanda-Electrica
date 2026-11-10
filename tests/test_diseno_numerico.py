"""Invariantes temporales del contrato ampliado, sin leer prueba ni entrenar."""
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from demanda.diseno_numerico import preparar_numericas_comunes

ROOT = Path(__file__).resolve().parents[1]


def ejemplo():
    idx = pd.date_range("2010-01-01", periods=240, freq="h", tz="America/Bogota")
    p = pd.DataFrame({"x_demanda": np.arange(240.), "x_obs_temp_ciudad": np.arange(240.), "cal_hora": idx.hour}, index=idx)
    reg = {"x_demanda": {"clase": "past_only"}, "x_obs_temp_ciudad": {"clase": "past_only", "rezago_h": 1}, "cal_hora": {"clase": "known_future"}}
    cfg = {"numericas": {"historicas": ["x_demanda", "x_obs_temp_ciudad"], "calendario": ["cal_hora"], "rezago_clima_h": 168}}
    return p, reg, cfg


def test_rezago_adicional_y_falta_sin_imputar():
    p, reg, cfg = ejemplo()
    original = p.copy(deep=True)
    out, m = preparar_numericas_comunes(p, reg, cfg)
    assert out.iloc[166].x_obs_temp_ciudad != out.iloc[166].x_obs_temp_ciudad
    assert out.iloc[167].x_obs_temp_ciudad == 0
    assert m.iloc[166].ausente_x_obs_temp_ciudad == 1
    pd.testing.assert_frame_equal(p, original)


def test_cambios_futuros_no_alteran_pasado():
    p, reg, cfg = ejemplo()
    a, _ = preparar_numericas_comunes(p, reg, cfg)
    p.iloc[200:, :] = 999999
    b, _ = preparar_numericas_comunes(p, reg, cfg)
    pd.testing.assert_frame_equal(a.iloc[:200], b.iloc[:200])


def test_rechaza_adelantos_huecos_y_rol_futuro_invalido():
    p, reg, cfg = ejemplo()
    with pytest.raises(ValueError):
        preparar_numericas_comunes(p.drop(p.index[20]), reg, cfg)
    malo = copy.deepcopy(cfg); malo["numericas"]["rezago_clima_h"] = 0
    with pytest.raises(ValueError):
        preparar_numericas_comunes(p, reg, malo)
    malo = copy.deepcopy(reg); malo["cal_hora"]["clase"] = "target"
    with pytest.raises(ValueError):
        preparar_numericas_comunes(p, malo, cfg)


def test_contrato_comun_cubre_los_cinco_bloques():
    c = json.loads((ROOT/"config/diseno_entrenamiento_texto.json").read_text(encoding="utf-8"))
    x = c["numericas"]["historicas"]
    assert len(x) == 30 and len(c["numericas"]["calendario"]) == 23
    assert sum(v.startswith("x_obs_") for v in x) == 24
    assert sum(v.startswith("x_hid_") for v in x) == 4
    assert "x_precio_bolsa_cop_kwh" in x
    assert len(c["variantes"]) == 3
    assert c["metrica"]["direccion"] == "min"

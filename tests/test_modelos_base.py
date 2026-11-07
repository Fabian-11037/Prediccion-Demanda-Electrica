"""Pruebas del pipeline de entrenamiento SIN entrenar: muestreo, normalización, referencias y formas."""
import json

import numpy as np
import pandas as pd
import pytest

from demanda import config as C
from demanda import modelos_base as M
from demanda.metricas import evaluar, n_semillas_requerido


@pytest.fixture(scope="module")
def datos(panel, registro):
    return M.Datos(panel, registro, M.ConfigDatos(covariables="completo", max_origenes_train=200))


def test_mismo_presupuesto_y_reproducibilidad_del_muestreo():
    for nombre in M.MODELOS_PYTORCH:
        a = [M.muestrear(M.ESPACIOS[nombre], np.random.default_rng(1)) for _ in range(5)]
        b = [M.muestrear(M.ESPACIOS[nombre], np.random.default_rng(1)) for _ in range(5)]
        assert a == b


def test_fijar_saca_el_parametro_de_la_busqueda():
    rng = np.random.default_rng(0)
    cfgs = [M.muestrear(M.ESPACIOS["lstm"], rng, {"unidades": 64}) for _ in range(10)]
    assert all(c["unidades"] == 64 for c in cfgs)
    assert len({c["dropout"] for c in cfgs}) > 1


def test_alpha_condicional_respeta_el_rango_de_cada_penalizacion():
    rng = np.random.default_rng(0)
    for _ in range(50):
        p = M.muestrear(M.ESPACIOS["arx_lineal"], rng)
        lo, hi = (1e-6, 1e-1) if p["penalizacion"] == "ridge" else (1e-7, 1e-2)
        assert lo <= p["lambda"] <= hi


def test_normalizacion_solo_con_train_y_sin_nan(datos):
    for s in ("train", "val"):
        c = datos.conjunto(s)
        assert not np.isnan(c.x).any() and not np.isnan(c.cal_f).any()
        assert c.x.shape[1] == C.VENTANA and c.y.shape[1] == C.HORIZONTE
    # la demanda de la ventana (modo nivel) tiene media 1 por construcción
    assert np.allclose(datos.conjunto("train").x[:, :, 0].mean(axis=1), 1.0, atol=1e-4)
    # el conjunto de prueba sigue protegido
    with pytest.raises(PermissionError):
        datos.conjunto("test")


def test_objetivo_se_invierte(datos):
    c = datos.conjunto("val")
    assert np.allclose(datos.invertir(datos.objetivo_norm(c), c), c.y, rtol=1e-5)


def test_referencias_respetan_el_rezago_de_publicacion(datos, panel):
    refs = datos.referencias("val")
    c = datos.conjunto("val")
    y = panel["y_" + C.OBJETIVO]
    p = int(np.where(panel.index == c.origen[3])[0][0])
    # ingenuo de 24 h: para h=23 se usa el día anterior al previo (el dato de t0-1h aún no es utilizable)
    assert refs["ingenuo_24h"][3, 23] == pytest.approx(y.iloc[p + 23 - 48])
    assert refs["ingenuo_24h"][3, 0] == pytest.approx(y.iloc[p - 24])
    assert refs["ingenuo_168h"][3, 5] == pytest.approx(y.iloc[p + 5 - 168])


def test_formas_y_forward_de_las_redes(datos):
    torch = pytest.importorskip("torch")
    tr = datos.conjunto("train")
    n = 8
    xp = torch.from_numpy(np.concatenate([tr.x[:n], tr.cal_p[:n]], axis=2))
    cf = torch.from_numpy(tr.cal_f[:n])
    params = {"arx_lineal": {"penalizacion": "lasso", "lambda": 1e-4, "lags": 48},
              "mlp": {"capas": 2, "unidades": 16, "dropout": 0.1},
              "lstm": {"capas": 2, "unidades": 8, "dropout": 0.1}}
    for nombre, p in params.items():
        red = M.construir_red(nombre, p, tr)
        red.preparar(torch.from_numpy(np.concatenate([tr.x, tr.cal_p], axis=2)), torch.from_numpy(tr.cal_f))
        assert red(xp, cf).shape == (n, C.HORIZONTE)
        assert float(torch.as_tensor(red.penalizacion()).detach()) >= 0


def test_dispositivo_cuda_sin_gpu_da_error_claro():
    torch = pytest.importorskip("torch")
    if torch.cuda.is_available():
        pytest.skip("hay GPU")
    with pytest.raises(RuntimeError):
        M.resolver_dispositivo("cuda")
    assert M.resolver_dispositivo("auto").type == "cpu"


def test_metricas_y_potencia():
    y = np.array([[1.0, 2.0], [3.0, 4.0]])
    m = evaluar(y, y + 1, escala=2.0, horizontes=(1, 2))
    assert m["MASE"] == pytest.approx(0.5) and m["MAE"] == pytest.approx(1.0) and m["MASE_h2"] == pytest.approx(0.5)
    assert n_semillas_requerido(0.0, 1.0, 0.03) == 1
    assert n_semillas_requerido(0.02, 1.0, 0.03) > n_semillas_requerido(0.01, 1.0, 0.03)

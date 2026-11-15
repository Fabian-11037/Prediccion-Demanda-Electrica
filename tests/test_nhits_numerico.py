"""Equivalencia con la interpolación oficial y uso de las entradas ampliadas."""
import torch
from demanda.nhits_numerico import NHITSNumerico, BaseLinealDeterminista
from demanda.vendor.nhits.core import _IdentityBasis


def test_interpolacion_equivale_oficial_en_salida_y_gradiente():
    torch.set_num_threads(1)
    for nodos in (6, 12, 24):
        a = torch.randn(3, 168 + nodos, requires_grad=True)
        b = a.detach().clone().requires_grad_()
        ba, ya = _IdentityBasis(168, 24, 'linear')(a)
        bb, yb = BaseLinealDeterminista(168, 24, nodos)(b)
        torch.testing.assert_close(ba, bb)
        torch.testing.assert_close(ya, yb)
        pesos = torch.randn_like(ya)
        ((ba ** 2).sum() + (ya * pesos).sum()).backward()
        ((bb ** 2).sum() + (yb * pesos).sum()).backward()
        torch.testing.assert_close(a.grad, b.grad)


def test_nhits_dimensiones_y_gradientes_de_covariables():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    model = NHITSNumerico(dict(dropout=.1), .1).eval()
    x = torch.randn(2, 168, 83, requires_grad=True)
    cal = torch.randn(2, 24, 23, requires_grad=True)
    y = model(x, cal)
    assert y.shape == (2, 24) and torch.isfinite(y).all()
    y.square().mean().backward()
    assert torch.isfinite(x.grad).all() and torch.isfinite(cal.grad).all()
    assert x.grad[:, :, 1:30].abs().sum() > 0
    assert x.grad[:, :, 53:].abs().sum() > 0
    assert cal.grad.abs().sum() > 0


def test_prueba_exige_seleccion_y_semillas_antes_de_leer_datos(tmp_path, monkeypatch):
    import importlib.util
    from pathlib import Path
    import pytest
    ruta = Path(__file__).resolve().parents[1] / 'scripts/entrenar_nhits_numerico.py'
    spec = importlib.util.spec_from_file_location('runner_nhits_test', ruta)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    monkeypatch.setattr(runner, 'OUT', tmp_path)
    with pytest.raises(AssertionError, match='selección y semillas'):
        runner.pack('final')

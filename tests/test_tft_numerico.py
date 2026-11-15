"""Entradas ampliadas y máscara causal del TFT de referencia."""
import torch
from demanda.tft_numerico import TFTNumerico


def test_tft_usa_covariables_y_propaga_gradientes_finitos():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    m = TFTNumerico(dict(dropout=.1), .1).eval()
    x = torch.randn(2, 168, 83, requires_grad=True)
    cal = torch.randn(2, 24, 23, requires_grad=True)
    y = m(x, cal)
    assert y.shape == (2, 24) and torch.isfinite(y).all()
    y.square().mean().backward()
    for g in (x.grad, cal.grad):
        assert torch.isfinite(g).all()
    assert x.grad[:, :, 1:30].abs().sum() > 0
    assert x.grad[:, :, 53:].abs().sum() > 0
    assert cal.grad.abs().sum() > 0
    # El calendario posterior no puede alterar la primera hora pronosticada.
    with torch.no_grad():
        alterado = cal.detach().clone()
        alterado[:, 1:] += 10
        torch.testing.assert_close(m(x, alterado)[:, 0], y[:, 0])

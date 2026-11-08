"""Contratos de formas, dilataciones y gradientes de las diez variantes."""
from types import SimpleNamespace
import numpy as np
import torch
from demanda.variantes_v4 import Secuencial,configuraciones

def test_diez_variantes_formas_y_gradientes():
    torch.set_num_threads(1)
    cfg=configuraciones();assert len(cfg['lstm'])==4 and len(cfg['cnn1d'])==6
    tr=SimpleNamespace(x=np.empty((2,168,1)),cal_p=np.empty((2,168,23)),cal_f=np.empty((2,24,23)))
    for nombre,cc in cfg.items():
        for c in cc:
            net=Secuencial(nombre,c['params'],tr)
            q=net(torch.randn(2,168,24),torch.randn(2,24,23))
            assert q.shape==(2,24) and torch.isfinite(q).all()
            q.square().mean().backward()
            assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in net.parameters())
            if nombre=='lstm':assert net.encoder.num_layers==c['params']['capas']
            else:assert [m.dilation[0] for m in net.encoder if isinstance(m,torch.nn.Conv1d)]==c['params']['dilataciones']

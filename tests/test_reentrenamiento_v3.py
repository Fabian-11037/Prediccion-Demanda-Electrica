"""Aislamiento del escalado y contratos de la reevaluación v3."""
import numpy as np
import pandas as pd
import torch
from demanda import modelos_base as M
from demanda.reentrenamiento_v3 import DatosFold,Entrenador
from demanda.particion import folds_forward_chaining

def test_es_y_validacion_no_cambian_escaladores_v3(panel,registro):
    f=folds_forward_chaining(panel.index,[2018])[0]
    cfg=M.ConfigDatos(ventana=72)
    d=DatosFold(panel,registro,cfg,f['tr'],True)
    futuro=panel.copy();corte=panel.index[f['es'][0]]
    cols=d.cols_x+d.cols_cal+['y_demanda_real_kwh']
    futuro.loc[futuro.index>=corte,cols]=999999.
    otro=DatosFold(futuro,registro,cfg,f['tr'],True)
    pd.testing.assert_series_equal(d.escala.media_,otro.escala.media_)
    pd.testing.assert_series_equal(d.escala.std_,otro.escala.std_)
    assert d.escaladores()['escala_mase']>0
    assert d.sigma_demanda==otro.sigma_demanda
    assert pd.Timestamp(d.auditoria['fin_ajuste'])<corte
    c=d.conjunto_pos(f['tr'][-16:]);assert abs(c.x[:,:,0].mean())<2
    assert np.allclose(d.invertir(d.objetivo_norm(c),c),c.y,rtol=1e-6)

def test_redes_v3_gradientes_y_checkpoint(panel,registro):
    f=folds_forward_chaining(panel.index,[2018],ventana=72)[0]
    d=DatosFold(panel,registro,M.ConfigDatos(ventana=72),f['tr'],True)
    tr=d.conjunto_pos(f['tr'][-16:]);es=d.conjunto_pos(f['es'][:8])
    p=dict(capas=1,unidades=16,filtros=8,kernel=3,dropout=.1,lr=.001,weight_decay=.001,perdida='huber',scheduler='plateau',batch_size=256)
    for nombre in ('lstm','cnn1d'):
        model=Entrenador(nombre);r=model.ajustar(tr,es,d,p,42,M.Opciones(max_epocas=2,dispositivo='cpu'))
        pred=model.predecir(es)
        assert pred.shape==es.y.shape and np.isfinite(pred).all()
        assert r['epocas']==2 and 1<=r['mejor_epoca']<=2
        assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in model.red.parameters())

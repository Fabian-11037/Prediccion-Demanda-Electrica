"""Contratos de datos ampliados, núcleo oficial y dimensionamiento."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from demanda.patchtst_numerico import Preparador,PatchTSTNumerico
from demanda.potencia_pareada import dimensionar
ROOT=Path(__file__).resolve().parents[1]
def fixture_numerico():
    c=json.loads((ROOT/'config/diseno_entrenamiento_texto.json').read_text());n=24*80;rng=np.random.default_rng(1)
    idx=pd.date_range('2017-01-01',periods=n,freq='h',tz='America/Bogota')
    cols=c['numericas']['historicas']+c['numericas']['calendario'];panel=pd.DataFrame(rng.normal(size=(n,len(cols))),index=idx,columns=cols)
    panel['y_demanda_real_kwh']=100+10*np.sin(np.arange(n)/24)+rng.normal(size=n)
    panel['x_demanda_real_kwh']=panel.y_demanda_real_kwh.shift(1)
    reg={k:dict(clase='past_only',rezago_h=1) for k in c['numericas']['historicas']}
    reg.update({k:dict(clase='known_future') for k in c['numericas']['calendario']})
    return c,panel,reg

def test_preparador_no_lee_es_y_conserva_mascaras():
    c,p,r=fixture_numerico();tr=np.arange(24*10,24*30,24);es=np.arange(24*40,24*43,24)
    pre=Preparador(p,r,c).ajustar(tr);alter=p.copy();alter.loc[alter.index>=p.index[es[0]]]=999999.
    other=Preparador(alter,r,c).ajustar(tr)
    assert pre.stats==other.stats
    q=pre.transformar(es);assert q.x.shape==(3,168,83) and q.calendario.shape==(3,24,23) and q.y.shape==(3,24)
    assert np.isfinite(q.x).all() and np.allclose(q.y[0],p.y_demanda_real_kwh.iloc[es[0]:es[0]+24])
    # Ausencias climáticas iniciales no causan descarte de orígenes.
    a=pre.transformar(np.array([24*8]));assert len(a.y)==1 and a.x[:,:,53:].sum()>0

def test_patchtst_forward_gradientes_y_uso_covariables():
    torch.set_num_threads(1);torch.manual_seed(42)
    model=PatchTSTNumerico(dict(dropout=.1),.1);x=torch.randn(2,168,83,requires_grad=True);c=torch.randn(2,24,23)
    y=model(x,c);assert y.shape==(2,24) and torch.isfinite(y).all()
    y.square().mean().backward();assert torch.isfinite(x.grad).all()
    assert x.grad[:,:,1:30].abs().sum()>0
    model.eval()
    with torch.no_grad():
        a=model(x,c);z=x.detach().clone();z[:,:,1]+=1;assert not torch.allclose(a,model(z,c))

def test_dimensionamiento_constante_y_determinista():
    losses=np.ones((5,60))*.5;ref=np.ones(60);years=np.repeat([2018,2019],30)
    a=dimensionar(losses,ref,years,replicas=200);b=dimensionar(losses,ref,years,replicas=200)
    assert a==b and a['n_final']==5 and a['potencia_limite_temporal']==1.

def test_presupuesto_temporal_detiene_antes_del_primer_lote():
    import pytest
    from demanda.patchtst_numerico import entrenar
    c,p,r=fixture_numerico();pre=Preparador(p,r,c).ajustar(np.arange(24*10,24*30,24))
    tr=pre.transformar(np.array([24*12,24*13]));es=pre.transformar(np.array([24*40,24*41]))
    params=dict(dropout=.1,lr=.001,weight_decay=.0001,batch_size=2,semilla=42)
    model=PatchTSTNumerico(params,pre.stats['sigma_demanda'])
    with pytest.raises(TimeoutError):entrenar(model,tr,es,params,max_epocas=1,device='cpu',deadline=0.)

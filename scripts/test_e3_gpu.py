"""Invariantes causales y de restauración del nuevo adaptador; no selecciona hiperparámetros."""
import numpy as np
import pandas as pd
import pytest
from e3_gpu_comun import lags_diarios,torch,ajustar,cargar,motor,verificar_json_portable
from e3_modelo1 import Router,complementariedad,dividir,controles
from corrector import modelo as MO,residuos as RS

def test_equivalencia_json_solo_finales_de_linea(tmp_path):
    import hashlib
    p=tmp_path/'registro.json';p.write_bytes(b'{\n"x": 1\n}\n')
    h=hashlib.sha256(p.read_bytes().replace(b'\n',b'\r\n')).hexdigest()
    assert 'CRLF' in verificar_json_portable(p,h)['equivalencia']
    p.write_bytes(b'{\n"x": 2\n}\n')
    with pytest.raises(ValueError):verificar_json_portable(p,h)

def test_errores_observables_solo_d_menos_2():
    dias=np.arange(np.datetime64('2020-01-01'),np.datetime64('2020-01-21'))
    e=np.arange(20*24*3,dtype='float32').reshape(20,24,3)
    a=lags_diarios(dias,e);otro=e.copy();otro[10:]=1e8;b=lags_diarios(dias,otro)
    assert np.array_equal(a[:12],b[:12])
    assert not np.array_equal(a[12],b[12])
    assert np.all(a[:2]==0)

def test_corrector_no_lee_error_actual_ni_siguiente():
    dias=pd.date_range('2020-01-01',periods=20);pred=np.full((20,24),100.,dtype='float32');y=pred+np.arange(20)[:,None]
    cr=dict(dia=dias.strftime('%Y-%m-%d').to_numpy(),anio=np.full(20,2020),y=y,yhat_cr=pred)
    cal=pd.DataFrame({'cal':np.arange(20,dtype='float32')},index=dias);cand=dict(dias=cr['dia'])
    a=MO.Datos(cr,cal,cand,1.).num;cr=dict(cr,y=y.copy());cr['y'][10:]=1e8
    b=MO.Datos(cr,cal,cand,1.).num
    assert np.array_equal(a[:12],b[:12])

def test_ridge_respeta_retraso_temporal():
    dias=pd.date_range('2020-01-01',periods=20);t=pd.DataFrame({'dia':dias});cal=pd.DataFrame({'cal':np.ones(20)},index=dias)
    r=np.arange(20,dtype=float);a=RS._rasgos_cr(t,cal,r);r[10:]=1e9;b=RS._rasgos_cr(t,cal,r)
    assert np.array_equal(a[:12],b[:12])

def test_complementariedad_de_signos():
    y=np.zeros((1,24),dtype='float32');p=np.tile(np.array([-2.,2.,3.]),(1,24,1))
    c=complementariedad(y,p,1.)
    assert np.all(c[:,:,0]==4.) and np.all(c[:,:,1]==4.) and np.all(c[:,:,2]==0.)

def test_router_convexo_y_gradientes():
    r=Router(6,dict(ancho=32,dropout=0.));x=torch.randn(4,6);pr=torch.randn(4,24,3)
    y,c,w=r(x,pr)
    assert torch.all(w>=0) and torch.allclose(w.sum(-1),torch.ones(4,24))
    assert torch.all(y>=pr.min(-1).values-1e-6) and torch.all(y<=pr.max(-1).values+1e-6)
    (y.square().mean()+c.square().mean()).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in r.parameters())

def test_particiones_meta_y_ajuste_final():
    dias=pd.date_range('2011-01-01','2021-12-31');m=dict(dia=dias.strftime('%Y-%m-%d').to_numpy(),anio=dias.year.to_numpy())
    for anio in [2018,2019,2020,2021,2022]:
        a,es,ev=dividir(m,anio)
        assert m['anio'][a].max()==anio-2 and m['anio'][es].min()==anio-1
        assert np.datetime64(m['dia'][a[-1]],'D')+np.timedelta64(2,'D')<=np.datetime64(m['dia'][es[0]],'D')
        assert (len(ev)==0)==(anio==2022)

def test_pesos_controles_restaurables(tmp_path):
    rng=np.random.default_rng(3);pr=rng.uniform(100,200,(15,24,3)).astype('float32');y=pr[:,:,0]*.7+pr[:,:,1]*.3
    m=dict(dia=pd.date_range('2020-01-01',periods=15).strftime('%Y-%m-%d').to_numpy(),pred=pr,y=y)
    controles(m,np.arange(5),np.arange(5,12),np.arange(12,15),tmp_path,100.)
    import json
    w=json.loads((tmp_path/'pesos.json').read_text());nn=np.array(w['NNLS']);cv=np.array(w['convexa'])
    assert nn.shape==(24,3) and cv.shape==(24,3) and np.allclose(cv.sum(1),1.)
    assert np.max(np.abs((pr[12:]*cv).sum(-1)-y[12:]))<.01

@pytest.mark.skipif(not torch.cuda.is_available(),reason='Requiere GPU real')
def test_checkpoint_gpu_y_reanudacion(tmp_path):
    motor.fijar_semilla(123);p=dict(ancho=32,dropout=.1,id=1,lambda_complementariedad=.01)
    r=Router(6,p).cuda();x=torch.randn(12,6,device='cuda');pr=torch.randn(12,24,3,device='cuda');y=pr.mean(-1)
    def loss(ix):return (r(x[ix],pr[ix])[0]-y[ix]).abs().mean()
    def ev():return (r(x,pr)[0]-y).abs().mean().item()
    cfg=dict(lr=.001,weight_decay=.0001,lote=4);op=dict(max_epocas=2,min_epocas=1,paciencia=2,min_delta=.0001)
    a=ajustar(r,12,loss,ev,cfg,op,123,tmp_path,lambda **k:None,'Prueba técnica GPU')
    ck=cargar(tmp_path/'reanudar.pt');assert 'optimizador' in ck and 'rng_cuda' in ck and ck['epoca']==2
    with torch.no_grad():antes=r(x,pr)[0].clone()
    b=ajustar(r,12,loss,ev,cfg,op,123,tmp_path,lambda **k:None,'Reanudar prueba técnica')
    with torch.no_grad():despues=r(x,pr)[0]
    assert torch.equal(antes,despues) and a['epocas']==b['epocas']==2

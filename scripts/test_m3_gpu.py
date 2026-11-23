"""Validaciones de M3 sin etiquetas de prueba ni entrenamiento de producción."""
import pytest
from e3_modelo3 import *


def entradas():
    motor.fijar_semilla(101);b=4;m=6
    num=torch.randn(b,12,device='cuda');expertos=torch.rand(b,24,3,device='cuda')*1e6+1e6
    logits=torch.randn(b,24,3,device='cuda');base=(logits.softmax(-1)*expertos).sum(-1)
    c=torch.zeros(b,24,3,device='cuda');emb=torch.randn(b,m,768,device='cuda')
    feat=torch.zeros(b,m,7,device='cuda');feat[:,:3,0]=1;feat[:,3:,1]=1
    f=torch.zeros(b,m,dtype=torch.long,device='cuda');mascara=torch.ones(b,m,dtype=torch.bool,device='cuda')
    return [num,expertos,logits,base,c,emb,feat,f,mascara]


@pytest.mark.parametrize('clase',[SelectorTexto,SelectorNumerico])
def test_gradientes_cotas_y_restauracion(clase,tmp_path):
    cfg=dict(id=1,d_texto=32,lambda_penalizacion_delta=.001,epsilon_logits=.25)
    red=clase(12,7,3,cfg).cuda();args=entradas()
    p,c,w,d=red(*args)
    assert torch.isfinite(p).all() and torch.allclose(w.sum(-1),torch.ones_like(p),atol=2e-7)
    assert d.abs().max()<=.25
    loss=p.mean()/215617.9+.1*c.square().mean();loss.backward()
    assert any(v.grad is not None and v.grad.abs().sum()>0 for v in red.parameters())
    assert all(v.grad is None or torch.isfinite(v.grad).all() for v in red.parameters())
    red.eval();path=tmp_path/'modelo.pt';checkpoint(path,dict(modelo=red.state_dict()))
    otra=clase(12,7,3,cfg).cuda().eval();otra.load_state_dict(cargar(path)['modelo'])
    with torch.no_grad():assert torch.equal(red(*args)[0],otra(*args)[0])


def test_identidad_sin_texto_y_aislamiento_de_canal():
    cfg=dict(d_texto=32,epsilon_logits=.5)
    red=SelectorTexto(12,7,3,cfg).cuda().eval();args=entradas()
    with torch.no_grad():
        nn.init.normal_(red.salida[-1].weight,.1,.1)
        args[-1][:]=False
        assert torch.equal(red(*args)[0],args[3])
        args[-1][:,:3]=True
        antes=red(*args)[0];args[5][:,3:]=99999.
        assert torch.equal(antes,red(*args)[0]),'Candidatos de prensa enmascarados afectan el canal oficial'


def test_seleccion_prefijo_no_depende_del_futuro():
    rej=[dict(id=1),dict(id=2)]
    filas=[dict(cfg=1,anio=2013,MASE=.7,n_parametros=10),dict(cfg=2,anio=2013,MASE=.8,n_parametros=20),
           dict(cfg=1,anio=2015,MASE=20,n_parametros=10),dict(cfg=2,anio=2015,MASE=0,n_parametros=20)]
    cfg,d=elegir_prefijo(filas,2014,rej)
    assert cfg['id']==1 and d['ultimo_anio_seleccion']==2013
    assert elegir_prefijo(filas,2013,rej)[0]['id']==1
    filas[2]['MASE']=-100
    assert elegir_prefijo(filas,2014,rej)[0]['id']==1


def test_fuentes_se_ajustan_solo_al_pasado():
    tx=Texto.__new__(Texto);tx.n_fuentes=3;tx.reservas=dict(alineados=[0,0],oficial=[0,1],prensa=[1,2])
    tx.mapa={'2015-01-01':0,'2016-01-01':1}
    feat=np.zeros((2,2,7),dtype='float32');feat[:,:,0]=1
    tx.k=dict(cand=np.array([[0,1],[2,3]]),feat=feat,fuente=np.array([[1,1],[2,3]]))
    mapa=tx.fuentes_ajuste(np.array(['2015-01-01']),'M3_OP')
    assert mapa.tolist()==[0,1,0,0]

"""Invariantes físicos, temporales y de restauración de M2; sin datos de prueba."""
from e3_modelo2 import *
import pytest


@pytest.mark.parametrize('modo',MODOS)
def test_conservacion_gradientes_y_restauracion(modo,tmp_path):
    assert torch.cuda.is_available()
    motor.fijar_semilla(101)
    cfg=dict(id=1,ancho=32,dropout=.1,lambda_energia=.1)
    red=EnergiaPerfil(12,cfg,2e8,modo).cuda()
    x=torch.randn(8,12,device='cuda');ancla=torch.rand(8,24,device='cuda')*1e7+1
    y,e,s=red(x,ancla)
    assert torch.isfinite(y).all() and (y>0).all()
    assert torch.allclose(y.sum(-1),e,rtol=3e-7)
    assert torch.allclose(s.sum(-1),torch.ones(8,device='cuda'),atol=2e-7)
    target=ancla*.8
    loss=(y-target).abs().mean()/215617.9;loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in red.parameters())
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in red.parameters())
    red.eval();p=tmp_path/'modelo.pt'
    checkpoint(p,dict(modelo=red.state_dict(),n_entrada=12,configuracion=cfg,escala_energia=2e8,modo=modo))
    otra=restaurar(p)
    with torch.no_grad():assert torch.equal(red(x,ancla)[0],otra(x,ancla)[0])
    if modo=='solo_distribucion':assert torch.allclose(e,ancla.sum(-1))
    if modo=='solo_energia':assert torch.allclose(s,ancla/ancla.sum(-1,keepdim=True))


def test_extraccion_congela_encoder_sin_cambiar_pronostico():
    cfg=dict(nucleo='patchtst',d_model=16,capas=1,cabezas=4,dropout=.1)
    red=redes.construir('tats',cfg,4,3,168,24).cuda().eval()
    tens=dict(x=torch.randn(3,168,4,device='cuda'),cal_p=torch.randn(3,168,3,device='cuda'),
              cal_f=torch.randn(3,24,3,device='cuda'),dia=torch.arange(3,device='cuda'))
    with torch.no_grad():antes=red(tens['x'],tens['cal_p'],tens['cal_f'],tens['dia']).cpu().numpy()
    estado,despues=contexto_congelado(red,tens)
    assert np.array_equal(antes,despues) and np.isfinite(estado).all()
    assert not any(p.requires_grad for p in red.parameters())


def test_rasgos_errores_causales_y_purga():
    dias=pd.date_range('2016-01-01','2021-12-31').strftime('%Y-%m-%d').to_numpy().astype('U10')
    n=len(dias);rng=np.random.default_rng(10)
    p=dict(dia=dias,y=rng.uniform(1e6,1e7,(n,24)).astype('float32'),pred=rng.uniform(1e6,1e7,(n,24)).astype('float32'),estado=np.zeros((n,5),dtype='float32'),nivel=np.ones(n,dtype='float32')*1e7)
    m=unir([p],215617.9);otro={**p,'y':p['y'].copy()}
    j=np.flatnonzero(dias=='2019-01-01')[0];otro['y'][j:]+=1e8
    mm=unir([otro],215617.9)
    assert np.array_equal(m['x'][:j+2],mm['x'][:j+2])
    assert not np.array_equal(m['x'][j+2:],mm['x'][j+2:])
    a,es,ev=dividir(m,2020)
    assert m['dia'][a[-1]]=='2018-12-30' and m['dia'][es[0]]=='2019-01-01'
    assert (m['anio'][a]<=2018).all() and (m['anio'][ev]==2020).all()

"""Causalidad, identidad sin vecinos y gradientes de M4; sin datos de prueba."""
import pytest
from types import SimpleNamespace
from e3_modelo4 import *


def crear():
    motor.fijar_semilla(101)
    cfg=dict(k_vecinos=8,edad_maxima_dias=365,ancho_compuerta=32)
    red=Memoria(12,7,3,cfg).cuda()
    q=torch.randn(20,32,device='cuda',requires_grad=True);k=torch.randn(20,32,device='cuda',requires_grad=True)
    dias=torch.arange(20,device='cuda')+15000
    r=torch.randn(20,24,device='cuda')*.2
    return red,q,k,dias,r


def test_identidad_sin_vecinos_y_exclusion_d1():
    red,q,k,dias,r=crear();red.eval();j=torch.tensor([0,1],device='cuda');base=torch.ones(2,24,device='cuda')*1e7
    p,g,ids,a,ok=red.recuperar(q,k,dias,r,j,base)
    assert torch.equal(p,base) and not ok.any() and not g.any()
    j=torch.tensor([2,9],device='cuda');base=torch.ones(2,24,device='cuda')*1e7
    p,g,ids,a,ok=red.recuperar(q,k,dias,r,j,base)
    assert ((~ok)|(dias[ids]<=dias[j,None]-2)).all()
    assert torch.allclose(a.sum(-1),torch.ones(2,device='cuda'))


def test_futuro_no_afecta_predicciones():
    red,q,k,dias,r=crear();red.eval();j=torch.tensor([5],device='cuda');base=torch.ones(1,24,device='cuda')*1e7
    with torch.no_grad():
        antes=red.recuperar(q,k,dias,r,j,base)[0]
        kk=k.clone();rr=r.clone();kk[4:]=1e8;rr[4:]=-1e7
        despues=red.recuperar(q,kk,dias,rr,j,base)[0]
        assert torch.equal(antes,despues)


def test_gradientes_finitos_incluso_sin_vecinos():
    red,q,k,dias,r=crear();j=torch.tensor([0,1,2,9],device='cuda');base=torch.ones(4,24,device='cuda')*1e7
    p=red.recuperar(q,k,dias,r,j,base)[0]
    loss=(p-base*.9).abs().mean()/215617.9;loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in red.parameters())
    assert torch.isfinite(q.grad).all() and torch.isfinite(k.grad).all()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in red.parameters())


def test_memoria_fija_y_restauracion(tmp_path):
    red,q,k,dias,r=crear();red.eval();j=torch.tensor([19],device='cuda');base=torch.ones(1,24,device='cuda')*1e7
    p,g,ids,a,ok=red.recuperar(q,k,dias,r,j,base,limite=15005)
    assert ((~ok)|(dias[ids]<=15005)).all()
    path=tmp_path/'modelo.pt';checkpoint(path,dict(modelo=red.state_dict(),n_num=12,n_feat=7,n_fuentes=3,configuracion=red.cfg,variante='M4_OP'))
    otra=construir(cargar(path));otra.load_state_dict(cargar(path)['modelo']);otra.eval()
    with torch.no_grad():assert torch.equal(p,otra.recuperar(q,k,dias,r,j,base,limite=15005)[0])


def test_codificacion_aprende_relevancia_sin_e5_entrenable():
    red,q,k,dias,r=crear()
    num=torch.randn(8,12,device='cuda');emb=torch.randn(8,6,768,device='cuda');feat=torch.zeros(8,6,7,device='cuda')
    feat[:,:3,0]=1;feat[:,3:,1]=1;fte=torch.zeros(8,6,dtype=torch.long,device='cuda');m=torch.ones(8,6,dtype=torch.bool,device='cuda')
    b=SimpleNamespace(num=num,texto_lote=lambda ix:(emb[ix],feat[ix],fte[ix],m[ix]))
    qq,kk=red.codificar(b,8);loss=(qq-kk).square().mean();loss.backward()
    assert not emb.requires_grad
    assert red.proy[0].weight.grad is not None and torch.isfinite(red.proy[0].weight.grad).all()
    assert torch.isfinite(qq).all() and torch.isfinite(kk).all()

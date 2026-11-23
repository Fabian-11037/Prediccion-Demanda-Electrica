"""Prueba integrada sintética GPU: restauración, causalidad y aislamiento del banco."""
from e3_gpu_comun import *
import argparse
import m4_cierre_procedencia as P

def ejecutar(car):
    if (car/'RESULTADO.json').exists():return
    assert torch.cuda.is_available();torch.set_num_threads(2);rng=np.random.default_rng(20261123);n=84
    base=np.full((n,24),20.,dtype='float32');y=base+(.3*np.sin(np.arange(n)[:,None]/7)+np.zeros((n,24))).astype('float32')
    meta=dict(dia=(np.datetime64('2019-01-01')+np.arange(n)).astype('U10'),anio=np.full(n,2019),num=rng.normal(size=(n,24)).astype('float32'),y=y,pred_base=base,expertos=np.repeat(base[:,:,None],3,2))
    a=np.arange(48);es=np.arange(49,68);ev=np.arange(69,84);rr=P.banco_residual(meta,base+.01,a,1.)
    P.ajustar(meta,rr,a,es,ev,101,car/'ajuste',1.,lambda **kw:None)
    ck=cargar(car/'ajuste/modelo.pt');antes=P.predecir(ck,meta,rr,ev)
    futuro={k:v.copy() for k,v in meta.items()};futuro['num'][78:]*=1e5;futuro['y'][78:]+=1e7;rr2=rr.copy();rr2[78:]+=1e7
    despues=P.predecir(ck,futuro,rr2,ev)
    assert np.array_equal(antes[ev<78],despues[ev<78]),'El futuro alteró pronósticos anteriores'
    assert np.isfinite(antes).all()
    with np.load(car/'ajuste/predicciones.npz') as z:assert np.array_equal(z['pred'],antes),'Restauración no idéntica'
    guardar(car/'RESULTADO.json',dict(aprobado=True,gpu=torch.cuda.get_device_name(0),datos='sintéticos; excluidos de resultados y conteos de semillas',restauracion_exacta=True,causalidad_prefijo=True,aislamiento_residuos=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);ejecutar(p.parse_args().out)

"""Controles deterministas sobre bases independientes; sin réplicas ficticias."""
from e3_gpu_comun import *
from scipy.optimize import nnls
from scipy.linalg import solve
from m4_cierre_controles import ewma

def ajustar(meta,a,es,car,q):
    if (car/'controles.npz').exists():return
    car.mkdir(parents=True,exist_ok=True);mu=meta['num'][a].mean(0);sd=meta['num'][a].std(0);sd=np.where(sd>1e-6,sd,1.)
    x=((meta['num'][a]-mu)/sd).astype('float64');r=((meta['y'][a]-meta['pred_base'][a])/q).astype('float64')
    # Intercepto explícito; lambda fija1, sin búsqueda por prueba.
    x=np.column_stack([x,np.ones(len(x))]);pen=np.eye(x.shape[1]);pen[-1,-1]=1e-8
    beta=solve(x.T@x+pen,x.T@r,assume_a='pos')
    pesos=np.array([nnls(meta['expertos'][es,h,:].astype('float64'),meta['y'][es,h].astype('float64'))[0] for h in range(24)])
    np.savez_compressed(car/'controles.npz',media=mu,escala=sd,beta_ridge=beta,NNLS=pesos)
    guardar(car/'contrato.json',dict(ridge_lambda=1,NNLS_ajuste='reserva interna anterior a prueba',EWMA_semivida_dias=30,banco_dias=365,retraso_dias=2,kNN=16,estocasticidad='Solo la de expertos/M1 independientes; controles condicionalmente deterministas'))

def predecir(meta,ix,car,q):
    with np.load(car/'controles.npz') as z:p={k:z[k] for k in z.files}
    x=((meta['num']-p['media'])/p['escala']).astype('float64');r=(meta['y']-meta['pred_base'])/q;dias=np.asarray(meta['dia'],dtype='datetime64[D]').astype('int64')
    ee,hay=ewma(meta,q);ridge=np.column_stack([x[ix],np.ones(len(ix))])@p['beta_ridge'];knn=[]
    for j in ix:
        ids=np.flatnonzero((dias[j]-dias>=2)&(dias[j]-dias<=365))
        if len(ids):
            dd=((x[ids]-x[j])**2).mean(1);ids=ids[np.argsort(dd,kind='stable')[:16]];knn.append(r[ids].mean(0))
        else:knn.append(np.zeros(24))
    return dict(NNLS=(meta['expertos'][ix]*p['NNLS']).sum(-1),EWMA_residual=meta['pred_base'][ix]+q*ee[ix],Ridge_residual=meta['pred_base'][ix]+q*np.where(hay[ix,None],ridge,0),kNN_residual=meta['pred_base'][ix]+q*np.asarray(knn))

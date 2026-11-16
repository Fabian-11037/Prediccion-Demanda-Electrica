"""Inferencia exploratoria sobre predicciones guardadas, sin entrenar modelos."""
from pathlib import Path
from itertools import combinations
import json,shutil,warnings
from seleccion_patchtst import RETIRAR,CONSERVAR,NOMBRE
import numpy as np
import pandas as pd
from scipy import stats,optimize,integrate,special
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.diagnostic import acorr_ljungbox,het_arch
from statsmodels.tsa.stattools import bds
import statsmodels.api as sm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R=next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/'data/processed/panel_horario.parquet').is_file())
O=R/'results/evaluacion_e2';O.mkdir(exist_ok=True)
C=O/'entradas'
Q=215617.90361589016;B=10000;SEED=20261115
TEXT_PAIRS=[('adj_modelo6_20261112_1035__N+O','adj_modelo6_20261112_1035__N'),
            ('adj_fusion_tardia_20261113_1728__N+O','adj_modelo6_20261112_1035__N'),
            ('adj_modelo7_20261114_0430__N+O','adj_modelo7_20261114_0430__N')]
def guardar(n,x):pd.DataFrame(x).to_csv(O/n,index=False,encoding='utf-8')
P={};nombres={};fuentes={}
df=pd.read_parquet(C/'consolidado/predicciones_finales.parquet')
Y=None
for (e,v),g in df.groupby(['experimento','variante'],sort=False):
    key=e+'__'+v
    if key==RETIRAR:continue
    arr=[]
    for s,gs in g.groupby('semilla',sort=True):
        gs=gs.sort_values(['origen','horizonte'])
        arr.append(gs.predicho_kwh.to_numpy().reshape(1461,24))
        y=gs.observado_kwh.to_numpy().reshape(1461,24)
        if Y is None:Y=y
        else:assert np.array_equal(y,Y)
    P[key]=np.stack(arr)
res=pd.read_csv(C/'consolidado/resumen_principal.csv')
for f in res.itertuples():
    key=f.experimento+'__'+f.variante
    if key==RETIRAR:continue
    nombres[key]=f.modelo+' · '+f.variante;fuentes[key]='numerico_ampliado'
nombres[CONSERVAR]=NOMBRE
snapshot=R/'results/repositorio/d1ae9a4e09dd1dead4af1329f16aa3e8d0f2c3d3'
reg=json.loads((snapshot/'results/consolidado/modelos.json').read_text())
for key in ['mlp','lstm','cnn1d','arx_lineal']:
    pp=[]
    for a in sorted(reg[key]['artefactos'],key=lambda a:a['semilla']):
        with np.load(snapshot/a['predicciones'],allow_pickle=False) as z:
            assert np.allclose(z['y'],Y,atol=1e-6)
            esperados=pd.date_range('2022-01-01','2025-12-31',freq='D',tz='America/Bogota').as_unit('ns').asi8
            assert np.array_equal(z['origen'],esperados)
            pp.append(z['yhat'])
    P[key]=np.stack(pp);nombres[key]={'mlp':'MLP','lstm':'LSTM','cnn1d':'CNN1D','arx_lineal':'ARX'}[key];fuentes[key]='demanda_calendario'
keys=list(P);K=len(keys);N=Y.shape[0]
origen=pd.date_range('2022-01-01','2025-12-31',freq='D',tz='America/Bogota')
panel=pd.read_parquet(R/'data/processed/panel_horario.parquet',columns=['y_demanda_real_kwh'])
pos=panel.index.get_indexer(origen);raw=panel.y_demanda_real_kwh.to_numpy()
horas=np.arange(24)
lag_diario=24*np.maximum(1,np.ceil((horas+1+1)/24).astype(int))
naives={'ingenuo_168h':raw[pos[:,None]+horas-168],
        'ingenuo_24h':raw[pos[:,None]+horas-lag_diario]}
assert abs(np.abs(naives['ingenuo_24h']-Y).mean()/Q-2.0767245486389916)<1e-10
for k,p in naives.items():P[k]=p[None];nombres[k]=k.replace('_',' ');fuentes[k]='referencia'
allkeys=list(P)
loss=np.column_stack([np.abs(P[k]-Y).mean(axis=(0,2))/Q for k in allkeys])
means=loss.mean(0)
seed_metrics=np.column_stack([np.abs(P[k]-Y).mean(axis=(1,2))/Q for k in keys])
assert seed_metrics.shape==(5,K)

def boot_medias(a,L,B=B,seed=SEED):
    """Bootstrap circular pareado: idénticos días remuestreados en todos los métodos."""
    n=len(a);rng=np.random.default_rng(seed);out=np.empty((B,a.shape[1]))
    for i in range(0,B,200):
        b=min(200,B-i)
        starts=rng.integers(n,size=(b,int(np.ceil(n/L))))
        ix=(starts[:,:,None]+np.arange(L))%n
        ix=ix.reshape(b,-1)[:,:n]
        out[i:i+b]=a[ix].mean(1)
    return out

def dm_hln(d,lag=13):
    """DM diario, pérdidas de horizontes no solapados: h efectivo=1 origen."""
    d=np.asarray(d,float);n=len(d);z=d-d.mean()
    lr=float(z@z/n)
    for j in range(1,lag+1):lr+=2*(1-j/(lag+1))*float(z[j:]@z[:-j]/n)
    se=np.sqrt(max(lr,1e-30)/n)
    t=float(d.mean()/se*np.sqrt((n-1)/n))
    return t,float(2*stats.t.sf(abs(t),df=n-1))

def ci_dz(d):
    n=len(d);t=d.mean()/(d.std(ddof=1)/np.sqrt(n))
    def cdf(nc):
        value=stats.nct.cdf(t,n-1,nc)
        if np.isfinite(value):return value
        # Definición de T no central como Z/sqrt(chi2/nu): evita NaN numérico.
        nu=n-1
        return integrate.quad(lambda x:special.ndtr(t*np.sqrt(x/nu)-nc)*np.exp(
            (nu/2-1)*np.log(max(x,1e-300))-x/2-(nu/2)*np.log(2)-special.gammaln(nu/2)),
            0,np.inf,epsabs=1e-9,limit=200)[0]
    def limite(prob):
        return optimize.brentq(lambda nc:cdf(nc)-prob,-1000,1000)/np.sqrt(n)
    return d.mean()/d.std(ddof=1),limite(.975),limite(.025)

print('Bootstrap de métricas',flush=True)
boot=boot_medias(loss,14)
np.savez_compressed(O/'perdidas_y_bootstrap.npz',loss=loss,bootstrap=boot,semillas=seed_metrics,
                    nombres=np.array(allkeys),origen=origen.as_unit('ns').asi8)
bench=[]
for i,k in enumerate(allkeys):
    abs_e=np.abs(P[k]-Y)
    rmse=np.sqrt(((P[k]-Y)**2).mean(axis=(1,2))).mean()
    bench.append(dict(clave=k,modelo=nombres[k],entradas=fuentes[k],semillas=len(P[k]),
        MASE=means[i],MASE_sd=abs_e.mean(axis=(1,2)).std(ddof=1)/Q if len(P[k])>1 else np.nan,
        IC95_inf=np.quantile(boot[:,i],.025),IC95_sup=np.quantile(boot[:,i],.975),
        MAE=means[i]*Q,RMSE=rmse,
        MASE_h1=abs_e[:,:,0].mean()/Q,MASE_h12=abs_e[:,:,11].mean()/Q,MASE_h24=abs_e[:,:,23].mean()/Q))
guardar('benchmark.csv',bench)
pares=[]
for i,j in combinations(range(len(allkeys)),2):
    d=loss[:,i]-loss[:,j];bd=boot[:,i]-boot[:,j]
    dm,pdm=dm_hln(d)
    centered=bd-d.mean()
    pb=(1+np.sum(np.abs(centered)>=abs(d.mean())))/(B+1)
    r=100*(means[i]-means[j])/means[j]
    rb=100*(boot[:,i]-boot[:,j])/boot[:,j]
    cr=rb-r
    p_lo=(1+np.sum(cr>=r+3))/(B+1)
    p_hi=(1+np.sum(cr<=r-3))/(B+1)
    pares.append(dict(A=allkeys[i],B=allkeys[j],diferencia_MASE=d.mean(),
        IC95_inf=np.quantile(bd,.025),IC95_sup=np.quantile(bd,.975),
        relativa_pct=r,IC95_rel_inf=np.quantile(rb,.025),IC95_rel_sup=np.quantile(rb,.975),
        DM_HLN=dm,p_DM=pdm,p_bootstrap=pb,
        p_TOST=max(p_lo,p_hi),IC90_rel_inf=np.quantile(rb,.05),IC90_rel_sup=np.quantile(rb,.95)))
pares=pd.DataFrame(pares)
for col in ['p_DM','p_bootstrap','p_TOST']:pares[col+'_Holm']=multipletests(pares[col],method='holm')[1]
guardar('comparaciones_temporales.csv',pares)

# Dos contrastes de contenido con dirección y denominador numérico explícitos.
texto=[]
for a,b in TEXT_PAIRS:
    i,j=allkeys.index(a),allkeys.index(b)
    delta=means[i]-means[j];bd=boot[:,i]-boot[:,j]
    r=100*delta/means[j];rb=100*bd/boot[:,j];cr=rb-r
    t,pdm=dm_hln(loss[:,i]-loss[:,j])
    pb=(1+np.sum(np.abs(bd-delta)>=abs(delta)))/(B+1)
    pt=max((1+np.sum(cr>=r+3))/(B+1),(1+np.sum(cr<=r-3))/(B+1))
    texto.append(dict(A=a,B=b,modelo=nombres[a],relativa_pct=r,
        IC95_inf=np.quantile(rb,.025),IC95_sup=np.quantile(rb,.975),
        p_DM=pdm,p_bootstrap=pb,p_TOST=pt,
        IC90_inf=np.quantile(rb,.05),IC90_sup=np.quantile(rb,.95)))
texto=pd.DataFrame(texto)
for col in ['p_DM','p_bootstrap','p_TOST']:texto[col+'_Holm_texto']=multipletests(texto[col],method='holm')[1]
guardar('contrastes_texto.csv',texto)

pp=[]
for i,j in combinations(range(K),2):
    d=seed_metrics[:,i]-seed_metrics[:,j]
    dz,lo,hi=ci_dz(d)
    pp.append(dict(A=keys[i],B=keys[j],diferencia_MASE=d.mean(),
        p_Shapiro=stats.shapiro(d).pvalue,p_t=stats.ttest_rel(seed_metrics[:,i],seed_metrics[:,j]).pvalue,
        p_Wilcoxon=stats.wilcoxon(d,method='exact').pvalue,Cohen_dz=dz,IC95_dz_inf=lo,IC95_dz_sup=hi))
pp=pd.DataFrame(pp)
for col in ['p_t','p_Wilcoxon']:pp[col+'_Holm']=multipletests(pp[col],method='holm')[1]
guardar('comparaciones_semillas.csv',pp)
fried=stats.friedmanchisquare(*[seed_metrics[:,j] for j in range(K)])
rr=np.apply_along_axis(stats.rankdata,1,seed_metrics)
rng_f=np.random.default_rng(SEED+99)
permutaciones=np.argsort(rng_f.random((B,5,K)),axis=2)
rp=np.take_along_axis(np.broadcast_to(rr,(B,5,K)),permutaciones,axis=2).mean(1)
q_perm=12*5/(K*(K+1))*((rp-(K+1)/2)**2).sum(1)
p_friedman_perm=float((1+np.sum(q_perm>=fried.statistic))/(B+1))
levene=stats.levene(*[seed_metrics[:,j] for j in range(K)],center='median')
ranks=np.apply_along_axis(stats.rankdata,1,seed_metrics).mean(0)
cd=stats.studentized_range.ppf(.95,K,np.inf)/np.sqrt(2)*np.sqrt(K*(K+1)/(6*5))
nem=[]
for i,j in combinations(range(K),2):
    q=abs(ranks[i]-ranks[j])/np.sqrt(K*(K+1)/(6*5))*np.sqrt(2)
    nem.append(dict(A=keys[i],B=keys[j],p_Nemenyi=stats.studentized_range.sf(q,K,np.inf)))
guardar('nemenyi.csv',nem)
guardar('rangos.csv',[dict(clave=k,modelo=nombres[k],rango=ranks[i]) for i,k in enumerate(keys)])

# Model Confidence Set, estadístico de rango studentizado y bootstrap circular.
activos=list(range(len(allkeys)));eliminados=[];p_acum=0.
while len(activos)>1:
    mm=means[activos];bb=boot[:,activos]
    dif=mm[:,None]-mm[None,:]
    bdiff=bb[:,:,None]-bb[:,None,:]
    se=bdiff.std(0,ddof=1);se[se<1e-12]=np.inf
    t=dif/se;obs=np.max(np.abs(t))
    bs=np.max(np.abs((bdiff-dif)/se),axis=(1,2))
    pv=float((1+np.sum(bs>=obs))/(B+1));p_acum=max(p_acum,pv)
    worst=int(np.argmax(t.max(1)));ix=activos[worst]
    eliminados.append(dict(clave=allkeys[ix],modelo=nombres[allkeys[ix]],paso=len(eliminados)+1,
                          p_crudo=pv,p_MCS=p_acum,retenido_90=p_acum>=.1))
    del activos[worst]
eliminados.append(dict(clave=allkeys[activos[0]],modelo=nombres[allkeys[activos[0]]],paso=len(eliminados)+1,p_crudo=1.,p_MCS=1.,retenido_90=True))
guardar('mcs.csv',eliminados)

diag=[];horizonte=[]
for k in keys:
    # Predicción media entre semillas sólo para diagnóstico, no reemplaza el benchmark.
    pred=P[k].mean(0);err=Y-pred;e=err.mean(1)
    lb=acorr_ljungbox(e,lags=[7,14,28],return_df=True)
    lbs=acorr_ljungbox((e-e.mean())**2,lags=[7,14,28],return_df=True)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',FutureWarning);arch=het_arch(e,nlags=14)
    jb=stats.jarque_bera(e)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore');bd,bdp=bds(e,max_dim=2,distance=1.5)
    X=sm.add_constant(pred.mean(1)/Q)
    fit=sm.OLS(Y.mean(1)/Q,X).fit(cov_type='HAC',cov_kwds={'maxlags':13})
    mz=fit.wald_test((np.eye(2),np.array([0.,1.])),scalar=True)
    diag.append(dict(clave=k,modelo=nombres[k],sesgo_kwh=e.mean(),p_LB7=lb.lb_pvalue.iloc[0],p_LB14=lb.lb_pvalue.iloc[1],p_LB28=lb.lb_pvalue.iloc[2],
        p_LB_cuadrados14=lbs.lb_pvalue.iloc[1],p_ARCH=arch[1],p_BDS=float(np.asarray(bdp)),
        p_JB=jb.pvalue,p_Mincer_Zarnowitz=float(mz.pvalue),intercepto=fit.params[0]*Q,pendiente=fit.params[1]))
    for h in range(24):
        lb1=acorr_ljungbox(err[:,h],lags=[14],return_df=True)
        horizonte.append(dict(clave=k,horizonte=h+1,MASE=np.abs(P[k][:,:,h]-Y[:,h]).mean()/Q,
                               sesgo_kwh=err[:,h].mean(),p_LB14=lb1.lb_pvalue.iloc[0]))
diag=pd.DataFrame(diag)
for col in [c for c in diag if c.startswith('p_')]:diag[col+'_Holm']=multipletests(diag[col],method='holm')[1]
guardar('diagnostico_residuos.csv',diag)
guardar('resultados_horizonte.csv',horizonte)
dmh=[]
for a,b in TEXT_PAIRS:
    for h in range(24):
        d=(np.abs(P[a][:,:,h]-Y[:,h]).mean(0)-np.abs(P[b][:,:,h]-Y[:,h]).mean(0))/Q
        t,p=dm_hln(d);dmh.append(dict(A=a,B=b,horizonte=h+1,DM_HLN=t,p=p))
dmh=pd.DataFrame(dmh);dmh['p_Holm']=multipletests(dmh.p,method='holm')[1];guardar('DM_por_horizonte_texto.csv',dmh)
sens=[]
for L in [7,14,28]:
    bs=boot if L==14 else boot_medias(loss,L,seed=SEED+L)
    for a,b in TEXT_PAIRS:
        i,j=allkeys.index(a),allkeys.index(b)
        delta=100*(bs[:,i]-bs[:,j])/bs[:,j]
        sens.append(dict(A=a,B=b,bloque=L,relativa_pct=100*(means[i]-means[j])/means[j],
                         IC95_inf=np.quantile(delta,.025),IC95_sup=np.quantile(delta,.975)))
guardar('sensibilidad_bloques_texto.csv',sens)

protocolo=dict(n_modelos=K,n_referencias=2,n_origenes=N,n_semillas=5,
    origen_unidad='ns desde epoch UTC',zona_presentacion='America/Bogota',
    comparaciones_temporales=len(pares),comparaciones_semillas=len(pp),
    B=B,bloque_dias=14,semilla=SEED,q168=Q,lag_HAC=13,h_efectivo_DM=1,
    friedman=float(fried.statistic),p_Friedman=float(fried.pvalue),
    p_Friedman_permutacion=p_friedman_perm,
    Levene=float(levene.statistic),p_Levene=float(levene.pvalue),CD_Nemenyi=float(cd),
    seleccion_tft='adjunto_20261115',alcance='exploratorio_retrospectivo',
    fuentes=nombres,entrenamiento_ejecutado=False,seleccion_patchtst=json.loads((R/'config/seleccion_patchtst.json').read_text(encoding='utf-8')),
    familias_Holm=f'cada tipo de contraste: {len(pares)} pares temporales, {len(pp)} de semillas; familia textual específica de {len(TEXT_PAIRS)} contrastes exploratorios',
    n_contrastes_textuales=len(TEXT_PAIRS),
    limitaciones=['Entradas, máscaras y presupuestos distintos: compara procedimientos, no efectos puros de arquitectura.',
                  'Semillas emparejadas por índice; no son cinco datasets independientes.',
                  'IC temporal condicionado a las semillas ejecutadas; no integra nuevas inicializaciones.',
                  'La pauta E2 no queda íntegramente satisfecha: faltan ARIMA/ETS, presupuesto homogéneo, potencia y búsqueda exigida.',
                  'Clark-West no se aplica a pérdidas MAE ni a redes no lineales sin supuestos de su formulación.',
                  'PIT y cobertura probabilística no se calculan con pronósticos puntuales.',
                  'El ingenuo diario respeta rezago de publicación de 1 h: usa 48 h para la última hora del horizonte.'])
(O/'protocolo.json').write_text(json.dumps(protocolo,ensure_ascii=False,indent=2),encoding='utf-8')
shutil.copy2(__file__,O/'codigo_ejecutado.py')
print(json.dumps({k:v for k,v in protocolo.items() if k not in ['fuentes','limitaciones']},ensure_ascii=False),flush=True)
print('Pruebas estadísticas guardadas',flush=True)

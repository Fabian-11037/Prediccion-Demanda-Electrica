"""Ablaciones y sensibilidad estadística de M4 desde predicciones conservadas."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from threadpoolctl import threadpool_limits
from evaluar_incertidumbre_comun import predictions
ROOT=Path(__file__).resolve().parents[1]
OUT='complementos_m4_20261125'
Q=215617.90361589016
B=9999
SEED=20261125
CONTROLS={
 'M1':('Base M1 sin memoria','Retira la corrección residual y conserva los tres expertos y su combinación.'),
 'M4_N_fija':('Banco sin actualización','Congela los episodios al cierre de 2021; conserva la edad máxima de 365 días.'),
 'M4_N_sin_compuerta':('Sin compuerta','Aplica completa la corrección; reajuste conservado de las proyecciones numéricas.'),
 'M4_N_uniforme':('Recientes con pesos iguales','Promedia los 16 episodios admisibles más recientes, sin proyección aprendida ni compuerta.'),
 'M4_N_similitud':('Distancia numérica sin aprendizaje','Recupera 16 contextos estandarizados por distancia L2 y pondera con softmax, sin proyección aprendida ni compuerta.'),
 'MLP_CE':('Control de capacidad igualada','MLP residual con el mismo contexto numérico y la misma base M1; 60.929 parámetros activos.')}

def sha(p):
 with Path(p).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,allow_nan=False,indent=2,default=lambda x:x.item())+'\n',encoding='utf-8')
def counts(n,seed=SEED):
 """Pesos de bootstrap estacionario por día; bloque esperado de 56 días."""
 rng=np.random.default_rng(seed);ix=np.empty((B,n),np.int32);ix[:,0]=rng.integers(n,size=B)
 for t in range(1,n):ix[:,t]=np.where(rng.random(B)<1/56,rng.integers(n,size=B),(ix[:,t-1]+1)%n)
 w=np.zeros((B,n),dtype=float)
 for start in range(0,B,128):
  z=ix[start:start+128];np.add.at(w[start:start+len(z)],(np.arange(len(z))[:,None],z),1/n)
 assert np.allclose(w.sum(1),1)
 return w

def temporal_tests(d,boot):
 """Mejora unilateral media; GW bilateral con diferencial observable d−2."""
 n=len(d);avg=float(d.mean());c=d-avg;lag=55
 gam=np.array([np.dot(c[h:],c[:n-h])/n for h in range(lag+1)])
 lrv=float(gam[0]+2*np.dot(1-np.arange(1,lag+1)/(lag+1),gam[1:]))
 assert lrv>0
 dm=avg/np.sqrt(lrv/n)*np.sqrt((n-1)/n)
 x=np.column_stack([np.ones(n),np.r_[np.zeros(2),d[:-2]]]);mom=x*d[:,None];mean=mom.mean(0);mc=mom-mean
 cov=mc.T@mc/n
 for h in range(1,lag+1):
  a=mc[h:].T@mc[:-h]/n;cov+=(1-h/(lag+1))*(a+a.T)
 gw=n*float(mean@np.linalg.pinv(cov)@mean)
 return [('bootstrap',avg,float((1+np.sum(boot-avg>=avg))/(B+1))),('DM_HLN',float(dm),float(stats.t.sf(dm,n-1))),('GW_bilateral',gw,float(stats.chi2.sf(gw,2)))]

def calculate(root=ROOT):
 root=Path(root);out=root/'results'/OUT;out.mkdir(parents=True,exist_ok=True)
 protocol=dict(fecha='2026-11-25',tipo='Análisis complementario de artefactos existentes',
  periodo=['2022-01-01','2025-12-31'],unidad='Origen diario con 24 errores; pérdidas individuales promedio entre semillas',
  Q_kWh=Q,replicas=B,bloque_medio_dias=56,HAC=55,semilla=SEED,margenes=[0,.03],
  escenarios=['todas','primeras_3','ultimas_3'],
  ablations=list(CONTROLS),familia_ablaciones=36,familia_sensibilidad=408,
  efecto='d_z temporal=media de L_ref−L_M4 / DE muestral diaria de ese diferencial. IC puntual mediante bloques56; no se supone independencia de días ni se emparejan identificadores de semillas.',
  RMSE='Promedio entre semillas de sqrt(promedio de errores cuadrados en todas las horas y días). El bootstrap recalcula esta operación no lineal en cada réplica.',
  tests='MASE: bootstrap unilateral, DM-HLN unilateral y GW bilateral. RMSE global: bootstrap unilateral de la métrica exacta. Holm36 para ablaciones; Holm408 para escenarios×márgenes×métricas/métodos.',
  selecciones='Primeras/últimas3 por orden de semilla; los ingenuos conservan su única realización. Sin selección por desempeño.',
  controles='Banco, compuerta y presencia de memoria: intervenciones específicas. Recientes uniformes y distancia L2 cambian recuperación y compuerta conjuntamente. No interpretarlos como retirada aislada de una capa.',
  actualizacion=False,entrenamiento=False,publicacion=False)
 p=out/'protocolo.json'
 if p.exists():assert json.loads(p.read_text(encoding='utf-8'))==protocol
 else:write(p,protocol)
 models,y,dates,master,sources=predictions(root);T=len(dates)
 run=root/'results/m4_cierre_gpu_20261123/reducido_B';control_arrays={};meta_rows=[]
 for key,(label,meaning)in CONTROLS.items():
  a=[];params=[]
  for seed in range(101,111):
   path=run/f'colombia/prueba/{key}_semilla_{seed}.npz';sources.append(path)
   with np.load(path,allow_pickle=False)as z:
    assert np.array_equal(z['dia'],dates.strftime('%Y-%m-%d'))and np.array_equal(z['y'],y)
    a.append(z['pred'].astype(float))
   if key!='M1':
    path=run/f'colombia/final/{key}/semilla_{seed}/TERMINADO.json';sources.append(path)
    ck=json.loads(path.read_text(encoding='utf-8'));params.append(ck['parametros_activos'])
  control_arrays[key]=np.stack(a)
  meta_rows.append(dict(clave=key,control=label,operacion=meaning,semillas=10,parametros_activos=params[0]if params else np.nan))
  assert not params or len(set(params))==1
 all_arrays={**models,**{'control_'+k:v for k,v in control_arrays.items()}}
 ma={k:np.abs(p-y).mean(2).T/Q for k,p in all_arrays.items()}
 sq={k:np.square((p-y)/1e6).mean(2).T for k,p in all_arrays.items()}
 for k in models:assert abs(ma[k].mean()-master.loc[k,'MASE_prueba'])<2e-6
 source_hash={p.relative_to(root).as_posix():sha(p)for p in sources}
 print('Datos alineados; calculando bootstrap de 56 días.',flush=True)
 with threadpool_limits(limits=4):
  w=counts(T)
  bm={k:w@v for k,v in ma.items()};bs={k:w@v for k,v in sq.items()}
  effects=[]
  primary=ma['m4_n'].mean(1);primary_boot=bm['m4_n'].mean(1)
  for key in master.index:
   if key=='m4_n':continue
   d=ma[key].mean(1)-primary;bd=bm[key].mean(1)-primary_boot
   sd=d.std(ddof=1);dz=d.mean()/sd
   bsd=np.sqrt(np.maximum((w@(d*d)-bd*bd)*T/(T-1),0));bdz=bd/bsd
   lo,hi=np.quantile(bdz,[.025,.975]);prob=(d>0).mean()+.5*(d==0).mean()
   effects.append(dict(clave=key,comparador=master.loc[key,'modelo'],d_z=dz,ic95_inf=lo,ic95_sup=hi,
    diferencia_MASE=d.mean(),DE_diferencial=sd,probabilidad_mejora_diaria=prob,supera_02=abs(dz)>=.2))
  pd.DataFrame(effects).to_csv(out/'efectos_M4.csv',index=False)
  ab_rows=[];ab_tests=[];seed_rows=[]
  for item in meta_rows:
   key=item['clave'];k='control_'+key;d=ma[k].mean(1)-primary;bd=bm[k].mean(1)-primary_boot
   gain=1-primary.mean()/ma[k].mean();ci=np.quantile(1-primary_boot/bm[k].mean(1),[.025,.975])
   mse=ma[k].mean(0)
   ab_rows.append(dict(**item,MASE=mse.mean(),DE_MASE=mse.std(ddof=1),RMSE_GWh=np.sqrt(sq[k].mean(0)).mean(),
    ganancia_M4_pct=100*gain,ic95_inf_pct=100*ci[0],ic95_sup_pct=100*ci[1],
    d_z=d.mean()/d.std(ddof=1)))
   for i,v in enumerate(mse):seed_rows.append(dict(clave=key,control=item['control'],semilla=101+i,MASE=v))
   for margin in [0,.03]:
    dd=(1-margin)*ma[k].mean(1)-primary;bb=(1-margin)*bm[k].mean(1)-primary_boot
    for method,stat,pv in temporal_tests(dd,bb):ab_tests.append(dict(clave=key,control=item['control'],margen=margin,metodo=method,estadistico=stat,p=pv))
  for i,v in enumerate(ma['m4_n'].mean(0)):seed_rows.append(dict(clave='m4_n',control='M4',semilla=101+i,MASE=v))
  ab=pd.DataFrame(ab_rows);at=pd.DataFrame(ab_tests);assert len(at)==36
  at['p_Holm']=multipletests(at.p,method='holm')[1];at['rechaza_005']=at.p_Holm<.05
  for margin,suffix in [(0,'0'),(.03,'3')]:
   g=at[at.margen.eq(margin)].groupby('clave').p_Holm.max()
   ab['p_max_Holm_'+suffix]=ab.clave.map(g)
   ab['respaldo_'+suffix]=(ab['p_max_Holm_'+suffix]<.05)&(ab.ganancia_M4_pct>100*margin)
  ab.to_csv(out/'ablaciones.csv',index=False);at.to_csv(out/'contrastes_ablaciones.csv',index=False)
  pd.DataFrame(seed_rows).to_csv(out/'ablaciones_semillas.csv',index=False)
  sensitivity=[];test_rows=[]
  slices={'todas':slice(None),'primeras_3':slice(0,3),'ultimas_3':slice(-3,None)}
  for scenario,part in slices.items():
   a=ma['m4_n'][:,part].mean(1);ba=bm['m4_n'][:,part].mean(1)
   rm=float(np.sqrt(sq['m4_n'][:,part].mean(0)).mean());brm=np.sqrt(bs['m4_n'][:,part]).mean(1)
   for key in master.index:
    if key=='m4_n':continue
    r=ma[key][:,part].mean(1);bref=bm[key][:,part].mean(1)
    rr=float(np.sqrt(sq[key][:,part].mean(0)).mean());brr=np.sqrt(bs[key][:,part]).mean(1)
    for metric,ref,mod,bbref,bbmod in [('MASE',float(r.mean()),float(a.mean()),bref,ba),('RMSE_GWh',rr,rm,brr,brm)]:
     gain=1-mod/ref;lo,hi=np.quantile(1-bbmod/bbref,[.025,.975])
     sensitivity.append(dict(escenario=scenario,metrica=metric,clave=key,comparador=master.loc[key,'modelo'],
      n_M4=ma['m4_n'][:,part].shape[1],n_ref=ma[key][:,part].shape[1],valor_M4=mod,valor_ref=ref,
      ganancia_pct=100*gain,ic95_inf_pct=100*lo,ic95_sup_pct=100*hi))
     for margin in [0,.03]:
      obs=(1-margin)*ref-mod;boot=(1-margin)*bbref-bbmod
      tests=temporal_tests((1-margin)*r-a,boot)if metric=='MASE'else[('bootstrap',obs,float((1+np.sum(boot-obs>=obs))/(B+1)))]
      for method,stat,pv in tests:test_rows.append(dict(escenario=scenario,metrica=metric,clave=key,comparador=master.loc[key,'modelo'],margen=margin,metodo=method,estadistico=stat,p=pv))
  se=pd.DataFrame(sensitivity);st=pd.DataFrame(test_rows);assert len(st)==408
  st['p_Holm']=multipletests(st.p,method='holm')[1];st['rechaza_005']=st.p_Holm<.05
  for margin,suffix in [(0,'0'),(.03,'3')]:
   groups=st[st.margen.eq(margin)].groupby(['escenario','metrica','clave']).p_Holm.max()
   se['p_max_Holm_'+suffix]=[groups.loc[(v.escenario,v.metrica,v.clave)]for v in se.itertuples()]
   se['respaldo_'+suffix]=(se['p_max_Holm_'+suffix]<.05)&(se.ganancia_pct>100*margin)
  se.to_csv(out/'sensibilidad.csv',index=False);st.to_csv(out/'contrastes_sensibilidad.csv',index=False)
  pd.DataFrame({'dia':dates.strftime('%Y-%m-%d'),**{k:v.mean(1)for k,v in ma.items()}}).to_csv(out/'perdidas_diarias.csv',index=False)
 summary=dict(ablaciones=len(ab),familia_ablaciones=len(at),familia_sensibilidad=len(st),
  d_z_TimesNet=next(e for e in effects if e['clave']=='nuevo_timesnet__N'),
  efectos_mayores_02=sum(e['supera_02']for e in effects),
  sensibilidad_respaldo_0=int(se.respaldo_0.sum()),sensibilidad_respaldo_3=int(se.respaldo_3.sum()),sensibilidad_combinaciones=len(se),
  ablaciones_respaldo_0=int(ab.respaldo_0.sum()),ablaciones_respaldo_3=int(ab.respaldo_3.sum()),
  parametros_M4=60993,parametros_control=60929,diferencia_capacidad_pct=100*(60929/60993-1),
  entrenamiento=False,publicacion=False)
 write(out/'resumen.json',summary)
 assert all(sha(root/p)==h for p,h in source_hash.items())
 write(out/'verificacion.json',dict(fechas_alineadas=True,paridad_MASE=True,fuentes_sin_cambios=True,fuentes_sha256=source_hash))
 print(json.dumps(summary,ensure_ascii=False,indent=2,default=lambda x:x.item()),flush=True)
if __name__=='__main__':calculate()

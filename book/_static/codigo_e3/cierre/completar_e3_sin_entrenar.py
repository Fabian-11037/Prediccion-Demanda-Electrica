"""Evaluación de artefactos congelados: estadística, propiedad y calibración temporal."""
from pathlib import Path
from types import SimpleNamespace
import ast,hashlib,json,sys,time
ROOT=Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT/'.bookenv/Lib/site-packages'))
import numpy as np
import pandas as pd
import torch
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.diagnostic import acorr_ljungbox,het_arch
from arch.bootstrap import MCS,StationaryBootstrap

OUT=ROOT/'results/cierre_e3_sin_entrenar_20261124'
RUN=ROOT/'results/m4_cierre_gpu_20261123/reducido_B'
Q=215617.90361589016
OUT.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(1)
torch.manual_seed(20261124)

def write(name,obj):
 (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def memory_class():
 p=ROOT/'scripts/documentacion_m4/pipeline_m4_n.py';tree=ast.parse(p.read_text(encoding='utf-8-sig'))
 node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Memoria')
 scope=dict(torch=torch,nn=torch.nn,np=np)
 exec(compile(ast.Module(body=[node],type_ignores=[]),str(p),'exec'),scope)
 return scope['Memoria']

def bank(meta,ck):
 return SimpleNamespace(num=torch.from_numpy((meta['num']-ck['media'])/ck['escala']),
  dias=torch.from_numpy(np.asarray(meta['dia'],dtype='datetime64[D]').astype('int64')),
  residuos=torch.from_numpy((meta['y']-meta['pred_base'])/Q),base=torch.from_numpy(meta['pred_base']),limite=None)

def load(seed):
 ck=torch.load(RUN/f'colombia/final/M4_N/semilla_{seed}/modelo.pt',map_location='cpu',weights_only=False)
 red=memory_class()(ck['n_num'],ck['n_feat'],ck['n_fuentes'],ck['configuracion'],'M4_N')
 red.load_state_dict(ck['modelo']);red.eval();red.requires_grad_(False)
 with np.load(RUN/f'colombia/base_oof_semilla_{seed}.npz',allow_pickle=False) as z:meta={k:z[k] for k in z.files}
 return red,ck,meta,bank(meta,ck)

def protocol():
 files=[RUN/f'colombia/prueba/M4_N_semilla_{s}.npz' for s in range(101,111)]
 files +=[RUN/f'colombia/final/M4_N/semilla_{s}/modelo.pt' for s in range(101,111)]
 files +=[ROOT/'results/evaluacion_e2/perdidas_y_bootstrap.npz']
 p=dict(fecha='2026-11-24',entrenamiento=False,semilla=20261124,block_days=14,
  bootstrap_reps=9999,mcs_reps=10000,mcs_size=.10,mcs_method='R',hac_lags=13,
  margins=[0,.03],calibration_year=2021,calibration_role='Reserva usada para selección de checkpoint; cobertura empírica, sin garantía split-conformal independiente',
  interval_levels=[.8,.9,.95],rolling_days=365,availability_lag_days=2,
  main_interval='Por horizonte, ventana móvil causal; predicción media de diez semillas',
  fixed_interval='Sensibilidad con cuantiles de 2021 fijos',
  correction='Contrastes históricos Top10: conservar familia1536; evaluación complementaria completa102: Holm separado',
  source_sha256={str(f.relative_to(ROOT)):sha(f) for f in files})
 if not (OUT/'protocolo.json').exists():write('protocolo.json',p)
 return p

def math_check():
 red,ck,meta,b=load(101)
 tests={};rng=np.random.default_rng(20261124)
 js=torch.tensor(np.linspace(370,len(meta['dia'])-1,128,dtype=int))
 with torch.no_grad():
  q,k=red.codificar(b,len(b.dias));pred,g,ids,a,ok=red(b,js,(q,k))
  rr=b.residuos[ids];mu=(a[:,:,None]*rr).sum(1)
  bound=Q*g[:,None]*rr.abs().amax(1)
  slack=bound-(pred-b.base[js]).abs()
  tests['cota_correccion']=bool((slack>=-2.).all())
  tests['compuerta_0_1']=bool(((g>=0)&(g<=1)).all())
  tests['atencion_simplex']=bool(torch.allclose(a.sum(-1),torch.ones(len(js)),atol=1e-6))
  age=b.dias[js,None]-b.dias[ids]
  tests['edad_2_365']=bool((~ok|((age>=2)&(age<=365))).all())
  empty=red(b,torch.tensor([0]),(q,k))
  tests['sin_vecinos_igual_base']=bool(torch.equal(empty[0],b.base[:1]) and empty[1].item()==0)
  # Una consulta no puede leer su etiqueta ni valores futuros, incluso si son enormes.
  j=torch.tensor([len(b.dias)-30]);old=b.residuos.clone();p0=red(b,j,(q,k))[0]
  b.residuos[j.item()-1:]=1e8
  tests['invarianza_valores_no_disponibles']=bool(torch.equal(p0,red(b,j,(q,k))[0]));b.residuos=old
  # Banco y vecinos fijos: perturbación solo de valores; la compuerta cambia por dispersión.
  perturb=torch.randn_like(rr)*.01;rp=rr+perturb;mup=(a[:,:,None]*rp).sum(1)
  tests['cota_convexidad_valores']=bool(((mup-mu).abs()<=perturb.abs().amax(1)+1e-6).all())
  # Cota local original: mismo banco y mismos vecinos; varían atención y compuerta.
  scores2=(q[js,None,:]*k[ids]).sum(-1)/(32**.5)+torch.randn_like(a)*.02
  a2=scores2.softmax(-1)*ok;a2=a2/a2.sum(-1,keepdim=True).clamp_min(1e-8)
  g2=(g+.03).clamp(0,1);mu2=(a2[:,:,None]*rr).sum(1)
  change=(Q*g2[:,None]*mu2-Q*g[:,None]*mu).abs().amax(-1)
  R=rr.abs().amax((1,2));limit=Q*R*((g2-g).abs()+(a2-a).abs().sum(-1))
  tests['cota_estabilidad_atencion_compuerta']=bool((change<=limit+2.).all())
  wq=red.proy_q[0].weight[:,:ck['n_num']].double();wk=red.proy_k[0].weight[:,:ck['n_num']].double()
  lq=float(torch.linalg.matrix_norm(wq,ord=2));lk=float(torch.linalg.matrix_norm(wk,ord=2))
 # Gradiente local de la atención y compuerta, con índices fijos y sin actualizar pesos.
 z=torch.randn(1,32,dtype=torch.float64,requires_grad=True);v=torch.randn(16,32,dtype=torch.float64)
 r=torch.randn(16,24,dtype=torch.float64);gate=red.compuerta.double()
 def f(x):
  s=x@v.T/(32**.5);a=s.softmax(-1);m=a@r
  disp=((r-m[:,None]).square()*a[:,:,None]).sum(1).mean(-1).sqrt()
  feat=torch.cat([x,s.max(-1).values[:,None],disp[:,None],x.new_ones((1,1)),x.new_ones((1,1))*.5],1)
  return torch.sigmoid(gate(feat))*m
 tests['gradcheck_atencion_compuerta']=bool(torch.autograd.gradcheck(f,(z,),eps=1e-6,atol=1e-4,rtol=1e-3))
 project=red.proy_q.double();inp=torch.randn(1,ck['n_num']+64,dtype=torch.float64,requires_grad=True)
 tests['gradcheck_proyeccion_LayerNorm']=bool(torch.autograd.gradcheck(project,(inp,),fast_mode=True))
 assert all(tests.values()),tests
 result=dict(tests=tests,n_queries=len(js),max_violation_kwh=float((-slack).clamp_min(0).max()),
  minimum_slack_kwh=float(slack.min()),query_weight_spectral_norm=lq,key_weight_spectral_norm=lk,
  LN_global_bound_query=lq*float(red.proy_q[1].weight.abs().max())/(red.proy_q[1].eps**.5),
  scope='Banco y vecindario fijos; top-k no se supone continuo globalmente; gradientes comprobados sin optimización',entrenamiento=False)
 write('propiedad.json',result);print('Propiedad y gradientes:',tests,flush=True)

def statistical():
 study=json.loads((ROOT/'dashboard/site/data/study.json').read_text(encoding='utf-8'))
 benchmark=pd.read_csv(ROOT/'results/evaluacion_e2/benchmark.csv').set_index('clave')
 z=np.load(ROOT/'results/evaluacion_e2/perdidas_y_bootstrap.npz',allow_pickle=False)
 loss=pd.DataFrame(z['loss'],columns=z['nombres'])
 # El archivo conserva nombres de clave o etiquetas; resolver de forma comprobada.
 mapping={r['modelo']:k for k,r in benchmark.iterrows()}
 loss=loss.rename(columns=mapping)
 assert set(loss.columns)==set(benchmark.index),(loss.columns.tolist(),benchmark.index.tolist())
 preds=[]
 for seed in range(101,111):
  with np.load(RUN/f'colombia/prueba/M4_N_semilla_{seed}.npz',allow_pickle=False) as a:
   preds.append(a['pred'].astype(float));y=a['y'];dates=a['dia']
 loss['m4_n']=np.abs(np.stack(preds)-y).mean((0,2))/Q
 # Verificar que se comparan pérdidas individuales y no el error del ensemble.
 assert abs(loss.m4_n.mean()-.5770369087419158)<1e-8
 for k in benchmark.index:assert abs(loss[k].mean()-benchmark.loc[k,'MASE'])<2e-6,k
 loss.to_csv(OUT/'perdidas_diarias.csv',index=False)
 top={m['id'] for m in study['models'] if m['group']=='Top 10'}
 extra=list(benchmark.index)
 rng=np.random.default_rng(20261124);N=len(loss);B=9999
 # Bootstrap estacionario común a todas las parejas.
 index=np.empty((B,N),dtype=np.int32);index[:,0]=rng.integers(N,size=B)
 for t in range(1,N):
  reset=rng.random(B)<1/14
  index[:,t]=np.where(reset,rng.integers(N,size=B),(index[:,t-1]+1)%N)
 rows=[];diagnostics=[]
 for key in extra:
  ref=loss[key].to_numpy();model=loss.m4_n.to_numpy();gain=1-model.mean()/ref.mean()
  br=ref[index].mean(1);bm=model[index].mean(1);ci=np.quantile(1-bm/br,[.025,.975])
  for margin in [0.,.03]:
   d=(1-margin)*ref-model;dm=d.mean();c=d-dm
   gamma=np.array([np.dot(c[h:],c[:N-h])/N for h in range(14)])
   lrv=max(gamma[0]+2*np.dot(1-np.arange(1,14)/14,gamma[1:]),1e-15)
   stat=dm/np.sqrt(lrv/N)*np.sqrt((N-1)/N)
   p_dm=float(stats.t.sf(stat,N-1))
   centered=((1-margin)*br-bm)-dm
   p_boot=(1+np.sum(centered>=dm))/(B+1)
   # GW condicional: constante y diferencial conocido d-2, HAC13.
   lag=np.r_[np.zeros(2),d[:-2]];X=np.column_stack([np.ones(N),lag])
   moments=X*d[:,None];avg=moments.mean(0);mc=moments-avg
   cov=mc.T@mc/N
   for h in range(1,14):
    c_h=mc[h:].T@mc[:-h]/N;cov+=(1-h/14)*(c_h+c_h.T)
   gw=N*float(avg@np.linalg.pinv(cov)@avg);p_gw=float(stats.chi2.sf(gw,2))
   for method,statistic,p,null in [('bootstrap_estacionario',float(dm),float(p_boot),'E[(1-m)L_ref-L_M4] <= 0'),('DM_HLN',float(stat),p_dm,'E[(1-m)L_ref-L_M4] <= 0'),('GW_bilateral',gw,p_gw,'E[z_t((1-m)L_ref-L_M4)] = 0')]:
    rows.append(dict(comparator_id=key,Comparador=benchmark.loc[key,'modelo'],metodo=method,margen_relativo=margin,
     estadistico=statistic,p=p,ganancia_relativa=gain,ic95_inf=float(ci[0]),ic95_sup=float(ci[1]),hipotesis_nula=null,
     familia='complementaria_102',alcance='Temporal condicionado a pesos y semillas conservadas',n=N))
  d=ref-model;c=d-d.mean()
  lb=acorr_ljungbox(d,lags=[7,14,28],return_df=True)
  diagnostics.append(dict(comparator_id=key,Comparador=benchmark.loc[key,'modelo'],media_d=float(d.mean()),
   LjungBox7_p=float(lb.lb_pvalue.iloc[0]),LjungBox14_p=float(lb.lb_pvalue.iloc[1]),LjungBox28_p=float(lb.lb_pvalue.iloc[2]),
   ARCH14_p=float(het_arch(c,nlags=14)[1]),JB_p=float(stats.jarque_bera(d).pvalue),
   dependencia='HAC y bootstrap por bloques; normalidad del diferencial no exigida'))
 df=pd.DataFrame(rows);assert len(df)==102
 df['p_ajustado']=multipletests(df.p,method='holm')[1]
 df.to_csv(OUT/'contrastes_completos.csv',index=False)
 # Diagnósticos de las diez parejas históricas, sin cambiar sus valores p.
 for key in sorted(top-set(extra)):
  d=loss[key].to_numpy()-loss.m4_n.to_numpy();lb=acorr_ljungbox(d,lags=[7,14,28],return_df=True)
  diagnostics.append(dict(comparator_id=key,Comparador=benchmark.loc[key,'modelo'],media_d=float(d.mean()),
   LjungBox7_p=float(lb.lb_pvalue.iloc[0]),LjungBox14_p=float(lb.lb_pvalue.iloc[1]),LjungBox28_p=float(lb.lb_pvalue.iloc[2]),
   ARCH14_p=float(het_arch(d-d.mean(),nlags=14)[1]),JB_p=float(stats.jarque_bera(d).pvalue),
   dependencia='HAC y bloques; no independencia diaria ni normalidad impuestas'))
 diag=pd.DataFrame(diagnostics)
 for col in ['LjungBox7_p','LjungBox14_p','LjungBox28_p','ARCH14_p','JB_p']:diag[col+'_Holm']=multipletests(diag[col],method='holm')[1]
 diag.to_csv(OUT/'diagnosticos_parejas.csv',index=False)
 print('Calculando MCS conjunto de 18 procedimientos...',flush=True)
 mcs=MCS(loss,size=.10,reps=10000,block_size=14,method='R',bootstrap='stationary',seed=20261124);mcs.compute()
 result=mcs.pvalues.copy();result['retenido_90']=result.index.isin(mcs.included)
 result.to_csv(OUT/'mcs_integrado.csv',index_label='clave')
 write('estadistica.json',dict(n_modelos=len(loss.columns),n_dias=N,contrastes_nuevos=len(df),
  mcs_retenidos=list(mcs.included),n_rechazos_ajustados=int((df.p_ajustado<.05).sum()),
  familia_historica_preservada=1536,familia_adicional=len(df),bootstrap_resolution=1/(B+1),entrenamiento=False))
 print('Estadística terminada. MCS:',mcs.included,flush=True)

def calibration():
 cached=OUT/'predicciones_calibracion.npz'
 if not cached.exists():
  outputs=[]
  with torch.inference_mode():
   for seed in range(101,111):
    red,ck,meta,b=load(seed);ix=np.flatnonzero(np.array(meta['dia'])>='2021-01-01')
    rep=red.codificar(b,len(meta['dia']));pred=[]
    for start in range(0,len(ix),64):pred.append(red(b,torch.tensor(ix[start:start+64]),rep)[0].numpy())
    outputs.append(np.concatenate(pred));cy=meta['y'][ix];cd=meta['dia'][ix]
    print('Calibración congelada 2021, semilla',seed,flush=True)
  np.savez_compressed(cached,pred=np.stack(outputs),y=cy,dia=cd)
 with np.load(cached,allow_pickle=False) as z:cal_pred=z['pred'].mean(0).astype(float);cy=z['y'].astype(float);cd=z['dia']
 test=[]
 for seed in range(101,111):
  with np.load(RUN/f'colombia/prueba/M4_N_semilla_{seed}.npz',allow_pickle=False) as z:test.append(z['pred'].astype(float));y=z['y'];dates=z['dia']
 center=np.mean(test,0);scores=np.abs(cy-cal_pred);res=np.abs(y-center)
 all_scores=np.concatenate([scores,res]);all_dates=np.asarray(np.r_[cd,dates],dtype='datetime64[D]')
 rows=[];hourly=[];bounds={}
 for level in [.8,.9,.95]:
  alpha=1-level
  for method in ['fijo_2021','movil_causal_365','movil_simultaneo24']:
   widths=[]
   for i,date in enumerate(np.asarray(dates,dtype='datetime64[D]')):
    eligible=(all_dates<=date-np.timedelta64(2,'D'))&(all_dates>=date-np.timedelta64(366,'D')) if method!='fijo_2021' else all_dates<np.datetime64('2022-01-01')
    sample=all_scores[eligible];n=len(sample);rank=min(int(np.ceil((n+1)*level)),n)
    widths.append(np.repeat(np.partition(sample.max(1),rank-1)[rank-1],24) if method=='movil_simultaneo24' else np.partition(sample,rank-1,axis=0)[rank-1])
   radius=np.asarray(widths);lo=center-radius;hi=center+radius;covered=(y>=lo)&(y<=hi)
   interval_score=2*radius+2/alpha*np.maximum(lo-y,0)+2/alpha*np.maximum(y-hi,0)
   bounds[f'{method}_{int(level*100)}']=radius
   for year in ['total',2022,2023,2024,2025]:
    sel=np.ones(len(dates),dtype=bool) if year=='total' else np.char.startswith(dates,str(year))
    rows.append(dict(metodo=method,nivel=level,segmento=str(year),n_dias=int(sel.sum()),cobertura=float(covered[sel].mean()),
     cobertura_trayectoria24=float(covered[sel].all(1).mean()),ancho_GWh=float((2*radius[sel]).mean()/1e6),interval_score_GWh=float(interval_score[sel].mean()/1e6)))
   for h in range(24):hourly.append(dict(metodo=method,nivel=level,horizonte=h+1,cobertura=float(covered[:,h].mean()),ancho_GWh=float((2*radius[:,h]).mean()/1e6)))
 np.savez_compressed(OUT/'intervalos.npz',**bounds,center=center,y=y,dia=dates)
 pd.DataFrame(rows).to_csv(OUT/'cobertura_intervalos.csv',index=False)
 pd.DataFrame(hourly).to_csv(OUT/'cobertura_horizonte.csv',index=False)
 write('calibracion.json',dict(n_calibracion=len(cy),inicio=str(cd[0]),fin=str(cd[-1]),
  rol='2021 también intervino en selección del checkpoint; evaluación empírica temporal, sin garantía de cobertura independiente',
  main='movil_causal_365',lag_dias=2,actualiza_pesos=False,point='Media de diez predicciones; distinto del promedio de pérdidas individuales del benchmark',
  fixed_widths_90_gwh=(np.partition(scores,min(int(np.ceil((len(scores)+1)*.9)),len(scores))-1,axis=0)[min(int(np.ceil((len(scores)+1)*.9)),len(scores))-1]/1e6).tolist()))
 print('Intervalos calculados sin ajustar pesos.',flush=True)

if __name__=='__main__':
 stage=sys.argv[1] if len(sys.argv)>1 else 'all';protocol()
 if stage in ['all','math']:math_check()
 if stage in ['all','stats']:statistical()
 if stage in ['all','calibration']:calibration()

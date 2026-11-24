"""Importancia, estabilidad y coste de M4 y TimesNet. No contiene optimizadores."""
from pathlib import Path
from types import SimpleNamespace
import sys,json,time,platform,copy,tracemalloc
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'dashboard'),str(ROOT/'scripts')]
from completar_e3_sin_entrenar import OUT,Q,load,sha,write
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from runtime_inferencia import get_runtime

def parameter_groups():
 return {'demanda':[0],'precio':[1],'hidrologia':list(range(2,6)),
  'temperatura':list(range(6,14)),'humedad':list(range(14,22)),'precipitacion':list(range(22,30))}

def raw_importance():
 runtime=get_runtime();I=runtime.inputs;rng=np.random.default_rng(20261124)
 dates=[f'{y}-{m:02d}-15' for y in range(2022,2026) for m in range(1,13)]
 rows=[];measures=[];models=['m4_n','nuevo_timesnet__N']
 with torch.inference_mode():
  for key in models:
   for number,date in enumerate(dates):
    I.configure(date);original=I.window.copy();cp=I.cp.copy();cf=I.cf.copy()
    base,_,_=runtime.forward(key)
    origin=pd.Timestamp(date,tz='America/Bogota');target=I.index.get_indexer(pd.date_range(origin,periods=24,freq='h'))
    # El objetivo se usa exclusivamente para puntuar, después de la inferencia.
    y=pd.read_parquet(ROOT/'data/processed/panel_horario.parquet',columns=['y_demanda_real_kwh']).iloc[target,0].to_numpy()
    error=np.abs(base-y).mean()/Q
    for repetition in range(3):
     # Donante pasado, igual día de semana; conserva trayectoria y máscaras del bloque.
     donor=origin-pd.Timedelta(days=int(rng.choice([14,21,28,35,42,49,56])))
     donor_window=I.raw.reindex(pd.date_range(donor-pd.Timedelta(hours=168),periods=168,freq='h')).to_numpy(copy=True)
     for group,cols in parameter_groups().items():
      I.window=original.copy();I.window[:,cols]=donor_window[:,cols]
      pred,_,_=runtime.forward(key)
      rows.append(dict(modelo=key,origen=date,repeticion=repetition,grupo=group,MASE_original=float(error),
       delta_MASE=float(np.abs(pred-y).mean()/Q-error),cambio_pred_GWh=float(np.abs(pred-base).mean()/1e6)))
     I.window=original;I.cp=cp;I.cf=cf
    if number%12==0:print('Importancia',key,date,flush=True)
   # Mismo origen, calentamiento, un hilo, incluye las tres redes y la memoria para M4.
   I.configure('2023-07-01')
   for _ in range(5):runtime.forward(key)
   times=[];tracemalloc.start()
   for _ in range(50):
    start=time.perf_counter();runtime.forward(key);times.append(1000*(time.perf_counter()-start))
   _,peak_python=tracemalloc.get_traced_memory();tracemalloc.stop()
   with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],profile_memory=True,with_flops=True) as prof:runtime.forward(key)
   events=prof.key_averages();flops=sum(e.flops for e in events)
   peak_positive=sum(max(0,e.self_cpu_memory_usage) for e in events)
   if key=='m4_n':
    modules=[r for r,_ in runtime.experts]+[runtime.router,runtime.memory]
   else:modules=[runtime.additional(key)[0]]
   measures.append(dict(modelo=key,semilla=101 if key=='m4_n' else 1000,n_mediciones=50,
    mediana_ms=float(np.median(times)),p95_ms=float(np.quantile(times,.95)),
    parametros_almacenados=sum(p.numel() for r in modules for p in r.parameters()),
    FLOPs_operadores_cubiertos=int(flops),asignaciones_positivas_torch_bytes=int(peak_positive),
    pico_python_bytes=peak_python,alcance='Forward completo con entradas preparadas; M4 incluye expertos, M1 y búsqueda; FLOPs parciales de operadores soportados por profiler'))
 pd.DataFrame(rows).to_csv(OUT/'importancia_numerica_diaria.csv',index=False)
 summary=pd.DataFrame(rows).groupby(['modelo','grupo']).agg(delta_MASE=('delta_MASE','mean'),DE=('delta_MASE','std'),cambio_pred_GWh=('cambio_pred_GWh','mean')).reset_index()
 summary.to_csv(OUT/'importancia_numerica.csv',index=False)
 pd.DataFrame(measures).to_csv(OUT/'eficiencia.csv',index=False)
 write('equipo.json',dict(system=platform.platform(),processor=platform.processor(),python=sys.version,torch=torch.__version__,threads=torch.get_num_threads(),device='CPU'))

def stability():
 groups={'resumen_variables':slice(0,120),'calendario':slice(120,672),'pronosticos_expertos':slice(672,744),
  'dispersion_expertos':slice(744,768),'errores_recientes':slice(768,912)}
 rows=[];deletion=[];attention={};rng=np.random.default_rng(20261124)
 with torch.inference_mode():
  for seed in range(101,111):
   red,ck,meta,b=load(seed);ix=np.flatnonzero(meta['dia']>='2021-01-01')[::30]
   q,k=red.codificar(b,len(b.dias));p,g,ids,a,ok=red(b,torch.tensor(ix),(q,k))
   attention[seed]=[]
   for pos,j in enumerate(ix):
    dist=np.zeros(365)
    ages=(b.dias[j]-b.dias[ids[pos]]).numpy()
    for age,weight,valid in zip(ages,a[pos].numpy(),ok[pos].numpy()):
     if valid:dist[int(age)-1]=weight
    attention[seed].append(dist)
   for name,columns in groups.items():
    changed=q.clone();z=b.num[torch.tensor(ix)].clone()
    # Cambiar solo la consulta por un episodio anterior, manteniendo banco y pesos.
    donor=torch.tensor(ix-28);z[:,columns]=b.num[donor,columns]
    augmented=torch.cat([z,z.new_zeros((len(z),64))],1);changed[torch.tensor(ix)]=red.proy_q(augmented)
    alt=red(b,torch.tensor(ix),(changed,k))[0]
    for pos,j in enumerate(ix):rows.append(dict(semilla=seed,grupo=name,origen=str(meta['dia'][j]),
     cambio_GWh=float((alt[pos]-p[pos]).abs().mean()/1e6)))
   for pos,j in enumerate(ix):
    original=b.residuos.clone();eligible=torch.nonzero((b.dias[j]-b.dias>=2)&(b.dias[j]-b.dias<=365)).flatten().numpy()
    for count in [1,4,8,16]:
     top=ids[pos][ok[pos]][:count]
     for condition in ['atencion','aleatorio']:
      b.residuos=original.clone();selected=top if condition=='atencion' else torch.from_numpy(rng.choice(eligible,len(top),replace=False))
      b.residuos[selected]=0.;out=red(b,torch.tensor([j]),(q,k))[0][0]
      deletion.append(dict(semilla=seed,origen=str(meta['dia'][j]),n=count,seleccion=condition,cambio_GWh=float((out-p[pos]).abs().mean()/1e6)))
    b.residuos=original
   print('Estabilidad y fidelidad, semilla',seed,flush=True)
 pairs=[]
 for a in range(101,111):
  for b in range(a+1,111):
   for i in range(len(attention[a])):
    rho=spearmanr(attention[a][i],attention[b][i]).statistic
    pairs.append(dict(semilla_A=a,semilla_B=b,consulta=i,rho_atencion=float(rho)))
 pd.DataFrame(rows).to_csv(OUT/'estabilidad_importancia.csv',index=False)
 pd.DataFrame(deletion).to_csv(OUT/'fidelidad_atencion.csv',index=False)
 pd.DataFrame(pairs).to_csv(OUT/'estabilidad_atencion.csv',index=False)
 imp=pd.DataFrame(rows).groupby(['semilla','grupo']).cambio_GWh.mean().unstack()
 ir=[]
 for a in imp.index:
  for b in imp.index:
   if a<b:ir.append(float(spearmanr(imp.loc[a],imp.loc[b]).statistic))
 write('interpretabilidad.json',dict(rho_atencion_mediana=float(pd.DataFrame(pairs).rho_atencion.median()),
  rho_importancia_mediana=float(np.median(ir)),n_semillas=10,n_consultas_por_semilla=len(ix),
  periodo='2021, reserva interna de selección; explicaciones descriptivas',
  nota='Atención mide uso de memoria; eliminación verifica efecto sobre la salida, no causalidad ni superioridad predictiva'))

if __name__=='__main__':
 stage=sys.argv[1] if len(sys.argv)>1 else 'all'
 if stage in ['all','importance']:raw_importance()
 if stage in ['all','stability']:stability()

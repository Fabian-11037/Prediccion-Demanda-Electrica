"""Diagnósticos con pesos congelados: sin entrenamiento ni selección."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from zipfile import ZipFile
import sys,json,io,hashlib,time,platform
ROOT=Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT/'.bookenv/Lib/site-packages'))
BUNDLE=ROOT/'results/analisis_nbeats_timesnet_20261120/fuentes_paquete/NBEATS_TimesNet_E2'
sys.path.insert(0,str(BUNDLE/'src'))
import torch,numpy as np,pandas as pd
from entrenamiento.datos import DatosNumericos
from entrenamiento import redes_e2
from entrenamiento.protocolo_e2 import _escalas
OUT=ROOT/'results/integracion_top10_20261120';OUT.mkdir(exist_ok=True)
SOURCE=ROOT/'results/analisis_nbeats_timesnet_20261120/entrada_resultados/resultados_e2_nbeats_timesnet'
WEIGHTS=ROOT/'results/integracion_top10_20261120/pesos/ENVIAR_pesos_NBEATS_TimesNet_20261119_2125.zip'
GROUPS=['perfil_demanda','nivel_ventana','precio','reservas','aportes','temperatura','humedad','precipitacion','calendario']
Q=215617.90361589016

def load(family,seed):
 folder=SOURCE/f'nuevo_{family}/final/N/semilla_{seed}'
 meta=json.loads((folder/'resultado.json').read_text(encoding='utf8'));esc=json.loads((folder/'escaladores.json').read_text(encoding='utf8'))
 with ZipFile(WEIGHTS) as z:raw=z.read(f'resultados_e2_nbeats_timesnet/nuevo_{family}/final/N/semilla_{seed}/modelo.pt')
 model=redes_e2.construir(family,meta['configuracion'],33,23,168,24,None,None)
 model.load_state_dict(torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True),strict=True)
 model.eval();model.requires_grad_(False)
 return model,esc,hashlib.sha256(raw).hexdigest()

def donors(dates,rep):
 rng=np.random.default_rng(11037+rep);ix=np.arange(len(dates))
 for year in sorted(set(dates.year)):
  p=np.flatnonzero(dates.year==year);ix[p]=np.roll(p,int(rng.integers(28,len(p)-27)))
 return ix

def task(job):
 family,seed=job;torch.set_num_threads(1)
 data=DatosNumericos(BUNDLE);dates=pd.date_range('2022-01-01','2025-12-31',tz='America/Bogota');pos=data.indice.get_indexer(dates)
 model,esc,sha=load(family,seed);c=data.construir(pos,_escalas(esc,data));sample=np.flatnonzero(dates.day.isin([1,8,15,22]))
 def pred(x,cp,cf,level):
  result=[];batch=1 if family=='timesnet' else 16
  with torch.inference_mode():
   for i in range(0,len(x),batch):
    args=[torch.from_numpy(a[i:i+batch].copy()) for a in [x,cp,cf,c.dia[sample]]]
    result.append(model(*args).numpy().astype(float)*level[i:i+batch,None])
  return np.concatenate(result)
 x,cp,cf,level=c.x[sample],c.cal_p[sample],c.cal_f[sample],c.nivel[sample];y=c.y[sample]
 basal=pred(x,cp,cf,level);mae=np.abs(basal-y).mean();rows=[]
 columns={'perfil_demanda':[0],'precio':[1],'reservas':[2,3,4],'aportes':[5],'temperatura':list(range(6,14)),'humedad':list(range(14,22)),'precipitacion':list(range(22,30))}
 for rep in range(3):
  donor=donors(dates,rep)[sample];cx=data.construir(pos[donor],_escalas(esc,data))
  for group in GROUPS:
   a,b,d,l=x.copy(),cp,cf,level
   if group=='nivel_ventana':l=cx.nivel
   elif group=='calendario':b,d=cx.cal_p,cx.cal_f
   else:
    cols=columns[group];a[:,:,cols]=cx.x[:,:,cols]
    if group in ['precio','reservas','aportes']:assert not a[:,:,30:32].any()
   p=pred(a,b,d,l);m=float(np.abs(p-y).mean())
   rows.append(dict(clave=f'nuevo_{family}__N',semilla=seed,repeticion=rep,grupo=group,MAE_base=mae,MAE_perturbado=m,delta_MASE=(m-mae)/Q,cambio_pct=100*(m/mae-1),divergencia_kwh=float(np.abs(p-basal).mean()),origenes=len(sample)))
 pd.DataFrame(rows).to_csv(OUT/f'importancia_{family}_{seed}.csv',index=False)
 print(f'Importancia {family} {seed}: completa',flush=True)
 return family,seed,sha

def main():
 jobs=[(f,s) for f in ['nbeats','timesnet'] for s in range(1000,1005)]
 pending=[j for j in jobs if not (OUT/f'importancia_{j[0]}_{j[1]}.csv').is_file()]
 if pending:
  with ProcessPoolExecutor(max_workers=5) as pool:list(pool.map(task,pending))
 # Medición secuencial después de terminar los trabajadores, sin contención entre modelos.
 torch.set_num_threads(1);torch.set_num_interop_threads(1)
 data=DatosNumericos(BUNDLE);dates=pd.date_range('2022-01-01','2025-12-31',tz='America/Bogota');pos=data.indice.get_indexer(dates)
 rows=[]
 for family,seed in jobs:
  model,esc,sha=load(family,seed);c=data.construir(pos[[0,730,1460]],_escalas(esc,data))
  with torch.inference_mode():
   for i,o in enumerate([0,730,1460]):
    args=[torch.from_numpy(a[i:i+1].copy()) for a in [c.x,c.cal_p,c.cal_f,c.dia]]
    for _ in range(20):model(*args)
    times=[]
    for _ in range(100):
     t=time.perf_counter_ns();p=model(*args);times.append((time.perf_counter_ns()-t)/1e6)
    rows.append(dict(clave=f'nuevo_{family}__N',semilla=seed,origen=str(dates[o]),mediana_ms=float(np.median(times)),p95_ms=float(np.quantile(times,.95)),media_ms=float(np.mean(times)),repeticiones=100,calentamiento=20,batch=1,hilos=1,documentos=0,cierres_textuales=0,documentos_origen=0,checkpoint_sha256=sha))
  print(f'Latencia {family} {seed}: completa',flush=True)
 pd.DataFrame(rows).to_csv(OUT/'latencia_nuevos.csv',index=False)
 (OUT/'protocolo_diagnosticos.json').write_text(json.dumps(dict(fecha='2026-11-20',torch=torch.__version__,sistema=platform.platform(),cpu=platform.processor(),origenes_importancia=192,semillas=5,repeticiones_importancia=3,donantes='Desplazamiento circular dentro del año; semilla 11037 + repetición; separación mínima 28 días',grupos=GROUPS,lote_timesnet=1,hilos_latencia=1,calentamientos=20,repeticiones_latencia=100,origenes_latencia=[0,730,1460],alcance_latencia='Forward con entradas preparadas; sin lectura ni escalado',entrenamiento=False,publicacion=False),ensure_ascii=False,indent=2),encoding='utf8')
if __name__=='__main__':main()


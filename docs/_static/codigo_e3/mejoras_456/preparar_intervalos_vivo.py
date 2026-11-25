"""Calibración causal del checkpoint utilizado en vivo; sin ajuste neuronal."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from evaluar_incertidumbre_comun import predictions
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/intervalos_vivo_20261125'
DEST=ROOT/'book/_static/inferencia_m4/calibracion'
LEVELS=[.8,.9,.95]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,z):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(z,ensure_ascii=False,allow_nan=False,indent=2)+'\n',encoding='utf-8')
def radii(scores,days,origin,scales,level=.9,scope='horario'):
 if origin<np.datetime64('2023-01-01'):return None
 mask=(days<=origin-np.timedelta64(2,'D'))&(days>=origin-np.timedelta64(366,'D'))
 sample=scores[mask];n=len(sample)
 if n<60:return None
 rank=int(np.ceil((n+1)*level))
 if rank>n:return None
 if scope=='horario':return np.sort(sample,axis=0)[rank-1]
 return np.sort(np.max(sample/scales,axis=1))[rank-1]*scales

def main():
 OUT.mkdir(exist_ok=True);DEST.mkdir(exist_ok=True)
 models,y,dates,master,sources=predictions(ROOT);days=np.array(dates.strftime('%Y-%m-%d'),dtype='datetime64[D]')
 protocol=dict(fecha='2026-11-25',semilla_M4=101,semilla_comparadores=1000,calibracion_inicial=['2022-01-01','2022-12-30'],inicio_intervalos='2023-01-01',
  metodo='365 últimos días disponibles con edades2…366; mínimo60 observaciones, estadístico de orden ceil((n+1)*nivel)',
  escala_conjunta='Mediana del error absoluto por horizonte en los364días iniciales; piso1kWh, congelada',niveles=LEVELS,alcances=['horario','trayectoria24'],
  escenarios='Mismo checkpoint del panel; distinto de la media de semillas del benchmark',disponibilidad='No emplear el error del día objetivo ni de d−1. Sin reserva elegible no emitir intervalos.',
  precision='Errores almacenados FP32 en kWh; cortes y cuantiles reproducibles entrePython yJavaScript',
  alcance='Cobertura empírica con dependencia temporal; no garantía exacta ni calibración condicional para historias intervenidas.',entrenamiento=False,publicacion=False)
 write(OUT/'protocolo.json',protocol);meta=dict(protocol=protocol,dias=[str(d)for d in days],models={});rows=[]
 for key,p in models.items():
  if key.startswith('ingenuo_'):continue
  score=np.abs(p[0]-y).astype('<f4');initial=score[days<=np.datetime64('2022-12-30')]
  scale=np.maximum(np.median(initial,axis=0).astype(float),1.)
  file=key.replace('+','_')+'.bin';path=DEST/file;path.write_bytes(score.tobytes())
  m=dict(name=master.loc[key,'modelo'],seed=101 if key=='m4_n'else 1000,file=file,sha256=sha(path),shape=list(score.shape),scales=scale.tolist())
  meta['models'][key]=m
  for level in LEVELS:
   for scope in ['horario','trayectoria24']:
    covered=[];joint=[];width=[]
    for t in np.flatnonzero(days>=np.datetime64('2023-01-01')):
     q=radii(score,days,days[t],scale,level,scope);assert q is not None
     hit=score[t]<=q;covered.append(hit.mean());joint.append(hit.all());width.append(2*q.mean()/1e6)
    rows.append(dict(clave=key,modelo=m['name'],semilla=m['seed'],nivel=level,alcance=scope,dias=len(covered),cobertura_horaria=float(np.mean(covered)),cobertura_conjunta=float(np.mean(joint)),ancho_medio_GWh=float(np.mean(width))))
  print('Calibración del checkpoint:',key,flush=True)
 write(DEST/'manifest.json',meta);write(OUT/'manifest.json',meta)
 pd.DataFrame(rows).to_csv(OUT/'cobertura_checkpoint.csv',index=False)
 write(OUT/'verificacion.json',dict(modelos=len(meta['models']),resumenes=len(rows),fuentes_sha256={p.relative_to(ROOT).as_posix():sha(p)for p in sources},entrenamiento=False,publicacion=False))
 print('Intervalos de16checkpoints preparados con cortes causales.',flush=True)
if __name__=='__main__':main()

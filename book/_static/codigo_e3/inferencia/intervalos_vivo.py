"""Intervalos del checkpoint de inferencia con errores estrictamente anteriores."""
from pathlib import Path
from functools import lru_cache
import hashlib,json
import numpy as np
BASE=Path(__file__).resolve().parents[1]/'book/_static/inferencia_m4/calibracion'
@lru_cache(maxsize=1)
def manifest():return json.loads((BASE/'manifest.json').read_text(encoding='utf-8'))
@lru_cache(maxsize=16)
def scores(key):
 m=manifest()['models'][key];raw=(BASE/m['file']).read_bytes()
 if hashlib.sha256(raw).hexdigest()!=m['sha256']:raise ValueError('Calibración incompatible con su registro.')
 return np.frombuffer(raw,dtype='<f4').reshape(m['shape'])
def interval(key,origin,prediction_kwh,level=.9,scope='horario'):
 level=float(level)
 if level not in [.8,.9,.95]or scope not in ['horario','trayectoria24']:raise ValueError('Nivel o alcance de intervalo incompatible.')
 meta=manifest();m=meta['models'][key];t=np.datetime64(origin,'D');days=np.asarray(meta['dias'],dtype='datetime64[D]')
 empty=lambda reason:dict(available=False,reason=reason,level=level,scope=scope,n=0)
 if t<np.datetime64('2023-01-01'):return empty('No hay reserva de calibración anterior al origen. Los intervalos comienzan en 2023.')
 mask=(days<=t-np.timedelta64(2,'D'))&(days>=t-np.timedelta64(366,'D'));sample=scores(key)[mask];n=len(sample)
 rank=int(np.ceil((n+1)*level))
 if n<60 or rank>n:return empty('No hay suficientes errores recientes disponibles para calibrar este origen.')
 if scope=='horario':radius=np.sort(sample,axis=0)[rank-1].astype(float)
 else:
  scale=np.asarray(m['scales']);radius=np.sort(np.max(sample/scale,axis=1))[rank-1]*scale
 p=np.asarray(prediction_kwh,dtype=float)
 return dict(available=True,level=level,scope=scope,n=n,start=str(days[mask][0]),end=str(days[mask][-1]),lower_kwh=(p-radius).tolist(),upper_kwh=(p+radius).tolist(),seed=m['seed'],method='Cuantil móvil causal365; retraso2días; checkpoint individual')

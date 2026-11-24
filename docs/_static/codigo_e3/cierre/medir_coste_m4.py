"""Coste de forward en proceso nuevo por modelo; no entrenamiento."""
from pathlib import Path
import sys,time,json,threading,ctypes,subprocess
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'dashboard'),str(ROOT/'scripts')]
from completar_e3_sin_entrenar import OUT,write
from runtime_inferencia import get_runtime
import numpy as np,pandas as pd,torch

def rss():
 if sys.platform=='win32':
  from ctypes import wintypes
  class Counters(ctypes.Structure):
   _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[(n,ctypes.c_size_t) for n in ['PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage']]
  c=Counters();c.cb=ctypes.sizeof(c)
  handle=ctypes.windll.kernel32.GetCurrentProcess
  handle.restype=ctypes.c_void_p
  fn=ctypes.windll.psapi.GetProcessMemoryInfo;fn.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_ulong]
  assert fn(handle(),ctypes.byref(c),c.cb)
  return c.WorkingSetSize
 else:
  # Linux: residente actual, en páginas, sin dependencia psutil.
  import os
  return int(Path('/proc/self/statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE')

def worker(key):
 r=get_runtime();I=r.inputs;I.configure('2023-07-01');torch.set_num_threads(1)
 with torch.inference_mode():
  for _ in range(10):r.forward(key)
  values=[]
  for _ in range(100):
   t=time.perf_counter();r.forward(key);values.append((time.perf_counter()-t)*1000)
  baseline=rss();samples=[];stop=threading.Event()
  def sample():
   while not stop.is_set():samples.append(rss());stop.wait(.001)
  thread=threading.Thread(target=sample);thread.start()
  try:
   for _ in range(30):r.forward(key)
  finally:stop.set();thread.join()
  with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],profile_memory=True,with_flops=True) as p:r.forward(key)
  modules=[m for m,_ in r.experts]+[r.router,r.memory] if key=='m4_n' else [r.additional(key)[0]]
  result=dict(modelo=key,semilla=101 if key=='m4_n' else 1000,n_mediciones=100,
   mediana_ms=float(np.median(values)),p95_ms=float(np.quantile(values,.95)),
   parametros_almacenados=sum(x.numel() for m in modules for x in m.parameters()),
   bytes_parametros=sum(x.numel()*x.element_size() for m in modules for x in m.parameters()),
   FLOPs_operadores_cubiertos=int(sum(e.flops for e in p.key_averages())),
   RSS_base_bytes=baseline,RSS_max_observado_bytes=max(samples+[baseline]),RSS_incremento_observado_bytes=max(samples+[baseline])-baseline,
   frecuencia_muestreo_ms=1,threads=1,preparacion_incluida=False,
   alcance='Proceso aislado por modelo; forward CPU, batch1 y banco temporal; M4 incluye expertos+M1+memoria; RSS total incluye datos/runtime; FLOPs parciales')
 (OUT/('coste_'+key+'.json')).write_text(json.dumps(result,indent=2),encoding='utf-8')
 print(key,result['mediana_ms'],result['p95_ms'],flush=True)

if __name__=='__main__':
 if len(sys.argv)>1:worker(sys.argv[1])
 else:
  for key in ['m4_n','nuevo_timesnet__N']:subprocess.run([sys.executable,str(Path(__file__).resolve()),key],check=True)
  rows=[json.loads(p.read_text()) for p in OUT.glob('coste_*.json')]
  pd.DataFrame(rows).to_csv(OUT/'eficiencia.csv',index=False)

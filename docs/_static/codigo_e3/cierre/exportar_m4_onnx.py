"""Exporta los cinco módulos activos de M4, verifica el pipeline y crea paquete CPU."""
from pathlib import Path
import sys,json,hashlib,shutil,zipfile,subprocess,time
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'dashboard'),str(ROOT/'scripts')]
from completar_e3_sin_entrenar import OUT,Q,sha,write
from runtime_inferencia import get_runtime
import torch,numpy as np
import onnx,onnxruntime as ort

class MemoryExport(torch.nn.Module):
 def __init__(self,red):super().__init__();self.red=red
 def forward(self,query,keys,days,origin,residuals,base):
  q=self.red.proy_q(torch.cat([query,query.new_zeros((1,64))],1))
  k=self.red.proy_k(torch.cat([keys,keys.new_zeros((365,64))],1))
  age=origin-days;valid=(age>=2)&(age<=365)
  score=q@k.T/(32**.5);score=score.masked_fill(~valid[None],-1e9)
  values,ids=score.topk(16,-1);ok=valid[ids];any_valid=ok.any(-1)
  a=values.softmax(-1)*ok;a=a/a.sum(-1,keepdim=True).clamp_min(1e-8)
  r=residuals[ids];mu=(a[:,:,None]*r).sum(1)
  dispersion=((r-mu[:,None]).square()*a[:,:,None]).sum(1).mean(-1).clamp_min(1e-12).sqrt()
  maxsim=torch.where(any_valid,values[:,0],torch.zeros_like(values[:,0]))
  features=torch.cat([q,maxsim[:,None],dispersion[:,None],ok.sum(-1).to(q.dtype)[:,None]/16,(a*age[ids]).sum(-1)[:,None]/365],1)
  gate=torch.sigmoid(self.red.compuerta(features))[:,0]*any_valid
  output=base+Q*gate[:,None]*mu
  return torch.where(any_valid[:,None],output,base),gate,a,ids

def session(p):
 options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
 return ort.InferenceSession(str(p),sess_options=options,providers=['CPUExecutionProvider'])

def export(red,args,file,names,outputs):
 torch.onnx.export(red,args,file,input_names=names,output_names=outputs,opset_version=18,dynamo=False)
 onnx.checker.check_model(str(file));return session(file)

def main():
 runtime=get_runtime();runtime.prepare_memory();torch.set_num_threads(1)
 torch.backends.mha.set_fastpath_enabled(False)
 dest=OUT/'paquete_m4_cpu';dest.mkdir(exist_ok=True)
 models=dest/'modelos';models.mkdir(exist_ok=True)
 exported={};results=[];dates=['2022-01-01','2023-07-01','2024-02-15','2025-12-31','2026-01-01']
 wrapper=MemoryExport(runtime.memory).eval();constants={
  'router_media':runtime.router_ck['media'],'router_escala':runtime.router_ck['escala'],
  'memory_media':runtime.memory_ck['media'],'memory_escala':runtime.memory_ck['escala']}
 np.savez_compressed(dest/'escalas.npz',**constants)
 with torch.no_grad():
  for number,date in enumerate(dates):
   I=runtime.inputs;I.configure(date);captured={}
   def router_hook(module,args):captured['router']=tuple(x.detach().clone() for x in args)
   def memory_hook(module,args):
    b,j=args;captured['memory']=(b,j.clone())
   h=runtime.router.register_forward_pre_hook(router_hook);m=runtime.memory.register_forward_pre_hook(memory_hook)
   truth,_,detail=runtime.forward('m4_n');h.remove();m.remove()
   data={};expert_outputs=[]
   for i,(red,sc) in enumerate(runtime.experts):
    d=I.construir(sc);args=tuple(torch.from_numpy(np.ascontiguousarray(d[k])) for k in ['x','cp','cf','dia'])
    for name,value in zip(['x','cp','cf','dia'],args):data[f'{name}_{i}']=value.numpy()
    data[f'nivel_{i}']=np.asarray(d['nivel'],dtype='float32')
    file=models/f'experto_{i}.onnx'
    if number==0:exported[f'experto_{i}']=export(red,args,file,['x','cp','cf','dia'],['pred'])
    s=exported[f'experto_{i}'];feed={x.name:data[f'{x.name}_{i}'] for x in s.get_inputs()}
    output=s.run(None,feed)[0]*d['nivel'][:,None];expert_outputs.append(output.astype('float32'))
   raw_context=captured['router'][0].numpy()*constants['router_escala']+constants['router_media']
   data['contexto']=raw_context.astype('float32')
   data['nivel_contexto']=np.array([I.window[:,0].mean()],dtype='float32')
   xp=np.stack(expert_outputs,-1)
   raw_context[:,672:744]=(xp[0]/data['nivel_contexto'][0]).ravel()
   raw_context[:,744:768]=xp[0].std(-1)/data['nivel_contexto'][0]
   rx=((raw_context-constants['router_media'])/constants['router_escala']).astype('float32')
   if number==0:exported['router']=export(runtime.router,(torch.from_numpy(rx),torch.from_numpy(xp)),models/'router.onnx',['x','pred'],['base','c','w'])
   base=exported['router'].run(None,{'x':rx,'pred':xp})[0]
   b,j=captured['memory'];n=j.item();assert n<=365
   keys=np.zeros((365,912),dtype='float32');keys[:n]=b.num[:n].numpy()
   origin=b.dias[-1:].numpy();days=np.full(365,int(origin[0]),dtype='int64');days[:n]=b.dias[:n].numpy()
   residuals=np.zeros((365,24),dtype='float32');residuals[:n]=b.residuos[:n].numpy()
   data.update(keys=keys,days=days,origin=origin,residuals=residuals)
   mq=((raw_context-constants['memory_media'])/constants['memory_escala']).astype('float32')
   feed=dict(query=mq,keys=keys,days=days,origin=origin,residuals=residuals,base=base.astype('float32'))
   if number==0:exported['memory']=export(wrapper,tuple(torch.from_numpy(x) for x in feed.values()),models/'memoria.onnx',list(feed),['pred','gate','attention','neighbors'])
   out=exported['memory'].run(None,feed)[0][0]
   error=float(np.max(np.abs(out-truth)));relative=float(np.max(np.abs(out-truth)/np.maximum(np.abs(truth),1)))
   results.append(dict(origen=date,max_error_kwh=error,max_error_relativo=relative))
   assert relative<1e-4,(date,error,relative)
   np.savez_compressed(dest/f'ejemplo_{number}.npz',**data)
   np.save(dest/f'esperado_{number}.npy',truth)
   print('ONNX round-trip completo',date,error,'kWh',flush=True)
 write('exportacion.json',dict(opset=18,provider='CPUExecutionProvider',semilla=101,modulos=5,sha256_checkpoints=runtime.hashes,
  resultados=results,tolerancia_relativa=1e-4,entrenamiento=False,
  alcance='Tres expertos + M1 + memoria; entradas preparadas sin datos futuros. Banco relleno hasta365 episodios, máscara causal conservada.',
  archivos={p.relative_to(dest).as_posix():sha(p) for p in models.glob('*.onnx')}))
 shutil.copy2(OUT/'exportacion.json',dest/'exportacion.json')
 # Fuentes de ejecución separadas de las clases de entrenamiento.
 source=ROOT/'scripts/ejecutar_m4_onnx.py';shutil.copy2(source,dest/'ejecutar.py')
 (dest/'requirements.txt').write_text('numpy==2.5.4\nonnxruntime==1.32.0\nflatbuffers==25.12.19\npackaging==26.4\nprotobuf==7.36.2\n',encoding='utf-8')
 (dest/'Dockerfile').write_text('FROM python:3.13-slim\nWORKDIR /app\nCOPY requirements.txt .\nRUN pip install --no-cache-dir -r requirements.txt\nCOPY . .\nCMD ["python", "ejecutar.py", "--verificar"]\n',encoding='utf-8')
 (dest/'README.md').write_text('''# Inferencia portable de M4_N

Python 3.13, CPU. No se necesitan PyTorch, el repositorio ni rutas externas.

```
python -m venv .venv
# Windows: .venv/Scripts/python; Linux: .venv/bin/python
python -m pip install -r requirements.txt
python ejecutar.py --verificar
python ejecutar.py --entrada ejemplo_0.npz --salida pronostico.json
```

El pipeline ejecuta tres expertos, reconstruye sus atributos del contexto, combina con M1 y consulta la memoria. Los cinco ejemplos tienen entradas anteriores a su origen, escaladas con parámetros conservados. El contrato NPZ contiene x/cp/cf/dia/nivel por experto; contexto numérico912; nivel_contexto; keys365×912 ya escaladas; days; origin; residuals365×24 normalizados por q. Los episodios ausentes se rellenan con edad cero y residuo cero. Antes de usar datos propios se debe construir este contrato con el preprocesamiento y las reglas de disponibilidad documentadas en el JBook; los ejemplos no sustituyen una adquisición automática de datos.

La predicción corresponde a la semilla101, prefijada; no es el promedio de pérdidas de las diez semillas. El manifiesto contiene hashes, tolerancia y errores del round-trip. exportacion.json conserva la procedencia.

Contenedor: `docker build -t m4-cpu .` y `docker run --rm m4-cpu`. El Dockerfile queda preparado; su construcción requiere un motor Docker disponible.
''',encoding='utf-8')
 files={p.relative_to(dest).as_posix():sha(p) for p in dest.rglob('*') if p.is_file() and p.name!='MANIFIESTO.json'}
 (dest/'MANIFIESTO.json').write_text(json.dumps(files,indent=2),encoding='utf-8')
 # Ejecutar en un proceso aislado con solo el paquete en sys.path local.
 run=subprocess.run([sys.executable,'-I',str(dest/'ejecutar.py'),'--verificar'],cwd=dest,text=True,capture_output=True,check=True)
 (OUT/'verificacion_paquete.log').write_text(run.stdout+run.stderr,encoding='utf-8')
 files={p.relative_to(dest).as_posix():sha(p) for p in dest.rglob('*') if p.is_file() and p.name!='MANIFIESTO.json'}
 (dest/'MANIFIESTO.json').write_text(json.dumps(files,indent=2),encoding='utf-8')
 with zipfile.ZipFile(OUT/'M4_inferencia_CPU_portable.zip','w',zipfile.ZIP_DEFLATED) as z:
  for p in dest.rglob('*'):
   if p.is_file():z.write(p,p.relative_to(dest))
 print('Paquete ONNX creado y verificado en proceso aislado.',flush=True)

if __name__=='__main__':main()

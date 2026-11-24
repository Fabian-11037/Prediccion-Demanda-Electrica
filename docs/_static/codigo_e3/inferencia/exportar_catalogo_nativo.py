"""Exportación CPU de los quince comparadores con pesos congelados, sin ajuste."""
from pathlib import Path
import sys,json,hashlib,copy,math,traceback
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'dashboard'),str(ROOT/'scripts')]
import numpy as np
import torch
import onnx
import onnxruntime as ort
from runtime_inferencia import get_runtime,MODELS,IA
from preparar_inferencia_nativa import plain
DEST=ROOT/'book/_static/inferencia_m4'
OUT=ROOT/'results/inferencia_nativa_catalogo_20261124'
DATES=['2022-01-01','2023-07-01','2024-02-15','2025-12-31','2026-01-01']

def sha(data):return hashlib.sha256(data).hexdigest()
def jsonwrite(p,obj):p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def graph(model,args,names,file,outputs=['pred'],dynamic=None):
 model.eval();model.requires_grad_(False)
 torch.onnx.export(model,args,file,input_names=names,output_names=outputs,opset_version=18,dynamo=False,dynamic_axes=dynamic)
 onnx.checker.check_model(str(file))
 options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
 s=ort.InferenceSession(str(file),sess_options=options,providers=['CPUExecutionProvider'])
 feed={name:arg.detach().numpy() for name,arg in zip(names,args) if name in [v.name for v in s.get_inputs()]}
 expected=model(*args)
 if isinstance(expected,tuple):expected=expected[0]
 actual=s.run(None,feed)[0]
 scale=max(float(expected.detach().abs().max()),1.0)
 error=float(np.max(np.abs(actual-expected.detach().numpy())))
 assert error<=1e-4*scale,(file,error,scale)
 print('Paridad de subgrafo',file.name,'error_max',error,'escala',scale,flush=True)
 # Cada parte cabe en GitHub; el navegador reconstruye el grafo original FP32.
 data=file.read_bytes();parts=[]
 for i,start in enumerate(range(0,len(data),32*1024*1024)):
  block=data[start:start+32*1024*1024];name=f'{file.stem}_{i:02d}.bin'
  (DEST/'catalogo'/name).write_bytes(block);parts.append({'file':'catalogo/'+name,'sha256':sha(block),'bytes':len(block)})
 return {'parts':parts,'sha256':sha(data),'bytes':len(data),'inputs':[v.name for v in s.get_inputs()]}

class TNPrepare(torch.nn.Module):
 def __init__(self,red):super().__init__();self.red=red
 def forward(self,x,cp):
  r=self.red;mean=x.mean(1,keepdim=True);z=x-mean;std=torch.sqrt(torch.var(z,dim=1,keepdim=True,unbiased=False)+1e-5)
  e=r.drop(r.valor(z/std)+r.posicion(z/std)+r.marca(cp))
  return r.predict_linear(e.permute(0,2,1)).permute(0,2,1),mean,std

class TNFinish(torch.nn.Module):
 def __init__(self,red):super().__init__();self.red=red
 def forward(self,e,mean,std,cf):
  r=self.red;out=r.proyeccion(e)*std[:,0,:].unsqueeze(1)+mean[:,0,:].unsqueeze(1)
  return 1+r.lectura(torch.cat([out[:,-r.H:,:1],cf],2)).squeeze(-1)

def scaler(sc,I,kind):
 def arr(key):return plain(sc[key]) if not isinstance(sc[key],dict) else [sc[key][c] for c in I.hist]
 if kind=='baseline':
  return dict(kind=kind,cal_mean=[sc['media'][c] for c in I.cal],cal_std=[sc['std'][c] for c in I.cal],sigma=sc.get('sigma_demanda',1),sequential=sc['secuencial'])
 cm=sc.get('cal_media',sc.get('media_cal'))
 cs=sc.get('cal_std',sc.get('std_cal'))
 if cm is None:cm=[sc['media_calendario'][c] for c in I.cal];cs=[sc['std_calendario'][c] for c in I.cal]
 return dict(kind=kind,median=arr('mediana'),mean=arr('media'),std=arr('std'),cal_mean=plain(cm),cal_std=plain(cs),sigma=sc['sigma_demanda'])

def compact_text(red,I):
 agg=getattr(red,'agregador',None) or getattr(red,'relevancia',None)
 if agg is None:return None
 rows=[];counts=[]
 for date in DATES:
  I.configure(date);day=I.day
  ids=I.lista[day-6:day+1];rows.append(ids)
  counts.append(int((ids[-1]>=0).sum()))
 rows=torch.cat(rows,0);unique=torch.unique(rows[rows>=0]);mapping={int(v):i for i,v in enumerate(unique.tolist())}
 compact=rows.clone()
 for t in range(compact.shape[0]):
  for j in range(compact.shape[1]):
   if compact[t,j]>=0:compact[t,j]=mapping[int(compact[t,j])]
 agg.vectores=agg.vectores[unique].clone();agg.lista=compact
 return counts

def main():
 OUT.mkdir(exist_ok=True);(DEST/'catalogo').mkdir(exist_ok=True)
 torch.set_num_threads(1);torch.backends.mha.set_fastpath_enabled(False)
 runtime=get_runtime();I=runtime.inputs
 I.configure(DATES[-1]);I.prepare_text()
 manifest={'models':{},'dates':DATES,'seed':1000,'training':False,'tolerance':1e-4}
 refs=[]
 keys=[k for k in MODELS if k!='m4_n']
 selected=sys.argv[1:] or keys
 if (OUT/'catalogue.json').exists():manifest=json.loads((OUT/'catalogue.json').read_text(encoding='utf-8'))
 if (OUT/'referencias_python.json').exists():refs=json.loads((OUT/'referencias_python.json').read_text(encoding='utf-8'))
 for number,key in enumerate(keys):
  if key not in selected:continue
  print('Exportando',key,flush=True)
  I.configure(DATES[-1])
  with torch.inference_mode():
   if key in ['mlp','arx_lineal','cnn1d','lstm']:
    red,sc,h=runtime.baseline(key);kind='baseline';repo=False
   elif key in ['nuevo_nbeats__N','nuevo_timesnet__N']:
    red,sc,h=runtime.additional(key);kind='sequence';repo=False
   else:
    record=next(m for m in runtime.registry if m['experimento']+'__'+m['variante']==key and m['semilla']==1000)
    red,data,repo,_=runtime.expanded(key);h=record['sha256_checkpoint'];sc=json.loads((IA.CON/record['artefactos']['escaladores']).read_text(encoding='utf-8'));kind='repo' if repo else 'sequence'
   # Referencia de la implementación sin compactar, para cinco orígenes y perturbación.
   refs=[r for r in refs if r['model']!=key]
   for index,date in enumerate(DATES):
    I.configure(date)
    p,_,_=runtime.forward(key)
    refs.append(dict(model=key,origin=date,index=index,change=0,prediccion_kwh=p.tolist()))
    if index==4:
     I.window[-24:,0]*=1.03;p,_,_=runtime.forward(key)
     refs.append(dict(model=key,origin=date,index=index,change=3,prediccion_kwh=p.tolist()))
   I.configure(DATES[-1]);counts=compact_text(red,I)
   d=runtime.baseline_inputs(sc) if kind=='baseline' else I.construir(sc,repo)
   inputs=['x','cf'] if kind=='baseline' or repo else ['x','cp','cf','dia']
   if counts is not None:d['dia']=np.array([34],dtype=np.int64)
   args=tuple(torch.from_numpy(np.ascontiguousarray(d[k])) for k in inputs)
   item=dict(name=MODELS[key],checkpoint_sha256=h,scaler=scaler(sc,I,kind),seed=1000,text_counts=counts,inputs=inputs,channels=int(d['x'].shape[-1]))
   if key=='nuevo_timesnet__N':
    x,cp,cf,dia=args;prepare=TNPrepare(red);e,mean,std=prepare(x,cp)
    item.update(mode='timesnet',top_k=red.bloques[0].k,length=int(e.shape[1]),d_model=int(e.shape[2]),layers=len(red.bloques))
    item['prepare']=graph(prepare,(x,cp),['x','cp'],OUT/'timesnet_prepare.onnx',['e','mean','std'])
    item['norm']=graph(red.norma,(e,),['e'],OUT/'timesnet_norm.onnx')
    item['convs']=[]
    for i,block in enumerate(red.bloques):
     sample=e.transpose(1,2).reshape(1,e.shape[2],8,24)
     item['convs'].append(graph(block.conv,(sample,),['x'],OUT/f'timesnet_conv{i}.onnx',dynamic={'x':{2:'rows',3:'period'},'pred':{2:'rows',3:'period'}}))
    item['finish']=graph(TNFinish(red),(e,mean,std,cf),['e','mean','std','cf'],OUT/'timesnet_finish.onnx')
   else:
    item['mode']='direct';item['graph']=graph(red,args,inputs,OUT/f'model_{number:02d}.onnx')
   manifest['models'][key]=item
   jsonwrite(OUT/'catalogue.json',manifest);jsonwrite(DEST/'catalogue.json',manifest);jsonwrite(OUT/'referencias_python.json',refs)
   print('OK',key,flush=True)
 print('Exportación congelada completada.',flush=True)

if __name__=='__main__':main()

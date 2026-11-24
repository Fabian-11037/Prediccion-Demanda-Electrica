"""Inferencia CPU portable. El contrato preparado se describe en README.md."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import onnxruntime as ort

def run(folder,entrada):
 folder=Path(folder)
 options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
 def session(name):return ort.InferenceSession(str(folder/'modelos'/name),sess_options=options,providers=['CPUExecutionProvider'])
 with np.load(entrada,allow_pickle=False) as z:d={k:z[k] for k in z.files}
 with np.load(folder/'escalas.npz',allow_pickle=False) as z:c={k:z[k] for k in z.files}
 p=[]
 for i in range(3):
  s=session(f'experto_{i}.onnx');p.append(s.run(None,{x.name:d[f'{x.name}_{i}'] for x in s.get_inputs()})[0]*d[f'nivel_{i}'][:,None])
 xp=np.stack(p,-1).astype('float32');context=d['contexto'].copy()
 context[:,672:744]=(xp[0]/d['nivel_contexto'][0]).ravel();context[:,744:768]=xp[0].std(-1)/d['nivel_contexto'][0]
 x=((context-c['router_media'])/c['router_escala']).astype('float32')
 base=session('router.onnx').run(None,{'x':x,'pred':xp})[0]
 query=((context-c['memory_media'])/c['memory_escala']).astype('float32')
 feed={k:d[k] for k in ['keys','days','origin','residuals']};feed.update(query=query,base=base)
 out=session('memoria.onnx').run(None,feed)
 return dict(prediccion_kwh=out[0][0].tolist(),compuerta=float(out[1][0]),entrenamiento=False)

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--entrada',default='ejemplo_0.npz');parser.add_argument('--salida');parser.add_argument('--verificar',action='store_true');args=parser.parse_args()
 folder=Path(__file__).resolve().parent
 manifest=folder/'MANIFIESTO.json'
 if manifest.exists():
  for name,digest in json.loads(manifest.read_text(encoding='utf-8')).items():
   assert hashlib.sha256((folder/name).read_bytes()).hexdigest()==digest,name
 if args.verificar:
  results=[]
  for i in range(5):
   out=run(folder,folder/f'ejemplo_{i}.npz');expected=np.load(folder/f'esperado_{i}.npy',allow_pickle=False)
   error=float(np.max(np.abs(np.asarray(out['prediccion_kwh'])-expected)/np.maximum(np.abs(expected),1)))
   assert error<1e-4,error;results.append(dict(ejemplo=i,max_error_relativo=error))
  print(json.dumps(dict(verificado=True,entrenamiento=False,resultados=results)))
 else:
  result=json.dumps(run(folder,Path(args.entrada)),ensure_ascii=False,indent=2)
  if args.salida:Path(args.salida).write_text(result,encoding='utf-8')
  else:print(result)

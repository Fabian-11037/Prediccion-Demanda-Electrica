"""Recorrido portable contra referencias GPU guardadas; no acredita todos los casos.

Requiere numpy y onnxruntime. Puede ejecutarse nativamente en Windows/Linux.
El rechazo histórico en empates permanece aunque este recorrido apruebe.
"""
import argparse, hashlib, json, platform, time
from pathlib import Path
import numpy as np
import onnxruntime as ort
Q=215617.90361589016

def ejecutar(raiz,proveedor):
    assert proveedor in ort.get_available_providers(),'Proveedor solicitado no disponible; no se permite fallback silencioso'
    filas=[]
    for car in sorted(raiz.glob('semilla_*')):
        with np.load(car/'entradas_portables.npz') as z:d={k:z[k] for k in z.files}
        op=ort.SessionOptions();op.intra_op_num_threads=1;op.inter_op_num_threads=1
        enc=ort.InferenceSession(str(car/'m4_n_encoder.onnx'),sess_options=op,providers=[proveedor])
        cor=ort.InferenceSession(str(car/'m4_n_corrector.onnx'),sess_options=op,providers=[proveedor])
        assert cor.get_providers()[0]==proveedor
        q=[];k=[]
        for x in d['num']:
            qq,kk=enc.run(None,{'num':x[None]});q.append(qq[0]);k.append(kk[0])
        q,k=np.asarray(q),np.asarray(k);pred=[];fallos=0;lat=[]
        for i,ref in zip(d['indices'],d['referencia_GPU']):
            edad=d['dias'][i]-d['dias'];ids=np.flatnonzero((edad>=2)&(edad<=365))
            keys=np.zeros((365,32),dtype='float32');r=np.zeros((365,24),dtype='float32');fechas=np.zeros(365,dtype='int64');valid=np.zeros(365,dtype=bool)
            keys[:len(ids)]=k[ids];r[:len(ids)]=d['residuos'][ids];fechas[:len(ids)]=d['dias'][ids];valid[:len(ids)]=True
            t=time.perf_counter();oo=cor.run(None,dict(consulta=q[i:i+1],claves=keys,residuos=r,fechas=fechas,validas=valid,origen=d['dias'][i:i+1],base=d['base'][i:i+1]));lat.append(time.perf_counter()-t)
            pr=oo[0][0];fallos+=int(not np.isfinite(pr).all() or not np.all(np.abs((pr-ref)/Q)<=1e-5+1e-5*np.abs(ref/Q)));pred.append(pr)
        pred=np.asarray(pred);y=d['y'];mase=float(np.abs(y.astype('float64')-pred).mean()/Q);mref=float(np.abs(y.astype('float64')-d['referencia_GPU']).mean()/Q)
        filas.append(dict(semilla=car.name,origenes=len(pred),fallos_observacion=fallos,diferencia_MASE=abs(mase-mref),recorrido_aprobado=not fallos and abs(mase-mref)<=1e-5,latencia_mediana_ms=float(np.median(lat[20:])*1000),latencia_p95_ms=float(np.percentile(lat[20:],95)*1000)))
    resultado=dict(sistema=platform.platform(),onnxruntime=ort.__version__,proveedor=proveedor,filas=filas,alcance='Solo recorrido diario y base externa. No levanta rechazo de empates ni certifica M1 completo.')
    (raiz/f'RECORRIDO_PORTABLE_{platform.system()}_{proveedor}.json').write_text(json.dumps(resultado,indent=2));print(json.dumps(resultado,indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('raiz',type=Path);ap.add_argument('--proveedor',default='CPUExecutionProvider');a=ap.parse_args();ejecutar(a.raiz,a.proveedor)

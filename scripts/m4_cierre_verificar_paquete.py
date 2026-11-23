"""Verificador nativo Windows/Linux de casos guardados; requiere numpy y onnxruntime."""
import argparse,json,platform,hashlib,time
from pathlib import Path
import numpy as np
import onnxruntime as ort

def verificar(raiz,proveedor):
    assert proveedor in ort.get_available_providers(),'Proveedor ausente; no se permite sustituirlo por CPU'
    contrato=json.loads((raiz/'PAQUETE.json').read_text());filas=[]
    for reg in contrato['grafos']:
        archivo=raiz/reg['archivo'];assert hashlib.sha256(archivo.read_bytes()).hexdigest()==reg['sha256']
        op=ort.SessionOptions();op.intra_op_num_threads=1;op.inter_op_num_threads=1
        if proveedor!='CPUExecutionProvider':op.add_session_config_entry('session.disable_cpu_ep_fallback','1')
        sesion=ort.InferenceSession(str(archivo),sess_options=op,providers=[proveedor]);sesion.disable_fallback()
        assert sesion.get_providers()[0]==proveedor
        for caso in reg['casos']:
            with np.load(raiz/caso) as z:
                activos={i.name for i in sesion.get_inputs()};entradas={k[3:]:z[k] for k in z.files if k.startswith('in_') and k[3:] in activos};assert set(entradas)==activos;salidas=sesion.run(None,entradas)
                bien=True;errores=[];t0=time.perf_counter()
                for nombre,pr in zip(reg['salidas'],salidas):
                    if 'ref_'+nombre not in z:continue
                    ref=z['ref_'+nombre];escala=reg.get('escala',1.) if nombre=='pred' else 1.
                    ok=np.isfinite(pr).all() and np.all(np.abs((pr-ref)/escala)<=1e-5+1e-5*np.abs(ref/escala));bien=bool(bien and ok)
                    errores.append(float(np.nanmax(np.abs(pr-ref)/escala)))
                filas.append(dict(grafo=reg['archivo'],caso=caso,aprobado=bien,max_errores=errores))
    r=dict(sistema=platform.platform(),proveedor=proveedor,onnxruntime=ort.__version__,casos=filas,aprobado=all(f['aprobado'] for f in filas),alcance='Subgrafos y casos conservados; no certifica plataformas no ejecutadas ni pipeline completo.')
    destino=raiz/f'RESULTADO_{platform.system()}_{proveedor}.json';destino.write_text(json.dumps(r,indent=2));print(json.dumps(dict(archivo=str(destino),aprobado=r['aprobado'],casos=len(filas))))
    return r

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('raiz',type=Path);p.add_argument('--proveedor',default='CPUExecutionProvider');a=p.parse_args();verificar(a.raiz,a.proveedor)

"""Latencia CPU de checkpoints congelados, sin ajuste ni lectura de etiquetas."""
from pathlib import Path
from types import SimpleNamespace
import json,sys,time,platform,hashlib
import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[1]
sha=json.loads((ROOT/'results/jbook/modelos_actualizados.json').read_text())['commit']
snapshot=ROOT/'results/repositorio'/sha
sys.path.insert(0,str(snapshot/'src'))
from demanda import modelos_base as M
from demanda.reentrenamiento_v3 import Secuencial as Convolucional
from demanda.variantes_v4 import Secuencial as Recurrente

def medir():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.manual_seed(11037)
    registro=json.loads((snapshot/'results/consolidado/modelos.json').read_text())
    entradas=SimpleNamespace(x=np.zeros((1,168,1),dtype=np.float32),
        cal_p=np.zeros((1,168,23),dtype=np.float32),cal_f=np.zeros((1,24,23),dtype=np.float32))
    x=torch.randn(1,168,24);c=torch.zeros(1,24,23)
    resultados=[]
    with torch.inference_mode():
        for nombre in ['mlp','arx_lineal','cnn1d','lstm']:
            for artefacto in sorted(registro[nombre]['artefactos'],key=lambda a:a['semilla']):
                path=snapshot/artefacto['pesos']
                checkpoint=torch.load(path,map_location='cpu',weights_only=True)
                p=registro[nombre]['configuracion']['params']
                if nombre=='lstm':red=Recurrente(nombre,p,entradas)
                elif nombre=='cnn1d':red=Convolucional(nombre,p,entradas)
                else:red=M.construir_red(nombre,p,entradas)
                red.load_state_dict(checkpoint['state_dict'],strict=True);red.eval()
                for _ in range(20):red(x,c)
                tiempos=[]
                for _ in range(200):
                    inicio=time.perf_counter_ns();salida=red(x,c);tiempos.append((time.perf_counter_ns()-inicio)/1e6)
                assert salida.shape==(1,24) and torch.isfinite(salida).all()
                resultados.append(dict(modelo=nombre,semilla=artefacto['semilla'],mediana_ms=float(np.median(tiempos)),p95_ms=float(np.quantile(tiempos,.95)),
                    media_ms=float(np.mean(tiempos)),repeticiones=200,calentamiento=20,batch=1,dispositivo='cpu',hilos=1,checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    out=ROOT/'results/jbook/modelos_actuales';out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(resultados).to_csv(out/'latencia_inferencia.csv',index=False)
    entorno=dict(fecha=pd.Timestamp.now(tz='America/Bogota').isoformat(),python=sys.version,torch=torch.__version__,numpy=np.__version__,pandas=pd.__version__,
        sistema=platform.platform(),procesador=platform.processor(),dispositivo='cpu',hilos=1,commit=sha,
        alcance='Sólo forward de un origen con 24 salidas; entradas sintéticas finitas; sin carga de pesos, preparación, transferencia ni escritura en el intervalo medido',
        semillas=list(range(1000,1005)),calentamiento=20,repeticiones=200,nuevo_entrenamiento=False,etiquetas_leidas=False)
    (out/'entorno_inferencia.json').write_text(json.dumps(entorno,ensure_ascii=False,indent=2),encoding='utf-8')
    print(pd.DataFrame(resultados).groupby('modelo')[['mediana_ms','p95_ms']].median().round(4).to_string())

if __name__=='__main__':medir()

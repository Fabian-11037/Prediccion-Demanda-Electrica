"""Coste homogéneo de dieciséis pipelines congelados; no ajusta modelos."""
from __future__ import annotations
from pathlib import Path
from collections import OrderedDict
import argparse,gc,hashlib,json,os,platform,subprocess,sys,threading,time
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'dashboard'),str(ROOT/'scripts')]
import numpy as np
import torch
from runtime_inferencia import Runtime,read,digest
from evaluar_robustez_congelada import Evaluator,RobustInputs
from medir_coste_m4 import rss
OUT=ROOT/'results/tabla_maestra_20261125'
DATES=['2022-01-01','2023-07-01','2025-12-31']
WARMUP=20
REPEATS=100

def write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,allow_nan=False,indent=2)+'\n',encoding='utf-8')

def worker(key):
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.manual_seed(20261125)
    r=Evaluator.__new__(Evaluator);Runtime.__init__(r)
    r.inputs.__class__=RobustInputs
    r.files={};r.selected_seed=None;r.red=None;r.aggregator=None
    I=r.inputs;I.configure(DATES[0])
    if '+O'in key or 'modelo7'in key:I.prepare_text()
    gc.collect();before=rss()
    reference=key.startswith('ingenuo_')
    seed=0 if reference else 101 if key=='m4_n'else 1000
    if not reference:r.load_model(key,seed)
    modules=[]if reference else [m for m,_ in r.experts]+[r.router,r.memory]if key=='m4_n'else[r.red]
    assert all(not m.training and not any(p.requires_grad for p in m.parameters())for m in modules)
    parameters={id(p):p for m in modules for p in m.parameters()}
    refs=read(ROOT/'results/inferencia_nativa_catalogo_20261124/referencias_python.json')
    latencies=[];bydate=[];parity=[]
    if reference:
        hours=np.arange(24)
        lag=168 if key=='ingenuo_168h'else 24*np.maximum(1,np.ceil((hours+2)/24).astype(int))
        offsets=hours-lag
        assert np.max(offsets)<=-2
        # x_demanda tiene rezago de una hora: +1 recupera la hora original.
        historical=I.raw['x_demanda_real_kwh'].to_numpy(copy=True)
        def predict():return historical[I.index.get_loc(I.origin)+offsets+1].copy()
    else:predict=r.predict_loaded
    with torch.inference_mode():
        for date in DATES:
            I.configure(date)
            output=predict()
            assert output.shape==(24,)and np.isfinite(output).all()
            if reference:
                # La referencia se contrasta con la serie fuente; nunca se
                # incorpora el objetivo futuro en el procedimiento cronometrado.
                import pandas as pd
                panel=pd.read_parquet(ROOT/'data/processed/panel_horario.parquet',columns=['y_demanda_real_kwh'])
                position=panel.index.get_loc(I.origin)
                expected=panel.iloc[position+offsets,0].to_numpy()
                del panel
            elif key=='m4_n':expected=r.expected(date)
            else:expected=np.array(next(a['prediccion_kwh']for a in refs if a['model']==key and a['origin']==date and a['change']==0))
            error=float(np.max(np.abs(output-expected)/np.maximum(np.abs(expected),1)))
            assert error<1e-4,(key,date,error)
            parity.append(dict(origen=date,max_error_relativo=error))
            for _ in range(WARMUP):predict()
            values=[]
            for rep in range(REPEATS):
                t=time.perf_counter_ns();predict();ms=(time.perf_counter_ns()-t)/1e6
                values.append(ms);latencies.append(dict(clave=key,semilla=seed,origen=date,repeticion=rep,ms=ms))
            bydate.append(dict(origen=date,mediana_ms=float(np.median(values)),p95_ms=float(np.quantile(values,.95))))
        baseline=rss();samples=[baseline];stop=threading.Event()
        def sample():
            while not stop.is_set():samples.append(rss());stop.wait(.001)
        thread=threading.Thread(target=sample,daemon=True);thread.start()
        try:
            for date in DATES:
                I.configure(date)
                for _ in range(30):predict()
        finally:stop.set();thread.join()
    for path,h in r.files.items():assert digest(Path(path))==h,path
    values=[a['ms']for a in latencies]
    result=dict(clave=key,semilla_coste=seed,mediana_ms=float(np.median(values)),p95_ms=float(np.quantile(values,.95)),
        parametros_pipeline=sum(p.numel()for p in parameters.values()),bytes_parametros=sum(p.numel()*p.element_size()for p in parameters.values()),
        RSS_antes_modelo_bytes=before,RSS_cargado_bytes=baseline,RSS_max_observado_bytes=max(samples),
        RSS_incremento_inferencia_bytes=max(samples)-baseline,por_origen=bydate,paridad=parity,
        mediciones=len(values),calentamiento_por_origen=WARMUP,lote=1,hilos=1,preparacion_numerica_incluida=True,
        lectura_disco_incluida=False,embeddings_precalculados=True,entrenamiento=False,fuentes_sha256=r.files,
        entorno=dict(sistema=platform.platform(),procesador=platform.processor(),python=platform.python_version(),torch=torch.__version__,cpu_logicas=os.cpu_count()))
    write(OUT/'costes'/f'{key.replace("+","_")}.json',result)
    write(OUT/'costes'/f'{key.replace("+","_")}_tiempos.json',latencies)
    print(f'{key}: {result["mediana_ms"]:.3f} ms; p95 {result["p95_ms"]:.3f}; parámetros {result["parametros_pipeline"]:,}',flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--modelo');args=parser.parse_args()
    if args.modelo:return worker(args.modelo)
    study=read(ROOT/'dashboard/site/data/study.json')
    protocol=dict(fecha='2026-11-25',origenes=DATES,semillas='101 M4; 1000 comparadores, prefijadas',
        dispositivo='CPU',hilos=1,lote=1,calentamiento=WARMUP,repeticiones_por_origen=REPEATS,
        alcance='Pipeline preparado: normalización y máscaras, forward, desnormalización; M4 incluye tres expertos, M1, recuperación y corrección. Excluye lectura de disco, construcción inicial de índices y generación de embeddings.',
        memoria='RSS total del proceso con datos y runtime, muestreado cada 1 ms; informar además tamaño de parámetros. La RSS observada no es memoria exclusiva del modelo ni pico garantizado.',
        agregacion='Mediana y percentil95 sobre 300 mediciones, tres orígenes con el mismo peso; una realización prefijada por procedimiento.',
        parametros='Total almacenado de módulos del pipeline, incluidos parámetros inactivos del checkpoint; sin confundir con parámetros entrenados del corrector M4.',
        paridad='24 salidas finitas, error relativo máximo menor que1e-4 respecto a referencias congeladas del mismo origen y semilla.',
        entrenamiento=False,publicacion=False)
    file=OUT/'protocolo_coste.json'
    if file.exists():assert read(file)==protocol
    else:write(file,protocol)
    for key in ['m4_n']+[m['id']for m in study['models']]+['ingenuo_168h','ingenuo_24h']:
        file=OUT/'costes'/f'{key.replace("+","_")}.json'
        if file.exists():
            print(key+': medición conservada',flush=True);continue
        subprocess.run([sys.executable,str(Path(__file__).resolve()),'--modelo',key],check=True)

if __name__=='__main__':main()

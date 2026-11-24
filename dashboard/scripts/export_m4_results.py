"""Exporta la evaluación conservada de M4_N al dashboard; no ejecuta redes."""
from pathlib import Path
import json,hashlib,shutil,sys
import numpy as np
import pandas as pd
import nbformat

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
SITE=ROOT/'dashboard/site'
RUN=ROOT/'results/m4_cierre_gpu_20261123/reducido_B'
DERIVED=ROOT/'results/evaluacion_m4_jbook'

def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,allow_nan=False,separators=(',',':')),encoding='utf-8')

def rows(df):return json.loads(df.to_json(orient='records',double_precision=14))

def main():
    study=json.loads((SITE/'data/study.json').read_text(encoding='utf-8'))
    q=study['protocol']['q168'];pred=[];hist=[];check=[]
    sources=[DERIVED/'benchmark_m4_top10_baselines.csv',DERIVED/'contrastes_m4_top10.csv']
    for seed in range(101,111):
        f=RUN/f'colombia/prueba/M4_N_semilla_{seed}.npz';sources.append(f)
        with np.load(f,allow_pickle=False) as z:
            if not pred:y=z['y'].astype(float);dates=z['dia'].copy()
            assert np.array_equal(z['y'],y) and np.array_equal(z['dia'],dates)
            pred.append(z['pred'].astype(float))
        folder=RUN/f'colombia/final/M4_N/semilla_{seed}'
        h=pd.read_csv(folder/'historia.csv');h['semilla']=seed
        hist.append(h);check.append(json.loads((folder/'TERMINADO.json').read_text(encoding='utf-8')))
        sources.extend([folder/'historia.csv',folder/'TERMINADO.json'])
    pred=np.stack(pred);err=pred-y[None];hist=pd.concat(hist,ignore_index=True);check=pd.DataFrame(check)
    assert pred.shape==(10,1461,24) and np.isfinite(pred).all()
    assert dates.tolist()==study['dates']
    observed=json.loads((SITE/'data/observed.json').read_text(encoding='utf-8'))
    assert np.allclose(y/1e6,observed['values'],atol=1e-8)
    table=pd.read_csv(DERIVED/'benchmark_m4_top10_baselines.csv')
    m=table.set_index('clave').loc['M4_N']
    assert np.isclose(np.abs(err).mean()/q,m.MASE,rtol=1e-12)
    assert np.isclose(np.sqrt(np.square(err).mean(axis=(1,2))).mean(),m.RMSE,rtol=1e-12)
    prediction=dict(mean=np.round(pred.mean(0)/1e6,8).tolist(),p10=np.round(np.quantile(pred,.1,axis=0)/1e6,8).tolist(),
        p90=np.round(np.quantile(pred,.9,axis=0)/1e6,8).tolist(),mae_daily=np.round(np.abs(err).mean(axis=(0,2)),6).tolist(),
        mse_daily=np.round(np.square(err).mean(2),3).tolist(),mae_hourly=np.round(np.abs(err).mean(0),3).tolist(),seeds=10)
    write(SITE/'data/predictions/m4_n.json',prediction)
    tests=pd.read_csv(DERIVED/'contrastes_m4_top10.csv')
    assert len(tests)==60 and tests.Comparador.nunique()==10
    catalog=json.loads((DERIVED/'catalogo_comparadores.json').read_text(encoding='utf-8'))
    byname={r['nombre']:r['clave'] for r in catalog}
    tests['comparator_id']=tests.Comparador.map(byname)
    assert tests.comparator_id.notna().all()
    table['clave']=table.clave.replace({'M4_N':'m4_n'})
    table['modelo']=table.clave.map({r['clave']:r['nombre'] for r in catalog}|{'m4_n':'M4ₙ'})
    # La fila agregada no recibe intervalos de un estimando distinto.
    result=dict(clave='m4_n',modelo='M4ₙ',MASE=float(m.MASE),MAE=float(m.MAE),RMSE=float(m.RMSE),
        MASE_sd=float(m.MASE_sd),semillas=10,IC95_inf=None,IC95_sup=None,entradas='numerico_ampliado')
    model=dict(id='m4_n',name='M4ₙ',family='m4_n',group='Propuesta',inputs='Numéricas ampliadas',params=60993,
        description='Memoria numérica de residuos sobre la combinación congelada de PatchTST, E y el experto periódico.',
        validation=float(check.mejor_perdida.mean()),validation_period='2021 · parada interna',validation_source='Parada interna',
        epochs=float(check.epocas.median()),training_seconds=float(check.segundos.median()),latency=None,
        source='./downloads/fichas/m4_n.md',seeds=10)
    ref=[]
    for f in sorted((ROOT/'results/m4_cierre_gpu_20261123/historicos_nbeats_timesnet/predicciones/TimesNet').glob('*.npz')):
        with np.load(f,allow_pickle=False) as z:
            assert np.array_equal(z['y'],y) and np.array_equal(z['dia'],dates)
            ref.append(z['yhat_lote1_kwh'].astype(float))
    ref=np.stack(ref);referr=ref-y[None];years=np.array([int(x[:4]) for x in dates])
    yearly=[dict(year=yy,mase=float(np.abs(err[:,years==yy]).mean()/q),reference_mase=float(np.abs(referr[:,years==yy]).mean()/q)) for yy in range(2022,2026)]
    interpretations={}
    n=nbformat.read(ROOT/'book/10_evaluacion.ipynb',4)
    for i,c in enumerate(n.cells):
        if c.metadata.get('integracion_m4') and c.cell_type=='code' and any('image/png' in o.get('data',{}) for o in c.outputs):
            if "'ganancia_top10.png'" in c.source:key='gain'
            elif "'curvas_aprendizaje.png'" in c.source:key='learning'
            elif "'observado_predicho.png'" in c.source:key='observed'
            elif "'semana_observado_predicho.png'" in c.source:key='week'
            elif "'horizonte_anio.png'" in c.source:key='segments'
            else:continue
            interpretations[key]=n.cells[i+1].source
    assert len(interpretations)==5
    assets=SITE/'figures/m4';assets.mkdir(parents=True,exist_ok=True)
    for name in ['observado_predicho.png','curvas_aprendizaje.png','ganancia_top10.png','horizonte_anio.png','semana_observado_predicho.png']:
        shutil.copy2(DERIVED/name,assets/name)
    for name in ['benchmark_m4_top10_baselines.csv','contrastes_m4_top10.csv','metricas_m4_semillas.csv','curvas_m4.csv','manifiesto.json']:
        shutil.copy2(DERIVED/name,SITE/'downloads'/name)
    curves=rows(hist.rename(columns={'perdida_ajuste':'train_loss','perdida_parada':'val_loss'}))
    data=dict(model=model,benchmark=result,comparison_benchmark=rows(table),comparators=catalog,tests=rows(tests),
        curves=curves,checkpoints=rows(check),yearly=yearly,horizon=dict(mae=np.abs(err).mean(axis=(0,1)).tolist(),reference_mae=np.abs(referr).mean(axis=(0,1)).tolist()),
        protocol=dict(q168=q,seeds=list(range(101,111)),origins=1461,horizon=24,bootstrap_replicates=49999,block_days=14,holm_family=1536,retrospective=True),
        interpretations=interpretations)
    write(SITE/'data/m4_results.json',data)
    write(SITE/'data/m4_results_manifest.json',dict(training_executed=False,neural_inference=False,published=False,
        seed_count=10,comparators=len(catalog),top10_comparisons=len(tests),source_sha256={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}))
    # Sincroniza únicamente la prosa actual de la ficha; conserva fórmulas y código.
    old='Las cinco semillas históricas son 101–105. La evaluación adicional registra su diseño antes de ejecutar nuevos experimentos.'
    new='La evaluación de M4<sub>N</sub> comprende diez semillas, 101–110, con configuración 6 y restauración del mejor checkpoint interno. Los resultados y las comparaciones frente al Top 10 se presentan en la evaluación integrada.'
    for name in ['theory/m4_n.fragment.html','theory/m4_n.html','downloads/fichas/m4_n.md']:
        p=SITE/name;s=p.read_text(encoding='utf-8');p.write_text(s.replace(old,new),encoding='utf-8',newline='\n')
    theory_manifest=SITE/'data/m4_teoria_manifest.json'
    metadata=json.loads(theory_manifest.read_text(encoding='utf-8'))
    metadata['alcance']='teoría y código; resultados en m4_results.json'
    metadata['sha256_fuentes']={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in metadata['sha256_fuentes']}
    write(theory_manifest,metadata)
    if (ROOT/'results/cierre_e3_sin_entrenar_20261124/estadistica.json').exists():
        from integrar_cierre_e3 import dashboard as exportar_cierre_e3
        exportar_cierre_e3()
    print(json.dumps({'M4_MASE':float(m.MASE),'seeds':10,'comparisons':len(tests),'training':False,'published':False}))

if __name__=='__main__':main()

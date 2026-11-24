from pathlib import Path
import sys,json,time,numpy as np,pandas as pd
R=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(R/'dashboard'))
import runtime_inferencia as L
rt=L.get_runtime();rows=[]
for key in L.MODELS:
    t=time.perf_counter()
    try:
        result=rt.predict(key,'2023-07-01')
        if key=='m4_n':
            with np.load(R/'results/m4_cierre_gpu_20261123/reducido_B/colombia/prueba/M4_N_semilla_101.npz',allow_pickle=False) as z:
                expected=z['pred'][z['dia']=='2023-07-01'][0]
        elif key in ['mlp','lstm','cnn1d','arx_lineal']:
            a=next(x for x in rt.baselines[key]['artefactos'] if x['semilla']==1000)
            with np.load(rt.snapshot/a['predicciones'],allow_pickle=False) as z:
                dates=pd.to_datetime(z['origen'],unit='ns',utc=True).tz_convert('America/Bogota')
                expected=z['yhat'][dates.strftime('%Y-%m-%d')=='2023-07-01'][0]
        elif key=='nuevo_timesnet__N':
            with np.load(R/'results/reevaluacion_timesnet_lote1_20261120/predicciones_1000.npz',allow_pickle=False) as z:
                expected=z['yhat_lote1_kwh'][z['origen']=='2023-07-01'][0]
        elif key=='nuevo_nbeats__N':
            with np.load(R/'results/analisis_nbeats_timesnet_20261120/entrada_resultados/resultados_e2_nbeats_timesnet/nuevo_nbeats/final/N/semilla_1000/pred_prueba.npz',allow_pickle=False) as z:
                expected=z['yhat_kwh'][z['origen']=='2023-07-01'][0]
        else:
            m=next(m for m in rt.registry if m['experimento']+'__'+m['variante']==key and m['semilla']==1000)
            expected=L.IA.guardadas(m)[pd.date_range('2022-01-01','2025-12-31').strftime('%Y-%m-%d')=='2023-07-01'][0]
        delta=float(np.max(np.abs(np.asarray(result['prediction_gwh'])*1e6-expected)))
        relative=float(np.max(np.abs(np.asarray(result['prediction_gwh'])*1e6-expected)/np.maximum(np.abs(expected),1)))
        row=dict(model=key,ok=relative<.001,max_diff_kwh=delta,max_relative=relative,seconds=time.perf_counter()-t)
    except Exception as e:
        import traceback;traceback.print_exc();row=dict(model=key,ok=False,error=str(e))
    rows.append(row);print(json.dumps(row,ensure_ascii=False),flush=True)
    (R/'results/revision_e3_integral_20261123/paridad_inferencia_vivo.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
print('PASS',sum(x['ok'] for x in rows),'/',len(rows))

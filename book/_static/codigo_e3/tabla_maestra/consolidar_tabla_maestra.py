"""Tabla única de rendimiento y coste desde resultados y pesos conservados."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/tabla_maestra_20261125'
Q=215617.90361589016

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    with Path(p).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,value):p.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2)+'\n',encoding='utf-8')

def consolidar():
    sources=[ROOT/p for p in ['dashboard/site/data/study.json','dashboard/site/data/m4_results.json',
        'results/evaluacion_e2/benchmark.csv','results/evaluacion_e2/metadatos_procedimientos.csv',
        'results/sensibilidad_temporal_m4_20261125/contrastes_sensibilidad.csv',
        'results/sensibilidad_temporal_m4_20261125/bootstrap_bloque_14.npz']]
    study=read(sources[0]);m4=read(sources[1]);benchmark=pd.read_csv(sources[2]).set_index('clave')
    metadata=pd.read_csv(sources[3]);tests=pd.read_csv(sources[4],converters={'p':float,'p_Holm':float})
    models={m['id']:m for m in study['models']}
    models['m4_n']=dict(id='m4_n',name='M4ₙ',group='Propuesta',inputs='Numéricas ampliadas')
    for key in ['ingenuo_168h','ingenuo_24h']:
        models[key]=dict(id=key,name=benchmark.loc[key,'modelo'],group='Referencias ingenuas',inputs='Demanda histórica')
    with np.load(sources[5],allow_pickle=False)as z:
        ci=np.quantile(z['medias'],[.025,.975],axis=0);ci_by={k:ci[:,i]for i,k in enumerate(z['claves'])}
    records=[];costdates=[];timings=[];weights={};trace=[]
    for key,model in models.items():
        cf=OUT/'costes'/f'{key.replace("+","_")}.json';cost=read(cf);sources.append(cf)
        raw=OUT/'costes'/f'{key.replace("+","_")}_tiempos.json';timings.extend(read(raw));sources.append(raw)
        b=m4['benchmark']if key=='m4_n'else benchmark.loc[key].to_dict()
        learned=key not in ['ingenuo_168h','ingenuo_24h']
        dev=float(np.mean([c['mejor_perdida']for c in m4['checkpoints']]))if key=='m4_n'else model.get('validation')
        dev_source='Parada interna'if key=='m4_n'else'Validación de selección'if learned else'No ajustado'
        period='2021'if key=='m4_n'else model.get('validation_period','—')
        selected=tests.loc[tests.comparator_id.eq(key)&tests.margen_relativo.eq(0)]
        assert len(selected)==(0 if key=='m4_n'else 9)
        pvalues={method:float(g.p_Holm.max())for method,g in selected.groupby('metodo')}
        train_sources=[]
        if key=='m4_n':
            base=ROOT/'results/corrector_final_y_m1_gpu_20261122_150345/M1'
            train_sources=[base/f'expertos_finales/{name}/semilla_101/TERMINADO.json'for name in ['patchtst','E','periodica']]
            train_sources+=[base/'final/semilla_101/TERMINADO.json',ROOT/'results/m4_cierre_gpu_20261123/reducido_B/colombia/final/M4_N/semilla_101/TERMINADO.json']
            training=float(sum(read(p)['segundos']for p in train_sources));device='CUDA, según registros de componentes'
        elif learned:
            a=metadata.loc[metadata.clave.eq(key)&metadata.semilla.eq(1000)]
            assert len(a)==1,key
            training=float(a.segundos.iloc[0]);device=str(a.dispositivo.iloc[0])
        else:training=0.;device='No ajustado'
        sources.extend(train_sources)
        for rel,h in cost['fuentes_sha256'].items():
            p=Path(rel);assert sha(p)==h,rel
            name=Path(__import__('os').path.relpath(p,ROOT)).as_posix()
            assert name not in weights or weights[name]==h
            weights[name]=h
        lo,hi=ci_by[key]
        r=dict(clave=key,modelo=model['name'],grupo=model['group'],entradas=model['inputs'],
            MASE_desarrollo=dev,fuente_desarrollo=dev_source,periodo_desarrollo=period,
            MASE_prueba=float(b['MASE']),MASE_DE_semillas=float(b['MASE_sd']),
            MASE_IC95_inf=float(lo),MASE_IC95_sup=float(hi),MAE_prueba_kWh=float(b['MAE']),RMSE_prueba_kWh=float(b['RMSE']),
            semillas_prueba=int(b['semillas']),p_conjunto_Holm=None if key=='m4_n'else max(pvalues.values()),
            p_bootstrap_Holm=pvalues.get('bootstrap_estacionario'),p_DM_HLN_Holm=pvalues.get('DM_HLN'),p_GW_Holm=pvalues.get('GW_bilateral'),
            parametros_pipeline=cost['parametros_pipeline'],parametros_MiB=cost['bytes_parametros']/2**20,
            latencia_mediana_ms=cost['mediana_ms'],latencia_p95_ms=cost['p95_ms'],RSS_MiB=cost['RSS_max_observado_bytes']/2**20,
            RSS_incremento_inferencia_MiB=cost['RSS_incremento_inferencia_bytes']/2**20,
            semilla_coste=cost['semilla_coste'],mediciones_coste=cost['mediciones'],
            ajuste_final_registrado_s=training,dispositivo_ajuste=device)
        assert abs(r['MAE_prueba_kWh']/Q-r['MASE_prueba'])<1e-10,key
        assert lo<r['MASE_prueba']<hi,key
        if learned and key!='m4_n':assert cost['parametros_pipeline']==model['params'],key
        records.append(r)
        costdates.extend([dict(clave=key,semilla=cost['semilla_coste'],**a)for a in cost['por_origen']])
        trace.append(dict(clave=key,metricas='m4_results.json'if key=='m4_n'else'results/evaluacion_e2/benchmark.csv',
            desarrollo='Media de diez mejores pérdidas internas M4 de 2021'if key=='m4_n'else'dashboard/site/data/study.json',
            coste=cf.relative_to(ROOT).as_posix(),ajuste_final=';'.join(p.relative_to(ROOT).as_posix()for p in train_sources)if train_sources else'results/evaluacion_e2/metadatos_procedimientos.csv'if learned else'No ajustado'))
    frame=pd.DataFrame(records).sort_values('MASE_prueba').reset_index(drop=True)
    assert len(frame)==18 and frame.clave.is_unique and frame.grupo.eq('Top 10').sum()==10
    frame.insert(0,'orden_descriptivo',np.arange(1,len(frame)+1))
    frame.to_csv(OUT/'tabla_maestra.csv',index=False)
    pd.DataFrame(costdates).to_csv(OUT/'coste_por_origen.csv',index=False)
    pd.DataFrame(timings).to_csv(OUT/'tiempos_inferencia.csv',index=False)
    pd.DataFrame(trace).to_csv(OUT/'trazabilidad.csv',index=False)
    write(OUT/'pesos_y_escaladores_sha256.json',weights)
    protocol=dict(fecha='2026-11-25',procedimientos=18,top10=10,baselines=4,control_numerico=1,propuesta=1,referencias_ingenuas=2,
        periodo_prueba=['2022-01-01','2025-12-31'],origenes=1461,horizontes=24,Q_kWh=Q,
        rendimiento='MASE y MAE: promedio de errores individuales de semillas; RMSE: media de RMSE individuales conservados. No puntuar el ensamble medio como si fuera la misma métrica.',
        desarrollo='Validación de selección de cada arquitectura y sus años; M4 utiliza media de mejores pérdidas internas de 2021 de diez semillas. No ordenar estos valores como validación homogénea.',
        IC_MASE='Percentiles puntuales del bootstrap estacionario pareado, 49.999 réplicas, bloque medio14 días, condicionado a semillas conservadas; misma familia de remuestreo para18procedimientos.',
        significancia='p conjunto es el máximo de bootstrap unilateral, DM–HLN unilateral y GW bilateral en las configuraciones14/13,28/27,56/55, margen0, ya ajustados conjuntamente porHolm(306). No es un nuevo test ómnibus; se exige rechazo en los nueve contrastes para resumir respaldo estable. M4 no se compara consigo mismo.',
        coste=read(OUT/'protocolo_coste.json'),
        tiempo_ajuste='Registro del ajuste final de la semilla de coste:1000 por comparador,101 M4. M4 suma tres expertos, M1 y memoria; excluye búsqueda y generación de bases OOF. Hardware de origen heterogéneo: no usar estos segundos como comparación controlada de entrenamiento.',
        entrenamiento=False,publicacion=False)
    write(OUT/'protocolo.json',protocol)
    m=frame.set_index('clave').loc['m4_n'];tn=frame.set_index('clave').loc['nuevo_timesnet__N']
    eligible=frame.loc[~frame.grupo.eq('Referencias ingenuas')]
    bestcost=eligible.sort_values('latencia_mediana_ms').iloc[0]
    summary=dict(procedimientos=len(frame),semillas_prueba=int(frame.semillas_prueba.sum()),mediciones=len(timings),
        M4_MASE=float(m.MASE_prueba),M4_latencia_ms=float(m.latencia_mediana_ms),TimesNet_latencia_ms=float(tn.latencia_mediana_ms),
        reduccion_MASE_TimesNet_pct=float(100*(1-m.MASE_prueba/tn.MASE_prueba)),ratio_latencia_M4_TimesNet=float(m.latencia_mediana_ms/tn.latencia_mediana_ms),
        parametros_M4=int(m.parametros_pipeline),procedimiento_aprendido_menor_latencia=bestcost['modelo'],
        rechazo_estable_comparadores=int(frame.p_conjunto_Holm.lt(.05).sum()),entrenamiento=False,publicacion=False)
    write(OUT/'resumen.json',summary)
    write(OUT/'verificacion.json',dict(fuentes_sha256={p.relative_to(ROOT).as_posix():sha(p)for p in sources},
        pesos_verificados=len(weights),paridad_Q=True,procedimientos_completos=True,mediciones_por_procedimiento=300,
        resultados_sha256={p.name:sha(p)for p in OUT.glob('*.csv')},entrenamiento=False,publicacion=False))
    print(frame[['modelo','MASE_prueba','latencia_mediana_ms','parametros_pipeline','RSS_MiB']].to_string(index=False))

if __name__=='__main__':consolidar()

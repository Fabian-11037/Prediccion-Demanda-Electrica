"""Intervalos comunes desde predicciones guardadas; no importa redes."""
from __future__ import annotations
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
ROOT=Path(__file__).resolve().parents[1]
OUT='incertidumbre_comun_20261125'
LEVELS=np.array([.8,.9,.95])
METHODS=['fijo_2022','movil_365']
SCOPES=['horario','trayectoria24']
Q=215617.90361589016
B=9999

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    with Path(p).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,obj):p.write_text(json.dumps(obj,ensure_ascii=False,allow_nan=False,indent=2)+'\n',encoding='utf-8')

def predictions(root):
    sources=[]
    def src(rel):p=root/rel;sources.append(p);return p
    master=pd.read_csv(src('results/tabla_maestra_20261125/tabla_maestra.csv')).set_index('clave')
    dates=pd.date_range('2022-01-01','2025-12-31',tz='America/Bogota',freq='D')
    days=dates.strftime('%Y-%m-%d').to_numpy();ns=dates.as_unit('ns').asi8
    parts=[];truth=None
    for seed in range(101,111):
        p=src(f'results/m4_cierre_gpu_20261123/reducido_B/colombia/prueba/M4_N_semilla_{seed}.npz')
        with np.load(p,allow_pickle=False)as z:
            if truth is None:truth=z['y'].astype(float)
            assert np.array_equal(z['dia'],days)and np.array_equal(z['y'],truth)
            parts.append(z['pred'].astype(float))
    models={'m4_n':np.stack(parts)}
    frame=pd.read_parquet(src('results/evaluacion_e2/entradas/consolidado/predicciones_finales.parquet'))
    for (exp,var),g in frame.groupby(['experimento','variante'],sort=False):
        key=exp+'__'+var
        if key not in master.index:continue
        parts=[]
        for seed,q in g.groupby('semilla',sort=True):
            assert seed in range(1000,1005)
            q=q.sort_values(['origen','horizonte'])
            assert np.array_equal(pd.DatetimeIndex(q.origen.drop_duplicates()).as_unit('ns').asi8,ns)
            assert np.array_equal(q.horizonte,np.tile(np.arange(1,25),1461))
            assert np.array_equal(q.observado_kwh.to_numpy().reshape(1461,24),truth)
            parts.append(q.predicho_kwh.to_numpy().reshape(1461,24))
        models[key]=np.stack(parts)
    snap=root/'results/repositorio/d1ae9a4e09dd1dead4af1329f16aa3e8d0f2c3d3'
    registry=read(src(snap.relative_to(root)/'results/consolidado/modelos.json'))
    for key in ['mlp','lstm','cnn1d','arx_lineal']:
        parts=[]
        for item in sorted(registry[key]['artefactos'],key=lambda r:r['semilla']):
            p=src(snap.relative_to(root)/item['predicciones'])
            with np.load(p,allow_pickle=False)as z:
                assert np.array_equal(z['origen'],ns)and np.allclose(z['y'],truth,rtol=0,atol=1e-6)
                parts.append(z['yhat'].astype(float))
        models[key]=np.stack(parts)
    panel=pd.read_parquet(src('data/processed/panel_horario.parquet'),columns=['y_demanda_real_kwh'])
    raw=panel.y_demanda_real_kwh.to_numpy();pos=panel.index.get_indexer(dates);hours=np.arange(24)
    models['ingenuo_168h']=raw[pos[:,None]+hours-168][None]
    lag=24*np.maximum(1,np.ceil((hours+2)/24).astype(int))
    models['ingenuo_24h']=raw[pos[:,None]+hours-lag][None]
    assert set(models)==set(master.index)and len(models)==18
    for key,p in models.items():
        assert p.shape==(int(master.loc[key,'semillas_prueba']),1461,24)and np.isfinite(p).all()
        assert abs(np.abs(p-truth).mean()/Q-master.loc[key,'MASE_prueba'])<2e-6,key
    return models,truth,dates,master,sources

def finite_quantiles(scores):
    """Estadísticos de orden ceil((n+1)*nivel), sobre días del eje1."""
    n=scores.shape[1];ranks=np.ceil((n+1)*LEVELS).astype(int)
    assert np.all((ranks>=1)&(ranks<=n))
    return np.take(np.partition(scores,ranks-1,axis=1),ranks-1,axis=1)

def stationary_counts(n):
    rng=np.random.default_rng(20261125);indices=np.empty((B,n),dtype=np.int32)
    indices[:,0]=rng.integers(n,size=B)
    for t in range(1,n):indices[:,t]=np.where(rng.random(B)<1/56,rng.integers(n,size=B),(indices[:,t-1]+1)%n)
    counts=np.zeros((B,n),dtype=np.float64)
    for start in range(0,B,100):
        ix=indices[start:start+100];row=np.repeat(np.arange(len(ix)),n)
        local=np.zeros((len(ix),n));np.add.at(local,(row,ix.ravel()),1)
        counts[start:start+len(ix)]=local/n
    assert np.allclose(counts.sum(1),1)
    return counts

def calculate(root=ROOT):
    root=Path(root);out=root/'results'/OUT;out.mkdir(parents=True,exist_ok=True)
    protocol=dict(fecha='2026-11-25',calibracion=['2022-01-01','2022-12-30'],dias_calibracion_inicial=364,
        evaluacion=['2023-01-01','2025-12-31'],dias_evaluacion=1096,origen='00:00 America/Bogota',
        disponibilidad='Solo errores de días s≤t−2; el 31-dic-2022 entra en la ventana móvil desde2-ene-2023.',
        metodos=METHODS,principal='movil_365',ventana='Últimos365 errores diarios disponibles;364 en el primer origen',
        centro='Media de predicciones guardadas:10semillas M4,5por comparador aprendido y1por ingenuo; no es la media de pérdidas del benchmark puntual.',
        niveles=LEVELS.tolist(),alcances=SCOPES,
        horario='Cuantil separado de error absoluto por horizonte; rango ceil((n+1)*nivel).',
        trayectoria='Cuantil del máximo diario de |error_h|/escala_h; escala_h=mediana de |error_h| en reserva2022, piso1e-6GWh, congelada. Radio_h=cuantil×escala_h.',
        incertidumbre='Cobertura empírica bajo dependencia temporal; no se supone intercambiabilidad ni se certifica cobertura exacta.',
        puntuaciones='IS_alpha=ancho+2/alpha×distancia fuera del intervalo; WIS=(0.5×error absoluto+sum(alpha/2×IS_alpha))/(3+0.5). WIS solo para intervalos marginales horarios.',
        bootstrap=dict(replicas=B,bloque_medio_dias=56,semilla=20261125,unidad='día;24h agregadas antes de remuestrear',IC='Percentiles95% puntuales, condicionados a calibración, pesos y semillas conservados'),
        contrastes='WIS diario del método móvil horario: M4 frente a17comparadores, bootstrap unilateral y DM–HLN HAC55,h=1. Holm sobre34, margen0. No modifica inferencia deMASE.',
        seleccion=False,entrenamiento=False,inferencia_neuronal=False,publicacion=False,
        referencias=['https://arxiv.org/abs/2005.12881','https://arxiv.org/abs/2107.07511'])
    pp=out/'protocolo.json'
    if pp.exists():assert read(pp)==protocol,'Se conserva el protocolo antes de recalcular.'
    else:write(pp,protocol)
    models,truth,dates,master,sources=predictions(root)
    keys=master.index.tolist();M=len(keys);ev=np.flatnonzero(dates.year>=2023);N=len(ev)
    cal=np.flatnonzero(dates<=pd.Timestamp('2022-12-30',tz='America/Bogota'))
    assert len(cal)==364 and N==1096 and cal.max()<=ev.min()-2
    center=np.stack([models[k].mean(0)for k in keys])/1e6;y=truth/1e6;errors=np.abs(center-y)
    scales=np.maximum(np.median(errors[:,cal],axis=1),1e-6)
    radii=np.empty((M,2,2,3,N,24),dtype=np.float64)
    audit=[]
    for a,method in enumerate(METHODS):
        for j,t in enumerate(ev):
            pool=cal if a==0 else np.arange(max(0,t-1-365),t-1)
            assert pool.max()<=t-2 and len(pool)>=364
            radii[:,a,0,:,j,:]=finite_quantiles(errors[:,pool])
            maxima=(errors[:,pool]/scales[:,None,:]).max(2)
            radii[:,a,1,:,j,:]=finite_quantiles(maxima)[:,:,None]*scales[:,None,:]
            audit.append(dict(metodo=method,origen=str(dates[t].date()),inicio=str(dates[pool[0]].date()),fin=str(dates[pool[-1]].date()),n=len(pool)))
        print(f'Intervalos {method}:18procedimientos,1096orígenes.',flush=True)
    assert np.isfinite(radii).all()and(radii>=0).all()
    assert np.all(np.diff(radii,axis=3)>=-1e-12)
    np.savez_compressed(out/'bandas.npz',radio_GWh=radii,centro_GWh=center[:,ev],observado_GWh=y[ev],dias=dates[ev].strftime('%Y-%m-%d').to_numpy(dtype='U10'),claves=np.array(keys),escala_h_GWh=scales,errores_absolutos_GWh=errors)
    pd.DataFrame(audit).to_csv(out/'auditoria_disponibilidad.csv',index=False)
    summaries=[];horizons=[];months=[];daily=[];columns=[];features=[];wis=np.empty((M,2,N))
    years=dates[ev].year.to_numpy();month_labels=dates[ev].strftime('%Y-%m').to_numpy()
    residual=errors[:,ev];cy=center[:,ev];ty=y[ev]
    for m,key in enumerate(keys):
        for a,method in enumerate(METHODS):
            wis_value=.5*residual[m]
            for s,scope in enumerate(SCOPES):
                for l,level in enumerate(LEVELS):
                    rad=radii[m,a,s,l];covered=residual[m]<=rad;alpha=1-level
                    scores=2*rad+2/alpha*np.maximum(residual[m]-rad,0)
                    if s==0:wis_value=wis_value+alpha/2*scores
                    metrics=dict(cobertura=covered.mean(1),cobertura24=covered.all(1).astype(float),ancho_GWh=(2*rad).mean(1))
                    if s==0:metrics['IS_GWh']=scores.mean(1)
                    record=dict(clave=key,modelo=master.loc[key,'modelo'],metodo=method,alcance=scope,nivel=float(level))
                    for metric,vals in metrics.items():columns.append((key,method,scope,float(level),metric));features.append(vals)
                    daily.append(pd.DataFrame(dict(**record,dia=dates[ev].strftime('%Y-%m-%d'),**metrics)))
                    for year in ['total',2023,2024,2025]:
                        mask=np.ones(N,dtype=bool)if year=='total'else years==year
                        summaries.append(dict(**record,segmento=str(year),n_dias=int(mask.sum()),**{k:float(v[mask].mean())for k,v in metrics.items()}))
                        for h in range(24):horizons.append(dict(**record,segmento=str(year),horizonte=h+1,n_dias=int(mask.sum()),cobertura=float(covered[mask,h].mean()),ancho_GWh=float((2*rad[mask,h]).mean())))
                    for month in np.unique(month_labels):
                        mask=month_labels==month
                        months.append(dict(**record,mes=month,n_dias=int(mask.sum()),**{k:float(v[mask].mean())for k,v in metrics.items()}))
            wis[m,a]=wis_value.mean(1)/3.5
            columns.append((key,method,'horario',0.,'WIS_GWh'));features.append(wis[m,a])
    print('Bootstrap temporal:9.999réplicas, bloque56días.',flush=True)
    weights=stationary_counts(N);boot=weights@np.stack(features,axis=1)
    ci=np.quantile(boot,[.025,.975],axis=0);indices={k:i for i,k in enumerate(columns)}
    result=pd.DataFrame(summaries)
    for i,r in result.loc[result.segmento.eq('total')].iterrows():
        for metric in ['cobertura','cobertura24','ancho_GWh','IS_GWh']:
            k=(r.clave,r.metodo,r.alcance,float(r.nivel),metric)
            if k in indices:
                j=indices[k];result.loc[i,metric+'_IC95_inf']=ci[0,j];result.loc[i,metric+'_IC95_sup']=ci[1,j]
    result.to_csv(out/'resumen.csv',index=False)
    pd.DataFrame(horizons).to_csv(out/'horizontes.csv',index=False)
    pd.DataFrame(months).to_csv(out/'mensual.csv',index=False)
    pd.concat(daily,ignore_index=True).to_parquet(out/'metricas_diarias.parquet',index=False)
    wr=[];wj=np.array([[indices[(k,method,'horario',0.,'WIS_GWh')]for method in METHODS]for k in keys]);wb=boot[:,wj]
    for m,key in enumerate(keys):
        for a,method in enumerate(METHODS):
            for year in ['total',2023,2024,2025]:
                mask=np.ones(N,dtype=bool)if year=='total'else years==year
                rec=dict(clave=key,modelo=master.loc[key,'modelo'],metodo=method,segmento=str(year),n_dias=int(mask.sum()),WIS_GWh=float(wis[m,a,mask].mean()))
                if year=='total':rec.update(IC95_inf=float(ci[0,wj[m,a]]),IC95_sup=float(ci[1,wj[m,a]]))
                wr.append(rec)
    pd.DataFrame(wr).to_csv(out/'wis.csv',index=False)
    mm=keys.index('m4_n');rows=[];comparisons=[]
    for m,key in enumerate(keys):
        if m==mm:continue
        d=wis[m,1]-wis[mm,1];mean=d.mean();c=d-mean
        cov=np.array([np.dot(c[h:],c[:N-h])/N for h in range(56)])
        lrv=cov[0]+2*np.dot(1-np.arange(1,56)/56,cov[1:]);assert lrv>0
        dm=float(mean/np.sqrt(lrv/N)*np.sqrt((N-1)/N));pdm=float(stats.t.sf(dm,N-1))
        bd=wb[:,m,1]-wb[:,mm,1];pb=float((1+np.count_nonzero(bd-mean>=mean))/(B+1))
        gain=1-wis[mm,1].mean()/wis[m,1].mean();gain_ci=np.quantile(1-wb[:,mm,1]/wb[:,m,1],[.025,.975])
        for method,p,meaning in [('bootstrap_estacionario',pb,'Evalúa si M4 reduce el WIS medio frente al comparador, remuestreando bloques diarios compartidos.'),('DM_HLN',pdm,'Contrasta menor WIS medio de M4 con covarianza HAC de55rezagos y correcciónHLN; prueba unilateral.')]:
            rows.append(dict(comparador=key,modelo=master.loc[key,'modelo'],contraste=method,significado=meaning,p=p,diferencia_WIS_GWh=float(mean),reduccion_pct=float(gain*100),IC95_inf_pct=float(gain_ci[0]*100),IC95_sup_pct=float(gain_ci[1]*100),n_dias=N))
    tests=pd.DataFrame(rows);tests['p_Holm']=multipletests(tests.p,method='holm')[1];tests['rechaza_005']=tests.p_Holm.lt(.05)
    tests.to_csv(out/'contrastes_WIS.csv',index=False)
    for key,g in tests.groupby('comparador',sort=False):
        r=g.iloc[0];pv=g.set_index('contraste').p_Holm
        comparisons.append(dict(comparador=key,modelo=r.modelo,reduccion_pct=r.reduccion_pct,IC95_inf_pct=r.IC95_inf_pct,IC95_sup_pct=r.IC95_sup_pct,p_bootstrap=float(pv['bootstrap_estacionario']),p_DM_HLN=float(pv['DM_HLN']),respaldo_menor_WIS=bool(g.rechaza_005.all()and r.diferencia_WIS_GWh>0)))
    pd.DataFrame(comparisons).to_csv(out/'comparaciones_WIS.csv',index=False)
    np.savez_compressed(out/'wis_y_bootstrap.npz',wis_diario_GWh=wis,bootstrap_WIS_GWh=wb,claves=np.array(keys))
    ranking=pd.DataFrame(wr).query('metodo=="movil_365" and segmento=="total"').sort_values('WIS_GWh')
    selected=result.query('metodo=="movil_365" and segmento=="total" and nivel==0.9')
    summary=dict(procedimientos=M,dias_evaluacion=N,dias_calibracion_inicial=len(cal),principal='movil_365',
        mejor_WIS=ranking.iloc[0].modelo,WIS_M4_GWh=float(wis[mm,1].mean()),
        WIS_TimesNet_GWh=float(wis[keys.index('nuevo_timesnet__N'),1].mean()),
        comparadores_con_menor_WIS_M4=int(sum(r['respaldo_menor_WIS']for r in comparisons)),familia_Holm=34,
        M4_90=selected.loc[selected.clave.eq('m4_n')].astype(object).where(selected.loc[selected.clave.eq('m4_n')].notna(),None).to_dict('records'),
        calibracion_inicial=['2022-01-01','2022-12-30'],evaluacion=['2023-01-01','2025-12-31'],
        entrenamiento=False,inferencia_neuronal=False,publicacion=False)
    write(out/'resumen.json',summary)
    write(out/'verificacion.json',dict(origenes_completos_unicos=True,paridad_benchmark=True,calibracion_causal=True,
        banda_anidada=True,procedimientos=M,origenes=N,fuentes_sha256={p.relative_to(root).as_posix():sha(p)for p in sources},
        resultados_sha256={p.name:sha(p)for p in out.iterdir()if p.suffix in ['.csv','.parquet','.npz']},entrenamiento=False,publicacion=False))
    print(ranking[['modelo','WIS_GWh']].to_string(index=False));print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':calculate()

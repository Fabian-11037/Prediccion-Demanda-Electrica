"""Sensibilidad a dependencia prolongada desde pérdidas diarias guardadas."""
from __future__ import annotations
import hashlib,json,time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.diagnostic import acorr_ljungbox

ROOT=Path(__file__).resolve().parents[1]
OUT_NAME='sensibilidad_temporal_m4_20261125'
Q=215617.90361589016
CONFIGS=[(14,13),(28,27),(56,55)]
REPLICAS=49999
SEED=20261125


def calcular(root=ROOT):
    root=Path(root);out=root/'results'/OUT_NAME;out.mkdir(parents=True,exist_ok=True)
    sources=[root/'results/cierre_e3_sin_entrenar_20261124/perdidas_diarias.csv',
        root/'results/evaluacion_e2/perdidas_y_bootstrap.npz',
        root/'results/evaluacion_e2/benchmark.csv',root/'dashboard/site/data/study.json']
    losses=pd.read_csv(sources[0]);bench=pd.read_csv(sources[2]).set_index('clave')
    with np.load(sources[1],allow_pickle=False)as z:
        days=pd.to_datetime(z['origen'],utc=True).tz_convert('America/Bogota')
        original=pd.DataFrame(z['loss'],columns=z['nombres'])
    original=original.rename(columns={r.modelo:k for k,r in bench.iterrows()})
    assert set(losses)==set(bench.index)|{'m4_n'}
    assert np.allclose(losses[original.columns],original,rtol=1e-12,atol=1e-12)
    dates=days.strftime('%Y-%m-%d').to_numpy();N=len(days)
    assert N==1461 and days.is_unique and np.all(np.diff(days.asi8)==86400*10**9)
    assert dates[0]=='2022-01-01' and dates[-1]=='2025-12-31'
    raw=[]
    for seed in range(101,111):
        p=root/f'results/m4_cierre_gpu_20261123/reducido_B/colombia/prueba/M4_N_semilla_{seed}.npz';sources.append(p)
        with np.load(p,allow_pickle=False)as z:
            assert np.array_equal(z['dia'],dates)
            raw.append(np.abs(z['pred'].astype(float)-z['y']).mean(1)/Q)
    assert np.allclose(np.mean(raw,axis=0),losses.m4_n,rtol=1e-12,atol=1e-12)
    for key in bench.index:assert abs(losses[key].mean()-bench.loc[key,'MASE'])<2e-6
    assert abs(losses.m4_n.mean()-.5770369087419158)<1e-12
    study=json.loads(sources[3].read_text(encoding='utf-8'))
    top={m['id']for m in study['models']if m['group']=='Top 10'}
    keys=list(bench.index);allkeys=keys+['m4_n'];matrix=losses[allkeys].to_numpy()
    assert matrix.shape==(1461,18)and np.isfinite(matrix).all()
    methods={
        'bootstrap_estacionario':'Evalúa si la reducción media supera el margen indicado, remuestreando bloques de días compartidos.',
        'DM_HLN':'Contrasta una mejora media del MASE con varianza HAC y corrección de muestra finita HLN; es unilateral.',
        'GW_bilateral':'Contrasta igualdad condicional de pérdidas usando una constante y el diferencial observable de d−2; es bilateral y no determina por sí solo la dirección.'}
    rows=[];diagnostics=[];gain_intervals=[]
    names=bench.modelo.to_dict();start=time.perf_counter()
    for block,hac in CONFIGS:
        print(f'Bootstrap estacionario: bloque medio {block} días, {REPLICAS} réplicas.',flush=True)
        rng=np.random.default_rng(SEED)
        ix=np.empty((REPLICAS,N),dtype=np.int32);ix[:,0]=rng.integers(N,size=REPLICAS)
        # Mismos números aleatorios de reinicio/posición entre configuraciones.
        for t in range(1,N):
            ix[:,t]=np.where(rng.random(REPLICAS)<1/block,rng.integers(N,size=REPLICAS),(ix[:,t-1]+1)%N)
        boot=np.empty((REPLICAS,len(allkeys)))
        for j in range(0,REPLICAS,128):boot[j:j+128]=matrix[ix[j:j+128]].mean(1)
        del ix
        np.savez_compressed(out/f'bootstrap_bloque_{block}.npz',medias=boot,claves=np.array(allkeys))
        bm=boot[:,-1];model=matrix[:,-1]
        for i,key in enumerate(keys):
            ref=matrix[:,i];br=boot[:,i];gain=1-model.mean()/ref.mean()
            gain_ci=np.quantile(1-bm/br,[.025,.975])
            gain_intervals.append(dict(comparator_id=key,Comparador=names[key],bloque_dias=block,
                lag_HAC=hac,ganancia_pct=100*gain,ic95_inf_pct=100*gain_ci[0],ic95_sup_pct=100*gain_ci[1]))
            for margin in [0.,.03]:
                d=(1-margin)*ref-model;mean=d.mean();c=d-mean
                gamma=np.array([np.dot(c[h:],c[:N-h])/N for h in range(hac+1)])
                lrv=float(gamma[0]+2*np.dot(1-np.arange(1,hac+1)/(hac+1),gamma[1:]))
                assert lrv>0,(key,block,margin)
                dm=float(mean/np.sqrt(lrv/N)*np.sqrt((N-1)/N))
                pdm=float(stats.t.sf(dm,N-1))
                bd=(1-margin)*br-bm
                pboot=float((1+np.count_nonzero(bd-mean>=mean))/(REPLICAS+1))
                X=np.column_stack([np.ones(N),np.r_[np.zeros(2),d[:-2]]])
                moments=X*d[:,None];avg=moments.mean(0);centered=moments-avg
                cov=centered.T@centered/N
                for h in range(1,hac+1):
                    ch=centered[h:].T@centered[:-h]/N
                    cov+=(1-h/(hac+1))*(ch+ch.T)
                assert np.linalg.eigvalsh(cov).min()>-1e-12
                gw=float(N*avg@np.linalg.pinv(cov)@avg);pgw=float(stats.chi2.sf(gw,2))
                ci_d=np.quantile(bd,[.025,.975])
                for method,statistic,p,null in [
                    ('bootstrap_estacionario',float(mean),pboot,'E[(1-m)L_ref−L_M4] ≤ 0'),
                    ('DM_HLN',dm,pdm,'E[(1-m)L_ref−L_M4] ≤ 0'),
                    ('GW_bilateral',gw,pgw,'E[z_t((1-m)L_ref−L_M4)] = 0')]:
                    rows.append(dict(comparator_id=key,Comparador=names[key],grupo='Top 10'if key in top else'Baseline / control',
                        bloque_dias=block,lag_HAC=hac,margen_relativo=margin,metodo=method,
                        significado=methods[method],estadistico=statistic,p=p,hipotesis_nula=null,
                        diferencia_ajustada_MASE=float(mean),diferencia_ic95_inf=float(ci_d[0]),diferencia_ic95_sup=float(ci_d[1]),
                        ganancia_relativa=float(gain),ic95_inf=float(gain_ci[0]),ic95_sup=float(gain_ci[1]),
                        MASE_M4=float(model.mean()),MASE_ref=float(ref.mean()),n=N,
                        replicas_bootstrap=REPLICAS,familia='sensibilidad_temporal_306'))
                lb=acorr_ljungbox(d,lags=[14,28,56],return_df=True)
                for lag,v in lb.iterrows():diagnostics.append(dict(comparator_id=key,Comparador=names[key],
                    margen_relativo=margin,rezago=int(lag),estadistico=float(v.lb_stat),p=float(v.lb_pvalue)))
        print(f'Configuración {block}/{hac} terminada; {time.perf_counter()-start:.1f} s acumulados.',flush=True)
    df=pd.DataFrame(rows);assert len(df)==306
    df['p_Holm']=multipletests(df.p,method='holm')[1]
    df['rechaza_005']=df.p_Holm.lt(.05)
    df.to_csv(out/'contrastes_sensibilidad.csv',index=False)
    pd.DataFrame(gain_intervals).to_csv(out/'intervalos_ganancia.csv',index=False)
    # Los diagnósticos de diferencial no dependen de la configuración HAC/bootstrap.
    diag=pd.DataFrame(diagnostics).drop_duplicates(['comparator_id','margen_relativo','rezago'])
    assert len(diag)==102
    diag['p_Holm']=multipletests(diag.p,method='holm')[1]
    diag.to_csv(out/'diagnosticos_diferenciales.csv',index=False)
    verdicts=[]
    for (key,block,margin),g in df.groupby(['comparator_id','bloque_dias','margen_relativo'],sort=False):
        assert set(g.metodo)==set(methods)
        common=g.iloc[0];pv=g.set_index('metodo').p_Holm
        verdicts.append(dict(comparator_id=key,Comparador=common.Comparador,grupo=common.grupo,
            bloque_dias=int(block),lag_HAC=int(common.lag_HAC),margen_relativo=float(margin),
            ganancia_pct=float(100*common.ganancia_relativa),ic95_inf_pct=float(100*common.ic95_inf),
            ic95_sup_pct=float(100*common.ic95_sup),p_bootstrap=float(pv['bootstrap_estacionario']),
            p_DM_HLN=float(pv['DM_HLN']),p_GW_bilateral=float(pv['GW_bilateral']),
            p_max_ajustado=float(pv.max()),mejora_media_respaldada=bool(pv['bootstrap_estacionario']<.05 and pv['DM_HLN']<.05),
            respaldo_tres_contrastes=bool(g.rechaza_005.all() and common.ganancia_relativa>margin)))
    verdict=pd.DataFrame(verdicts);assert len(verdict)==102
    verdict.to_csv(out/'resumen_comparaciones.csv',index=False)
    times=verdict.query('comparator_id=="nuevo_timesnet__N"').copy()
    summary=dict(periodo=['2022-01-01','2025-12-31'],dias=N,comparadores=17,modelos_top10=10,
        semillas_M4=10,semillas_comparadores_aprendidos=5,referencias_deterministas=2,
        MASE_M4=float(losses.m4_n.mean()),MASE_TimesNet=float(losses['nuevo_timesnet__N'].mean()),
        ganancia_TimesNet_pct=float(times.ganancia_pct.iloc[0]),
        configuraciones=[dict(bloque_dias=b,lag_HAC=h)for b,h in CONFIGS],
        familia_Holm=len(df),replicas=REPLICAS,semilla=SEED,contrastes_rechazados=int(df.rechaza_005.sum()),
        comparaciones_respaldadas=int(verdict.respaldo_tres_contrastes.sum()),comparaciones=len(verdict),
        comparaciones_top10_respaldadas=int(verdict.loc[verdict.grupo.eq('Top 10'),'respaldo_tres_contrastes'].sum()),
        comparaciones_top10=int(verdict.grupo.eq('Top 10').sum()),
        TimesNet=times.astype(object).where(times.notna(),None).to_dict(orient='records'),
        entrenamiento=False,modificacion_resultados_historicos=False)
    protocol=dict(**{k:summary[k]for k in ['periodo','dias','configuraciones','familia_Holm','replicas','semilla']},
        fecha='2026-11-25',unidad='origen diario; pérdidas individuales promedio entre semillas',
        metrica='MAE / Q de entrenamiento',Q_kWh=Q,margenes=[0,.03],
        bootstrap='Estacionario pareado; medias de bloque14/28/56; mismos días para todos los procedimientos',
        intervalos='Percentiles95% puntuales de la reducción relativa, no intervalos simultáneos',
        DM='Unilateral, Bartlett HAC13/27/55, h efectivo HLN=1 origen, referencia t con N−1 grados de libertad',
        GW='Bilateral χ²2, HAC Bartlett13/27/55, instrumentos constante y diferencial disponible de d−2',
        criterio='Ambas pruebas unilaterales respaldan el margen; respaldo de tres exige además rechazo GW bilateral y dirección favorable observada',
        control_multiple='Holm único sobre17comparadores×2márgenes×3métodos×3configuraciones=306',
        diagnosticos='Ljung–Box de los diferenciales; Holm separado sobre102 diagnósticos, no sobre residuos del modelo',
        fuente_semillas='Diez realizaciones M4, cinco por comparador aprendido; no emparejar identificadores de semillas',
        alcance='Sensibilidad temporal condicionada a pesos y semillas conservados; no cambia selección ni configura modelos',
        entrenamiento=False,inferencia_neuronal=False,publicacion=False)
    hashes={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()for p in sources}
    for name,obj in [('resumen.json',summary),('protocolo.json',protocol),('verificacion.json',dict(
        fechas_completas_unicas=True,paridad_benchmark=True,paridad_perdidas_M4=True,
        n_hipotesis=306,n_resumenes=102,fuentes_sha256=hashes,
        resultados_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in out.glob('*.csv')}))]:
        (out/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':calcular()

"""Dispersión, residuos y segmentos de M4 a partir de predicciones guardadas."""
from __future__ import annotations
import hashlib
import json
import warnings
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.stattools import acf, bds

SIGNIFICADOS_CONTRASTES={'Ljung–Box': 'Evalúa si los errores diarios conservan autocorrelación hasta el número de días indicado.', 'Ljung–Box cuadrados': 'Evalúa si la magnitud de los errores presenta dependencia temporal; usa residuos centrados al cuadrado.', 'ARCH–LM': 'Busca dependencia de la varianza del error respecto a errores pasados al cuadrado, hasta 14 días.', 'BDS': 'Evalúa la hipótesis de residuos independientes e idénticamente distribuidos; su rechazo no identifica por sí solo no linealidad.', 'Jarque–Bera': 'Evalúa si la asimetría y la curtosis de los residuos son compatibles con una distribución normal.', 'Mincer–Zarnowitz': 'Evalúa conjuntamente intercepto cero y pendiente uno al relacionar observado con predicción media diaria, usando covarianza HAC13.', 'Ljung–Box por horizonte': 'Evalúa la autocorrelación entre días de los errores del mismo horizonte, hasta 14 días.'}

ROOT=Path(__file__).resolve().parents[1]
OUT_NAME='diagnosticos_m4_20261124'
Q=215617.90361589016


def calcular(root=ROOT):
    root=Path(root);out=root/'results'/OUT_NAME;out.mkdir(parents=True,exist_ok=True)
    sources=[]
    def source(rel):
        p=root/rel;sources.append(p);return p
    arrays=[];truth=None
    for seed in range(101,111):
        p=source(f'results/m4_cierre_gpu_20261123/reducido_B/colombia/prueba/M4_N_semilla_{seed}.npz')
        with np.load(p,allow_pickle=False)as z:
            if truth is None:truth=z['y'].astype(float);days=z['dia'].copy()
            assert np.array_equal(days,z['dia']) and np.array_equal(truth,z['y'])
            arrays.append(z['pred'].astype(float))
    P=np.stack(arrays);T,H=truth.shape
    dates=pd.DatetimeIndex(pd.to_datetime(days)).tz_localize('America/Bogota')
    assert P.shape==(10,1461,24) and dates.is_unique and dates.is_monotonic_increasing
    assert np.array_equal(days,pd.date_range('2022-01-01','2025-12-31').strftime('%Y-%m-%d'))
    assert np.isfinite(P).all() and np.isfinite(truth).all()
    center=P.mean(0);spread=P.std(0,ddof=1);variance=spread**2
    residual=truth-center;abs_error=np.abs(P-truth);ensemble_abs=np.abs(residual)
    cv=100*spread/np.abs(center);lo,hi=np.quantile(P,[.1,.9],axis=0)
    daily_spread=spread.mean(1);daily_ae=ensemble_abs.mean(1);daily_e=residual.mean(1)
    # Identidad exacta de la descomposición del MSE, distinta del MAE primario.
    mse_ind=np.mean((P-truth)**2);mse_center=np.mean(residual**2)
    initialization_mse=np.mean((P-center)**2)
    assert np.isclose(mse_ind,mse_center+initialization_mse,rtol=1e-12)
    assert abs(abs_error.mean()/Q-.5770369087419158)<1e-10
    daily=pd.DataFrame(dict(dia=days,DE_horaria_media_kWh=daily_spread,
        DE_horaria_max_kWh=spread.max(1),CV_horario_medio_pct=cv.mean(1),
        DE_media_diaria_kWh=P.mean(2).std(0,ddof=1),
        ancho_p10_p90_medio_kWh=(hi-lo).mean(1),MAE_individual_kWh=abs_error.mean((0,2)),
        MAE_media_predicciones_kWh=daily_ae,sesgo_media_kWh=daily_e,
        observado_medio_kWh=truth.mean(1),predicho_medio_kWh=center.mean(1),
        MASE_individual=abs_error.mean((0,2))/Q))
    daily.to_csv(out/'dispersion_diaria.csv',index=False)
    hourly=pd.DataFrame(dict(dia=np.repeat(days,H),horizonte=np.tile(np.arange(1,H+1),T),
        observado_kWh=truth.ravel(),prediccion_media_kWh=center.ravel(),
        DE_kWh=spread.ravel(),CV_pct=cv.ravel(),p10_kWh=lo.ravel(),p90_kWh=hi.ravel(),
        residuo_media_kWh=residual.ravel(),MAE_individual_kWh=abs_error.mean(0).ravel()))
    hourly.to_parquet(out/'dispersion_por_origen_horizonte.parquet',index=False)
    horizon=pd.DataFrame([dict(horizonte=h+1,DE_media_kWh=spread[:,h].mean(),
        DE_mediana_kWh=np.median(spread[:,h]),DE_p95_kWh=np.quantile(spread[:,h],.95),
        CV_medio_pct=cv[:,h].mean(),MAE_individual_kWh=abs_error[:,:,h].mean(),
        MAE_media_predicciones_kWh=ensemble_abs[:,h].mean(),sesgo_kWh=residual[:,h].mean(),
        rho_dispersion_error=stats.spearmanr(spread[:,h],ensemble_abs[:,h]).statistic)
        for h in range(H)])
    seed_loss=abs_error.mean(2)/Q
    pairs=pd.DataFrame([dict(semilla_A=a+101,semilla_B=b+101,
        diferencia_absoluta_media_kWh=np.abs(P[a]-P[b]).mean(),
        rho_MAE_diario=stats.spearmanr(seed_loss[a],seed_loss[b]).statistic)
        for a,b in combinations(range(10),2)])
    pairs.to_csv(out/'pares_semillas.csv',index=False)
    pd.DataFrame(dict(semilla=np.arange(101,111),MASE=seed_loss.mean(1),
        MAE_kWh=abs_error.mean((1,2)),RMSE_kWh=np.sqrt(((P-truth)**2).mean((1,2))))).to_csv(out/'metricas_semillas.csv',index=False)

    # Los orígenes diarios, no las horas, son la unidad de remuestreo temporal.
    B,L,seed=9999,14,20261124
    rng=np.random.default_rng(seed);ix=np.empty((B,T),dtype=np.int32)
    ix[:,0]=rng.integers(T,size=B)
    for t in range(1,T):
        ix[:,t]=np.where(rng.random(B)<1/L,rng.integers(T,size=B),(ix[:,t-1]+1)%T)
    vals=np.column_stack([daily_spread,daily_ae,daily_e,residual])
    boot=np.empty((B,vals.shape[1]));rho_boot=np.empty(B)
    for start in range(0,B,128):
        indices=ix[start:start+128]
        boot[start:start+128]=vals[indices].mean(1)
        ar=stats.rankdata(daily_spread[indices],axis=1);br=stats.rankdata(daily_ae[indices],axis=1)
        ar-=ar.mean(1,keepdims=True);br-=br.mean(1,keepdims=True)
        rho_boot[start:start+128]=np.sum(ar*br,axis=1)/np.sqrt(np.sum(ar**2,axis=1)*np.sum(br**2,axis=1))
    ci=np.quantile(boot,[.025,.975],axis=0)
    horizon['sesgo_ic95_inf_kWh']=ci[0,3:];horizon['sesgo_ic95_sup_kWh']=ci[1,3:]
    horizon.to_csv(out/'resumen_horizonte.csv',index=False)
    np.savez_compressed(out/'bootstrap.npz',medias=boot,rho_dispersion_error=rho_boot)

    # Diagnósticos sobre el residuo diario de la predicción media de las semillas.
    tests=[]
    def test(name,statistic,p,null,lag=None,h=None):
        tests.append(dict(prueba=name,estadistico=float(statistic),p=float(p),
                          H0=null,rezagos=lag,horizonte=h))
    for label,v in [('Ljung–Box',daily_e),('Ljung–Box cuadrados',(daily_e-daily_e.mean())**2)]:
        lb=acorr_ljungbox(v,lags=[7,14,28],return_df=True)
        for lag,r in lb.iterrows():test(label,r.lb_stat,r.lb_pvalue,'Autocorrelaciones hasta el rezago indicado iguales a cero',int(lag))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',FutureWarning)
        arch=het_arch(daily_e/Q,nlags=14)
    test('ARCH–LM',arch[0],arch[1],'Ausencia de efectos ARCH hasta 14 días',14)
    bd,pbd=bds(daily_e/Q,max_dim=2,distance=1.5)
    test('BDS',np.asarray(bd).item(),np.asarray(pbd).item(),'Residuos independientes e idénticamente distribuidos')
    jb=stats.jarque_bera(daily_e)
    test('Jarque–Bera',jb.statistic,jb.pvalue,'Normalidad del residuo diario')
    X=sm.add_constant(center.mean(1)/Q)
    fit=sm.OLS(truth.mean(1)/Q,X).fit(cov_type='HAC',cov_kwds={'maxlags':13})
    mz=fit.wald_test((np.eye(2),np.array([0.,1.])),scalar=True)
    test('Mincer–Zarnowitz',np.asarray(mz.statistic).item(),mz.pvalue,'Intercepto cero y pendiente uno, con covarianza HAC13')
    for h in range(H):
        lb=acorr_ljungbox(residual[:,h],lags=[14],return_df=True).iloc[0]
        test('Ljung–Box por horizonte',lb.lb_stat,lb.lb_pvalue,'Ausencia de autocorrelación hasta 14 días en este horizonte',14,h+1)
    tests=pd.DataFrame(tests);assert len(tests)==34
    tests['p_Holm']=multipletests(tests.p,method='holm')[1]
    tests['rechaza_005']=tests.p_Holm.lt(.05)
    tests['significado']=tests.prueba.map(SIGNIFICADOS_CONTRASTES)
    tests.to_csv(out/'contrastes_residuos.csv',index=False)
    pd.DataFrame(dict(rezago=np.arange(36),ACF=acf(daily_e,nlags=35,fft=True))).to_csv(out/'autocorrelacion.csv',index=False)
    qx,qy=stats.probplot(daily_e/1e6,fit=False)
    pd.DataFrame(dict(cuantil_normal=qx,residuo_GWh=qy)).to_csv(out/'qq_residuos.csv',index=False)

    # Segmentación descriptiva: calendario del objetivo y umbrales de entrenamiento.
    panel_path=source('data/processed/panel_horario.parquet')
    panel=pd.read_parquet(panel_path,columns=['y_demanda_real_kwh','cal_es_festivo'])
    ypanel=panel.loc['2022':'2025'].y_demanda_real_kwh.to_numpy().reshape(T,H)
    assert np.array_equal(ypanel,truth)
    train=panel.loc['2013':'2020'].y_demanda_real_kwh
    cut=train.quantile([.25,.5,.75]).to_numpy()
    holidays=panel.reindex(dates).cal_es_festivo.to_numpy().astype(bool)
    types=np.where(holidays,'Festivo',np.where(dates.day_of_week>=5,'Fin de semana','Laborable'))
    type_names=['Laborable','Fin de semana','Festivo'];level_names=['Q1 · demanda baja','Q2','Q3','Q4 · demanda alta']
    levels=np.searchsorted(cut,truth,side='right')
    masks=[('tipo_dia',name,np.broadcast_to((types==name)[:,None],(T,H)))for name in type_names]
    masks += [('nivel_demanda',name,levels==i)for i,name in enumerate(level_names)]
    daily['tipo_dia']=types;daily.to_csv(out/'dispersion_diaria.csv',index=False)

    bench=pd.read_csv(source('results/evaluacion_e2/benchmark.csv')).set_index('clave')
    names=bench.modelo.to_dict()|{'m4_n':'M4ₙ'};models={'m4_n':P}
    consolidated=source('results/evaluacion_e2/entradas/consolidado/predicciones_finales.parquet')
    frame=pd.read_parquet(consolidated)
    expected_ns=dates.as_unit('ns').asi8
    for (exp,var),group in frame.groupby(['experimento','variante'],sort=False):
        key=exp+'__'+var
        if key not in bench.index:continue
        parts=[]
        for s,g in group.groupby('semilla',sort=True):
            g=g.sort_values(['origen','horizonte'])
            assert s in range(1000,1005)
            assert np.array_equal(pd.DatetimeIndex(g.origen.drop_duplicates()).as_unit('ns').asi8,expected_ns)
            assert np.array_equal(g.observado_kwh.to_numpy().reshape(T,H),truth)
            assert np.array_equal(g.horizonte.to_numpy(),np.tile(np.arange(1,H+1),T))
            parts.append(g.predicho_kwh.to_numpy().reshape(T,H))
        models[key]=np.stack(parts)
    snap=root/'results/repositorio/d1ae9a4e09dd1dead4af1329f16aa3e8d0f2c3d3'
    reg_path=source('results/repositorio/d1ae9a4e09dd1dead4af1329f16aa3e8d0f2c3d3/results/consolidado/modelos.json')
    reg=json.loads(reg_path.read_text(encoding='utf-8'))
    for key in ['mlp','lstm','cnn1d','arx_lineal']:
        rows=[]
        for item in sorted(reg[key]['artefactos'],key=lambda r:r['semilla']):
            path=snap/item['predicciones'];sources.append(path)
            with np.load(path,allow_pickle=False)as z:
                assert np.array_equal(z['origen'],expected_ns) and np.allclose(z['y'],truth,rtol=0,atol=1e-6)
                rows.append(z['yhat'].astype(float))
        models[key]=np.stack(rows)
    raw=panel.y_demanda_real_kwh.to_numpy();pos=panel.index.get_indexer(dates);hours=np.arange(24)
    lag_daily=24*np.maximum(1,np.ceil((hours+2)/24).astype(int))
    models['ingenuo_168h']=raw[pos[:,None]+hours-168][None]
    models['ingenuo_24h']=raw[pos[:,None]+hours-lag_daily][None]
    assert len(models)==18
    segments=[]
    for key,arr in models.items():
        assert np.isfinite(arr).all()
        assert len(arr)==(10 if key=='m4_n' else 1 if key.startswith('ingenuo_') else 5)
        errors=arr-truth;absolute=np.abs(errors).mean(0)
        if key!='m4_n':assert abs(absolute.mean()/Q-bench.loc[key,'MASE'])<2e-6,key
        for family,label,mask in masks:
            n=int(mask.sum());n_days=int(mask.any(1).sum())
            segments.append(dict(clave=key,modelo=names[key],semillas=len(arr),familia=family,segmento=label,
                horas=n,origenes=n_days,MASE=absolute[mask].mean()/Q,
                MAE_kWh=absolute[mask].mean(),RMSE_kWh=np.sqrt((errors[:,mask]**2).mean(1)).mean(),
                sesgo_kWh=(truth-arr.mean(0))[mask].mean()))
    segments=pd.DataFrame(segments);segments.to_csv(out/'segmentos.csv',index=False)
    for family in ['tipo_dia','nivel_demanda']:
        m=segments.query('clave=="m4_n" and familia==@family')
        assert m.horas.sum()==T*H
        assert np.isclose(np.average(m.MASE,weights=m.horas),abs_error.mean()/Q)

    peak_index=int(daily_spread.argmax());peak_point=np.unravel_index(spread.argmax(),spread.shape)
    daily.sort_values('DE_horaria_media_kWh',ascending=False).head(20).to_csv(out/'dias_mayor_dispersion.csv',index=False)
    np.savez_compressed(out/'matrices.npz',dia=days,media_kWh=center,observado_kWh=truth,DE_kWh=spread,
        CV_pct=cv,p10_kWh=lo,p90_kWh=hi,residuo_kWh=residual,
        predicciones_dia_max_kWh=P[:,peak_index],dia_max=days[peak_index])
    seed_mase=seed_loss.mean(1);rho=stats.spearmanr(daily_spread,daily_ae).statistic
    summary=dict(dias=T,horizontes=H,semillas=10,Q_kWh=Q,MASE_individual=float(abs_error.mean()/Q),
        MASE_DE_semillas=float(seed_mase.std(ddof=1)),
        DE_media_GWh=float(spread.mean()/1e6),DE_mediana_GWh=float(np.median(spread)/1e6),
        DE_p95_GWh=float(np.quantile(spread,.95)/1e6),CV_medio_pct=float(cv.mean()),
        CV_p95_pct=float(np.quantile(cv,.95)),
        DE_media_ic95_GWh=(ci[:,0]/1e6).tolist(),
        dia_mayor_dispersion=str(days[peak_index]),DE_horaria_media_dia_max_GWh=float(daily_spread[peak_index]/1e6),
        maxima_DE_GWh=float(spread[peak_point]/1e6),dia_maxima_DE=str(days[peak_point[0]]),
        horizonte_maxima_DE=int(peak_point[1]+1),
        correlacion_dispersion_error_diario=float(rho),correlacion_dispersion_error_ic95=np.quantile(rho_boot,[.025,.975]).tolist(),
        rho_MAE_diario_entre_semillas_mediana=float(pairs.rho_MAE_diario.median()),
        cobertura_banda_semillas_p10_p90=float(((truth>=lo)&(truth<=hi)).mean()),
        ancho_medio_banda_semillas_GWh=float((hi-lo).mean()/1e6),
        MAE_media_predicciones_GWh=float(daily_ae.mean()/1e6),
        RMSE_media_predicciones_GWh=float(np.sqrt(mse_center)/1e6),
        sesgo_GWh=float(daily_e.mean()/1e6),sesgo_ic95_GWh=(ci[:,2]/1e6).tolist(),
        intercepto_calibracion_GWh=float(fit.params[0]*Q/1e6),pendiente_calibracion=float(fit.params[1]),
        MSE_individual_kWh2=float(mse_ind),MSE_media_predicciones_kWh2=float(mse_center),
        contribucion_inicializacion_MSE_kWh2=float(initialization_mse),
        porcentaje_MSE_asociado_dispersion=float(100*initialization_mse/mse_ind),
        contrastes=len(tests),rechazos_Holm=int(tests.rechaza_005.sum()),
        horizontes_con_autocorrelacion=int(tests.query('prueba=="Ljung–Box por horizonte"').rechaza_005.sum()),
        umbrales_demanda_kWh=cut.tolist(),periodo_umbrales=['2013-01-01','2020-12-31'],
        periodo=['2022-01-01','2025-12-31'])
    protocol=dict(unidad_remuestreo='origen diario de 24 horas',dispersion='DE muestral entre diez semillas (ddof=1)',
        bootstrap=dict(tipo='estacionario pareado',replicas=B,longitud_media_bloque_dias=L,semilla=seed,
                       nivel=.95,intervalos='percentiles puntuales',remuestreo_semillas=False),
        estimando_benchmark='Media de las pérdidas individuales por semilla',
        estimando_residuos='Observado menos media de diez predicciones; agregado por día para contrastes globales',
        familia_Holm=34,alfa=.05,periodo=summary['periodo'],
        BDS=dict(dimension=2,epsilon='1.5 desviaciones del residuo diario'),
        calibracion='Mincer–Zarnowitz, Wald conjunto intercepto=0 y pendiente=1, HAC13',
        segmentos=dict(tipo_dia='Festivo tiene prioridad sobre fin de semana; categorías excluyentes',
            nivel='Cuartiles de demanda horaria observada de 2013–2020; solo diagnóstico posterior',
            umbrales_kWh=cut.tolist(),intervalos=False),
        bandas_semillas='p10–p90 describe inicializaciones, no cobertura predictiva nominal',
        atencion='Su estabilidad se estudió en 13 consultas de 2021; no se correlaciona directamente con las salidas de prueba de otros días',
        entrenamiento=False,inferencia_neuronal=False,modificacion_benchmark=False)
    hashes={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()for p in sorted(set(sources))}
    for name,obj in [('resumen.json',summary),('protocolo.json',protocol),('verificacion.json',dict(
        n_archivos_predicciones_M4=10,fechas_completas_unicas=True,paridad_objetivo=True,
        paridad_MASE=True,identidad_MSE=True,segmentos_excluyentes_y_exhaustivos=True,
        comparadores=17,fuentes_sha256=hashes))]:
        (out/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':calcular()

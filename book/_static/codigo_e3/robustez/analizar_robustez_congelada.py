"""Agregación por origen, bootstrap temporal/semillas y Holm de robustez."""
from __future__ import annotations
from pathlib import Path
import hashlib,json,sys,os
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/robustez_congelada_20261124'
Q=215617.90361589016


def calidad_entradas(root=ROOT):
    root=Path(root);out=root/'results/robustez_congelada_20261124'
    contract=json.loads((root/'config/diseno_entrenamiento_texto.json').read_text(encoding='utf-8'))
    columns=contract['numericas']['historicas']
    panel=pd.read_parquet(root/'data/processed/panel_horario.parquet',columns=columns)
    registry=json.loads((root/'data/processed/feature_registry.json').read_text(encoding='utf-8'))
    for column in columns:
        if column.startswith('x_obs_'):
            panel[column]=panel[column].shift(contract['numericas']['rezago_clima_h']-registry[column]['rezago_h'])
    protocol=json.loads((out/'protocolo.json').read_text(encoding='utf-8'))
    windows=np.stack([panel.reindex(pd.date_range(pd.Timestamp(day,tz='America/Bogota')-pd.Timedelta(hours=168),periods=168,freq='h')).to_numpy() for day in protocol['origenes']])
    missing=~np.isfinite(windows)
    result=dict(origenes=len(windows),horas_por_ventana=168,variables=len(columns),celdas=int(windows.size),
                celdas_ausentes=int(missing.sum()),porcentaje_ausente=float(missing.mean()*100),
                ventanas_con_ausencias=int(missing.any((1,2)).sum()),horas_impulsadas_simuladas=9,
                porcentaje_horas_impulsadas=100*9/168,
                alcance='Ausencias en las entradas originales de las 96 ventanas, con los rezagos del contrato; no estima frecuencia natural de atípicos.')
    (out/'calidad_entradas_originales.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return result


def main(root=ROOT):
    root=Path(root);out=root/'results/robustez_congelada_20261124'
    calidad_entradas(root)
    protocol=json.loads((out/'protocolo.json').read_text(encoding='utf-8'))
    cases=protocol['casos'];dates=protocol['origenes'];study=json.loads((root/'dashboard/site/data/study.json').read_text(encoding='utf-8'))
    names={r['id']:r['name']for r in study['models']}|{'m4_n':'M4ₙ','naive_diario':'Ingenuo diario','naive_semanal':'Ingenuo semanal'}
    groups={r['id']:r['group']for r in study['models']}|{'m4_n':'Propuesta','naive_diario':'Referencia estacional','naive_semanal':'Referencia estacional'}
    conditions=[('original',0)]+[(k,i)for k in protocol['tipos']for i in range(1,4)]
    cond_indices={c:[j for j,v in enumerate(cases)if(v['tipo'],v['nivel'])==c]for c in conditions}
    maes={};mses={};seed_ids={};parity=[];detail=[];protected={};expected_means={}
    for key in names:
        files=sorted((out/'predicciones').glob(key+'__semilla_*.npz'))
        count=1 if key.startswith('naive_')else 5
        assert len(files)==count,(key,len(files),count)
        er=[];sq=[];clean=[];seeds=[]
        for file in files:
            with np.load(file,allow_pickle=False)as z:
                assert z['dias'].tolist()==dates and z['casos'].tolist()==[c['id']for c in cases]
                p=z['pred'];y=z['y'];assert p.shape==(96,len(cases),24)and np.isfinite(p).all()
                if not key.endswith('__N+O'):
                    ix=[j for j,c in enumerate(cases)if c['tipo']=='perdida_texto']
                    assert np.array_equal(p[:,ix],np.broadcast_to(p[:,0,None,:],p[:,ix].shape)),(key,'Predicción numérica alterada por pérdida textual')
                error=p-y[:,None,:];er.append(np.abs(error).mean(-1));sq.append(np.square(error).mean(-1));clean.append(p[:,0])
            info=json.loads(file.with_suffix('.json').read_text(encoding='utf-8'));seed=info['semilla'];seeds.append(seed)
            protected.update(info['pesos_sha256'])
            for r in info['paridad']:parity.append(dict(modelo=key,semilla=seed,**r))
            for j,case in enumerate(cases):
                for n,date in enumerate(dates):
                    detail.append(dict(clave=key,semilla=seed,origen=date,tipo=case['tipo'],nivel=case['nivel'],repeticion=case['repeticion'],
                                       MAE=er[-1][n,j],MSE=sq[-1][n,j],MASE=er[-1][n,j]/Q))
        rawmae=np.stack(er)/Q;rawmse=np.stack(sq)
        maes[key]=np.stack([rawmae[:,:,cond_indices[c]].mean(-1)for c in conditions],axis=1)
        mses[key]=np.stack([rawmse[:,:,cond_indices[c]].mean(-1)for c in conditions],axis=1)
        seed_ids[key]=seeds
        if key!='m4_n'and not key.startswith('naive_'):
            saved=json.loads((root/f'dashboard/site/data/predictions/{key}.json').read_text(encoding='utf-8'))
            j=np.asarray([(pd.Timestamp(d)-pd.Timestamp('2022-01-01')).days for d in dates])
            expected=np.asarray(saved['mean'])[j]*1e6;pred=np.mean(clean,axis=0)
            error=float(np.max(np.abs(pred-expected)));relative=error/max(float(np.max(np.abs(expected))),1)
            assert relative<1e-4,(key,'Paridad media',relative,error)
            expected_means[key]=dict(error_max_kWh=error,error_relativo=relative)
        # Las redes sin canal documental deben conservar exactamente la predicción.
        if not key.endswith('__N+O'):
            for c in range(10,13):
                assert np.allclose(maes[key][:,0],maes[key][:,c],rtol=1e-14,atol=1e-14)
                # Predicciones idénticas: conservar la identidad exacta al promediar tres copias.
                maes[key][:,c]=maes[key][:,0]
                mses[key][:,c]=mses[key][:,0]
    B=protocol['bootstrap']['reps'];rng=np.random.default_rng(20261124)
    starts=rng.integers(0,48,size=(B,16))
    months=((starts[:,:,None]+np.arange(3))%48).reshape(B,48)
    distributions={};summary=[];degradation=[]
    for key in names:
        # Dos orígenes por mes; las réplicas estocásticas ya se promediaron.
        monthly=maes[key].reshape(len(seed_ids[key]),13,48,2).mean(-1)
        sd=rng.integers(0,len(seed_ids[key]),size=(B,len(seed_ids[key])))
        boots=[]
        for c in range(13):
            boots.append(monthly[:,c,:][sd[:,:,None],months[:,None,:]].mean((1,2)))
        boots=np.stack(boots,axis=1);distributions[key]=boots
        base=maes[key][:,0].mean()
        for c,(kind,level)in enumerate(conditions):
            metric=maes[key][:,c].mean();seeds_mean=maes[key][:,c].mean(-1)
            lo,hi=np.quantile(boots[:,c],[.025,.975])
            diff=metric-base;db=boots[:,c]-boots[:,0];dlo,dhi=np.quantile(db,[.025,.975])
            rel=100*(metric/base-1);rb=100*(boots[:,c]/boots[:,0]-1);rlo,rhi=np.quantile(rb,[.025,.975])
            severity=0 if level==0 else protocol['tipos'][kind][level-1]
            summary.append(dict(clave=key,modelo=names[key],grupo=groups[key],tipo=kind,nivel=level,severidad=severity,
                                semillas=len(seed_ids[key]),origenes=96,MASE=metric,MASE_sd=seeds_mean.std(ddof=1)if len(seeds_mean)>1 else 0.,
                                MASE_ic_inf=lo,MASE_ic_sup=hi,MAE=metric*Q,RMSE=np.sqrt(mses[key][:,c].mean(-1)).mean(),
                                delta_MASE=diff,delta_MASE_ic_inf=dlo,delta_MASE_ic_sup=dhi,
                                degradacion_pct=rel,degradacion_ic_inf=rlo,degradacion_ic_sup=rhi))
            if level:
                p=float((1+np.sum(db-diff>=diff))/(B+1))
                degradation.append(dict(clave=key,modelo=names[key],tipo=kind,nivel=level,delta_MASE=diff,ic_inf=dlo,ic_sup=dhi,
                                        p=p,nula='E[MASE_perturbado − MASE_original] ≤ 0'))
    contrasts=[]
    for key in names:
        if key=='m4_n':continue
        for c,(kind,level)in enumerate(conditions):
            if not level:continue
            a=maes['m4_n'][:,c].mean();b=maes[key][:,c].mean()
            difference=.97*b-a;db=.97*distributions[key][:,c]-distributions['m4_n'][:,c]
            p=float((1+np.sum(db-difference>=difference))/(B+1))
            gain=100*(1-a/b);gb=100*(1-distributions['m4_n'][:,c]/distributions[key][:,c]);lo,hi=np.quantile(gb,[.025,.975])
            dlo,dhi=np.quantile(db,[.025,.975])
            contrasts.append(dict(comparador=key,modelo=names[key],tipo=kind,nivel=level,MASE_M4=a,MASE_comparador=b,
                                  mejora_pct=gain,mejora_ic_inf=lo,mejora_ic_sup=hi,contraste_MASE=difference,
                                  ic_inf=dlo,ic_sup=dhi,p=p,margen=.03,nula='E[0,97 MASE_comparador − MASE_M4] ≤ 0'))
    deg=pd.DataFrame(degradation);comp=pd.DataFrame(contrasts)
    assert len(deg)==216 and len(comp)==204
    deg['p_ajustado']=multipletests(deg.p,method='holm')[1]
    comp['p_ajustado']=multipletests(comp.p,method='holm')[1]
    comp['supera_3pct']=comp.p_ajustado.lt(.05)
    summary=pd.DataFrame(summary);summary.to_csv(out/'resumen.csv',index=False)
    deg.to_csv(out/'contrastes_degradacion.csv',index=False);comp.to_csv(out/'contrastes_M4.csv',index=False)
    pd.DataFrame(detail).to_parquet(out/'metricas_por_origen.parquet',index=False)
    pd.DataFrame(parity).to_csv(out/'paridad_predicciones.csv',index=False)
    np.savez_compressed(out/'distribuciones_bootstrap.npz',**{key:arr for key,arr in distributions.items()})
    costs=pd.read_csv(out/'paridad_predicciones.csv')
    invariance={}
    for path,expected in protected.items():
        with open(path,'rb')as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
        assert actual==expected,path
        try:label=Path(path).relative_to(root).as_posix()
        except ValueError:
            try:label=os.path.relpath(path,root).replace('\\','/')
            except ValueError:label=str(Path(path))
        invariance[label]=actual
    audit=dict(modelos=18,pipelines_aprendidos=16,semillas_por_pipeline=5,referencias_estacionales=2,origenes=96,
               casos=len(cases),niveles_no_nulos=3,tipos=4,replicas_estocasticas=3,contrastes_degradacion=216,contrastes_M4=204,
               paridad_max_kWh=float(costs.error_max_kWh.max()),paridad_max_relativo=float(costs.error_relativo.max()),
               paridad_media=expected_means,artefactos_intactos=invariance,entrenamiento=False,publicacion=False)
    (out/'verificacion.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(summary.query("clave=='m4_n'")[['tipo','nivel','MASE','degradacion_pct']].to_string(index=False))
    print('Resumen: 234 filas; 216 contrastes de degradación y 204 frente a M4, Holm por familia.')
    return summary,comp,deg


if __name__=='__main__':main()

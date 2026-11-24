"""Diagnósticos retrospectivos de pesos congelados. No contiene optimización."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse,copy,hashlib,json,os,platform,sys,time,warnings
import numpy as np
import pandas as pd
import torch
from inferencia_ampliada import ROOT,CON,FU,Q,Entradas,cargar,registro,predecir,guardadas,forward,sha
from importancia_variables import indices_donantes

OUT=ROOT/'results/diagnosticos_ampliados'
GRUPOS=['perfil_demanda','nivel_ventana','precio','reservas','aportes','temperatura','humedad','precipitacion','calendario']
REPS=3
E=None

def iniciar():
    global E
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(11037)
    torch.use_deterministic_algorithms(True)
    warnings.filterwarnings('ignore',message='.*weight_norm.*deprecated.*',category=FutureWarning)
    E=Entradas()

def clave(m):return m['experimento']+'__'+m['variante']
def nombre(m):return clave(m).replace('+','_')+'_'+str(m['semilla'])
def muestra(e):return np.flatnonzero(e.fechas.day.isin([1,8,15,22]))

def perturbada(data,donantes,grupo,repo,e):
    d=data.copy()
    if grupo=='nivel_ventana':d['nivel']=data['nivel'][donantes];return d
    if grupo=='calendario':
        d['cp']=data['cp'][donantes];d['cf']=data['cf'][donantes]
        if repo:
            d['x']=data['x'].copy();d['x'][:,:,30:53]=data['x'][donantes,:,30:53]
        return d
    cols={'perfil_demanda':[0],'precio':[1],'reservas':[2,3,4],'aportes':[5],
          'temperatura':list(range(6,14)),'humedad':list(range(14,22)),
          'precipitacion':list(range(22,30))}[grupo]
    d['x']=data['x'].copy();d['x'][:,:,cols]=data['x'][donantes][:,:,cols]
    # Las máscaras individuales de la variante numérica se trasladan con su variable.
    if repo:
        maskcols=np.array(cols)+53;d['x'][:,:,maskcols]=data['x'][donantes][:,:,maskcols]
    elif grupo in ['precio','reservas','aportes']:
        # Las cuatro magnitudes hídricas comparten un indicador de ausencia.
        j=30+e.mnames.index('ausente_precio' if grupo=='precio' else 'ausente_hidrologia')
        assert not data['x'][:,:,j].any(), 'El periodo de diagnóstico debe tener estos grupos completos'
    return d

def canal(red):return getattr(red,'agregador',None) or getattr(red,'relevancia',None)

def trabajar(m):
    e=E;folder=OUT/'por_semilla'/nombre(m);folder.mkdir(parents=True,exist_ok=True)
    if (folder/'completo.json').exists():return nombre(m)+' conservado'
    t0=time.perf_counter();red,data,repo=cargar(m,e);ix=np.arange(len(e.fechas));sample=muestra(e)
    pred=predecir(red,data,ix,repo);saved=guardadas(m)
    assert pred.shape==saved.shape==e.y.shape
    error=np.abs(pred-e.y);old=np.abs(saved-e.y);diff=np.abs(pred-saved)
    # Tolerancia de equivalencia numérica entre motores; no se confunde con δ del estudio.
    maxrel=float((diff/np.maximum(np.abs(saved),1)).max())
    reldmae=float(abs(error.mean()-old.mean())/old.mean())
    assert maxrel<.001 and reldmae<.005,(clave(m),maxrel,reldmae)
    valid=dict(clave=clave(m),semilla=m['semilla'],checkpoint_sha256=m['sha256_checkpoint'],
        max_diferencia_kwh=float(diff.max()),media_diferencia_kwh=float(diff.mean()),
        max_diferencia_relativa=maxrel,MAE_guardado=float(old.mean()),MAE_cpu=float(error.mean()),
        cambio_relativo_MAE=reldmae,origenes=len(ix),reproduccion_numerica=True)
    pd.DataFrame([valid]).to_csv(folder/'reproduccion.csv',index=False)
    np.savez_compressed(folder/'basal.npz',pred=pred,loss=error.mean(1)/Q)
    filas=[];effects=[]
    for r in range(REPS):
        donor=indices_donantes(e.fechas,r)
        for group in GRUPOS:
            dp=perturbada(data,donor,group,repo,e);p=predecir(red,dp,sample,repo)
            er=np.abs(p-e.y[sample]).mean(1);base=error[sample].mean(1)
            delta=(er-base)/Q
            filas.append(dict(clave=clave(m),semilla=m['semilla'],repeticion=r,grupo=group,
                MAE_base=float(base.mean()),MAE_perturbado=float(er.mean()),
                delta_MASE=float(delta.mean()),cambio_pct=float(100*(er.mean()/base.mean()-1)),
                divergencia_kwh=float(np.abs(p-pred[sample]).mean()),origenes=len(sample)))
            effects.append(delta)
    pd.DataFrame(filas).to_csv(folder/'importancia.csv',index=False)
    np.savez_compressed(folder/'importancia_diaria.npz',delta=np.stack(effects),indices=sample)
    # Las contrafactuales textuales mantienen x, calendarios, nivel y pesos.
    ca=canal(red);sens=[];losses=[];labels=[];repids=[]
    if ca is not None:
        lista_original=ca.lista;vec_original=ca.vectores
        conditions=[('sin_documentos',0),('contenido_cero',0)]+[('contenido_historico',r) for r in range(REPS)]
        for condition,r in conditions:
            ca.lista=lista_original;ca.vectores=vec_original
            if condition=='sin_documentos':ca.lista=torch.full_like(lista_original,-1)
            elif condition=='contenido_cero':ca.vectores=torch.zeros_like(vec_original)
            else:
                ca.vectores,donors=e.placebo_historico(r)
                if not (OUT/f'donantes_texto_{r}.npy').exists():np.save(OUT/f'donantes_texto_{r}.npy',donors,allow_pickle=False)
            p=predecir(red,data,ix,repo);er=np.abs(p-e.y).mean(1)
            sens.append(dict(clave=clave(m),semilla=m['semilla'],condicion=condition,repeticion=r,
                MASE_base=float(error.mean()/Q),MASE=float(er.mean()/Q),
                cambio_pct=float(100*(er.mean()/error.mean()-1)),divergencia_kwh=float(np.abs(p-pred).mean()),origenes=len(ix)))
            losses.append(er/Q);labels.append(condition);repids.append(r)
        ca.lista=lista_original;ca.vectores=vec_original
    # TGForecaster usa también descripciones estáticas en N y N+O.
    if hasattr(red,'descripciones'):
        original=red.descripciones
        for condition,r in [('descripciones_cero',0)]+[('descripciones_permutadas',r) for r in range(REPS)]:
            gen=torch.Generator().manual_seed(11037+r)
            red.descripciones=torch.zeros_like(original) if condition=='descripciones_cero' else original[torch.randperm(len(original),generator=gen)]
            # Relevancia conserva la descripción del objetivo: se modifica únicamente la asociación canal/memoria.
            p=predecir(red,data,ix,repo);er=np.abs(p-e.y).mean(1)
            sens.append(dict(clave=clave(m),semilla=m['semilla'],condicion=condition,repeticion=r,
                MASE_base=float(error.mean()/Q),MASE=float(er.mean()/Q),cambio_pct=float(100*(er.mean()/error.mean()-1)),
                divergencia_kwh=float(np.abs(p-pred).mean()),origenes=len(ix)))
            losses.append(er/Q);labels.append(condition);repids.append(r)
        red.descripciones=original
    if sens:
        pd.DataFrame(sens).to_csv(folder/'sensibilidad.csv',index=False)
        np.savez_compressed(folder/'sensibilidad_diaria.npz',loss=np.stack(losses),condiciones=np.array(labels),repeticiones=np.array(repids))
    assert sha(CON/m['artefactos']['pesos'])==m['sha256_checkpoint']
    end=dict(clave=clave(m),semilla=m['semilla'],segundos=time.perf_counter()-t0,entrenamiento=False)
    (folder/'completo.json').write_text(json.dumps(end,indent=2),encoding='utf-8')
    return nombre(m)+' '+str(round(end['segundos']))+' s'

def resumir():
    models=registro()
    assert models and len({(clave(m),m['semilla']) for m in models})==len(models)
    for name in ['reproduccion','importancia','sensibilidad']:
        files=[OUT/'por_semilla'/nombre(m)/(name+'.csv') for m in models]
        data=pd.concat([pd.read_csv(p) for p in files if p.exists()],ignore_index=True)
        data.to_csv(OUT/(name+'_semillas.csv'),index=False)
    imp=pd.read_csv(OUT/'importancia_semillas.csv')
    assert len(imp)==len(models)*REPS*len(GRUPOS)
    # Primero se promedian las cinco semillas para cada desplazamiento; rango p10–p90 entre desplazamientos.
    reps=imp.groupby(['clave','grupo','repeticion']).agg(delta_MASE=('delta_MASE','mean'),
        base=('MAE_base','mean'),pert=('MAE_perturbado','mean')).reset_index()
    reps['cambio_pct']=100*(reps.pert/reps.base-1)
    summary=reps.groupby(['clave','grupo']).agg(delta_MASE=('delta_MASE','mean'),cambio_pct=('cambio_pct','mean'),
        p10=('cambio_pct',lambda s:s.quantile(.1)),p90=('cambio_pct',lambda s:s.quantile(.9))).reset_index()
    summary.to_csv(OUT/'importancia_resumen.csv',index=False)
    sens=pd.read_csv(OUT/'sensibilidad_semillas.csv')
    summaries=[]
    for (k,condition),g in sens.groupby(['clave','condicion']):
        losses=[];bases=[]
        for m in [m for m in models if clave(m)==k]:
            f=OUT/'por_semilla'/nombre(m)
            with np.load(f/'sensibilidad_diaria.npz',allow_pickle=False) as z:
                losses.append(z['loss'][z['condiciones']==condition].mean(0))
            with np.load(f/'basal.npz',allow_pickle=False) as z:bases.append(z['loss'])
        b=np.mean(bases,0);a=np.mean(losses,0);rng=np.random.default_rng(11037);ratios=[]
        for j in range(10000):
            starts=rng.integers(len(a),size=int(np.ceil(len(a)/14)))
            ix=((starts[:,None]+np.arange(14))%len(a)).ravel()[:len(a)]
            ratios.append(100*(a[ix].mean()/b[ix].mean()-1))
        lo,hi=np.quantile(ratios,[.025,.975]);nrep=g.repeticion.nunique()
        summaries.append(dict(clave=k,condicion=condition,MASE_base=float(b.mean()),MASE=float(a.mean()),
            cambio_pct=float(100*(a.mean()/b.mean()-1)),IC95_inf=lo,IC95_sup=hi,
            divergencia_kwh=float(g.divergencia_kwh.mean()),semillas=5,repeticiones=nrep,origenes=len(a)))
    pd.DataFrame(summaries).to_csv(OUT/'sensibilidad_resumen.csv',index=False)

def medir_latencia():
    # Ejecutar sólo cuando hayan terminado los procesos de perturbación.
    e=Entradas();models=registro();rows=[]
    ix=np.array([0,730,1460])  # mismos tres orígenes, fijados antes de medir.
    for m in models:
        red,data,repo=cargar(m,e)
        with torch.inference_mode():
            for origin in ix:
                x=torch.from_numpy(data['x'][origin:origin+1]);cf=torch.from_numpy(data['cf'][origin:origin+1])
                if repo:args=(x,cf)
                else:args=(x,torch.from_numpy(data['cp'][origin:origin+1]),cf,torch.from_numpy(data['dia'][origin:origin+1]))
                for _ in range(20):red(*args)
                ts=[]
                for _ in range(100):
                    t=time.perf_counter_ns();out=red(*args);ts.append((time.perf_counter_ns()-t)/1e6)
                assert out.shape==(1,24) and torch.isfinite(out).all()
                ca=canal(red)
                cierres=7 if getattr(red,'agregador',None) is not None else 1 if ca is not None else 0
                dias=data['dia'][origin]+np.arange(-(cierres-1),1) if cierres else np.array([],dtype=int)
                docs=int((e.lista[dias]>=0).sum()) if cierres else 0
                rows.append(dict(clave=clave(m),semilla=m['semilla'],origen=str(e.fechas[origin]),mediana_ms=float(np.median(ts)),
                    p95_ms=float(np.quantile(ts,.95)),media_ms=float(np.mean(ts)),repeticiones=100,calentamiento=20,batch=1,hilos=1,
                    documentos=docs,cierres_textuales=cierres,documentos_origen=int((e.lista[e.dia[origin]]>=0).sum()) if cierres else 0,
                    checkpoint_sha256=m['sha256_checkpoint']))
        print('Latencia',nombre(m),flush=True)
    pd.DataFrame(rows).to_csv(OUT/'latencia_semillas.csv',index=False)
    # Referencias base: el mismo ordenador, un hilo y los tres mismos orígenes.
    from types import SimpleNamespace
    snapshot=ROOT/'results/repositorio/d1ae9a4e09dd1dead4af1329f16aa3e8d0f2c3d3'
    sys.path.insert(0,str(snapshot/'src'))
    from demanda import modelos_base as M
    from demanda.reentrenamiento_v3 import Secuencial as CNN
    from demanda.variantes_v4 import Secuencial as LSTM
    from importancia_variables import restaurar_entradas
    reg=json.loads((snapshot/'results/consolidado/modelos.json').read_text(encoding='utf-8'))
    panel=pd.read_parquet(ROOT/'data/processed/panel_horario.parquet')
    for key in ['mlp','arx_lineal','cnn1d','lstm']:
        for item in sorted(reg[key]['artefactos'],key=lambda x:x['semilla']):
            sc=json.loads((snapshot/item['escaladores']).read_text(encoding='utf-8'))
            xp,cf,level,y,cols=restaurar_entradas(panel,e.fechas,sc)
            ent=SimpleNamespace(x=np.zeros((1,168,1),dtype=np.float32),cal_p=np.zeros((1,168,23),dtype=np.float32),cal_f=np.zeros((1,24,23),dtype=np.float32))
            p=reg[key]['configuracion']['params']
            red=LSTM(key,p,ent) if key=='lstm' else CNN(key,p,ent) if key=='cnn1d' else M.construir_red(key,p,ent)
            path=snapshot/item['pesos'];red.load_state_dict(torch.load(path,map_location='cpu',weights_only=True)['state_dict']);red.eval()
            with torch.inference_mode():
                for origin in ix:
                    args=(torch.from_numpy(xp[origin:origin+1]),torch.from_numpy(cf[origin:origin+1]))
                    for _ in range(20):red(*args)
                    ts=[]
                    for _ in range(100):
                        t=time.perf_counter_ns();out=red(*args);ts.append((time.perf_counter_ns()-t)/1e6)
                    rows.append(dict(clave=key,semilla=item['semilla'],origen=str(e.fechas[origin]),mediana_ms=float(np.median(ts)),
                        p95_ms=float(np.quantile(ts,.95)),media_ms=float(np.mean(ts)),repeticiones=100,calentamiento=20,batch=1,hilos=1,
                        documentos=0,cierres_textuales=0,documentos_origen=0,checkpoint_sha256=sha(path)))
            print('Latencia',key,item['semilla'],flush=True)
    d=pd.DataFrame(rows);assert len(d)==(len(models)+sum(len(v['artefactos']) for v in reg.values()))*len(ix);d.to_csv(OUT/'latencia_semillas.csv',index=False)
    d.groupby('clave').agg(mediana_ms=('mediana_ms','median'),p95_ms=('p95_ms','median'),
        minimo_ms=('mediana_ms','min'),maximo_ms=('mediana_ms','max'),mediciones=('mediana_ms','size')).reset_index().to_csv(OUT/'latencia_resumen.csv',index=False)
    (OUT/'entorno.json').write_text(json.dumps(dict(fecha=pd.Timestamp.now(tz='America/Bogota').isoformat(),python=sys.version,
        torch=torch.__version__,numpy=np.__version__,pandas=pd.__version__,sistema=platform.platform(),procesador=platform.processor(),
        cpu_logicas=os.cpu_count(),dispositivo='cpu',hilos=1,entrenamiento=False,
        alcance_latencia='Sólo forward, entradas reales preparadas y vectores congelados en memoria; sin carga, escalado, tokenización, encoder E5 ni escritura'),ensure_ascii=False,indent=2),encoding='utf-8')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--fase',choices=['inferir','latencia','resumir'],default='inferir');ap.add_argument('--workers',type=int,default=4);args=ap.parse_args()
    OUT.mkdir(exist_ok=True);(OUT/'por_semilla').mkdir(exist_ok=True)
    if args.fase=='inferir':
        dates=pd.date_range('2022-01-01','2025-12-31',freq='D',tz='America/Bogota');sample=dates[dates.day.isin([1,8,15,22])]
        assert len(sample)==192
        protocol=dict(fecha=pd.Timestamp.now(tz='America/Bogota').isoformat(),escenario='diagnostico_retrospectivo',
            procedimientos=len({clave(m) for m in registro()}),semillas=list(range(1000,1005)),grupos=GRUPOS,repeticiones=REPS,importancia_origenes=192,
            fechas_importancia=sample.strftime('%Y-%m-%d').tolist(),regla_muestra='Días 1,8,15,22 de cada mes; misma muestra para todos',
            sensibilidad_origenes=1461,donantes_numericos='desplazamiento circular por año, mínimo 28 días; semilla11037+r',
            placebo_texto='Remuestreo con reemplazo dentro de fuente; contenido disponible antes de 2022; lista, volumen y presencia originales',
            sensibilidad_estatica='Descripciones TG a cero o permutadas entre canales; descripción de consulta de relevancia conservada',
            incertidumbre_importancia='p10–p90 de tres desplazamientos tras promediar cinco semillas; no IC',
            incertidumbre_sensibilidad='Bootstrap circular pareado de 14 días, 10000 remuestras; semillas/reemplazos promediados primero',
            base_importancia='Predicción CPU sin perturbar del mismo checkpoint sobre los 192 orígenes',
            base_sensibilidad='Predicción CPU sin perturbar del mismo checkpoint sobre los 1461 orígenes',
            tolerancia_reproduccion=dict(max_rel_pred=.001,max_rel_MAE=.005),q168=Q,entrenamiento=False,seleccion=False,publicacion=False)
        p=OUT/'protocolo.json'
        if not p.exists():p.write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8')
        else:
            old=json.loads(p.read_text(encoding='utf-8'));assert {k:v for k,v in old.items() if k!='fecha'}=={k:v for k,v in protocol.items() if k!='fecha'}
        with ProcessPoolExecutor(max_workers=args.workers,initializer=iniciar) as pool:
            jobs=[pool.submit(trabajar,m) for m in registro()]
            for f in as_completed(jobs):print(f.result(),flush=True)
        resumir()
    elif args.fase=='latencia':
        torch.set_num_threads(1);torch.set_num_interop_threads(1);medir_latencia()
    else:resumir()

if __name__=='__main__':main()

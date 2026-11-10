"""Ruta 1: PatchTST adaptado, núcleo fijo y búsqueda de hiperparámetros."""
import fcntl,hashlib,html,json,os,sys,time,traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from demanda.patchtst_numerico import Preparador,PatchTSTNumerico,entrenar,predecir
from demanda import particion as P,modelos_base as M
from demanda.metricas import evaluar
from demanda.potencia_pareada import dimensionar
OUT=ROOT/'results/patchtst_numerico_r1';OUT.mkdir(parents=True,exist_ok=True)
S={};DONE=0;TOTAL=260;DEADLINE=None

def save(p,obj):
    temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str));temp.replace(p)
def state(**kw):
    S.update(kw);S.update(completados=DONE,total=TOTAL,actualizado=pd.Timestamp.now(tz='America/Bogota').isoformat(),pid=os.getpid())
    save(OUT/'estado.json',S)
    txt=' · '.join(f'{k}: {v}' for k,v in S.items() if k in ('estado','etapa','configuracion','semilla','fold','epoca'))
    body=f'''<!doctype html><html lang="es"><meta charset="utf-8"><meta http-equiv="refresh" content="8"><title>PatchTST numérico</title><style>body{{font:18px system-ui;background:#14202c;color:#eef5fa;max-width:950px;margin:50px auto;padding:20px}}progress{{width:100%;height:32px}}p{{line-height:1.7}}</style><h1>PatchTST · numéricas ampliadas · GPU</h1><progress value="{DONE}" max="{TOTAL}"></progress><p>{DONE}/{TOTAL} ajustes previstos · {100*DONE/max(TOTAL,1):.1f}%</p><p>{html.escape(txt)}</p><p>20 configuraciones × 3 semillas × 4 pliegues = 240 ajustes.<br>Piloto: 5 semillas × 4 pliegues = 20 ajustes.<br>Evaluación final: R semillas, pendientes del dimensionamiento del piloto.</p><p>30 variables históricas + 23 de calendario + 30 máscaras. Historia168h · horizonte24h · MAE normalizada · máximo200épocas.<br>Arquitectura fija basada en código oficial; lectura afín adaptada. Evaluación retrospectiva2022–2025.</p><p>{html.escape(S.get('detalle',''))}</p><small>{S['actualizado']}</small></html>'''
    temp=OUT/'progreso.tmp';temp.write_text(body);temp.replace(OUT/'progreso.html')

def configuraciones():
    rng=np.random.default_rng(20261110);cc=[dict(id='P01',lr=.0001,batch_size=32,dropout=.3,weight_decay=.0001)]
    for i in range(1,20):cc.append(dict(id=f'P{i+1:02d}',lr=float(np.exp(rng.uniform(np.log(1e-4),np.log(3e-3)))),batch_size=int(rng.choice([16,32,64])),dropout=float(rng.uniform(.05,.3)),weight_decay=float(np.exp(rng.uniform(np.log(1e-6),np.log(1e-2))))))
    return cc
CONFIG=configuraciones();DATA={};PREPARER=None;PANEL=None;CONTRACT=None

def pack(fold):
    if fold in DATA:return DATA[fold]
    if fold=='final':
        tr,es=P.pretest(PANEL.index);ev=P.origenes(PANEL.index,'test',allow_test=True)
    else:
        f=P.folds_forward_chaining(PANEL.index,[fold])[0];tr,es,ev=f['tr'],f['es'],f['val']
    prep=Preparador(PANEL,REG,CONTRACT).ajustar(tr)
    a,b,c=prep.transformar(tr),prep.transformar(es),prep.transformar(ev)
    assert pd.Timestamp(prep.stats['fin_ajuste'])<b.origen.min()
    assert a.origen.max()+pd.Timedelta(hours=23)<b.origen.min()
    assert b.origen.max()+pd.Timedelta(hours=23)<c.origen.min()
    DATA.clear();DATA[fold]=(prep,a,b,c)
    return DATA[fold]

def references(c):
    pos=PANEL.index.get_indexer(c.origen);h=np.arange(24)
    y=PANEL.y_demanda_real_kwh.to_numpy();weekly=y[pos[:,None]+h-168]
    daily=y[pos[:,None]+h-np.where(h-24<=-2,24,48)]
    return dict(ingenuo_168h=weekly,ingenuo_24h=daily)

def fit(stage,cfg,seed,fold):
    global DONE
    identity=dict(etapa=stage,configuracion=cfg,semilla=seed,fold=fold,modelo='PatchTST_adaptado')
    key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:20]
    folder=OUT/'ajustes'/stage/key;folder.mkdir(parents=True,exist_ok=True)
    state(etapa=stage,configuracion=cfg['id'],semilla=seed,fold=fold,epoca=0)
    if (folder/'resultado.json').exists():
        result=json.loads((folder/'resultado.json').read_text());DONE+=1;state();return result
    if DEADLINE is not None and time.time()>=DEADLINE:raise TimeoutError('Presupuesto de24horas agotado')
    prep,tr,es,ev=pack(fold);save(folder/'identidad.json',identity);save(folder/'escaladores.json',prep.stats)
    M.fijar_semillas(seed,M.Opciones(dispositivo='cuda',hilos=2))
    p=dict(cfg,semilla=seed);model=PatchTSTNumerico(p,prep.stats['sigma_demanda']);history=[]
    def cb(item):
        history.append(item);save(folder/'historia_en_curso.json',history);state(epoca=item['epoca'],loss_train=item['train_loss'],loss_es=item['val_loss'])
    started=time.time()
    try:hist=entrenar(model,tr,es,p,cb,deadline=DEADLINE)
    except TimeoutError:
        torch.save(dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},configuracion=cfg,semilla=seed,escaladores=prep.stats,parcial=True),folder/'modelo_parcial.pt')
        save(folder/'historia_parcial.json',history)
        raise
    q=predecir(model,ev,cfg['batch_size']);assert np.isfinite(q).all()
    qscale=CONTRACT['metrica']['q168_kwh'];metrics=evaluar(ev.y,q,qscale)
    result=dict(**identity,segundos=time.time()-started,n_parametros=sum(v.numel() for v in model.parameters()),epocas=len(hist),mejor_epoca=int(np.argmin([v['val_loss'] for v in hist])+1),**metrics)
    torch.save(dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},configuracion=cfg,semilla=seed,escaladores=prep.stats,arquitectura=ARQUITECTURA,modelo='PatchTST_adaptado'),folder/'modelo.pt')
    np.savez_compressed(folder/'predicciones.npz',y=ev.y,yhat=q,origen=ev.origen.as_unit('ns').asi8)
    save(folder/'historia.json',hist);save(folder/'resultado.json',result);DONE+=1;state()
    print(stage,cfg['id'],seed,fold,round(metrics['MASE'],4),round(result['segundos'],1),flush=True)
    return result

def table(name,rows):
    pd.DataFrame([{**r,'configuracion':json.dumps(r['configuracion'],ensure_ascii=False)} for r in rows]).to_csv(OUT/name,index=False)
ARQUITECTURA=dict(origen='yuqinie98/PatchTST',revision='204c21efe0b39603ad6e2ca640ef5896646ab1a9',configuracion='scripts/PatchTST/etth1.sh',capas=3,d_model=16,n_heads=4,d_ff=128,patch_len=16,stride=8,head='flatten compartido original + lectura afín exógena fija',revin=True,affine=False,historia=168,horizonte=24)

def main():
    global PANEL,REG,CONTRACT,TOTAL,DEADLINE
    lock=(OUT/'gpu.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (OUT/'ejecucion_completa.json').exists():return
    assert torch.cuda.is_available(),'Se requiere GPU'
    CONTRACT=json.loads((ROOT/'config/diseno_entrenamiento_texto.json').read_text())
    paths=['PLAN_PATCHTST_NUMERICO_R1.md','config/diseno_entrenamiento_texto.json','src/demanda/patchtst_numerico.py','src/demanda/potencia_pareada.py','scripts/entrenar_patchtst_numerico.py','src/demanda/diseno_numerico.py','src/demanda/particion.py','src/demanda/modelos_base.py','data/processed/panel_horario.parquet','data/processed/feature_registry.json']
    paths+=[str(p.relative_to(ROOT)) for p in (ROOT/'src/demanda/vendor/patchtst').glob('*') if p.is_file()]
    frozen=dict(sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},arquitectura=ARQUITECTURA,configuraciones=CONFIG)
    if (OUT/'plan.json').exists():assert json.loads((OUT/'plan.json').read_text())==frozen,'Fuentes o configuración modificadas'
    save(OUT/'plan.json',frozen);save(OUT/'hardware.json',dict(torch=torch.__version__,hip=torch.version.hip,nombre=torch.cuda.get_device_name(),memoria=torch.cuda.get_device_properties(0).total_memory))
    PANEL=pd.read_parquet(ROOT/'data/processed/panel_horario.parquet');REG=json.loads((ROOT/'data/processed/feature_registry.json').read_text())
    assert frozen['sha256']['data/processed/panel_horario.parquet']==CONTRACT['panel_sha256']
    budget=OUT/'limite_tiempo.json'
    if not budget.exists():save(budget,dict(inicio=time.time(),deadline=time.time()+24*3600,horas=24))
    DEADLINE=json.loads(budget.read_text())['deadline']
    if time.time()>=DEADLINE:raise TimeoutError('Presupuesto de24horas agotado; se requiere autorización para renovar el plazo')
    state(estado='Entrenando');rows=[];scores=[]
    # Agrupación por fold reduce recomputación sin cambiar candidatos, semillas o ponderación.
    for year in (2018,2019,2020,2021):
        for cfg in CONFIG:
            for seed in (43,44,45):rows.append(fit('busqueda',cfg,seed,year));table('busqueda.csv',rows)
    for cfg in CONFIG:scores.append((float(np.mean([r['MASE'] for r in rows if r['configuracion']==cfg])),cfg))
    score,best=min(scores,key=lambda x:(x[0],x[1]['id']));save(OUT/'mejor_config.json',dict(configuracion=best,MASE_cv=score))
    pilot=[];losses={s:[] for s in range(100,105)};refs=[];years=[]
    for year in (2018,2019,2020,2021):
        for seed in range(100,105):
            result=fit('piloto',best,seed,year);pilot.append(result);table('piloto.csv',pilot)
            key=hashlib.sha256(json.dumps(dict(etapa='piloto',configuracion=best,semilla=seed,fold=year,modelo='PatchTST_adaptado'),sort_keys=True).encode()).hexdigest()[:20]
            with np.load(OUT/'ajustes/piloto'/key/'predicciones.npz') as z:losses[seed].append(np.abs(z['yhat']-z['y']).mean(1)/CONTRACT['metrica']['q168_kwh'])
        prep,tr,es,ev=pack(year);weekly=references(ev)['ingenuo_168h'];refs.append(np.abs(weekly-ev.y).mean(1)/CONTRACT['metrica']['q168_kwh']);years.extend([year]*len(ev.y))
    state(etapa='dimensionamiento',detalle='Bootstrap pareado por semilla y bloques temporales de14d;10000remuestreos')
    dim=dimensionar(np.stack([np.concatenate(losses[s]) for s in losses]),np.concatenate(refs),np.asarray(years));save(OUT/'dimensionamiento.json',dim)
    save(OUT/'seleccion_cerrada.json',dict(mejor=best,MASE_cv=score,dimensionamiento=dim,fecha=S['actualizado']))
    if dim['n_final'] is None:
        state(estado='Requiere decisión',detalle='El piloto no permite acreditar el objetivo de precisión con más semillas en el mismo periodo. Prueba no ejecutada. Consultar dimensionamiento.json.');return
    n=dim['n_final'];TOTAL=260+n;final=[]
    for seed in range(1000,1000+n):final.append(fit('final',best,seed,'final'));table('final_runs.csv',final)
    prep,tr,es,ev=pack('final');reference={name:evaluar(ev.y,q,CONTRACT['metrica']['q168_kwh']) for name,q in references(ev).items()};save(OUT/'referencias_test.json',reference)
    vals=np.array([r['MASE'] for r in final]);summary=dict(modelo='PatchTST_adaptado',n_final=n,MASE_validacion=score,MASE_prueba=float(vals.mean()),MASE_sd=float(vals.std(ddof=1)),MAE_prueba=float(np.mean([r['MAE'] for r in final])),dimensionamiento=dim,nota='Reevaluación retrospectiva. El piloto no certifica potencia futura ni una exención E2 del adaptador afín.')
    save(OUT/'resumen_final.json',summary)
    (OUT/'RESUMEN.md').write_text('# PatchTST numérico adaptado\n\n'+json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
    save(OUT/'ejecucion_completa.json',dict(ajustes=DONE,semillas=n,fecha=S['actualizado']));state(estado='Completado',detalle='Resultados y artefactos guardados.')
if __name__=='__main__':
    try:main()
    except TimeoutError as e:state(estado='Presupuesto temporal agotado',detalle=str(e))
    except BaseException as e:state(estado='Error',detalle=str(e));traceback.print_exc();raise

"""Ruta 1: TFT adaptado, núcleo fijo y búsqueda de hiperparámetros."""
import fcntl,hashlib,html,json,os,sys,time,traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from demanda.tft_numerico import Preparador,TFTNumerico,entrenar,predecir,ARQUITECTURA
from demanda import particion as P,modelos_base as M
from demanda.metricas import evaluar
from demanda.potencia_pareada import dimensionar
from demanda.seleccion_escalonada import clasificar
OUT=ROOT/'results/tft_numerico_r1';OUT.mkdir(parents=True,exist_ok=True)
S={};DONE=0;TOTAL=130;DEADLINE=None

def save(p,obj):
    temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str));temp.replace(p)
def state(**kw):
    S.update(kw);S.update(completados=DONE,total=TOTAL,actualizado=pd.Timestamp.now(tz='America/Bogota').isoformat(),pid=os.getpid())
    save(OUT/'estado.json',S)
    txt=' · '.join(f'{k}: {v}' for k,v in S.items() if k in ('estado','etapa','configuracion','semilla','fold','epoca'))
    body=f'''<!doctype html><html lang="es"><meta charset="utf-8"><meta http-equiv="refresh" content="8"><title>TFT numérico</title><style>body{{font:18px system-ui;background:#14202c;color:#eef5fa;max-width:950px;margin:50px auto;padding:20px}}progress{{width:100%;height:32px}}p{{line-height:1.7}}</style><h1>TFT · numéricas ampliadas · GPU</h1><progress value="{DONE}" max="{TOTAL}"></progress><p>{DONE}/{TOTAL} ajustes previstos · {100*DONE/max(TOTAL,1):.1f}%</p><p>{html.escape(txt)}</p><p>Filtro2018:20configuraciones × 3semillas = 60ajustes.<br>Confirmación2019–2021:5finalistas × 3semillas × 3pliegues = 45ajustes.<br>Piloto: 5 semillas × 4 pliegues = 20 ajustes.<br>Prueba: R semillas dimensionadas; cinco exploratorias si no es dimensionable.</p><p>30 variables históricas + 23 de calendario + 30 máscaras. Historia168h · horizonte24h · MAE normalizada · máximo200épocas.<br>TFT compacto: dimensión16, una capa LSTM, cuatro cabezas, selección de variables y atención causal. Evaluación retrospectiva2022–2025.</p><p>{html.escape(S.get('detalle',''))}</p><small>{S['actualizado']}</small></html>'''
    temp=OUT/'progreso.tmp';temp.write_text(body);temp.replace(OUT/'progreso.html')

def configuraciones():
    rng=np.random.default_rng(2026111406);cc=[dict(id='T01',lr=.001,batch_size=32,dropout=.1,weight_decay=.0001)]
    for i in range(1,20):cc.append(dict(id=f'T{i+1:02d}',lr=float(np.exp(rng.uniform(np.log(1e-4),np.log(3e-3)))),batch_size=int(rng.choice([16,32,64])),dropout=float(rng.uniform(.05,.3)),weight_decay=float(np.exp(rng.uniform(np.log(1e-6),np.log(1e-2))))))
    return cc
CONFIG=configuraciones();DATA={};PREPARER=None;PANEL=None;CONTRACT=None

def pack(fold):
    if fold in DATA:return DATA[fold]
    if fold=='final':
        assert (OUT/'seleccion_cerrada.json').exists() and (OUT/'plan_prueba.json').exists(), 'La prueba requiere selección y semillas congeladas'
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
    identity=dict(etapa=stage,configuracion=cfg,semilla=seed,fold=fold,modelo='TFT_adaptado')
    key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:20]
    folder=OUT/'ajustes'/stage/key;folder.mkdir(parents=True,exist_ok=True)
    state(etapa=stage,configuracion=cfg['id'],semilla=seed,fold=fold,epoca=0)
    if (folder/'resultado.json').exists():
        result=json.loads((folder/'resultado.json').read_text());DONE+=1;state();return result
    if DEADLINE is not None and time.time()>=DEADLINE:raise TimeoutError('Presupuesto de24horas agotado')
    prep,tr,es,ev=pack(fold);save(folder/'identidad.json',identity);save(folder/'escaladores.json',prep.stats)
    M.fijar_semillas(seed,M.Opciones(dispositivo='cuda',hilos=2))
    p=dict(cfg,semilla=seed);torch.cuda.reset_peak_memory_stats();model=TFTNumerico(p,prep.stats['sigma_demanda']);history=[]
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
    result=dict(**identity,segundos=time.time()-started,n_parametros=sum(v.numel() for v in model.parameters()),memoria_gpu_max_bytes=torch.cuda.max_memory_allocated(),epocas=len(hist),mejor_epoca=int(np.argmin([v['val_loss'] for v in hist])+1),**metrics)
    torch.save(dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},configuracion=cfg,semilla=seed,escaladores=prep.stats,arquitectura=ARQUITECTURA,modelo='TFT_adaptado'),folder/'modelo.pt')
    np.savez_compressed(folder/'predicciones.npz',y=ev.y,yhat=q,origen=ev.origen.as_unit('ns').asi8,semilla=seed,zona_origen='UTC')
    save(folder/'historia.json',hist);save(folder/'resultado.json',result);DONE+=1;state()
    print(stage,cfg['id'],seed,fold,round(metrics['MASE'],4),round(result['segundos'],1),flush=True)
    return result

def table(name,rows):
    pd.DataFrame([{**r,'configuracion':json.dumps(r['configuracion'],ensure_ascii=False)} for r in rows]).to_csv(OUT/name,index=False)

def main():
    global PANEL,REG,CONTRACT,TOTAL,DEADLINE
    lock=(OUT/'gpu.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (OUT/'ejecucion_completa.json').exists():return
    CONTRACT=json.loads((ROOT/'config/diseno_entrenamiento_texto.json').read_text())
    paths=['PLAN_TFT_NUMERICO_R1.md','src/demanda/seleccion_escalonada.py','config/diseno_entrenamiento_texto.json','src/demanda/tft_numerico.py','src/demanda/patchtst_numerico.py','src/demanda/potencia_pareada.py','scripts/entrenar_tft_numerico.py','src/demanda/diseno_numerico.py','src/demanda/particion.py','src/demanda/modelos_base.py','src/demanda/metricas.py','data/processed/panel_horario.parquet','data/processed/feature_registry.json']
    paths += [str(p.relative_to(ROOT)) for folder in ['tft','patchtst'] for p in (ROOT/'src/demanda/vendor'/folder).glob('*') if p.is_file()]
    frozen=dict(politica_busqueda=dict(filtro_anios=[2018],finalistas=5,confirmacion_anios=[2019,2020,2021],total_preprueba=125),politica_prueba=dict(semilla_inicial=1000,semillas_si_no_dimensionable=5,exploratorio=True),sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},arquitectura=ARQUITECTURA,configuraciones=CONFIG)
    if (OUT/'plan.json').exists():
        assert json.loads((OUT/'plan.json').read_text())==frozen,'Fuentes o configuración modificadas; no se reanuda con cambios'
    save(OUT/'plan.json',frozen)
    if '--preparar' in sys.argv:
        print('Plan TFT congelado; no se ha accedido a prueba ni a la GPU.',flush=True);return
    assert torch.cuda.is_available(),'Se requiere GPU'
    save(OUT/'hardware.json',dict(torch=torch.__version__,hip=torch.version.hip,nombre=torch.cuda.get_device_name(),memoria=torch.cuda.get_device_properties(0).total_memory))
    PANEL=pd.read_parquet(ROOT/'data/processed/panel_horario.parquet');REG=json.loads((ROOT/'data/processed/feature_registry.json').read_text())
    assert frozen['sha256']['data/processed/panel_horario.parquet']==CONTRACT['panel_sha256']
    budget=OUT/'limite_tiempo.json'
    if not budget.exists():save(budget,dict(inicio=time.time(),deadline=time.time()+24*3600,horas=24))
    DEADLINE=json.loads(budget.read_text())['deadline']
    if time.time()>=DEADLINE:raise TimeoutError('Presupuesto de24horas agotado; se requiere autorización para renovar el plazo')
    state(estado='Verificando GPU',etapa='verificacion',detalle='Comprobación técnica después de finalizar N-HiTS')
    M.fijar_semillas(42,M.Opciones(dispositivo='cuda',hilos=2))
    comprobacion=TFTNumerico(dict(dropout=.1),.1).cuda()
    x=torch.randn(64,168,83,device='cuda');cal=torch.randn(64,24,23,device='cuda')
    opt=torch.optim.AdamW(comprobacion.parameters(),lr=.001)
    inicio=time.time();pred=comprobacion(x,cal);loss=(pred-1).abs().mean()
    assert torch.isfinite(loss);loss.backward();opt.step();torch.cuda.synchronize()
    save(OUT/'verificacion_gpu.json',dict(segundos=time.time()-inicio,n_parametros=sum(p.numel() for p in comprobacion.parameters()),batch_size=64,salida=list(pred.shape),determinismo=torch.are_deterministic_algorithms_enabled()))
    del comprobacion,x,cal,opt,pred,loss;torch.cuda.empty_cache()
    state(estado='Entrenando',etapa='busqueda',detalle='N-HiTS finalizado; TFT en ejecución.');rows=[];scores=[]
    # Conservar íntegros los60ajustes del filtro2018.
    for cfg in CONFIG:
        for seed in (43,44,45):rows.append(fit('busqueda',cfg,seed,2018));table('busqueda.csv',rows)
    ranking=clasificar(rows,CONFIG,[2018],[43,44,45]);finalistas=[cfg for _,cfg in ranking[:5]]
    save(OUT/'seleccion_2018.json',dict(criterio='Media de las3semillas en2018;empates por ID',ranking=[dict(MASE=value,configuracion=cfg) for value,cfg in ranking],finalistas=finalistas))
    for year in (2019,2020,2021):
        for cfg in finalistas:
            for seed in (43,44,45):rows.append(fit('busqueda',cfg,seed,year));table('busqueda.csv',rows)
    score,best=clasificar(rows,finalistas,[2019,2020,2021],[43,44,45])[0]
    save(OUT/'mejor_config.json',dict(configuracion=best,MASE_cv=score,anios_confirmacion=[2019,2020,2021],filtro_2018=True))
    pilot=[];losses={s:[] for s in range(100,105)};refs=[];years=[]
    for year in (2018,2019,2020,2021):
        for seed in range(100,105):
            result=fit('piloto',best,seed,year);pilot.append(result);table('piloto.csv',pilot)
            key=hashlib.sha256(json.dumps(dict(etapa='piloto',configuracion=best,semilla=seed,fold=year,modelo='TFT_adaptado'),sort_keys=True).encode()).hexdigest()[:20]
            with np.load(OUT/'ajustes/piloto'/key/'predicciones.npz') as z:losses[seed].append(np.abs(z['yhat']-z['y']).mean(1)/CONTRACT['metrica']['q168_kwh'])
        prep,tr,es,ev=pack(year);weekly=references(ev)['ingenuo_168h'];refs.append(np.abs(weekly-ev.y).mean(1)/CONTRACT['metrica']['q168_kwh']);years.extend([year]*len(ev.y))
    state(etapa='dimensionamiento',detalle='Bootstrap pareado por semilla y bloques temporales de14d;10000remuestreos')
    dim=dimensionar(np.stack([np.concatenate(losses[s]) for s in losses]),np.concatenate(refs),np.asarray(years));save(OUT/'dimensionamiento.json',dim)
    save(OUT/'seleccion_cerrada.json',dict(mejor=best,MASE_cv=score,dimensionamiento=dim,fecha=S['actualizado']))
    n=dim['n_final'] if dim['n_final'] is not None else 5
    TOTAL=125+n;final=[]
    prueba=dict(configuracion=best,semillas=list(range(1000,1000+n)),exploratorio=True,potencia_acreditada=False,dimensionamiento=dim,fecha=pd.Timestamp.now(tz='America/Bogota').isoformat())
    if (OUT/'plan_prueba.json').exists():
        antiguo=json.loads((OUT/'plan_prueba.json').read_text())
        assert antiguo['configuracion']==best and antiguo['semillas']==prueba['semillas']
    else:save(OUT/'plan_prueba.json',prueba)
    state(etapa='final',detalle='Prueba retrospectiva exploratoria; cinco semillas si el piloto no permite dimensionar. No se acredita potencia futura.')
    for seed in range(1000,1000+n):final.append(fit('final',best,seed,'final'));table('final_runs.csv',final)
    prep,tr,es,ev=pack('final');reference={name:evaluar(ev.y,q,CONTRACT['metrica']['q168_kwh']) for name,q in references(ev).items()};save(OUT/'referencias_test.json',reference)
    vals=np.array([r['MASE'] for r in final]);summary=dict(modelo='TFT_adaptado',n_final=n,MASE_validacion=score,anios_confirmacion=[2019,2020,2021],MASE_prueba=float(vals.mean()),MASE_sd=float(vals.std(ddof=1)),MAE_prueba=float(np.mean([r['MAE'] for r in final])),dimensionamiento=dim,nota='Reevaluación retrospectiva exploratoria. El piloto no certifica potencia futura. Arquitectura compacta adaptada desde Nixtla; sin exención E2 automática.')
    summary.update(exploratorio=True,potencia_acreditada=False,semillas=list(range(1000,1000+n)),mejora_ingenuo_semanal_pct=100*(1-vals.mean()/reference['ingenuo_168h']['MASE']))
    for metrica in ['RMSE','sMAPE','MAPE']:summary[metrica+'_prueba']=float(np.mean([r[metrica] for r in final]))
    save(OUT/'resumen_final.json',summary)
    (OUT/'RESUMEN.md').write_text('# TFT numérico adaptado\n\n'+json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
    save(OUT/'ejecucion_completa.json',dict(ajustes=DONE,semillas=n,exploratorio=True,fecha=S['actualizado']));state(estado='Completado',detalle='Resultados y artefactos guardados.')
if __name__=='__main__':
    try:main()
    except TimeoutError as e:state(estado='Presupuesto temporal agotado',detalle=str(e))
    except BaseException as e:state(estado='Error',detalle=str(e));traceback.print_exc();raise

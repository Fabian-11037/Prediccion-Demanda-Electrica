"""Ejecuta el plan v3 acotado, persistente y con progreso por época."""
import csv,fcntl,hashlib,html,json,os,sys,time,traceback
from collections import OrderedDict
from pathlib import Path
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from demanda import modelos_base as M,particion as P
from demanda.reentrenamiento_v3 import DatosFold,Entrenador
from demanda.metricas import evaluar
OUT=ROOT/'results/base_v3';OUT.mkdir(parents=True,exist_ok=True)
STATE={};DONE=0;TOTAL=276;CACHE=OrderedDict()
def guardar(path,obj):
 t=path.with_suffix(path.suffix+'.tmp');t.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str));t.replace(path)
def estado(**kw):
 STATE.update(kw);STATE.update(completados=DONE,total=TOTAL,actualizado=pd.Timestamp.now(tz='America/Bogota').isoformat(),pid=os.getpid())
 guardar(OUT/'estado.json',STATE)
 desc=' · '.join(f'{k}: {v}' for k,v in STATE.items() if k in ('estado','etapa','modelo','configuracion','semilla','fold','epoca'))
 body=f'''<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="5"><title>Reentrenamiento GPU v3</title><style>body{{font:18px system-ui;max-width:950px;margin:60px auto;background:#14202c;color:#eef5fa}}progress{{width:100%;height:30px}}p{{line-height:1.7}}</style><h1>Reentrenamiento GPU v3</h1><progress value="{DONE}" max="{TOTAL}"></progress><p>{DONE}/{TOTAL} ajustes · {DONE/TOTAL:.1%}</p><p>{html.escape(desc)}</p><p>Escalado exclusivo de entrenamiento por fold. Redes pequeñas y regularizadas.<br>Filtro80 · confirmación96 · piloto80 · prueba20.<br>Prueba 2022–2025 observada previamente: reevaluación exploratoria.</p><p>{html.escape(STATE.get('error',''))}</p><small>{STATE['actualizado']}</small>'''
 (OUT/'progreso.html').write_text(body)
def configs():
 result={}
 for m in ('arx_lineal','mlp'):
  rng=np.random.default_rng(1);result[m]=[dict(id=f'{m}_{i+1:02d}',ventana=168,params=M.muestrear(M.ESPACIOS[m],rng)) for i in range(20)]
 opts=[(.001,.1,1e-4,3),(.0003,.1,1e-4,5),(.001,.2,1e-3,3),(.0003,.2,1e-3,5),(.001,.3,1e-3,3)]
 for m in ('lstm','cnn1d'):
  v=[]
  for w in ((16,32) if m=='lstm' else (8,16)):
   for L in (72,168):
    for lr,drop,wd,k in opts:
     p=dict(capas=1,dropout=drop,weight_decay=wd,batch_size=256,lr=lr,scheduler='plateau',perdida='huber',paciencia=6)
     if m=='lstm':p['unidades']=w
     else:p.update(filtros=w,kernel=k)
     v.append(dict(id=f'{m}_{len(v)+1:02d}',ventana=L,params=p))
  result[m]=v
 return result
CONFIG=configs()
PANEL=None;REG=None

def datos_para(m,L,fold):
 key=(m in ('lstm','cnn1d'),L,fold)
 if key in CACHE:CACHE.move_to_end(key);return CACHE[key]
 cfg=M.ConfigDatos(ventana=L)
 if fold=='final':tr,es=P.pretest(PANEL.index,ventana=L);ev=P.origenes(PANEL.index,'test',ventana=L,allow_test=True)
 else:
  f=P.folds_forward_chaining(PANEL.index,[fold],ventana=L)[0];tr,es,ev=f['tr'],f['es'],f['val']
 d=DatosFold(PANEL,REG,cfg,tr,secuencial=key[0]);train=d.conjunto_pos(tr);stop=d.conjunto_pos(es);validation=d.conjunto_pos(ev)
 assert pd.Timestamp(d.auditoria['fin_ajuste'])<stop.origen.min()
 assert train.origen.max()+pd.Timedelta(hours=23)<stop.origen.min()
 assert stop.origen.max()+pd.Timedelta(hours=23)<validation.origen.min()
 pack=(d,train,stop,validation);CACHE[key]=pack
 while len(CACHE)>2:CACHE.popitem(last=False)
 return pack

def ajuste(stage,m,config,seed,fold,epochs):
 global DONE
 identity=dict(etapa=stage,modelo=m,configuracion=config,semilla=seed,fold=fold,max_epocas=epochs)
 key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:20];folder=OUT/'ajustes'/stage/key;folder.mkdir(parents=True,exist_ok=True)
 estado(etapa=stage,modelo=m,configuracion=config['id'],semilla=seed,fold=fold,epoca=0)
 if (folder/'resultado.json').exists():
  r=json.loads((folder/'resultado.json').read_text());DONE+=1;estado();return r
 d,tr,es,ev=datos_para(m,config['ventana'],fold)
 guardar(folder/'identidad.json',identity);guardar(folder/'escaladores.json',d.escaladores())
 op=M.Opciones(max_epocas=epochs,dispositivo='cuda',hilos=2)
 model=Entrenador(m)
 history=[]
 def cb(info):
  history.append(info);guardar(folder/'historia_en_curso.json',history);estado(epoca=info['epoca'],loss_train=info['train_loss'],loss_es=info['val_loss'])
 model.observador=cb;start=time.time();info=model.ajustar(tr,es,d,config['params'],seed,op)
 q=d.invertir(model.predecir(ev),ev)
 assert np.isfinite(q).all()
 metrics=evaluar(ev.y,q,d.escala_mase)
 r=dict(etapa=stage,modelo=m,configuracion=config['id'],semilla=seed,fold=fold,ventana=config['ventana'],params=config['params'],segundos=time.time()-start,**metrics,**{k:v for k,v in info.items() if k!='historia'})
 # Separar pérdidas de ES de la evaluación anual; no confundir dropout activo con brecha de generalización.
 if stage in ('piloto','final'):
  train_pred=model.predecir(tr);es_pred=model.predecir(es)
  r['MAE_train_normalizado_eval']=float(np.abs(train_pred-d.objetivo_norm(tr)).mean())
  r['MAE_es_normalizado_eval']=float(np.abs(es_pred-d.objetivo_norm(es)).mean())
 if stage=='piloto':
  from dataclasses import replace
  perm=np.random.default_rng(2026).permutation(len(ev));xx=ev.x.copy();xx[:,:,0]=xx[perm,:,0]
  modified=replace(ev,x=xx)
  r['MASE_demanda_permutada']=evaluar(ev.y,d.invertir(model.predecir(modified),ev),d.escala_mase)['MASE']
  r['MASE_referencia_168h']=evaluar(ev.y,d.referencias_pos(PANEL.index.get_indexer(ev.origen),ev)['ingenuo_168h'],d.escala_mase)['MASE']
 np.savez_compressed(folder/'predicciones.npz',y=ev.y,yhat=q,origen=ev.origen.as_unit('ns').asi8)
 guardar(folder/'historia.json',info['historia'])
 torch.save(dict(state_dict={k:v.detach().cpu() for k,v in model.red.state_dict().items()},modelo=m,params=config['params'],escaladores=d.escaladores(),version=3,informacion={k:v for k,v in info.items() if k!='historia'}),folder/'modelo.pt')
 guardar(folder/'resultado.json',r);DONE+=1;estado()
 print(stage,m,config['id'],seed,fold,'MASE',round(r['MASE'],4),'s',round(r['segundos'],1),flush=True)
 return r

def tabla(path,rows):pd.DataFrame(rows).to_csv(OUT/path,index=False)
def main():
 global PANEL,REG
 lock=(OUT/'gpu.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if (OUT/'ejecucion_completa.json').exists():print('Ejecución ya completa');return
 assert torch.cuda.is_available(),'Se requiere GPU'
 sources=['PLAN_REENTRENAMIENTO_V3.md','src/demanda/reentrenamiento_v3.py','src/demanda/modelos_base.py','scripts/reentrenar_v3.py','data/processed/panel_horario.parquet','data/processed/feature_registry.json']
 plan=dict(configuraciones=CONFIG,sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources},total_ajustes=TOTAL,n_final=5)
 if (OUT/'plan.json').exists():assert json.loads((OUT/'plan.json').read_text())==plan,'Plan o fuentes modificados'
 guardar(OUT/'plan.json',plan);guardar(OUT/'configuraciones.json',CONFIG)
 guardar(OUT/'hardware.json',dict(torch=torch.__version__,hip=torch.version.hip,gpu=torch.cuda.get_device_name(),memoria=torch.cuda.get_device_properties(0).total_memory))
 PANEL=pd.read_parquet(ROOT/'data/processed/panel_horario.parquet');REG=json.loads((ROOT/'data/processed/feature_registry.json').read_text())
 estado(estado='Entrenando')
 filtered=[];candidates={}
 for m,cc in CONFIG.items():
  scores=[]
  for conf in cc:
   r=ajuste('filtro',m,conf,42,2019,30);filtered.append(r);tabla('filtro.csv',filtered);scores.append((r['MASE'],conf))
  candidates[m]=[c for _,c in sorted(scores,key=lambda t:t[0])[:3]]
 guardar(OUT/'finalistas.json',candidates)
 confirmed=[];best={}
 for m,cc in candidates.items():
  scores=[]
  for conf in cc:
   rr=[]
   for year in (2018,2019,2020,2021):
    for seed in (43,44):
     r=ajuste('confirmacion',m,conf,seed,year,60);confirmed.append(r);rr.append(r['MASE']);tabla('confirmacion.csv',confirmed)
   scores.append((float(np.mean(rr)),conf))
  value,c=min(scores,key=lambda t:t[0]);best[m]=dict(configuracion=c,MASE_cv=value)
 guardar(OUT/'mejor_config.json',best)
 piloto=[]
 for m,b in best.items():
  for year in (2018,2019,2020,2021):
   for seed in range(100,105):
    piloto.append(ajuste('piloto',m,b['configuracion'],seed,year,60));tabla('piloto.csv',piloto)
 guardar(OUT/'seleccion_cerrada.json',dict(mejores=best,fecha=STATE['actualizado'],nota='Selección y piloto completos antes de acceder a prueba en v3'))
 final=[];summ=[]
 for m,b in best.items():
  for seed in range(1000,1005):
   final.append(ajuste('final',m,b['configuracion'],seed,'final',60));tabla('final_runs.csv',final)
  d,tr,es,te=datos_para(m,b['configuracion']['ventana'],'final')
  rr={name:evaluar(te.y,pred,d.escala_mase) for name,pred in d.referencias_pos(PANEL.index.get_indexer(te.origen),te).items()}
  guardar(OUT/'referencias_test.json',rr)
 p=pd.DataFrame(piloto);f=pd.DataFrame(final)
 original=pd.read_csv(ROOT/'results/base_v2/resumen_final.csv').set_index('modelo')
 for m in CONFIG:
  pp=p[p.modelo==m];ff=f[f.modelo==m]
  v=dict(modelo=m,n_final=len(ff),MASE_cv_seleccion=best[m]['MASE_cv'],MASE_validacion_piloto=pp.MASE.mean(),MASE_prueba=ff.MASE.mean(),MASE_sd_prueba=ff.MASE.std(),MAE_prueba=ff.MAE.mean(),sMAPE_prueba=ff.sMAPE.mean(),MASE_prueba_v2=float(original.loc[m,'MASE_media']),MAE_train_normalizado_eval=ff.MAE_train_normalizado_eval.mean(),MAE_es_normalizado_eval=ff.MAE_es_normalizado_eval.mean(),MASE_demanda_permutada_validacion=pp.MASE_demanda_permutada.mean())
  v['mejora_MASE_vs_v2_pct']=100*(1-v['MASE_prueba']/v['MASE_prueba_v2']);v['mejora_vs_referencia_semanal_pct']=100*(1-v['MASE_prueba']/rr['ingenuo_168h']['MASE']);summ.append(v)
 tabla('resumen_final.csv',summ)
 lines=['# Resultados exploratorios v3','', 'Escalado por fold exclusivamente con entrenamiento. Validación 2018–2021; reevaluación de prueba 2022–2025 previamente observada. Cinco semillas finales por familia; sin afirmación de potencia estadística ni confirmación independiente.','',pd.DataFrame(summ).to_markdown(index=False),'','La comparación con v2 combina correcciones de escalado, presupuesto y cambios de arquitectura; no permite atribuir la diferencia a un único factor. Los errores train/ES se calculan con dropout desactivado. Predicciones, historiales, pesos y escaladores se conservan por ajuste en ajustes/.']
 (OUT/'RESUMEN.md').write_text('\n'.join(lines))
 guardar(OUT/'ejecucion_completa.json',dict(ajustes=DONE,fecha=STATE['actualizado'],reevaluacion_exploratoria=True))
 estado(estado='Completado',epoca=0)
 print(pd.DataFrame(summ).to_string(index=False),flush=True)
if __name__=='__main__':
 try:main()
 except BaseException as exc:
  estado(estado='Error',error=str(exc));traceback.print_exc();raise

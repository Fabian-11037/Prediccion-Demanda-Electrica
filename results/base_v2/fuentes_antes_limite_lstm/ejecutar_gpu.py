"""Ejecuta búsqueda, piloto y final v2 con observación y persistencia por ajuste."""
import fcntl,hashlib,html,importlib.util,json,os,sys,time,traceback
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
spec=importlib.util.spec_from_file_location('entrenador',ROOT/'scripts/entrenar_modelos_base.py')
E=importlib.util.module_from_spec(spec);spec.loader.exec_module(E)
M=E.M
OUT=ROOT/'results/base_v2'
STATE={}
ACTIVE={}

def guardar(path,values):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(values,ensure_ascii=False,indent=2,default=str));temp.replace(path)

def publicar(**values):
    STATE.update(values);STATE['actualizado']=E.pd.Timestamp.now(tz='America/Bogota').isoformat()
    guardar(OUT/'estado_gpu.json',STATE)
    total=STATE.get('total_etapa',1);done=STATE.get('completados_etapa',0)
    current=STATE.get('ajuste',{})
    description=' · '.join(f'{k}: {v}' for k,v in current.items())
    body=f'''<!doctype html><html lang="es"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><meta http-equiv="refresh" content="5"><title>Entrenamiento de demanda eléctrica</title><style>body{{font:18px system-ui;max-width:940px;margin:48px auto;padding:24px;background:#14202c;color:#eff5fa}}progress{{width:100%;height:32px;accent-color:#60c4ed}}p{{line-height:1.65}}small{{color:#b2c5d4}}</style><h1>Pronóstico de demanda eléctrica · GPU</h1><p>{html.escape(STATE.get('estado','Preparación'))} · etapa {html.escape(STATE.get('etapa','preparación'))}</p><progress value="{done}" max="{total}"></progress><p><strong>{done} / {total} ajustes de la etapa · {100*done/max(total,1):.2f}%</strong></p><p>{html.escape(description)}</p><p>Búsqueda: 20 configuraciones × 4 modelos × 3 semillas × 4 pliegues = 960 ajustes.<br>Piloto: 5 semillas × 4 modelos × 4 pliegues = 80 ajustes.<br>Etapa final: número de semillas determinado por el piloto, mínimo 5 por modelo.</p><p>Máximo 200 épocas · parada temprana en tramo interno · criterio MASE.<br>Demanda histórica y calendario · prueba sólo en etapa final preregistrada.</p><small>Actualización: {html.escape(STATE['actualizado'])} · refresco cada 5 segundos</small></html>'''
    temp=OUT/'progreso.tmp';temp.write_text(body);temp.replace(OUT/'progreso.html')

def epoca(info):
    ACTIVE['historia'].append(dict(info))
    publicar(ajuste=dict(ACTIVE['descripcion'],epoca=info['epoca'],perdida_entrenamiento=round(info['train_loss'],8),
        perdida_parada=round(info['val_loss'],8),mejor_perdida=round(min(v['val_loss'] for v in ACTIVE['historia']),8)))
    guardar(ACTIVE['carpeta']/'historia_en_curso.json',ACTIVE['historia'])

ORIGINAL_AJUSTAR=M.ModeloTorch.ajustar
ORIGINAL_EVALUAR=M.entrenar_y_evaluar

def ajustar(self,tr,va,datos,p,semilla,op):
    result=ORIGINAL_AJUSTAR(self,tr,va,datos,p,semilla,op)
    torch.save(dict(state_dict={k:v.detach().cpu() for k,v in self.red.state_dict().items()},modelo=self.nombre,
        params=p,semilla=semilla,datos=asdict(datos.cfg),columnas_x=datos.cols_x,columnas_cal=datos.cols_cal,
        media=datos.escala.media_.to_dict(),std=datos.escala.std_.to_dict(),escala_mase=datos.escala_mase,
        informacion={k:v for k,v in result.items() if k!='historia'}),ACTIVE['carpeta']/'modelo.pt')
    return result

def evaluar(nombre,params,semilla,datos,op,tr,es,ev,guardar_pred=False):
    identity=dict(etapa=STATE['etapa'],modelo=nombre,params=params,semilla=semilla,
        inicio_evaluacion=str(ev.origen[0]),fin_evaluacion=str(ev.origen[-1]),n_train=len(tr),n_es=len(es),n_eval=len(ev))
    key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:20]
    folder=OUT/'ajustes'/STATE['etapa']/key;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'resultado.json').exists():
        result=json.loads((folder/'resultado.json').read_text())
        result['_historia']=json.loads((folder/'historia.json').read_text())
        if guardar_pred:
            with np.load(folder/'predicciones.npz') as p:result['_yhat']=p['yhat']
    else:
        ACTIVE.clear();ACTIVE.update(carpeta=folder,historia=[],descripcion=dict(modelo=nombre,semilla=semilla,
            pliegue=str(ev.origen[0])[:4],id=key))
        guardar(folder/'identidad.json',identity)
        publicar(ajuste=dict(ACTIVE['descripcion'],epoca=0))
        result=ORIGINAL_EVALUAR(nombre,params,semilla,datos,op,tr,es,ev,guardar_pred=True)
        np.savez_compressed(folder/'predicciones.npz',y=ev.y,yhat=result['_yhat'],origen=ev.origen.astype('int64'))
        guardar(folder/'historia.json',result['_historia'])
        guardar(folder/'resultado.json',{k:v for k,v in result.items() if not k.startswith('_')})
        if not guardar_pred:result.pop('_yhat',None)
    publicar(completados_etapa=STATE['completados_etapa']+1)
    return result

def main():
    OUT.mkdir(exist_ok=True,parents=True)
    lock=(OUT/'gpu.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (OUT/'ejecucion_completa.json').exists():print('La ejecución v2 ya está completa; no se repite prueba.');return
    if not torch.cuda.is_available():raise RuntimeError('La GPU es obligatoria para esta ejecución')
    source_paths=['PREREGISTRO.md','src/demanda/modelos_base.py','src/demanda/particion.py','src/demanda/ventanas.py',
        'scripts/entrenar_modelos_base.py','scripts/ejecutar_gpu.py','PREREGISTRO_CNN1D.md','data/processed/panel_horario.parquet','data/processed/feature_registry.json']
    frozen=dict(sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in source_paths},
        dispositivo='cuda',determinismo='estricto',n_config=20,semillas_por_config=3,max_epocas=200,
        modelos=list(M.MODELOS_PYTORCH),delta=.03,amp=False)
    if (OUT/'plan_gpu.json').exists() and json.loads((OUT/'plan_gpu.json').read_text())!=frozen:
        anterior=json.loads((OUT/'plan_gpu.json').read_text())
        if '--incorporar-cnn' not in sys.argv:
            raise RuntimeError('Las fuentes o el plan difieren de la ejecución existente')
        assert anterior['modelos']==['arx_lineal','mlp','lstm']
        assert frozen['modelos']==['arx_lineal','mlp','lstm','cnn1d']
        assert not (OUT/'final_completo.json').exists()
        for key,value in anterior.items():
            if key not in ['modelos','sha256']:assert frozen[key]==value,key
        cambiables={'src/demanda/modelos_base.py','scripts/ejecutar_gpu.py'}
        for key,value in anterior['sha256'].items():
            if key not in cambiables:assert frozen['sha256'][key]==value,key
        guardar(OUT/'incorporacion_cnn.json',dict(plan_anterior=anterior,plan_ampliado=frozen,
            motivo='CNN1D y espacio confirmados antes de su entrenamiento',
            fecha=E.pd.Timestamp.now(tz='America/Bogota').isoformat()))
    guardar(OUT/'plan_gpu.json',frozen)
    guardar(OUT/'hardware_gpu.json',dict(torch=torch.__version__,hip=torch.version.hip,gpu=torch.cuda.get_device_name(0),
        memoria=torch.cuda.get_device_properties(0).total_memory,compatibilidad=os.environ.get('HSA_OVERRIDE_GFX_VERSION')))
    M.PROGRESO_ENTRENAMIENTO=epoca;M.ModeloTorch.ajustar=ajustar;M.entrenar_y_evaluar=evaluar
    common=['--salida',str(OUT),'--dispositivo','cuda','--determinismo','estricto','--max-epocas','200',
        '--n-config','20','--semillas-por-config','3','--delta','0.03','--prereg',str(ROOT/'PREREGISTRO.md')]
    started=time.time()
    for stage,call in [('buscar',E.etapa_buscar),('piloto',E.etapa_piloto),('final',E.etapa_final)]:
        if (OUT/f'{stage}_completo.json').exists():continue
        total=960 if stage=='buscar' else 80
        done=0
        if stage=='buscar' and (OUT/'busqueda.csv').exists():done=len(E.pd.read_csv(OUT/'busqueda.csv'))*12
        if stage=='final':total=4*max(5,json.loads((OUT/'piloto_resumen.json').read_text())['n_final'])
        publicar(estado='Entrenando',etapa=stage,total_etapa=total,completados_etapa=done,pid=os.getpid(),ajuste={})
        args=E.parser().parse_args([stage,*common,*(['--allow-test'] if stage=='final' else [])])
        call(args)
        guardar(OUT/f'{stage}_completo.json',dict(ajustes=total,fecha=STATE['actualizado']))
    publicar(estado='Completado',ajuste={})
    guardar(OUT/'ejecucion_completa.json',dict(segundos=time.time()-started,final_preregistrado=True))

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        if OUT.exists():publicar(estado='Error',error=str(exc))
        traceback.print_exc();raise

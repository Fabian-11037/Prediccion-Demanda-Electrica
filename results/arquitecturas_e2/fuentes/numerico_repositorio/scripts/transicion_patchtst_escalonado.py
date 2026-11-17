"""Termina el ajuste PatchTST activo y reanuda con la selección autorizada."""
import fcntl,hashlib,json,os,signal,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/patchtst_numerico_r1'
def registrar(**data):
    p=OUT/'transicion_escalonada.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(dict(fecha=time.time(),**data),indent=2));t.replace(p)
if __name__=='__main__':
    lock=(OUT/'transicion.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=json.loads((OUT/'estado.json').read_text());services=json.loads((OUT/'servicios_locales.json').read_text())
    assert state['estado']=='Entrenando' and state['etapa']=='busqueda' and state['fold']==2018
    cfg=next(c for c in json.loads((OUT/'configuraciones.json').read_text()) if c['id']==state['configuracion'])
    identity=dict(etapa='busqueda',configuracion=cfg,semilla=state['semilla'],fold=2018,modelo='PatchTST_adaptado')
    key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:20]
    target=OUT/'ajustes/busqueda'/key/'resultado.json';pid=services['pid_entrenamiento']
    registrar(estado='esperando_fin_ajuste_activo',pid_monitor=os.getpid(),pid_original=pid,configuracion=cfg['id'],semilla=state['semilla'],ajuste=key)
    while not target.exists():
        state=json.loads((OUT/'estado.json').read_text())
        if state['estado'] in ('Error','Presupuesto temporal agotado'):
            registrar(estado='cancelado_por_estado_entrenamiento',detalle=state);raise SystemExit(1)
        os.kill(pid,0);time.sleep(2)
    cmd=Path(f'/proc/{pid}/cmdline').read_bytes()
    assert b'entrenar_patchtst_numerico.py' in cmd,'PID no corresponde al entrenamiento esperado'
    os.kill(pid,signal.SIGTERM)
    gpu=(OUT/'gpu.lock').open('w');fcntl.flock(gpu,fcntl.LOCK_EX);fcntl.flock(gpu,fcntl.LOCK_UN);gpu.close()
    env=os.environ.copy();env['HSA_OVERRIDE_GFX_VERSION']='10.3.0';env['PYTHONPATH']=str(ROOT.parent/'.demanda-python-deps')
    with (OUT/'entrenamiento.log').open('a') as log:
        p=subprocess.Popen(['/home/fabian/Documents/Machine Learning/mlenv/bin/python','-u','scripts/entrenar_patchtst_numerico.py','--seleccion-escalonada'],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
    services['pid_entrenamiento']=p.pid;(OUT/'servicios_locales.json').write_text(json.dumps(services,indent=2))
    registrar(estado='reanudado',pid_entrenamiento=p.pid,ajuste_completado=key)

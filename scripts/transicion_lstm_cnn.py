"""Completa la configuración LSTM activa y reanuda con presupuesto reducido."""
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/base_v2'
def guardar(nombre, datos):
    p=OUT/nombre;t=p.with_suffix('.tmp');t.write_text(json.dumps(datos,indent=2));t.replace(p)
if __name__=='__main__':
    lock=(OUT/'transicion.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    guardar('transicion_lstm.json',dict(estado='esperando_configuracion_5',pid=os.getpid()))
    while True:
        estado=json.loads((OUT/'estado_gpu.json').read_text())
        if estado['estado']=='Error':
            guardar('transicion_lstm.json',dict(estado='error_entrenamiento'));raise SystemExit(1)
        try:
            with (OUT/'busqueda.csv').open() as f: rows=list(csv.DictReader(f))
            completo=any(r['modelo']=='lstm' and int(r['config'])==4 for r in rows)
        except (OSError,ValueError,KeyError): completo=False
        if completo:break
        time.sleep(2)
    plan=json.loads((OUT/'plan_gpu.json').read_text())
    cambios={'scripts/entrenar_modelos_base.py','scripts/ejecutar_gpu.py'}
    for p,h in plan['sha256'].items():
        actual=hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
        if p not in cambios:assert actual==h,p
    pid=json.loads((OUT/'servicios_locales.json').read_text())['pid_entrenamiento']
    os.kill(pid,signal.SIGTERM)
    gpu_lock=(OUT/'gpu.lock').open('w')
    fcntl.flock(gpu_lock,fcntl.LOCK_EX)
    nuevo=json.loads(json.dumps(plan))
    for p in cambios:nuevo['sha256'][p]=hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
    guardar('reduccion_lstm.json',dict(plan_anterior=plan,plan_actual=nuevo,limites={'lstm':5},motivo='Solicitud del autor: completar configuración activa y proceder a CNN1D',fecha=time.time()))
    guardar('plan_gpu.json',nuevo)
    fcntl.flock(gpu_lock,fcntl.LOCK_UN);gpu_lock.close()
    env=os.environ.copy();env['HSA_OVERRIDE_GFX_VERSION']='10.3.0';env['PYTHONPATH']=str(ROOT.parent/'.demanda-python-deps')
    with (OUT/'entrenamiento_gpu.log').open('a') as log:
        proc=subprocess.Popen(['/home/fabian/Documents/Machine Learning/mlenv/bin/python','-u','scripts/ejecutar_gpu.py'],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
    servicios=json.loads((OUT/'servicios_locales.json').read_text());servicios['pid_entrenamiento']=proc.pid;guardar('servicios_locales.json',servicios)
    guardar('transicion_lstm.json',dict(estado='reanudado',pid_entrenamiento=proc.pid,fecha=time.time()))

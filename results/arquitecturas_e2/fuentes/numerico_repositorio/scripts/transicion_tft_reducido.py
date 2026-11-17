"""Espera el ajuste activo, conserva resultados y reanuda TFT con62ajustes."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/tft_numerico_r1'


def leer(nombre):
    return json.loads((OUT / nombre).read_text())


def registrar(estado, **datos):
    p = OUT / 'transicion_reducida.json'
    t = p.with_suffix('.tmp')
    t.write_text(json.dumps(dict(estado=estado, fecha=time.time(), pid=os.getpid(), **datos), indent=2))
    t.replace(p)


def coincide(pid, nombre):
    try:
        stat = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return stat[0] != 'Z' and nombre.encode() in Path(f'/proc/{pid}/cmdline').read_bytes()
    except OSError:
        return False


def main():
    with (OUT / 'transicion_reducida.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert (OUT / 'plan_reducido.json').exists()
        estado = leer('estado.json')
        servicios = leer('servicios_locales.json')
        pid = servicios['pid_entrenamiento']
        assert estado['estado'] == 'Entrenando' and estado['etapa'] == 'busqueda'
        assert coincide(pid, 'entrenar_tft_numerico.py')
        cfg = next(c for c in leer('plan.json')['configuraciones'] if c['id'] == estado['configuracion'])
        identidad = dict(etapa='busqueda', configuracion=cfg, semilla=estado['semilla'],
                         fold=estado['fold'], modelo='TFT_adaptado')
        key = hashlib.sha256(json.dumps(identidad, sort_keys=True).encode()).hexdigest()[:20]
        objetivo = OUT / 'ajustes/busqueda' / key / 'resultado.json'
        registrar('esperando_fin_ajuste_activo', pid_original=pid, configuracion=cfg['id'], semilla=estado['semilla'], ajuste=key)
        while not objetivo.exists():
            actual = leer('estado.json')
            assert actual['estado'] == 'Entrenando', actual
            assert coincide(pid, 'entrenar_tft_numerico.py'), 'Proceso original desaparecido'
            time.sleep(2)
        carpeta = objetivo.parent
        assert all((carpeta / n).is_file() for n in ['modelo.pt', 'historia.json', 'predicciones.npz', 'escaladores.json'])
        # Detener solo el monitor de la instancia sustituida; el nuevo se inicia después.
        apagado = servicios.get('pid_apagado')
        if apagado and coincide(apagado, 'apagar_patchtst_al_terminar.py'):
            cmd = Path(f'/proc/{apagado}/cmdline').read_bytes()
            assert b'tft' in cmd
            os.kill(apagado, signal.SIGTERM)
            for _ in range(50):
                if not coincide(apagado, 'apagar_patchtst_al_terminar.py'):
                    break
                time.sleep(.2)
            assert not coincide(apagado, 'apagar_patchtst_al_terminar.py')
        if coincide(pid, 'entrenar_tft_numerico.py'):
            os.kill(pid, signal.SIGTERM)
        with (OUT / 'gpu.lock').open('a') as gpu:
            for _ in range(100):
                try:
                    fcntl.flock(gpu, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    time.sleep(.2)
            else:
                raise RuntimeError('La instancia original no liberó el bloqueo GPU')
            fcntl.flock(gpu, fcntl.LOCK_UN)
        env = os.environ.copy()
        env['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'
        env['PYTHONPATH'] = str(ROOT.parent / '.demanda-python-deps')
        with (OUT / 'entrenamiento_reducido.log').open('ab') as log:
            nuevo = subprocess.Popen(['/home/fabian/Documents/Machine Learning/mlenv/bin/python', '-u',
                'scripts/entrenar_tft_reducido.py'], cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        for _ in range(60):
            time.sleep(1)
            assert nuevo.poll() is None, 'El ejecutor reducido no pudo iniciar'
            if leer('estado.json').get('pid') == nuevo.pid:
                break
        else:
            raise RuntimeError('No se confirmó el inicio de la nueva instancia')
        with (OUT / 'apagado_automatico.log').open('ab') as log:
            monitor = subprocess.Popen(['/usr/bin/python3', '-u', 'scripts/apagar_patchtst_al_terminar.py',
                '--modelo', 'tft'], cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True)
        time.sleep(1)
        assert monitor.poll() is None, 'No se pudo restablecer el monitor de apagado'
        servicios.update(pid_entrenamiento=nuevo.pid, pid_apagado=monitor.pid)
        (OUT / 'servicios_locales.json').write_text(json.dumps(servicios, indent=2))
        registrar('reanudado_reducido', pid_entrenamiento=nuevo.pid, pid_apagado=monitor.pid, ajuste_completado=key)


if __name__ == '__main__':
    try:
        main()
    except BlockingIOError:
        print('Ya existe una transición activa.', flush=True)
    except BaseException as error:
        registrar('error_transicion', detalle=str(error))
        raise

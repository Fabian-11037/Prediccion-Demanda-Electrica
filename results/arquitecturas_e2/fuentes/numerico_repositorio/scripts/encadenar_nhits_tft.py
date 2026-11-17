"""Cola local: espera el cierre y guardado de N-HiTS antes de iniciar TFT."""
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
PREVIO = ROOT / 'results/nhits_numerico_r1'
OUT = ROOT / 'results/tft_numerico_r1'
PYTHON = '/home/fabian/Documents/Machine Learning/mlenv/bin/python'


def leer(path):
    return json.loads(path.read_text())


def guardar(path, datos):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(datos, indent=2, ensure_ascii=False))
    tmp.replace(path)


def registrar(estado, **datos):
    guardar(OUT / 'cola.json', dict(estado=estado, pid=os.getpid(), fecha=time.time(), **datos))


def activo(pid, nombre):
    try:
        datos = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return datos[0] != 'Z' and nombre.encode() in Path(f'/proc/{pid}/cmdline').read_bytes()
    except OSError:
        return False


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'cola.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert (OUT / 'plan.json').exists(), 'Preparar el plan TFT antes de encolarlo'
        if (OUT / 'ejecucion_completa.json').exists():
            registrar('tft_ya_completado')
            return
        if (OUT / 'estado.json').exists():
            previo_tft = leer(OUT / 'estado.json')
            if activo(previo_tft.get('pid', 0), 'entrenar_tft_numerico.py'):
                registrar('tft_ya_activo', pid_entrenamiento=previo_tft['pid'])
                return
        registrar('esperando_nhits')
        guardar(OUT / 'estado.json', dict(estado='En cola', etapa='espera_nhits',
                pid=os.getpid(), completados=0, total=130, actualizado=time.time()))
        (OUT / 'progreso.html').write_text('''<!doctype html><html lang="es"><meta charset="utf-8"><meta http-equiv="refresh" content="10"><title>TFT · En cola</title><style>body{font:18px system-ui;background:#14202c;color:#eef5fa;max-width:950px;margin:50px auto;padding:20px}progress{width:100%;height:32px}p{line-height:1.7}a{color:#8cd4ff}</style><h1>TFT compacto · En cola</h1><progress value="0" max="130"></progress><p>Esperando que N-HiTS termine y guarde sus resultados. TFT comenzará automáticamente después.</p><p>Dimensión16 · una capa LSTM · cuatro cabezas · historia168h · horizonte24h.<br>125 ajustes de selección y piloto, seguidos de prueba.<br>El límite24h de TFT empieza al iniciar, sin contar la espera.</p><p><a href="http://127.0.0.1:39973/progreso.html">Ver progreso de N-HiTS</a></p><p>El apagado automático queda al final de TFT, después de guardar resultados.</p></html>''')
        while True:
            estado = leer(PREVIO / 'estado.json')
            corriendo = activo(estado['pid'], 'entrenar_nhits_numerico.py')
            if estado['estado'] == 'Error':
                raise RuntimeError('N-HiTS terminó con error; TFT no se inicia automáticamente')
            if not corriendo:
                assert estado['estado'] in {'Completado', 'Presupuesto temporal agotado'}, 'N-HiTS salió sin cierre válido'
                with (PREVIO / 'gpu.lock').open('a') as gpu:
                    fcntl.flock(gpu, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    spec = importlib.util.spec_from_file_location('apagador', ROOT / 'scripts/apagar_patchtst_al_terminar.py')
                    apagador = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(apagador)
                    apagador.OUT = PREVIO
                    apagador.verificar(estado)
                    os.sync()
                break
            time.sleep(30)
        registrar('iniciando_tft', estado_nhits=estado)
        env = os.environ.copy()
        env['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'
        env['PYTHONPATH'] = str(ROOT.parent / '.demanda-python-deps')
        with (OUT / 'entrenamiento.log').open('ab') as log:
            proceso = subprocess.Popen([PYTHON, '-u', 'scripts/entrenar_tft_numerico.py'],
                cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True)
        for _ in range(60):
            time.sleep(1)
            if proceso.poll() is not None:
                raise RuntimeError('TFT no pudo iniciar; consultar entrenamiento.log')
            estado_tft = leer(OUT / 'estado.json')
            if estado_tft.get('pid') == proceso.pid:
                break
        else:
            raise RuntimeError('TFT no confirmó inicio; no se programó apagado')
        with (OUT / 'apagado_automatico.log').open('ab') as log:
            monitor = subprocess.Popen(['/usr/bin/python3', '-u', 'scripts/apagar_patchtst_al_terminar.py', '--modelo', 'tft'],
                cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True)
        time.sleep(1)
        assert monitor.poll() is None, 'El monitor de apagado TFT no pudo iniciar'
        servicios = leer(OUT / 'servicios_locales.json')
        servicios.update(pid_entrenamiento=proceso.pid, pid_apagado=monitor.pid)
        guardar(OUT / 'servicios_locales.json', servicios)
        registrar('tft_iniciado', pid_entrenamiento=proceso.pid, pid_apagado=monitor.pid)


if __name__ == '__main__':
    try:
        main()
    except BlockingIOError:
        print('La cola o el recurso GPU ya están bloqueados; no se duplica la ejecución.', flush=True)
    except BaseException as error:
        registrar('error_cola', detalle=str(error))
        actual = leer(OUT / 'estado.json') if (OUT / 'estado.json').exists() else {}
        if not activo(actual.get('pid', 0), 'entrenar_tft_numerico.py'):
            guardar(OUT / 'estado.json', dict(estado='Error de cola', detalle=str(error),
                    pid=os.getpid(), completados=0, total=130))
        raise

"""Apaga el equipo únicamente tras guardar la ejecución GPU completa."""
import json
import csv
import os
from pathlib import Path
import subprocess
import time

OUT = Path(__file__).resolve().parents[1] / 'results/base_v2'

def registrar(estado, **extra):
    target = OUT / 'apagado_automatico.json'
    tmp = target.with_suffix('.tmp')
    tmp.write_text(json.dumps(dict(estado=estado, pid=os.getpid(), fecha=time.time(), **extra), indent=2))
    tmp.replace(target)

if __name__ == '__main__':
    import fcntl
    lock = (OUT / 'apagado.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    registrar('esperando_finalizacion')
    while True:
        try:
            estado = json.loads((OUT / 'estado_gpu.json').read_text())
            completo = json.loads((OUT / 'ejecucion_completa.json').read_text()) if (OUT / 'ejecucion_completa.json').exists() else {}
            if estado.get('estado') == 'Error':
                registrar('cancelado_por_error_entrenamiento')
                break
            if completo.get('final_preregistrado') and estado.get('estado') == 'Completado' and all((OUT / f'{etapa}_completo.json').exists() for etapa in ('buscar', 'piloto', 'final')):
                requeridos = ('resumen_final.csv', 'final_runs.csv', 'mejor_config.json', 'piloto_resumen.json', 'referencias_test.json')
                if any(not (OUT / nombre).is_file() or (OUT / nombre).stat().st_size == 0 for nombre in requeridos):
                    registrar('apagado_bloqueado_resultados_incompletos')
                    break
                with (OUT / 'final_runs.csv').open() as archivo:
                    filas = list(csv.DictReader(archivo))
                n_final = max(5, json.loads((OUT / 'piloto_resumen.json').read_text())['n_final'])
                modelos = ('arx_lineal', 'mlp', 'lstm', 'cnn1d')
                if any(sum(fila['modelo'] == modelo for fila in filas) != n_final for modelo in modelos):
                    registrar('apagado_bloqueado_conteo_final_incompleto')
                    break
                import hashlib
                inventario = {}
                for ruta in OUT.rglob('*'):
                    if ruta.is_file() and ruta.suffix in ('.pt', '.npz', '.csv'):
                        inventario[str(ruta.relative_to(OUT))] = dict(bytes=ruta.stat().st_size, sha256=hashlib.file_digest(ruta.open('rb'), 'sha256').hexdigest())
                (OUT / 'resultados_guardados.json').write_text(json.dumps(dict(archivos=inventario, semillas_por_modelo=n_final, verificado=time.time()), indent=2))
                registrar('apagado_solicitado')
                os.sync()
                result = subprocess.run(['busctl', 'call', 'org.freedesktop.login1', '/org/freedesktop/login1', 'org.freedesktop.login1.Manager', 'PowerOff', 'b', 'false'], capture_output=True, text=True)
                if result.returncode:
                    registrar('error_apagado', detalle=result.stderr)
                break
        except (OSError, ValueError):
            pass
        time.sleep(30)

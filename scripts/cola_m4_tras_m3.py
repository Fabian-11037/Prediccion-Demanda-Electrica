"""Entrega la GPU a M4 solo después del cierre correcto y guardado de M3."""
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback


def main():
    config=json.loads(Path(sys.argv[1]).read_text());out=Path(config['salida_m4']);prev=Path(config['salida_m3'])
    out.mkdir(parents=True,exist_ok=True)
    lock=(out/'cola.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def guardar(nombre,obj):
        p=out/nombre;t=p.with_suffix('.tmp');t.write_text(json.dumps(obj,ensure_ascii=False,indent=1));t.replace(p)
    guardar('COLA_ESTADO.json',dict(estado='Esperando entrenamiento y prueba de M3',apagado_final='M4'))
    try:
        while True:
            if (prev/'ERROR.json').exists():
                guardar('COLA_ESTADO.json',dict(estado='Error en M3; no iniciar M4 ni apagar'));return
            if (prev/'MANIFIESTO_FINAL.json').exists() and (prev/'FINALIZADO.json').exists() and (out/'PREPARACION_LISTA.json').exists():break
            time.sleep(5)
        listo=json.loads((out/'PREPARACION_LISTA.json').read_text())
        with (out/'ejecucion.log').open('a') as f:p=subprocess.Popen(listo['comando'],cwd=config['repositorio'],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        guardar('M4_PROCESO.json',dict(pid=p.pid,comando=listo['comando']))
        guardar('COLA_ESTADO.json',dict(estado='M4 iniciado después de M3',pid=p.pid))
    except BaseException:guardar('ERROR_COLA.json',dict(error=traceback.format_exc(),apagado=False))


if __name__=='__main__':main()

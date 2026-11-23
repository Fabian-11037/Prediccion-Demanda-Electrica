"""Cancela el apagado de M2 y entrega la GPU a M3 cuando su preparación está lista."""
import datetime as dt
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback
from zoneinfo import ZoneInfo


def main():
    cola=Path(sys.argv[1]).resolve();c=json.loads(cola.read_text());out=Path(c['salida_m3']);m2=Path(c['salida_m2'])
    out.mkdir(parents=True,exist_ok=True)
    lock=(out/'cola.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def guardar(nombre,obj):
        p=out/nombre;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(obj,ensure_ascii=False,indent=1));t.replace(p)
    def estado(s,**kw):guardar('COLA_ESTADO.json',dict(estado=s,fecha=dt.datetime.now(ZoneInfo('America/Bogota')).isoformat(),**kw))
    estado('Esperando finalización de M2 y preparación de M3')
    try:
        while True:
            if (m2/'ERROR.json').exists():
                estado('Bloqueada: M2 terminó con error; sin apagado ni inicio de M3');return
            if (m2/'APAGADO.json').exists():
                a=json.loads((m2/'APAGADO.json').read_text())
                if a['codigo']==0:
                    r=subprocess.run(['shutdown','-c'],capture_output=True,text=True)
                    guardar('APAGADO_M2_CANCELADO.json',dict(codigo=r.returncode,salida=r.stdout,error=r.stderr))
                    if r.returncode:raise RuntimeError('No se pudo cancelar el apagado de M2: '+r.stderr)
                else:guardar('APAGADO_M2_CANCELADO.json',dict(codigo=0,salida='M2 no había programado apagado',error=''))
                break
            time.sleep(2)
        assert (m2/'FINALIZADO.json').exists() and (m2/'MANIFIESTO_FINAL.json').exists()
        estado('M2 finalizado; apagado pospuesto hasta terminar M3')
        while not (out/'PREPARACION_LISTA.json').exists():time.sleep(2)
        listo=json.loads((out/'PREPARACION_LISTA.json').read_text())
        with (out/'ejecucion.log').open('a') as f:
            p=subprocess.Popen(listo['comando'],cwd=c['repositorio'],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        guardar('M3_PROCESO.json',dict(pid=p.pid,comando=listo['comando']))
        estado('M3 iniciado después de M2',pid=p.pid)
    except BaseException:
        guardar('ERROR_COLA.json',dict(error=traceback.format_exc(),apagado=False));estado('Error; equipo encendido')


if __name__=='__main__':main()

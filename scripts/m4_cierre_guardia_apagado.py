"""Finaliza la inferencia registrada y su publicación antes del apagado."""
import json,subprocess,time,sys,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/m4_cierre_gpu_20261123'

def ejecutar():
    while True:
        if (OUT/'ERROR_COLA_COMPLETA.json').exists():return
        fallo=subprocess.run(['systemctl','--user','is-failed','--quiet','demanda-m4-principal.service']).returncode==0
        if fallo:
            (OUT/'ERROR_RUNTIME_CONTROLADOR.json').write_text(json.dumps(dict(error='El servicio principal terminó inesperadamente; revisar cola_completa.log',apagado=False)))
            return
        if (OUT/'APAGADO.json').exists():
            r=subprocess.run(['shutdown','-c'],capture_output=True,text=True)
            if r.returncode!=0:raise RuntimeError('No se pudo cancelar el apagado inicial: '+r.stderr)
            estado=dict(estado='Ejecutando',fase='Ampliación del plan y cierre estadístico',tareas=0,total=1,gpu='Entrenamiento completado; estadística CPU')
            (OUT/'progreso/estado.json').write_text(json.dumps(estado))
            while not (OUT/'complemento/LISTO_PARA_COLA.json').exists():time.sleep(5)
            if not (OUT/'historicos_nbeats_timesnet/FINALIZADO.json').exists():
                subprocess.run([sys.executable,str(ROOT/'scripts/m4_cierre_predecir_historicos.py'),'--out',str(OUT)],check=True)
            subprocess.run([sys.executable,str(ROOT/'scripts/m4_cierre_complemento.py'),'--out',str(OUT)],check=True)
            assert json.loads((OUT/'complemento/PUBLICADO.json').read_text())['verificado']
            assert all((OUT/'insumos'/f'{n}.json').exists() for n in ['nuevo_timesnet__N','nuevo_nbeats__N'])
            subprocess.run([sys.executable,str(ROOT/'scripts/m4_cierre_completar_top10.py'),'--out',str(OUT)],check=True)
            assert json.loads((OUT/'AMPLIACION_TOP10_PUBLICADA.json').read_text())['verificado']
            rr=subprocess.run(['shutdown','-h','+2'],capture_output=True,text=True)
            (OUT/'APAGADO_TRAS_TOP10.json').write_text(json.dumps(dict(retorno=rr.returncode,salida=rr.stdout,error=rr.stderr,comparaciones_completadas=True)))
            assert rr.returncode==0
            estado.update(estado='Finalizado',fase='Resultados publicados y verificados; apagado en dos minutos',tareas=1)
            (OUT/'progreso/estado.json').write_text(json.dumps(estado))
            return
        time.sleep(5)

if __name__=='__main__':
    try:ejecutar()
    except Exception:
        (OUT/'ERROR_CIERRE_ESTADISTICO.json').write_text(json.dumps(dict(error=traceback.format_exc(),apagado=False)));raise

"""Cola persistente GPU: prueba final del corrector, entrenamiento de M1 y apagado autorizado."""
from __future__ import annotations
import argparse
import datetime as dt
import fcntl
import functools
import http.server
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time
import traceback
from zoneinfo import ZoneInfo
from e3_gpu_comun import *
from e3_corrector_final import ejecutar_final
from e3_modelo1 import ejecutar_m1

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--salida',type=Path,required=True);ap.add_argument('--puerto',type=int,default=39976)
    ap.add_argument('--apagar-al-final',action='store_true');args=ap.parse_args()
    out=args.salida.resolve();out.mkdir(parents=True,exist_ok=True)
    bloqueo=(out/'proceso.lock').open('w');fcntl.flock(bloqueo,fcntl.LOCK_EX|fcntl.LOCK_NB)
    web=out/'progreso';web.mkdir(exist_ok=True)
    shutil.copy2(REPO/'scripts/e3_progreso.html',web/'progreso.html')
    inicio=time.time();estado=dict(estado='Preparando',fase='verificar',inicio=inicio,ajustes_completados=0,ajustes_totales=401,
           etapa_corrector='pendiente',etapa_m1='en cola',apagado_al_final=args.apagar_al_final)
    lock=threading.RLock()
    def progreso(**kw):
        with lock:
            estado.update(kw);estado['actualizado']=dt.datetime.now(ZoneInfo('America/Bogota')).isoformat(timespec='seconds');estado['pid']=os.getpid()
            guardar(web/'estado.json',estado)
    def completar():progreso(ajustes_completados=estado['ajustes_completados']+1)
    def log(msg,consola=True):
        linea=f'[{dt.datetime.now(ZoneInfo("America/Bogota")).isoformat(timespec="seconds")}] {msg}'
        with (out/'bitacora.log').open('a') as f:f.write(linea+'\n')
        if consola:print(linea,flush=True)
    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self,*a,**k):super().__init__(*a,directory=str(web),**k)
        def log_message(self,*a):pass
        def end_headers(self):self.send_header('Cache-Control','no-store');super().end_headers()
    servidor=http.server.ThreadingHTTPServer(('127.0.0.1',args.puerto),Handler)
    threading.Thread(target=servidor.serve_forever,daemon=True).start()
    def pulso():
        while True:time.sleep(15);progreso()
    threading.Thread(target=pulso,daemon=True).start();progreso()
    try:
        cfg=leer(REPO/'continuidad/CONFIGURACIONES_PRUEBA_E3_GPU.json');diseno=leer(REPO/'continuidad/DISENO_CUATRO_MODELOS_GPU.json')
        fuentes=list((PAQUETE/'src').rglob('*.py'))+list((PAQUETE/'config').glob('*.json'))+list((REPO/'scripts').glob('e3_*.py'))+[Path(__file__),REPO/'scripts/test_e3_gpu.py',REPO/'scripts/e3_progreso.html']
        guias=[REPO/'continuidad'/x for x in ['CONFIGURACIONES_PRUEBA_E3_GPU.json','DISENO_CUATRO_MODELOS_GPU.json','ENTRENAMIENTO_GPU_E3.md','DISENO_CUATRO_MODELOS_GPU.md']]
        insumos=[REPO/'data/processed'/x for x in ['panel_horario.parquet','feature_registry.json','corpus_texto.parquet']]
        huellas={str(p):sha(p) for p in fuentes+guias+insumos}
        plan=dict(fecha=dt.datetime.now(ZoneInfo('America/Bogota')).isoformat(),autorizacion='Evaluación final del corrector y luego entrenamiento M1; apagado al finalizar autorizado por el usuario.',
             corrector=cfg,diseno_m1=diseno,huellas=huellas,apagado=args.apagar_al_final,
             alcance_M1='Desarrollo: 8 candidatos x 5 semillas x 4 folds, controles y ajuste final anterior a 2022. No se predice prueba de M1 ni se ejecutan M2-M4.',
             implementacion_M1=dict(capacidad='MLP de dos capas ocultas, ancho de rejilla; cabeza lineal auxiliar por pareja; logits por hora',suavidad='Media del cuadrado de diferencias de pesos entre horas; coeficiente .001',
                  EWA='eta=1 fijado antes de ejecutar; errores observables d-2; arranque uniforme',convexa_fija='MAE por programación lineal sobre parada previa',
                  purga_meta='un día completo entre último objetivo de ajuste e inicio de parada'),
             secuencia=['corrector_ajuste_pre2022','congelacion_36_checkpoints','texto_y_prediccion_prueba','estadistica_corrector','M1_OOF_expertos','M1_desarrollo_y_controles','M1_ajuste_final_preprueba','integridad_y_apagado'],
             comparacion_informada='El benchmark 2022–2025 ya fue conocido; no se denomina prueba nunca consultada. Las métricas finales nuevas del corrector no modifican M1.')
        if (out/'REGISTRO_PREVIO.json').exists():
            anterior=leer(out/'REGISTRO_PREVIO.json');assert anterior['huellas']==huellas,'El código o los datos cambiaron; crear una corrida nueva'
        else:guardar(out/'REGISTRO_PREVIO.json',plan)
        for p in fuentes+guias:
            relativo=p.relative_to(REPO) if p.is_relative_to(REPO) else Path('paquete_corrector')/p.relative_to(PAQUETE)
            dest=out/'fuentes'/relativo;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
        assert sha(insumos[0])==cfg['panel_sha256'] and sha(insumos[2])==cfg['corpus_sha256']
        registro_portable=verificar_json_portable(insumos[1],diseno['datos']['hashes_entrada']['data/processed/feature_registry.json'])
        guardar(out/'EQUIVALENCIA_REGISTRO_JSON.json',registro_portable)
        assert all(sha(REPO/'data/processed'/p.name)==sha(PAQUETE/'data/processed'/p.name) for p in insumos[:2])
        assert torch.cuda.is_available(),'GPU no disponible; no se continúa en CPU'
        torch.set_num_threads(8);x=torch.ones(8,device='cuda',requires_grad=True);x.square().sum().backward();torch.cuda.synchronize();assert x.grad.is_cuda
        gpu=dict(nombre=torch.cuda.get_device_name(0),memoria_gb=round(torch.cuda.get_device_properties(0).total_memory/2**30,2),torch=torch.__version__,rocm=torch.version.hip)
        guardar(out/'ENTORNO_GPU.json',gpu);progreso(gpu=gpu,estado='Verificando implementación')
        # Pruebas de invariantes del adaptador, sin utilizar datos de prueba.
        test=subprocess.run([sys.executable,'-m','pytest','-q',str(REPO/'scripts/test_e3_gpu.py'),'-p','no:cacheprovider'],cwd=REPO,capture_output=True,text=True)
        (out/'pruebas_implementacion.txt').write_text(test.stdout+'\n'+test.stderr)
        if test.returncode:raise RuntimeError('Fallaron pruebas de implementación: '+test.stdout[-2500:]+test.stderr[-500:])
        datos=DatosNumericos(PAQUETE)
        from entrenamiento import pruebas
        checks=pruebas.numericas(datos);assert all(checks.values()),checks;guardar(out/'auditoria_numerica.json',checks)
        for p in insumos:assert sha(p)==huellas[str(p)]
        progreso(estado='Ejecutando',etapa_corrector='en ejecución')
        log('Inicia ajuste final del corrector; M1 queda en cola. GPU '+gpu['nombre'])
        ejecutar_final(out/'corrector_final',datos,cfg,progreso,completar,log)
        progreso(etapa_corrector='finalizado',etapa_m1='en ejecución',epoca=None)
        ejecutar_m1(out/'M1',datos,diseno,cfg,out/'corrector_final',progreso,completar,log)
        assert (out/'corrector_final/FINALIZADO.json').exists() and (out/'M1/FINALIZADO.json').exists()
        progreso(etapa_m1='finalizado',fase='guardando',etiqueta='Guardando manifiesto de resultados y checkpoints',epoca=None)
        archivos=[]
        for p in sorted(out.rglob('*')):
            if p.is_file() and p.suffix not in ['.log','.tmp'] and 'progreso' not in p.parts and p.name not in ['proceso.lock','MANIFIESTO_FINAL.json']:
                archivos.append(dict(archivo=str(p.relative_to(out)),bytes=p.stat().st_size,sha256=sha(p)))
        guardar(out/'MANIFIESTO_FINAL.json',archivos)
        guardar(out/'FINALIZADO.json',dict(fecha=dt.datetime.now(ZoneInfo('America/Bogota')).isoformat(),corrector_prueba=True,M1_entrenado=True,M1_prueba=False,archivos=len(archivos)))
        progreso(estado='Finalizado',fase='finalizado',fin=time.time(),etiqueta='Resultados y checkpoints guardados')
        log('Cola terminada e integridad registrada.')
        if args.apagar_al_final:
            r=subprocess.run(['shutdown','-h','+2'],capture_output=True,text=True)
            guardar(out/'APAGADO.json',dict(codigo=r.returncode,salida=r.stdout,error=r.stderr,fecha=dt.datetime.now(ZoneInfo('America/Bogota')).isoformat()))
            if r.returncode:progreso(estado='Finalizado; apagado pendiente',etiqueta=r.stderr)
            else:progreso(estado='Finalizado; apagado programado',etiqueta=r.stdout.strip())
    except BaseException:
        error=traceback.format_exc();log(error);guardar(out/'ERROR.json',dict(error=error,apagado=False));progreso(estado='Error; equipo encendido',error=error,fin=time.time())
    while True:time.sleep(30)

if __name__=='__main__':main()

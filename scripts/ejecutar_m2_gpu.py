"""Cola persistente M2: desarrollo, controles, prueba, guardado y apagado autorizado."""
from __future__ import annotations
import argparse
import datetime as dt
import fcntl
import http.server
import shutil
import subprocess
import threading
import time
import traceback
from zoneinfo import ZoneInfo
from e3_gpu_comun import *
from e3_modelo2 import ejecutar


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--salida',type=Path,required=True)
    ap.add_argument('--entrenamiento-m1',type=Path,required=True)
    ap.add_argument('--prueba-m1',type=Path,required=True)
    ap.add_argument('--puerto',type=int,default=39977)
    ap.add_argument('--apagar-al-final',action='store_true')
    args=ap.parse_args();out=args.salida.resolve();out.mkdir(parents=True,exist_ok=True)
    bloqueo=(out/'proceso.lock').open('w');fcntl.flock(bloqueo,fcntl.LOCK_EX|fcntl.LOCK_NB)
    run=args.entrenamiento_m1.resolve();m1=run/'M1';test=args.prueba_m1.resolve()
    web=out/'progreso';web.mkdir(exist_ok=True)
    shutil.copy2(REPO/'scripts/m2_progreso.html',web/'progreso.html')
    estado=dict(estado='Preparando',fase='verificacion',inicio=time.time(),ajustes_completados=0,ajustes_totales=300,apagado_al_final=args.apagar_al_final)
    mutex=threading.RLock()
    def progreso(**kw):
        with mutex:
            estado.update(kw);estado.update(actualizado=dt.datetime.now(ZoneInfo('America/Bogota')).isoformat(timespec='seconds'),pid=os.getpid())
            guardar(web/'estado.json',estado)
    def completar():progreso(ajustes_completados=estado['ajustes_completados']+1)
    def log(msg):
        with (out/'bitacora.log').open('a') as f:f.write(dt.datetime.now(ZoneInfo('America/Bogota')).isoformat()+' '+msg+'\n')
        print(msg,flush=True)
    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self,*a,**k):super().__init__(*a,directory=str(web),**k)
        def log_message(self,*a):pass
        def end_headers(self):self.send_header('Cache-Control','no-store');super().end_headers()
    servidor=http.server.ThreadingHTTPServer(('127.0.0.1',args.puerto),Handler)
    threading.Thread(target=servidor.serve_forever,daemon=True).start()
    progreso()
    try:
        diseno=leer(REPO/'continuidad/DISENO_CUATRO_MODELOS_GPU.json')
        fuentes=list((PAQUETE/'src').rglob('*.py'))+[REPO/'scripts'/n for n in ['e3_gpu_comun.py','e3_modelo1.py','e3_modelo2.py','ejecutar_m2_gpu.py','test_m2_gpu.py','m2_progreso.html']]
        entradas=[m1/'CONFIGURACION_ELEGIDA.json',test/'REGISTRO_PREVIO.json',test/'contrastes.csv']
        for s in diseno['validacion']['semillas']:
            entradas += [m1/'expertos/patchtst'/f'anio_{a}'/f'semilla_{s}'/'modelo.pt' for a in range(2011,2022)]
            entradas += [m1/'expertos/patchtst'/f'anio_{a}'/f'semilla_{s}'/'predicciones.npz' for a in range(2011,2022)]
            entradas += [m1/'expertos_finales/patchtst'/f'semilla_{s}'/'modelo.pt',test/f'predicciones_semilla_{s}.npz']
        datos_arch=[PAQUETE/'data/processed/panel_horario.parquet',PAQUETE/'data/processed/feature_registry.json']
        huellas={str(p):sha(p) for p in fuentes+entradas+datos_arch}
        anterior={r['archivo']:r['sha256'] for r in leer(run/'MANIFIESTO_FINAL.json')}
        assert all(huellas[str(p)]==anterior[str(p.relative_to(run))] for p in entradas if p.is_relative_to(run))
        assert huellas[str(datos_arch[0])]==diseno['datos']['hashes_entrada']['data/processed/panel_horario.parquet']
        verificar_json_portable(datos_arch[1],diseno['datos']['hashes_entrada']['data/processed/feature_registry.json'])
        plan=dict(fecha=dt.datetime.now(ZoneInfo('America/Bogota')).isoformat(),
                  autorizacion='Usuario: inicia entrenamiento M2, incluye prueba y apaga cuando termine.',
                  diseno=diseno,huellas=huellas,semillas=diseno['validacion']['semillas'],
                  rejilla='8 candidatos x 5 semillas x 4 folds =160 ajustes',
                  controles='3 controles entrenables x5 semillas x4 folds =60; ancla intacta reutilizada',
                  finales='M2 y3 controles x5 semillas =20; total240 ajustes;60 extracciones congeladas',
                  implementacion=dict(estado='Entrada del MLP contextual de PatchTST: media de parches por canal y calendario, congelados por año y semilla',
                     extras='Pronóstico ancla/nivel histórico y error ancla d-2/media d-8..d-2',
                     cabezas='MLP compartido de dos capas del ancho registrado; energía softplus y logits horarios',
                     ancla='recorte inferior1e-6 kWh, normalización por suma, epsilon1e-8 antes de log',
                     escala_energia='Media de suma diaria de objetivos exclusivamente en ajuste',
                     directa24='Mismo MLP;24 salidas softplus multiplicadas por escala_energia/24; capacidad difiere solo en una salida',
                     latentes='Los modelos OOF anuales pueden cambiar de base latente; se declara como limitación'),
                  seleccion_prueba=False,prueba='2022–2025,1461 orígenes diarios; se abre tras congelar20 checkpoints finales',
                  estadistica='Familia global28:4 hipótesis previas de M1 +4 nuevas de M2;20 pendientes completadas con1 para cota conservadora de Holm; bootstrap9999,14 días, sensibilidad7/28',
                  comparadores_primarios=['M1','PatchTST mejor individual de desarrollo'],
                  comparadores_descriptivos=['NNLS','ancla','solo_energia','solo_distribucion','directa_24'],
                  retrospectivo='Benchmark conocido durante el diseño; no prueba nunca consultada',
                  apagado_al_final=args.apagar_al_final,publicacion_automatica=False,
                  publicacion_previa='Corrector y M1 publicados en85b35df antes de iniciar esta cola')
        if (out/'REGISTRO_PREVIO.json').exists():
            assert leer(out/'REGISTRO_PREVIO.json')['huellas']==huellas,'Código o insumos modificados: crear otra corrida'
        else:guardar(out/'REGISTRO_PREVIO.json',plan)
        for p in fuentes:
            rel=p.relative_to(REPO) if p.is_relative_to(REPO) else Path('paquete_corrector')/p.relative_to(PAQUETE)
            dest=out/'fuentes'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
        assert torch.cuda.is_available(),'GPU requerida, no se permite continuar en CPU'
        torch.set_num_threads(8)
        x=torch.ones(4,device='cuda',requires_grad=True);x.square().sum().backward();torch.cuda.synchronize();assert x.grad.is_cuda
        gpu=dict(nombre=torch.cuda.get_device_name(0),memoria_gb=round(torch.cuda.get_device_properties(0).total_memory/2**30,2),torch=torch.__version__,rocm=torch.version.hip)
        guardar(out/'ENTORNO_GPU.json',gpu);progreso(gpu=gpu)
        env=dict(os.environ);env['PYTHONPATH']=os.pathsep.join([str(REPO.parent/'.demanda-python-deps'),str(REPO/'scripts')])
        t=subprocess.run([sys.executable,'-m','pytest','-q',str(REPO/'scripts/test_m2_gpu.py'),'-p','no:cacheprovider'],env=env,cwd=REPO,capture_output=True,text=True)
        (out/'pruebas_implementacion.txt').write_text(t.stdout+'\n'+t.stderr)
        if t.returncode:raise RuntimeError('Falló validación de M2: '+t.stdout[-3000:]+t.stderr[-1000:])
        datos=DatosNumericos(PAQUETE)
        progreso(estado='Ejecutando',fase='extraccion_oof');log('M2 inicia en GPU '+gpu['nombre'])
        ejecutar(out,datos,diseno,m1,test,progreso,completar,log)
        progreso(fase='guardando',epoca=None,etiqueta='Comprobando conservación y manifiesto')
        assert all(sha(p)==h for p,h in huellas.items()),'Se modificaron insumos congelados'
        assert estado['ajustes_completados']==300
        guardar(out/'VERIFICACION_FINAL.json',dict(insumos_conservados=True,tareas_completadas=300,checkpoints_finales=20,prueba=1461))
        guardar(out/'MANIFIESTO_FINAL.json',[dict(archivo=str(p.relative_to(out)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(out.rglob('*')) if p.is_file() and p.name not in ['MANIFIESTO_FINAL.json','proceso.lock'] and p.suffix not in ['.log','.tmp'] and 'progreso' not in p.parts])
        progreso(estado='Finalizado',fase='finalizado',fin=time.time(),etiqueta='Entrenamiento y prueba guardados')
        if args.apagar_al_final:
            r=subprocess.run(['shutdown','-h','+2'],capture_output=True,text=True)
            guardar(out/'APAGADO.json',dict(codigo=r.returncode,salida=r.stdout,error=r.stderr,fecha=dt.datetime.now(ZoneInfo('America/Bogota')).isoformat()))
            progreso(estado='Finalizado; apagado programado' if not r.returncode else 'Finalizado; apagado pendiente',etiqueta=r.stdout.strip() or r.stderr)
    except BaseException:
        e=traceback.format_exc();log(e);guardar(out/'ERROR.json',dict(error=e,apagado=False));progreso(estado='Error; equipo encendido',error=e,fin=time.time())
    while True:time.sleep(30)


if __name__=='__main__':main()

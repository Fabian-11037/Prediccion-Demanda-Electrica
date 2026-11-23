"""Entrenamiento y prueba M3 tras M2. M4 tiene reservado el apagado final."""
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
from e3_modelo3 import ejecutar, Texto


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--salida',type=Path,required=True)
    ap.add_argument('--run-m1',type=Path,required=True);ap.add_argument('--prueba-m1',type=Path,required=True)
    ap.add_argument('--run-m2',type=Path,required=True);ap.add_argument('--puerto',type=int,default=39978)
    args=ap.parse_args();out=args.salida.resolve();out.mkdir(parents=True,exist_ok=True)
    lock=(out/'proceso.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    run=args.run_m1.resolve();test=args.prueba_m1.resolve();m2=args.run_m2.resolve()
    web=out/'progreso';web.mkdir(exist_ok=True);shutil.copy2(REPO/'scripts/m3_progreso.html',web/'progreso.html')
    estado=dict(estado='Preparando',fase='verificacion',inicio=time.time(),ajustes_completados=0,ajustes_totales=665,apagado_al_final=False)
    mutex=threading.RLock()
    def progreso(**kw):
        with mutex:
            estado.update(kw);estado.update(actualizado=dt.datetime.now(ZoneInfo('America/Bogota')).isoformat(timespec='seconds'),pid=os.getpid())
            guardar(web/'estado.json',estado)
    def completar():progreso(ajustes_completados=estado['ajustes_completados']+1)
    def log(s):
        with (out/'bitacora.log').open('a') as f:f.write(dt.datetime.now(ZoneInfo('America/Bogota')).isoformat()+' '+s+'\n')
        print(s,flush=True)
    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self,*a,**k):super().__init__(*a,directory=str(web),**k)
        def log_message(self,*a):pass
        def end_headers(self):self.send_header('Cache-Control','no-store');super().end_headers()
    servidor=http.server.ThreadingHTTPServer(('127.0.0.1',args.puerto),Handler)
    threading.Thread(target=servidor.serve_forever,daemon=True).start();progreso()
    try:
        assert (m2/'MANIFIESTO_FINAL.json').exists() and (m2/'FINALIZADO.json').exists()
        diseno=leer(REPO/'continuidad/DISENO_CUATRO_MODELOS_GPU.json')
        fuentes=[REPO/'scripts'/n for n in ['e3_modelo3.py','ejecutar_m3_gpu.py','test_m3_gpu.py','m3_progreso.html','e3_modelo1.py','e3_gpu_comun.py','probar_m1_gpu.py']]+list((PAQUETE/'src').rglob('*.py'))
        entradas=[REPO/'continuidad/DISENO_CUATRO_MODELOS_GPU.json',REPO/'continuidad/DISENO_CUATRO_MODELOS_GPU.md',REPO/'data/processed/corpus_texto.parquet',PAQUETE/'data/processed/panel_horario.parquet',PAQUETE/'data/processed/feature_registry.json']
        entradas+=list((run/'M1').rglob('modelo.pt'))+list((run/'M1/expertos').rglob('predicciones.npz'))
        entradas+=[run/'M1/rejilla_registrada.json',m2/'prueba/contrastes_globales_parciales.csv']+list(test.glob('predicciones_semilla_*.npz'))
        base=PAQUETE/'resultados_corrector_e3'
        for col in ['disponible_declarado','disponible_estricto']:
            h=base/('texto' if col=='disponible_declarado' else 'texto_'+col)
            entradas += [h/'resumen.json',h/'candidatos.npz',base/'texto_vectores'/f'vectores_{col}.npy']
            entradas += [run/'corrector_final/texto'/col/n for n in ['LISTO.json','candidatos.npz','vectores.npy']]
        huellas={str(p):sha(p) for p in fuentes+entradas}
        assert sha(REPO/'data/processed/corpus_texto.parquet')==diseno['datos']['hashes_entrada']['data/processed/corpus_texto.parquet']
        assert sha(PAQUETE/'data/processed/panel_horario.parquet')==diseno['datos']['hashes_entrada']['data/processed/panel_horario.parquet']
        verificar_json_portable(PAQUETE/'data/processed/feature_registry.json',diseno['datos']['hashes_entrada']['data/processed/feature_registry.json'])
        plan=dict(fecha=dt.datetime.now(ZoneInfo('America/Bogota')).isoformat(),autorizacion='Usuario solicita M3 tras M2 y añade M4 después; apagado final reservado a M4.',
          diseno=diseno,huellas=huellas,semillas=[101,102,103,104,105],
          arquitectura='Dos agregadores E5 separados con atención de4 cabezas y consultas numéricas/horarias; M1 y E5 congelados; delta de logits con compuerta, epsilon y tanh. Identidad exacta con M1 sin candidatos.',
          oof_anidado='Primer prefijo2013 usa cfg1 predefinida. Cada Y selecciona con folds2013..Y-1 disponibles, igual peso por fold/semilla; se añaden200 ajustes para2013–2017. Los demás checkpoints se reutilizan. No hay selección con Y ni prueba.',
          contrato_texto='Se prioriza la instrucción específica M3/M4 del Markdown §5: fragmentos1500 caracteres/solape200/máximo16 y reservas32+48+48,72h y alineados120 días. Difiere del bloque encoder genérico del JSON; decisión anterior al entrenamiento. E5 revisión y pooling del contrato sin cambios.',
          fuentes='Mapa aprendido solo en ajuste de cada fold; fuente posterior desconocida0',
          entrenamiento='AdamW lr.001,wd.0001,lote64,clip1,max100,min10,paciencia10,min_delta MASE.0001; FP32. Evaluación lote128 registrado por memoria, sin gradientes.',
          perdida='MAE/q +.1 SmoothL1(complementariedad) +lambda_delta media(delta²) +.001 suavidad de pesos',
          tareas='200 ajustes M1 de prefijos+45 extracciones OOF+160 búsqueda M3_OP+200 controles+55 finales+5 extracciones finales=665',
          variantes=['M3_OP','M3_O','M3_P','M3_metadatos','M3_N','desalineado49/91/154/217/301','estricto'],
          prueba='Solo después de congelar55 checkpoints finales;2022–2025,1461 orígenes,24h; sin actualizaciones',
          estadistica='18 hipótesis observadas M1/M2/M3,10 M4 pendientes; cota conservadora Holm global28. Bootstrap9999,bloques14,sensibilidad7/28. Controles desplazados: media de pérdidas.',
          retrospectivo=True,publicacion_automatica=False,apagado=False,siguiente='M4')
        if (out/'REGISTRO_PREVIO.json').exists():assert leer(out/'REGISTRO_PREVIO.json')['huellas']==huellas
        else:guardar(out/'REGISTRO_PREVIO.json',plan)
        for p in fuentes:
            rel=p.relative_to(REPO) if p.is_relative_to(REPO) else Path('paquete_corrector')/p.relative_to(PAQUETE)
            d=out/'fuentes'/rel;d.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,d)
        assert torch.cuda.is_available(),'GPU requerida; no hay fallback CPU'
        torch.set_num_threads(8);x=torch.ones(4,device='cuda',requires_grad=True);x.square().sum().backward();torch.cuda.synchronize();assert x.grad.is_cuda
        gpu=dict(nombre=torch.cuda.get_device_name(0),memoria_gb=round(torch.cuda.get_device_properties(0).total_memory/2**30,2),torch=torch.__version__,rocm=torch.version.hip)
        guardar(out/'ENTORNO_GPU.json',gpu);progreso(gpu=gpu)
        env=dict(os.environ);env['PYTHONPATH']=os.pathsep.join([str(REPO.parent/'.demanda-python-deps'),str(REPO/'scripts')])
        t=subprocess.run([sys.executable,'-m','pytest','-q',str(REPO/'scripts/test_m3_gpu.py'),'-p','no:cacheprovider'],env=env,capture_output=True,text=True,cwd=REPO)
        (out/'pruebas_implementacion.txt').write_text(t.stdout+'\n'+t.stderr)
        if t.returncode:raise RuntimeError(t.stdout[-3500:]+t.stderr[-1000:])
        datos=DatosNumericos(PAQUETE);progreso(estado='Ejecutando');log('M3 inicia en GPU '+gpu['nombre'])
        ejecutar(out,datos,diseno,run,test,m2,progreso,completar,log)
        assert estado['ajustes_completados']==665
        assert all(sha(p)==h for p,h in huellas.items()),'Cambiaron insumos o código de M3'
        guardar(out/'VERIFICACION_FINAL.json',dict(insumos_conservados=True,tareas=665,checkpoints_finales=55))
        guardar(out/'MANIFIESTO_FINAL.json',[dict(archivo=str(p.relative_to(out)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(out.rglob('*')) if p.is_file() and p.suffix not in ['.tmp','.log'] and p.name not in ['proceso.lock','cola.lock','MANIFIESTO_FINAL.json'] and 'progreso' not in p.parts])
        progreso(estado='Finalizado; continúa M4',fase='finalizado',fin=time.time(),epoca=None,etiqueta='M3 guardado; apagado al terminar M4')
    except BaseException:
        e=traceback.format_exc();log(e);guardar(out/'ERROR.json',dict(error=e,apagado=False));progreso(estado='Error; equipo encendido',error=e,fin=time.time())
    while True:time.sleep(30)


if __name__=='__main__':main()

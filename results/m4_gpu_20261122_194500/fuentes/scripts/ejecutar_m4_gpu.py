"""M4 después de M3: búsqueda, controles, prueba, cierre global y apagado."""
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
from e3_modelo4 import ejecutar


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--salida',type=Path,required=True)
    ap.add_argument('--run-m1',type=Path,required=True);ap.add_argument('--prueba-m1',type=Path,required=True)
    ap.add_argument('--run-m3',type=Path,required=True);ap.add_argument('--puerto',type=int,default=39979)
    ap.add_argument('--apagar-al-final',action='store_true')
    ap.add_argument('--publicar-al-final',action='store_true');args=ap.parse_args()
    out=args.salida.resolve();out.mkdir(parents=True,exist_ok=True)
    bloqueo=(out/'proceso.lock').open('w');fcntl.flock(bloqueo,fcntl.LOCK_EX|fcntl.LOCK_NB)
    run=args.run_m1.resolve();test=args.prueba_m1.resolve();m3=args.run_m3.resolve()
    web=out/'progreso';web.mkdir(exist_ok=True);shutil.copy2(REPO/'scripts/m4_progreso.html',web/'progreso.html')
    estado=dict(estado='Preparando',fase='verificacion',inicio=time.time(),ajustes_completados=0,ajustes_totales=465,apagado_al_final=args.apagar_al_final)
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
        assert (m3/'MANIFIESTO_FINAL.json').exists() and (m3/'FINALIZADO.json').exists() and not (m3/'ERROR.json').exists()
        diseno=leer(REPO/'continuidad/DISENO_CUATRO_MODELOS_GPU.json')
        fuentes=[REPO/'scripts'/n for n in ['e3_modelo4.py','ejecutar_m4_gpu.py','test_m4_gpu.py','m4_progreso.html','e3_modelo3.py','e3_modelo1.py','e3_gpu_comun.py','publicar_resultados_m2_m3_m4.py']]+list((PAQUETE/'src').rglob('*.py'))
        entradas=[REPO/'continuidad/DISENO_CUATRO_MODELOS_GPU.json',REPO/'continuidad/DISENO_CUATRO_MODELOS_GPU.md',REPO/'data/processed/corpus_texto.parquet',PAQUETE/'data/processed/panel_horario.parquet',PAQUETE/'data/processed/feature_registry.json']
        entradas+=list((m3/'base_oof').glob('*.npz'))+list((m3/'base_final').glob('*.npz'))+list((m3/'prefijos').glob('*.json'))
        entradas += [m3/'REGISTRO_PREVIO.json',m3/'prueba/contrastes_globales_parciales.csv']+list(test.glob('predicciones_semilla_*.npz'))
        base=PAQUETE/'resultados_corrector_e3'
        for col in ['disponible_declarado','disponible_estricto']:
            h=base/('texto' if col=='disponible_declarado' else 'texto_'+col)
            entradas += [h/'resumen.json',h/'candidatos.npz',base/'texto_vectores'/f'vectores_{col}.npy']
            entradas += [run/'corrector_final/texto'/col/n for n in ['LISTO.json','candidatos.npz','vectores.npy']]
        huellas={str(p):sha(p) for p in fuentes+entradas}
        manifiesto={r['archivo']:r['sha256'] for r in leer(m3/'MANIFIESTO_FINAL.json')}
        assert all(huellas[str(p)]==manifiesto[str(p.relative_to(m3))] for p in entradas if p.is_relative_to(m3))
        for n in ['corpus_texto.parquet','panel_horario.parquet']:
            p=REPO/'data/processed'/n
            assert sha(p)==diseno['datos']['hashes_entrada']['data/processed/'+n]
        verificar_json_portable(PAQUETE/'data/processed/feature_registry.json',diseno['datos']['hashes_entrada']['data/processed/feature_registry.json'])
        plan=dict(fecha=dt.datetime.now(ZoneInfo('America/Bogota')).isoformat(),autorizacion='Usuario agrega M4 después de M3; entrenamiento y prueba, apagado al finalizar toda la cola.',
          diseno=diseno,huellas=huellas,semillas=[101,102,103,104,105],
          base='OOF anidado de M1 generado por M3,2013–2021. M4 no usa predicciones de M3. Base final M1 congelada para2022–2025.',
          arquitectura='Atención aprendida por fragmento/canal sobre E5 congelado, proyección de consulta y clave32 con LayerNorm, similitud dot/sqrt32,top-k y compuerta MLP contextual.',
          elegibilidad='Edad>=2 y<=365/730 días, máscara ANTES de top-k y de consultar residuos. Cero vecinos devuelve M1 exactamente. Regla independiente del orden de minibatches.',
          compuerta='Contexto de consulta32 +similitud máxima,dispersión de residuos,cantidad y antigüedad de vecinos. Bias inicial−2; desviación con piso1e−12 para gradientes finitos.',
          texto='Contrato específico del Markdown §5 idéntico a M3:1500 caracteres,solape200,max16 fragmentos,reservas32/48/48,72h/120d. Se registra discrepancia con bloque encoder genérico del JSON. E5 sin ajuste; fuentes ajustadas solo en entrenamiento.',
          entrenamiento='AdamW lr.001,wd.0001,lote64,clip1,max100,min10,paciencia10,min_delta MASE.0001; FP32. Agregación textual por bloques64 para memoria, evaluación128 sin gradientes.',
          perdida='MAE horario/q168',rejilla='8x5 semillasx4 folds=160 ajustes',
          controles='12 variantesx5 semillasx4 folds=240 ajustes; ancla M1 sin ajuste',
          finales='13 variantesx5 semillas=65; total465 ajustes',
          ablaciones=['memoria fija al fin de parada (2021 en prueba)','sin compuerta','sin texto=M4_N','sin contexto numérico','estricto'],
          prueba='1461 días2022–2025 solo tras congelar65 checkpoints. Nuevos episodios se incorporan en d+2, sin optimizador ni reajuste. Se guardan vecinos/fechas/atenciones/compuertas y log de disponibilidad.',
          estadistica='5 pares nuevos/10 hipótesis +18 anteriores =Holm global28 completo;bootstrap9999,bloques14,sensibilidad7/28;desplazados agregados por pérdidas.',
          retrospectivo=True,publicacion_automatica=args.publicar_al_final,apagado_al_final=args.apagar_al_final,
          politica_apagado='Guardar resultados, publicar M2/M3/M4 y verificar commit/manifiesto remoto. Si falla publicación, no apagar.')
        if (out/'REGISTRO_PREVIO.json').exists():assert leer(out/'REGISTRO_PREVIO.json')['huellas']==huellas
        else:guardar(out/'REGISTRO_PREVIO.json',plan)
        for p in fuentes:
            rel=p.relative_to(REPO) if p.is_relative_to(REPO) else Path('paquete_corrector')/p.relative_to(PAQUETE)
            d=out/'fuentes'/rel;d.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,d)
        assert torch.cuda.is_available(),'GPU requerida; no se continúa en CPU'
        torch.set_num_threads(8);x=torch.ones(4,device='cuda',requires_grad=True);x.square().sum().backward();torch.cuda.synchronize();assert x.grad.is_cuda
        gpu=dict(nombre=torch.cuda.get_device_name(0),memoria_gb=round(torch.cuda.get_device_properties(0).total_memory/2**30,2),torch=torch.__version__,rocm=torch.version.hip)
        guardar(out/'ENTORNO_GPU.json',gpu);progreso(gpu=gpu)
        env=dict(os.environ);env['PYTHONPATH']=os.pathsep.join([str(REPO.parent/'.demanda-python-deps'),str(REPO/'scripts')])
        t=subprocess.run([sys.executable,'-m','pytest','-q',str(REPO/'scripts/test_m4_gpu.py'),'-p','no:cacheprovider'],env=env,capture_output=True,text=True,cwd=REPO)
        (out/'pruebas_implementacion.txt').write_text(t.stdout+'\n'+t.stderr)
        if t.returncode:raise RuntimeError(t.stdout[-3500:]+t.stderr[-1000:])
        datos=DatosNumericos(PAQUETE);progreso(estado='Ejecutando');log('M4 inicia en GPU '+gpu['nombre'])
        ejecutar(out,datos,diseno,run,m3,test,progreso,completar,log)
        assert estado['ajustes_completados']==465
        assert all(sha(p)==h for p,h in huellas.items()),'Se modificaron insumos o código'
        guardar(out/'VERIFICACION_FINAL.json',dict(insumos_conservados=True,tareas=465,checkpoints_finales=65))
        guardar(out/'MANIFIESTO_FINAL.json',[dict(archivo=str(p.relative_to(out)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(out.rglob('*')) if p.is_file() and p.suffix not in ['.tmp','.log'] and p.name not in ['proceso.lock','cola.lock','MANIFIESTO_FINAL.json'] and 'progreso' not in p.parts])
        progreso(estado='Finalizado',fase='finalizado',fin=time.time(),epoca=None,etiqueta='M3 y M4 entrenados, evaluados y guardados')
        publicacion_confirmada=False
        if args.publicar_al_final:
            progreso(estado='Publicando resultados',fase='publicacion',etiqueta='Publicación M2/M3/M4; apagado condicionado a confirmación remota')
            destino=REPO/'results'/('publicacion_m2_m3_m4_'+out.name.removeprefix('m4_gpu_'))
            with (out/'publicacion.log').open('a') as f:
                r=subprocess.run([sys.executable,str(REPO/'scripts/publicar_resultados_m2_m3_m4.py'),
                  '--cola',str(REPO/'results/COLA_M2_M3_M4.json'),'--salida',str(destino)],cwd=REPO,stdout=f,stderr=subprocess.STDOUT)
            if r.returncode:raise RuntimeError('La publicación falló; no se apaga. Consultar publicacion.log y '+str(destino))
            recibo=leer(out/'PUBLICACION.json')
            assert recibo['verificado'] and str(out) in recibo['ejecuciones'],'No se confirmó la publicación de esta ejecución'
            publicacion_confirmada=True
            progreso(estado='Resultados publicados',fase='publicacion',etiqueta='GitHub confirmado: '+recibo['commit'])
        if args.apagar_al_final:
            assert publicacion_confirmada,'Apagado bloqueado: falta publicación confirmada'
            r=subprocess.run(['shutdown','-h','+2'],capture_output=True,text=True)
            guardar(out/'APAGADO.json',dict(codigo=r.returncode,salida=r.stdout,error=r.stderr,fecha=dt.datetime.now(ZoneInfo('America/Bogota')).isoformat()))
            progreso(estado='Finalizado; apagado programado' if not r.returncode else 'Finalizado; apagado pendiente',etiqueta=r.stdout.strip() or r.stderr)
    except BaseException:
        e=traceback.format_exc();log(e)
        nombre='ERROR_PUBLICACION.json' if estado.get('fase')=='publicacion' else 'ERROR.json'
        guardar(out/nombre,dict(error=e,apagado=False))
        progreso(estado='Error de publicación; equipo encendido' if nombre=='ERROR_PUBLICACION.json' else 'Error; equipo encendido',error=e,fin=time.time())
    while True:time.sleep(30)


if __name__=='__main__':main()

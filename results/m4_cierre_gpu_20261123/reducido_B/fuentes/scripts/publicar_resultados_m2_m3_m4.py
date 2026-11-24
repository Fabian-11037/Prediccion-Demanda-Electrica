"""Publicación transaccional de resultados cerrados; no ejecuta apagado."""
from __future__ import annotations
import argparse
import csv
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from zoneinfo import ZoneInfo

REPO=Path(__file__).resolve().parents[1]
GH=REPO.parent/'.tools/gh-publicacion'
URL='https://github.com/Fabian-11037/Prediccion-Demanda-Electrica.git'
OWNER='Fabian-11037/Prediccion-Demanda-Electrica'


def fecha():return dt.datetime.now(ZoneInfo('America/Bogota')).isoformat(timespec='seconds')


def leer(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def guardar(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp')
    t.write_text(json.dumps(obj,ensure_ascii=False,indent=1),encoding='utf-8');t.replace(p)


def ejecutar(cmd,cwd=None,timeout=1800):
    env=dict(os.environ,GIT_TERMINAL_PROMPT='0',GIT_LFS_SKIP_SMUDGE='1',GCM_INTERACTIVE='never')
    r=subprocess.run([str(c) for c in cmd],cwd=cwd,env=env,capture_output=True,text=True,timeout=timeout)
    if r.returncode:raise RuntimeError(f'Falló {Path(str(cmd[0])).name}: '+r.stdout[-1500:]+r.stderr[-2500:])
    return r.stdout.strip()


def git(*args,cwd=None,timeout=1800):
    helper='!'+shlex.quote(str(GH))+' auth git-credential'
    return ejecutar(['git','-c','credential.helper=','-c','credential.helper='+helper,*args],cwd,timeout)


def incluir(p,run):
    rel=p.relative_to(run)
    if 'progreso' in rel.parts or '__pycache__' in rel.parts:return False
    if p.suffix in ['.npy','.log','.pid','.lock','.tmp','.pyc']:return False
    return p.name not in ['reanudar.pt','PREPARACION_LISTA.json','COLA_ESTADO.json','M3_PROCESO.json','M4_PROCESO.json','APAGADO.json','APAGADO_M2_CANCELADO.json','PUBLICACION.json','ERROR_PUBLICACION.json']


def validar_cierre(runs):
    for run in runs:
        if (run/'ERROR.json').exists():raise RuntimeError('Ejecución con error: '+str(run))
        if not (run/'MANIFIESTO_FINAL.json').exists() or not (run/'FINALIZADO.json').exists():
            raise RuntimeError('La ejecución todavía no está cerrada: '+str(run))
        f=leer(run/'FINALIZADO.json')
        if not (f.get('entrenamiento') and f.get('prueba')):raise RuntimeError('Falta entrenamiento o prueba: '+str(run))


def fuentes_publicacion():
    nombres=['e3_modelo2.py','e3_modelo3.py','e3_modelo4.py','ejecutar_m2_gpu.py','ejecutar_m3_gpu.py','ejecutar_m4_gpu.py',
             'test_m2_gpu.py','test_m3_gpu.py','test_m4_gpu.py','m2_progreso.html','m3_progreso.html','m4_progreso.html',
             'cola_m3_tras_m2.py','cola_m4_tras_m3.py','publicar_resultados_m2_m3_m4.py']
    return [REPO/'scripts'/n for n in nombres]


def preparar(runs,fuentes,dest,pubrel):
    publicados=[];omitidos=[]
    for run in runs:
        original={r['archivo']:r['sha256'] for r in leer(run/'MANIFIESTO_FINAL.json')}
        for p in sorted(run.rglob('*')):
            if not p.is_file():continue
            rel=p.relative_to(REPO)
            if not incluir(p,run):
                omitidos.append(dict(archivo=str(rel),bytes=p.stat().st_size));continue
            if p.stat().st_size>95*2**20:raise RuntimeError('Se necesita LFS para el archivo: '+str(rel))
            h=sha(p);clave=str(p.relative_to(run))
            if clave in original and original[clave]!=h:raise RuntimeError('Un resultado cambió tras el cierre: '+str(p))
            q=dest/rel;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
            publicados.append(dict(archivo=str(rel),bytes=q.stat().st_size,sha256=h))
    for p in fuentes:
        rel=p.relative_to(REPO);q=dest/rel;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
        publicados.append(dict(archivo=str(rel),bytes=q.stat().st_size,sha256=sha(q)))
    pub=dest/pubrel;pub.mkdir(parents=True,exist_ok=True)
    guardar(pub/'MANIFIESTO_PUBLICADO.json',publicados)
    guardar(pub/'ALCANCE.json',dict(fecha=fecha(),archivos_publicados=len(publicados),bytes=sum(p['bytes'] for p in publicados),
      omitidos_locales=omitidos,criterio='Se conservan resultados, predicciones, historiales, checkpoints, escaladores y fuentes. Embeddings NPY y estados reanudables quedan locales. Se omiten servicios, locks y logs temporales. Los manifiestos originales describen también esos archivos excluidos.',
      sin_despliegue=['JBook','dashboard']))
    lineas=['# Resultados GPU de M2, M3 y M4','','Entrenamiento y prueba retrospectiva2022–2025:1.461 orígenes diarios,24 horas y cinco semillas101–105. Configuraciones y transformaciones congeladas antes de prueba.','','| Modelo | Configuración | MASE de prueba | Informe |','|---|---:|---:|---|']
    resumen=[]
    for nombre,run in zip(['M2','M3','M4'],runs):
        cfg=leer(run/'CONFIGURACION_ELEGIDA.json')['configuracion']
        clave=nombre if nombre=='M2' else nombre+'_OP'
        with (run/'prueba/resumen_resultados.csv').open() as f:filas=list(csv.DictReader(f))
        valor=next(float(r['MASE']) for r in filas if r['modelo']==clave)
        informe='../'+run.name+'/INFORME.md'
        lineas.append(f'| {clave} | {cfg["id"]} | {valor:.6f} | [Informe]({informe}) |')
        resumen.append(dict(modelo=clave,configuracion=cfg,MASE=valor))
    lineas+=['','M3 y M4 utilizan M1 fuera de muestra mediante selección anidada por prefijo. Sus fuentes y registros documentan arquitectura, disponibilidad, controles, ablaciones y presupuesto. M4 incluye el ajuste Holm conjunto de las28 hipótesis; consultar sus contrastes completos antes de afirmar superioridad.','','El benchmark ya era conocido al diseñar los modelos: evaluación retrospectiva informada, no un holdout nunca consultado. No se descartan resultados desfavorables. La publicación incluye los checkpoints intermedios conservados y los finales; no repite entrenamientos ni modifica métricas.','','ALCANCE.json enumera las exclusiones locales y MANIFIESTO_PUBLICADO.json identifica los archivos incluidos. Los manifiestos originales se preservan. No se modifica ni despliega el JBook o el dashboard.','']
    (pub/'README.md').write_text('\n'.join(lineas),encoding='utf-8')
    return resumen,publicados


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cola',type=Path,required=True)
    ap.add_argument('--salida',type=Path,required=True);ap.add_argument('--comprobar-entorno',action='store_true')
    args=ap.parse_args();cola=leer(args.cola);out=args.salida.resolve();out.mkdir(parents=True,exist_ok=True)
    runs=[Path(cola[k]) for k in ['salida_m2','salida_m3','salida_m4']]
    if not GH.exists():raise RuntimeError('Falta GitHub CLI preparado: '+str(GH))
    ejecutar([GH,'auth','status'],timeout=60)
    if args.comprobar_entorno:
        head=git('ls-remote',URL,'refs/heads/main',timeout=60)
        print('Autenticación y lectura del remoto correctas; no se publicaron archivos. '+head)
        return
    validar_cierre(runs)
    fuentes=fuentes_publicacion()
    # Regla AGENTS: pruebas del proyecto antes de crear el commit. No se repiten pruebas GPU.
    env=dict(os.environ);env['PYTHONPATH']=os.pathsep.join([str(REPO.parent/'.demanda-python-deps'),str(REPO/'src')])
    t=subprocess.run([sys.executable,'-m','pytest','-q','tests','-p','no:cacheprovider'],cwd=REPO,env=env,capture_output=True,text=True,timeout=600)
    (out/'pruebas_proyecto.txt').write_text(t.stdout+'\n'+t.stderr,encoding='utf-8')
    if t.returncode:raise RuntimeError('No se publica: fallaron pruebas del proyecto. '+t.stdout[-1800:]+t.stderr[-1000:])
    pubrel=Path('results')/out.name
    inicio=dt.datetime.now(ZoneInfo('America/Bogota')).strftime('%Y%m%d_%H%M%S')
    for intento in range(1,4):
        dest=REPO.parent/f'Publicacion-M2-M3-M4-{inicio}-{intento}'
        git('clone','--depth','1','--filter=blob:none','--no-checkout',URL,str(dest))
        git('sparse-checkout','set','--no-cone','/AGENTS.md','/CONTEXTO_PROYECTO.md',cwd=dest)
        git('-c','filter.lfs.process=','-c','filter.lfs.required=false','checkout','main',cwd=dest)
        resumen,publicados=preparar(runs,fuentes,dest,pubrel)
        shutil.copy2(out/'pruebas_proyecto.txt',dest/pubrel/'pruebas_proyecto.txt')
        contexto=dest/'CONTEXTO_PROYECTO.md';texto=contexto.read_text(encoding='utf-8')
        titulo='## Cierre GPU de M2, M3 y M4 · '+fecha()[:10]
        if titulo not in texto:
            filas='; '.join(f'{r["modelo"]}: MASE {r["MASE"]:.6f}' for r in resumen)
            extra='\n\n'+titulo+'\n\nEntrenamiento y prueba finalizados en GPU con cinco semillas101–105,1.461 días2022–2025 y24 horas. '+filas+'. Se publican resultados favorables y desfavorables, predicciones, historiales, pesos, escaladores, fuentes y registros. La base M1 de M3/M4 es OOF anidada con selección por prefijo. M4 reúne las28 hipótesis para Holm conjunto. Evaluación retrospectiva informada por un benchmark conocido.\n\nEntrada de resultados: `'+str(pubrel/'README.md')+'`. Su alcance y manifiesto distinguen archivos publicados de embeddings/estados reconstruibles conservados localmente. Se preservan los manifiestos originales. No se modifica ni despliega JBook o dashboard.\n'
            contexto.write_text(texto+extra,encoding='utf-8')
        paths=[str(r.relative_to(REPO)) for r in runs]+[str(p.relative_to(REPO)) for p in fuentes]+[str(pubrel),'CONTEXTO_PROYECTO.md']
        git('add','--sparse','--force','--',*paths,cwd=dest)
        git('commit','--quiet','-m','Publica M2, M3 y M4 con predicciones, pesos y contrastes globales [skip ci]',cwd=dest)
        commit=git('rev-parse','HEAD',cwd=dest)
        guardar(out/'INTENTO.json',dict(fecha=fecha(),intento=intento,commit=commit,copia=str(dest)))
        try:git('push','origin','HEAD:main',cwd=dest,timeout=3600)
        except RuntimeError as e:
            guardar(out/f'error_push_{intento}.json',dict(fecha=fecha(),error=str(e)))
            if intento==3:raise
            continue
        remoto=git('ls-remote',URL,'refs/heads/main',timeout=60).split()[0]
        if remoto!=commit:
            git('fetch','--depth','50','origin','main',cwd=dest)
            git('merge-base','--is-ancestor',commit,'FETCH_HEAD',cwd=dest)
        manifest=str(pubrel/'MANIFIESTO_PUBLICADO.json')
        blob=git('rev-parse',commit+':'+manifest,cwd=dest)
        comprobado=ejecutar([GH,'api',f'repos/{OWNER}/contents/{manifest}?ref={commit}','--jq','.sha'],timeout=120)
        if comprobado!=blob:raise RuntimeError('El manifiesto remoto no coincide con el commit publicado')
        recibo=dict(fecha=fecha(),commit=commit,rama='main',head_remoto=remoto,verificado=True,ejecuciones=[str(r) for r in runs],manifiesto_remoto=manifest,
                    sha_git_manifiesto=blob,archivos=len(publicados),resumen=resumen,url='https://github.com/'+OWNER+'/commit/'+commit)
        guardar(out/'PUBLICADO.json',recibo);guardar(runs[-1]/'PUBLICACION.json',recibo)
        guardar(REPO/'results/PUBLICACION_M2_M3_M4.json',recibo)
        print('Publicación confirmada: '+recibo['url'],flush=True)
        return
    raise RuntimeError('No se confirmó la publicación; no apagar')


if __name__=='__main__':main()

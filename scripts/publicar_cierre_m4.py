"""Publica un cierre completo; verifica manifiesto remoto. Nunca apaga el PC."""
import argparse, datetime as dt, hashlib, json, shutil, subprocess, sys, time
from pathlib import Path
from zoneinfo import ZoneInfo
from publicar_resultados_m2_m3_m4 import git, ejecutar, GH, OWNER, URL, sha, guardar, leer
REPO=Path(__file__).resolve().parents[1]

def incluir(p,run):
    rel=p.relative_to(run)
    return not any(x in rel.parts for x in ['progreso','__pycache__']) and p.suffix not in ['.log','.lock','.tmp','.pyc'] and p.name not in ['reanudar.pt','PUBLICADO.json','ERROR_PUBLICACION.json','APAGADO.json','SERVIDOR.json']

def preparar(run,dest):
    manifiesto=[];omitidos=[];candidatos=[p for p in run.rglob('*') if p.is_file()]
    candidatos += list((REPO/'scripts').glob('m4_cierre_*.py'))+[REPO/'scripts/ejecutar_cierre_m4.py',REPO/'scripts/publicar_cierre_m4.py',REPO/'scripts/adquirir_datasets_cierre_m4.py',REPO/'scripts/m4_onnx_congelado.py',REPO/'scripts/pruebas_m4_congelado.py',REPO/'scripts/fase_capacidad_colombia_m4.py']
    candidatos += [REPO/'scripts'/n for n in ['reconstruir_artefactos_m4.py','verificar_onnx_portable.py','verificar_propiedad_m4.py','auditar_capacidad_soporte_m4.py']]
    candidatos += list((REPO/'experiments/m4_originalidad').rglob('*'))+list((REPO/'config/m4_cierre_benchmark').glob('*.json'))
    # Fuentes y paneles ajenos: publicar procedencia/recetas, no datos brutos.
    candidatos += [REPO/'data/external/m4_cierre_20261123'/n for n in ['FUENTES.json','PANELES.json']]
    nuevos=[]
    for p in sorted(set(candidatos)):
        if not p.is_file():continue
        if run in p.parents and not incluir(p,run):omitidos.append(str(p.relative_to(REPO)));continue
        rel=p.relative_to(REPO);q=dest/rel;q.parent.mkdir(parents=True,exist_ok=True)
        if p.stat().st_size<=90*2**20:
            shutil.copy2(p,q);reg=dict(archivo=str(rel),sha256=sha(p),bytes=p.stat().st_size);manifiesto.append(reg);nuevos.append(q)
        else:
            # Fragmentos exactos de64MiB, sin cuantizar ni modificar pesos.
            partes=[]
            with p.open('rb') as f:
                i=0
                while bloque:=f.read(64*2**20):
                    pp=q.with_name(q.name+f'.parte_{i:04d}');pp.write_bytes(bloque);h=sha(pp)
                    partes.append(dict(archivo=pp.name,sha256=h,bytes=len(bloque)));nuevos.append(pp);i+=1
            indice=q.with_name(q.name+'.partes.json');guardar(indice,dict(original=q.name,sha256=sha(p),bytes=p.stat().st_size,partes=partes));nuevos.append(indice)
            manifiesto.append(dict(archivo=str(rel),sha256=sha(p),bytes=p.stat().st_size,fragmentado=True,indice=str(indice.relative_to(dest)),partes=partes))
    pubrel=run.relative_to(REPO)/'publicacion';pub=dest/pubrel;pub.mkdir(exist_ok=True)
    guardar(pub/'MANIFIESTO_PUBLICADO.json',manifiesto);guardar(pub/'ALCANCE.json',dict(omitidos=omitidos,datos_brutos_terceros='conservados locales; se publican recetas y SHA256',optimizadores='reanudar.pt permanece local; checkpoints seleccionados publicados completos',fragmentos='Concatenar partes en orden y verificar SHA256 original; no cuantización'))
    nuevos.extend([pub/'MANIFIESTO_PUBLICADO.json',pub/'ALCANCE.json'])
    # No reemplazar el contexto remoto por la copia antigua del entrenamiento.
    ctx=dest/'CONTEXTO_PROYECTO.md';texto=ctx.read_text();marca='## Evaluación adicional M4_N en GPU · cierre20261123'
    if marca not in texto:
        texto+='\n\n'+marca+'\n\nSe ejecuta la cola autorizada sobre Colombia,UCI,GEFCom2014-L y AlemaniaOPSD con diez realizaciones101–110, controles de capacidad, ablaciones y comparadores numéricos adicionales. Los resultados se conservan en `'+str(run.relative_to(REPO))+'/INFORME.md`. La selección colombiana es retrospectiva; no se certifica originalidad ni potencia0,90. ONNX conserva su rechazo de portabilidad en empates y las plataformas no ejecutadas permanecen pendientes. El benchmark numérico adicional no sustituye al Top10 histórico con texto. No se modifica ni despliega JBook/dashboard.\n'
        ctx.write_text(texto);nuevos.append(ctx)
    return nuevos,pubrel,len(manifiesto)

def publicar(run):
    assert leer(run/'FINALIZADO.json')['prueba'] and (run/'MANIFIESTO_FINAL.json').exists()
    assert not (run/'ERROR_COLA_COMPLETA.json').exists()
    # Exigencia AGENTS, una vez al cierre; sin reentrenar ni repetir GPU.
    import os
    env=dict(os.environ,PYTHONPATH=str(REPO.parent/'.demanda-python-deps')+':'+str(REPO/'src'))
    r=subprocess.run([sys.executable,'-m','pytest','-q','tests'],cwd=REPO,env=env,capture_output=True,text=True)
    (run/'pytest_publicacion.txt').write_text(r.stdout+r.stderr);assert r.returncode==0,'Fallaron pruebas del proyecto antes de commit'
    for intento in range(1,4):
        dest=REPO.parent/f'Publicacion-Cierre-M4-{dt.datetime.now(ZoneInfo("America/Bogota")).strftime("%Y%m%d_%H%M%S")}-{intento}'
        git('clone','--depth','1','--filter=blob:none','--sparse',URL,str(dest))
        archivos,pubrel,n=preparar(run,dest)
        # Mantener cada push por debajo de700MiB. No forzar ni reescribir remoto.
        grupos=[];grupo=[];tam=0
        for p in archivos:
            b=p.stat().st_size
            if grupo and tam+b>650*2**20:grupos.append(grupo);grupo=[];tam=0
            grupo.append(p);tam+=b
        if grupo:grupos.append(grupo)
        try:
            for i,grupo in enumerate(grupos,1):
                lista=dest.parent/f'.rutas-publicacion-m4-{i}.txt';lista.write_text('\n'.join(str(p.relative_to(dest)) for p in grupo)+'\n')
                git('add','--sparse','--force','--pathspec-from-file='+str(lista),cwd=dest)
                if git('diff','--cached','--name-only',cwd=dest):
                    git('-c','user.name=Fabian-11037','-c','user.email=Fabian-11037@users.noreply.github.com','commit','-m',f'Publica cierre M4_N: artefactos {i}/{len(grupos)} [skip ci]',cwd=dest)
                    git('push','origin','HEAD:main',cwd=dest)
            commit=git('rev-parse','HEAD',cwd=dest)
            remoto=ejecutar([GH,'api',f'repos/{OWNER}/commits/main','--jq','.sha']);assert remoto==commit
            ruta=str(pubrel/'MANIFIESTO_PUBLICADO.json');blob=git('hash-object',ruta,cwd=dest)
            remoto_blob=ejecutar([GH,'api',f'repos/{OWNER}/contents/{ruta}?ref={commit}','--jq','.sha']);assert blob==remoto_blob
            recibo=dict(fecha=dt.datetime.now(ZoneInfo('America/Bogota')).isoformat(),commit=commit,verificado=True,manifiesto=ruta,sha_git_manifiesto=blob,archivos_originales=n,url=f'https://github.com/{OWNER}/commit/{commit}')
            guardar(run/'PUBLICADO.json',recibo);return
        except Exception:
            if intento==3:raise
            time.sleep(2)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    try:publicar(a.out.resolve())
    except Exception:
        import traceback
        guardar(a.out/'ERROR_PUBLICACION.json',dict(error=traceback.format_exc(),apagado=False));raise

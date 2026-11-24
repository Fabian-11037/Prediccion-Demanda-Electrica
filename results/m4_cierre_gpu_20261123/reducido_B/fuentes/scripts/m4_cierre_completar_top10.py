"""Completa dos contrastes históricos con pérdidas publicadas, sin inventar predicciones."""
import argparse,datetime,hashlib,json,shutil,subprocess,sys
from pathlib import Path
from zoneinfo import ZoneInfo
from m4_cierre_estadistica import np,pd,stats,hac,estacionario,holm
from publicar_resultados_m2_m3_m4 import git,ejecutar,GH,OWNER,URL,sha,guardar
ROOT=Path(__file__).resolve().parents[1]

def completar_familia(out):
    """Resolución Monte Carlo fijada antes de la nueva inferencia de prueba."""
    car=out/'ampliacion_top10';car.mkdir(exist_ok=True);filas=[];B=49999;fuentes=[]
    for archivo in sorted((out/'estadistica').glob('perdidas_*.npz')):
        dataset=archivo.stem.replace('perdidas_','')
        ampliado=out/'complemento/estadistica_final'/archivo.name
        if ampliado.exists():archivo=ampliado
        with np.load(archivo) as z:
            dias=z['dia'];errores={k:z[k].mean(0) for k in z.files if k!='dia'}
        if dataset=='colombia':
            for nombre,clave,esperado in [('TimesNet','nuevo_timesnet__N',.642927818004554),('NBEATS','nuevo_nbeats__N',.9533426171053975)]:
                p=out/'insumos'/f'{clave}.json';d=json.loads(p.read_text());individuales='historico_'+nombre in errores
                v=errores['historico_'+nombre] if individuales else np.asarray(d['mae_daily'],dtype='float64')/215617.90361589016
                assert v.shape==(len(dias),) and abs(v.mean()-esperado)<(1e-5 if individuales else 1e-10)
                errores['historico_'+nombre]=v;fuentes.append(dict(archivo=str((archivo if individuales else p).relative_to(ROOT)),sha256=sha(archivo if individuales else p),MASE=float(v.mean()),semillas=5,tipo='pérdidas de predicciones individuales restauradas' if individuales else 'pérdidas publicadas promediadas',precision_mae_kwh=None if individuales else 1e-6))
        parejas=[('M4_N',k) for k in errores if k!='M4_N']
        if dataset=='colombia' and 'M4_OP' in errores:
            parejas += [('M4_O','M4_N'),('M4_OP','M4_N'),('M4_OP','M4_O'),('M4_OP','M4_metadatos'),('M4_OP','M4_desalineado_promedio')]
        n=len(dias);idx=estacionario(n,B=B)
        for a,nombre in parejas:
            objetivo=errores[a];rival=errores[nombre];co=np.column_stack([np.ones(n-2),objetivo[:-2]])
            ganancia=1-objetivo.mean()/rival.mean();gb=[]
            for inds in np.array_split(idx,50):gb.append(1-objetivo[inds].mean(1)/rival[inds].mean(1))
            ic=np.quantile(np.concatenate(gb),[.025,.975])
            for margen in [0.,.03]:
                dd=(1-margen)*rival-objetivo;z=dd-dd.mean();vb=max(float(hac(dd)[0,0]),1e-30)
                t=dd.mean()/np.sqrt(vb/n)*np.sqrt((n-1)/n);pdm=float(stats.t.sf(t,n-1))
                mayores=sum(int(np.sum(z[inds].mean(1)>=dd.mean())) for inds in np.array_split(idx,50));pb=(1+mayores)/(B+1)
                g=co*dd[2:,None];media=g.mean(0);cov=hac(g);rango=np.linalg.matrix_rank(cov);pgw=float(stats.chi2.sf((n-2)*media@np.linalg.pinv(cov)@media,rango)) if rango else 1.
                for metodo,pv in [('bootstrap_estacionario',pb),('DM_HLN',pdm),('GW_bilateral',pgw)]:
                    filas.append(dict(dataset=dataset,A=a,B=nombre,metodo=metodo,margen_relativo=margen,ganancia_relativa=ganancia,ic95_inf=ic[0],ic95_sup=ic[1],p=pv,favorable=ganancia>margen,replicas_bootstrap=B,rezago_instrumento_GW=2,alcance='retrospectivo' if dataset=='colombia' else 'evaluación adicional condicionada a supuestos'))
        del idx
    assert len(filas)<=1536
    ajustes=holm([f['p'] for f in filas]+[1.]*(1536-len(filas)))[:len(filas)]
    for f,p in zip(filas,ajustes):f['p_holm_cota_familia1536']=float(p)
    pd.DataFrame(filas).to_csv(car/'contrastes_F2_Top10_completo.csv',index=False)
    guardar(car/'PROCEDENCIA.json',dict(fecha=datetime.datetime.now(ZoneInfo('America/Bogota')).isoformat(),fuentes=fuentes,replicas_bootstrap=B,familia_maxima=1536,observadas=len(filas),resolucion_minima_p=1/(B+1),
        adenda='experiments/m4_originalidad/ADENDA_COMPLETAR_PLAN.md',no_reconstruccion='Se usan predicciones individuales restauradas de los cinco pesos originales cuando están disponibles; en su ausencia, pérdidas diarias publicadas. Nunca se reconstruyen realizaciones desde agregados.',
        alcance='La resolución se fijó por multiplicidad antes de nuevas pruebas; no se adaptan datos, modelos ni hipótesis. El paso inicial9999 se conserva como resultado de precisión menor. La inferencia final es este CSV49999.',familias_historicas_intactas=[28,220]))
    (car/'INFORME.md').write_text('# Evaluación temporal final y Top10 histórico\n\nSe integran TimesNet y N-BEATS con las pérdidas de las predicciones individuales restauradas de sus cinco checkpoints originales; si faltan, se usan las pérdidas diarias publicadas y se identifica la procedencia. Son errores individuales promediados, distintos del error de la predicción ensemble. No se inventan pronósticos ni réplicas.\n\nEl CSV final utiliza49999 réplicas, tres procedimientos y Holm de la familia máxima1536. Sustituye únicamente la resolución Monte Carlo de la nueva evaluación, conservando el paso preliminar y las familias históricas28/220. Colombia sigue siendo retrospectiva. El Top10 histórico conserva cinco semillas, frente a diez de los ajustes nuevos. No se certifica potencia0,90, originalidad ni validez confirmatoria de pruebas cuyos supuestos no estén establecidos.\n',encoding='utf-8')
    with (car/'INFORME.md').open('a') as f:f.write('\nGW utiliza la pérdida observable de dos días antes y excluye los dos orígenes iniciales. El GW preliminar de rezago1 no se interpreta; se conserva como procedencia. Consultar la adenda registrada antes de nuevas pruebas.\n')
    (out/'RESULTADOS_FINALES.md').write_text('# Resultados finales de la evaluación adicional\n\n- [Métricas, entrenamiento y predicciones](INFORME.md).\n- [Evaluación complementaria y pendientes](complemento/INFORME.md).\n- [Inferencia estadística final y Top10 completo](ampliacion_top10/INFORME.md).\n- [Contrastes finales49999 y Holm](ampliacion_top10/contrastes_F2_Top10_completo.csv).\n- [Auditoría ONNX y rechazo de portabilidad en empates](pruebas_congeladas/RESULTADO.json).\n\nLas tablas de estadistica/ corresponden al paso preliminar9999. Para conclusiones F2 usar la tabla final49999 con instrumento GW observable de rezago2. Las familias históricas28/220 permanecen intactas. La selección colombiana es retrospectiva y esta ejecución no certifica el cierre metodológico del E3.\n',encoding='utf-8')
    return car

def publicar(out,car):
    import os
    env=dict(os.environ,PYTHONPATH=str(ROOT.parent/'.demanda-python-deps')+':'+str(ROOT/'src'))
    t=subprocess.run([sys.executable,'-m','pytest','-q','tests'],cwd=ROOT,env=env,capture_output=True,text=True)
    (car/'pytest_publicacion.txt').write_text(t.stdout+t.stderr);assert t.returncode==0
    guardar(car/'MANIFIESTO.json',[dict(archivo=str(p.relative_to(ROOT)),sha256=sha(p),bytes=p.stat().st_size) for p in [*sorted(car.rglob('*')),out/'RESULTADOS_FINALES.md'] if p.is_file() and p.name!='MANIFIESTO.json'])
    destino=ROOT.parent/f'Publicacion-Cierre-M4-Top10-{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}'
    git('clone','--depth','1','--filter=blob:none','--sparse',URL,str(destino))
    for p in [*car.rglob('*'),out/'RESULTADOS_FINALES.md',Path(__file__),ROOT/'scripts/m4_cierre_guardia_apagado.py']:
        if not p.is_file():continue
        q=destino/p.relative_to(ROOT);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
    ctx=destino/'CONTEXTO_PROYECTO.md'
    with ctx.open('a') as f:f.write('\n\n## Cierre estadístico de la evaluación adicional M4_N\n\nLa etapa final utiliza49999 réplicas y GW con pérdidas observables de dos días antes. Integra las pérdidas canónicas publicadas de TimesNet y N-BEATS, promediadas entre cinco semillas, sin reconstruir pronósticos. La entrada vigente es `'+str(out.relative_to(ROOT))+'/RESULTADOS_FINALES.md`; las tablas preliminares9999 se conservan solo como procedencia. Modelos y datos no se reajustan.\n')
    rel=car.relative_to(ROOT);git('add','--sparse','--force',str(rel),str((out/'RESULTADOS_FINALES.md').relative_to(ROOT)),'CONTEXTO_PROYECTO.md','scripts/m4_cierre_completar_top10.py','scripts/m4_cierre_guardia_apagado.py',cwd=destino)
    git('-c','user.name=Fabian-11037','-c','user.email=Fabian-11037@users.noreply.github.com','commit','-m','Completa contrastes canónicos del Top10 sin reentrenar [skip ci]',cwd=destino);git('push','origin','HEAD:main',cwd=destino)
    commit=git('rev-parse','HEAD',cwd=destino);remoto=ejecutar([GH,'api',f'repos/{OWNER}/commits/main','--jq','.sha']);assert commit==remoto
    ruta=str(rel/'MANIFIESTO.json');blob=git('hash-object',ruta,cwd=destino);assert blob==ejecutar([GH,'api',f'repos/{OWNER}/contents/{ruta}?ref={commit}','--jq','.sha'])
    guardar(out/'AMPLIACION_TOP10_PUBLICADA.json',dict(verificado=True,commit=commit,manifiesto=ruta,sha_git_manifiesto=blob,url=f'https://github.com/{OWNER}/commit/{commit}'))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--out',type=Path,required=True);args=a.parse_args();c=completar_familia(args.out);publicar(args.out,c)

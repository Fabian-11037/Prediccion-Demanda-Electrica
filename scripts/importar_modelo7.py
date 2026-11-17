"""Importa TGForecaster de archivos guardados; no deserializa pesos ni entrena."""
from pathlib import Path
import hashlib,io,json,pickletools,re,shutil,zipfile
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
CONS=ROOT.parent/'Consolidado-Entrenamientos-20261115'
OUT=CONS/'consolidado'
EID='adj_modelo7_20261114_0430'
Q=215617.90361589016

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def rel(p):return p.relative_to(CONS).as_posix()

def origen_estatico(path):
    with zipfile.ZipFile(path) as z,z.open('origen.npy') as f:
        assert np.lib.format.read_magic(f)==(1,0)
        shape,_,dtype=np.lib.format.read_array_header_1_0(f)
        assert shape==(1461,) and dtype.hasobject
        dates=[arg for op,arg,_ in pickletools.genops(f.read())
            if op.name in {'SHORT_BINUNICODE','BINUNICODE','BINUNICODE8'}
            and isinstance(arg,str) and re.fullmatch(r'\d{4}-\d{2}-\d{2}',arg)]
    expected=pd.date_range('2022-01-01','2025-12-31',freq='D')
    assert dates==expected.strftime('%Y-%m-%d').tolist()
    return expected.tz_localize('America/Bogota')

def main():
    backup=ROOT/'results/modelo7_integracion/respaldo';backup.mkdir(parents=True,exist_ok=True)
    for name in ['indice_modelos.json','metricas_por_semilla.csv','resumen_todos.csv','resumen_principal.csv',
                 'predicciones_finales.parquet','registro_ajustes.csv','validacion_por_ajuste.csv','metricas_por_horizonte.csv']:
        p=OUT/name
        if not (backup/name).exists():shutil.copy2(p,backup/name)
    dest=CONS/'fuentes/adjuntos/modelo7_20261114_0430';dest.mkdir(parents=True,exist_ok=True)
    inventory=[];packages=[]
    for name in ['ENVIAR_resultados_modelo7_20261114_0430.zip','ENVIAR_pesos_y_vectores_modelo7_20261114_0430.zip']:
        p=Path('C:/Users/Work/Downloads')/name
        packages.append(dict(archivo=name,procedencia='adjunto',sha256=sha(p),bytes_zip=p.stat().st_size))
        with zipfile.ZipFile(p) as z:
            for member in z.infolist():
                if member.is_dir():continue
                target=(dest/member.filename).resolve()
                assert target.is_relative_to(dest.resolve()) and not member.filename.startswith(('/','\\'))
                content=z.read(member)
                if target.exists():assert target.read_bytes()==content
                else:
                    target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(content)
                inventory.append(dict(archivo=name,ruta=rel(target),bytes=len(content),sha256=hashlib.sha256(content).hexdigest()))
    base=dest/'resultados_modelo7';register=read(base/'REGISTRO.json')
    source=ROOT/'results/arquitecturas_e2/fuentes/Modelo7_TGForecaster'
    checks=[]
    for name,h in register['huellas_sha256'].items():
        p=source/name
        if p.is_file():
            data=p.read_bytes();actual=hashlib.sha256(data).hexdigest();lf=hashlib.sha256(data.replace(b'\r\n',b'\n')).hexdigest()
            assert h in {actual,lf},name
            checks.append(dict(archivo=name,sha256_registrado=h,sha256_local=actual,equivalente_LF=actual!=h))
    assert any(x['archivo']=='src/entrenamiento/redes.py' for x in checks)
    selected=read(base/'busqueda/configuracion_elegida.json')
    contract=read(base/'final/resumen_prueba.json');assert contract['q168_kwh']==Q
    panel=pd.read_parquet(ROOT/'data/processed/panel_horario.parquet',columns=['y_demanda_real_kwh'])
    dates=pd.date_range('2022-01-01','2025-12-31',freq='D',tz='America/Bogota')
    pos=panel.index.get_indexer(dates);Y=panel.y_demanda_real_kwh.to_numpy()[pos[:,None]+np.arange(24)]
    models=[];metrics=[];frames=[];horizons=[];trials=[];summaries=[]
    for variant in ['N','N+O']:
        runs=[]
        for seed in range(1000,1005):
            folder=base/'final'/variant.replace('+','_')/f'semilla_{seed}'
            result=read(folder/'resultado.json');predpath=folder/'pred_prueba.npz'
            assert origen_estatico(predpath).equals(dates)
            with np.load(predpath,allow_pickle=False) as z:p=z['yhat_kwh'].astype(float);labels=z['y_kwh'].astype(float)
            assert p.shape==labels.shape==Y.shape==(1461,24) and np.isfinite(p).all()
            difference=float(np.max(np.abs(labels-Y)));assert difference<=1.01
            err=p-Y;ae=np.abs(err);stored=result['metricas']
            m=dict(MAE=float(ae.mean()),MASE=float(ae.mean()/Q),RMSE=float(np.sqrt((err**2).mean())),
                MAPE=float(100*(ae/np.abs(Y)).mean()),sMAPE=float(200*(ae/(np.abs(Y)+np.abs(p))).mean()))
            assert abs(m['MASE']-stored['MASE'])<1e-5
            assert result['semilla']==seed and result['variante']==variant
            row=dict(experimento=EID,modelo='TGForecaster',variante=variant,semilla=seed,**m,
                MASE_reportado=stored['MASE'],diferencia_MASE_recalculada=abs(m['MASE']-stored['MASE']),
                epocas=result['epocas'],mejor_epoca=result['mejor_epoca'],segundos=result['segundos'],parametros=result['n_parametros'],
                MASE_ajuste=result['MASE_ajuste'],MASE_parada=result['MASE_parada'],n_origenes=1461,horas=35064,
                origen_fuente='adjunto',principal=True,estado='principal')
            runs.append(row);metrics.append(row)
            artifacts={k:rel(folder/name) for k,name in [('pesos','modelo.pt'),('escaladores','escaladores.json'),
                ('historia','historia.csv'),('predicciones','pred_prueba.npz'),('resultado','resultado.json')]}
            models.append(dict(experimento=EID,modelo='TGForecaster',variante=variant,semilla=seed,
                artefactos=artifacts,sha256_checkpoint=sha(folder/'modelo.pt'),origenes_verificados=True,
                max_diferencia_etiqueta_kwh=difference,diferencia_MASE=abs(m['MASE']-stored['MASE']),principal=True))
            frames.append(pd.DataFrame(dict(experimento=EID,variante=variant,semilla=seed,origen=np.repeat(dates,24),
                horizonte=np.tile(np.arange(1,25),1461),observado_kwh=Y.ravel(),predicho_kwh=p.ravel())))
            horizons.extend(dict(experimento=EID,variante=variant,semilla=seed,horizonte=h+1,MASE=float(ae[:,h].mean()/Q)) for h in range(24))
            trials.append(dict(experimento=EID,variante=variant,etapa='final',configuracion=result['configuracion']['id'],
                semilla=seed,pliegue='final',MASE=stored['MASE'],epocas=result['epocas'],mejor_epoca=result['mejor_epoca'],
                segundos=result['segundos'],ruta=rel(folder/'resultado.json')))
        vtrials=[]
        for path in sorted((base/'busqueda'/variant.replace('+','_')).glob('pliegue_*/cfg_*/resultado.json')):
            result=read(path)
            vtrials.append(dict(experimento=EID,variante=variant,etapa='busqueda',configuracion=result['configuracion']['id'],
                semilla=result['semilla'],pliegue=path.parents[1].name,MASE=result['metricas']['MASE'],
                epocas=result['epocas'],mejor_epoca=result['mejor_epoca'],segundos=result['segundos'],ruta=rel(path)))
        assert len(vtrials)==35;trials.extend(vtrials)
        g=pd.DataFrame(runs);secs=sum(x['segundos'] for x in vtrials)+g.segundos.sum()
        summaries.append(dict(experimento=EID,modelo='TGForecaster',variante=variant,principal=True,semillas=5,
            MASE_validacion=selected[variant]['MASE_medio_validacion'],anios_validacion='2018,2019,2020,2021',
            MASE_prueba=g.MASE.mean(),MASE_sd=g.MASE.std(ddof=1),MAE_kwh=g.MAE.mean(),RMSE_kwh=g.RMSE.mean(),
            MAPE_pct=g.MAPE.mean(),sMAPE_pct=g.sMAPE.mean(),mejora_semanal_pct=100*(1-g.MASE.mean()/1.354906444254724),
            MASE_ajuste=g.MASE_ajuste.mean(),MASE_parada=g.MASE_parada.mean(),epocas_media=g.epocas.mean(),
            epocas_min=int(g.epocas.min()),epocas_max=int(g.epocas.max()),mejor_epoca_media=g.mejor_epoca.mean(),
            parametros=int(g.parametros.iloc[0]),segundos_finales=g.segundos.sum(),ajustes=40,segundos_ajustes=secs,
            configuracion=str(selected[variant]['configuracion']['id']),indicadores_ausencia=3,codigo_fuente_incluido=True))
    def append_csv(name,rows):
        old=pd.read_csv(OUT/name);old=old.loc[old.experimento.ne(EID)]
        value=pd.concat([old,pd.DataFrame(rows)],ignore_index=True);value.to_csv(OUT/name,index=False);return value
    append_csv('metricas_por_semilla.csv',metrics);append_csv('metricas_por_horizonte.csv',horizons)
    append_csv('registro_ajustes.csv',trials);append_csv('validacion_por_ajuste.csv',[x for x in trials if x['etapa']!='final'])
    total=append_csv('resumen_todos.csv',summaries)
    main=total.loc[total.principal].sort_values('MASE_prueba').reset_index(drop=True)
    main.insert(0,'orden_descriptivo',np.arange(1,len(main)+1));main.to_csv(OUT/'resumen_principal.csv',index=False)
    old=pd.read_parquet(OUT/'predicciones_finales.parquet');old=old.loc[old.experimento.ne(EID)]
    combined=pd.concat([old,*frames],ignore_index=True)
    assert not combined.duplicated(['experimento','variante','semilla','origen','horizonte']).any()
    combined.to_parquet(OUT/'predicciones_finales.parquet',index=False,compression='zstd')
    index=read(OUT/'indice_modelos.json');index['modelos']=[m for m in index['modelos'] if m['experimento']!=EID]+models
    index['configuraciones'][EID]=dict(familia='TGForecaster',fuente='adjunto',val={v:s['MASE_medio_validacion'] for v,s in selected.items()},
        configuracion={v:s['configuracion'] for v,s in selected.items()},anios_media_validacion=[2018,2019,2020,2021],
        mascaras=3,configuraciones_filtro=20,semillas_filtro=[43],finalistas=5,semillas_confirmacion=[43],
        piloto='sin piloto independiente',arquitectura=register['plan']['arquitectura_fija'],codigo_fuente_incluido=True,
        inicio=register['creado'],fin=register['terminado'],registro=rel(base/'REGISTRO.json'))
    (OUT/'indice_modelos.json').write_text(json.dumps(index,ensure_ascii=False,indent=2),encoding='utf-8')
    cache=[]
    for path in [base/'texto/vectores_documento.npy',base/'texto/descripciones/vectores_documento.npy']:
        array=np.load(path,allow_pickle=False,mmap_mode='r');assert array.shape[1]==768 and np.isfinite(array).all()
        ids=pd.read_parquet(path.with_name('indice.parquet'));assert len(ids)==len(array)
        cache.append(dict(ruta=rel(path),sha256=sha(path),shape=list(array.shape),dtype=str(array.dtype)))
    audit=dict(experimento=EID,variantes=['N','N+O'],semillas=list(range(1000,1005)),finales=10,
        ajustes_busqueda=70,filas_nuevas=350640,origenes=1461,horizontes=24,fuentes_verificadas=checks,
        paquetes=packages,archivos=inventory,vectores=cache,checkpoints_cargados=False,entrenamiento=False,publicacion=False)
    report=ROOT/'results/modelo7_integracion';(report/'importacion.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    pd.DataFrame(inventory).to_csv(report/'inventario.csv',index=False)
    manifest=read(CONS/'manifest_fuentes.json');manifest['paquetes']=[x for x in manifest['paquetes'] if 'modelo7_' not in x['archivo']]+packages
    global_inventory=pd.read_csv(CONS/'inventario_archivos.csv')
    global_inventory=global_inventory.loc[~global_inventory.paquete.str.contains('modelo7_',regex=False)]
    extra=[dict(procedencia='adjunto',paquete=row['archivo'],ruta_interna=row['ruta'].split('modelo7_20261114_0430/',1)[1],
        ruta_local=row['ruta'],bytes=row['bytes'],sha256=row['sha256']) for row in inventory]
    global_inventory=pd.concat([global_inventory,pd.DataFrame(extra)],ignore_index=True)
    global_inventory.to_csv(CONS/'inventario_archivos.csv',index=False)
    manifest['archivos']=len(global_inventory)
    (CONS/'manifest_fuentes.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    vectors=read(OUT/'indice_vectores.json')
    vectors['alias']=[x for x in vectors['alias'] if 'modelo7_' not in x['ruta']]
    vectors['alias'].append(dict(ruta=cache[0]['ruta'],indice_documentos=rel(base/'texto/indice.parquet'),
        sha256=cache[0]['sha256'],documentos=14162,dimension=768,dtype=cache[0]['dtype']))
    vectors['caches_unicos']=len({x['sha256'] for x in vectors['alias']})
    vectors['descripciones_canales']=cache[1]
    (OUT/'indice_vectores.json').write_text(json.dumps(vectors,ensure_ascii=False,indent=2),encoding='utf-8')
    for name in ['RESUMEN_ENTRENAMIENTO_Y_PRUEBA.md','README.md']:
        if not (backup/name).exists():shutil.copy2(CONS/name,backup/name)
    lines=['# Resultados consolidados de entrenamiento y prueba','',
        'La consolidación ampliada incluye diez procedimientos principales con cinco semillas cada uno. '
        'El libro incorpora además cuatro familias de demanda y calendario, para catorce procedimientos y dos referencias deterministas.',
        '', '| Modelo | Variante | MASE validación | MASE prueba | MAE prueba (kWh) |', '|---|---|---:|---:|---:|']
    for row in main.itertuples():lines.append(f'| {row.modelo} | {row.variante} | {row.MASE_validacion:.6f} | {row.MASE_prueba:.6f} | {row.MAE_kwh:.2f} |')
    lines+=['','TGForecaster conserva 20 candidatos por variante, cinco finalistas y cinco semillas finales: 80 ajustes en total. '
        'N utiliza descripciones estáticas de 33 canales; N+O añade documentos oficiales. Se validan diez checkpoints, '
        '350.640 predicciones nuevas, fechas completas y métricas reproducidas, sin cargar pesos ni ejecutar redes.',
        '', 'Los artefactos y vectores conservan sus SHA-256. Los 14.162 embeddings oficiales coinciden con el caché de TaTS/fusión; '
        'las 33 descripciones de canal son un recurso adicional. Los originales y alternativas permanecen inventariados.',
        '', 'Las diferencias de presupuesto, máscaras y entradas impiden atribuir los resultados exclusivamente a la arquitectura. '
        'La inferencia conjunta, los contrastes textuales y la precisión de desarrollo se reproducen en results/evaluacion_e2 del JBook. '
        'La integración no ejecuta entrenamiento ni publicación.']
    (CONS/'RESUMEN_ENTRENAMIENTO_Y_PRUEBA.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    cached=ROOT/'results/evaluacion_e2/entradas/consolidado'
    for name in ['predicciones_finales.parquet','resumen_principal.csv']:shutil.copy2(OUT/name,cached/name)
    print(pd.DataFrame(summaries)[['modelo','variante','MASE_validacion','MASE_prueba','MAE_kwh','MASE_parada']].to_string(index=False))
    print('Verificados:',len(models),'checkpoints sin cargar;',len(checks),'fuentes;',len(inventory),'archivos.')

if __name__=='__main__':main()

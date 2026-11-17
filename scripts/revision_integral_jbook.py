"""Contratos numéricos, alcance editorial e inventario de tablas y gráficas."""
from pathlib import Path
from html.parser import HTMLParser
from itertools import combinations
import hashlib,io,json,re,shutil,zipfile
import numpy as np
import pandas as pd
import nbformat as nb
from scipy import stats
from statsmodels.stats.multitest import multipletests


def corregir_capitulo10(root,cells):
    out=Path(root)/'results/evaluacion_e2'
    p=json.loads((out/'protocolo.json').read_text(encoding='utf-8'))
    temporal=pd.read_csv(out/'comparaciones_temporales.csv')
    seeds=pd.read_csv(out/'comparaciones_semillas.csv')
    horizons=pd.read_csv(out/'DM_por_horizonte_texto.csv')
    text_pairs=horizons[['A','B']].drop_duplicates()
    stop=json.loads((out/'precision_parada_protocolo.json').read_text(encoding='utf-8'))
    for cell in cells:
        if cell.cell_type=='markdown':
            if cell.source.startswith('## Comparaciones temporales de todos los procedimientos'):
                cell.source=f'''## Comparaciones temporales de todos los procedimientos

La matriz incluye las {len(temporal)} parejas de los {p['n_modelos']} procedimientos y las dos referencias. Cada celda muestra p de Diebold–Mariano con corrección HLN y Holm({len(temporal)}), calculado sobre diferenciales diarios con HAC de {p['lag_HAC']} rezagos. Las diferencias e intervalos temporales completos se conservan en la tabla descargable.'''
            elif cell.source.startswith('DM–HLN con Holm rechaza'):
                cell.source=re.sub(r'DM–HLN con Holm rechaza en \d+ de las \d+ parejas\.',
                    f'DM–HLN con Holm rechaza en {int(temporal.p_DM_Holm.lt(.05).sum())} de las {len(temporal)} parejas.',cell.source)
            elif cell.source.startswith('## Comparaciones entre semillas y tamaño del efecto'):
                cell.source=f'''## Comparaciones entre semillas y tamaño del efecto

Se incluyen las {len(seeds)} parejas de los {p['n_modelos']} procedimientos, emparejadas por índice de semilla 1000–1004. Se reportan diferencia media de MASE, Shapiro–Wilk, t pareada y Wilcoxon exacto con Holm({len(seeds)}), y Cohen d_z con IC95 % por inversión de t no central. La t es un análisis de sensibilidad; cinco pares no acreditan normalidad porque Shapiro no rechace.'''
            cell.source=cell.source.replace('Los dos grupos de paneles separan las curvas','Los tres grupos de paneles separan las curvas')
            cell.source=re.sub(r'Levene centrado en mediana rechaza igualdad de varianzas \(p≈[0-9.,]+\)',
                f"Levene centrado en mediana rechaza igualdad de varianzas (p≈{p['p_Levene']:.6f})",cell.source)
            cell.source=cell.source.replace('Se conservan 72 contrastes DM–HLN de los dos aportes textuales',
                f'Se conservan {len(horizons)} contrastes DM–HLN de los {len(text_pairs)} aportes textuales')
            cell.source=cell.source.replace('[Quince diferencias y dimensionamiento]',f"[{stop['parejas']} diferencias y dimensionamiento de parada]")
            cell.source=cell.source.replace('La planificación conservadora alcanza máximos de 82 semillas en los pilotos externos y el máximo indicado en la tabla de parada interna.',
                f"La planificación conservadora alcanza máximos de 82 semillas en los pilotos externos y {stop['R_max_Bonferroni']} en la reserva interna.")
        elif cell.cell_type=='code' and 'diagnostico_residuos.csv' in cell.source:
            cell.source=cell.source.replace("'{:.3g}'", "(lambda p: '<1e-300' if p==0 else f'{p:.3g}')")
    from diagnosticos_ampliados_jbook import integrar
    return integrar(root,cells)


def preparar_corpus_modelos(root):
    root=Path(root);out=root/'results/revision_integral';out.mkdir(exist_ok=True)
    dest=root/'results/evaluacion_e2/entradas/texto';dest.mkdir(parents=True,exist_ok=True)
    target=dest/'corpus_oficial.parquet'
    if not target.exists():
        with zipfile.ZipFile('C:/Users/Work/Downloads/Modelo7_TGForecaster.zip') as z:
            target.write_bytes(z.read('Modelo7_TGForecaster/data/processed/corpus_oficial.parquet'))
    assert hashlib.sha256(target.read_bytes()).hexdigest()=='3bda4dc2bd80c07e9800053030a335d0dc7202e9a4e76dacae39abd61f3d3c89'
    source=root.parent/'Consolidado-Entrenamientos-20261115/fuentes/adjuntos/modelo7_20261114_0430/resultados_modelo7/texto'
    for a,b in [('indice.parquet','indice_encoder.parquet'),('ficha.json','ficha_encoder.json'),('cobertura_por_dia.csv','cobertura_original.csv')]:
        if (source/a).exists():shutil.copy2(source/a,dest/b)
    corpus=pd.read_parquet(target);index=pd.read_parquet(dest/'indice_encoder.parquet')
    assert corpus.id.tolist()==index.id.tolist() and len(corpus)==14162 and corpus.id.is_unique
    hashes=[]
    for row in corpus.itertuples():
        body=row.texto.strip()
        content=row.titulo.strip()+('\n'+body if body else '')
        hashes.append(hashlib.sha256(content.encode('utf-8')).hexdigest())
    assert hashes==index.sha_contenido.tolist()
    assert set(corpus.tipo_fuente)=={'official_report'}
    old_ids=pd.read_parquet(root/'data/processed/corpus_texto.parquet',columns=['id'])
    assert corpus.id.isin(old_ids.id).all()
    fich=json.loads((dest/'ficha_encoder.json').read_text(encoding='utf-8'))
    capacity=fich['capacidad_tokens_por_fragmento'];overlap=64
    covered=np.minimum(index.tokens,capacity+(index.fragmentos_usados-1)*(capacity-overlap))
    facts=dict(registros=len(corpus),contenidos_distintos=index.sha_contenido.nunique(),
        solo_titular=int(corpus.texto.str.strip().eq('').sum()),truncados=int((index.fragmentos_totales>index.fragmentos_usados).sum()),
        descarte_tokens_reportado_pct=fich['porcentaje_tokens_descartados'],
        descarte_tokens_sin_doble_conteo_pct=float(100*(1-covered.sum()/index.tokens.sum())),
        inicio=str(corpus.disponible_desde.min()),fin=str(corpus.disponible_desde.max()))
    (out/'corpus_modelos_auditoria.json').write_text(json.dumps(facts,ensure_ascii=False,indent=2),encoding='utf-8')
    return facts


def celdas_corpus_modelos(root):
    facts=preparar_corpus_modelos(root);md=nb.v4.new_markdown_cell;code=nb.v4.new_code_cell
    return [md('''## Corpus oficial utilizado en los modelos

Las variantes con documentos utilizan un caché E5 de 14.162 registros oficiales, con identificadores únicos y contenidos procedentes del inventario integrado. Incluye boletines IDEAM que solo tienen titular, además de documentos con cuerpo; por ello su tamaño no equivale al de las 7.907 versiones documentales de la ampliación. El EDA léxico de prensa que sigue caracteriza otro conjunto de entradas.

El encoder conserva la revisión fijada de multilingual-E5-base. Cada fragmento se representa mediante promedio enmascarado y norma L2; los fragmentos admitidos se promedian y normalizan para obtener un vector por documento. La relevancia entre documentos se aprende dentro de la red. Las variantes ejecutadas son N y N+O; la incorporación de prensa permanece pendiente. TGForecaster conserva descripciones estáticas de canales en ambas variantes.'''),code('''TEXTO_MODELOS=ROOT/'results/evaluacion_e2/entradas/texto'
corpus_m=pd.read_parquet(TEXTO_MODELOS/'corpus_oficial.parquet')
indice_m=pd.read_parquet(TEXTO_MODELOS/'indice_encoder.parquet')
assert len(corpus_m)==14162 and corpus_m.id.is_unique
assert corpus_m.id.tolist()==indice_m.id.tolist()
datos_m=corpus_m.merge(indice_m.drop(columns='fuente'),on='id',validate='one_to_one')
datos_m['solo_titular']=datos_m.texto.str.strip().eq('')
datos_m['truncado']=datos_m.fragmentos_totales>datos_m.fragmentos_usados
fecha_limite=pd.Timestamp('2026-01-01',tz='America/Bogota')
datos_m['disponible_hasta_2025']=pd.to_datetime(datos_m.disponible_desde,utc=True)<fecha_limite
resumen_m=datos_m.groupby('fuente').agg(Registros=('id','size'),
    Disponibles_hasta_2025=('disponible_hasta_2025','sum'),Solo_titular=('solo_titular','sum'),
    Tokens_mediana=('tokens','median'),Truncados=('truncado','sum'))
display(resumen_m)
'''),md(f'''El caché contiene {facts['solo_titular']:,} registros sin cuerpo y {facts['contenidos_distintos']:,} contenidos distintos por SHA-256. La concentración de titulares cortos de IDEAM y las diferencias de longitud entre fuentes limitan una interpretación homogénea del texto. Los registros posteriores al horizonte histórico no se habilitan antes de su fecha de disponibilidad.

Se limitan a ocho fragmentos {facts['truncados']:,} documentos. El indicador del paquete reporta {facts['descarte_tokens_reportado_pct']:.2f} % de tokens descartados al sumar las capacidades de los fragmentos; esa suma cuenta repetidamente el solapamiento. Al contar posiciones originales cubiertas, el descarte es {facts['descarte_tokens_sin_doble_conteo_pct']:.2f} %. La diferencia describe la convención de conteo y no modifica los embeddings conservados.'''),code('''# Cobertura por origen: ventana inclusiva de 72 h y 64 documentos más recientes.
fechas_disponibles=pd.DatetimeIndex(pd.to_datetime(corpus_m.disponible_desde,utc=True)).as_unit('ns').asi8
assert np.all(np.diff(fechas_disponibles)>=0)
origenes_m=pd.date_range('2000-01-09','2025-12-31',freq='D',tz='America/Bogota')
instantes_m=origenes_m.as_unit('ns').asi8
hi_m=np.searchsorted(fechas_disponibles,instantes_m,side='right')
lo_m=np.searchsorted(fechas_disponibles,instantes_m-pd.Timedelta(hours=72).value,side='left')
total_m=hi_m-lo_m;usados_m=np.minimum(total_m,64)
cobertura_m=pd.DataFrame({'origen':origenes_m,'total':total_m,'usados':usados_m})
anual_m=cobertura_m.groupby(cobertura_m.origen.dt.year).agg(
    cobertura=('total',lambda x:100*x.gt(0).mean()),media_usados=('usados','mean'))
original_m=pd.read_csv(TEXTO_MODELOS/'cobertura_original.csv').set_index('dia')
seleccion_m=original_m.reindex(origenes_m.strftime('%Y-%m-%d'))
assert np.array_equal(total_m,seleccion_m.documentos_en_ventana)
assert np.array_equal(usados_m,seleccion_m.documentos_usados)
fig,axes=plt.subplots(1,2,figsize=(13,4),layout='constrained')
axes[0].plot(anual_m.index,anual_m.cobertura,marker='o')
axes[0].set(xlabel='Año',ylabel='Orígenes con documentos (%)',ylim=(0,105))
axes[1].plot(anual_m.index,anual_m.media_usados,marker='o',color='#287f83')
axes[1].set(xlabel='Año',ylabel='Documentos admitidos por origen · media',ylim=(0,65))
for ax in axes:ax.axvline(2022,color='grey',ls='--',lw=1)
display(fig);plt.close(fig)
'''),md('''La disponibilidad de texto cambia entre años incluso con una regla de entrada constante. La línea de 2022 señala el inicio de prueba; las curvas describen cobertura y volumen de las listas diarias, sin medir utilidad predictiva. TaTS consulta cierres diarios de su historia y los otros mecanismos documentales consultan el origen. El límite de 64 documentos y los días sin texto forman parte del procedimiento; no se eliminan orígenes para mejorar sus métricas.''')]


def actualizar_capitulos(root):
    root=Path(root)
    from reorganizar_capitulo10 import ejecutar
    p=root/'book/04_estacionalidad.ipynb';n=nb.read(p,4)
    for c in n.cells:
        if c.cell_type=='code' and 'from statsmodels.tsa.seasonal import STL' in c.source:
            c.source='''from statsmodels.tsa.seasonal import STL
diaria=gwh.resample('D').mean()
stl=STL(diaria,period=7,robust=True).fit()
fig=stl.plot();fig.set_size_inches(11,7)
fig.axes[0].set_title('STL de la demanda media diaria')
for ax,nombre in zip(fig.axes,['Observada','Tendencia','Semanal','Residuo']):
    ax.set_ylabel(nombre+'\\n(GWh por intervalo)')
fig.tight_layout();display(fig);plt.close(fig)
amplitud=float(stl.seasonal.quantile(.95)-stl.seasonal.quantile(.05))
residuo=float(stl.resid.std())
'''
    ejecutar(n.cells);nb.write(n,p)
    p=root/'book/06_covariables.ipynb';n=nb.read(p,4)
    for c in n.cells:
        if c.cell_type=='code' and "label=ciudad)" in c.source:
            c.source=c.source.replace('label=ciudad)',"label={'bogota':'Bogotá','medellin':'Medellín','cucuta':'Cúcuta'}.get(ciudad,ciudad.title()))")
            c.source=c.source.replace('Precipitación (mm/h)','Precipitación (mm en la hora previa)')
        if c.cell_type=='markdown' and c.source.startswith('Se caracteriza el panel ampliado') and 'selección de asociaciones' not in c.source:
            c.source+=' Las figuras muestran una selección de asociaciones y agregados urbanos; no equivalen a importancia predictiva. El diccionario conserva las variables individuales y el diseño especifica los rezagos efectivos del entrenamiento.'
    ejecutar(n.cells);nb.write(n,p)
    p=root/'book/09_validacion.ipynb';n=nb.read(p,4)
    for c in n.cells:
        if c.cell_type!='markdown':continue
        c.source=c.source.replace('los seis procedimientos ampliados restantes','los ocho procedimientos ampliados restantes')
        c.source=c.source.replace('La configuración base utiliza demanda y calendario; clima, hidrología y titulares fundamentan extensiones posteriores.',
            'Las cuatro familias base utilizan demanda y calendario. Las arquitecturas ampliadas incorporan precio, hidrología y clima histórico; TaTS, fusión tardía y TGForecaster contrastan además documentos oficiales.')
        c.source=c.source.replace('Los resultados del capítulo de modelos corresponden a sus entradas efectivamente entrenadas, demanda y calendario.',
            'Los resultados se atribuyen a las entradas efectivamente utilizadas: demanda/calendario en las cuatro familias base y conjunto numérico ampliado, con o sin documentos, en los demás procedimientos.')
    nb.write(n,p)
    p=root/'book/07_texto.ipynb';n=nb.read(p,4)
    for c in n.cells:
        if c.cell_type=='code' and c.source.startswith('coberturas={}'): 
            c.source='''coberturas={}
for escenario in ['declarado','estricto']:
    c=pd.read_csv(ROOT/f'data/processed/cobertura_titulares_{escenario}.csv',index_col=0)
    c.index=c.index.astype(int);c=c.loc[c.index<2022]
    suma_antes=int(c.sum().sum())
    c=c.reindex(range(2000,2022),fill_value=0)
    assert int(c.sum().sum())==suma_antes
    coberturas[escenario]=c
fig,axes=plt.subplots(1,2,figsize=(12,4))
for ax,(escenario,c) in zip(axes,coberturas.items()):
    c.plot(ax=ax,marker='.',markersize=3)
    ax.set(title=escenario,xlabel='Año de disponibilidad',ylabel='Titulares',xlim=(2000,2021),xticks=[2000,2005,2010,2015,2021])
    ax.legend(fontsize=6)
fig.tight_layout();display(fig);plt.close(fig)
total_declarado=int(coberturas['declarado'].sum().sum())
total_estricto=int(coberturas['estricto'].sum().sum())
'''
        if c.cell_type!='markdown':continue
        c.source=c.source.replace('Se compararán tres variantes','El diseño contempla tres variantes')
        c.source=c.source.replace('una ponderación aprendida agrega los fragmentos disponibles','una ponderación aprendida agrega los documentos disponibles')
        c.source=c.source.replace('contiene únicamente fragmentos de documentos disponibles al origen','contiene únicamente documentos disponibles al origen')
        c.source=c.source.replace('la relevancia se aprenderá entre los fragmentos admitidos','los modelos ejecutados aprenden relevancia entre los vectores documentales obtenidos de los fragmentos admitidos')
        c.source=c.source.replace('| Sin texto | Historia numérica y calendario | Referencia numérica |',
            '| Sin documentos temporales | Demanda, precio, hidrología, clima y calendario | Referencia numérica; TGForecaster conserva descripciones estáticas |')
        c.source=c.source.replace('Se mantienen semillas, métrica primaria MASE, margen relativo del 3 % y presupuesto de búsqueda comparables.',
            'El diseño conserva MASE y margen relativo del 3 %; los presupuestos efectivamente ejecutados se detallan en el capítulo de modelos y no son iguales entre todos los procedimientos.')
        c.source=c.source.replace('Esta ventana descriptiva no fija el hiperparámetro del futuro modelo.','Las ventanas predictivas y sus límites efectivos se documentan en las fichas de los modelos.')
        c.source=c.source.replace('un futuro entrenamiento textual deberá ajustar su representación dentro del tramo permitido de cada pliegue.',
            'las representaciones predictivas usan E5 congelado y relevancia ajustada en el tramo de entrenamiento de cada pliegue.')
        c.source=c.source.replace('La comparación principal será sin texto frente a texto oficial.','La comparación principal contrasta N frente a N+O; la variante con prensa no cuenta con resultados.')
    if not any(c.cell_type=='markdown' and c.source.startswith('## Corpus oficial utilizado en los modelos') for c in n.cells):
        inicio=next(i for i,c in enumerate(n.cells) if c.cell_type=='code')+1
        n.cells[inicio:inicio]=celdas_corpus_modelos(root)
    else:
        a=next(i for i,c in enumerate(n.cells) if c.cell_type=='markdown' and c.source.startswith('## Corpus oficial utilizado en los modelos'))
        n.cells[a:a+5]=celdas_corpus_modelos(root)
    ejecutar(n.cells);nb.write(n,p)
    from diagnosticos_ampliados_jbook import armonizar
    armonizar(root)


def verificar_contratos(root):
    root=Path(root);o=root/'results/evaluacion_e2'
    b=pd.read_csv(o/'benchmark.csv').set_index('clave');modelos=b.index[b.semillas.eq(5)].tolist();allkeys=b.index.tolist()
    p=json.loads((o/'protocolo.json').read_text(encoding='utf-8'))
    assert p['n_modelos']==len(modelos) and len(b)==len(modelos)+2
    with np.load(o/'perdidas_y_bootstrap.npz',allow_pickle=False) as z:
        assert z['loss'].shape==(1461,len(b)) and z['semillas'].shape==(5,len(modelos))
        assert z['nombres'].tolist()==allkeys
        assert np.allclose(z['loss'].mean(axis=0),b.MASE)
        assert np.allclose(z['semillas'].mean(axis=0),b.MASE.reindex(modelos))
        f=stats.friedmanchisquare(*z['semillas'].T)
        assert np.isclose(f.statistic,p['friedman']) and np.isclose(f.pvalue,p['p_Friedman'],rtol=1e-8)
    assert np.allclose(b.MAE,b.MASE*p['q168'])
    assert (b.IC95_inf<=b.MASE).all() and (b.MASE<=b.IC95_sup).all()
    for name,keys,ncols in [('comparaciones_temporales.csv',allkeys,['p_DM','p_bootstrap','p_TOST']),
                            ('comparaciones_semillas.csv',modelos,['p_t','p_Wilcoxon']),('nemenyi.csv',modelos,[])]:
        d=pd.read_csv(o/name);expected={frozenset(x) for x in combinations(keys,2)}
        assert len(d)==len(expected) and {frozenset(x) for x in zip(d.A,d.B)}==expected
        for col in ncols:assert np.allclose(d[col+'_Holm'],multipletests(d[col],method='holm')[1])
    txt=pd.read_csv(o/'contrastes_texto.csv');hor=pd.read_csv(o/'DM_por_horizonte_texto.csv')
    assert len(txt)==p['n_contrastes_textuales'] and len(hor)==24*len(txt)
    for col in ['p_DM','p_bootstrap','p_TOST']:assert np.allclose(txt[col+'_Holm_texto'],multipletests(txt[col],method='holm')[1])
    assert np.allclose(hor.p_Holm,multipletests(hor.p,method='holm')[1])
    mcs=pd.read_csv(o/'mcs.csv');assert len(mcs)==len(b) and set(mcs.clave)==set(allkeys)
    meta=pd.read_csv(o/'metadatos_procedimientos.csv').set_index(['clave','semilla'])
    curves=pd.read_csv(o/'curvas_aprendizaje.csv')
    assert len(curves.groupby(['clave','semilla']))==5*len(modelos)
    for (k,s),d in curves.groupby(['clave','semilla']):
        assert d.epoca.tolist()==list(range(1,len(d)+1))
        assert len(d)==meta.loc[(k,s),'epocas']
        assert int(d.loc[d.val_loss.idxmin(),'epoca'])==meta.loc[(k,s),'mejor_epoca']
    prec=pd.read_csv(o/'precision_desarrollo_resumen.csv').set_index('clave');assert set(prec.index)==set(modelos)
    assert prec.Semillas.eq(5).all() and prec[['MASE_media','DE','CV_pct','IC95_inf','IC95_sup']].notna().all().all()
    gen=pd.read_csv(o/'generalizacion_integrada.csv').set_index('clave')
    assert np.allclose(gen.Prueba,b.MASE.reindex(gen.index)) and np.allclose(gen['Precisión desarrollo'],prec.MASE_media.reindex(gen.index))
    metrics=pd.read_csv(o/'resultados_horizonte.csv');assert metrics.groupby('clave').size().eq(24).all()
    assert set(metrics.clave)==set(modelos)
    assert np.allclose(metrics.groupby('clave').MASE.mean(),b.MASE.reindex(metrics.groupby('clave').MASE.mean().index))
    return dict(procedimientos=len(modelos),realizaciones=len(meta),referencias=2,parejas_temporales=len(pd.read_csv(o/'comparaciones_temporales.csv')),
        parejas_semillas=len(pd.read_csv(o/'comparaciones_semillas.csv')),contrastes_texto=len(txt),horizontes_texto=len(hor),
        tablas_recalculadas_desde_artefactos=True,entrenamiento=False,publicacion=False)


def inventariar(root):
    root=Path(root);rows=[]
    for path in sorted((root/'book').glob('*.ipynb')):
        n=nb.read(path,4);heading=''
        for i,c in enumerate(n.cells):
            if c.cell_type=='markdown':
                hs=[x for x in c.source.splitlines() if x.startswith('#')]
                if hs:heading=hs[-1]
                # Las tablas Markdown son contenido nativo, fuera de bloques de código.
                fence=False
                for j,line in enumerate(c.source.splitlines()):
                    if line.startswith('```'):fence=not fence
                    if not fence and re.match(r'^\|[ :|-]+\|\s*$',line):rows.append(dict(archivo=path.name,celda=i,tipo='tabla_markdown',seccion=heading))
            else:
                assert c.execution_count is not None
                for out in c.outputs:
                    assert out.output_type!='error'
                    if out.output_type=='stream' and out.get('name')=='stderr':raise AssertionError((path.name,i,out.text))
                    if 'image/png' in out.get('data',{}):
                        assert i+1<len(n.cells) and n.cells[i+1].cell_type=='markdown' and len(n.cells[i+1].source)>35
                        rows.append(dict(archivo=path.name,celda=i,tipo='figura',seccion=heading))
                    html=out.get('data',{}).get('text/html','')
                    for _ in range(html.count('<table')):rows.append(dict(archivo=path.name,celda=i,tipo='tabla_calculada',seccion=heading))
    for path in [*(root/'book').glob('*.md'),*(root/'book/arquitecturas').glob('*.md')]:
        lines=path.read_text(encoding='utf-8').splitlines();fence=False;heading=''
        for i,line in enumerate(lines):
            if line.startswith('```'):fence=not fence
            if fence:continue
            if line.startswith('#'):heading=line
            if re.match(r'^\|[ :|-]+\|\s*$',line):rows.append(dict(archivo=path.relative_to(root/'book').as_posix(),celda=i,tipo='tabla_markdown',seccion=heading))
            if re.search(r'!\[.*?\]\([^)]*\)',line):rows.append(dict(archivo=path.relative_to(root/'book').as_posix(),celda=i,tipo='esquema',seccion=heading))
    return pd.DataFrame(rows)


def auditar(root):
    root=Path(root);out=root/'results/revision_integral';out.mkdir(exist_ok=True)
    checks=verificar_contratos(root);inventory=inventariar(root)
    inventory.to_csv(out/'inventario_tablas_graficas.csv',index=False)
    counts=inventory.tipo.value_counts().to_dict();checks['inventario']=counts
    (out/'auditoria.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(checks,ensure_ascii=False))
    return checks


if __name__=='__main__':auditar(Path(__file__).resolve().parents[1])

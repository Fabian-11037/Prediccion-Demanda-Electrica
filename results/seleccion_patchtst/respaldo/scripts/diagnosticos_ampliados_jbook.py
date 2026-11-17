"""Presentación de diagnósticos calculados con checkpoints congelados."""
from pathlib import Path
import json,shutil
import nbformat as nb
import pandas as pd

def listo(root):return (Path(root)/'results/diagnosticos_ampliados/latencia_resumen.csv').is_file()

def integrar(root,cells):
    root=Path(root)
    if not listo(root):return cells
    out=root/'results/diagnosticos_ampliados';md=nb.v4.new_markdown_cell;code=nb.v4.new_code_cell
    names=pd.read_csv(root/'results/evaluacion_e2/benchmark.csv').set_index('clave').modelo.to_dict()
    imp=pd.read_csv(out/'importancia_resumen.csv');sens=pd.read_csv(out/'sensibilidad_resumen.csv')
    check=pd.read_csv(out/'reproduccion_semillas.csv')
    top=imp.loc[imp.cambio_pct.idxmax()]
    ex=imp[~imp.grupo.isin(['perfil_demanda','nivel_ventana','calendario'])]
    extop=ex.loc[ex.cambio_pct.idxmax()]
    setup=code('''d_root=ROOT/'results/diagnosticos_ampliados'
d_imp=pd.read_csv(d_root/'importancia_resumen.csv')
d_sens=pd.read_csv(d_root/'sensibilidad_resumen.csv')
d_lat=pd.read_csv(d_root/'latencia_resumen.csv').set_index('clave')
d_names=pd.read_csv(ROOT/'results/evaluacion_e2/benchmark.csv').set_index('clave').modelo.to_dict()
''')
    labels={'perfil_demanda':'Perfil de demanda','nivel_ventana':'Nivel de ventana','precio':'Precio',
        'reservas':'Reservas hídricas','aportes':'Aportes hídricos','temperatura':'Temperatura',
        'humedad':'Humedad','precipitacion':'Precipitación','calendario':'Calendario'}
    intro=md(f'''## Importancia de variables en los modelos ampliados

La importancia se estima manteniendo pesos y escaladores: se traslada circularmente un grupo de entradas entre orígenes del mismo año, con una separación mínima de 28 días. Se conservan las 168 horas y el orden interno de cada historia; la etiqueta permanece en su fecha. Los nueve grupos cubren demanda, nivel de restitución, precio, reservas y aportes, temperatura, humedad, precipitación y calendario. Las magnitudes hídricas correlacionadas se agrupan como reservas; cada variable climática reúne las ocho ciudades. El calendario incluye pasado y horizonte, pero la hora del día es invariante entre estos orígenes diarios y su importancia no queda identificada.

Se utilizan las cinco inicializaciones de cada uno de los diez procedimientos y 192 fechas: días 1, 8, 15 y 22 de todos los meses de 2022–2025. Los donantes se toman de los 1.461 orígenes del mismo periodo. Se evalúan tres desplazamientos reproducibles y se promedian primero las cinco semillas. Esta muestra fija no se elige por error ni sustituye el benchmark de prueba completa.

Antes de perturbar se reconstruyeron las 50 predicciones basales sobre los 1.461 días. La diferencia absoluta máxima respecto de los archivos conservados fue {check.max_diferencia_kwh.max():.2f} kWh por pronóstico y el mayor cambio relativo del MAE fue {100*check.cambio_relativo_MAE.max():.4f} %. Son comprobaciones de reproducción entre motores CPU/GPU; los resultados de prueba publicados no se reemplazan. Las comparaciones de sensibilidad utilizan siempre el basal CPU del mismo checkpoint.''')
    plot=code('''from matplotlib.colors import SymLogNorm
d_groups=['perfil_demanda','nivel_ventana','precio','reservas','aportes','temperatura','humedad','precipitacion','calendario']
d_labels=['Perfil de demanda','Nivel de ventana','Precio','Reservas hídricas','Aportes hídricos','Temperatura','Humedad','Precipitación','Calendario']
d_keys=[k for k in g_keys if k in set(d_imp.clave)]
d_matrix=d_imp.pivot(index='clave',columns='grupo',values='cambio_pct').reindex(index=d_keys,columns=d_groups)
d_lim=max(abs(d_matrix.to_numpy()).max(),1.)
fig,ax=plt.subplots(figsize=(15,8),layout='constrained')
d_im=ax.imshow(d_matrix,cmap='RdBu_r',norm=SymLogNorm(linthresh=1,vmin=-d_lim,vmax=d_lim))
ax.set(xticks=range(len(d_groups)),yticks=range(len(d_keys)),xticklabels=d_labels,yticklabels=[d_names[k] for k in d_keys])
ax.tick_params(axis='x',rotation=35,labelsize=9);ax.tick_params(axis='y',labelsize=9);ax.grid(False)
for i in range(len(d_keys)):
    for j in range(len(d_groups)):
        v=d_matrix.iloc[i,j];ax.text(j,i,f'{v:.2g}',ha='center',va='center',fontsize=8,color='white' if abs(v)>d_lim/5 else '#151515')
fig.colorbar(d_im,ax=ax,label='Cambio del MAE (%) · escala simétrica logarítmica fuera de ±1 %')
display(fig);plt.close(fig)
''')
    interpretation=md(f'''El mayor deterioro aparece al perturbar {labels[top.grupo].lower()} en {names[top.clave]} ({top.cambio_pct:.2f} %). Entre las covariables externas, el mayor incremento corresponde a {labels[extop.grupo].lower()} en {names[extop.clave]} ({extop.cambio_pct:.2f} %). Las celdas negativas indican que la perturbación redujo el error en estas fechas; las próximas a cero describen poca sensibilidad bajo este mecanismo, sin demostrar irrelevancia intrínseca. El nivel participa también en la restitución a kWh y su efecto no es atribuible solo a una entrada explícita de la red.

La permutación rompe combinaciones observadas de calendario, demanda y exógenas; es una dependencia condicional del modelo, no un efecto causal ni el resultado de retirar variables y reentrenar. En los canales textuales también puede cambiar la consulta de relevancia al cambiar la historia. Los CSV conservan los tres desplazamientos y su p10–p90 después de promediar semillas; ese rango no es un intervalo de confianza temporal. Los diagnósticos de las cuatro familias base utilizan 1.461 fechas y diez desplazamientos, por lo que sus magnitudes deben leerse con ese alcance distinto.''')
    dynamic=md('''## Sensibilidad al contenido de los documentos oficiales

Se mantienen la historia numérica, el calendario, los pesos y las listas documentales de cada origen. Se contrastan tres intervenciones: eliminar documentos y su presencia; anular únicamente sus vectores manteniendo la presencia; y sustituir el contenido por documentos de la misma fuente con disponibilidad declarada anterior a 2022, con tres remuestreos reproducibles con reemplazo. Este último control conserva volumen y fuente, pero rompe la correspondencia entre el contenido y el acontecimiento reciente. No es una permutación biyectiva ni introduce documentos con fecha declarada posterior al origen. El control a cero retiene los sesgos aprendidos y distingue contenido de presencia.

El análisis incluye los 1.461 días y las cinco semillas de TaTS, fusión tardía y TGForecaster con documentos. La variación del error es respecto del mismo procedimiento con sus documentos originales; no representa la comparación N frente a N+O entrenados por separado. Los IC95 % se obtienen mediante 10.000 remuestras circulares pareadas de 14 días, promediando antes semillas y repeticiones del placebo.''')
    dplot=code('''d_conditions=['sin_documentos','contenido_cero','contenido_historico']
d_clabels=['Sin documentos','Contenido cero','Contenido histórico']
d_text=d_sens[d_sens.condicion.isin(d_conditions)]
d_textkeys=[k for k in g_keys if k in set(d_text.clave)]
fig,axes=plt.subplots(1,len(d_textkeys),figsize=(16,5),layout='constrained')
for ax,k in zip(np.atleast_1d(axes),d_textkeys):
    s=d_text[d_text.clave.eq(k)].set_index('condicion').reindex(d_conditions)
    ax.errorbar(s.cambio_pct,np.arange(3),xerr=np.vstack([s.cambio_pct-s.IC95_inf,s.IC95_sup-s.cambio_pct]),fmt='o',capsize=4,color='#258a83')
    ax.axvline(0,color='#777',linestyle='--',linewidth=1);ax.set(yticks=range(3),yticklabels=d_clabels,xlabel='Cambio del MAE (%) e IC95 % temporal',title=d_names[k])
    ax.tick_params(axis='y',labelsize=9)
display(fig);plt.close(fig)
''')
    dlines=[]
    for k,g in sens[sens.condicion.isin(['sin_documentos','contenido_historico'])].groupby('clave'):
        s=g.set_index('condicion');r=s.loc['sin_documentos'];h=s.loc['contenido_historico']
        dlines.append(f"En {names[k]}, retirar documentos cambia el MAE {r.cambio_pct:+.2f} % (IC95 % [{r.IC95_inf:+.2f}; {r.IC95_sup:+.2f}]) y sustituir su contenido por texto histórico lo cambia {h.cambio_pct:+.2f} % (IC95 % [{h.IC95_inf:+.2f}; {h.IC95_sup:+.2f}]). La sustitución modifica el pronóstico en promedio {h.divergencia_kwh:.1f} kWh por intervalo.")
    din=md(' '.join(dlines)+''' La comparación con contenido cero completa la separación entre señal semántica y activación de la rama. Un cambio pequeño de pronóstico con texto sustituido sugiere poca sensibilidad al contenido bajo este control; un cambio grande demuestra dependencia, pero no garantiza que esa dependencia sea útil. La dependencia de una rama puede coexistir con un error mayor que el control N ajustado por separado. Los intervalos son puntuales y exploratorios, condicionados a las inicializaciones disponibles; no se emplean para seleccionar arquitecturas ni para declarar equivalencia dentro del margen del estudio.''')
    ta=sens[sens.clave.str.contains('modelo6')&sens.condicion.eq('contenido_historico')].iloc[0]
    tg=sens[sens.clave.str.contains('modelo7')&sens.condicion.eq('contenido_historico')].iloc[0]
    din.source+=f'''\n\nEn TaTS, el reemplazo histórico cambia el MAE apenas {ta.cambio_pct:+.3f} % y desplaza el pronóstico medio {ta.divergencia_kwh:.1f} kWh. La sensibilidad a anular el canal no implica, por tanto, una utilización comparable de la información específica del día: gran parte de su efecto se conserva con contenido antiguo de la misma fuente. En TGForecaster con documentos, el contenido histórico reduce descriptivamente el error ({tg.cambio_pct:+.2f} %) frente al original. El resultado cuestiona la utilidad de su alineación reciente bajo este control, sin convertir el placebo en una nueva configuración seleccionada.'''
    static=md('''### Descripciones estáticas de los canales en TGForecaster

Las dos variantes TGForecaster usan 33 descripciones de canales. Se evalúan a cero y con tres permutaciones reproducibles de su asociación a los canales. Se altera su uso en la memoria y en las consultas del horizonte; la descripción del objetivo en el mecanismo de relevancia documental permanece fija. Se conserva el canal documental original en N+O. Es una intervención sobre la identificación estática de variables, distinta de retirar noticias recientes; puede producir entradas alejadas de las observadas durante el ajuste.''')
    splot=code('''d_stat=d_sens[d_sens.condicion.str.startswith('descripciones_')]
d_statkeys=[k for k in g_keys if k in set(d_stat.clave)]
fig,axes=plt.subplots(1,len(d_statkeys),figsize=(13,4),layout='constrained')
for ax,k in zip(np.atleast_1d(axes),d_statkeys):
    s=d_stat[d_stat.clave.eq(k)].set_index('condicion').reindex(['descripciones_cero','descripciones_permutadas'])
    ax.errorbar(s.cambio_pct,np.arange(2),xerr=np.vstack([s.cambio_pct-s.IC95_inf,s.IC95_sup-s.cambio_pct]),fmt='o',capsize=4,color='#447ba0')
    ax.axvline(0,color='#777',linestyle='--',linewidth=1)
    ax.set(yticks=range(2),yticklabels=['Descripciones cero','Descripciones permutadas'],xlabel='Cambio del MAE (%) e IC95 % temporal',title=d_names[k])
display(fig);plt.close(fig)
''')
    slines=[]
    for k,g in sens[sens.condicion.str.startswith('descripciones_')].groupby('clave'):
        s=g.set_index('condicion');a=s.loc['descripciones_cero'];b=s.loc['descripciones_permutadas']
        slines.append(f"En {names[k]}, anular las descripciones cambia el MAE {a.cambio_pct:+.2f} % y permutarlas {b.cambio_pct:+.2f} %.")
    sin=md(' '.join(slines)+''' La sensibilidad a estas descripciones no acredita por sí misma que los documentos variables en el tiempo mejoren el pronóstico. El canal estático y el documental tienen funciones distintas; sus efectos se presentan por separado.''')
    pipeline=md('''### Código de restauración y diagnóstico

Los siguientes archivos restauran las entradas y los checkpoints, verifican sus SHA-256 y calculan las perturbaciones, los controles textuales y las mediciones de latencia. Utilizan el entorno de inferencia CPU y los archivos de la consolidación. La construcción del libro lee los resultados conservados; no inicia la evaluación ni ajusta parámetros. El remuestreo numérico se define en `indices_donantes`, con semilla 11037 más el índice de repetición y desplazamientos circulares por año entre 28 y n−28 días.

Restauración de entradas y pesos:

```python
'''+(root/'scripts/inferencia_ampliada.py').read_text(encoding='utf-8')+'''
```

Cálculo de los diagnósticos:

```python
'''+(root/'scripts/diagnosticos_ampliados.py').read_text(encoding='utf-8')+'''
```

El archivo `protocolo.json` conserva fechas, grupos, semillas, donantes y tolerancias. Las mediciones de latencia se ejecutan después de terminar los procesos de perturbación, con un hilo, batch uno, 20 calentamientos y 100 repeticiones en cada uno de tres orígenes comunes. No incluyen carga de pesos, preparación, tokenización ni el encoder E5; los vectores ya están en memoria.''')
    cells[:]=[c for c in cells if not (c.cell_type=='markdown' and c.source.startswith('### Código de restauración y diagnóstico'))]
    insert=[intro,setup,plot,interpretation,dynamic,dplot,din,static,splot,sin]
    # Idempotente en un libro ya integrado.
    begin=next((i for i,c in enumerate(cells) if c.cell_type=='markdown' and c.source.startswith('## Importancia de variables en los modelos ampliados')),None)
    end=next(i for i,c in enumerate(cells) if c.cell_type=='markdown' and c.source.startswith('## Coste de inferencia'))
    if begin is None:cells[end:end]=insert
    else:cells[begin:end]=insert
    cells.append(pipeline)
    for c in cells:
        if c.cell_type=='markdown':
            c.source=c.source.replace('La importancia por perturbación y la latencia CPU conservan el alcance de las cuatro familias medidas.',
                'La importancia agrupada abarca los catorce procedimientos; la latencia se mide para sus 70 realizaciones. Los modelos ampliados incorporan controles de contenido documental y descripciones estáticas.')
            if c.source.startswith('## Coste de inferencia'):
                c.source='''## Coste de inferencia medido y coste de ajuste registrado

Se presentan parámetros, duración del ajuste final y épocas para los catorce procedimientos. La duración de ajuste es la mediana de cinco semillas; los dispositivos y protocolos distintos limitan su comparación directa. La latencia CPU se vuelve a medir en el mismo ordenador para las 70 realizaciones: un hilo, batch uno y los mismos tres orígenes reales. Cada corrida tiene 20 calentamientos y 100 mediciones. Se informa la mediana de las 15 medianas (cinco semillas × tres orígenes) y la mediana de sus p95; esta última no es el p95 combinado de todas las llamadas.'''
            elif c.source.startswith('Parámetros y tiempo de ajuste caracterizan'):
                lat=pd.read_csv(out/'latencia_resumen.csv');low=lat.loc[lat.mediana_ms.idxmin()];high=lat.loc[lat.mediana_ms.idxmax()]
                c.source=f'''La menor latencia mediana corresponde a {names[low.clave]} ({low.mediana_ms:.3f} ms) y la mayor a {names[high.clave]} ({high.mediana_ms:.3f} ms). Los tres orígenes fijados incluyen distinta carga documental y todas las redes producen 24 salidas. El valor medido describe el forward en esta CPU, con entradas y vectores preparados; no es el tiempo de un servicio completo ni incluye la codificación de nuevos documentos. Las diferencias de hardware impiden trasladar directamente estas cifras a GPU u otro ordenador.'''
            elif c.source.startswith('Los catorce procedimientos completados están representados'):
                c.source='''Los catorce procedimientos tienen benchmark, curvas, observado–predicho, contrastes e importancia agrupada. La latencia se mide para las 70 realizaciones en condiciones comunes. La importancia ampliada utiliza 192 fechas y tres desplazamientos; la de las cuatro familias base, 1.461 fechas y diez desplazamientos. La sensibilidad documental cubre los 1.461 días de los tres procedimientos con documentos, y TGForecaster añade el diagnóstico de descripciones estáticas. Los pilotos independientes siguen limitados a seis procedimientos.'''
            elif c.source.startswith('## Tablas y reproducción'):
                c.source+='\n\n[Importancia ampliada](_static/resultados/diagnosticos_ampliados/importancia_resumen.csv) · [Sensibilidad al texto](_static/resultados/diagnosticos_ampliados/sensibilidad_resumen.csv) · [Latencia de 70 realizaciones](_static/resultados/diagnosticos_ampliados/latencia_semillas.csv) · [Reproducción CPU](_static/resultados/diagnosticos_ampliados/reproduccion_semillas.csv) · [Protocolo de diagnósticos](_static/resultados/diagnosticos_ampliados/protocolo.json).'
        elif c.cell_type=='code':
            if c.source.startswith('g_cost='):
                c.source='''g_cost=g_meta.groupby('clave').agg(Parametros=('parametros','first'),Ajuste_s_mediana=('segundos','median'),Epocas_mediana=('epocas','median'),Dispositivo=('dispositivo','first')).reindex(g_keys)
g_latency=pd.read_csv(ROOT/'results/diagnosticos_ampliados/latencia_resumen.csv').set_index('clave')
g_cost['Latencia_CPU_mediana_ms']=g_latency.mediana_ms.reindex(g_keys)
g_cost['Mediana_p95_por_corrida_ms']=g_latency.p95_ms.reindex(g_keys)
assert g_cost[['Latencia_CPU_mediana_ms','Mediana_p95_por_corrida_ms']].notna().all().all()
g_cost.index=[g_names[k] for k in g_cost.index]
display(g_cost.rename(columns={'Parametros':'Parámetros','Ajuste_s_mediana':'Ajuste final mediano (s)',
    'Epocas_mediana':'Épocas medianas','Latencia_CPU_mediana_ms':'Latencia CPU mediana (ms)',
    'Mediana_p95_por_corrida_ms':'p95 mediano (ms)'}).style.format(precision=3))
'''
            if "'Importancia y latencia CPU':" in c.source:
                c.source=c.source.replace("'Medidas' if g_k in modelos else 'Pendientes'","'Medidas'")
                if "g_coverage['Sensibilidad textual']" not in c.source:
                    c.source=c.source.replace('display(g_coverage)', '''g_sensitivity=[]
for g_k in g_keys:
    if 'modelo7' in g_k:
        g_sensitivity.append('Documentos y descripciones · 1461 días' if g_k.endswith('__N+O') else 'Descripciones · 1461 días')
    elif g_k.endswith('__N+O'):
        g_sensitivity.append('Documentos · 1461 días')
    else:
        g_sensitivity.append('No aplica · entradas numéricas')
g_coverage['Sensibilidad textual']=g_sensitivity
display(g_coverage)''')
    dest=root/'book/_static/resultados/diagnosticos_ampliados';dest.mkdir(exist_ok=True,parents=True)
    for f in out.iterdir():
        if f.is_file() and f.suffix in ['.csv','.json']:shutil.copy2(f,dest/f.name)
    for f in ['inferencia_ampliada.py','diagnosticos_ampliados.py']:
        shutil.copy2(root/'scripts'/f,root/'book/_static/codigo_e2'/f)
    return cells

def armonizar(root):
    root=Path(root)
    if not listo(root):return
    replacements={
        'La importancia por perturbación y la latencia CPU permanecen limitadas a las cuatro familias de demanda/calendario.':
        'La importancia agrupada incluye los catorce procedimientos y la latencia CPU se mide para sus 70 realizaciones; los modelos ampliados añaden controles de sensibilidad al contenido documental y a las descripciones estáticas.',
        'La importancia por perturbación y la latencia CPU siguen disponibles solo para las cuatro familias medidas; no se inventan para las diez ampliadas.':
        'La importancia y la latencia se documentan para los catorce procedimientos, con los tamaños de muestra y las condiciones de cada diagnóstico.'}
    for p in (root/'book').glob('*.ipynb'):
        if p.name=='10_evaluacion.ipynb':continue
        n=nb.read(p,4);changed=False
        for c in n.cells:
            if c.cell_type!='markdown':continue
            for a,b in replacements.items():
                if a in c.source:c.source=c.source.replace(a,b);changed=True
        if p.name.startswith('12'):
            heading='## Reproducción de los diagnósticos ampliados'
            n.cells[:]=[c for c in n.cells if not (c.cell_type=='markdown' and c.source.startswith(heading))]
            n.cells.append(nb.v4.new_markdown_cell(heading+'''

La [evaluación integrada](10_evaluacion.ipynb) incluye importancia agrupada, latencia CPU y sensibilidad al contenido de los documentos y a las descripciones estáticas. Sus figuras se reconstruyen con los CSV de `results/diagnosticos_ampliados`, sin cargar redes ni iniciar una evaluación nueva.

Para repetir las mediciones se utiliza el entorno de inferencia con PyTorch CPU y la carpeta hermana `Consolidado-Entrenamientos-20261115`, cuyo índice identifica los checkpoints, escaladores y vectores congelados. La inferencia conserva los pesos y registra cada realización completada para reanudarla. La latencia debe medirse cuando hayan terminado los procesos de perturbación.

```bash
python scripts/diagnosticos_ampliados.py --fase inferir --workers 4
python scripts/diagnosticos_ampliados.py --fase latencia
```

Después, en el entorno del JBook, se reconstruye únicamente la presentación a partir de esos resultados:

```bash
python scripts/actualizar_diagnosticos_jbook.py
jupyter-book build book --warningiserror --keep-going
python scripts/verificar_jbook.py
```

El protocolo registra fechas, donantes, grupos, semillas, controles e incertidumbre. `reproduccion_semillas.csv` documenta la correspondencia numérica con las predicciones conservadas; `entorno.json` identifica el ordenador y las condiciones de latencia. El código de restauración y diagnóstico se presenta íntegro al final de la evaluación integrada.
'''))
            changed=True
        if changed:nb.write(n,p)

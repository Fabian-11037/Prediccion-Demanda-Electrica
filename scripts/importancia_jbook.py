"""Integra importancia agrupada y su pipeline en el informe documental."""

INTRO = r'''(importancia-variables)=
## Importancia de variables y bloques de rezagos

Se mide cuánto aumenta el error al perturbar un grupo de entradas, manteniendo los pesos y los escaladores. El análisis incluye los 1.461 orígenes diarios de 2022–2025, cuatro familias y cinco inicializaciones por familia. La demanda histórica y el calendario son las entradas de estos modelos; el clima, la hidrología y el texto no reciben puntuaciones porque no intervienen en sus pronósticos.

La [importancia por permutación](https://scikit-learn.org/stable/modules/permutation_importance.html) describe la dependencia de un modelo ajustado respecto de sus entradas. Aquí se adapta a ventanas temporales: se trasladan circularmente orígenes completos dentro de cada año, con diez desplazamientos reproducibles y una separación circular mínima de 28 días. Se conserva el orden de las horas dentro de cada bloque y se utilizan los mismos donantes para familias, semillas y grupos. Las etiquetas permanecen en sus fechas originales. Esta perturbación es un diagnóstico retrospectivo, no un pronóstico operativo ni una estimación causal.

La importancia se expresa como diferencia de MASE:

$$I_g=\frac{1}{50}\sum_{s=1}^{5}\sum_{r=1}^{10}
\frac{\operatorname{MAE}_{s,r,g}-\operatorname{MAE}_{s}}{q_{168}}.$$

Las figuras muestran el cambio porcentual respecto del MAE basal de cada modelo, equivalente al cambio relativo de MASE con el denominador fijo. Un valor positivo indica deterioro al perturbar la entrada; uno negativo indica menor error bajo esa perturbación. Las comparaciones son condicionales al modelo, al periodo y al esquema de donantes.

### Demanda y calendario

La demanda se examina como perfil normalizado y nivel medio de la ventana, tanto por separado como conjuntamente. Perturbar el nivel modifica el factor usado al restituir la salida a kWh. El calendario se transfiere simultáneamente en pasado y horizonte, conservando juntas las codificaciones de cada grupo. Se distinguen calendario completo, semana y festivos, y año y temporadas. Los grupos se solapan y sus importancias no se suman.'''

CARGA_GRAFICO = '''FI = ROOT/'results/jbook/importancia_variables'
reporte_fi = json.loads((FI/'ejecucion.json').read_text(encoding='utf-8'))
protocolo_fi = json.loads((FI/'protocolo.json').read_text(encoding='utf-8'))
fi = pd.read_csv(FI/'permutaciones.csv')
assert reporte_fi['completo'] and not reporte_fi['entrenamiento']
assert len(fi) == 2600 and not fi.duplicated(['modelo','semilla','grupo','repeticion']).any()
assert set(fi.modelo) == set(modelos)
assert set(fi.semilla) == set(range(1000,1005)) and set(fi.repeticion) == set(range(10))
# Primero se promedian las cinco inicializaciones dentro de cada permutación.
por_perm = fi.groupby(['modelo','grupo','repeticion'],sort=False)[['MAE_base','delta_MAE','delta_MASE']].mean().reset_index()
por_perm['delta_rel_pct'] = 100 * por_perm.delta_MAE / por_perm.MAE_base
resumen_fi = por_perm.groupby(['modelo','grupo'],sort=False).agg(
    cambio_MAE_pct=('delta_rel_pct','mean'),
    p10_perm_pct=('delta_rel_pct',lambda x:x.quantile(.1)),
    p90_perm_pct=('delta_rel_pct',lambda x:x.quantile(.9)),
    delta_MAE_kWh=('delta_MAE','mean'), delta_MASE=('delta_MASE','mean')).reset_index()
resumen_fi.to_csv(FI/'resumen.csv',index=False)
grupos_principales = ['demanda_completa','perfil_demanda','nivel_ventana',
                      'calendario_completo','semanal_festivos','anual_temporadas']
nombres_grupos = ['Demanda: perfil y nivel','Perfil de demanda, nivel fijo',
                 'Nivel medio de la ventana','Calendario completo',
                 'Semana y festivos','Año y temporadas']
matriz_fi = resumen_fi.pivot(index='grupo',columns='modelo',values='cambio_MAE_pct').reindex(
    index=grupos_principales,columns=modelos)
from matplotlib.colors import TwoSlopeNorm
lim_fi = float(np.max(np.abs(matriz_fi.to_numpy())))
fig, ax = plt.subplots(figsize=(10,5.7),layout='constrained')
im = ax.imshow(matriz_fi.to_numpy(),cmap='RdBu_r',
               norm=TwoSlopeNorm(vmin=-lim_fi,vcenter=0,vmax=lim_fi),aspect='auto')
ax.set(xticks=range(4),xticklabels=[etiquetas[m] for m in modelos],
       yticks=range(6),yticklabels=nombres_grupos)
ax.grid(False)
for i in range(6):
    for j in range(4):
        valor=matriz_fi.iloc[i,j]
        ax.text(j,i,f'{valor:+.1f} %',ha='center',va='center',
                color='white' if abs(valor)>.55*lim_fi else '#202020',fontsize=10)
fig.colorbar(im,ax=ax,label='Cambio del MAE frente al basal (%)')
fig.savefig(FI/'importancia_grupos.png',dpi=170)
display(fig)
plt.close(fig)
mlp_perfil=float(matriz_fi.loc['perfil_demanda','mlp'])
mlp_semana=float(matriz_fi.loc['semanal_festivos','mlp'])
mlp_anual=float(matriz_fi.loc['anual_temporadas','mlp'])
grupo_lider={etiquetas[m]:nombres_grupos[int(np.argmax(matriz_fi[m].to_numpy()))] for m in modelos}
'''

LECTURA_GRUPOS = '''En MLP, perturbar el perfil de demanda con nivel fijo aumenta el MAE un {mlp_perfil:.1f} %, mientras alterar semana y festivos lo incrementa un {mlp_semana:.1f} %. Año y temporadas produce un cambio menor, del {mlp_anual:.1f} %. El calendario laboral y la trayectoria reciente aportan información complementaria para reproducir el perfil diario. La importancia del nivel corresponde al procedimiento de normalización y restitución, además de las entradas explícitas de la red. La magnitud no es un porcentaje de varianza explicada ni una contribución aditiva. La correlación entre grupos y las combinaciones artificiales creadas al permutarlos limitan atribuciones individuales.'''

INTRO_REZAGOS = '''### Distribución de la sensibilidad a lo largo de la historia

Se perturban por separado los siete bloques consecutivos de 24 horas de demanda normalizada, manteniendo fijos el nivel y el calendario. El bloque 1 es el más reciente y el bloque 7 el más antiguo dentro de las 168 horas disponibles. Los puntos muestran la media; los segmentos, p10–p90 entre los diez desplazamientos después de promediar las cinco inicializaciones. Son dispersión de perturbaciones, no intervalos de confianza temporal.'''

GRAFICO_REZAGOS = '''grupos_dias=[f'dia_{i}' for i in range(1,8)]
fig, axes=plt.subplots(2,2,figsize=(12,8),sharex=True,sharey=True,layout='constrained')
q_dias=resumen_fi.loc[resumen_fi.grupo.isin(grupos_dias)]
xmin=min(0.,float(q_dias.p10_perm_pct.min()))
xmax=max(0.,float(q_dias.p90_perm_pct.max()))
for ax,m in zip(axes.ravel(),modelos):
    q=resumen_fi.loc[resumen_fi.modelo.eq(m)].set_index('grupo').loc[grupos_dias]
    for i,g in enumerate(grupos_dias):
        if m=='arx_lineal' and i>=2:
            ax.text(.06,i,'No utiliza este bloque',transform=ax.get_yaxis_transform(),
                    va='center',color='#777777',fontsize=8)
        else:
            ax.hlines(i,q.loc[g,'p10_perm_pct'],q.loc[g,'p90_perm_pct'],color='#247ba0',lw=3)
            ax.scatter(q.loc[g,'cambio_MAE_pct'],i,color='#173f5f',s=30,zorder=3)
    ax.axvline(0,color='#666666',ls='--',lw=1)
    ax.set(title=etiquetas[m],xlabel='Cambio del MAE frente al basal (%)',
           yticks=range(7),yticklabels=[f'Bloque {i}' for i in range(1,8)],
           xlim=(xmin-.04*(xmax-xmin),xmax*1.05),ylim=(6.6,-.6))
fig.savefig(FI/'importancia_rezagos.png',dpi=170)
display(fig)
plt.close(fig)
mlp_dias=resumen_fi.loc[resumen_fi.modelo.eq('mlp') & resumen_fi.grupo.isin(grupos_dias)].set_index('grupo')
mlp_bloque_lider=int(mlp_dias.cambio_MAE_pct.idxmax().split('_')[1])
mlp_ultimo_dia=float(mlp_dias.loc['dia_1','cambio_MAE_pct'])
lstm_antiguo=float(resumen_fi.loc[resumen_fi.modelo.eq('lstm') & resumen_fi.grupo.isin(grupos_dias[1:]),'cambio_MAE_pct'].abs().max())
lstm_antiguo_texto='inferior al 0,001 %' if lstm_antiguo<.001 else f'del {lstm_antiguo:.4f} %'
destino_fi=ROOT/'book/_static/resultados/importancia_variables'
destino_fi.mkdir(parents=True,exist_ok=True)
resumen_fi.to_csv(destino_fi/'resumen.csv',index=False)
fi.to_csv(destino_fi/'permutaciones.csv',index=False)
pd.read_csv(FI/'reproduccion_baseline.csv').to_csv(destino_fi/'reproduccion_baseline.csv',index=False)
for nombre in ['protocolo.json','ejecucion.json']:
    (destino_fi/nombre).write_text((FI/nombre).read_text(encoding='utf-8'),encoding='utf-8')
'''

LECTURA_REZAGOS = '''El bloque {mlp_bloque_lider} es el de mayor sensibilidad en MLP: perturbar las últimas 24 horas aumenta su MAE un {mlp_ultimo_dia:.1f} %. La sensibilidad a los bloques anteriores es menor. ARX utiliza únicamente 48 rezagos; los bloques 3–7 quedan identificados como ausentes de su entrada. En LSTM, el cambio absoluto máximo de los bloques 2–7 es {lstm_antiguo_texto}, compatible con una sensibilidad muy pequeña a esa parte del historial en este diagnóstico. Esa observación no demuestra que todas las redes recurrentes ignoren la historia larga.'''

LIMITES = '''### Alcance de la atribución

Las tres codificaciones de hora son idénticas entre orígenes diarios iniciados a medianoche. Permutar orígenes deja ese bloque inalterado; su importancia no es identificable mediante este procedimiento y no se reporta como cero. La salida directa de 24 posiciones también representa la posición horaria.

El desplazamiento preserva las trayectorias de cada bloque, pero rompe dependencias entre demanda, calendario y objetivo. Por ello mide sensibilidad ante perturbaciones fuera de la combinación observada y no estima el efecto de retirar una variable y reentrenar. Los grupos correlacionados pueden sustituirse parcialmente; una importancia pequeña no acredita irrelevancia intrínseca. La prueba conserva su alcance retrospectivo y estos resultados no modifican arquitecturas, selección de entradas ni métricas del benchmark.

[Resumen de importancias](_static/resultados/importancia_variables/resumen.csv) · [Resultados por semilla y desplazamiento](_static/resultados/importancia_variables/permutaciones.csv) · [Reproducción del baseline](_static/resultados/importancia_variables/reproduccion_baseline.csv) · [Protocolo](_static/resultados/importancia_variables/protocolo.json) · [Registro de cálculo](_static/resultados/importancia_variables/ejecucion.json).'''


def integrar_importancia(books,md,code,note,root):
    cells=books['10_evaluacion']
    start=next((i for i,c in enumerate(cells) if c.source.startswith('(importancia-variables)=')),None)
    if start is not None:
        end=next(i for i in range(start+1,len(cells)) if cells[i].source.startswith('## Lectura conjunta'))
        del cells[start:end]
    pos=next(i for i,c in enumerate(cells) if c.source.startswith('## Lectura conjunta'))
    source=(root/'scripts/importancia_variables.py').read_text(encoding='utf-8')
    pipeline='''### Pipeline reproducible de inferencia y perturbación

El siguiente código utiliza los checkpoints y escaladores conservados, verifica sus huellas, reproduce el baseline y calcula las perturbaciones. La construcción documental lee sus CSV; el cálculo se ejecuta explícitamente con `python scripts/importancia_variables.py` en un entorno con PyTorch CPU. No contiene pasos de optimización.

```python
'''+source+'\n```'
    cells[pos:pos]=[md(INTRO),code(CARGA_GRAFICO),note(LECTURA_GRUPOS),
        md(INTRO_REZAGOS),code(GRAFICO_REZAGOS),note(LECTURA_REZAGOS),md(LIMITES),md(pipeline)]
    for c in books.get('10_modelos',[]):
        if c.cell_type=='markdown' and c.source.startswith('## 10.5.'):
            c.source=c.source.replace('[importancia agrupada de variables](10_evaluacion.ipynb#importancia-variables)',
                '{ref}`importancia agrupada de variables <importancia-variables>`')
            if 'importancia agrupada de variables' not in c.source:
                c.source+='\n\nLa {ref}`importancia agrupada de variables <importancia-variables>` complementa el rendimiento global mediante la sensibilidad de los modelos a demanda y calendario.'

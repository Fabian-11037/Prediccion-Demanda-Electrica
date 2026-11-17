"""Integra la precisión disponible en tablas y figuras sin mezclar estimandos."""
import nbformat as nb


def integrar_presentacion_precision(root,cells):
    md=nb.v4.new_markdown_cell;code=nb.v4.new_code_cell
    setup=code('''# Inventario de precisión de desarrollo: una fuente por procedimiento.
v_pilot=g_pilots.groupby(['clave','semilla'],as_index=False).MASE.mean()
v_pilot['Fuente']='Piloto externo';v_pilot['Periodo']='2018–2021 · media de cuatro años'
v_stop=pd.read_csv(g_result/'entradas_precision_parada.csv')
v_stop=v_stop[['clave','semilla','MASE_parada']].rename(columns={'MASE_parada':'MASE'})
v_stop['Fuente']='Parada interna';v_stop['Periodo']='2019-10-21 a 2021-12-31 · 803 días'
assert set(v_pilot.clave).isdisjoint(set(v_stop.clave))
v_seeds=pd.concat([v_pilot,v_stop],ignore_index=True)
assert len(v_seeds)==5*len(g_keys) and set(v_seeds.clave)==set(g_keys)
assert not v_seeds.duplicated(['clave','semilla']).any()
assert v_seeds.groupby('clave').size().eq(5).all()
v_precision=v_seeds.groupby('clave').agg(MASE_media=('MASE','mean'),DE=('MASE','std'),
    Semillas=('semilla','count'),Fuente=('Fuente','first'),Periodo=('Periodo','first')).reindex(g_keys)
v_precision['CV_pct']=100*v_precision.DE/v_precision.MASE_media
v_width=stats.t.ppf(.975,4)*v_precision.DE/np.sqrt(5)
v_precision['IC95_inf']=v_precision.MASE_media-v_width
v_precision['IC95_sup']=v_precision.MASE_media+v_width
v_precision.insert(0,'Procedimiento',[g_names[k] for k in v_precision.index])
v_precision.to_csv(g_result/'precision_desarrollo_resumen.csv',index_label='clave')
v_seeds.to_csv(g_result/'precision_desarrollo_semillas.csv',index=False)
assert np.allclose(v_precision.loc[v_stop.clave.unique(),'MASE_media'],
    v_stop.groupby('clave').MASE.mean().reindex(v_stop.clave.unique()))
''')
    i=next(i for i,c in enumerate(cells) if c.cell_type=='code' and c.source.startswith('# Predicciones de los catorce'))
    cells.insert(i+1,setup)
    # La tabla de selección identifica también la fuente de precisión completada.
    c=next(c for c in cells if c.cell_type=='code' and c.source.startswith('g_design=pd.read_csv'))
    c.source=c.source.replace("g_design.index=[g_names[k] for k in g_keys]",
        "g_design['precision']=v_precision.Fuente\ng_design.index=[g_names[k] for k in g_keys]")
    c.source=c.source.replace("'max_epocas':'Épocas máx. filtro / resto','paciencia':'Paciencia'",
        "'max_epocas':'Épocas máx. filtro / resto','paciencia':'Paciencia','precision':'Precisión disponible'")
    # La figura de selección conserva el detalle de todos los datos disponibles.
    i=next(i for i,c in enumerate(cells) if c.cell_type=='code' and c.source.startswith('fig,axes=plt.subplots(1,3') and 'g_pilot_keys=' in c.source)
    cells[i]=code('''fig,axes=plt.subplots(2,2,figsize=(18,11),layout='constrained')
v_pilot_keys=[k for k in g_keys if k in set(v_pilot.clave)]
v_stop_keys=[k for k in g_keys if k in set(v_stop.clave)]
for v_k in v_pilot_keys:
    v_year=g_pilots[g_pilots.clave.eq(v_k)].groupby('fold').MASE
    axes[0,0].errorbar(v_year.mean().index,v_year.mean(),yerr=v_year.std(),marker='o',capsize=3,label=g_names[v_k])
axes[0,0].set(title='Piloto externo · seis procedimientos',xlabel='Año externo',ylabel='MASE · media ± DE',xticks=[2018,2019,2020,2021])
axes[0,0].legend(fontsize=8)
for ax,v_frame,v_ks,v_title in [(axes[0,1],v_pilot,v_pilot_keys,'Piloto · media de cuatro años por semilla'),
    (axes[1,0],v_stop,v_stop_keys,'Parada interna · 803 días por semilla')]:
    for v_j,v_k in enumerate(v_ks):
        v_rows=v_frame[v_frame.clave.eq(v_k)].sort_values('semilla')
        ax.scatter(v_j+np.linspace(-.10,.10,len(v_rows)),v_rows.MASE,s=28,label=None)
    ax.set(title=v_title,xticks=range(len(v_ks)),xticklabels=[g_names[k] for k in v_ks],ylabel='MASE por inicialización')
    ax.tick_params(axis='x',rotation=30,labelsize=8)
for v_k in modelos:
    v_perm=piloto[piloto.modelo.eq(v_k)].groupby('fold')[['MASE','MASE_demanda_permutada']].mean()
    axes[1,1].plot(v_perm.index,v_perm.MASE_demanda_permutada/v_perm.MASE,marker='o',label=etiquetas[v_k])
axes[1,1].axhline(1,color='grey',ls='--')
axes[1,1].set(title='Control de demanda permutada · cuatro familias',xlabel='Año externo',ylabel='MASE permutado / original',xticks=[2018,2019,2020,2021])
axes[1,1].legend(fontsize=8)
fig.savefig(g_figures/'precision_desarrollo_fuentes.png',dpi=160)
display(fig);plt.close(fig)
''')
    cells[i+1]=md('''Los catorce procedimientos tienen una estimación de precisión de desarrollo: seis a partir del piloto externo y ocho a partir de la parada interna. Los puntos muestran cinco inicializaciones por procedimiento, sin tratar los años como semillas adicionales. Las dos fuentes cubren periodos y mecanismos de selección distintos; sus medias no constituyen un ranking común de validación. El control de permutación conserva las cuatro familias para las que fue calculado.''')
    # Un solo resumen visible evita repetir las dos tablas de seis procedimientos.
    c=next(c for c in cells if c.cell_type=='code' and c.source.startswith('g_pilot_seed='))
    lines=[line for line in c.source.splitlines() if not line.startswith('display(g_pilot_summary')]
    c.source='\n'.join(lines)+'''\ndisplay(v_precision[['Procedimiento','Fuente','Periodo','Semillas','MASE_media','DE','CV_pct','IC95_inf','IC95_sup']].rename(columns={
    'Fuente':'Fuente de precisión','MASE_media':'MASE desarrollo','CV_pct':'CV (%)',
    'IC95_inf':'IC95 inferior','IC95_sup':'IC95 superior'}).reset_index(drop=True).style.format(
        {'MASE desarrollo':'{:.5f}','DE':'{:.5f}','CV (%)':'{:.2f}','IC95 inferior':'{:.5f}','IC95 superior':'{:.5f}'}))'''
    c=next(c for c in cells if c.cell_type=='code' and 'p_summary,p_pairs,p_matrix=' in c.source)
    start=c.source.index('display(p_summary[[')
    c.source=c.source[:start]+'''assert np.allclose(p_summary.set_index('clave').MASE_media,
    v_precision.MASE_media.reindex(p_summary.clave))
'''
    # Generalización integra una precisión de desarrollo disponible en los catorce.
    i=next(i for i,c in enumerate(cells) if c.cell_type=='code' and c.source.startswith('g_general = '))
    cells[i]=code('''from matplotlib.patches import Patch
g_general=g_meta.groupby('clave').validacion.first().reindex(g_keys).to_frame('Validación externa')
g_general['Precisión desarrollo']=v_precision.MASE_media
g_general['Prueba']=g_benchmark.loc[g_keys,'MASE']
g_general['Fuente precisión']=v_precision.Fuente
g_general['Periodo precisión']=v_precision.Periodo
g_general['Periodo validación']=g_meta.groupby('clave').anios_validacion.first().reindex(g_keys)
assert g_general.notna().all().all() and len(g_general)==len(g_keys)
fig,ax=plt.subplots(figsize=(16,7))
v_pos=np.arange(len(g_keys));v_w=.25
ax.bar(v_pos-v_w,g_general['Validación externa'],v_w,color='#8295a5',label='Validación externa de selección')
v_colors=['#417da2' if f=='Piloto externo' else '#d18b38' for f in g_general['Fuente precisión']]
v_bars=ax.bar(v_pos,g_general['Precisión desarrollo'],v_w,color=v_colors)
for v_bar,v_source in zip(v_bars,g_general['Fuente precisión']):
    if v_source=='Parada interna':v_bar.set_hatch('//')
ax.bar(v_pos+v_w,g_general.Prueba,v_w,color='#258a83',label='Prueba · 2022–2025')
ax.set(xticks=v_pos,xticklabels=[g_names[k] for k in g_keys],ylabel='MASE',xlabel='')
ax.tick_params(axis='x',rotation=40,labelsize=8)
ax.legend(handles=[Patch(color='#8295a5',label='Validación externa de selección'),
    Patch(color='#417da2',label='Precisión: piloto externo · 2018–2021'),
    Patch(facecolor='#d18b38',hatch='//',label='Precisión: parada interna · 2019-10-21 a 2021-12-31'),
    Patch(color='#258a83',label='Prueba · 2022–2025')],fontsize=9)
fig.tight_layout();fig.savefig(g_figures/'generalizacion_completa.png',dpi=160)
display(fig);plt.close(fig)
g_general.to_csv(g_result/'generalizacion_integrada.csv',index_label='clave')
''')
    cells[i-1]=md('''## Validación y generalización

Los catorce procedimientos incluyen validación de selección, precisión de desarrollo y prueba. La barra central utiliza el piloto externo donde existe y el MASE de parada interna en los otros ocho; color y trama identifican su fuente. La validación de PatchTST/N-HiTS numéricos promedia 2019–2021; las demás, 2018–2021. La parada interna no se presenta como validación externa.''')
    cells[i+1]=md('''El error de prueba supera al de validación en los catorce procedimientos. La figura incorpora la precisión de desarrollo sin huecos: piloto externo en azul y parada interna en naranja con trama. La parada participó en la selección de época y puede ofrecer una estimación optimista; el aumento hasta prueba no se interpreta aisladamente como sobreajuste ni compara periodos idénticos.''')
    # Ambos dimensionamientos se visualizan con referencias y familias explícitas.
    figure=[md('''### Precisión requerida entre inicializaciones

Las matrices muestran el número aproximado de semillas por procedimiento para cada pareja, con Bonferroni por grupo (15 parejas externas y 28 internas), δ=3 % y potencia objetivo de 0,80. Cada panel conserva su fuente y referencia de escala; no se combinan pares entre fuentes distintas.'''),code('''v_dimension_groups=[(g_dimension,'A','B','R Bonferroni(15)','Piloto externo · referencia MLP'),
    (p_pairs,'A','B','R_Bonferroni','Parada interna · referencia control N')]
from matplotlib.colors import LogNorm
fig,axes=plt.subplots(1,2,figsize=(18,8),layout='constrained')
v_max=max(float(g_dimension['R Bonferroni(15)'].max()),float(p_pairs.R_Bonferroni.max()))
for ax,(v_df,v_a,v_b,v_r,v_title) in zip(axes,v_dimension_groups):
    v_ks=[k for k in g_keys if k in set(v_df[v_a])|set(v_df[v_b])]
    v_matrix=np.full((len(v_ks),len(v_ks)),np.nan)
    for _,v_row in v_df.iterrows():
        v_i,v_j=v_ks.index(v_row[v_a]),v_ks.index(v_row[v_b])
        v_matrix[v_i,v_j]=v_matrix[v_j,v_i]=v_row[v_r]
    v_im=ax.imshow(v_matrix,cmap='YlOrRd',norm=LogNorm(vmin=5,vmax=v_max))
    ax.set(title=v_title,xticks=range(len(v_ks)),yticks=range(len(v_ks)),
        xticklabels=[g_names[k] for k in v_ks],yticklabels=[g_names[k] for k in v_ks])
    ax.tick_params(axis='x',rotation=60,labelsize=8);ax.tick_params(axis='y',labelsize=8);ax.grid(False)
    for v_i in range(len(v_ks)):
        for v_j in range(len(v_ks)):
            if v_i!=v_j:ax.text(v_j,v_i,f'{v_matrix[v_i,v_j]:.0f}',ha='center',va='center',fontsize=10,
                color='white' if v_matrix[v_i,v_j]>.6*v_max else '#222222')
fig.colorbar(v_im,ax=axes.tolist(),shrink=.65,label='Semillas aproximadas · escala logarítmica; Bonferroni por grupo')
fig.savefig(g_figures/'dimensionamiento_fuentes.png',dpi=160)
display(fig);plt.close(fig)
'''),md('''La planificación conservadora alcanza máximos de 82 semillas en los pilotos externos y el máximo indicado en la tabla de parada interna. Las mayores cifras reflejan la dispersión de las diferencias frente al margen fijado, no una mayor diferencia media de rendimiento. Cinco semillas no acreditan la potencia objetivo en todas las parejas; el cálculo retrospectivo no modifica el número ejecutado.''')]
    i=next(i for i,c in enumerate(cells) if c.cell_type=='markdown' and c.source.startswith('## Curvas de aprendizaje'))
    cells[i:i]=figure
    # La tabla principal mantiene las métricas de prueba y enlaza su precisión.
    c=next(c for c in cells if c.cell_type=='code' and 'e2_tabla = ' in c.source)
    c.source='''v_test_table=g_benchmark[['modelo','MASE','MAE','RMSE','semillas']].copy()
v_test_table['Precisión de desarrollo']=v_precision.Fuente.reindex(v_test_table.index).fillna('Referencia determinista')
v_test_table=v_test_table.sort_values('MAE').rename(columns={'modelo':'Procedimiento',
    'MASE':'MASE de prueba','MAE':'MAE de prueba (kWh)','RMSE':'RMSE de prueba (kWh)','semillas':'Realizaciones'})
display(v_test_table.reset_index(drop=True).round({'MASE de prueba':4,'MAE de prueba (kWh)':2,'RMSE de prueba (kWh)':2}))
'''
    for cell in cells:
        if cell.cell_type=='markdown':
            cell.source=cell.source.replace('### Dimensionamiento y precisión del piloto','### Precisión de desarrollo y dimensionamiento')
            if cell.source.startswith('### Precisión de desarrollo y dimensionamiento'):
                cell.source='''### Precisión de desarrollo y dimensionamiento

El resumen reúne los catorce procedimientos e identifica la fuente y el periodo de cada estimación. Los seis pilotos externos y las ocho reservas internas se dimensionan por separado. Cada semilla piloto 100–104 resume cuatro años externos 2018–2021; cada semilla final 1000–1004 resume 803 días de parada. Se muestran media de MASE, DE, CV e IC95 % t entre cinco inicializaciones. Los años no se consideran semillas adicionales y los errores de prueba no intervienen en el dimensionamiento.'''
            cell.source=cell.source.replace('Se resumen los seis procedimientos con piloto independiente y se dimensionan sus quince parejas. Los otros seis se analizan a continuación mediante sus métricas de parada interna.',
                'El resumen reúne los catorce procedimientos, identificando la fuente y el periodo de cada estimación. Los seis pilotos externos y las ocho reservas internas se dimensionan por separado.')
            cell.source=cell.source.replace('La tabla presenta media de MASE, desviación estándar (DE) y coeficiente de variación (CV).',
                'La tabla presenta media de MASE, desviación estándar (DE), coeficiente de variación (CV) e IC95 % t entre las cinco inicializaciones. Para el piloto, cada inicialización es la media de cuatro años; para parada, resume los 803 días de reserva.')
            cell.source=cell.source.replace('esta tabla no sustituye el benchmark de prueba.','el resumen integrado no sustituye el benchmark de prueba.')
            if cell.source.startswith('## Discusión y alcance del Entregable 2'):
                cell.source+='\n\nLa precisión de desarrollo está estimada para los catorce procedimientos: seis mediante piloto externo y ocho mediante parada interna. Esto completa la descripción de variación y la planificación retrospectiva; no acredita por sí solo potencia de 0,80 ni convierte la reserva de selección de época en una evaluación externa.'
            if cell.source.startswith('## Tablas y reproducción'):
                cell.source+='\n\n[Precisión de desarrollo de catorce procedimientos](_static/resultados/evaluacion_e2/precision_desarrollo_resumen.csv) · [Setenta métricas de desarrollo](_static/resultados/evaluacion_e2/precision_desarrollo_semillas.csv) · [Generalización integrada](_static/resultados/evaluacion_e2/generalizacion_integrada.csv).'
    return cells

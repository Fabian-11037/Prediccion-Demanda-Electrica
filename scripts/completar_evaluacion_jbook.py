"""Amplía los diagnósticos a todos los procedimientos con predicciones conservadas.

No entrena, modifica checkpoints ni vuelve a seleccionar configuraciones.
Las celdas generadas contienen el cálculo y la gráfica, no una llamada opaca.
"""
from pathlib import Path
import copy
import json
import shutil
import numpy as np
import pandas as pd
import nbformat as nb
from seleccion_patchtst import filtrar,RETIRAR


def metadatos(root):
    """Normaliza registros de ajuste; nunca sustituye una etapa ausente por cero."""
    root = Path(root)
    source = root.parent / 'Consolidado-Entrenamientos-20261115'
    if not (source/'consolidado/indice_modelos.json').is_file():
        # La copia de entrega conserva estas tablas de entrada ya normalizadas.
        cached=root/'results/evaluacion_e2'
        frame=pd.read_csv(cached/'metadatos_procedimientos.csv')
        pilots=pd.read_csv(cached/'pilotos_disponibles.csv')
        assert len(frame)==5*pd.read_csv(root/'results/evaluacion_e2/benchmark.csv').semillas.eq(5).sum() and not frame.duplicated(['clave','semilla']).any()
        assert len(pilots)==100 and not pilots.duplicated(['clave','semilla','fold']).any()
        for name in ['metadatos_procedimientos.csv','pilotos_disponibles.csv']:
            shutil.copy2(cached/name,root/'book/_static/resultados/evaluacion_e2'/name)
        return
    index = json.loads((source/'consolidado/indice_modelos.json').read_text(encoding='utf-8'))
    old = root/'results/repositorio/d1ae9a4e09dd1dead4af1329f16aa3e8d0f2c3d3'
    registry = json.loads((old/'results/consolidado/modelos.json').read_text(encoding='utf-8'))
    rows, pilots = [], []
    for key, model in registry.items():
        p = pd.read_csv(old/'results'/model['ejecucion']/'piloto.csv')
        p = p.loc[(p.modelo==key)&(p.configuracion==model['configuracion']['id'])]
        pilots.extend(dict(clave=key,semilla=int(x.semilla),fold=int(x.fold),MASE=x.MASE) for x in p.itertuples())
        for artifact in model['artefactos']:
            result = json.loads((old/artifact['metricas']).read_text(encoding='utf-8'))
            rows.append(dict(clave=key,semilla=result['semilla'],validacion=model['MASE_cv'],
                anios_validacion='2018–2021',parametros=result['n_parametros'],segundos=result['segundos'],
                epocas=result['epocas'],mejor_epoca=result['mejor_epoca'],
                MAE_ajuste_normalizado=result['MAE_train_normalizado_eval'],
                MAE_parada_normalizado=result['MAE_es_normalizado_eval'],
                MASE_ajuste=np.nan,MASE_parada=np.nan,dispositivo=result['dispositivo']))
    for model in index['modelos']:
        if not model['principal']:
            continue
        key=model['experimento']+'__'+model['variante']
        config=index['configuraciones'][model['experimento']]
        result=json.loads((source/model['artefactos']['resultado']).read_text(encoding='utf-8'))
        rows.append(dict(clave=key,semilla=model['semilla'],validacion=config['val'][model['variante']],
            anios_validacion='–'.join(str(v) for v in [min(config['anios_media_validacion']),max(config['anios_media_validacion'])]),
            parametros=result['n_parametros'],segundos=result['segundos'],epocas=result['epocas'],
            mejor_epoca=result['mejor_epoca'],MAE_ajuste_normalizado=np.nan,MAE_parada_normalizado=np.nan,
            MASE_ajuste=result.get('MASE_ajuste',np.nan),MASE_parada=result.get('MASE_parada',np.nan),
            dispositivo=result.get('dispositivo','GPU · registro de hardware')))
    for exp in ['repo_patchtst_numerico_r1','repo_nhits_numerico_r1']:
        p=source/'fuentes/repositorio_66111003f387/results'/exp.removeprefix('repo_')/'piloto.csv'
        table=pd.read_csv(p)
        assert len(table)==20 and not table.duplicated(['semilla','fold']).any()
        pilots.extend(dict(clave=exp+'__N',semilla=int(x.semilla),fold=int(x.fold),MASE=x.MASE) for x in table.itertuples())
    out=root/'results/evaluacion_e2'
    frame=filtrar(pd.DataFrame(rows))
    pilots=filtrar(pd.DataFrame(pilots)).to_dict('records')
    assert len(frame)==5*pd.read_csv(root/'results/evaluacion_e2/benchmark.csv').semillas.eq(5).sum() and not frame.duplicated(['clave','semilla']).any()
    assert all(len(g)==5 for _,g in frame.groupby('clave'))
    frame.to_csv(out/'metadatos_procedimientos.csv',index=False)
    pd.DataFrame(pilots).to_csv(out/'pilotos_disponibles.csv',index=False)
    for name in ['metadatos_procedimientos.csv','pilotos_disponibles.csv']:
        shutil.copy2(out/name,root/'book/_static/resultados/evaluacion_e2'/name)


def metadatos_seleccion(root):
    """Conserva presupuestos y diagnósticos del piloto sin usar prueba."""
    root=Path(root);source=root.parent/'Consolidado-Entrenamientos-20261115'
    out=root/'results/evaluacion_e2'
    names=['diseno_seleccion.csv','dimensionamiento_temporal_registrado.csv']
    if (source/'consolidado/indice_modelos.json').is_file():
        index=json.loads((source/'consolidado/indice_modelos.json').read_text(encoding='utf-8'))
        old=root/'results/repositorio/d1ae9a4e09dd1dead4af1329f16aa3e8d0f2c3d3'
        registry=json.loads((old/'results/consolidado/modelos.json').read_text(encoding='utf-8'))
        phases={stage:pd.concat([pd.read_csv(old/'results'/run/(stage+'.csv'))
            for run in sorted({m['ejecucion'] for m in registry.values()})],ignore_index=True)
            for stage in ['filtro','confirmacion']}
        rows=[]
        for key in registry:
            f=phases['filtro'].loc[phases['filtro'].modelo.eq(key)]
            c=phases['confirmacion'].loc[phases['confirmacion'].modelo.eq(key)]
            rows.append(dict(clave=key,candidatos_filtro=f.configuracion.nunique(),
                filtro=f"{len(f)} ajustes; {f.semilla.nunique()} semilla; 2019",
                confirmacion=f"{len(c)} ajustes; {c.semilla.nunique()} semillas; 2018–2021",
                piloto='5 semillas × 4 años',final='5 semillas',max_epocas='30 / 60',paciencia=6))
        completed={(m['experimento'],m['variante']) for m in index['modelos'] if m['principal']}
        for exp,variant in sorted(completed):
            config=index['configuraciones'][exp];repo=config['fuente']=='repositorio'
            seeds=len(config['semillas_filtro']);confirm_seeds=len(config['semillas_confirmacion'])
            confirms=config['finalistas']*confirm_seeds*3
            seed_word='semilla' if seeds==1 else 'semillas'
            confirm_word='semilla' if confirm_seeds==1 else 'semillas'
            rows.append(dict(clave=exp+'__'+variant,candidatos_filtro=config['configuraciones_filtro'],
                filtro=f"{config['configuraciones_filtro']*seeds} ajustes; {seeds} {seed_word}; 2018",
                confirmacion=f'{confirms} ajustes; {confirm_seeds} {confirm_word}; 2019–2021',
                piloto='5 semillas × 4 años' if repo else 'Sin piloto independiente',
                final='5 semillas',max_epocas='200 / 200',paciencia=10 if repo else 6))
        table=filtrar(pd.DataFrame(rows))
        assert len(table)==pd.read_csv(out/'benchmark.csv').semillas.eq(5).sum() and not table.clave.duplicated().any()
        table.to_csv(out/names[0],index=False)
        temporal=[]
        for exp in ['repo_patchtst_numerico_r1','repo_nhits_numerico_r1']:
            path=source/'fuentes/repositorio_66111003f387/results'/exp.removeprefix('repo_')/'dimensionamiento.json'
            record=json.loads(path.read_text(encoding='utf-8'))
            temporal.append(dict(clave=exp+'__N',referencia='Ingenuo semanal en validación',
                MASE_referencia=record['MASE_referencia'],margen_relativo=record['margen_relativo'],
                potencia_limite_temporal=record['potencia_limite_temporal'],
                semillas_piloto=record['semillas_piloto'],n_final=record['n_final'],estado=record['estado']))
        filtrar(pd.DataFrame(temporal)).to_csv(out/names[1],index=False)
    for name in names:
        shutil.copy2(out/name,root/'book/_static/resultados/evaluacion_e2'/name)


def ampliar(root,cells):
    metadatos(root)
    metadatos_seleccion(root)
    md=nb.v4.new_markdown_cell
    code=nb.v4.new_code_cell
    # Estas variables son independientes de las del diagnóstico de importancia.
    setup=code('''# Predicciones de los catorce procedimientos, alineadas por semilla, origen y hora.
g_result = ROOT/'results/evaluacion_e2'
g_benchmark = pd.read_csv(g_result/'benchmark.csv').set_index('clave')
g_keys = g_benchmark.index[g_benchmark.semillas.eq(5)].tolist()
g_names = g_benchmark.modelo.to_dict()
g_meta = pd.read_csv(g_result/'metadatos_procedimientos.csv')
g_pilots = pd.read_csv(g_result/'pilotos_disponibles.csv')
g_pred = {k: pred[k].copy() for k in modelos}
g_y = y.copy()
g_dates = fechas.copy()
g_q = escala
g_df = pd.read_parquet(g_result/'entradas/consolidado/predicciones_finales.parquet')
for (g_exp,g_variant),g_group in g_df.groupby(['experimento','variante'],sort=False):
    g_key = g_exp+'__'+g_variant
    g_arrays = []
    for g_seed,g_rows in g_group.groupby('semilla',sort=True):
        g_rows = g_rows.sort_values(['origen','horizonte'])
        assert g_seed in range(1000,1005)
        assert np.allclose(g_rows.observado_kwh.to_numpy().reshape(1461,24),g_y,atol=1e-6)
        assert pd.DatetimeIndex(g_rows.origen.drop_duplicates()).tz_convert('America/Bogota').equals(g_dates)
        assert np.array_equal(g_rows.horizonte.to_numpy(),np.tile(np.arange(1,25),1461))
        g_arrays.append(g_rows.predicho_kwh.to_numpy().reshape(1461,24))
    g_pred[g_key] = np.stack(g_arrays)
assert set(g_pred)==set(g_keys) 
g_daily = {}
for g_k in g_keys:
    assert g_pred[g_k].shape==(5,1461,24) and np.isfinite(g_pred[g_k]).all()
    g_error = g_pred[g_k]-g_y[None]
    g_daily[g_k] = np.abs(g_error).mean(axis=(0,2))/g_q
    assert np.isclose(g_daily[g_k].mean(),g_benchmark.loc[g_k,'MASE'],rtol=2e-6)
    assert np.isclose(np.sqrt((g_error**2).mean(axis=(1,2))).mean(),g_benchmark.loc[g_k,'RMSE'],rtol=2e-6)
g_figures = g_result/'figuras'
g_figures.mkdir(exist_ok=True)
''')
    general=[md('''## Validación y generalización

Se comparan validación, piloto independiente cuando existe y prueba en los catorce procedimientos. La validación de PatchTST y N-HiTS numéricos promedia 2019–2021; las demás, 2018–2021. El piloto de seis procedimientos promedia cuatro años y cinco semillas. Un espacio vacío indica una etapa no disponible, no un error igual a cero.'''),setup,code('''g_general = g_meta.groupby('clave').validacion.first().reindex(g_keys).to_frame('Validación')
g_general['Piloto'] = g_pilots.groupby('clave').MASE.mean().reindex(g_keys)
g_general['Prueba'] = g_benchmark.loc[g_keys,'MASE']
g_general.index = [g_names[k] for k in g_keys]
fig,ax = plt.subplots(figsize=(15,6))
g_general.plot.bar(ax=ax,rot=45,width=.8,color=['#447ba0','#dc923b','#258a83'])
ax.set(ylabel='MASE',xlabel='');ax.tick_params(axis='x',labelsize=8)
ax.legend();fig.tight_layout();fig.savefig(g_figures/'generalizacion_completa.png',dpi=160)
display(fig);plt.close(fig)
'''),md('''El error de prueba supera al de validación en los catorce procedimientos, con aumentos de distinta magnitud. El cambio de periodo afecta a todas las familias y no se interpreta aisladamente como sobreajuste. Las barras de validación resumen ventanas diferentes, por lo que su comparación no equivale a una búsqueda con protocolo idéntico. Los seis procedimientos sin piloto independiente conservan esa ausencia.'''),
md('''### Ajuste y parada interna

Los registros conservan MAE normalizado en modo evaluación para las cuatro familias de demanda/calendario y MASE de ajuste/parada para ocho variantes ampliadas. Se muestran por separado para preservar sus unidades. PatchTST y N-HiTS numéricos conservan curvas de pérdida, pero no estas métricas de evaluación de ambos tramos; no se reemplazan por pérdidas de entrenamiento calculadas con dropout.'''),code('''fig,axes = plt.subplots(1,2,figsize=(16,5))
for ax,g_cols,g_label in zip(axes,[['MAE_ajuste_normalizado','MAE_parada_normalizado'],['MASE_ajuste','MASE_parada']],['MAE normalizado · modo evaluación','MASE · modo evaluación']):
    g_present = [k for k in g_keys if g_meta.loc[g_meta.clave.eq(k),g_cols].notna().all(axis=1).all()]
    for g_i,g_k in enumerate(g_present):
        g_rows = g_meta[g_meta.clave.eq(g_k)].sort_values('semilla')
        for g_offset,g_col,g_color,g_label_col in zip([-.09,.09],g_cols,['#447ba0','#dc923b'],['Ajuste','Parada interna']):
            ax.scatter(np.full(5,g_i)+g_offset,g_rows[g_col],color=g_color,label=g_label_col if g_i==0 else None,s=25)
    ax.set(xticks=np.arange(len(g_present)),xticklabels=[g_names[k] for k in g_present],ylabel=g_label)
    ax.tick_params(axis='x',rotation=35,labelsize=8);ax.legend()
fig.tight_layout();display(fig);plt.close(fig)
'''),md('''La parada interna presenta mayor error que el ajuste en las realizaciones mostradas. La separación describe generalización dentro del periodo de desarrollo; su magnitud no se compara entre paneles porque las escalas difieren. Las curvas de aprendizaje del capítulo incluyen también los dos procedimientos sin estas métricas desagregadas.''')]
    observed=[md('''(observado-prediccion)=
## Demanda observada frente a predicción

Cada panel reúne 35.064 horas y cinco semillas: 175.320 pares hora–semilla por procedimiento. La diagonal representa predicción exacta; ejes y escala logarítmica de densidad son comunes. MAE y RMSE se promedian entre realizaciones, igual que en el benchmark.'''),code('''from matplotlib.colors import LogNorm
g_obs = np.broadcast_to(g_y,(5,*g_y.shape)).reshape(-1)/1e6
g_ext = [g_y.min()/1e6,g_y.max()/1e6]
for g_k in g_keys:
    g_ext.extend([g_pred[g_k].min()/1e6,g_pred[g_k].max()/1e6])
g_pad = .025*(max(g_ext)-min(g_ext))
g_limits = (min(g_ext)-g_pad,max(g_ext)+g_pad)
fig,axes = plt.subplots(int(np.ceil(len(g_keys)/3)),3,figsize=(16,21),sharex=True,sharey=True,layout='constrained')
for ax,g_k in zip(axes.flat,g_keys):
    g_density = ax.hexbin(g_obs,g_pred[g_k].reshape(-1)/1e6,gridsize=55,mincnt=1,cmap='viridis',norm=LogNorm(1,len(g_obs)),extent=(*g_limits,*g_limits))
    ax.plot(g_limits,g_limits,'--',color='#c44e52',lw=1.2)
    ax.set(title=g_names[g_k],xlim=g_limits,ylim=g_limits,xlabel='Observado (GWh por intervalo)',ylabel='Predicho (GWh por intervalo)')
    ax.set_aspect('equal',adjustable='box')
    ax.text(.04,.96,f"MAE: {g_benchmark.loc[g_k,'MAE']/1e6:.3f} GWh\\nRMSE: {g_benchmark.loc[g_k,'RMSE']/1e6:.3f} GWh",transform=ax.transAxes,va='top',fontsize=9,bbox=dict(facecolor='white',alpha=.85,edgecolor='none'))
for g_ax in list(axes.flat)[len(g_keys):]:g_ax.set_visible(False)
fig.colorbar(g_density,ax=axes.ravel().tolist(),shrink=.65,label='Pares hora–semilla por celda · escala logarítmica')
fig.savefig(g_figures/'observado_predicho_completo.png',dpi=160)
display(fig);plt.close(fig)
'''),md('''TaTS y su control numérico muestran los menores errores agregados y una nube más próxima a la diagonal. Las desviaciones en demandas altas y bajas revelan errores de amplitud que el promedio global puede ocultar. La dispersión incluye variación temporal y entre semillas; su densidad no convierte las realizaciones compartidas en observaciones independientes.'''),md('''### Perfil horario en una semana de prueba

Se utiliza la misma ventana del 1 al 7 de julio de 2023 para los catorce procedimientos. La banda p10–p90 representa dispersión entre cinco semillas y no un intervalo predictivo calibrado. El MAE semanal se calcula con los errores individuales, mientras la predicción media se utiliza para visualizar el perfil.'''),code('''import matplotlib.dates as mdates
g_week = (g_dates>=pd.Timestamp('2023-07-01',tz='America/Bogota'))&(g_dates<pd.Timestamp('2023-07-08',tz='America/Bogota'))
assert g_week.sum()==7
g_time = pd.DatetimeIndex([d+pd.Timedelta(hours=h) for d in g_dates[g_week] for h in range(24)])
g_week_mae = {}
fig,axes = plt.subplots(int(np.ceil(len(g_keys)/3)),3,figsize=(18,18),sharex=True,sharey=True,layout='constrained')
for ax,g_k in zip(axes.flat,g_keys):
    g_v = g_pred[g_k][:,g_week,:].reshape(5,-1)/1e6
    g_week_mae[g_k] = np.abs(g_pred[g_k][:,g_week,:]-g_y[None,g_week,:]).mean()/1e6
    ax.fill_between(g_time,np.quantile(g_v,.1,axis=0),np.quantile(g_v,.9,axis=0),color='#447ba0',alpha=.2,label='p10–p90 entre semillas')
    ax.plot(g_time,g_y[g_week].reshape(-1)/1e6,color='#222222',lw=1.5,label='Observado')
    ax.plot(g_time,g_v.mean(axis=0),'--',color='#447ba0',lw=1.2,label='Predicción media')
    ax.set(title=f'{g_names[g_k]}\\nMAE semanal: {g_week_mae[g_k]:.3f} GWh',xlabel='Julio de 2023 · Colombia',ylabel='GWh por intervalo')
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2,tz=g_time.tz))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m',tz=g_time.tz))
    ax.legend(fontsize=7,loc='lower right')
for g_ax in list(axes.flat)[len(g_keys):]:g_ax.set_visible(False)
fig.savefig(g_figures/'semana_completa.png',dpi=160)
display(fig);plt.close(fig)
'''),md('''Los procedimientos reproducen el ciclo diario, pero difieren en la altura de los máximos, los mínimos nocturnos y las transiciones. Esta ventana permite contextualizar las métricas en unidades de demanda; siete días no representan por sí solos el ranking del periodo completo. La banda entre semillas describe estabilidad de inicialización y puede ser estrecha aun cuando exista un error común.''')]
    segments=[md('''## Segmentos de evaluación

Se conserva la evaluación por año, tipo de día y cuartil de demanda para los catorce procedimientos. Cada panel muestra pérdidas promediadas entre semillas dentro del segmento. Los cortes describen heterogeneidad de prueba y no se utilizan para seleccionar configuraciones.'''),code('''g_panel = pd.read_parquet(ROOT/'data/processed/panel_horario.parquet',columns=['cal_es_festivo'])
g_festivo = g_panel.reindex(g_dates).cal_es_festivo.to_numpy().astype(bool)
g_tipo = np.where((g_dates.day_of_week==6)|g_festivo,'Domingo/festivo',np.where(g_dates.day_of_week==5,'Sábado','Lun–vie'))
g_segments = {'Año':g_dates.year.astype(str),'Tipo de día':g_tipo,'Nivel de demanda':pd.qcut(g_y.mean(axis=1),4,labels=['Q1','Q2','Q3','Q4']).astype(str)}
g_groups = [[k for k in g_keys if k not in modelos and '+O' not in k],[k for k in g_keys if '+O' in k],[k for k in g_keys if k in modelos]]
fig,axes = plt.subplots(3,3,figsize=(17,14),sharey=True,layout='constrained')
for g_row,g_group in enumerate(g_groups):
    for ax,(g_title,g_labels) in zip(axes[g_row],g_segments.items()):
        g_order = ['Lun–vie','Sábado','Domingo/festivo'] if g_title=='Tipo de día' else sorted(set(g_labels))
        for g_k in g_group:
            ax.plot(g_order,[g_daily[g_k][np.asarray(g_labels)==g].mean() for g in g_order],marker='o',label=g_names[g_k])
        ax.set(xlabel=g_title,ylabel='MASE');ax.legend(fontsize=7)
fig.savefig(g_figures/'segmentos_completos.png',dpi=160)
display(fig);plt.close(fig)
'''),md('''El error cambia entre años y niveles de demanda, por lo que una única media no resume todas las condiciones operativas. La comparación por tipo de día permite distinguir la dificultad de días laborales y no laborales. Los dos grupos de paneles separan las curvas para facilitar la lectura; utilizan los mismos segmentos y el mismo denominador MASE.''')]
    seeds=[md('''## Comparaciones entre semillas y tamaño del efecto

Se incluyen las 120 parejas de los catorce procedimientos, emparejadas por índice de semilla 1000–1004. Se reportan diferencia media de MASE, Shapiro–Wilk, t pareada y Wilcoxon exacto con Holm(91), y Cohen d_z con IC95 % por inversión de t no central. La t es un análisis de sensibilidad; cinco pares no acreditan normalidad porque Shapiro no rechace.'''),code('''g_pairs = pd.read_csv(g_result/'comparaciones_semillas.csv')
assert len(g_pairs)==len(g_keys)*(len(g_keys)-1)//2 and set(g_pairs.A)|set(g_pairs.B)==set(g_keys)
g_view = g_pairs.copy()
g_view.insert(0,'Comparación',[g_names[a]+' − '+g_names[b] for a,b in zip(g_view.A,g_view.B)])
g_view = g_view[['Comparación','diferencia_MASE','p_Shapiro','p_t_Holm','p_Wilcoxon_Holm','Cohen_dz','IC95_dz_inf','IC95_dz_sup']]
with pd.option_context('display.max_rows',100):
    display(g_view.style.format({c:'{:.4g}' for c in g_view.columns if c!='Comparación'}))
'''),md('''Las pruebas entre semillas y las temporales responden a fuentes de incertidumbre distintas. Con cinco pares, el mínimo p bilateral de Wilcoxon exacto sin empates es 0,0625; ninguna de las 91 comparaciones rechaza tras Holm. Cohen d_z mide la diferencia respecto de su dispersión entre inicializaciones, no un porcentaje de mejora ni la importancia práctica del error.'''),code('''from matplotlib.colors import TwoSlopeNorm
g_effect = np.full((len(g_keys),len(g_keys)),np.nan)
g_pt = np.full((len(g_keys),len(g_keys)),np.nan)
for g_pair in g_pairs.itertuples():
    g_i,g_j = g_keys.index(g_pair.A),g_keys.index(g_pair.B)
    g_effect[g_i,g_j]=g_pair.Cohen_dz;g_effect[g_j,g_i]=-g_pair.Cohen_dz
    g_pt[g_i,g_j]=g_pt[g_j,g_i]=g_pair.p_t_Holm
g_lim = np.nanmax(np.abs(g_effect))
fig,ax = plt.subplots(figsize=(14,11),layout='constrained')
g_im = ax.imshow(g_effect,cmap='RdBu_r',norm=TwoSlopeNorm(vmin=-g_lim,vcenter=0,vmax=g_lim))
ax.set(xticks=range(len(g_keys)),yticks=range(len(g_keys)),xticklabels=[g_names[k] for k in g_keys],yticklabels=[g_names[k] for k in g_keys])
ax.tick_params(axis='x',rotation=60,labelsize=8);ax.tick_params(axis='y',labelsize=9);ax.grid(False)
for g_i in range(len(g_keys)):
    for g_j in range(len(g_keys)):
        if g_i!=g_j:
            g_v=g_effect[g_i,g_j]
            ax.text(g_j,g_i,f'{g_v:.1f}'+('*' if g_pt[g_i,g_j]<.05 else ''),ha='center',va='center',fontsize=8,color='white' if abs(g_v)>.55*g_lim else '#222222')
fig.colorbar(g_im,ax=ax,label='Cohen d_z · fila menos columna; * t pareada Holm(91) < 0,05')
fig.savefig(g_figures/'efectos_semillas_completos.png',dpi=160)
display(fig);plt.close(fig)
'''),md('''Los valores negativos favorecen al procedimiento de la fila. Los asteriscos identifican rechazos de la t pareada con Holm, sujetos a sus supuestos, y no alteran el resultado de Wilcoxon. Un efecto estandarizado grande puede coexistir con una diferencia absoluta pequeña cuando la variación entre semillas es reducida; el margen práctico se evalúa por separado.'''),md('''### Distribución de diferencias entre inicializaciones

Los gráficos Q–Q comparan cada procedimiento con el control numérico de TaTS/fusión. Son trece contrastes diagnósticos sobre cinco diferencias; la tabla anterior conserva las 91 parejas completas.'''),code('''with np.load(g_result/'perdidas_y_bootstrap.npz',allow_pickle=False) as g_archive:
    g_seed_matrix=g_archive['semillas'];g_seed_keys=g_archive['nombres'].tolist()[:len(g_keys)]
g_reference='adj_modelo6_20261112_1035__N'
fig,axes=plt.subplots(int(np.ceil((len(g_keys)-1)/3)),3,figsize=(16,18),layout='constrained')
for ax,g_k in zip(axes.flat,[k for k in g_keys if k!=g_reference]):
    g_d=g_seed_matrix[:,g_seed_keys.index(g_k)]-g_seed_matrix[:,g_seed_keys.index(g_reference)]
    stats.probplot(g_d,dist='norm',plot=ax)
    ax.set(title=g_names[g_k],xlabel='Cuantil normal teórico',ylabel='MASE − control N')
for g_ax in list(axes.flat)[len(g_keys)-1:]:g_ax.set_visible(False)
display(fig);plt.close(fig)
'''),md('''Cinco puntos ofrecen poca resolución para detectar desviaciones de normalidad. Estos gráficos describen las diferencias de inicialización y no sustituyen la inferencia temporal por bloques ni justifican elegir una prueba después de observar su resultado.''')]
    mcs=[md('''### Post hoc completo de Nemenyi

La matriz presenta las 91 comparaciones post hoc entre los catorce procedimientos. Nemenyi ya controla su familia de comparaciones mediante la distribución del rango studentizado; no se añade Holm a esos mismos valores p.'''),code('''g_nemenyi=pd.read_csv(g_result/'nemenyi.csv')
assert len(g_nemenyi)==len(g_keys)*(len(g_keys)-1)//2
g_matrix=np.full((len(g_keys),len(g_keys)),np.nan)
for g_pair in g_nemenyi.itertuples():
    g_i,g_j=g_keys.index(g_pair.A),g_keys.index(g_pair.B)
    g_matrix[g_i,g_j]=g_matrix[g_j,g_i]=g_pair.p_Nemenyi
fig,ax=plt.subplots(figsize=(14,11),layout='constrained')
g_im=ax.imshow(g_matrix,cmap='viridis_r',vmin=0,vmax=1)
ax.set(xticks=range(len(g_keys)),yticks=range(len(g_keys)),xticklabels=[g_names[k] for k in g_keys],yticklabels=[g_names[k] for k in g_keys])
ax.tick_params(axis='x',rotation=60,labelsize=8);ax.tick_params(axis='y',labelsize=9);ax.grid(False)
for g_i in range(len(g_keys)):
    for g_j in range(len(g_keys)):
        if g_i!=g_j:
            ax.text(g_j,g_i,f'{g_matrix[g_i,g_j]:.3f}'+('*' if g_matrix[g_i,g_j]<.05 else ''),ha='center',va='center',fontsize=7,color='white' if g_matrix[g_i,g_j]>.5 else '#222222')
fig.colorbar(g_im,ax=ax,label='p Nemenyi · * < 0,05')
fig.savefig(g_figures/'nemenyi_completo.png',dpi=160)
display(fig);plt.close(fig)
'''),md('''El rechazo global de Friedman no se extiende automáticamente a todos los pares. Nemenyi exige separaciones de rango grandes con cinco bloques y catorce procedimientos. Las celdas sin rechazo no acreditan equivalencia; se conservan los contrastes pareados y sus tamaños del efecto para valorar magnitud e incertidumbre.'''),code('''g_mcs=pd.read_csv(g_result/'mcs.csv').sort_values(['retenido_90','paso'],ascending=[False,True])
assert len(g_mcs)==len(g_benchmark) and set(g_mcs.clave)==set(g_benchmark.index)
g_mcs['Estado']=np.where(g_mcs.retenido_90,'Retenido','Excluido')
display(g_mcs[['modelo','Estado','paso','p_crudo','p_MCS']].rename(columns={'modelo':'Procedimiento','paso':'Paso','p_crudo':'p del paso','p_MCS':'p MCS'}).style.format({'p del paso':'{:.4g}','p MCS':'{:.4g}'}))
'''),md('''El procedimiento evaluó los catorce modelos y las dos referencias. Retiene TaTS con texto oficial y su control numérico en el MCS del 90 %; los demás figuran como excluidos con su paso y valor p. Mostrar dos retenidos no significa que solo se hayan comparado dos modelos, ni establece un ganador único o la necesidad del texto.''')]
    variance=[md('''## Variación entre fechas e inicializaciones

La pérdida diaria de las cinco semillas se descompone, para los catorce procedimientos, en efectos descriptivos de fecha, inicialización e interacción. No se supone independencia entre días ni se interpreta la descomposición como ANOVA inferencial.'''),code('''g_components=[]
for g_k in g_keys:
    g_values=np.abs(g_pred[g_k]-g_y[None]).mean(axis=2)/g_q
    g_mean=g_values.mean()
    g_components.append({'Procedimiento':g_names[g_k],
        'Inicialización':np.mean((g_values.mean(axis=1)-g_mean)**2),
        'Fecha':np.mean((g_values.mean(axis=0)-g_mean)**2),
        'Interacción':np.mean((g_values-g_values.mean(axis=1)[:,None]-g_values.mean(axis=0)[None,:]+g_mean)**2)})
g_components=pd.DataFrame(g_components).set_index('Procedimiento')
fig,ax=plt.subplots(figsize=(14,6))
g_components.div(g_components.sum(axis=1),axis=0).plot.bar(stacked=True,ax=ax,rot=45)
ax.set(xlabel='',ylabel='Fracción de variación descriptiva');ax.tick_params(axis='x',labelsize=8)
fig.tight_layout();display(fig);plt.close(fig)
'''),md('''La componente de fecha recoge cambios de dificultad compartidos por las inicializaciones. La interacción representa cambios que no se explican por efectos aditivos de fecha y semilla. Estas fracciones complementan los intervalos de error; no estiman por sí solas la incertidumbre ante otros periodos o nuevas series.''')]
    outliers=[md('''## Impacto de valores atípicos sobre el error e identidad de prueba

Los umbrales de Tukey se fijan sobre log(y/y_lag168) en 2000–2017 y se aplican sin modificación a prueba. Se comparan errores en horas señaladas y restantes para los catorce procedimientos; ninguna hora se elimina del benchmark.'''),code('''g_series=pd.read_parquet(ROOT/'data/processed/panel_horario.parquet',columns=['y_demanda_real_kwh']).y_demanda_real_kwh
g_development=g_series.loc[g_series.index<'2018-01-01']
g_log=np.log(g_development/g_development.shift(168)).dropna()
g_q1,g_q3=g_log.quantile([.25,.75]);g_iqr=g_q3-g_q1
g_times=pd.DatetimeIndex([d+pd.Timedelta(hours=h) for d in g_dates for h in range(24)])
g_ref=g_series.reindex(g_times-pd.Timedelta(hours=168)).to_numpy().reshape(g_y.shape)
g_ratio=np.log(g_y/g_ref)
g_outlier=(g_ratio<g_q1-1.5*g_iqr)|(g_ratio>g_q3+1.5*g_iqr)
g_impact=[]
for g_k in g_keys:
    g_e=np.abs(g_pred[g_k]-g_y[None]).mean(axis=0)
    g_impact.append({'Procedimiento':g_names[g_k],'MAE atípicos (kWh)':g_e[g_outlier].mean(),
        'MAE restantes (kWh)':g_e[~g_outlier].mean(),'Error en atípicos (%)':100*g_e[g_outlier].sum()/g_e.sum(),
        'Cambio MAE al excluir (%)':100*(g_e[~g_outlier].mean()/g_e.mean()-1)})
g_impact=pd.DataFrame(g_impact)
display(g_impact.round(3))
print(f'Horas señaladas: {g_outlier.sum()} de {g_outlier.size} ({100*g_outlier.mean():.2f} %)')
g_digest=hashlib.sha256();g_digest.update(b'SIN-demand-test-v1\\n')
g_digest.update(np.asarray(origen,dtype='<i8').tobytes());g_digest.update(np.asarray(g_y,dtype='<f8').tobytes())
assert g_digest.hexdigest()=='595f00dd186280542cfd7fed9a580ebe1b19c8cc4e0e742ef12e2e6d2eb0f49a'
'''),md('''Los umbrales señalan el 7,19 % de las horas. La tabla muestra su contribución al error y cuánto cambiaría el MAE al excluirlas; es un análisis de sensibilidad, no una propuesta de depuración de prueba. El efecto varía entre procedimientos y no permite atribuir toda la brecha de generalización a atípicos.

La huella SHA-256 de orígenes y etiquetas de prueba es `595f00dd186280542cfd7fed9a580ebe1b19c8cc4e0e742ef12e2e6d2eb0f49a`. Identifica su contenido, sin acreditar por sí sola la fecha de congelación.''')]
    cost=[md('''## Coste de inferencia medido y coste de ajuste registrado

Se presentan parámetros, duración del ajuste final y épocas para los catorce procedimientos. La duración es la mediana de cinco semillas; los dispositivos y protocolos distintos impiden atribuir sus diferencias solo a la eficiencia arquitectónica. La latencia CPU se midió para las cuatro familias de demanda/calendario con un hilo, batch uno, 20 calentamientos y 200 mediciones por checkpoint, sin preparación ni carga de datos.'''),code('''g_cost=g_meta.groupby('clave').agg(Parametros=('parametros','first'),Ajuste_s_mediana=('segundos','median'),Epocas_mediana=('epocas','median'),Dispositivo=('dispositivo','first')).reindex(g_keys)
g_latency=pd.read_csv(ROOT/'results/jbook/modelos_actuales/latencia_inferencia.csv')
g_cost['Latencia_CPU_mediana_ms']=g_latency.groupby('modelo').mediana_ms.median().reindex(g_keys)
g_cost['Latencia_CPU_p95_ms']=g_latency.groupby('modelo').p95_ms.median().reindex(g_keys)
g_cost.index=[g_names[k] for k in g_cost.index]
display(g_cost.style.format(precision=3,na_rep='Sin medición'))
'''),md('''Parámetros y tiempo de ajuste caracterizan el coste registrado, mientras la latencia aislada describe únicamente el forward en la CPU medida. No se dispone de una medición equivalente de latencia para los diez procedimientos ampliados ni del coste completo de codificación del texto. Estos campos permanecen sin medir y no se infieren del número de parámetros.''')]
    coverage=[md('''## Cobertura de los diagnósticos

La tabla distingue resultados presentes de diagnósticos que requieren inferencia adicional. Un resultado no disponible no se representa con cero ni se reconstruye con pérdidas de otro tipo. TGForecaster incorpora las variantes numérica y con texto oficial.'''),code('''g_coverage=[]
for g_k in g_keys:
    g_row=g_meta[g_meta.clave.eq(g_k)]
    g_internal=g_row[['MAE_ajuste_normalizado','MAE_parada_normalizado']].notna().all(axis=1).all() or g_row[['MASE_ajuste','MASE_parada']].notna().all(axis=1).all()
    g_coverage.append({'Procedimiento':g_names[g_k],'Prueba/curvas/segmentos/contrastes':'Completo · 5 semillas',
        'Piloto independiente':'Disponible' if g_k in set(g_pilots.clave) else 'No disponible',
        'Métricas ajuste/parada':'Disponibles' if g_internal else 'Solo curvas de pérdida',
        'Importancia y latencia CPU':'Medidas' if g_k in modelos else 'Pendientes'})

g_coverage=pd.DataFrame(g_coverage)
display(g_coverage)
g_coverage.to_csv(g_result/'cobertura_diagnosticos.csv',index=False)
'''),md('''Los catorce procedimientos completados están representados en el benchmark, las curvas, los perfiles observado–predicho y los contrastes generales. La importancia por perturbación y la latencia conservan el alcance de las cuatro familias para las que fueron medidas. El resultado no permite concluir que clima, hidrología, precio o texto carezcan de importancia en las otras arquitecturas.''')]
    cells=copy.deepcopy(cells)
    def replace(start,end,insert):
        nonlocal cells
        a=next(i for i,c in enumerate(cells) if c.cell_type=='markdown' and start in c.source)
        b=next(i for i in range(a+1,len(cells)) if cells[i].cell_type=='markdown' and end in cells[i].source)
        cells[a:b]=insert
    replace('## Validación y generalización','## Error por horizonte',general+observed)
    # La selección/piloto precede a las curvas; inicializar aquí los datos globales.
    setup_index=next(i for i,c in enumerate(cells) if c.cell_type=='code' and c.source.startswith('# Predicciones de los catorce'))
    setup_cell=cells.pop(setup_index)
    cells.insert(3,setup_cell)
    pilot_code=next(i for i,c in enumerate(cells) if c.cell_type=='code' and 'piloto.loc[' in c.source and 'permutacion' in c.source)
    cells[pilot_code]=code('''fig,axes=plt.subplots(1,3,figsize=(18,5))
g_pilot_keys=[k for k in g_keys if k in set(g_pilots.clave)]
for g_k in g_pilot_keys:
    g_g=g_pilots[g_pilots.clave.eq(g_k)].groupby('fold').MASE
    axes[0].errorbar(g_g.mean().index,g_g.mean(),yerr=g_g.std(),marker='o',capsize=3,label=g_names[g_k])
    g_by_seed=g_pilots[g_pilots.clave.eq(g_k)].groupby('semilla').MASE.mean()
    axes[1].scatter(np.full(len(g_by_seed),g_pilot_keys.index(g_k)),g_by_seed,s=25)
axes[0].set(xlabel='Año externo',ylabel='MASE · media ± DE entre semillas',xticks=[2018,2019,2020,2021]);axes[0].legend(fontsize=8)
axes[1].set(xticks=range(len(g_pilot_keys)),xticklabels=[g_names[k] for k in g_pilot_keys],ylabel='MASE medio de cuatro años por semilla')
axes[1].tick_params(axis='x',rotation=35,labelsize=8)
for g_k in modelos:
    g_perm=piloto[piloto.modelo.eq(g_k)].groupby('fold')[['MASE','MASE_demanda_permutada']].mean()
    axes[2].plot(g_perm.index,g_perm.MASE_demanda_permutada/g_perm.MASE,marker='o',label=etiquetas[g_k])
axes[2].axhline(1,color='grey',ls='--')
axes[2].set(xlabel='Año externo',ylabel='MASE permutado / MASE original',xticks=[2018,2019,2020,2021]);axes[2].legend(fontsize=8)
fig.tight_layout();display(fig);plt.close(fig)
''')
    cells[pilot_code+1]=md('''Los seis procedimientos con piloto muestran diferencias entre años e inicializaciones. El panel central agrega primero los cuatro años dentro de cada semilla; los veinte resultados anuales no son veinte semillas independientes. El panel derecho conserva el control de demanda permutada disponible para las cuatro familias de demanda/calendario: razones mayores que uno indican deterioro al romper esa información. Los otros ocho procedimientos no conservan un piloto independiente y se evalúan con sus cinco realizaciones finales.''')
    selection=next(i for i,c in enumerate(cells) if c.cell_type=='markdown' and '## Entrenamiento, selección y precisión del piloto' in c.source)
    cells[selection]=md('''## Entrenamiento, selección y precisión del piloto

La selección utiliza validación cronológica y una reserva interna para parada. La tabla reúne los presupuestos de los catorce procedimientos completados, las etapas de búsqueda y las cinco semillas finales por procedimiento. Los candidatos de las cuatro familias de demanda/calendario se cuentan sobre todos los registros de filtro conservados; los ajustes de confirmación incluyen sus rondas registradas. Las arquitecturas y los hiperparámetros seleccionados se detallan en sus fichas.

PatchTST y N-HiTS numéricos utilizan tres semillas en búsqueda, cinco finalistas y confirmación en 2019–2021; su criterio de selección promedia esos tres años. Los otros diez procedimientos ampliados utilizan una semilla en búsqueda, cinco finalistas y media de 2018–2021, que incorpora el filtro de 2018 y la confirmación de 2019–2021. Las cuatro familias de demanda/calendario confirman con dos semillas y media de cuatro años externos. Los presupuestos distintos limitan la atribución de diferencias exclusivamente a la arquitectura.''')
    cells[selection+1:selection+1]=[code('''g_design=pd.read_csv(g_result/'diseno_seleccion.csv').set_index('clave').reindex(g_keys)
assert len(g_design)==len(g_keys) and g_design.notna().all().all()
g_design.index=[g_names[k] for k in g_keys]
display(g_design.rename(columns={'candidatos_filtro':'Candidatos de filtro','filtro':'Filtro',
    'confirmacion':'Confirmación','piloto':'Piloto independiente','final':'Final',
    'max_epocas':'Épocas máx. filtro / resto','paciencia':'Paciencia'}))
'''),md('''Los procedimientos con tres semillas en búsqueda reciben más ajustes por candidato que los paquetes con una. El máximo de épocas tampoco es común a todas las familias. El piloto independiente existe para seis procedimientos; las cinco realizaciones finales no sustituyen esa etapa. TGForecaster utiliza cinco semillas para cada una de sus variantes N y N+O.''')]
    dimension=[md(r'''### Dimensionamiento y precisión del piloto

Se incluyen los catorce procedimientos en el resumen de disponibilidad y se dimensionan las quince parejas de los seis que conservan un piloto independiente. Cada semilla 100–104 se resume promediando primero los cuatro años externos 2018–2021; los veinte registros semilla–año no se consideran veinte inicializaciones independientes. La tabla presenta media de MASE, desviación estándar (DE) y coeficiente de variación (CV). No se utilizan errores de prueba para suplir pilotos ausentes.'''),code('''g_pilot_seed=g_pilots.groupby(['semilla','clave']).MASE.mean().unstack('clave')
assert g_pilot_seed.shape==(5,6) and g_pilot_seed.notna().all().all()
assert g_pilot_seed.index.tolist()==[100,101,102,103,104]
g_pilot_summary=g_pilot_seed.agg(['mean','std','count']).T.rename(columns={'mean':'MASE medio','std':'DE','count':'Semillas piloto'}).reindex(g_keys)
g_pilot_summary['CV (%)']=100*g_pilot_summary['DE']/g_pilot_summary['MASE medio']
g_pilot_summary['Disponibilidad']=np.where(g_pilot_summary['Semillas piloto'].notna(),'Piloto independiente','Sin piloto independiente')
g_pilot_summary.index=[g_names[k] for k in g_keys]
display(g_pilot_summary[['Disponibilidad','Semillas piloto','MASE medio','DE','CV (%)']].style.format(precision=5,na_rep='No estimable'))
g_pilot_summary.to_csv(g_result/'resumen_piloto_integrado.csv',index_label='Procedimiento')
'''),md(r'''El menor MASE medio del piloto corresponde a MLP (0,542), seguido de PatchTST numérico (0,569). N-HiTS registra 0,827. La dispersión entre semillas no sigue necesariamente ese orden; un error medio mayor puede ser estable entre inicializaciones.

Para expresar las diferencias en una escala relativa común se conserva como referencia la media piloto de MLP, obtenida exclusivamente de validación. Con $d_s=(M_{A,s}-M_{B,s})/\overline M_{\mathrm{MLP,piloto}}$, margen $\delta=0,03$, $\alpha=0,05$ y potencia objetivo $0,80$, se calcula

$$R\approx\left\lceil\left[\frac{(z_{1-\alpha/2}+z_{0,80})\widehat\sigma_d}{\delta}\right]^2\right\rceil,\qquad
\mathrm{MDD}_5=(z_{1-\alpha/2}+z_{0,80})\frac{\widehat\sigma_d}{\sqrt5}.$$

Se informa además $\delta/\widehat\sigma_d$ y una planificación conservadora con Bonferroni para las quince parejas. El mínimo de diseño es cinco semillas. Es una aproximación retrospectiva de variación entre inicializaciones, condicionada a los mismos cuatro años; no estima la incertidumbre de nuevas fechas ni acredita la potencia alcanzada por los ajustes ya ejecutados.'''),code('''from itertools import combinations
g_delta=.03;g_alpha=.05;g_power=.80
g_pilot_reference=float(g_pilot_seed['mlp'].mean())
g_pilot_pairs=list(combinations(g_pilot_seed.columns,2))
assert len(g_pilot_pairs)==15
g_z=stats.norm.ppf(1-g_alpha/2)+stats.norm.ppf(g_power)
g_z_bonf=stats.norm.ppf(1-g_alpha/(2*len(g_pilot_pairs)))+stats.norm.ppf(g_power)
g_dimension=[]
for g_a,g_b in g_pilot_pairs:
    g_difference=(g_pilot_seed[g_a]-g_pilot_seed[g_b])/g_pilot_reference
    g_sd=float(g_difference.std(ddof=1))
    g_dimension.append({'A':g_a,'B':g_b,'Comparación':g_names[g_a]+' − '+g_names[g_b],
        'DE diferencia relativa':g_sd,'δ / DE':g_delta/g_sd if g_sd>0 else np.inf,
        'R individual':max(5,int(np.ceil((g_z*g_sd/g_delta)**2))),
        'R Bonferroni(15)':max(5,int(np.ceil((g_z_bonf*g_sd/g_delta)**2))),
        'MDD con 5 (% referencia piloto)':100*g_z*g_sd/np.sqrt(5)})
g_dimension=pd.DataFrame(g_dimension)
display(g_dimension.drop(columns=['A','B']).round(5))
g_dimension.to_csv(g_result/'dimensionamiento_piloto_integrado.csv',index=False)
'''),md('''La aproximación individual exige entre cinco y 45 semillas por procedimiento, según la pareja; el máximo conservador para las quince comparaciones es 82. Las cinco semillas ejecutadas no acreditan potencia de 0,80 para el margen del 3 % en todas las parejas. En los ocho procedimientos sin piloto independiente, el número requerido permanece sin estimar.

Los registros de PatchTST y N-HiTS incluyen además un diagnóstico temporal por bootstrap de bloques de 14 días frente al ingenuo semanal de validación. Sus estimaciones límite son 0,0984 y 0,0843 y no determinan un número final de semillas. Ese diagnóstico conserva incertidumbre temporal y no equivale al cálculo anterior entre inicializaciones.'''),code('''g_recorded_power=pd.read_csv(g_result/'dimensionamiento_temporal_registrado.csv')
g_recorded_power['Procedimiento']=g_recorded_power.clave.map(g_names)
display(g_recorded_power[['Procedimiento','referencia','MASE_referencia','margen_relativo','potencia_limite_temporal','n_final','estado']].rename(columns={
    'referencia':'Referencia temporal','MASE_referencia':'MASE referencia','margen_relativo':'δ relativo',
    'potencia_limite_temporal':'Estimación límite temporal','n_final':'R final registrado','estado':'Estado'}).style.format(precision=4,na_rep='No determinado'))
'''),md('''Ambos registros indican precisión temporal insuficiente para la alternativa evaluada. Aumentar semillas reduce la variación de inicialización, pero no elimina por sí solo la incertidumbre debida a los días compartidos. La planificación normal y el diagnóstico temporal conservan sus referencias e hipótesis explícitas.

[Diseño de selección](_static/resultados/evaluacion_e2/diseno_seleccion.csv) · [Resumen piloto integrado](_static/resultados/evaluacion_e2/resumen_piloto_integrado.csv) · [Quince comparaciones de dimensionamiento](_static/resultados/evaluacion_e2/dimensionamiento_piloto_integrado.csv) · [Diagnóstico temporal registrado](_static/resultados/evaluacion_e2/dimensionamiento_temporal_registrado.csv).''')]
    replace('### Piloto y dimensionamiento','## Curvas de aprendizaje',dimension)
    temporal=[md('''## Comparaciones temporales de todos los procedimientos

La matriz incluye las 120 parejas de los catorce procedimientos y las dos referencias. Cada celda muestra p de Diebold–Mariano con corrección HLN y Holm(91), calculado sobre diferenciales diarios con HAC de 13 rezagos. Las diferencias e intervalos temporales completos se conservan en la tabla descargable.'''),code('''g_temporal=pd.read_csv(g_result/'comparaciones_temporales.csv')
g_all=g_benchmark.index.tolist()
assert len(g_temporal)==len(g_all)*(len(g_all)-1)//2
g_pdm=np.full((len(g_all),len(g_all)),np.nan)
for g_pair in g_temporal.itertuples():
    g_i,g_j=g_all.index(g_pair.A),g_all.index(g_pair.B)
    g_pdm[g_i,g_j]=g_pdm[g_j,g_i]=g_pair.p_DM_Holm
fig,ax=plt.subplots(figsize=(15,13),layout='constrained')
g_im=ax.imshow(g_pdm,cmap='viridis_r',vmin=0,vmax=1)
ax.set(xticks=range(len(g_all)),yticks=range(len(g_all)),xticklabels=[g_names[k] for k in g_all],yticklabels=[g_names[k] for k in g_all])
ax.tick_params(axis='x',rotation=60,labelsize=8);ax.tick_params(axis='y',labelsize=9);ax.grid(False)
for g_i in range(len(g_all)):
    for g_j in range(len(g_all)):
        if g_i!=g_j:
            g_v=g_pdm[g_i,g_j]
            ax.text(g_j,g_i,('<.001' if g_v<.001 else f'{g_v:.3f}')+('*' if g_v<.05 else ''),ha='center',va='center',fontsize=7,color='white' if g_v>.5 else '#222222')
fig.colorbar(g_im,ax=ax,label='p DM–HLN · Holm(120); * < 0,05')
fig.savefig(g_figures/'dm_completo.png',dpi=160)
display(fig);plt.close(fig)
'''),md('''Los rechazos son específicos de cada pareja y no convierten el orden de las medias en un ranking de diferencias significativas. Un valor p elevado tampoco establece equivalencia. Los contrastes temporales condicionan las cinco semillas ejecutadas; su resultado puede diferir de las pruebas entre inicializaciones porque cambia la unidad de análisis.''')]
    a=next(i for i,c in enumerate(cells) if c.cell_type=='markdown' and '## Validación y generalización' in c.source)
    cells[a:a]=temporal
    replace('## Segmentos de evaluación','## Aporte de texto oficial',segments)
    replace('## Comparaciones entre semillas','## Friedman, post hoc',seeds)
    # Preservar Friedman/rangos de doce modelos, sustituir la tabla filtrada del MCS.
    a=next(i for i,c in enumerate(cells) if c.cell_type=='code' and 'mcs.loc[' in c.source)
    cells[a:a+2]=mcs
    replace('## Variación entre fechas','## Impacto de valores atípicos',variance)
    replace('## Impacto de valores atípicos','## Coste de inferencia',outliers)
    replace('## Coste de inferencia','(importancia-variables)=',cost)
    a=next(i for i,c in enumerate(cells) if c.cell_type=='markdown' and '## Discusión y alcance del Entregable 2' in c.source)
    cells[a:a]=coverage
    cells[0].source=cells[0].source.replace('Los diagnósticos complementarios de observado–predicho, segmentos e importancia mantienen su alcance en las cuatro familias con demanda y calendario; las comparaciones estadísticas generales abarcan los catorce procedimientos disponibles.','Los diagnósticos de precisión, generalización, observado–predicho, segmentos, atípicos y contrastes abarcan los catorce procedimientos disponibles. La importancia por perturbación y la latencia CPU se presentan con su cobertura de medición explícita.')
    cells[-1].source+='\n\n[Metadatos de los catorce procedimientos](_static/resultados/evaluacion_e2/metadatos_procedimientos.csv) · [Pilotos disponibles](_static/resultados/evaluacion_e2/pilotos_disponibles.csv) · [Cobertura de diagnósticos](_static/resultados/evaluacion_e2/cobertura_diagnosticos.csv).'
    g_n_t=int((pd.read_csv(Path(root)/'results/evaluacion_e2/comparaciones_semillas.csv').p_t_Holm<.05).sum())
    g_n_nemenyi=int((pd.read_csv(Path(root)/'results/evaluacion_e2/nemenyi.csv').p_Nemenyi<.05).sum())
    g_n_dm=int((pd.read_csv(Path(root)/'results/evaluacion_e2/comparaciones_temporales.csv').p_DM_Holm<.05).sum())
    for cell in cells:
        if cell.cell_type!='markdown':continue
        cell.source=cell.source.replace('Los rechazos son específicos de cada pareja y no convierten',f'DM–HLN con Holm rechaza en {g_n_dm} de las 91 parejas. Los rechazos son específicos de cada pareja y no convierten')
        cell.source=cell.source.replace('Los valores negativos favorecen al procedimiento de la fila.',f'La t pareada con Holm rechaza en {g_n_t} de las 91 parejas. Los valores negativos favorecen al procedimiento de la fila.')
        cell.source=cell.source.replace('El rechazo global de Friedman no se extiende automáticamente a todos los pares.',f'Nemenyi rechaza en {g_n_nemenyi} de las 91 parejas: las parejas específicas se identifican en la matriz completa y su tabla descargable. El rechazo global de Friedman no se extiende automáticamente a todos los pares.')
    from precision_parada_jbook import integrar_precision_parada
    cells=integrar_precision_parada(root,cells)
    from revision_integral_jbook import corregir_capitulo10
    return corregir_capitulo10(root,cells)


def armonizar_secciones(root):
    """Corrige alcance vigente en introducción, pregunta, diseño y reproducción."""
    root=Path(root)
    p=root/'book/intro.md'
    text=p.read_text(encoding='utf-8')
    text=text.replace('junto con el benchmark numérico de MLP, LSTM y CNN1D y las referencias ARX y estacionales',
        'junto con catorce procedimientos de pronóstico: redes densas, recurrentes, convolucionales, multiescala y de atención, variantes con texto oficial y referencias ARX y estacionales')
    text=text.replace('junto con doce procedimientos de pronóstico','junto con catorce procedimientos de pronóstico')
    p.write_text(text,encoding='utf-8')
    for name in ['01_datos','06_covariables','08_literatura','09_validacion','11_sintesis','12_reproducibilidad']:
        p=root/'book'/(name+'.ipynb');notebook=nb.read(p,4)
        for cell in notebook.cells:
            if cell.cell_type!='markdown':continue
            text=cell.source
            if name=='01_datos':
                text=text.replace('El estudio contempla la integración posterior de información documental.','El estudio compara procedimientos numéricos y variantes que incorporan documentos oficiales disponibles antes del origen.')
                if '**Pregunta de investigación:**' in text:
                    text='''**Tipo de problema:** series de tiempo, con pronóstico directo de 24 pasos, covariables y rezagos del objetivo. **Pregunta de investigación:** ¿qué precisión y estabilidad presentan las arquitecturas numéricas para anticipar la demanda horaria del SIN colombiano durante las siguientes 24 horas, y qué contribución incremental aporta el texto oficial disponible antes del origen? La evaluación compara catorce procedimientos completados y dos referencias estacionales; TGForecaster participa con sus variantes N y N+O.'''
            if name=='06_covariables':
                text=text.replace('La comparación con texto empleará el mismo conjunto numérico en todas sus variantes','El diseño de comparación con texto establece el mismo conjunto numérico para sus variantes')
            if name=='08_literatura':
                text=text.replace('La MLP presenta la mayor capacidad parametrizada.','Entre las cuatro familias de demanda/calendario, la MLP presenta la mayor capacidad parametrizada.')
                text=text.replace('Los resultados actuales establecen la referencia numérica y no prueban todavía esa contribución del texto.','La evaluación disponible contrasta el texto oficial en TaTS y fusión tardía frente a un control numérico compartido; la variante adicional con prensa y las ablaciones de contenido siguen pendientes.')
                if text.startswith('### Extensiones respaldadas por la revisión') and 'El capítulo de arquitecturas desarrolla PatchTST' not in text:
                    text+='''\n\nEl capítulo de arquitecturas desarrolla PatchTST y su control numérico, N-HiTS, TFT, TCN e iTransformer, además de TaTS y fusión tardía con documentos oficiales. Cada ficha distingue el mecanismo implementado de su artículo orientador y enlaza el pipeline. Estas incorporaciones no alteran retrospectivamente el denominador de cinco estudios empíricos utilizado para contar comparadores. TGForecaster se evalúa con sus variantes N y N+O.'''
            if name=='09_validacion':
                text=text.replace('DM–HLN utiliza covarianza HAC de 14 rezagos diarios.','La evaluación integrada utiliza DM–HLN con covarianza HAC Bartlett de 13 rezagos diarios y h efectivo de un origen no solapado.')
                text=text.replace('La selección utiliza confirmación de validación con dos semillas y cuatro años externos.','Las cuatro familias de demanda/calendario utilizan confirmación con dos semillas y cuatro años externos. PatchTST y N-HiTS numéricos utilizan tres semillas y la media de 2019–2021; los diez procedimientos ampliados restantes utilizan una semilla y cuatro años externos. Los presupuestos y periodos se conservan como atributos del procedimiento.')
                if text.startswith('## Unidad de evaluación, bloques y comparaciones') and 'Se comparan las tres redes entre sí,' in text:
                    a=text.index('Se comparan las tres redes entre sí,')
                    b=text.index('\n\nEl dimensionamiento',a)
                    text=text[:a]+'''Se conservan 120 parejas temporales entre catorce procedimientos y dos referencias, y 91 parejas entre las doce series de cinco semillas. Holm se aplica por tipo de contraste a su familia completa. Los tres contrastes específicos de texto forman una familia exploratoria de tres parejas. Las diferencias, intervalos, tamaños del efecto y valores p se reportan junto con sus supuestos. El margen relativo es ±3 %: en los contrastes globales se expresa respecto del procedimiento de referencia de la pareja; en los contrastes textuales, respecto del control numérico N. No se infiere equivalencia por ausencia de significación.'''+text[b:]
                if text.startswith('## Ejecución y alcance estadístico') and 'Las configuraciones se seleccionan' in text:
                    a=text.index('Las configuraciones se seleccionan')
                    b=text.index('\n\nMASE es',a)
                    text=text[:a]+'''Las doce configuraciones conservan cinco realizaciones finales cada una. MLP, ARX, LSTM y CNN1D utilizan máximo 30 épocas en filtro y 60 en las demás etapas; los procedimientos ampliados, máximo 200. La paciencia efectiva es seis para las cuatro familias y los paquetes ampliados, y diez para PatchTST/N-HiTS numéricos del repositorio. Se restaura el mejor estado de la reserva interna. Los pilotos independientes están disponibles para seis procedimientos; no se sustituyen con las cinco realizaciones de prueba. El preprocesamiento se ajusta al tramo de optimización de cada pliegue.'''+text[b:]
                text=text.replace('Las etiquetas coinciden en las 20 realizaciones seleccionadas.','Las 60 realizaciones utilizan los mismos 1.461 orígenes y 24 horizontes; la consolidación verifica la alineación y registra diferencias de redondeo float32 en los paquetes. La evaluación utiliza etiquetas canónicas comunes.')
                text=text.replace('La preparación conserva la malla horaria y una máscara para cada entrada ausente.','La preparación conserva la malla horaria. Los procedimientos numéricos del repositorio utilizan 30 máscaras de ausencia; los paquetes ampliados utilizan tres por bloque. Esta diferencia se declara al interpretar el benchmark.')
                if text.startswith('## Unidad de evaluación, bloques y comparaciones') and 'precisión retrospectiva de parada' not in text:
                    text+='\n\nPara los ocho procedimientos sin piloto externo, la precisión retrospectiva de parada utiliza el MASE de los cinco checkpoints finales en la reserva interna de 803 días anterior a 2022. Se estiman dispersión, intervalos y dimensionamiento con δ=3 % respecto del control N en esa reserva. La selección de época utiliza el mismo tramo, por lo que el análisis no equivale a validación externa ni a planificación previa; los errores de prueba no intervienen en ese dimensionamiento.'
            if name=='12_reproducibilidad' and text.startswith('## Procedencia de modelos y reconstrucción'):
                text='''## Procedencia de modelos y reconstrucción

Las cuatro familias de demanda/calendario conservan el commit d1ae9a4e09dd1dead4af1329f16aa3e8d0f2c3d3 y su [registro de artefactos](_static/modelos/modelos.json); el [índice SHA-256](_static/modelos/indice_artefactos_v3_v4.csv) identifica 2.304 archivos. Las arquitecturas ampliadas proceden del commit 66111003f387b8452b1fc690dbc7fc503d5570df y de los paquetes consolidados. El inventario conserva pesos, escaladores, historias, predicciones y métricas por semilla; el control N compartido se cuenta una vez. TFT utiliza el paquete seleccionado por rigor del procedimiento y fidelidad arquitectónica.

La evaluación integra 70 realizaciones finales de catorce procedimientos, con cinco semillas cada uno, y dos referencias deterministas. Las métricas se reconstruyen con predicciones conservadas y orígenes alineados. [Metadatos de ajuste y validación](_static/resultados/evaluacion_e2/metadatos_procedimientos.csv) y [cobertura de diagnósticos](_static/resultados/evaluacion_e2/cobertura_diagnosticos.csv) identifican el alcance de cada análisis. TGForecaster conserva cinco realizaciones por variante.

Para reconstruir la evaluación integrada se ejecuta `python scripts/reorganizar_capitulo10.py`; después, `jupyter-book build book --warningiserror --keep-going` y `python scripts/verificar_jbook.py`. El generador completo `scripts/crear_jbook.py` integra la misma ampliación al finalizar. Los datos de predicción y las entradas documentales deben conservar su inventario de procedencia. La reproducción del análisis documental usa CPU; la inferencia y el entrenamiento requieren PyTorch compatible con CPU, CUDA o ROCm según el procedimiento y dispositivo. La reconstrucción del libro no inicia entrenamiento.'''
            text=text.replace('TGForecaster se documenta con resultados pendientes.','TGForecaster participa con cinco semillas en cada variante N y N+O.')
            text=text.replace('TGForecaster mantiene su ficha y pipeline con resultados pendientes.','TGForecaster incorpora N y N+O; el procedimiento con documentos presenta un error medio mayor que su control N.')
            if name=='09_validacion':
                text=text.replace('Las doce configuraciones','Las catorce configuraciones').replace('Las 60 realizaciones utilizan','Las 70 realizaciones utilizan')
                text=text.replace('91 parejas temporales entre doce procedimientos y dos referencias, y 66 parejas entre las doce series','120 parejas temporales entre catorce procedimientos y dos referencias, y 91 parejas entre las catorce series')
                text=text.replace('Los dos contrastes específicos de texto','Los tres contrastes específicos de texto').replace('familia exploratoria de dos parejas','familia exploratoria de tres parejas')
                text=text.replace('Para los seis procedimientos sin piloto externo','Para los ocho procedimientos sin piloto externo')
            cell.source=text
            if 'plantilla_resultados' in cell.metadata:
                cell.metadata.plantilla_resultados=text
        nb.write(notebook,p)

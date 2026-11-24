"""Integra literatura y resultados conservados de M4_N en el libro, sin redes."""
from pathlib import Path
import ast, base64, copy, hashlib, json, re, shutil, traceback
import nbformat as nb
import numpy as np
import pandas as pd
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output
from IPython.core.pylabtools import print_figure
from matplotlib.figure import Figure

VERSION='m4_top10_20261123'

SETUP=r'''from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from IPython.display import display

m4_root=next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/'results/m4_cierre_gpu_20261123/reducido_B/resumen_resultados.csv').exists())
m4_run=m4_root/'results/m4_cierre_gpu_20261123/reducido_B'
m4_out=m4_root/'results/evaluacion_m4_jbook'
m4_out.mkdir(exist_ok=True)
m4_catalog=pd.read_json(m4_out/'catalogo_comparadores.json')
m4_benchmark=pd.read_csv(m4_root/'results/evaluacion_e2/benchmark.csv').set_index('clave')
m4_q=float(json.loads((m4_root/'results/evaluacion_e2/protocolo.json').read_text(encoding='utf-8'))['q168'])
m4_seeds=list(range(101,111))
m4_predictions=[];m4_history=[];m4_checkpoints=[];m4_seed_metrics=[]
for m4_seed in m4_seeds:
    with np.load(m4_run/f'colombia/prueba/M4_N_semilla_{m4_seed}.npz',allow_pickle=False) as m4_z:
        if not m4_predictions:
            m4_dates=pd.to_datetime(m4_z['dia']);m4_y=m4_z['y'].astype(float)
        assert np.array_equal(m4_z['dia'],m4_dates.strftime('%Y-%m-%d'))
        assert np.array_equal(m4_z['y'],m4_y)
        m4_pred=m4_z['pred'].astype(float)
        assert m4_pred.shape==(1461,24) and np.isfinite(m4_pred).all()
        m4_predictions.append(m4_pred)
    m4_folder=m4_run/f'colombia/final/M4_N/semilla_{m4_seed}'
    m4_hist=pd.read_csv(m4_folder/'historia.csv');m4_hist['semilla']=m4_seed
    m4_ck=json.loads((m4_folder/'TERMINADO.json').read_text(encoding='utf-8'))
    assert m4_ck['mejor_epoca']==int(m4_hist.loc[m4_hist.perdida_parada.idxmin(),'epoca'])
    assert np.isclose(m4_ck['mejor_perdida'],m4_hist.perdida_parada.min())
    m4_history.append(m4_hist);m4_checkpoints.append(m4_ck)
    m4_error=m4_pred-m4_y
    m4_seed_metrics.append({'semilla':m4_seed,'MASE':np.abs(m4_error).mean()/m4_q,
        'MAE':np.abs(m4_error).mean(),'RMSE':np.sqrt(np.square(m4_error).mean())})
m4_predictions=np.stack(m4_predictions);m4_history=pd.concat(m4_history,ignore_index=True)
m4_seed_metrics=pd.DataFrame(m4_seed_metrics);m4_checkpoints=pd.DataFrame(m4_checkpoints)
m4_saved=pd.read_csv(m4_run/'metricas_por_semilla.csv').query("modelo=='M4_N'").set_index('semilla')
assert np.allclose(m4_seed_metrics.set_index('semilla').MASE,m4_saved.MASE,rtol=1e-8)
assert m4_dates.equals(pd.date_range('2022-01-01','2025-12-31',freq='D'))
m4_daily=np.abs(m4_predictions-m4_y[None]).mean(axis=(0,2))/m4_q
m4_table=m4_benchmark.loc[m4_catalog.clave,['modelo','MASE','MASE_sd','MAE','RMSE','semillas']].copy()
m4_table.loc['M4_N']=['M4ₙ',m4_seed_metrics.MASE.mean(),m4_seed_metrics.MASE.std(ddof=1),
    m4_seed_metrics.MAE.mean(),m4_seed_metrics.RMSE.mean(),len(m4_seeds)]
m4_table=m4_table.sort_values('MASE')
m4_table.to_csv(m4_out/'benchmark_m4_top10_baselines.csv',index_label='clave')
m4_top=m4_catalog.query("grupo=='Top 10'")
m4_best_key=m4_benchmark.loc[m4_top.clave].MASE.idxmin()
m4_best_name=m4_catalog.set_index('clave').loc[m4_best_key,'nombre']
assert m4_best_key=='nuevo_timesnet__N'
m4_ref_predictions=[]
for m4_file in sorted((m4_root/'results/m4_cierre_gpu_20261123/historicos_nbeats_timesnet/predicciones/TimesNet').glob('*.npz')):
    with np.load(m4_file,allow_pickle=False) as m4_z:
        assert np.array_equal(m4_z['y'],m4_y)
        assert np.array_equal(m4_z['dia'],m4_dates.strftime('%Y-%m-%d'))
        m4_ref_predictions.append(m4_z['yhat_lote1_kwh'].astype(float))
m4_ref_predictions=np.stack(m4_ref_predictions)
assert m4_ref_predictions.shape==(5,1461,24)
assert np.isclose(np.abs(m4_ref_predictions-m4_y).mean()/m4_q,m4_benchmark.loc[m4_best_key,'MASE'],rtol=1e-6,atol=1e-8)
m4_comparators={('historico_TimesNet' if r.clave=='nuevo_timesnet__N' else 'historico_NBEATS' if r.clave=='nuevo_nbeats__N' else 'historico_'+r.clave):r.nombre for r in m4_top.itertuples()}
m4_tests=pd.read_csv(m4_run/'ampliacion_top10/contrastes_F2_Top10_completo.csv')
m4_tests=m4_tests.loc[m4_tests.A.eq('M4_N')&m4_tests.B.isin(m4_comparators)].copy()
m4_tests['Comparador']=m4_tests.B.map(m4_comparators)
assert len(m4_tests)==len(m4_top)*3*2 and len(m4_top)==10
assert not m4_tests.duplicated(['B','metodo','margen_relativo']).any()
m4_tests.to_csv(m4_out/'contrastes_m4_top10.csv',index=False)
m4_forest=m4_tests.loc[m4_tests.metodo.eq('bootstrap_estacionario')&m4_tests.margen_relativo.eq(0)].sort_values('ganancia_relativa')
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':110})
display(m4_table.assign(MAE_GWh=m4_table.MAE/1e6,RMSE_GWh=m4_table.RMSE/1e6)[['modelo','MASE','MASE_sd','MAE_GWh','RMSE_GWh','semillas']].rename(columns={'modelo':'Procedimiento','MASE_sd':'DE entre semillas','MAE_GWh':'MAE (GWh)','RMSE_GWh':'RMSE (GWh)','semillas':'Semillas'}).style.format({'MASE':'{:.4f}','DE entre semillas':'{:.4f}','MAE (GWh)':'{:.4f}','RMSE (GWh)':'{:.4f}','Semillas':'{:.0f}'}))
'''

FOREST=r'''m4_fig,m4_ax=plt.subplots(figsize=(10,5.5))
m4_g=100*m4_forest.ganancia_relativa.to_numpy()
m4_lo=100*m4_forest.ic95_inf.to_numpy();m4_hi=100*m4_forest.ic95_sup.to_numpy()
m4_ax.errorbar(m4_g,np.arange(len(m4_g)),xerr=np.vstack([m4_g-m4_lo,m4_hi-m4_g]),fmt='o',color='#146c70',ecolor='#38535c',capsize=3)
m4_ax.set_yticks(np.arange(len(m4_g)),m4_forest.Comparador)
m4_ax.axvline(0,color='#71828a',lw=1);m4_ax.axvline(3,color='#bc7734',ls='--',label='Margen relativo del 3 %')
m4_ax.set_xlabel('Reducción del MAE/MASE de M4ₙ frente al comparador (%) · IC95 % puntual')
m4_ax.grid(axis='x',alpha=.15);m4_ax.legend(frameon=False)
m4_fig.tight_layout();m4_fig.savefig(m4_out/'ganancia_top10.png',dpi=170)
display(m4_fig);plt.close(m4_fig)
'''

TESTS=r'''m4_p=m4_tests.loc[m4_tests.margen_relativo.eq(0)].pivot(index='Comparador',columns='metodo',values='p_holm_cota_familia1536')
m4_p=m4_p.reindex(m4_forest.Comparador)
m4_p3=m4_tests.loc[m4_tests.metodo.eq('bootstrap_estacionario')&m4_tests.margen_relativo.eq(.03)].set_index('Comparador').p_holm_cota_familia1536
m4_p['Bootstrap · mejora >3 %']=m4_p3
m4_p=m4_p.rename(columns={'DM_HLN':'DM–HLN · mejora >0','GW_bilateral':'GW bilateral','bootstrap_estacionario':'Bootstrap · mejora >0'})
display(m4_p[['DM–HLN · mejora >0','Bootstrap · mejora >0','GW bilateral','Bootstrap · mejora >3 %']].style.format(lambda x:f'{x:.3g}'))
m4_p.to_csv(m4_out/'contrastes_top10_resumen.csv')
'''

LEARNING=r'''from matplotlib.lines import Line2D
m4_fig,m4_axes=plt.subplots(5,2,figsize=(12,15),sharex=True,sharey=True)
for m4_ax,m4_seed in zip(m4_axes.flat,m4_seeds):
    m4_h=m4_history.loc[m4_history.semilla.eq(m4_seed)]
    m4_best=m4_h.loc[m4_h.perdida_parada.idxmin()]
    m4_ax.plot(m4_h.epoca,m4_h.perdida_ajuste,color='#146c70',lw=1.7)
    m4_ax.plot(m4_h.epoca,m4_h.perdida_parada,color='#c27821',lw=1.7)
    m4_ax.scatter(m4_best.epoca,m4_best.perdida_parada,marker='*',s=100,color='#c27821',edgecolors='white',zorder=4)
    m4_ax.axvline(m4_best.epoca,color='#71828a',ls=':',alpha=.6)
    m4_ax.set_title(f'Semilla {m4_seed} · checkpoint en época {int(m4_best.epoca)}',fontsize=10)
    m4_ax.set_xlabel('Época');m4_ax.set_ylabel('MAE / q');m4_ax.grid(alpha=.15)
m4_fig.legend(handles=[Line2D([],[],color='#146c70',label='Ajuste · 2013–2020'),Line2D([],[],color='#c27821',label='Parada interna · 2021'),Line2D([],[],color='#c27821',marker='*',lw=0,label='Checkpoint restaurado')],loc='upper center',ncol=3,frameon=False)
m4_fig.tight_layout(rect=[0,0,1,.97]);m4_fig.savefig(m4_out/'curvas_aprendizaje.png',dpi=150)
display(m4_fig);plt.close(m4_fig)
'''

SCATTER=r'''from matplotlib.colors import LogNorm
m4_fig,m4_axes=plt.subplots(1,2,figsize=(12,5.2),sharex=True,sharey=True)
m4_min=min(m4_y.min(),m4_predictions.min(),m4_ref_predictions.min())/1e6
m4_max=max(m4_y.max(),m4_predictions.max(),m4_ref_predictions.max())/1e6
for m4_ax,m4_name,m4_stack in zip(m4_axes,['M4ₙ',m4_best_name],[m4_predictions,m4_ref_predictions]):
    m4_hex=m4_ax.hexbin(np.tile(m4_y.ravel(),len(m4_stack))/1e6,m4_stack.reshape(-1)/1e6,gridsize=65,mincnt=1,norm=LogNorm(),cmap='viridis')
    m4_ax.plot([m4_min,m4_max],[m4_min,m4_max],'--',color='#b4514b',lw=1.2)
    m4_ax.set_title(m4_name);m4_ax.set_xlabel('Observado (GWh por intervalo)')
    m4_ax.set_aspect('equal',adjustable='box')
    m4_fig.colorbar(m4_hex,ax=m4_ax,label='Pares origen–hora–semilla (escala log)')
m4_axes[0].set_ylabel('Predicho (GWh por intervalo)')
m4_fig.tight_layout();m4_fig.savefig(m4_out/'observado_predicho.png',dpi=170)
display(m4_fig);plt.close(m4_fig)
'''

WEEK=r'''import matplotlib.dates as mdates
m4_week=(m4_dates>=pd.Timestamp('2023-07-01'))&(m4_dates<pd.Timestamp('2023-07-08'))
m4_time=pd.date_range('2023-07-01',periods=7*24,freq='h')
m4_week_pred=m4_predictions[:,m4_week,:]
m4_fig,m4_ax=plt.subplots(figsize=(13,4.5))
m4_ax.fill_between(m4_time,np.quantile(m4_week_pred,.1,axis=0).ravel()/1e6,np.quantile(m4_week_pred,.9,axis=0).ravel()/1e6,color='#146c70',alpha=.18,label='M4ₙ · p10–p90 entre semillas')
m4_ax.plot(m4_time,m4_y[m4_week].ravel()/1e6,color='#29383d',lw=1.8,label='Observado')
m4_ax.plot(m4_time,m4_week_pred.mean(axis=0).ravel()/1e6,color='#146c70',lw=1.3,ls='--',label='M4ₙ · media de semillas')
m4_ax.plot(m4_time,m4_ref_predictions[:,m4_week,:].mean(axis=0).ravel()/1e6,color='#c27821',lw=1,alpha=.9,label=m4_best_name+' · media de semillas')
m4_ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))
m4_ax.set(xlabel='Fecha y hora de Colombia · 2023',ylabel='Demanda (GWh por intervalo)')
m4_ax.legend(ncol=2,fontsize=9,frameon=False);m4_ax.grid(alpha=.15)
m4_fig.tight_layout();m4_fig.savefig(m4_out/'semana_observado_predicho.png',dpi=170)
display(m4_fig);plt.close(m4_fig)
'''

SEGMENTS=r'''m4_abs=np.abs(m4_predictions-m4_y[None]);m4_ref_abs=np.abs(m4_ref_predictions-m4_y[None])
m4_fig,m4_axes=plt.subplots(1,2,figsize=(12,4.3))
for m4_label,m4_err,m4_color in [('M4ₙ',m4_abs,'#146c70'),(m4_best_name,m4_ref_abs,'#c27821')]:
    m4_axes[0].plot(np.arange(1,25),m4_err.mean(axis=(0,1))/1e6,label=m4_label,color=m4_color)
    m4_axes[1].plot(range(2022,2026),[m4_err[:,m4_dates.year==yy].mean()/m4_q for yy in range(2022,2026)],marker='o',label=m4_label,color=m4_color)
m4_axes[0].set(xlabel='Horizonte (hora)',ylabel='MAE (GWh por intervalo)',xticks=[1,6,12,18,24])
m4_axes[1].set(xlabel='Año de prueba',ylabel='MASE',xticks=range(2022,2026))
for m4_ax in m4_axes:m4_ax.grid(alpha=.15);m4_ax.legend(frameon=False)
m4_fig.tight_layout();m4_fig.savefig(m4_out/'horizonte_anio.png',dpi=170)
display(m4_fig);plt.close(m4_fig)
m4_seed_metrics.to_csv(m4_out/'metricas_m4_semillas.csv',index=False)
m4_history.to_csv(m4_out/'curvas_m4.csv',index=False)
'''

def marcar(c):
    c.metadata['integracion_m4']=VERSION
    return c

def md(text):return marcar(nb.v4.new_markdown_cell(text.strip()))
def code(text):return marcar(nb.v4.new_code_cell(text.strip()))
def dec(v,n=4):return f'{float(v):.{n}f}'.replace('.',',')

def ejecutar(cells,root):
    import os
    shell=InteractiveShell.instance()
    def imagen(fig):
        data=print_figure(fig,fmt='png',base64=True)
        return {'image/png':data,'text/plain':'Figura generada a partir de resultados conservados.'}
    shell.display_formatter.ipython_display_formatter.for_type(Figure,lambda fig:shell.display_pub.publish(data=imagen(fig)))
    old=Path.cwd();os.chdir(root)
    try:
        for i,c in enumerate(cells,1):
            if c.cell_type!='code':continue
            ast.parse(c.source)
            with capture_output() as cap:result=shell.run_cell(c.source,store_history=False)
            if result.error_in_exec or result.error_before_exec:
                error=result.error_in_exec or result.error_before_exec
                raise RuntimeError(f'M4 celda {i}: '+''.join(traceback.format_exception(error)))
            c.execution_count=i;c.outputs=[]
            if cap.stdout:c.outputs.append(nb.v4.new_output('stream',name='stdout',text=cap.stdout))
            if cap.stderr:c.outputs.append(nb.v4.new_output('stream',name='stderr',text=cap.stderr))
            for output in cap.outputs:c.outputs.append(nb.v4.new_output('display_data',data=output.data,metadata=output.metadata))
        return shell.user_ns
    finally:os.chdir(old)

def literatura():
    return [md('''## 8.8. Memoria residual para el pronóstico de demanda

La corrección residual separa la estimación del nivel de demanda de los errores que persisten después del pronóstico. M4<sub>N</sub> estudia si esos errores pueden transferirse entre episodios con contexto numérico comparable, manteniendo congelada la base. La recuperación de residuos y la actualización de memoria tienen antecedentes; la contribución investigada se concentra en su formulación temporal y en el aprendizaje de la recuperación.

### Antecedentes y relación con M4

| Estudio | Mecanismo y evidencia consultada | Relación con la propuesta |
|---|---|---|
| [RATL, He et al. (2026)](https://arxiv.org/html/2609.03937v1), prepublicación; texto completo | Base congelada, memoria de sus residuos y router por bloques y variables; incluye ECL. | Es el trabajo más cercano: la recuperación residual no se presenta como invención de M4. RATL emplea claves de la base y supervisión de un profesor; M4 aprende consulta y claves de contexto y una compuerta por origen. |
| [FreshCast, He et al. (2026)](https://arxiv.org/html/2610.07834v3), prepublicación; texto completo | Memoria actualizada con observaciones reveladas, regresión por kernel y peso calibrado; incluye Electricity. | Sustenta la utilidad de referencias recientes sin reajustar el pronosticador. M4 recupera trayectorias de errores fuera de muestra y regula su aportación con una red condicionada al respaldo recuperado. |
| [MMEMformer (2025)](https://doi.org/10.1016/j.energy.2025.135762); resumen editorial | Atención sobre un banco de patrones históricos y fusión de representaciones para carga de sistemas energéticos integrados. | La memoria de patrones ya tiene aplicación en carga; su comparación distingue valores y representaciones históricas de errores de un pronosticador específico. La lectura de texto completo queda pendiente. |
| [Chandrasekaran y Paramasivan (2026)](https://doi.org/10.3389/frai.2026.1935983); texto completo | Transformer congelado y corrector residual GRU con atención, para demanda a 24 horas en Tamil Nadu y Panamá. | Confirma antecedentes de corrección neuronal posterior al pronóstico. M4 conserva episodios recuperables en lugar de generar toda la corrección mediante un estado recurrente. |
| [ResMem, Yang et al. (2023)](https://proceedings.neurips.cc/paper_files/paper/2023/hash/bf0857cb9a41c73639f028a80301cdf0-Abstract-Conference.html); resumen oficial | Suma un regresor de residuos por vecinos a una base neuronal, con aplicaciones en visión y lenguaje. | Antecedente transversal de memorización residual; la adaptación temporal requiere definir cuándo están disponibles las etiquetas y cómo se construyen los residuos. |
| [Vargas-Forero et al. (2025)](https://doi.org/10.32479/ijeep.17528); texto completo | LSTM, ARIMA y Prophet para demanda nacional colombiana horaria del día siguiente. | Antecedente directo del fenómeno, sistema y horizonte; contextualiza la evaluación del SIN sin equiparar métricas ni periodos entre artículos. |

### Mecanismo, entrenamiento y pregunta de investigación

El diseño conserva una base M1 formada por PatchTST, E y el experto periódico. Sus residuos se obtienen mediante predicciones fuera del bloque de ajuste, con selección por prefijos y OOF temporal anidado. Dos proyecciones aprendidas representan el origen actual y los episodios históricos; la máscara de disponibilidad se aplica antes de recuperar vecinos. Una compuerta combina contexto, semejanza, dispersión, respaldo y antigüedad para regular la trayectoria residual de 24 horas.

Frente a RATL, las diferencias se sitúan en la representación aprendida del contexto, el esquema de generación de residuos y la lectura escalar por origen. Frente a FreshCast, se recuperan errores de la base y se aprende una compuerta contextual, en lugar de combinar un pronóstico de memoria mediante calibración cerrada. Los umbrales de antigüedad y el número de vecinos son decisiones de configuración; no constituyen por sí solos una contribución metodológica.

La pregunta contrastable es si esta integración reduce el error de demanda del SIN colombiano a 24 horas frente al Top 10 y los baselines del benchmark. La evaluación utiliza MASE, MAE y RMSE, una unidad diaria de 24 errores y contrastes temporales con multiplicidad controlada. La novedad investigada corresponde al mecanismo y al entrenamiento; su aplicación geográfica y el desempeño observado no bastan para demostrarla.

La búsqueda dirigida del 23 de noviembre de 2026 complementa el corpus bibliográfico. RATL y FreshCast se revisaron en sus versiones indicadas; la matriz distingue texto completo de resumen. Estos trabajos de mecanismos no se añaden al denominador histórico de frecuencia de uso como baseline. [Registro de búsqueda y evidencia](_static/literature/m4_revision.json).
''')]

def celdas_resultados():
    return [md(r'''(resultados-m4)=
## M4ₙ: evaluación frente al Top 10 y los baselines

La propuesta numérica conserva la configuración 6 y se evalúa con diez inicializaciones (101–110), sobre 1.461 orígenes diarios de 2022–2025 y 24 horas por origen. El ajuste de la memoria utiliza 2013–2020 y la reserva interna de 2021 determina el checkpoint restaurado. La base y los pesos de la memoria permanecen congelados durante prueba; el banco solo incorpora residuos completos admitidos por el retraso de dos días.

La pérdida diaria del procedimiento promedia primero los errores individuales de sus semillas:

$$L_{m,t}=\frac{1}{S_m\,24\,q_{168}}\sum_{s=1}^{S_m}\sum_{h=1}^{24}|y_{t,h}-\widehat y_{m,s,t,h}|.$$

M4ₙ utiliza diez semillas; el Top 10 y los baselines aprendidos conservan cinco. No se emparejan semillas por su identificador: la comparación temporal utiliza los mismos días y promedia las pérdidas entre inicializaciones. La evaluación es retrospectiva. Los resultados de la propuesta se distinguen de los diagnósticos del benchmark de referencia presentados después.

### Desempeño agregado
'''),code(SETUP),md('<!-- m4:metricas -->'),
    md('''### Ganancia relativa frente al Top 10

La reducción se calcula respecto de cada comparador: un porcentaje positivo favorece a M4ₙ. Los intervalos son puntuales del 95 % y se obtienen con bootstrap estacionario pareado; la significación se decide con las cotas ajustadas de Holm, no por la inspección aislada de cada intervalo.
'''),code(FOREST),md('<!-- m4:forest -->'),
    md('''### Contrastes temporales y margen de mejora

Se presentan los contrastes guardados de M4ₙ frente a las diez arquitecturas. El bootstrap utiliza 49.999 réplicas y longitud esperada de bloque de 14 días; DM–HLN utiliza HAC de 13 rezagos. GW es bilateral y usa un instrumento observable con retraso de dos días. Se conservan las cotas de Holm de la familia registrada de 1.536 hipótesis, sin recalcularlas sobre este subconjunto. Las hipótesis no ejecutadas no se interpretan como pruebas observadas. La columna de mejora superior al 3 % procede del contraste correspondiente, distinto de la mera superioridad sobre cero.
'''),code(TESTS),md('<!-- m4:tests -->'),
    md('''### Curvas de entrenamiento y validación interna

Cada panel superpone la pérdida de ajuste y la de parada interna de una semilla. La estrella identifica el checkpoint de menor pérdida interna restaurado antes de predecir prueba; no corresponde necesariamente a la última época ejecutada.
'''),code(LEARNING),md('<!-- m4:learning -->'),
    md('''### Demanda observada frente a predicción

Se comparan las predicciones individuales de M4ₙ con TimesNet, el modelo de menor MASE del Top 10. Cada panel utiliza todos los pares origen–hora–semilla disponibles y la misma escala de demanda. Las frecuencias de las celdas se representan en escala logarítmica; sus magnitudes dependen también del número de semillas.
'''),code(SCATTER),md('<!-- m4:scatter -->'),
    md('''### Perfil horario de una semana de prueba

La ventana del 1 al 7 de julio de 2023 coincide con la utilizada en el benchmark. Las líneas de pronóstico representan medias entre semillas; el rango p10–p90 describe dispersión entre inicializaciones y no un intervalo predictivo calibrado. Las métricas se calculan con errores individuales, no con el error de la predicción media.
'''),code(WEEK),md('<!-- m4:week -->'),
    md('''### Error por horizonte y año

El MAE por hora agrega los mismos orígenes de prueba. El MASE anual mantiene la escala semanal de entrenamiento fija; cada año conserva su número real de días y el error se promedia entre inicializaciones.
'''),code(SEGMENTS),md('<!-- m4:segments -->'),md('<!-- m4:discusion -->'),
    md('''### Archivos y reproducción de la evaluación de M4ₙ

[Benchmark de M4ₙ, Top 10 y baselines](_static/resultados/evaluacion_m4/benchmark_m4_top10_baselines.csv) · [Contrastes temporales del Top 10](_static/resultados/evaluacion_m4/contrastes_m4_top10.csv) · [Métricas por semilla de M4ₙ](_static/resultados/evaluacion_m4/metricas_m4_semillas.csv) · [Curvas de aprendizaje](_static/resultados/evaluacion_m4/curvas_m4.csv) · [Manifiesto de entradas](_static/resultados/evaluacion_m4/manifiesto.json).

Los bloques Python de esta sección reconstruyen métricas, tablas y figuras a partir de las predicciones e historiales guardados. No cargan checkpoints ni ejecutan entrenamiento. La reproducción estadística conserva las pérdidas diarias y el protocolo de los contrastes; el script documental no vuelve a seleccionar configuraciones.

## Benchmark de referencia: selección, precisión y diagnósticos

Los apartados siguientes caracterizan los quince procedimientos del benchmark de referencia y sus cinco semillas. Sus matrices de rangos, MCS, perturbaciones y latencia corresponden a ese conjunto; los contrastes de M4ₙ frente al Top 10 se presentan en el bloque anterior.
''')]

def aplicar(root,run_cells=True):
    root=Path(root).resolve();out=root/'results/evaluacion_m4_jbook';out.mkdir(parents=True,exist_ok=True)
    d=json.loads((root/'dashboard/site/data/study.json').read_text(encoding='utf-8'))
    catalog=[{'clave':x['id'],'nombre':x['name'],'grupo':x['group']} for x in d['models'] if x['group'] in ['Top 10','Baselines E1','Referencia clásica']]
    (out/'catalogo_comparadores.json').write_text(json.dumps(catalog,ensure_ascii=False,indent=2),encoding='utf-8')
    cells=celdas_resultados()
    if not run_cells:return cells
    ns=ejecutar(cells,root)
    table=ns['m4_table'];forest=ns['m4_forest'];check=ns['m4_checkpoints'];seeds=ns['m4_seed_metrics'];tests=ns['m4_tests']
    best_name=ns['m4_best_name'];m4=float(seeds.MASE.mean());mae=float(seeds.MAE.mean()/1e6);rmse=float(seeds.RMSE.mean()/1e6)
    best=forest.loc[forest.Comparador.eq(best_name)].iloc[0]
    pos=tests.loc[tests.margen_relativo.eq(0)&tests.metodo.eq('bootstrap_estacionario')]
    margin=tests.loc[tests.margen_relativo.eq(.03)&tests.metodo.eq('bootstrap_estacionario')]
    rejects=int(pos.p_holm_cota_familia1536.lt(.05).sum());rejects3=int(margin.p_holm_cota_familia1536.lt(.05).sum())
    smallest=forest.loc[forest.ganancia_relativa.idxmin()];largest=forest.loc[forest.ganancia_relativa.idxmax()]
    epoch_lo=int(check.mejor_epoca.min());epoch_hi=int(check.mejor_epoca.max())
    week=ns['m4_week'];pred=ns['m4_predictions'];y=ns['m4_y'];ref=ns['m4_ref_predictions'];dates=ns['m4_dates'];q=ns['m4_q']
    year_gains={str(yy):100*(1-np.abs(pred[:,dates.year==yy]-y[dates.year==yy]).mean()/np.abs(ref[:,dates.year==yy]-y[dates.year==yy]).mean()) for yy in range(2022,2026)}
    interpretation={
      'metricas':f'M4ₙ alcanza MASE {dec(m4)}, MAE {dec(mae)} GWh y RMSE {dec(rmse)} GWh por intervalo. La desviación estándar de MASE entre sus diez semillas es {dec(seeds.MASE.std(ddof=1))}. La pérdida normalizada de parada interna de 2021, en los checkpoints restaurados, es {dec(check.mejor_perdida.mean())} ± {dec(check.mejor_perdida.std(ddof=1))} entre semillas; esa reserva participa en la selección de época. Su error medio de prueba es menor que el de los diez modelos del Top 10 y los cuatro baselines de la tabla; este orden descriptivo se complementa con los contrastes temporales siguientes.',
      'forest':f'La reducción relativa frente al Top 10 va de {dec(100*smallest.ganancia_relativa,2)} % ({smallest.Comparador}) a {dec(100*largest.ganancia_relativa,2)} % ({largest.Comparador}). Frente a {best_name}, la mejora es {dec(100*best.ganancia_relativa,2)} %, con IC95 % puntual [{dec(100*best.ic95_inf,2)} %; {dec(100*best.ic95_sup,2)} %]. La ganancia se estima sobre los mismos días y no sobre muestras horarias tratadas como independientes.',
      'tests':f'El bootstrap detecta superioridad frente a {rejects} de los diez modelos con la cota de Holm registrada; el contraste de una mejora mayor del 3 % rechaza su nulo en {rejects3} de diez comparaciones. DM–HLN ofrece evidencia concordante de dirección favorable. El rechazo de GW indica diferencias condicionales y, por ser bilateral, no establece por sí solo qué procedimiento tiene menor error. La significación está condicionada al periodo, las realizaciones y los supuestos de dependencia temporal.',
      'learning':f'Los checkpoints restaurados corresponden a las épocas {epoch_lo}–{epoch_hi}; los ajustes finalizaron entre {int(check.epocas.min())} y {int(check.epocas.max())} épocas. La pérdida de entrenamiento disminuye, mientras la reserva interna alcanza su mínimo temprano y después fluctúa o aumenta. Ese patrón es compatible con sobreajuste al continuar la optimización y motiva la restauración de pesos por parada interna; la brecha entre periodos también depende de sus condiciones de demanda.',
      'scatter':'La proximidad a la diagonal representa acuerdos entre demanda observada y pronóstico; los desvíos por encima indican sobreestimación y por debajo, subestimación. La comparación utiliza la misma escala física. La concentración visual no sustituye las métricas agregadas ni los contrastes pareados, y la frecuencia absoluta es mayor cuando se incluyen más inicializaciones.',
      'week':f'En esta semana, el MAE individual medio es {dec(np.abs(pred[:,week]-y[week]).mean()/1e6)} GWh para M4ₙ y {dec(np.abs(ref[:,week]-y[week]).mean()/1e6)} GWh para {best_name}. Ambas trayectorias reproducen el ciclo diario; las diferencias alrededor de ascensos, máximos y descensos contextualizan el error. Esta ventana es ilustrativa y no reemplaza los 1.461 días del contraste.',
      'segments':'La reducción relativa de error de M4ₙ frente a '+best_name+' es '+', '.join(f'{dec(v,2)} % en {yy}' for yy,v in year_gains.items())+'. El perfil por horizonte permite localizar las horas de mayor dificultad; el agregado anual muestra la estabilidad o variación de la ganancia entre regímenes, sin utilizar esos cortes para volver a seleccionar el modelo.',
      'discusion':f'''### Síntesis de la propuesta numérica

M4ₙ reduce el MASE frente a las diez arquitecturas del Top 10 en la evaluación 2022–2025. Frente a {best_name}, su competidor de menor error medio, la reducción es {dec(100*best.ganancia_relativa,2)} %. Los contrastes temporales conservan la multiplicidad de la familia registrada y respaldan una mejora superior al margen del 3 % en {rejects3} de las diez comparaciones. La evidencia cuantifica el desempeño del procedimiento completo; no atribuye la ganancia exclusivamente a una capa.

El banco se actualiza con resultados ya observables y los parámetros permanecen congelados. La diferencia de cinco frente a diez semillas impide interpretar los identificadores como réplicas pareadas. La evaluación retrospectiva y las diferencias de presupuestos y entradas delimitan la generalización del resultado. La contrastación en conjuntos adicionales, el dimensionamiento para potencia y la verificación teórica se documentan separadamente del rendimiento en el SIN.'''
    }
    for c in cells:
        if c.cell_type=='markdown':
            for k,v in interpretation.items():c.source=c.source.replace(f'<!-- m4:{k} -->',v)
    p=root/'book/10_evaluacion.ipynb';n=nb.read(p,4)
    n.cells=[c for c in n.cells if not c.metadata.get('integracion_m4')]
    insert=next(i for i,c in enumerate(n.cells) if c.cell_type=='markdown' and c.source.startswith('## Entrenamiento, selección'))
    n.cells[insert:insert]=cells
    # Se conserva la semántica de los diagnósticos históricos del benchmark.
    nb.write(n,p)
    p=root/'book/08_literatura.ipynb';n=nb.read(p,4)
    n.cells=[c for c in n.cells if not c.metadata.get('integracion_m4')]
    for c in n.cells:
        if c.cell_type=='markdown':
            c.source=c.source.replace('La brecha operativa del proyecto es determinar si el contexto documental', 'La línea documental del estudio busca determinar si el contexto documental')
            c.source=c.source.replace('La comparación del proyecto conserva MASE como métrica primaria', 'El benchmark de referencia conserva MASE como métrica primaria')
    n.cells+=literatura();nb.write(n,p)
    for stem,text in [('10_modelos',f'''## Propuesta numérica con memoria

La {{ref}}`arquitectura de M4ₙ <m4-n>` desarrolla la memoria de errores y sus expertos. La {{ref}}`evaluación de la propuesta <resultados-m4>` reúne sus diez semillas, curvas de aprendizaje y comparación con el Top 10 y los baselines. En prueba, M4ₙ obtiene MASE {dec(m4)}; el benchmark de referencia conserva quince procedimientos y cinco semillas por procedimiento.'''),('11_sintesis',f'''## Memoria residual para el pronóstico de demanda

M4ₙ alcanza MASE {dec(m4)} sobre 1.461 orígenes diarios y diez semillas, frente a {dec(ns['m4_benchmark'].loc[ns['m4_best_key'],'MASE'])} de {best_name}, el menor error del Top 10. La reducción relativa es {dec(100*best.ganancia_relativa,2)} % y el bootstrap con cota de Holm respalda una mejora superior al 3 % frente a las diez arquitecturas. El aprendizaje restaura checkpoints tempranos mediante la reserva interna; durante prueba solo avanza el banco de observaciones disponibles. La {{ref}}`evaluación de M4ₙ <resultados-m4>` presenta las métricas y su alcance temporal.'''),('12_reproducibilidad','''## Reproducción de M4ₙ

`scripts/integrar_m4_resultados.py` reconstruye las métricas, tablas y figuras de M4ₙ desde las predicciones e historiales de diez semillas conservadas en `results/m4_cierre_gpu_20261123/reducido_B/`. La escala semanal y los comparadores proceden del benchmark; TimesNet se verifica con sus cinco predicciones de lote uno. Los bloques analíticos se muestran completos en la sección de resultados.

Ejecutar `python scripts/integrar_m4_resultados.py` desde la raíz regenera únicamente la documentación y los resúmenes derivados. El manifiesto registra las huellas de los archivos utilizados. No se importan módulos de redes ni se cargan pesos; los contrastes se leen de los resultados guardados y conservan sus ajustes por multiplicidad.''')]:
        p=root/f'book/{stem}.ipynb';n=nb.read(p,4);n.cells=[c for c in n.cells if not c.metadata.get('integracion_m4')]
        n.cells.append(md(text));nb.write(n,p)
    # Actualiza la exposición del número de semillas sin cambiar las fuentes históricas.
    for p in [root/'scripts/documentacion_m4/contenido_m4_n.md',root/'book/arquitecturas/m4_n.md']:
        text=p.read_text(encoding='utf-8').replace('Las cinco semillas históricas son 101–105. La evaluación adicional registra su diseño antes de ejecutar nuevos experimentos.','La evaluación de M4_N comprende diez semillas, 101–110, con configuración 6 y restauración del mejor checkpoint interno. Los resultados y las comparaciones frente al Top 10 se presentan en la evaluación integrada.')
        p.write_text(text,encoding='utf-8',newline='\n')
    assets=root/'book/_static/resultados/evaluacion_m4';assets.mkdir(parents=True,exist_ok=True)
    sourcefiles=[root/'results/evaluacion_e2/benchmark.csv',root/'results/evaluacion_e2/protocolo.json',ns['m4_run']/'metricas_por_semilla.csv',ns['m4_run']/'ampliacion_top10/contrastes_F2_Top10_completo.csv']
    sourcefiles+=list((ns['m4_run']/'colombia/final/M4_N').rglob('historia.csv'))+list((ns['m4_run']/'colombia/final/M4_N').rglob('TERMINADO.json'))
    sourcefiles+=[ns['m4_run']/f'colombia/prueba/M4_N_semilla_{s}.npz' for s in ns['m4_seeds']]
    sourcefiles+=list((root/'results/m4_cierre_gpu_20261123/historicos_nbeats_timesnet/predicciones/TimesNet').glob('*.npz'))
    manifest={'version':VERSION,'semillas_m4':ns['m4_seeds'],'comparadores':catalog,'archivos':[{'archivo':str(p.relative_to(root)).replace('\\','/'),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sourcefiles],
              'contrastes_top10':len(tests),'entrenamiento':False,'inferencia_redes':False,'publicacion':False,
              'MASE_M4_N':m4,'MAE_GWh':mae,'RMSE_GWh':rmse,'rechazos_top10_bootstrap':rejects,'rechazos_top10_margen3':rejects3}
    (out/'manifiesto.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    for p in out.glob('*.csv'):shutil.copy2(p,assets/p.name)
    shutil.copy2(out/'manifiesto.json',assets/'manifiesto.json')
    # El registro bibliográfico limita su contenido a los mecanismos de la propuesta.
    review={'fecha':'2026-11-23','alcance':'Búsqueda dirigida de fuentes primarias; no certifica revisión sistemática exhaustiva.',
      'consultas':['"load forecasting" "residual retrieval"','"Colombia" "load forecasting" "error correction"','"time series forecasting" "residual memory"','"load forecasting" "memory bank" attention gating','"Colombia" "electricity demand" "24 h"'],
      'fuentes':[{'titulo':'RATL','url':'https://arxiv.org/html/2609.03937v1','evidencia':'Texto completo'}, {'titulo':'FreshCast','url':'https://arxiv.org/html/2610.07834v3','evidencia':'Texto completo'}, {'titulo':'MMEMformer','url':'https://doi.org/10.1016/j.energy.2025.135762','evidencia':'Resumen editorial'}, {'titulo':'TE-AG-GRU-RC','url':'https://doi.org/10.3389/frai.2026.1935983','evidencia':'Texto completo'}, {'titulo':'ResMem','url':'https://proceedings.neurips.cc/paper_files/paper/2023/hash/bf0857cb9a41c73639f028a80301cdf0-Abstract-Conference.html','evidencia':'Resumen oficial'}, {'titulo':'Demanda nacional colombiana','url':'https://doi.org/10.32479/ijeep.17528','evidencia':'Texto completo'}]}
    literature_assets=root/'book/_static/literature';literature_assets.mkdir(exist_ok=True)
    (literature_assets/'m4_revision.json').write_text(json.dumps(review,ensure_ascii=False,indent=2),encoding='utf-8')
    shutil.copy2(Path(__file__),root/'book/_static/codigo_e3/m4/integrar_m4_resultados.py')
    intro=root/'book/intro.md'
    text=intro.read_text(encoding='utf-8')
    text=text.replace('[evaluación integrada](10_evaluacion.ipynb#resultados-m4)', '{ref}`evaluación integrada <resultados-m4>`')
    if 'La propuesta M4<sub>N</sub>' not in text:
        text+='\n\nLa propuesta M4<sub>N</sub> incorpora una memoria numérica de errores sobre una combinación de expertos. Su evaluación utiliza diez semillas y compara el pronóstico de 24 horas con el Top 10 y los baselines sobre los mismos orígenes. La {ref}`evaluación integrada <resultados-m4>` presenta métricas, curvas y contrastes temporales.\n'
    intro.write_text(text,encoding='utf-8',newline='\n')
    assert set(tests.B)==set(ns['m4_comparators']) and len(tests)==60
    assert len(ns['m4_predictions'])==10 and len(ns['m4_history'].semilla.unique())==10
    assert all(c.execution_count is not None for c in cells if c.cell_type=='code')
    assert not any('<!-- m4:' in c.source for c in cells)
    return manifest

if __name__=='__main__':
    result=aplicar(Path(__file__).resolve().parents[1])
    print(json.dumps({k:v for k,v in result.items() if k not in ['archivos','comparadores']},ensure_ascii=False))

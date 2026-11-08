"""Capítulo reproducible a partir de resultados publicados; no entrena."""
import ast
import hashlib
from pathlib import Path

SHA='02ff447d94fa78e1cde323fa2382da5bce24fb6e'


def ampliar_modelos(books,md,code,note,root):
    snapshot=root/'results/repositorio'/SHA
    assert snapshot.exists()
    c=[
        md('# 10. Modelos y benchmark de demanda'),
        md('El benchmark compara MLP, LSTM y CNN1D como tres familias de Deep Learning, acompañadas por una regresión ARX penalizada y dos referencias estacionales. Las configuraciones se seleccionaron por MASE de validación; la evaluación final conserva 195 inicializaciones por familia sobre los mismos días de prueba.'),
        code("""from pathlib import Path
import json,hashlib,itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.multitest import multipletests
from IPython.display import display
ROOT=next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/'data/processed/panel_horario.parquet').exists())
SHA='02ff447d94fa78e1cde323fa2382da5bce24fb6e'
SNAP=ROOT/'results/repositorio'/SHA
R=SNAP/'results/base_v2'; OUT=ROOT/'results/jbook/modelos'; OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'figure.figsize':(10,4),'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.15})
plan=json.loads((R/'plan_gpu.json').read_text(encoding='utf-8'))
analisis=json.loads((OUT/'plan_analisis_secundario.json').read_text(encoding='utf-8'))
registro_archivos=json.loads((R/'resultados_guardados.json').read_text(encoding='utf-8'))['archivos']
for nombre,huella in analisis['sha256'].items():
    assert hashlib.sha256((R/nombre).read_bytes()).hexdigest()==huella
verificados=[]
for nombre,huella in plan['sha256'].items():
    p=SNAP/nombre if (SNAP/nombre).exists() else ROOT/nombre
    contenido=p.read_bytes()
    # El panel binario coincide exactamente; los textos locales pueden tener CRLF.
    coincide=hashlib.sha256(contenido).hexdigest()==huella
    if not coincide and p.suffix in ['.json','.md','.py']:
        coincide=hashlib.sha256(contenido.replace(b'\\r\\n',b'\\n')).hexdigest()==huella
    assert coincide,nombre
    verificados.append(nombre)
for nombre,info in registro_archivos.items():
    if (R/nombre).is_file():
        assert hashlib.sha256((R/nombre).read_bytes()).hexdigest()==info['sha256'],nombre
runs=pd.read_csv(R/'final_runs.csv'); piloto=pd.read_csv(R/'piloto.csv')
busqueda=pd.read_csv(R/'busqueda.csv'); ajustes=pd.read_csv(R/'resumen_ajustes.csv')
mejores=json.loads((R/'mejor_config.json').read_text(encoding='utf-8'))
referencias=json.loads((R/'referencias_test.json').read_text(encoding='utf-8'))
ref_cv=json.loads((R/'referencias_cv.json').read_text(encoding='utf-8'))
env=json.loads((R/'entorno_final.json').read_text(encoding='utf-8'))
escala_mase=env['escala_mase']
neuronales=['mlp','lstm','cnn1d']; modelos=['mlp','arx_lineal','lstm','cnn1d']
etiquetas={'mlp':'MLP','arx_lineal':'ARX','lstm':'LSTM','cnn1d':'CNN1D','ingenuo_24h':'Ingenuo diario','ingenuo_168h':'Ingenuo semanal'}
assert not runs.duplicated(['modelo','semilla']).any()
assert set(runs.modelo)==set(modelos)
for nombre,d in runs.groupby('modelo'):
    assert set(d.semilla)==set(range(1000,1195))
    assert d.params.map(json.loads).map(lambda p:p==mejores[nombre]['params']).all()
assert np.isfinite(runs[['MASE','RMSE','MAE','MASE_h1','MASE_h12','MASE_h24']]).all().all()
assert np.allclose(runs.MAE/runs.MASE,escala_mase)
publicado=pd.read_csv(R/'resumen_final.csv').set_index('modelo')
for nombre,d in runs.groupby('modelo'):
    for metrica in ['MASE','RMSE','MAE','sMAPE','MAPE','MASE_h1','MASE_h12','MASE_h24']:
        assert np.isclose(d[metrica].mean(),publicado.loc[nombre,metrica+'_media'])
n_ajustes=len(ajustes); n_fuentes=len(verificados)
"""),
        note('Se verifican {n_fuentes} fuentes y archivos del plan y {n_ajustes} ajustes resumidos. La escala semanal de MASE es {escala_mase:,.2f} kWh, calculada en 2000–2017. Las medias publicadas se reconstruyen desde los registros individuales; no se ejecutan ajustes ni nuevas predicciones.'),
        md('## 10.1 Entradas, particiones y ejecución\n\nCada origen diario de las 00:00 produce 24 demandas horarias mediante salida directa. La historia contiene 168 filas anteriores al origen. Las entradas son demanda real y calendario: el calendario histórico se concatena a las secuencias LSTM/CNN y el calendario futuro conocido alimenta una rama separada. MLP y ARX utilizan demanda histórica y calendario del horizonte aplanados.\n\nLa demanda se divide por su nivel medio histórico y se devuelve a kWh al evaluar. El estandarizador inicial de calendario se fija con 2000–2017; MLP y ARX ajustan además su estandarización tabular con el tramo de optimización de cada pliegue. El escalado inicial incluye parte de la reserva interna de parada del primer pliegue, por lo que no representa una separación estricta de todas las transformaciones respecto de esa reserva. No utiliza el año externo ni la prueba.\n\nLa selección aplica cuatro años externos, 2018–2021, y reserva el último 10 % del historial previo para parada temprana. El ajuste final emplea el historial anterior a 2022 con la misma separación interna; la prueba comprende 2022–2025. Las ejecuciones usan Linux, Python 3.12.3, PyTorch 2.9.1 con ROCm 6.4 y GPU AMD de aproximadamente 12 GB, con determinismo estricto y sin precisión mixta.'),
        md(r'''## 10.2 Arquitecturas y complejidad

**MLP.** Para la ventana aplanada $\mathbf z$ y el calendario futuro $\mathbf k$, la salida es

$$\widehat{\mathbf y}=W_o\,\phi(W_2\,\phi(W_1[\mathbf z;\mathbf k]+\mathbf b_1)+\mathbf b_2)+\mathbf b_o.$$

La configuración seleccionada tiene dos capas ocultas de 128 unidades, ReLU y dropout. Con dimensión $d$, ancho $u$, $a$ capas y horizonte $H$, el costo dominante por origen es $O(du+(a-1)u^2+uH)$.

**LSTM.** Una capa de 256 unidades procesa la historia. Sus compuertas actualizan memoria y estado oculto:

$$\mathbf c_t=\mathbf f_t\odot\mathbf c_{t-1}+\mathbf i_t\odot\mathbf g_t,\qquad
\mathbf h_t=\mathbf o_t\odot\tanh(\mathbf c_t).$$

El estado final se concatena con una proyección del calendario futuro y una cabeza densa genera las 24 salidas. El costo recurrente es $O(Lu(f+u))$, más las proyecciones densas. El dropout seleccionado afecta las salidas y la cabeza; al haber una sola capa recurrente, no se aplica dropout entre capas LSTM.

**CNN1D.** Un bloque de 32 filtros, kernel 7, ReLU y Dropout1d transforma exclusivamente la historia:

$$\mathbf z_t=\phi\left(\sum_{j=-3}^{3}W_j\mathbf x_{t+j}+\mathbf b\right).$$

El padding simétrico opera dentro de una historia ya observada. La salida se promedia en siete bloques de 24 horas y se combina con el calendario futuro mediante capas densas. El costo convolucional es $O(Lkfc)$ para un bloque, con kernel $k$, canales $f$ y filtros $c$; se añaden las proyecciones densas.

**ARX de contexto.** Una transformación afín de rezagos y calendario incorpora penalización L1 o L2, con costo $O(dH)$. Implementarla en PyTorch no la convierte en una arquitectura profunda; se reporta fuera de las tres familias neuronales. La penalización L1 se optimiza por gradiente y no garantiza coeficientes exactamente nulos.'''),
        code("""configuraciones=[]
for nombre in modelos:
    p=mejores[nombre]['params']
    configuraciones.append({'Modelo':etiquetas[nombre],
        'Estructura':('2 × 128' if nombre=='mlp' else '1 × 256' if nombre=='lstm' else '1 × 32; kernel 7' if nombre=='cnn1d' else '168 rezagos; L1'),
        'Parámetros':int(runs.loc[runs.modelo==nombre,'n_parametros'].iloc[0]),
        'Dropout':p.get('dropout',0.),'LR':p['lr'],'Lote':p['batch_size'],
        'Planificador':p['scheduler'],'Pérdida':p['perdida'],'Paciencia':p['paciencia'],
        'Decaimiento':p.get('weight_decay',0.)})
display(pd.DataFrame(configuraciones).set_index('Modelo').style.format({'Dropout':'{:.4f}','LR':'{:.3g}','Decaimiento':'{:.3g}'}))
"""),
        md('El optimizador común es AdamW, con recorte de norma del gradiente a 1, máximo de 200 épocas y mejora mínima de 10⁻⁶. Se restaura el estado que mejora la pérdida interna por encima de ese umbral. Huber utiliza transición 0,05 en la escala normalizada por nivel; la función de selección externa sigue siendo MASE, independientemente de la pérdida de entrenamiento.'),
        md('## 10.3 Búsqueda y presupuesto ejecutado\n\nLa búsqueda aleatoria utiliza tres semillas por candidato y cuatro pliegues. La adenda de costo computacional redujo LSTM de veinte a cinco configuraciones durante la búsqueda. MLP, CNN1D y ARX conservaron veinte. Esta desigualdad limita la comparación del potencial de las familias; los resultados describen las configuraciones efectivamente exploradas.'),
        code("""presupuesto=ajustes.groupby(['modelo','etapa']).size().unstack(fill_value=0).reindex(modelos)
presupuesto['configuraciones']=busqueda.groupby('modelo').size()
display(presupuesto.rename(index=etiquetas).rename(columns={'buscar':'ajustes búsqueda','piloto':'ajustes piloto','final':'ajustes finales'}))
fig,axes=plt.subplots(1,2,figsize=(12,4))
for nombre in modelos:
    d=busqueda.loc[busqueda.modelo==nombre].sort_values('config')
    axes[0].plot(np.arange(1,len(d)+1),d.MASE.cummin(),marker='o',ms=3,label=etiquetas[nombre])
    axes[1].scatter(d.n_parametros,d.MASE,label=etiquetas[nombre],s=20,alpha=.65)
axes[0].set(xlabel='Candidatos evaluados',ylabel='Menor MASE acumulado'); axes[0].legend(fontsize=8)
axes[1].set(xscale='log',xlabel='Número de parámetros',ylabel='MASE de selección')
fig.tight_layout(); display(fig); plt.close(fig)
seleccion_mejor=min(neuronales,key=lambda n:mejores[n]['MASE_cv']); cv_mejor=mejores[seleccion_mejor]['MASE_cv']
"""),
        note('MLP alcanza el menor MASE de selección entre las redes ({cv_mejor:.3f}). La figura relaciona capacidad y error de los candidatos, sin demostrar que aumentar parámetros produzca una mejora. La curva LSTM termina en el quinto candidato; no equivale al presupuesto de veinte aplicado a las otras familias.'),
        md('## 10.4 Piloto y dimensionamiento\n\nEl piloto utiliza cinco semillas, 100–104, sobre los mismos cuatro pliegues. El dimensionamiento original aplica una aproximación normal de dos muestras y adopta el máximo requerido entre las cuatro familias, con margen relativo del 3 %, α = 0,05 y potencia objetivo 0,80. Se conserva esa decisión y se contrasta con un cálculo pareado retrospectivo.'),
        code("""piloto_est=piloto.groupby('modelo').MASE.agg(['count','mean','std']).reindex(modelos)
piloto_est['CV_pct']=100*piloto_est['std']/piloto_est['mean']
zpot=stats.norm.ppf(.975)+stats.norm.ppf(.8)
piloto_est['n_original']=np.ceil(2*(zpot*piloto_est['std']/(.03*piloto_est['mean']))**2).astype(int)
display(piloto_est.rename(index=etiquetas).rename(columns={'count':'Semillas','mean':'MASE medio','std':'DE'}).round(4))
n_final=int(json.loads((R/'piloto_resumen.json').read_text())['n_final'])
assert n_final==195 and piloto_est['n_original'].max()==n_final
fig,ax=plt.subplots()
ax.barh([etiquetas[n] for n in modelos],100*piloto_est['std']/piloto_est['mean'])
ax.set_xlabel('Coeficiente de variación del MASE piloto (%)')
fig.tight_layout(); display(fig); plt.close(fig)
cv_lstm=float(piloto_est.loc['lstm','CV_pct'])
"""),
        note('LSTM presenta un coeficiente de variación del {cv_lstm:.2f} % en el piloto y determina las {n_final} semillas finales comunes. El cálculo original dimensiona variación de inicialización bajo una aproximación de dos muestras; no estima potencia sobre días ni sustituye el dimensionamiento con diferencias pareadas solicitado por la pauta.'),
        md('## 10.5 Benchmark de prueba\n\nLas métricas se reconstruyen desde las 780 filas finales. Los intervalos t al 95 % describen variación entre inicializaciones sobre la misma prueba; no son intervalos de muestreo de días. Los tiempos corresponden al ajuste, no a latencia de inferencia.'),
        code("""filas=[]
for nombre in modelos:
    d=runs[runs.modelo==nombre]; n=len(d); media=d.MASE.mean(); sd=d.MASE.std(ddof=1)
    mitad=stats.t.ppf(.975,n-1)*sd/np.sqrt(n)
    filas.append({'Modelo':etiquetas[nombre],'Familia':nombre,'n':n,'MASE':media,'DE':sd,
        'IC95_inf':media-mitad,'IC95_sup':media+mitad,
        'MAE_kWh':d.MAE.mean(),'RMSE_kWh':d.RMSE.mean(),'sMAPE_pct':d.sMAPE.mean(),
        'Ajuste_s_mediana':d.segundos.median(),'Parámetros':d.n_parametros.iloc[0]})
for nombre,v in referencias.items():
    filas.append({'Modelo':etiquetas[nombre],'Familia':nombre,'n':1,'MASE':v['MASE'],
        'MAE_kWh':v['MAE'],'RMSE_kWh':v['RMSE'],'sMAPE_pct':v['sMAPE']})
benchmark=pd.DataFrame(filas).sort_values('MASE').reset_index(drop=True)
benchmark['Orden']=np.arange(1,len(benchmark)+1)
display(benchmark.set_index('Modelo')[['Orden','n','MASE','DE','IC95_inf','IC95_sup','MAE_kWh','RMSE_kWh','sMAPE_pct','Ajuste_s_mediana','Parámetros']].round(4))
benchmark.to_csv(OUT/'benchmark.csv',index=False)
baseline=benchmark.loc[benchmark.Familia=='mlp'].iloc[0]
reduccion_semanal=float(100*(1-baseline.MASE/referencias['ingenuo_168h']['MASE']))
umbral_practico=float(.97*baseline.MASE)
"""),
        note('MLP establece el baseline empírico con MASE {baseline.MASE:.3f} y MAE {baseline.MAE_kWh:,.0f} kWh por intervalo; reduce el error un {reduccion_semanal:.1f} % respecto del ingenuo semanal. LSTM y CNN1D no superan esa referencia. Una reducción práctica del 3 % respecto de MLP corresponde a MASE inferior a {umbral_practico:.3f}; demostrarla exige además una comparación pareada con incertidumbre adecuada.'),
        md('## 10.6 Distribución, generalización y costo'),
        code("""matriz=runs.pivot(index='semilla',columns='modelo',values='MASE').reindex(columns=modelos)
fig,axes=plt.subplots(1,2,figsize=(12,4))
axes[0].boxplot([matriz[n] for n in modelos],tick_labels=[etiquetas[n] for n in modelos],showfliers=True)
axes[0].axhline(referencias['ingenuo_168h']['MASE'],ls='--',color='grey',label='Ingenuo semanal')
axes[0].set_ylabel('MASE de prueba por semilla'); axes[0].legend(fontsize=8)
for nombre in modelos:
    d=runs[runs.modelo==nombre]
    axes[1].scatter(d.segundos,d.MASE,label=etiquetas[nombre],s=10,alpha=.4)
axes[1].set(xscale='log',xlabel='Tiempo de ajuste (s)',ylabel='MASE de prueba'); axes[1].legend(fontsize=8)
fig.tight_layout(); display(fig); plt.close(fig)
de_lstm=float(matriz.lstm.std()); de_mlp=float(matriz.mlp.std())
"""),
        note('La dispersión entre semillas es mayor en LSTM (DE {de_lstm:.3f}) que en MLP ({de_mlp:.3f}). El diagrama de costo y error permite valorar cada ajuste sin confundir tiempo de entrenamiento con latencia. Las semillas comparten el mismo periodo; su dispersión representa variabilidad algorítmica condicionada a esos datos.'),
        code("""comparacion=pd.read_csv(R/'comparacion_validacion_prueba.csv').set_index('modelo').reindex(modelos)
fig,axes=plt.subplots(1,2,figsize=(12,4)); pos=np.arange(len(modelos))
axes[0].bar(pos-.18,comparacion.MASE_validacion_piloto,width=.36,label='Piloto de validación')
axes[0].bar(pos+.18,comparacion.MASE_prueba,width=.36,label='Prueba')
axes[0].set(xticks=pos,xticklabels=[etiquetas[n] for n in modelos],ylabel='MASE'); axes[0].legend(fontsize=8)
for nombre in modelos:
    valores=[runs.loc[runs.modelo==nombre,f'MASE_h{h}'].mean() for h in [1,12,24]]
    axes[1].scatter([1,12,24],valores,label=etiquetas[nombre])
for nombre,v in referencias.items():
    axes[1].scatter([1,12,24],[v[f'MASE_h{h}'] for h in [1,12,24]],marker='x',label=etiquetas[nombre])
axes[1].set(xticks=[1,12,24],xlabel='Hora del horizonte',ylabel='MASE reportado'); axes[1].legend(fontsize=7,ncol=2)
fig.tight_layout(); display(fig); plt.close(fig)
brecha_mlp=float(comparacion.loc['mlp','brecha_MASE_pct'])
"""),
        note('El MASE de MLP aumenta un {brecha_mlp:.1f} % entre piloto y prueba. La comparación combina periodos y tamaños de historial distintos; la brecha no identifica por sí sola sobreajuste. Los puntos de la derecha corresponden sólo a h = 1, 12 y 24: no se interpolan los horizontes intermedios ni se reconstruye una curva de 24 errores a partir de tres valores.'),
        md('## 10.7 Parada temprana y pérdidas guardadas\n\nEl resumen de ajustes conserva la última pérdida de entrenamiento, la última pérdida interna y su mínimo, junto con épocas totales y época de mínimo reportada. Estos indicadores permiten describir la parada, pero no reconstruyen las curvas completas por época.'),
        code("""final_hist=ajustes.loc[ajustes.etapa=='final'].copy()
fig,axes=plt.subplots(1,2,figsize=(12,4))
axes[0].boxplot([runs.loc[runs.modelo==n,'mejor_epoca'] for n in modelos],tick_labels=[etiquetas[n] for n in modelos])
axes[0].set_ylabel('Época de menor pérdida interna reportada')
for nombre in modelos:
    d=final_hist[final_hist.modelo==nombre]
    # Las pérdidas sólo son comparables dentro de una misma función y escala.
    cociente=d.perdida_validacion_final/d.perdida_validacion_minima
    axes[1].scatter(d.epocas,cociente,label=etiquetas[nombre],s=10,alpha=.45)
axes[1].set(xlabel='Épocas ejecutadas',ylabel='Pérdida interna final / mínima'); axes[1].legend(fontsize=8)
fig.tight_layout(); display(fig); plt.close(fig)
tope=int((runs.epocas>=200).sum())
"""),
        note('{tope} ajustes finales alcanzan el máximo de 200 épocas. Un cociente mayor que uno indica que el último estado no coincide con el mínimo de pérdida; el pipeline restaura el estado seleccionado por parada temprana. No se comparan directamente pérdidas MAE, Huber y MSE entre familias ni se presentan estos resúmenes como curvas de aprendizaje.')]
    books['09_modelos']=c
    c.extend([
        md('## 10.8 Comparaciones pareadas entre inicializaciones\n\nEl análisis secundario se fija en plan_analisis_secundario.json después de la publicación de los resultados; no es un preregistro anterior al entrenamiento. Las semillas 1000–1194 se aparean entre familias. Se estiman diferencias medias e intervalos bootstrap por semillas, t pareada y Wilcoxon bilateral como sensibilidad; Holm se aplica a tres contrastes neuronales y, por separado, a tres comparaciones con ARX.\n\nLa unidad de este análisis es la inicialización sobre una prueba fija, distinta del origen diario del pronóstico. Los contrastes cuantifican variabilidad algorítmica condicionada a 2022–2025; no demuestran superioridad temporal en días no observados. Shapiro–Wilk y Q–Q describen las diferencias; la prueba por rangos también requiere considerar su simetría.'),
        code("""rng=np.random.default_rng(analisis['semilla_analisis'])
B=analisis['bootstrap_replicas']; pares=[]
familias={'Neuronales':list(itertools.combinations(neuronales,2)),
          'Contexto ARX':[(n,'arx_lineal') for n in neuronales]}
qq=[]
for familia,comparaciones in familias.items():
    for a,b in comparaciones:
        va=matriz[a].to_numpy(); vb=matriz[b].to_numpy(); d=vb-va
        n=len(d); sd=float(d.std(ddof=1)); indices=rng.integers(0,n,size=(B,n))
        delta_boot=d[indices].mean(axis=1)
        mejora_boot=100*(1-va[indices].mean(axis=1)/vb[indices].mean(axis=1))
        ci=np.quantile(delta_boot,[.025,.975]); ci_rel=np.quantile(mejora_boot,[.025,.975])
        t=stats.ttest_rel(vb,va)
        w=stats.wilcoxon(np.round(d,12),alternative='two-sided',method='asymptotic')
        sh=stats.shapiro(d)
        rangos=stats.rankdata(np.abs(d)); rb=float(np.sum(np.sign(d)*rangos)/rangos.sum())
        pares.append({'Familia':familia,'A':a,'B':b,'Delta_B_menos_A':d.mean(),
            'IC95_inf':ci[0],'IC95_sup':ci[1],'Reduccion_A_pct':100*(1-va.mean()/vb.mean()),
            'IC95_reduccion_inf':ci_rel[0],'IC95_reduccion_sup':ci_rel[1],
            'Cohen_dz':d.mean()/sd,'Rango_biserial':rb,'Victorias_A_pct':100*np.mean(d>0),
            't_p':float(t.pvalue),'Wilcoxon_p':float(w.pvalue),'Shapiro_p':float(sh.pvalue),
            'Asimetria_diferencias':float(stats.skew(d,bias=False))})
        qq.append((a,b,d))
contrastes=pd.DataFrame(pares)
for familia,idx in contrastes.groupby('Familia').groups.items():
    for prueba in ['t','Wilcoxon']:
        contrastes.loc[idx,prueba+'_Holm']=multipletests(contrastes.loc[idx,prueba+'_p'],method='holm')[1]
display(contrastes[['Familia','A','B','Delta_B_menos_A','IC95_inf','IC95_sup','Reduccion_A_pct',
    'IC95_reduccion_inf','IC95_reduccion_sup','Cohen_dz','t_p','t_Holm','Wilcoxon_p','Wilcoxon_Holm','Shapiro_p']].style.format(precision=4,
    formatter={k:'{:.3g}' for k in ['t_p','t_Holm','Wilcoxon_p','Wilcoxon_Holm','Shapiro_p']}))
contrastes.to_csv(OUT/'contrastes_semillas.csv',index=False)
fig,axes=plt.subplots(2,3,figsize=(13,6))
for ax,(a,b,d) in zip(axes.ravel(),qq):
    stats.probplot(d,dist='norm',plot=ax)
    ax.set(title=etiquetas[b]+' − '+etiquetas[a],xlabel='Cuantiles normales',ylabel='Diferencia MASE')
fig.tight_layout(); display(fig); plt.close(fig)
principal=contrastes[contrastes.Familia=='Neuronales']
n_rechazos=int((principal.t_Holm<.05).sum())
peor_shapiro=float(principal.Shapiro_p.min())
"""),
        note('Los {n_rechazos} contrastes de medias entre redes rechazan igualdad tras Holm en este análisis por semillas. Los Q–Q permiten reconocer colas y asimetría que la tabla no resume; el menor p de Shapiro–Wilk es {peor_shapiro:.3g}. Los intervalos y tamaños de efecto corresponden a inicializaciones sobre los mismos días, sin una conclusión confirmatoria sobre generalización temporal.'),
        md('## 10.9 Rangos y descomposición de variabilidad\n\nFriedman se calcula sobre las tres redes con semillas apareadas. Su p se calibra mediante 19.999 permutaciones de etiquetas dentro de cada semilla para evitar depender exclusivamente de la aproximación asintótica con tres métodos. Nemenyi se presenta como post-hoc aproximado de rangos. La descomposición de sumas de cuadrados separa diferencias entre familias, efecto común de semilla e interacción; no representa una estimación de ruido irreducible.'),
        code("""v=matriz[neuronales].to_numpy(); n,k=v.shape
rangos=stats.rankdata(v,axis=1); rangos_medios=rangos.mean(axis=0)
friedman=stats.friedmanchisquare(*[v[:,j] for j in range(k)])
q_sin_corr=12*n/(k*(k+1))*np.sum((rangos_medios-(k+1)/2)**2)
correccion=q_sin_corr/friedman.statistic if friedman.statistic>0 else 1.
rng=np.random.default_rng(11037); excedencias=0; Bf=analisis['permutaciones_friedman']
for inicio in range(0,Bf,1000):
    m=min(1000,Bf-inicio)
    perm=np.argsort(rng.random((m,n,k)),axis=2)
    rp=np.take_along_axis(np.broadcast_to(rangos,(m,n,k)),perm,axis=2).mean(axis=1)
    qp=12*n/(k*(k+1))*np.sum((rp-(k+1)/2)**2,axis=1)/correccion
    excedencias+=int(np.sum(qp>=friedman.statistic-1e-12))
p_perm=(excedencias+1)/(Bf+1)
se_rangos=np.sqrt(k*(k+1)/(6*n))
cd=float(stats.studentized_range.ppf(.95,k,np.inf)/np.sqrt(2)*se_rangos)
nemenyi=[]
for i,j in itertools.combinations(range(k),2):
    q=abs(rangos_medios[i]-rangos_medios[j])/se_rangos*np.sqrt(2)
    nemenyi.append({'A':etiquetas[neuronales[i]],'B':etiquetas[neuronales[j]],
        'Diferencia_rango':abs(rangos_medios[i]-rangos_medios[j]),
        'p_Nemenyi_aprox':float(stats.studentized_range.sf(q,k,np.inf))})
display(pd.DataFrame(nemenyi))
x=matriz.to_numpy(); total=x.mean(); medias_modelo=x.mean(axis=0); medias_semilla=x.mean(axis=1)
ss_modelo=len(x)*np.sum((medias_modelo-total)**2)
ss_semilla=x.shape[1]*np.sum((medias_semilla-total)**2)
ss_interaccion=np.sum((x-medias_modelo[None,:]-medias_semilla[:,None]+total)**2)
ss=np.array([ss_modelo,ss_semilla,ss_interaccion]); fracciones=100*ss/ss.sum()
fig,axes=plt.subplots(1,2,figsize=(12,4))
axes[0].barh([etiquetas[a] for a in neuronales],rangos_medios)
axes[0].set(xlabel='Rango medio por semilla (menor es mejor)',xlim=(0,3.6))
axes[0].plot([.5,.5+cd],[2.5,2.5],color='black',lw=2); axes[0].text(.5,2.6,'CD 0,05',fontsize=8)
axes[1].barh(['Familia','Semilla común','Interacción'],fracciones); axes[1].set_xlabel('Suma de cuadrados descriptiva (%)')
fig.tight_layout(); display(fig); plt.close(fig)
salida_rangos={'friedman_Q':float(friedman.statistic),'p_permutacion':float(p_perm),
    'replicas':Bf,'rangos':dict(zip(neuronales,map(float,rangos_medios))),
    'diferencia_critica':cd,'varianza_descriptiva_pct':list(map(float,fracciones))}
(OUT/'rangos_variabilidad.json').write_text(json.dumps(salida_rangos,indent=2),encoding='utf-8')
"""),
        note('Friedman obtiene Q = {friedman.statistic:.2f} y p de permutación {p_perm:.5f}, con resolución limitada por {Bf} réplicas. La diferencia crítica aproximada es {cd:.3f} rangos. Las diferencias entre familias explican el {fracciones[0]:.1f} % de la suma de cuadrados del cuadro de resultados. Estas semillas no representan bases de datos independientes; la lectura de rangos permanece condicionada a la prueba compartida.'),
        md('## 10.10 Dimensionamiento pareado y relevancia práctica\n\nSe utiliza el piloto, no las diferencias finales, para estimar la desviación de diferencias entre semillas coincidentes. El cálculo retrospectivo aplica la aproximación normal pareada de la pauta, un margen del 3 % del MASE piloto del comparador y un nivel conservador Bonferroni por familia. Se reporta la diferencia mínima detectable con 195 semillas; no se modifica el tamaño de las ejecuciones ya terminadas.'),
        code("""pil=piloto.pivot(index='semilla',columns='modelo',values='MASE')
dim=[]
for familia,comparaciones in familias.items():
    m=len(comparaciones); z=stats.norm.ppf(1-.05/(2*m))+stats.norm.ppf(.8)
    for a,b in comparaciones:
        dif=(pil[b]-pil[a]).dropna(); sd=float(dif.std(ddof=1))
        delta=.03*float(pil[b].mean()); R_pareado=max(1,int(np.ceil((z*sd/delta)**2)))
        mde=z*sd/np.sqrt(n_final)
        dim.append({'Familia':familia,'A':a,'B':b,'n_piloto_pares':len(dif),
            'DE_diferencias':sd,'delta_MASE':delta,'delta_sobre_DE':delta/sd,
            'n_estimado_pareado':R_pareado,'MDE_con_195':mde,
            'MDE_relativa_pct':100*mde/pil[b].mean(),'MDE_supera_delta':bool(mde>delta)})
dimensionamiento=pd.DataFrame(dim)
display(dimensionamiento.style.format(precision=4))
dimensionamiento.to_csv(OUT/'dimensionamiento_pareado.csv',index=False)
limite_mde=float(dimensionamiento.loc[dimensionamiento.Familia=='Neuronales','MDE_relativa_pct'].max())
"""),
        note('La mayor diferencia mínima detectable estimada entre redes es {limite_mde:.2f} % bajo los supuestos pareados y la dispersión de sólo cinco semillas piloto. Si supera el 3 %, las 195 ejecuciones no garantizan el objetivo de potencia para ese contraste. El valor es una aproximación de diseño, no potencia temporal ni evidencia de equivalencia.'),
        md('## 10.11 Alcance del cumplimiento y artefactos necesarios\n\n| Requisito del E1 | Evidencia incorporada | Alcance |\n|---|---|---|\n| Tres familias de Deep Learning, configuración y pipeline | MLP, LSTM y CNN1D; ARX como contexto; código íntegro en el apéndice | La frecuencia de uso en una revisión sistemática sigue sin completarse. |\n| Piloto, semillas y variabilidad | Cinco semillas piloto y 195 finales; medias, DE, IC y dimensionamiento original y pareado | Inferencia condicional a una prueba compartida. |\n| Equidad de búsqueda | Espacios y número de ajustes documentados | LSTM tuvo cinco candidatos frente a veinte; no cumple presupuesto igual. |\n| Ranking y baseline | MASE, MAE, RMSE, sMAPE, tiempos de ajuste, parámetros y margen del 3 % | MLP es el baseline empírico de estas configuraciones. |\n| Curvas de aprendizaje | Épocas y pérdidas resumidas | Faltan historiales completos v2 por época. |\n| Errores por horizonte y segmentos | Horizontes 1, 12 y 24 | Faltan predicciones fechadas para las 24 horas, años, tipos de día y rangos del objetivo. |\n| Inferencia temporal y residuos | Procedimiento y datos requeridos identificados | DM por horizonte, Ljung–Box, BDS y bootstrap por días no se calculan desde métricas por semilla. |\n| Reproducción exacta y costo | Fuentes, configuraciones, hashes y hardware | Pesos, escaladores, predicciones v2 y latencia de inferencia no están en GitHub. |\n\nLas fuentes numéricas y el conjunto de prueba son retrospectivos; la versión 2 se diseñó después de consultar resultados de la versión 1, conforme a las adendas. Los contrastes secundarios no corrigen esa limitación. No se mezclan predicciones de la versión 1 con métricas de la versión 2 ni se fabrican curvas o diagnósticos ausentes.'),
        md('## 10.12 Código y reproducción\n\nEl [apéndice del pipeline](10_pipeline.md) contiene el código completo de ventanas, partición, escalado, arquitecturas, pérdidas, optimizador, parada temprana, búsqueda, piloto y prueba, extraído de la instantánea verificada. El capítulo muestra además todo el código utilizado para reconstruir tablas, gráficos y contrastes.\n\nLos resultados se conservan en results/repositorio/'+SHA+'/results/base_v2. El registro IMPORTACION.json identifica la revisión descargada y sus archivos; el plan del análisis secundario y las tablas derivadas están en results/jbook/modelos. La importación no modifica las fuentes activas de entrenamiento.\n\nFundamento de rangos: [Demšar (2006)](https://jmlr.org/papers/v7/demsar06a.html). Las condiciones de [Wilcoxon](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html) se consideran para interpretar asimetrías. La permutación calibra el estadístico de [Friedman](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.friedmanchisquare.html) con tres redes.')])

    # Mostrar las fuentes completas como bloques nativos, sin ejecutarlas al leer.
    fuentes=['src/demanda/config.py','src/demanda/particion.py','src/demanda/ventanas.py',
             'src/demanda/metricas.py','src/demanda/modelos_base.py',
             'scripts/entrenar_modelos_base.py','scripts/ejecutar_gpu.py']
    fence=chr(96)*3
    paginas=['# Código completo del pipeline de entrenamiento y prueba',
        'Las fuentes corresponden al commit '+SHA+'. Los bloques siguientes conservan el código de la ejecución; el análisis del libro no invoca sus funciones de entrenamiento. El controlador GPU registra caché, pesos y predicciones por ajuste; la interfaz central admite CPU y GPU.',
        '## Reproducción desde una copia independiente',
        'Utilizar una copia del commit indicado con el mismo panel y registro de variables en data/processed. Instalar las dependencias del entrenamiento y la variante de PyTorch adecuada al dispositivo. Una reproducción debe escribir en un directorio nuevo y conservar las adendas y el límite LSTM.',
        fence+'python\nfrom pathlib import Path\nimport shutil\nsalida=Path("results/reproduccion_e1")\nsalida.mkdir(parents=True,exist_ok=True)\nshutil.copyfile("results/base_v2/limites_busqueda.json",salida/"limites_busqueda.json")\n'+fence,
        fence+'bash\npython scripts/entrenar_modelos_base.py buscar --salida results/reproduccion_e1 --n-config 20 --semillas-por-config 3 --max-epocas 200 --dispositivo auto\npython scripts/entrenar_modelos_base.py piloto --salida results/reproduccion_e1 --max-epocas 200 --dispositivo auto\npython scripts/entrenar_modelos_base.py final --salida results/reproduccion_e1 --max-epocas 200 --n-semillas 195 --dispositivo auto --allow-test --prereg PREREGISTRO.md\n'+fence,
        'Los comandos son una receta de reproducción y no se ejecutan al construir este libro. El controlador original de la GPU utiliza fcntl y es específico de Linux; el pipeline central anterior puede invocarse desde Windows o Linux con las dependencias apropiadas. No se afirma identidad bit a bit entre dispositivos.']
    for nombre in fuentes:
        contenido=(snapshot/nombre).read_text(encoding='utf-8')
        ast.parse(contenido,filename=nombre)
        paginas.extend(['## '+nombre,fence+'python\n'+contenido.rstrip()+'\n'+fence])
    (root/'book/10_pipeline.md').write_text('\n\n'.join(paginas)+'\n',encoding='utf-8')

    # Conciliar el diseño documentado con la implementación realmente ejecutada.
    for cell in books['08_validacion']:
        if cell.cell_type=='markdown':
            cell.source=cell.source.replace('La versión 2 fija el mismo presupuesto por familia: 20 configuraciones, tres semillas y cuatro pliegues anuales.',
                'La ejecución v2 evaluó veinte configuraciones para MLP, CNN1D y ARX, y cinco para LSTM tras una adenda de costo computacional, con tres semillas y cuatro pliegues anuales. El presupuesto ejecutado es desigual.')
            cell.source=cell.source.replace('Imputación y estandarización se ajustan con el tramo de optimización de cada pliegue.',
                'El pipeline ejecutado fija el estandarizador inicial con 2000–2017 y la escala MASE con ese mismo bloque; MLP y ARX añaden estandarización tabular ajustada con el tramo de optimización. La reserva interna del primer pliegue participa en el escalado inicial, limitación documentada en el capítulo de modelos.')
            cell.source=cell.source.replace('La comparación inferencial requiere un preregistro específico antes de calcular los contrastes:',
                'La inferencia temporal definitiva requiere un preregistro específico:')
            cell.source=cell.source.replace('La potencia se estimará mediante piloto.',
                'La ejecución v2 completó un piloto de cinco semillas y utilizó 195 semillas finales. El capítulo de modelos presenta el análisis secundario de esos resultados.')
        elif cell.cell_type=='code' and 'pre_2018=candidatos' in cell.source:
            inicio=cell.source.index('pre_2018=candidatos')
            fin=cell.source.index('; K_estandar=',inicio)
            cell.source=cell.source[:inicio]+"media_cal=train[calendario].mean(); escala_cal=train[calendario].std().replace(0,1)"+cell.source[fin:]
    books['08_literatura'].append(md('## Correspondencia con el benchmark ejecutado\n\nEl benchmark neuronal se integra por MLP, LSTM y CNN1D; ARX es una referencia lineal de contexto. MLP y LSTM tienen comparaciones directas de demanda en los estudios examinados. La familia convolucional se relaciona con filtros temporales y las comparaciones de TCN de Pelekis et al.; la CNN poco profunda implementada se especificó en PREREGISTRO_CNN1D.md y no se presenta como réplica exacta de una TCN ni del autoencoder textual de Bai et al. La síntesis dirigida no demuestra una frecuencia exhaustiva de uso de las tres arquitecturas.'))
    books['11_sintesis'].append(md('## Resultado del benchmark\n\nLa ejecución v2 completa la comparación de MLP, LSTM y CNN1D, con ARX y referencias estacionales. El capítulo 10 reconstruye el ranking y la variación entre 195 semillas a partir de resultados publicados. MLP obtiene el menor MASE; la desigualdad del presupuesto LSTM y la ausencia pública de predicciones y curvas completas limitan la inferencia temporal y la comprobación de todos los requisitos del E1. El análisis por semillas no sustituye la evaluación por días.'))

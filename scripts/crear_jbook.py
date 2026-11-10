"""Construye y ejecuta el libro de demanda exclusivamente con desarrollo."""
import base64,json,hashlib,shutil,sys,ast
from pathlib import Path
import nbformat as nb
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output
from IPython.core.pylabtools import print_figure
from matplotlib.figure import Figure

ROOT=Path(__file__).resolve().parents[1];BOOK=ROOT/'book'
SETUP='''from pathlib import Path
import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from IPython.display import display
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p/'data/processed/panel_horario.parquet').exists())
plt.rcParams.update({'figure.figsize':(10,4),'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.15})
panel = pq.read_table(ROOT/'data/processed/panel_horario.parquet', filters=[('inicio','<',pd.Timestamp('2022-01-01',tz='America/Bogota'))]).to_pandas()
assert panel.index.max() < pd.Timestamp('2022-01-01',tz='America/Bogota')
assert panel.index.is_monotonic_increasing and panel.index.is_unique
y = panel['y_demanda_real_kwh']
gwh = y / 1e6
registro = json.loads((ROOT/'data/processed/feature_registry.json').read_text(encoding='utf-8'))
'''

def md(s):return nb.v4.new_markdown_cell(s)
def code(s):return nb.v4.new_code_cell(s)
def note(s):
    cell=md(s);cell.metadata['plantilla_resultados']=s;return cell
def decision(d,i,r):return md(f'**Decisión.** {d}\n\n**Impacto.** {i}\n\n**Riesgo.** {r}')
def chapter(title,intro,cells):return [md('# '+title),md(intro),code(SETUP),*cells]

def main():
    BOOK.mkdir(exist_ok=True);(ROOT/'results/jbook').mkdir(parents=True,exist_ok=True)
    books={}
    books['01_datos']=chapter('1. Problema y datos','El objetivo es pronosticar la demanda real horaria de energía eléctrica del Sistema Interconectado Nacional de Colombia. Se formula un pronóstico de series de tiempo multivariado con salida directa de 24 intervalos y 168 horas de historia. El origen es diario a las 00:00 de Colombia. El estudio contempla la integración posterior de información documental.',[
        md('El conjunto numérico abarca 2000–2025. El análisis exploratorio caracteriza desarrollo 2000–2021; las etiquetas de prueba quedan excluidas. La demanda se expresa en kWh por intervalo horario; las figuras utilizan GWh para facilitar su lectura. Las entradas observadas se alinean según disponibilidad y el calendario puede conocerse para el horizonte.'),
        code("inventario=pd.DataFrame([{'bloque':'Panel de desarrollo','filas':len(panel),'columnas':len(panel.columns),'inicio':str(panel.index.min()),'fin':str(panel.index.max())}]); display(inventario)\nvariables=pd.DataFrame.from_dict(registro,orient='index'); display(variables.groupby('clase').size().rename('variables')); display(variables)"),
        md('Las demandas comercial y real no se consideran intercambiables: la comercial queda excluida como entrada. Temperatura, humedad y precipitación proceden de reanálisis ERA5; sus observaciones pasadas complementan el panel, pero no representan pronósticos meteorológicos del horizonte. La hidrología es diaria. Los promedios urbanos ponderados utilizan población estática del censo 2018, documentada mediante una fuente secundaria.'),
        decision('Conservar un registro explícito de rol y disponibilidad por variable.','Permite distinguir entradas pasadas, calendario y objetivo.','Los rezagos supuestos y las revisiones retrospectivas limitan la disponibilidad histórica verificable.')])
    books['02_calidad']=chapter('2. Calidad e integridad','Se comprueban continuidad horaria, duplicados, faltantes, rangos y observaciones extremas. Los valores anómalos se conservan en los datos originales.',[
        code("faltantes=len(pd.date_range(panel.index.min(),panel.index.max(),freq='h').difference(panel.index)); duplicados=panel.index.duplicated().sum(); display(pd.Series({'horas':len(panel),'horas_ausentes':faltantes,'duplicados':duplicados,'objetivo_nulo':int(y.isna().sum()),'objetivo_no_positivo':int((y<=0).sum())})); nulos=panel.isna().mean().sort_values(ascending=False); display(nulos[nulos>0].rename('fraccion_nula'))\nrazon=y/y.shift(168); anomalas=razon[razon<.5]; display(pd.DataFrame({'demanda_GWh':gwh.loc[anomalas.index],'razon_semanal':anomalas})); n_anomalas=len(anomalas)\nfig,axes=plt.subplots(1,2,figsize=(12,4)); axes[0].plot(y.index,razon,lw=.5); axes[0].axhline(.5,color='r',ls='--'); axes[0].set_ylabel('Demanda / demanda hace 168 h'); axes[1].barh(nulos.head(10).index,nulos.head(10)); axes[1].set_xlabel('Fracción de faltantes'); fig.tight_layout(); display(fig); plt.close(fig)"),
        note('Se identifican {faltantes} horas ausentes y {duplicados} duplicados en desarrollo. Hay {n_anomalas} horas con demanda inferior a la mitad de su referencia semanal; requieren revisión de fuente y contexto antes de cualquier corrección.'),
        decision('Mantener los extremos y documentar los nulos introducidos por rezagos.','Evita confundir indisponibilidad inicial con huecos de la fuente.','Eliminar observaciones atípicas sin evidencia puede suprimir eventos reales.')])
    books['03_demanda']=chapter('3. Distribución y evolución de la demanda','Se examinan el nivel, la dispersión y el crecimiento temporal de la demanda. Las estadísticas agregadas se calculan sobre observaciones horarias.',[
        code("resumen=gwh.describe(percentiles=[.01,.05,.5,.95,.99]); display(resumen); media=float(gwh.mean()); mediana=float(gwh.median()); asimetria=float(gwh.skew()); anual=gwh.groupby(gwh.index.year).agg(['mean','median','std','min','max']); display(anual); inicial=float(anual['mean'].iloc[0]); final=float(anual['mean'].iloc[-1]); crecimiento=100*(final/inicial-1)\nfig,axes=plt.subplots(1,2,figsize=(12,4)); gwh.resample('MS').mean().plot(ax=axes[0]); axes[0].set_ylabel('Demanda media (GWh/h)'); axes[1].hist(gwh,bins=70,color='#247ba0'); axes[1].axvline(media,color='r',label='Media'); axes[1].set_xlabel('Demanda (GWh por intervalo)'); axes[1].set_ylabel('Horas'); axes[1].legend(); fig.tight_layout(); display(fig); plt.close(fig)"),
        note('La demanda media anual pasa de {inicial:.2f} a {final:.2f} GWh por intervalo entre los extremos del desarrollo, un cambio del {crecimiento:.1f} %. La media global es {media:.2f} y la mediana {mediana:.2f}; la distribución reúne años con distintos niveles, por lo que no describe un régimen único.'),
        code("mensual=gwh.resample('MS').agg(['mean','std']); fig,axes=plt.subplots(1,2,figsize=(12,4)); axes[0].plot(mensual['mean'],mensual['std'],'.'); axes[0].set(xlabel='Media mensual (GWh)',ylabel='DE mensual (GWh)'); gwh.resample('D').mean().loc['2019':'2021'].plot(ax=axes[1]); axes[1].set_ylabel('Demanda media diaria (GWh/h)'); correlacion_nivel=float(mensual['mean'].corr(mensual['std'])); fig.tight_layout(); display(fig); plt.close(fig)"),
        note('La asociación entre nivel y dispersión mensual es {correlacion_nivel:.2f}. La ventana 2019–2021 permite distinguir cambios locales que el agregado de todo el histórico comprime; la comparación de errores absolutos entre regímenes debe considerar la escala de la demanda.'),
        decision('Examinar la normalización por el nivel de la ventana histórica.','Reduce la influencia del crecimiento sobre la representación de entradas y errores.','Una escala calculada con observaciones futuras produciría fuga temporal.')])
    books['04_estacionalidad']=chapter('4. Calendario y estacionalidad','Los perfiles horarios y semanales permiten estudiar ciclos recurrentes y diferencias entre días laborales, festivos y fines de semana.',[
        code("perfil=gwh.groupby([panel.index.day_of_week,panel.index.hour]).mean().unstack(0); pico=int(gwh.groupby(panel.index.hour).mean().idxmax()); valle=int(gwh.groupby(panel.index.hour).mean().idxmin()); fig,axes=plt.subplots(1,2,figsize=(12,4)); perfil.plot(ax=axes[0]); axes[0].set(xlabel='Hora',ylabel='Demanda media (GWh)'); axes[0].legend(['Lun','Mar','Mié','Jue','Vie','Sáb','Dom'],fontsize=8); imagen=axes[1].imshow(perfil.T,aspect='auto',cmap='viridis'); axes[1].set(xlabel='Hora',ylabel='Día de semana',yticks=range(7),yticklabels=['Lun','Mar','Mié','Jue','Vie','Sáb','Dom']); fig.colorbar(imagen,ax=axes[1],label='GWh'); fig.tight_layout(); display(fig); plt.close(fig)"),
        note('El perfil agregado alcanza su máximo a las {pico:02d}:00 y su mínimo a las {valle:02d}:00. La matriz muestra cómo varía el ciclo diario entre días de semana; los promedios mezclan niveles de distintos años y no representan efectos causales del calendario.'),
        code("laboral=gwh[panel.cal_es_no_laboral==0].mean(); no_laboral=gwh[panel.cal_es_no_laboral==1].mean(); diferencia=100*(no_laboral/laboral-1); fig,axes=plt.subplots(1,2,figsize=(12,4)); pd.DataFrame({'GWh':gwh,'hora':panel.index.hour,'no_laboral':panel.cal_es_no_laboral}).groupby(['hora','no_laboral']).GWh.mean().unstack().plot(ax=axes[0]); axes[0].set_ylabel('Demanda media (GWh)'); gwh.groupby(panel.index.month).mean().plot.bar(ax=axes[1]); axes[1].set(xlabel='Mes',ylabel='Demanda media (GWh)'); fig.tight_layout(); display(fig); plt.close(fig)"),
        note('La demanda media en días no laborales difiere un {diferencia:.1f} % de la de días laborales. El ciclo mensual y el intradiario aconsejan conservar calendario del horizonte; su asociación puede variar entre años.'),
        code("from statsmodels.tsa.seasonal import STL\ndiaria=gwh.resample('D').mean(); stl=STL(diaria,period=7,robust=True).fit(); fig=stl.plot(); fig.set_size_inches(11,7); display(fig); plt.close(fig); amplitud=float(stl.seasonal.quantile(.95)-stl.seasonal.quantile(.05)); residuo=float(stl.resid.std())"),
        note('La componente semanal de la demanda diaria presenta un rango p5–p95 de {amplitud:.2f} GWh, con DE residual {residuo:.2f}. La tendencia recoge variaciones lentas; el residuo conserva episodios que el ciclo semanal no representa. Esta descomposición es retrospectiva y no se utiliza como entrada del horizonte.'),
        decision('Incluir hora, día de semana y festivos conocidos de los próximos 24 intervalos.','Representa ciclos sin consultar covariables observadas del futuro.','Promedios globales pueden ocultar cambios estacionales y estructurales.')])
    books['05_dependencia']=chapter('5. Dependencia temporal y estacionariedad','La dependencia se estudia en escala horaria; las pruebas de raíz unitaria y estacionariedad se aplican a la media diaria y a su diferencia.',[
        code("from statsmodels.tsa.stattools import acf,adfuller,kpss\nfrom scipy.signal import periodogram\nv=acf(gwh,nlags=336,fft=True); a24=float(v[24]); a168=float(v[168]); fig,axes=plt.subplots(1,2,figsize=(12,4)); axes[0].plot(range(337),v); axes[0].set(xlabel='Rezago (h)',ylabel='ACF'); f,p=periodogram(gwh.to_numpy(),detrend='linear'); sel=(f>0)&(1/np.maximum(f,1e-12)<1000); axes[1].plot(1/f[sel],p[sel]); axes[1].set(xlabel='Periodo (h)',ylabel='Potencia espectral',xlim=(0,350)); fig.tight_layout(); display(fig); plt.close(fig)"),
        note('La autocorrelación a 24 y 168 horas es {a24:.3f} y {a168:.3f}, respectivamente. El espectro complementa estos rezagos al localizar ciclos; la tendencia puede elevar la ACF global y no debe confundirse con capacidad predictiva fuera de muestra.'),
        code("import warnings\nregistros=[]\nfor nombre,serie in [('Media diaria',gwh.resample('D').mean()),('Diferencia diaria',gwh.resample('D').mean().diff().dropna())]:\n    with warnings.catch_warnings():\n        warnings.simplefilter('ignore'); a=adfuller(serie,maxlag=30,autolag='AIC'); k=kpss(serie,regression='c',nlags='auto')\n    registros.append({'serie':nombre,'ADF_estadistico':a[0],'ADF_p':a[1],'KPSS_estadistico':k[0],'KPSS_p_tabulado':k[1]})\ndisplay(pd.DataFrame(registros))"),
        md('ADF contrasta raíz unitaria y KPSS estacionariedad alrededor de una constante. Se interpretan conjuntamente a frecuencia diaria. Los p de KPSS pueden corresponder al límite de su tabla; una diferencia no garantiza estabilidad frente a rupturas.'),
        decision('Conservar rezagos diarios y semanales como referencias temporales.','Permite evaluar memoria y ciclos presentes en los datos.','La persistencia y la tendencia pueden generar correlaciones altas sin mejoras de pronóstico.')])
    books['06_covariables']=chapter('6. Covariables y disponibilidad','Se caracteriza el panel ampliado con observaciones meteorológicas e hidrológicas pasadas. Las asociaciones descriptivas no implican disponibilidad futura ni causalidad.',[
        code("seleccion=['y_demanda_real_kwh','x_demanda_real_kwh','x_precio_bolsa_cop_kwh','x_obs_temperature_2m_c_ponderada','x_obs_relative_humidity_2m_pct_ponderada','x_obs_precipitation_preceding_hour_mm_ponderada','x_hid_llenado_valor_original','x_hid_aportes_kwh_dia']; correlacion=panel[seleccion].corr(method='spearman'); display(correlacion); fig,ax=plt.subplots(figsize=(10,7)); im=ax.imshow(correlacion,vmin=-1,vmax=1,cmap='RdBu_r'); ax.set_xticks(range(len(seleccion)),labels=seleccion,rotation=90,fontsize=8); ax.set_yticks(range(len(seleccion)),labels=seleccion,fontsize=8); fig.colorbar(im,ax=ax,label='Spearman'); fig.tight_layout(); display(fig); plt.close(fig); asociacion=float(correlacion.loc['y_demanda_real_kwh','x_obs_temperature_2m_c_ponderada'])"),
        note('La correlación de Spearman entre demanda y temperatura ponderada rezagada es {asociacion:.3f}. La matriz permite detectar asociaciones y redundancias; las tendencias compartidas y el calendario impiden interpretarlas como contribuciones causales.'),
        code("ciudades=['bogota','medellin','cali','barranquilla','cartagena','bucaramanga','pereira','cucuta']; fig,axes=plt.subplots(1,3,figsize=(14,4));\nfor ax,var,titulo in zip(axes,['temperature_2m_c','relative_humidity_2m_pct','precipitation_preceding_hour_mm'],['Temperatura (°C)','Humedad (%)','Precipitación (mm/h)']):\n    for ciudad in ciudades:\n        z=panel['x_obs_'+var+'_'+ciudad]; ax.plot(z.groupby(z.index.month).mean(),label=ciudad)\n    ax.set(xlabel='Mes',ylabel=titulo)\naxes[0].legend(fontsize=7); fig.tight_layout(); display(fig); plt.close(fig)"),
        md('Los perfiles urbanos muestran heterogeneidad espacial y estacional. La ponderación por población resume ciudades, pero no describe lluvia en cuencas aportantes ni sustituye pronósticos con emisión previa. Los pesos estáticos de 2018 introducen una limitación temporal para años anteriores.'),
        decision('Distinguir el conjunto de demanda y calendario del panel ampliado.','Permite evaluar ampliaciones con una regla de disponibilidad explícita.','Reanálisis, población estática y revisiones de fuente pueden introducir sesgos retrospectivos.')])
    books['07_texto']=chapter('7. Exploración de titulares','El corpus se analiza por disponibilidad, fuente y características léxicas. Los tópicos se aprenden automáticamente; no se asignan categorías temáticas manuales.',[
        code("coberturas={}\nfor escenario in ['declarado','estricto']:\n    c=pd.read_csv(ROOT/f'data/processed/cobertura_titulares_{escenario}.csv',index_col=0); c.index=c.index.astype(int); coberturas[escenario]=c.loc[c.index<2022]\nfig,axes=plt.subplots(1,2,figsize=(12,4));\nfor ax,(escenario,c) in zip(axes,coberturas.items()):\n    c.plot(ax=ax); ax.set(title=escenario,xlabel='Año de disponibilidad',ylabel='Titulares'); ax.legend(fontsize=6)\nfig.tight_layout(); display(fig); plt.close(fig); total_declarado=int(coberturas['declarado'].sum().sum()); total_estricto=int(coberturas['estricto'].sum().sum())"),
        note('Desarrollo contiene {total_declarado:,} registros en el escenario declarado y {total_estricto:,} en el estricto. Son escenarios alternativos, no conjuntos para sumar. Las diferencias de cobertura limitan comparaciones por fuente y tiempo; ausencia de registros recuperados no equivale a ausencia de publicación.'),
        code("tit=pq.read_table(ROOT/'data/raw/noticias/titulares.parquet',columns=['source','titular','available_at_assumed','quarantined'],filters=[('available_at_assumed','<','2018-01-01'),('available_at_assumed','>=','2000-01-01'),('quarantined','==',False)]).to_pandas(); tit=tit[tit.titular.notna() & tit.titular.str.strip().ne('')]; muestra=tit.sample(min(30000,len(tit)),random_state=11037); from sklearn.feature_extraction.text import TfidfVectorizer\nfrom sklearn.decomposition import NMF\nvectorizador=TfidfVectorizer(lowercase=True,strip_accents='unicode',min_df=10,max_df=.8,max_features=6000,stop_words=['de','la','el','en','y','a','los','del','las','un','una','por','para','con','que','se','al','su','es','no','lo']); X=vectorizador.fit_transform(muestra.titular); nmf=NMF(n_components=6,random_state=11037,max_iter=400); W=nmf.fit_transform(X); palabras=vectorizador.get_feature_names_out(); frecuencia=np.asarray(X.sum(axis=0)).ravel(); top=np.argsort(frecuencia)[-20:]; fig,ax=plt.subplots(); ax.barh(palabras[top],frecuencia[top]); ax.set_xlabel('Peso TF-IDF acumulado'); fig.tight_layout(); display(fig); plt.close(fig); n_textos=len(tit); n_muestra=len(muestra); termino=palabras[top[-1]]"),
        note('El vocabulario se ajusta con {n_muestra:,} titulares de una población de {n_textos:,} registros declarados anteriores a 2018. «{termino}» concentra el mayor peso TF-IDF acumulado; esta medida refleja composición léxica y no frecuencia bruta ni señal predictiva de demanda.'),
        code("filas=[]\nfor i,componente in enumerate(nmf.components_):\n    filas.append({'topico':f'T{i+1}','terminos':' · '.join(palabras[np.argsort(componente)[-8:][::-1]]),'peso_medio':float(W[:,i].mean())})\ndisplay(pd.DataFrame(filas)); fig,ax=plt.subplots(); ax.bar([r['topico'] for r in filas],[r['peso_medio'] for r in filas]); ax.set_ylabel('Peso NMF medio'); display(fig); plt.close(fig); dominante=filas[int(np.argmax([r['peso_medio'] for r in filas]))]['topico']"),
        note('{dominante} presenta el mayor peso medio en la muestra de ajuste. Los términos de la tabla definen componentes aprendidas, no etiquetas editoriales. El predominio de una componente puede reflejar concentración de fuentes o prácticas del archivo, sin demostrar anticipación de la demanda.'),
        decision('Ajustar vocabulario y tópicos sólo con entrenamiento y congelarlos para su proyección.','Evita incorporar el vocabulario posterior al ajuste.','La disponibilidad declarada no certifica la versión histórica exacta del titular.')])
    books['08_validacion']=chapter('8. Diseño de evaluación','El protocolo fija MASE como métrica primaria y MAE, RMSE y sMAPE como complementarias. La diferencia relativa de interés es δ = 3 %, con α = 0,05 y potencia objetivo 0,80. La unidad de evaluación es el origen diario que produce 24 etiquetas.',[
        md('La selección utiliza cuatro pliegues anuales: 2018, 2019, 2020 y 2021. El entrenamiento se amplía cronológicamente. El último 10 % de los orígenes anteriores a cada año se reserva para parada temprana. La prueba 2022–2025 queda fuera de este análisis. La escala semanal de MASE procede de entrenamiento; con demanda creciente no representa una escala intercambiable entre particiones.'),
        code("H=24; L=168; indice=panel.index; candidatos=np.arange(L,len(panel)-H+1); candidatos=candidatos[indice[candidatos].hour==0]; filas=[]; fig,ax=plt.subplots()\nfor fila,anio in enumerate([2018,2019,2020,2021]):\n    inicio=pd.Timestamp(f'{anio}-01-01',tz='America/Bogota'); fin=pd.Timestamp(f'{anio+1}-01-01',tz='America/Bogota'); pre=candidatos[indice[candidatos]+pd.Timedelta(hours=H-1)<inicio]; n_es=max(1,round(.1*len(pre))); es=pre[-n_es:]; tr=pre[indice[pre]+pd.Timedelta(hours=H-1)<indice[es[0]]]; val=candidatos[(indice[candidatos]>=inicio)&(indice[candidatos]+pd.Timedelta(hours=H-1)<fin)]; assert indice[tr[-1]]+pd.Timedelta(hours=23)<indice[es[0]]; assert indice[es[-1]]+pd.Timedelta(hours=23)<indice[val[0]]\n    filas.append({'año':anio,'entrenamiento':len(tr),'parada_interna':len(es),'validacion':len(val)})\n    for posiciones,color in [(tr,'#247ba0'),(es,'#d18a2d'),(val,'#2a9d8f')]: ax.plot([indice[posiciones[0]],indice[posiciones[-1]]],[fila,fila],color=color,lw=10)\nax.set(yticks=range(4),yticklabels=range(2018,2022),xlabel='Fecha',ylabel='Año validado'); fig.tight_layout(); display(fig); plt.close(fig); display(pd.DataFrame(filas))"),
        md('Cada bloque verde corresponde a un año posterior a la historia de ajuste; el tramo naranja reserva parada temprana dentro de esa historia. Los conteos verifican separación de etiquetas entre ajuste, parada y validación externa. El año 2020 conserva su propio pliegue, permitiendo identificar el comportamiento de ese régimen.'),
        md('### Pipeline por origen\n\nEl panel aplica los rezagos de disponibilidad. Para un origen en la posición `p`, la historia contiene filas `p−168` a `p−1`, mientras el calendario del horizonte utiliza `p` a `p+23`. La demanda futura sólo se emplea como etiqueta. La media de demanda de la ventana es una escala conocida al origen.'),
        code("p=int(candidatos[100]); historia=panel.iloc[p-L:p]; horizonte=panel.iloc[p:p+H]; X_pasado=historia[['x_demanda_real_kwh']].to_numpy(); calendario=[c for c in panel.columns if c.startswith('cal_') and pd.api.types.is_numeric_dtype(panel[c])]; K_futuro=horizonte[calendario].to_numpy(); Y=horizonte['y_demanda_real_kwh'].to_numpy(); nivel=float(X_pasado.mean()); X_nivel=X_pasado/nivel; Y_nivel=Y/nivel; assert historia.index.max()<horizonte.index.min(); assert nivel>0; display(pd.DataFrame([{'forma_historia':str(X_nivel.shape),'forma_calendario':str(K_futuro.shape),'forma_etiquetas':str(Y_nivel.shape),'nivel_kWh':nivel}]))\ntrain=panel.loc[panel.index<'2018-01-01']; media_cal=train[calendario].mean(); escala_cal=train[calendario].std().replace(0,1); K_estandar=(horizonte[calendario]-media_cal)/escala_cal; escala_mase=float(train.y_demanda_real_kwh.diff(168).abs().dropna().mean()); display(pd.Series({'escala_MASE_semanal_kWh':escala_mase,'margen_relativo':.03,'alfa':.05,'potencia_objetivo':.8}))"),
        decision('Separar ajuste, parada interna, validación externa y prueba.','Homogeneiza la comparación y preserva el orden temporal.','La prueba retrospectiva y las versiones revisadas de las fuentes limitan la independencia histórica; la inferencia debe seguir su preregistro.')])
    books['09_modelos']=[md('# 9. Modelos')]
    from ampliacion_jbook import ampliar
    ampliar(books,md,code,note,ROOT)
    from secciones_eda import ampliar_eda
    ampliar_eda(books,md,code,note,ROOT,SETUP)
    from modelos_jbook import ampliar_modelos
    ampliar_modelos(books,md,code,note,ROOT)
    from evaluacion_jbook import ampliar_evaluacion
    ampliar_evaluacion(books,md,code,note,ROOT)
    from referencias_jbook import completar_referencias
    completar_referencias(books,md,code,note,ROOT)
    from conciliar_modelos_jbook import conciliar
    conciliar(books,md)
    from graficos_modelos_jbook import integrar_graficos
    integrar_graficos(books,md,code,note)
    from literatura_integral_jbook import integrar_literatura
    integrar_literatura(books,md,code,note,ROOT)
    from modelos_actualizados_jbook import integrar_actualizacion
    integrar_actualizacion(books,md,code,note,ROOT)
    from correcciones_e1_jbook import corregir,exportar
    corregir(books,md,code,note,ROOT)
    from consistencia_jbook import armonizar,armonizar_archivos
    armonizar(books,md,ROOT)
    from diseno_texto_jbook import integrar_diseno
    integrar_diseno(books,md,code)
    from observado_prediccion_jbook import integrar_observado
    integrar_observado(books,md,code,note)
    from importancia_jbook import integrar_importancia
    integrar_importancia(books,md,code,note,ROOT)
    from diseno_ampliado_jbook import integrar_ampliado,escribir_pagina
    integrar_ampliado(books,md,code,ROOT)
    seleccion=set(sys.argv[1:])
    if seleccion-set(books):raise ValueError('Capítulos desconocidos: '+str(seleccion-set(books)))
    for stem,cells in books.items():
        if seleccion and stem not in seleccion:continue
        notebook=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'}})
        path=BOOK/(stem+'.ipynb'); shell=InteractiveShell.instance(); shell.reset(new_session=True)
        shell.display_formatter.formatters['image/png'].for_type(Figure,lambda f:base64.b64encode(print_figure(f,fmt='png')).decode())
        count=0
        for cell in notebook.cells:
            if cell.cell_type=='markdown':
                if cell.metadata.get('plantilla_resultados'):cell.source=cell.metadata.plantilla_resultados.format_map(shell.user_ns)
                continue
            count+=1
            with capture_output() as captured:result=shell.run_cell(cell.source,store_history=False)
            if result.error_before_exec or result.error_in_exec:raise RuntimeError(f'{stem}: {result.error_before_exec or result.error_in_exec}')
            cell.execution_count=count;cell.outputs=[]
            if captured.stdout:cell.outputs.append(nb.v4.new_output('stream',name='stdout',text=captured.stdout))
            if captured.stderr:cell.outputs.append(nb.v4.new_output('stream',name='stderr',text=captured.stderr))
            for output in captured.outputs:cell.outputs.append(nb.v4.new_output('display_data',data=output.data,metadata=output.metadata))
        nb.write(notebook,path); print('Ejecutado '+stem,flush=True)
    exportar(ROOT)
    (BOOK/'intro.md').write_text('# Pronóstico de la demanda eléctrica del SIN colombiano\n\n**Fabian José Manrique Morelos y Gloria Bastidas Rodriguez**  \nMaestría en Estadística Aplicada · Universidad del Norte\n\nEl estudio caracteriza la demanda real horaria y su relación con el calendario, las condiciones ambientales y la información documental disponible. El análisis exploratorio fundamenta la representación temporal y el diseño de evaluación de pronósticos diarios de 24 horas.\n\nEl libro presenta los datos, su calidad, distribución, estacionalidad, dependencia, covariables, titulares y protocolo temporal. Cada sección incluye el código analítico y las implicaciones de los resultados. El análisis integra calidad de medición, cambios de régimen, transferencia del vocabulario y estabilidad de tópicos, junto con una síntesis metodológica y una guía de reproducción.\n',encoding='utf-8')
    (BOOK/'fuentes.md').write_text('# Fuentes y fundamento metodológico\n\n| Fuente | Uso en el estudio | Alcance |\n|---|---|---|\n| XM, archivos numéricos y registro de variables | Demanda e información del SIN | Series retrospectivas con rezagos de disponibilidad supuestos |\n| ERA5 vía Open-Meteo | Condiciones meteorológicas observadas | Reanálisis; no equivale a un pronóstico con emisión previa |\n| Censo 2018 citado en el inventario de ciudades | Ponderación de observaciones urbanas | Pesos estáticos y fuente secundaria |\n| Archivos de titulares y cobertura | Diagnóstico documental | Disponibilidad declarada o versión archivada verificada |\n| Preregistro y configuración del repositorio | Horizonte, partición, métricas y control temporal | Versiones experimentales conservadas |\n\n[Hyndman y Athanasopoulos, *Forecasting: Principles and Practice*, capítulo 5](https://otexts.com/fpp3/toolbox.html) fundamentan las referencias estacionales, los errores escalados y la validación temporal. MASE divide el error absoluto del pronóstico por la escala de una referencia estacional calculada en entrenamiento. La comparación entre particiones requiere considerar el cambio del nivel de demanda. La validación temporal evalúa observaciones posteriores a la historia de ajuste.\n\nLa síntesis de textos completos sobre demanda y noticias se presenta en el capítulo de literatura. Los enlaces identifican las versiones consultadas; las cifras reproducidas se atribuyen a sus tablas originales.\n',encoding='utf-8')
    (BOOK/'_config.yml').write_text('''title: "Pronóstico de la demanda eléctrica del SIN colombiano"
author: "Fabian José Manrique Morelos y Gloria Bastidas Rodriguez"
copyright: "2026"
language: es
execute:
  execute_notebooks: "off"
html:
  extra_footer: "Maestría en Estadística Aplicada · Universidad del Norte"
parse:
  myst_enable_extensions: [dollarmath, amsmath]
sphinx:
  config:
    language: es
    suppress_warnings: ["myst.header"]
''',encoding='utf-8')
    (BOOK/'_toc.yml').write_text('format: jb-book\nroot: intro\nchapters:\n'+''.join('  - file: '+stem+'\n'+('    sections:\n      - file: diccionario\n' if stem=='01_datos' else '    sections:\n      - file: 10_evaluacion\n      - file: 10_pipeline\n' if stem=='10_modelos' else '') for stem in books if stem!='10_evaluacion')+'  - file: fuentes\n',encoding='utf-8')
    armonizar_archivos(ROOT)
    escribir_pagina(ROOT)
    audit=dict(development_start='2000-01-01',development_end_exclusive='2022-01-01',
        test_values_read=False,model_results_read=True,training_executed=False,model_chapter_empty=False,
        authors=['Fabian José Manrique Morelos','Gloria Bastidas Rodriguez'],notebooks=len(books))
    (ROOT/'results/jbook/auditoria.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')

if __name__=='__main__':main()

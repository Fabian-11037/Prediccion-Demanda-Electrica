"""Ampliación analítica del libro y eliminación de tablas redundantes."""
from diccionario_jbook import construir


def ampliar(books, md, code, note, root):
    construir(root)
    for cells in books.values():
        for cell in cells:
            if cell.cell_type != 'code':
                continue
            cell.source=cell.source.replace("; display(nulos[nulos>0].rename('fraccion_nula'))",'')
            cell.source=cell.source.replace('; display(anual)','').replace('; display(correlacion)','')
            cell.source=cell.source.replace('display(pd.DataFrame(filas)); fig,ax=plt.subplots(); ax.bar',
                "display(pd.DataFrame(filas)[['topico','terminos']]); fig,ax=plt.subplots(); ax.bar")
            if 'variables=pd.DataFrame.from_dict(registro' in cell.source:
                cell.source=cell.source.split('\nvariables=')[0]+"\nvariables=pd.DataFrame.from_dict({k:v for k,v in registro.items() if not k.startswith('_')},orient='index'); display(variables.groupby('clase').size().rename('variables'))"
    books['01_datos'].insert(4,md('El [diccionario de datos](diccionario.md) define cada variable, sus unidades y disponibilidad. Incluye el panel horario, el corpus de titulares y los metadatos geográficos y poblacionales de las ocho ciudades.'))
    books['04_estacionalidad'].insert(-1,md('La categoría «no laboral» comprende domingos y festivos. Los sábados no festivos se distinguen mediante cal_tipo_dia = 1; la comparación binaria no equivale a lunes–viernes frente a todo el fin de semana.'))
    books['06_covariables'][-1:-1]=[
        md('## Asociaciones disponibles al inicio del día\n\nLas covariables se extraen de la fila anterior a cada origen diario. Se relacionan con la demanda media de las siguientes 24 horas. Las primeras diferencias reducen la influencia de tendencias comunes, aunque no eliminan estacionalidad ni demuestran causalidad.'),
        code("""covars={
    'x_precio_bolsa_cop_kwh':'Precio de bolsa',
    'x_obs_temperature_2m_c_ponderada':'Temperatura urbana',
    'x_obs_relative_humidity_2m_pct_ponderada':'Humedad urbana',
    'x_obs_precipitation_preceding_hour_mm_ponderada':'Precipitación urbana',
    'x_hid_llenado_valor_original':'Llenado',
    'x_hid_aportes_kwh_dia':'Aportes hídricos'}
# Seleccionar la última fila permitida antes del origen de las 00:00.
diario=panel[list(covars)].shift(1).resample('D').asfreq()
diario['demanda']=gwh.resample('D').mean()
rho_nivel=diario.corr(method='spearman')['demanda'].drop('demanda')
rho_cambio=diario.diff().corr(method='spearman')['demanda'].drop('demanda')
pos=np.arange(len(covars)); fig,ax=plt.subplots(figsize=(10,4))
ax.barh(pos-.18,rho_nivel.reindex(covars),height=.36,label='Nivel diario')
ax.barh(pos+.18,rho_cambio.reindex(covars),height=.36,label='Cambio diario')
ax.set(yticks=pos,yticklabels=list(covars.values()),xlabel='Correlación de Spearman',xlim=(-1,1))
ax.axvline(0,color='grey',lw=.7); ax.legend(); fig.tight_layout(); display(fig); plt.close(fig)
etiqueta_cambio=covars[rho_cambio.abs().idxmax()]
valor_cambio=float(rho_cambio.loc[rho_cambio.abs().idxmax()])
"""),
        note('La mayor asociación en cambios diarios corresponde a {etiqueta_cambio}, con Spearman {valor_cambio:.2f}. Las diferencias entre nivel y cambio muestran cuánto depende cada relación de la evolución lenta de las series. La precipitación urbana describe condiciones locales de consumo; no sustituye la lluvia en las cuencas de los embalses.'),
        md('## Respuesta meteorológica y calendario\n\nDemanda y clima se centran descriptivamente por año, mes, día de semana y hora. Este ajuste separa parte del calendario y del crecimiento, pero utiliza todo el desarrollo y no se incorpora como transformación del entrenamiento.'),
        code("""meteo=['x_obs_temperature_2m_c_ponderada','x_obs_relative_humidity_2m_pct_ponderada']
z=panel[meteo].shift(1).copy(); z['demanda']=gwh
grupos=[z.index.year,z.index.month,z.index.day_of_week,z.index.hour]
centrado=z-z.groupby(grupos).transform('mean')
fig,axes=plt.subplots(1,2,figsize=(12,4)); rhos=[]
for ax,col,unidad in zip(axes,meteo,['°C','puntos porcentuales']):
    a=centrado[[col,'demanda']].dropna(); bins=pd.qcut(a[col],20,duplicates='drop')
    curva=a.groupby(bins,observed=True).agg(x=(col,'median'),y=('demanda','median'),
        inferior=('demanda',lambda s:s.quantile(.25)),superior=('demanda',lambda s:s.quantile(.75)))
    ax.plot(curva.x,curva.y,marker='o',ms=3)
    ax.fill_between(curva.x.to_numpy(),curva.inferior.to_numpy(),curva.superior.to_numpy(),alpha=.2)
    ax.set(xlabel='Desviación meteorológica ('+unidad+')',ylabel='Desviación de demanda (GWh)')
    rhos.append(float(a[col].corr(a.demanda,method='spearman')))
fig.tight_layout(); display(fig); plt.close(fig)
rho_temp,rho_hum=rhos
"""),
        note('Tras el centrado, las asociaciones de temperatura y humedad con demanda son {rho_temp:.2f} y {rho_hum:.2f}. Las líneas muestran medianas por intervalos meteorológicos y las bandas el rango intercuartílico de demanda, no intervalos de confianza. La dispersión permite valorar la variabilidad que una relación media no resume.'),
        md('## Estabilidad entre entrenamiento y validación\n\nLa distancia de Wasserstein compara distribuciones diarias de entrenamiento 2000–2017 y validación 2018–2021. Se divide por la desviación estándar de entrenamiento para comparar escalas; no se interpreta como prueba de hipótesis.'),
        code("""from scipy.stats import wasserstein_distance
ent=diario.loc[diario.index<'2018-01-01']; val=diario.loc[diario.index>='2018-01-01']
distancias={}
for col in ['demanda',*covars]:
    a=ent[col].dropna(); b=val[col].dropna(); sd=float(a.std())
    if sd>0 and len(b): distancias[col]=wasserstein_distance(a,b)/sd
drift=pd.Series(distancias).sort_values(); etiquetas={'demanda':'Demanda',**covars}
fig,ax=plt.subplots(); ax.barh([etiquetas[c] for c in drift.index],drift.values)
ax.set_xlabel('Wasserstein / DE de entrenamiento'); fig.tight_layout(); display(fig); plt.close(fig)
variable_drift=etiquetas[drift.idxmax()]; valor_drift=float(drift.max())
"""),
        note('{variable_drift} presenta el mayor desplazamiento normalizado ({valor_drift:.2f}). Una distancia elevada advierte que la validación puede exigir extrapolación; no demuestra por sí sola deterioro del pronóstico. Evaluar los años por separado permite identificar cambios de régimen que el agregado puede ocultar.')]
    literatura=[md('# 8. Literatura y fundamentos del diseño'),
        md('La síntesis se concentra en pronóstico de demanda e integración documental. Se consultan textos completos para distinguir resultados, arquitecturas y condiciones de transferencia al SIN. El país de afiliación se separa del sistema eléctrico analizado.'),
        md("""## Estudios de mayor pertinencia

| Estudio | Afiliación / datos | Modelo o enfoque | Resultados y pertinencia |
|---|---|---|---|
| [Nti et al. (2020), Electricity load forecasting: a systematic review](https://doi.org/10.1186/s43067-020-00021-8) | Ghana / múltiples países | Revisión de métodos estadísticos y de inteligencia artificial | Identifica clima, calendario y actividad socioeconómica como factores relevantes. La heterogeneidad de datos y horizontes impide trasladar un ranking único al SIN. |
| [Bai, Camal y Michiorri, News and Load (versión 2024)](https://arxiv.org/abs/2301.07535) | Francia / Gran Bretaña e Irlanda del Norte | ExtraTrees con demanda, calendario, temperatura y atributos de noticias | Algunas representaciones y conjuntos de noticias mejoran la referencia; no todas las fuentes o combinaciones son útiles. Motiva contrastar el aporte textual con una referencia numérica estable. |
| [Bai, Camal y Michiorri (2023), Electricity Demand Forecasting through Natural Language Processing with Long Short-Term Memory Networks](https://arxiv.org/abs/2309.06793) | Francia / Reino Unido | LSTM y autoencoder CNN para comprimir atributos textuales | La representación y compresión afectan el desempeño. Añadir todas las representaciones puede deteriorarlo; aporta un diseño de ablación y extensión probabilística. |

Los trabajos de Bai y colaboradores comparten una línea de investigación y referencias experimentales; no son replicaciones independientes en Colombia. La síntesis dirigida no equivale a una revisión sistemática ni permite calcular frecuencias representativas de arquitecturas."""),
        md('## Representación textual y desempeño\n\nLa tabla I del estudio LSTM permite comparar variantes con la misma referencia numérica. S representa sentimiento; G, GloVe; CG, GloVe comprimido mediante autoencoder CNN; W, frecuencias de palabras; T, tópicos.'),
        code("""import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from IPython.display import display
# MAE de la tabla I del texto completo, arXiv:2309.06793.
mae_lstm=2333.20
variantes={'S + G':2299.20,'S + G + CG':2248.55,'W + S + T + G':4094.80}
cambios={nombre:100*(valor/mae_lstm-1) for nombre,valor in variantes.items()}
fig,ax=plt.subplots(figsize=(9,3.5))
ax.barh(list(cambios),list(cambios.values()),color=['#2a9d8f' if v<0 else '#c65d47' for v in cambios.values()])
ax.axvline(0,color='grey',lw=.8); ax.set_xlabel('Cambio relativo del MAE frente a LSTM numérico (%)')
fig.tight_layout(); display(fig); plt.close(fig)
mejora_cg=-cambios['S + G + CG']; deterioro_total=cambios['W + S + T + G']
"""),
        note('S + G + CG reduce el MAE un {mejora_cg:.2f} %, mientras W + S + T + G lo incrementa un {deterioro_total:.2f} %. La utilidad depende de la representación y su selección; los valores pertenecen al experimento británico y no estiman la mejora esperable en el SIN. [Fuente: tabla I del artículo](https://arxiv.org/html/2309.06793).'),
        md("""## Implicaciones para el diseño

La memoria diaria y semanal, el calendario y los cambios de nivel orientan la representación temporal. Una referencia lineal cuantifica cuánto resuelve una combinación aditiva de rezagos; una red de alimentación directa representa interacciones no lineales; una arquitectura recurrente contrasta el procesamiento secuencial de la historia. La selección final y sus resultados se desarrollarán en el capítulo de modelos.

Clima e hidrología se consideran mediante ablaciones con los mismos orígenes y presupuesto. Para el texto se congela cada representación tras ajustarla en entrenamiento y se comparan escenarios de disponibilidad. Los tópicos exploratorios no se convierten en categorías manuales.

[Hyndman y Athanasopoulos, Forecasting: Principles and Practice](https://otexts.com/fpp3/toolbox.html) sustentan las referencias estacionales, los errores escalados y la evaluación temporal. La comparación se realiza sobre los mismos orígenes y se complementa con errores en la escala original.""")]
    orden={}
    for nombre,cells in books.items():
        if nombre=='08_validacion': orden['08_literatura']=literatura
        orden[nombre]=cells
    books.clear();books.update(orden)
    books['08_validacion'][0]=md('# 9. Diseño de evaluación')
    books['09_modelos']=[md('# 10. Modelos')]
    books['08_validacion'].insert(3,md(r"""## Formulación y entradas

Para un origen diario $o$, la salida es $\mathbf{y}_o=(y_o,\ldots,y_{o+23})$. El pronóstico directo se expresa como

$$\widehat{\mathbf{y}}_o=f_\theta(\mathbf{X}_{o-168:o-1},\mathbf{K}_{o:o+23}),$$

donde $\mathbf{X}$ contiene observaciones pasadas alineadas por disponibilidad y $\mathbf{K}$ contiene calendario conocido. La configuración base utiliza demanda y calendario; clima, hidrología y titulares fundamentan extensiones posteriores.

La unidad de evaluación es el origen diario, con 24 errores relacionados. Las semillas describen variación del ajuste y no hacen independientes los errores de un mismo día."""))
    books['08_validacion'].extend([
        md(r"""## Métricas y comparación

Para $N$ orígenes y $H=24$:

$$\operatorname{MAE}=\frac{1}{NH}\sum_o\sum_{h=0}^{H-1}|y_{o+h}-\widehat y_{o+h}|,$$

$$\operatorname{RMSE}=\sqrt{\frac{1}{NH}\sum_o\sum_{h=0}^{H-1}(y_{o+h}-\widehat y_{o+h})^2},\qquad
\operatorname{MASE}=\frac{\operatorname{MAE}}{q_{168}},$$

$$q_{168}=\frac{1}{T-168}\sum_{t=168}^{T-1}|y_t-y_{t-168}|.$$

La escala $q_{168}$ procede exclusivamente de entrenamiento 2000–2017. MASE es la métrica primaria; MAE se interpreta en kWh por intervalo y RMSE acentúa errores grandes. sMAPE promedia $200|y-\widehat y|/(|y|+|\widehat y|)$, con contribución cero cuando ambos son cero. Un MASE inferior a 1 compara con la escala histórica de entrenamiento; la comparación con el ingenuo en validación requiere calcular también su error sobre los mismos orígenes.

Persistencia y referencias diaria y semanal deben utilizar la misma disponibilidad que los modelos. El reporte desagrega por hora del horizonte, año y tipo de día, además de media, desviación e intervalos entre semillas. El margen relativo de interés es 3 %; α = 0,05 y potencia objetivo 0,80 son parámetros del diseño inferencial."""),
        md("""## Búsqueda y prevención de fuga

La versión 2 fija el mismo presupuesto por familia: 20 configuraciones, tres semillas y cuatro pliegues anuales. Cada ajuste tiene hasta 200 épocas; la parada temprana utiliza el tramo interno, paciencia entre 8, 15 y 25 y mejora mínima de $10^{-6}$. Se conserva el mejor estado según la pérdida interna y el año externo evalúa la configuración.

Imputación y estandarización se ajustan con el tramo de optimización de cada pliegue. El nivel de cada ventana se calcula con sus 168 filas permitidas y se conserva para devolver las salidas a kWh. Las etiquetas completas de cada origen permanecen dentro de su partición.

La inferencia requiere un preregistro específico: hipótesis pareadas, tratamiento de dependencia temporal, corrección de multiplicidad y equivalencia. Se contempla remuestreo por bloques de orígenes; su longitud y las pruebas definitivas quedan pendientes de fijar. La potencia se estimará mediante piloto.

## Reproducibilidad

La reproducción conserva configuraciones, semillas, versión del protocolo, diccionario y separación temporal. Debe registrar versiones de dependencias, dispositivo CPU/GPU, duración y huellas de los archivos de entrada. Los resultados y predicciones se almacenan por versión del experimento, sin sobrescribir corridas.""")])
    for cell in books['08_validacion']:
        if cell.cell_type=='code' and 'media_cal=train[calendario].mean()' in cell.source:
            cell.source=cell.source.replace("media_cal=train[calendario].mean(); escala_cal=train[calendario].std().replace(0,1)",
                "pre_2018=candidatos[indice[candidatos]+pd.Timedelta(hours=23)<pd.Timestamp('2018-01-01',tz='America/Bogota')]; inicio_es=indice[pre_2018[-max(1,round(.1*len(pre_2018)))]]; ajuste=train.loc[train.index<inicio_es]; media_cal=ajuste[calendario].mean(); escala_cal=ajuste[calendario].std().replace(0,1)")

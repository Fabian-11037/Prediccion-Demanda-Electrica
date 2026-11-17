"""Secciones descriptivas, síntesis e instrucciones reproducibles del libro."""


def ampliar_eda(books, md, code, note, root, setup):
    books['02_calidad'][-1:-1]=[
        md('## Rangos y coherencia de las mediciones\n\nSe comprueban valores finitos y restricciones de interpretación: humedad entre 0 y 100 %, precipitación no negativa y almacenamiento no negativo. El llenado superior a uno se registra por separado para contrastarlo con reserva y capacidad; no se corrige ni se interpreta automáticamente como un porcentaje.'),
        code("""controles=[]
for col in panel.select_dtypes(include='number'):
    s=panel[col]; valida=s.dropna()
    if 'relative_humidity' in col: fuera=(valida<0)|(valida>100)
    elif 'precipitation' in col or col in ['x_hid_reserva_util_kwh','x_hid_aportes_kwh_dia']: fuera=valida<0
    elif col=='x_hid_capacidad_util_kwh': fuera=valida<=0
    else: fuera=pd.Series(False,index=valida.index)
    controles.append({'variable':col,'no_finitos_no_nulos':int((~np.isfinite(valida)).sum()),'fuera_rango':int(fuera.sum())})
controles=pd.DataFrame(controles)
incidencias=controles.loc[(controles.no_finitos_no_nulos>0)|(controles.fuera_rango>0)]
if len(incidencias): display(incidencias)
n_rango=int(controles.fuera_rango.sum()); n_inf=int(controles.no_finitos_no_nulos.sum())
hid=panel[['x_hid_llenado_valor_original','x_hid_reserva_util_kwh','x_hid_capacidad_util_kwh']].dropna()
desacuerdo=(hid.iloc[:,0]-hid.iloc[:,1]/hid.iloc[:,2]).abs()
error_ratio=float(desacuerdo.max())
fechas_sobreuno=panel.loc[panel.x_hid_llenado_valor_original>1,'aux_hid_fecha_dato'].nunique()
"""),
        note('Se registran {n_rango} incumplimientos de los rangos evaluados y {n_inf} valores infinitos no nulos. El llenado supera uno en {fechas_sobreuno} fechas hidráulicas distintas; la discrepancia máxima entre el indicador y reserva/capacidad es {error_ratio:.2g}. Las cifras repetidas por hora reflejan la frecuencia diaria de la fuente, no mediciones hidráulicas independientes.'),
        md('## Representatividad geográfica y poblacional\n\nLas series meteorológicas representan puntos de malla próximos a ocho ciudades. La distancia al punto solicitado cuantifica una diferencia de localización, no el error de temperatura o precipitación. Los pesos poblacionales son estáticos: no describen la distribución histórica del consumo eléctrico ni la evolución demográfica anual.'),
        code("""geo=pd.read_csv(ROOT/'data/raw/numeric/ciudades.csv')
lat1=np.radians(geo.lat_solicitada); lat2=np.radians(geo.lat_malla)
dlat=lat2-lat1; dlon=np.radians(geo.lon_malla-geo.lon_solicitada)
a=np.sin(dlat/2)**2+np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
geo['distancia_km']=6371*2*np.arcsin(np.sqrt(a.clip(0,1)))
geo['peso']=geo.habitantes_censo_2018_ajustado/geo.habitantes_censo_2018_ajustado.sum()
fig,axes=plt.subplots(1,2,figsize=(12,4))
axes[0].barh(geo.ciudad,geo.distancia_km); axes[0].set_xlabel('Distancia al punto de malla (km)')
axes[1].barh(geo.ciudad,100*geo.peso); axes[1].set_xlabel('Peso poblacional estático (%)')
fig.tight_layout(); display(fig); plt.close(fig)
distancia_max=float(geo.distancia_km.max()); ciudad_peso=geo.loc[geo.peso.idxmax(),'ciudad']; peso_max=float(100*geo.peso.max())
"""),
        note('La separación máxima entre coordenada solicitada y malla es {distancia_max:.2f} km. {ciudad_peso} concentra el {peso_max:.1f} % del promedio ponderado, por lo que esa representación puede atenuar condiciones de ciudades con menor peso. Validar el error meteorológico requiere estaciones emparejadas; la proximidad geográfica no basta para estimarlo.')]

    books['03_demanda'][-1:-1]=[
        md('## Cambios exploratorios de régimen\n\nSe calcula el logaritmo de la demanda diaria y se retira el perfil medio de día de semana estimado en 2000–2017. Sobre la serie resultante se aplica segmentación binaria por reducción de suma de cuadrados, con segmentos mínimos de 180 días, candidatos semanales, penalización de tres veces el logaritmo del tamaño de muestra y un máximo de seis cortes. Es un diagnóstico retrospectivo de nivel; la tendencia puede generar cortes sin una ruptura abrupta y las fechas no se utilizan como entradas del pronóstico.'),
        code("""diaria_reg=gwh.resample('D').mean()
log_diaria=np.log(diaria_reg)
base=log_diaria.loc[log_diaria.index<'2018-01-01']
perfil_semana=base.groupby(base.index.day_of_week).mean()
ajustada=log_diaria-pd.Series(log_diaria.index.day_of_week,index=log_diaria.index).map(perfil_semana)
v=((ajustada-ajustada.mean())/ajustada.std()).to_numpy()
cs=np.r_[0,np.cumsum(v)]; cs2=np.r_[0,np.cumsum(v*v)]
def sse(a,b):
    return cs2[b]-cs2[a]-(cs[b]-cs[a])**2/(b-a)
segmentos=[(0,len(v))]; cortes=[]; minimo=180; umbral=3*np.log(len(v))
for _ in range(6):
    opciones=[]
    for a,b in segmentos:
        for k in range(a+minimo,b-minimo+1,7):
            opciones.append((sse(a,b)-sse(a,k)-sse(k,b),a,b,k))
    if not opciones: break
    ganancia,a,b,k=max(opciones)
    if ganancia<=umbral: break
    cortes.append(k); segmentos.remove((a,b)); segmentos.extend([(a,k),(k,b)])
fechas_corte=diaria_reg.index[sorted(cortes)]
fig,axes=plt.subplots(2,1,figsize=(11,6),sharex=True)
axes[0].plot(diaria_reg.index,diaria_reg,lw=.6); axes[0].set_ylabel('Demanda diaria (GWh/h)')
axes[1].plot(ajustada.index,ajustada,lw=.6); axes[1].set_ylabel('Log-demanda ajustada')
for a,b in sorted(segmentos):
    axes[1].plot([ajustada.index[a],ajustada.index[b-1]],[ajustada.iloc[a:b].mean()]*2,color='#c65d47',lw=2)
for fecha in fechas_corte:
    for ax in axes: ax.axvline(fecha,color='#c65d47',ls='--',alpha=.5)
fig.tight_layout(); display(fig); plt.close(fig)
n_cortes=len(cortes); fechas_texto=', '.join(f.strftime('%Y-%m-%d') for f in fechas_corte) or 'ninguna'
"""),
        note('El procedimiento identifica {n_cortes} cortes: {fechas_texto}. Los tramos resumen cambios de nivel de la demanda ajustada por día de semana, sin atribuirlos a causas económicas o sanitarias. La comparación anual y la normalización por nivel permiten evaluar estos regímenes sin convertir los cortes retrospectivos en información disponible al origen.')]

    books['05_dependencia'][-1:-1]=[
        md('## Memoria tras retirar el ciclo semanal\n\nLa diferencia semanal horaria elimina la comparación entre horas de distinto día de semana. Su ACF y PACF permiten examinar dependencia remanente; los picos negativos cercanos a 168 horas también pueden surgir por la propia diferenciación.'),
        code("""from statsmodels.tsa.stattools import pacf
dif_sem=gwh.diff(168).dropna()
acf_dif=acf(dif_sem,nlags=336,fft=True); pacf_dif=pacf(dif_sem,nlags=48,method='ywmle')
fig,axes=plt.subplots(1,2,figsize=(12,4))
axes[0].plot(np.arange(337),acf_dif); axes[0].set(xlabel='Rezago (h)',ylabel='ACF diferencia semanal')
axes[1].stem(np.arange(49),pacf_dif); axes[1].set(xlabel='Rezago (h)',ylabel='PACF diferencia semanal')
fig.tight_layout(); display(fig); plt.close(fig)
acf_1=float(acf_dif[1]); pacf_24=float(pacf_dif[24])
"""),
        note('La diferencia semanal conserva ACF de {acf_1:.2f} a una hora y PACF de {pacf_24:.2f} a 24 horas. Restar la semana anterior no agota necesariamente la memoria reciente; estos diagnósticos respaldan estudiar conjuntamente historia corta y calendario, sin interpretar cada rezago como un contraste independiente.'),
        md('## Variación de la dispersión y dependencia residual\n\nLa mediana y la desviación absoluta mediana móviles se calculan sobre diferencias semanales diarias con ventanas retrospectivas de 90 días. Ljung–Box y ARCH se aplican sólo a entrenamiento sobre esa diferencia centrada. Al no tratarse de innovaciones de un modelo ajustado, los resultados describen dependencia y agrupación de magnitudes; una señal ARCH también puede estar afectada por autocorrelación remanente.'),
        code("""from statsmodels.stats.diagnostic import acorr_ljungbox,het_arch
from statsmodels.tsa.stattools import bds
d_sem=gwh.resample('D').mean().diff(7).dropna()
mediana_mov=d_sem.rolling(90,min_periods=90).median()
mad_mov=d_sem.rolling(90,min_periods=90).apply(lambda x:np.median(np.abs(x-np.median(x))),raw=True)
fig,axes=plt.subplots(2,1,figsize=(11,5),sharex=True)
axes[0].plot(d_sem.index,d_sem,lw=.6); axes[0].plot(mediana_mov.index,mediana_mov,lw=1.3); axes[0].set_ylabel('Cambio semanal (GWh/h)')
axes[1].plot(mad_mov.index,mad_mov); axes[1].set_ylabel('MAD móvil, 90 días (GWh/h)')
fig.tight_layout(); display(fig); plt.close(fig)
mad_min=float(mad_mov.dropna().min()); mad_max=float(mad_mov.max())
e=d_sem.loc[d_sem.index<'2018-01-01']; e=e-e.mean()
lb=acorr_ljungbox(e,lags=[7,14],return_df=True)
lm,p_arch,_,_=het_arch(e,nlags=7,result_object=False)
bds_stat,bds_p=bds(e.iloc[-1000:].to_numpy(),max_dim=2)
diagnosticos=pd.DataFrame([
    {'contraste':'Ljung–Box 7 días','estadistico':lb.loc[7,'lb_stat'],'p_sin_ajustar':lb.loc[7,'lb_pvalue']},
    {'contraste':'Ljung–Box 14 días','estadistico':lb.loc[14,'lb_stat'],'p_sin_ajustar':lb.loc[14,'lb_pvalue']},
    {'contraste':'ARCH 7 días','estadistico':lm,'p_sin_ajustar':p_arch},
    {'contraste':'BDS dimensión 2, últimos 1000 días de train','estadistico':float(bds_stat),'p_sin_ajustar':float(bds_p)}])
display(diagnosticos)
p_lb=float(lb.loc[14,'lb_pvalue'])
"""),
        note('La MAD móvil varía entre {mad_min:.3f} y {mad_max:.3f} GWh/h, mostrando cuánto cambia la dispersión local de las diferencias semanales. Ljung–Box a 14 días obtiene p = {p_lb:.3g}; los contrastes son exploratorios y no comparan modelos. BDS contrasta independencia e idéntica distribución: su rechazo, en una serie con memoria, no aísla no linealidad. La heterocedasticidad se revisará después sobre errores predictivos genuinos.')]

    books['07_texto'].extend([
        md('## Cobertura mensual, longitud y repetición\n\nEl corpus de desarrollo incluye todos los titulares elegibles con disponibilidad declarada anterior a 2022; los registros en cuarentena y los encabezados vacíos quedan excluidos. El ajuste léxico inicial permanece limitado a 2000–2017. La normalización de duplicados aplica Unicode NFKC, minúsculas y eliminación de puntuación y espacios redundantes, conservando las letras acentuadas. Una repetición entre fuentes no se elimina automáticamente.'),
        code("""import re,unicodedata
texto=pq.read_table(ROOT/'data/raw/noticias/titulares.parquet',
    columns=['id','source','titular','available_at_assumed','quarantined'],
    filters=[('available_at_assumed','>=','2000-01-01'),('available_at_assumed','<','2022-01-02'),('quarantined','==',False)]).to_pandas()
texto=texto[texto.titular.notna() & texto.titular.str.strip().ne('')].copy()
texto['fecha']=pd.to_datetime(texto.available_at_assumed,utc=True,format='ISO8601').dt.tz_convert('America/Bogota')
texto=texto[(texto.fecha>=pd.Timestamp('2000-01-01',tz='America/Bogota')) & (texto.fecha<pd.Timestamp('2022-01-01',tz='America/Bogota'))].copy()
texto=texto.sort_values(['fecha','source','id'],kind='stable').reset_index(drop=True)
def normalizar_titulo(s):
    s=unicodedata.normalize('NFKC',s).lower()
    return re.sub(r'\\s+',' ',re.sub(r'[^\\w\\s]',' ',s)).strip()
texto['normalizado']=texto.titular.map(normalizar_titulo)
texto['n_palabras']=texto.normalizado.str.split().str.len()
texto['repetido_exacto']=texto.duplicated(['source','titular'],keep='first')
texto['repetido_normalizado']=texto.duplicated(['source','normalizado'],keep='first')
texto['mes']=texto.fecha.dt.tz_localize(None).dt.to_period('M')
texto['periodo']=np.where(texto.fecha.dt.year<2018,'Entrenamiento','Validación')
cob_mes=pd.crosstab(texto.mes,texto.source).reindex(pd.period_range('2000-01','2021-12',freq='M'),fill_value=0)
fig,axes=plt.subplots(1,2,figsize=(14,4))
im=axes[0].imshow(np.log1p(cob_mes.T.to_numpy()),aspect='auto',cmap='viridis')
axes[0].set(yticks=range(len(cob_mes.columns)),yticklabels=cob_mes.columns,
    xticks=np.arange(0,len(cob_mes),36),xticklabels=[str(x) for x in cob_mes.index[::36]])
fig.colorbar(im,ax=axes[0],label='log(1 + titulares/mes)')
for periodo,datos in texto.groupby('periodo'):
    axes[1].hist(datos.n_palabras,bins=np.arange(0,51),density=True,alpha=.45,label=periodo)
axes[1].set(xlabel='Palabras por titular (0–50)',ylabel='Densidad'); axes[1].legend()
fig.tight_layout(); display(fig); plt.close(fig)
n_desarrollo=len(texto); longitud_mediana=float(texto.n_palabras.median())
sin_cobertura=int((cob_mes.sum(axis=1)==0).sum()); cola_larga=int((texto.n_palabras>50).sum())
"""),
        note('Se analizan {n_desarrollo:,} titulares, con mediana de {longitud_mediana:.0f} palabras. Hay {sin_cobertura} meses sin registros recuperados en ninguna fuente; no equivalen a meses sin publicaciones. El histograma muestra hasta 50 palabras y deja fuera {cola_larga:,} encabezados más extensos. La matriz distingue vacíos del archivo de aumentos de volumen asociados a nuevas fuentes.'),
        code("""repeticion=texto.groupby('source')[['repetido_exacto','repetido_normalizado']].mean()*100
fig,ax=plt.subplots(figsize=(11,4))
repeticion.rename(columns={'repetido_exacto':'Exacta','repetido_normalizado':'Normalizada'}).plot.barh(ax=ax)
ax.set_xlabel('Repeticiones posteriores a la primera por fuente (%)')
fig.tight_layout(); display(fig); plt.close(fig)
dup_exacta=float(100*texto.repetido_exacto.mean()); dup_normal=float(100*texto.repetido_normalizado.mean())
"""),
        note('Las repeticiones exactas representan el {dup_exacta:.2f} % y las normalizadas el {dup_normal:.2f} % de los registros. La diferencia cuantifica variantes formales que la comparación literal omite. La repetición puede responder a recuperación duplicada, republicación o encabezados recurrentes; determinar su origen requiere revisar la trazabilidad.'),
        md('## Bigramas y diversidad léxica\n\nLos bigramas se cuentan por presencia documental sobre titulares de entrenamiento únicos por fuente y normalización. La diversidad se estima en bloques de 1.000 términos elegibles, ordenados por fecha y separados por año; se descarta el último bloque incompleto. Así se reduce la dependencia del indicador respecto del volumen anual.'),
        code("""from sklearn.feature_extraction.text import CountVectorizer
ent_texto=texto.loc[texto.periodo=='Entrenamiento'].drop_duplicates(['source','normalizado'])
stop_es=vectorizador.get_stop_words()
bigramador=CountVectorizer(ngram_range=(2,2),binary=True,min_df=10,max_features=3000,
    strip_accents='unicode',stop_words=list(stop_es))
xb=bigramador.fit_transform(ent_texto.titular)
frecuencia_bi=np.asarray(xb.sum(axis=0)).ravel(); terminos_bi=bigramador.get_feature_names_out()
indices=np.argsort(frecuencia_bi)[-15:]
analizador=vectorizador.build_analyzer()
diversidad=[]
for anio,grupo in texto.groupby(texto.fecha.dt.year):
    terminos=[t for s in grupo.titular for t in analizador(s)]
    valores=[len(set(terminos[k:k+1000]))/1000 for k in range(0,len(terminos)-999,1000)]
    if valores: diversidad.append({'anio':anio,'mediana':np.median(valores),'q25':np.quantile(valores,.25),'q75':np.quantile(valores,.75)})
diversidad=pd.DataFrame(diversidad).set_index('anio')
fig,axes=plt.subplots(1,2,figsize=(14,5))
axes[0].barh(terminos_bi[indices],100*frecuencia_bi[indices]/len(ent_texto)); axes[0].set_xlabel('Titulares de entrenamiento únicos (%)')
axes[1].plot(diversidad.index,diversidad.mediana,marker='o',ms=3)
axes[1].fill_between(diversidad.index,diversidad.q25,diversidad.q75,alpha=.2)
axes[1].set(xlabel='Año',ylabel='Proporción de términos únicos / bloque')
fig.tight_layout(); display(fig); plt.close(fig)
bigrama_principal=terminos_bi[indices[-1]]
div_inicio=float(diversidad.mediana.iloc[0]); div_final=float(diversidad.mediana.iloc[-1])
"""),
        note('«{bigrama_principal}» es el bigrama con mayor presencia entre los candidatos conservados. La diversidad mediana pasa de {div_inicio:.3f} a {div_final:.3f} entre el primer y último año representados; las bandas muestran dispersión entre bloques. Los cambios pueden reflejar composición editorial y repetición, además de variaciones del contenido.'),
        md('## Evolución de tópicos y composición por fuente\n\nTodos los documentos de desarrollo se proyectan con el vocabulario, IDF y seis componentes NMF ajustados en la muestra inicial de entrenamiento. Para comparar componentes se normalizan sus perfiles de términos y se expresa su contribución reconstruida por documento. Los titulares sin representación se contabilizan aparte; sus contribuciones no se imputan a cero.'),
        code("""pesos=np.empty((len(texto),nmf.n_components),dtype='float32')
representado=np.zeros(len(texto),dtype=bool); n_tokens=np.zeros(len(texto),dtype='int32')
n_fuera=np.zeros(len(texto),dtype='int32'); vocab=set(vectorizador.vocabulary_)
masa_componentes=nmf.components_.sum(axis=1)
for inicio in range(0,len(texto),20000):
    fin=min(inicio+20000,len(texto)); titulares=texto.titular.iloc[inicio:fin]
    x_lote=vectorizador.transform(titulares); w=nmf.transform(x_lote)
    contribucion=w*masa_componentes[None,:]; total=contribucion.sum(axis=1)
    valido=(x_lote.getnnz(axis=1)>0)&(total>0)
    contribucion=np.divide(contribucion,total[:,None],out=np.full_like(contribucion,np.nan),where=total[:,None]>0)
    pesos[inicio:fin]=contribucion; representado[inicio:fin]=valido
    for j,s in enumerate(titulares,inicio):
        tokens=analizador(s); n_tokens[j]=len(tokens); n_fuera[j]=sum(t not in vocab for t in tokens)
topicos=[f'T{k+1}' for k in range(nmf.n_components)]
proyeccion=pd.DataFrame(pesos,columns=topicos,index=texto.index)
proyeccion.loc[~representado,:]=np.nan
mensual=proyeccion.groupby(texto.mes).mean().reindex(cob_mes.index)
por_fuente=proyeccion.groupby(texto.source).mean()
fig,axes=plt.subplots(1,2,figsize=(14,4))
for topico in topicos: axes[0].plot(mensual.index.to_timestamp(),mensual[topico],label=topico,lw=1)
axes[0].set(xlabel='Disponibilidad mensual',ylabel='Contribución media normalizada'); axes[0].legend(ncol=3,fontsize=8)
im=axes[1].imshow(por_fuente,aspect='auto',vmin=0,vmax=1,cmap='viridis')
axes[1].set(yticks=range(len(por_fuente)),yticklabels=por_fuente.index,xticks=range(len(topicos)),xticklabels=topicos)
fig.colorbar(im,ax=axes[1],label='Contribución media')
fig.tight_layout(); display(fig); plt.close(fig)
cobertura_repr=float(100*representado.mean()); topico_global=proyeccion.mean().idxmax()
"""),
        note('El {cobertura_repr:.2f} % de los titulares tiene representación utilizable; {topico_global} concentra la mayor contribución media. Las curvas muestran evolución del contenido representado y la matriz permite identificar diferencias editoriales. Las interrupciones mensuales indican ausencia de documentos representados; no se interpolan. El volumen y la composición por fuente deben acompañar cualquier interpretación temática.'),
        md('## Transferencia del vocabulario y estabilidad de los tópicos\n\nLa tasa de términos fuera del vocabulario usa los tokens producidos por el mismo analizador de entrenamiento; incluye palabras descartadas por sus umbrales. La divergencia Jensen–Shannon se calcula entre distribuciones de unigramas de entrenamiento y validación en ese vocabulario común. La estabilidad NMF se evalúa con otra inicialización, manteniendo la misma muestra y emparejando componentes por máxima similitud coseno.'),
        code("""from scipy.spatial.distance import jensenshannon
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import cosine_similarity
texto['n_tokens']=n_tokens; texto['n_fuera']=n_fuera; texto['representado']=representado
oov=texto.groupby(['source','periodo'])[['n_tokens','n_fuera']].sum()
tasa=(100*oov.n_fuera/oov.n_tokens.replace(0,np.nan)).unstack('periodo')
conteos={p:np.zeros(len(vocab),dtype=np.float64) for p in ['Entrenamiento','Validación']}
contador=CountVectorizer(vocabulary=vectorizador.vocabulary_,lowercase=vectorizador.lowercase,
    strip_accents=vectorizador.strip_accents,stop_words=list(stop_es),token_pattern=vectorizador.token_pattern)
for periodo,grupo in texto.groupby('periodo'):
    for inicio in range(0,len(grupo),30000):
        conteos[periodo]+=np.asarray(contador.transform(grupo.titular.iloc[inicio:inicio+30000]).sum(axis=0)).ravel()
js=float(jensenshannon(conteos['Entrenamiento'],conteos['Validación'],base=2)**2)
nmf_otro=NMF(n_components=6,init='random',random_state=11038,max_iter=400)
nmf_otro.fit(X)
sim=cosine_similarity(nmf.components_,nmf_otro.components_)
rr,cc=linear_sum_assignment(-sim); estabilidad=sim[rr,cc]
fig,axes=plt.subplots(1,2,figsize=(13,4))
tasa.plot.barh(ax=axes[0]); axes[0].set_xlabel('Tokens fuera del vocabulario (%)')
axes[1].bar([f'T{i+1}' for i in rr],estabilidad); axes[1].set(ylabel='Similitud coseno emparejada',ylim=(0,1))
fig.tight_layout(); display(fig); plt.close(fig)
totales=texto.groupby('periodo')[['n_tokens','n_fuera']].sum()
oov_train=float(100*totales.loc['Entrenamiento','n_fuera']/totales.loc['Entrenamiento','n_tokens'])
oov_val=float(100*totales.loc['Validación','n_fuera']/totales.loc['Validación','n_tokens'])
sim_min=float(estabilidad.min()); sim_media=float(estabilidad.mean())
resumen_texto=dict(documentos=len(texto),oov_train_pct=oov_train,oov_val_pct=oov_val,
    jensen_shannon=js,similitud_media=sim_media,similitud_minima=sim_min,
    repeticion_exacta_pct=dup_exacta,repeticion_normalizada_pct=dup_normal,representados_pct=cobertura_repr)
(ROOT/'results/jbook/texto_resumen.json').write_text(json.dumps(resumen_texto,indent=2),encoding='utf-8')
mensual.to_csv(ROOT/'results/jbook/topicos_mensuales.csv')
por_fuente.to_csv(ROOT/'results/jbook/topicos_por_fuente.csv')
tasa.to_csv(ROOT/'results/jbook/vocabulario_por_fuente.csv')
cob_mes.to_csv(ROOT/'results/jbook/cobertura_texto_mensual.csv')
"""),
        note('La tasa global fuera del vocabulario pasa del {oov_train:.2f} % en entrenamiento al {oov_val:.2f} % en validación, con divergencia Jensen–Shannon de {js:.3f}. La similitud NMF media es {sim_media:.3f} y la mínima {sim_min:.3f}. La primera comparación mide transferencia temporal y entre fuentes; la segunda, sensibilidad a la inicialización dentro de una misma muestra. Una estabilidad alta no prueba relevancia predictiva.'),
        md('## Cobertura del historial documental por pliegue\n\nLos conteos distinguen todo el historial anterior al año validado de los documentos disponibles durante ese año. No equivalen al número de titulares de cada ventana predictiva, cuya longitud se fijará antes del experimento textual.'),
        code("""cob_folds=[]
for anio in [2018,2019,2020,2021]:
    antes=texto.fecha.dt.year<anio; externo=texto.fecha.dt.year==anio
    cob_folds.append({'año_validado':anio,'historial_documentos':int(antes.sum()),
        'documentos_año_externo':int(externo.sum()),'fuentes_historial':int(texto.loc[antes,'source'].nunique()),
        'fuentes_año_externo':int(texto.loc[externo,'source'].nunique()),
        'días_con_texto_externo':int(texto.loc[externo,'fecha'].dt.normalize().nunique())})
display(pd.DataFrame(cob_folds))
"""),
        md('La tabla permite comprobar si cada año externo dispone de fuentes y días comparables con su historial. Las proyecciones exploratorias usan una representación común ajustada antes de 2018; un futuro entrenamiento textual deberá ajustar su representación dentro del tramo permitido de cada pliegue.')])

    books['08_literatura'].extend([
        md('## Comparación de arquitecturas para demanda nacional\n\n[Pelekis et al. (2023), *A comparative assessment of deep learning models for day-ahead load forecasting: Investigating key accuracy drivers*](https://arxiv.org/abs/2302.12168), con afiliación en Grecia, estudian demanda nacional portuguesa horaria de 2013–2021. Comparan MLP, LSTM, N-BEATS, TCN y TFT con referencias estacionales. La tabla 4 separa versiones univariantes y extensiones con temperatura ex-post de Lisboa.'),
        code("""# Tabla 4 del texto completo, arXiv:2302.12168v2.
arquitecturas=['MLP','LSTM','N-BEATS','TCN','TFT']
univar=np.array([2.02,2.23,1.90,2.22,2.17])
temp_expost=np.array([1.92,2.20,2.00,2.12,2.06])
pos=np.arange(len(arquitecturas))
fig,ax=plt.subplots(figsize=(9,4))
ax.bar(pos-.18,univar,width=.36,label='Univariante')
ax.bar(pos+.18,temp_expost,width=.36,label='Con temperatura ex-post')
ax.set(xticks=pos,xticklabels=arquitecturas,ylabel='MAPE de prueba (%)'); ax.legend()
fig.tight_layout(); display(fig); plt.close(fig)
ganancia_mlp=float(100*(1-temp_expost[0]/univar[0]))
"""),
        note('N-BEATS univariante presenta el menor MAPE de esta comparación; MLP mejora un {ganancia_mlp:.2f} % relativo con temperatura ex-post. Una arquitectura más compleja no garantiza menor error y la temperatura no beneficia a todas. La extensión ex-post no demuestra disponibilidad meteorológica operativa. [Fuente: tabla 4](https://arxiv.org/html/2302.12168).'),
        md('## Criterios de extracción y justificación de familias\n\n| Familia | Evidencia y función en el estudio | Aspecto a contrastar |\n|---|---|---|\n| Regresión autorregresiva con entradas externas | La revisión de Nti et al. sitúa métodos estadísticos y regresión entre las referencias de demanda. La versión penalizada del protocolo ofrece una comparación aditiva de baja complejidad. | Utilidad de memoria y calendario sin interacciones no lineales. |\n| MLP | Pelekis et al. la comparan directamente con arquitecturas más complejas en demanda nacional. | Si las interacciones de una ventana aplanada bastan para representar la señal. |\n| LSTM | Aparece en Pelekis et al. y en el trabajo de Bai et al. con texto. | Si procesar secuencialmente la historia aporta frente a una representación densa. |\n\nLa revisión no establece que la regresión penalizada exacta del protocolo sea la más frecuente ni que estas tres familias sean universalmente superiores. N-BEATS, TCN, TFT y compresión textual constituyen arquitecturas de interés para ampliaciones; no modifican el presupuesto ni los modelos que ya están en entrenamiento.'),
        md('## Registro de la evidencia\n\nLa extracción conserva título, identificador DOI o arXiv, versión, país de afiliación, sistema eléctrico, frecuencia, horizonte, entradas, comparadores y localización del resultado. Las cifras de artículos se reproducen mediante celdas visibles y se distinguen de resultados del SIN. La calidad de transferencia se valora por similitud del objetivo, separación cronológica, disponibilidad de covariables y tratamiento de semillas. La selección dirigida no permite construir un flujo PRISMA ni una frecuencia exhaustiva de modelos; eso requiere búsquedas y criterios documentados sobre un inventario más amplio.')])
    # El cuaderno bibliográfico no usa el setup del panel.
    books['08_literatura'][4].source="import numpy as np\n"+books['08_literatura'][4].source

    books['08_validacion'].extend([
        md(r"""## Unidad de evaluación, bloques y comparaciones

Para dos métodos $A$ y $B$, se define una pérdida por origen

$$\ell_{m,o}=\frac{1}{24q_{168}}\sum_{h=0}^{23}|y_{o+h}-\widehat y_{m,o+h}|,\qquad
d_o=\ell_{A,o}-\ell_{B,o}.$$

El apareamiento exige los mismos días y horizontes; las 24 horas no se tratan como réplicas independientes. El intervalo temporal se estimaría con bloques contiguos de días para conservar dependencia serial. La variación entre semillas se reportaría separadamente: no se concatenan como si multiplicaran el número de días observados.

El protocolo inferencial por completar debe fijar las comparaciones de las tres familias entre sí y frente a las referencias, corrección de Holm dentro de una familia definida de hipótesis, longitud del bloque, número de remuestreos y tratamiento de faltantes. Para cada comparación se reportarán diferencia, mejora relativa, intervalo y p ajustado. La equivalencia requiere un contraste específico con el margen del 3 %; no se concluye por ausencia de significación.

El dimensionamiento debe usar dispersión de diferencias pareadas y dependencia temporal. Los errores de validación que participaron en selección no se presentan como confirmación independiente; las pruebas definitivas y la potencia se fijarán antes de calcularlas."""),
        md('## Implementación de remuestreo temporal\n\nLa función siguiente implementa índices de bootstrap circular por bloques. Su ejemplo utiliza un índice artificial, no errores ni resultados de modelos. La longitud ilustrativa de 14 días no constituye una elección inferencial del estudio.'),
        code("""def indices_bootstrap_bloques(n,longitud,rng):
    if not 1<=longitud<=n:
        raise ValueError('La longitud debe estar entre 1 y n.')
    comienzos=rng.integers(0,n,size=int(np.ceil(n/longitud)))
    return ((comienzos[:,None]+np.arange(longitud))%n).ravel()[:n]

rng=np.random.default_rng(11037)
ejemplo=indices_bootstrap_bloques(60,14,rng)
assert ejemplo.shape==(60,) and ejemplo.min()>=0 and ejemplo.max()<60
"""),
        md('En una evaluación futura, los mismos índices se aplicarían conjuntamente a las pérdidas de ambos métodos. El remuestreo debe respetar los bloques de evaluación y cualquier cambio del procedimiento de reajuste; no debe cruzarlos indiscriminadamente.')])

    books['11_sintesis']=[
        md('# 11. Síntesis del análisis exploratorio'),
        code(setup),
        code("""resumen_eda={
    'horas_desarrollo':len(panel),
    'crecimiento_pct':float(100*(gwh.loc['2021'].mean()/gwh.loc['2000'].mean()-1)),
    'acf24':float(acf(gwh,nlags=168,fft=True)[24]) if 'acf' in globals() else float(gwh.autocorr(24)),
    'acf168':float(gwh.autocorr(168))}
crecimiento_eda=resumen_eda['crecimiento_pct']; acf_eda24=resumen_eda['acf24']; acf_eda168=resumen_eda['acf168']
"""),
        note('Entre 2000 y 2021, la demanda media aumenta un {crecimiento_eda:.1f} %. Las autocorrelaciones a 24 y 168 horas son {acf_eda24:.2f} y {acf_eda168:.2f}. El crecimiento y la persistencia aconsejan comparar referencias estacionales y considerar el nivel histórico de cada ventana, manteniendo los parámetros de transformación dentro del entrenamiento.'),
        md('## Consecuencias para la representación y la evaluación\n\n| Hallazgo | Decisión metodológica | Riesgo que debe controlarse |\n|---|---|---|\n| Nivel cambiante y diferencias de dispersión | Normalización por nivel conocido de la ventana y reporte de errores por año. | Confundir cambio de escala con pérdida de capacidad predictiva. |\n| Ciclos diarios, semanales y festivos | Historia de 168 horas y calendario conocido para las 24 horas de salida. | Mezclar sábados con domingos o trasladar observaciones del horizonte a las entradas. |\n| Anomalías y valores originales de hidrología | Conservar datos, indicadores y trazabilidad; evaluar sensibilidad antes de corregir. | Eliminar eventos reales o interpretar una razón como porcentaje. |\n| Asociaciones climáticas y desplazamiento distributivo | Contrastar extensiones mediante ablación y los mismos orígenes. | Confundir asociación contemporánea, tendencia común y señal disponible al origen. |\n| Cobertura y léxico variables entre fuentes | Ajustar representaciones con entrenamiento y medir cobertura al proyectar. | Atribuir a eventos un cambio que proviene del archivo o de la composición editorial. |\n| Errores temporalmente relacionados | Comparaciones pareadas por origen y remuestreo de días en bloques. | Inflar precisión contando horas y semillas como observaciones independientes. |\n\nLa tabla integra decisiones y riesgos; los valores y patrones gráficos se presentan en sus capítulos correspondientes.'),
        md('## Alcance y amenazas a la validez\n\nLa demanda es nacional, mientras el clima resume ocho puntos urbanos; esa diferencia espacial limita la atribución de asociaciones a usos concretos. Los pesos de población de 2018 son estáticos y proceden de una fuente secundaria. El reanálisis meteorológico caracteriza retrospectivamente el tiempo observado, sin reconstruir por sí solo lo disponible en operación.\n\nLas series administrativas y los titulares pueden contener revisiones. La disponibilidad declarada de noticias no acredita una versión histórica exacta; el escenario estricto ofrece mayor trazabilidad con menor cobertura. La ausencia de titulares recuperados no debe convertirse en una señal de ausencia de eventos.\n\nLa validación 2018–2021 combina regímenes distintos y participa en selección. Las limitaciones de independencia de la prueba retrospectiva permanecen en el preregistro. El EDA no demuestra todavía superioridad predictiva, causalidad ni una ganancia atribuible al texto. Los resultados de entrenamiento se incorporarán una vez completada su evaluación.') ]

    books['12_reproducibilidad']=[
        md('# 12. Reproducibilidad y procedencia'),
        md('El libro se reconstruye desde los datos del repositorio y conserva las salidas de cada análisis. El entorno documental se separa del entrenamiento. Las rutas se resuelven a partir de la raíz del repositorio y las fechas se interpretan en America/Bogota.'),
        code("""from pathlib import Path
import sys,platform,json,hashlib,importlib.metadata as metadata
import pandas as pd
from IPython.display import display
ROOT=next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/'data/processed/panel_horario.parquet').exists())
paquetes=['jupyter-book','nbformat','ipython','matplotlib','pandas','pyarrow','scipy','statsmodels','scikit-learn']
entorno={'python':sys.version,'sistema':platform.platform(),
    'dependencias':{p:metadata.version(p) for p in paquetes},'dispositivo_eda':'CPU',
    'zona_horaria':'America/Bogota'}
(ROOT/'results/jbook/entorno.json').write_text(json.dumps(entorno,ensure_ascii=False,indent=2),encoding='utf-8')
display(pd.Series(entorno['dependencias'],name='versión documental'))
"""),
        md('## Identificación de los archivos\n\nLas huellas SHA-256 identifican el contenido binario exacto. Se incluyen fuentes numéricas, titulares, archivos externos y el panel consolidado; se excluyen resultados de entrenamiento. Calcular una huella no deserializa ni evalúa etiquetas de prueba.'),
        code("""def sha256_archivo(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for bloque in iter(lambda:f.read(1024*1024),b''):
            h.update(bloque)
    return h.hexdigest()

rutas=set(p for p in (ROOT/'data/raw/numeric').rglob('*') if p.is_file())
rutas.update(p for p in (ROOT/'data/external').rglob('*') if p.is_file())
rutas.update([ROOT/'data/raw/noticias/titulares.parquet',ROOT/'data/processed/panel_horario.parquet',
    ROOT/'data/processed/feature_registry.json',ROOT/'data/processed/diccionario_datos.json'])
manifiesto=pd.DataFrame([{'archivo':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,
    'sha256':sha256_archivo(p)} for p in sorted(rutas)])
manifiesto.to_csv(ROOT/'results/jbook/manifiesto_entradas.csv',index=False)
n_archivos=len(manifiesto); total_gb=float(manifiesto['bytes'].sum()/1e9)
"""),
        note('El manifiesto registra {n_archivos} archivos y {total_gb:.2f} GB de entradas. Sus huellas permiten detectar cambios de versión antes de reconstruir el análisis. El diccionario y el registro de disponibilidad complementan la identidad binaria con las definiciones semánticas y temporales.'),
        md('## Ejecución en Windows y Linux\n\nCrear un entorno con la versión de Python registrada y usar el archivo de dependencias fijadas. Los comandos siguientes se ejecutan desde la raíz; la activación del entorno depende del sistema operativo.'),
        md(r"""### Windows

```powershell
python -m venv .bookenv
.bookenv\Scripts\python.exe -m pip install -r requirements-book.lock.txt
.bookenv\Scripts\python.exe scripts\crear_jbook.py
.bookenv\Scripts\jupyter-book.exe build book --warningiserror --keep-going
.bookenv\Scripts\python.exe scripts\verificar_jbook.py
```

### Linux

```bash
python3 -m venv .bookenv
.bookenv/bin/python -m pip install -r requirements-book.lock.txt
.bookenv/bin/python scripts/crear_jbook.py
.bookenv/bin/jupyter-book build book --warningiserror --keep-going
.bookenv/bin/python scripts/verificar_jbook.py
```"""),
        md('Los análisis se ejecutan por CPU. El entrenamiento conserva sus dependencias en requirements.txt y permite seleccionar dispositivo mediante la interfaz del repositorio: --dispositivo auto, cpu o cuda. Una instalación de PyTorch compatible con CUDA y el controlador es necesaria para GPU; el entorno documental no la exige. La compatibilidad de rutas no equivale a una ejecución verificada en ambos sistemas: el entorno efectivamente utilizado queda registrado en entorno.json.'),
        md('## Conservación y reconstrucción de la entrega\n\nEl verificador comprueba estructura de cuadernos, celdas ejecutadas, ausencia de errores, interpretaciones junto a las figuras, cobertura del diccionario y enlaces HTML locales. Después copia el HTML construido a docs. No publica ni inicia entrenamiento. Los archivos del libro y sus evidencias se conservan en book, docs y results/jbook; una actualización del sitio requiere una solicitud expresa.')]

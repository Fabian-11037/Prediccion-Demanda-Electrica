"""Reporte estadístico, diagnóstico del EDA y coherencia documental del E1."""
from pathlib import Path
import json,re,shutil,subprocess

PILOTO=r'''# Una realización piloto es la media de sus cuatro años externos.
piloto_semillas=piloto.groupby(['modelo','semilla']).MASE.mean().unstack('modelo').reindex(columns=modelos)
assert piloto_semillas.shape==(5,4)
resumen_piloto=piloto_semillas.agg(['mean','std']).T.rename(columns={'mean':'MASE_media','std':'DE'})
resumen_piloto['CV_pct']=100*resumen_piloto.DE/resumen_piloto.MASE_media
resumen_piloto['n_semillas']=5
display(resumen_piloto.rename(index=etiquetas).round(5))
resumen_piloto.to_csv(OUT/'piloto_resumen.csv')
piloto.groupby(['modelo','fold']).MASE.agg(['mean','std','count']).to_csv(OUT/'piloto_por_fold.csv')
delta_rel=.03;referencia_piloto=float(resumen_piloto.loc['mlp','MASE_media'])
z=stats.norm.ppf(.975)+stats.norm.ppf(.8)
z_familia=stats.norm.ppf(1-.05/(2*3))+stats.norm.ppf(.8)
dimension=[]
for a,b in itertools.combinations(neuronales,2):
    d=(piloto_semillas[a]-piloto_semillas[b])/referencia_piloto;sd=float(d.std(ddof=1))
    dimension.append(dict(comparacion=etiquetas[a]+' − '+etiquetas[b],sigma_d_rel=sd,delta_rel=delta_rel,delta_sobre_sigma=delta_rel/sd,
        R_normal=max(5,int(np.ceil((z*sd/delta_rel)**2))),R_Bonferroni=max(5,int(np.ceil((z_familia*sd/delta_rel)**2))),
        MDD_5_rel=z*sd/np.sqrt(5),MDD_5_pct_MLP=100*z*sd/np.sqrt(5)))
dimension=pd.DataFrame(dimension);display(dimension.round(5));dimension.to_csv(OUT/'dimensionamiento_piloto.csv',index=False)
R_max=int(dimension.R_normal.max());R_familia=int(dimension.R_Bonferroni.max())
'''

PARES=r'''pares=[];qqfig,qqaxes=plt.subplots(1,3,figsize=(13,3.8));ix_boot=np.asarray(list(itertools.product(range(5),repeat=5)))
for ax,(a,b) in zip(qqaxes,itertools.combinations(neuronales,2)):
    aa=calculadas.loc[calculadas.modelo==a].sort_values('semilla').set_index('semilla')
    bb=calculadas.loc[calculadas.modelo==b].sort_values('semilla').set_index('semilla')
    assert aa.index.equals(bb.index)
    d=(aa.MASE-bb.MASE).to_numpy();sd=d.std(ddof=1);se=sd/np.sqrt(5)
    t=stats.ttest_rel(aa.MASE,bb.MASE);w=stats.wilcoxon(d,method='exact');sw=stats.shapiro(d)
    ci=stats.t.ppf(.975,4)*se;bs=d[ix_boot].mean(axis=1)
    stats.probplot(d,dist='norm',plot=ax);ax.set_title(etiquetas[a]+' − '+etiquetas[b])
    pares.append(dict(a=a,b=b,comparacion=etiquetas[a]+' − '+etiquetas[b],diferencia_MASE=d.mean(),IC95_t_inf=d.mean()-ci,IC95_t_sup=d.mean()+ci,
        IC95_boot_inf=np.quantile(bs,.025),IC95_boot_sup=np.quantile(bs,.975),Shapiro_W=sw.statistic,Shapiro_p=sw.pvalue,t=t.statistic,p_t=t.pvalue,
        Wilcoxon_W=w.statistic,p_Wilcoxon=w.pvalue,Cohen_dz=d.mean()/sd,
        reduccion_MAE_a_vs_b_pct=100*(1-aa.MAE.mean()/bb.MAE.mean()),reduccion_RMSE_a_vs_b_pct=100*(1-aa.RMSE.mean()/bb.RMSE.mean())))
qqfig.tight_layout();display(qqfig);plt.close(qqfig)
comparaciones=pd.DataFrame(pares)
comparaciones['p_t_Holm']=multipletests(comparaciones.p_t,method='holm')[1]
comparaciones['p_Wilcoxon_Holm']=multipletests(comparaciones.p_Wilcoxon,method='holm')[1]
comparaciones.to_csv(OUT/'comparaciones_pareadas.csv',index=False)
'''

FRIEDMAN=r'''matriz=calculadas.pivot(index='semilla',columns='modelo',values='MASE').reindex(columns=neuronales)
rk=stats.rankdata(matriz.to_numpy(),axis=1);n,k=rk.shape
assert all(len(np.unique(row))==k for row in matriz.to_numpy())
Q=12*n/(k*(k+1))*np.sum(rk.mean(axis=0)**2)-3*n*(k+1)
permutaciones=np.asarray(list(itertools.permutations(range(k))))
combinaciones=np.asarray(list(itertools.product(range(len(permutaciones)),repeat=n)))
sumas=np.zeros((len(combinaciones),k))
for i in range(n):sumas+=rk[i][permutaciones[combinaciones[:,i]]]
Q_nulo=12*n/(k*(k+1))*np.sum((sumas/n)**2,axis=1)-3*n*(k+1)
p_friedman=float(np.mean(Q_nulo>=Q-1e-12))
friedman=pd.DataFrame([{'Q':Q,'p_exacto':p_friedman,'permutaciones':len(Q_nulo),'bloques_semilla':n}]);display(friedman)
friedman.to_csv(OUT/'friedman_exacto.csv',index=False)
nemenyi=[]
for ia,ib in itertools.combinations(range(k),2):
    diferencia=abs(rk.mean(axis=0)[ia]-rk.mean(axis=0)[ib]);q=diferencia/np.sqrt(k*(k+1)/(6*n))
    nemenyi.append({'comparacion':etiquetas[neuronales[ia]]+' − '+etiquetas[neuronales[ib]],'diferencia_rangos':diferencia,'p_Nemenyi_aprox':stats.studentized_range.sf(q*np.sqrt(2),k,np.inf)})
nemenyi=pd.DataFrame(nemenyi);display(nemenyi.round(5));nemenyi.to_csv(OUT/'nemenyi.csv',index=False)
'''

TOST=r'''# El único margen es relativo a la media poblacional de MLP: ±3 %.
# theta_AB = (E[MASE_A] - E[MASE_B]) / E[MASE_MLP].
delta_rel=.03;equivalencias=[]
mm=calculadas.loc[calculadas.modelo=='mlp'].sort_values('semilla').MASE.to_numpy()
def fieller_pareado(numerador,denominador,confianza=.90):
    n=len(numerador);v=np.cov(numerador,denominador,ddof=1)/n
    x=denominador.mean();y=numerador.mean();q=stats.t.ppf((1+confianza)/2,n-1)**2
    a=x*x-q*v[1,1];b=-2*x*y+2*q*v[0,1];c=y*y-q*v[0,0]
    if a<=0 or b*b-4*a*c<0:return -np.inf,np.inf
    raiz=np.sqrt(b*b-4*a*c)
    return (-b-raiz)/(2*a),(-b+raiz)/(2*a)
for familia,lista in [('Redes',list(itertools.combinations(neuronales,2))),('Redes frente a ARX',[(m,'arx_lineal') for m in neuronales])]:
    filas=[]
    for a,b in lista:
        aa=calculadas.loc[calculadas.modelo==a].sort_values('semilla').MASE.to_numpy();bb=calculadas.loc[calculadas.modelo==b].sort_values('semilla').MASE.to_numpy()
        d=aa-bb
        # H0 inferior: E[d] <= -0.03 E[MLP]; H0 superior: E[d] >= +0.03 E[MLP].
        inferior=stats.ttest_1samp(d+delta_rel*mm,0,alternative='greater')
        superior=stats.ttest_1samp(d-delta_rel*mm,0,alternative='less')
        p=max(inferior.pvalue,superior.pvalue);lo,hi=fieller_pareado(d,mm)
        assert (lo>-delta_rel and hi<delta_rel)==bool(p<.05)
        filas.append({'familia':familia,'comparacion':etiquetas[a]+' − '+etiquetas[b],
            'diferencia_rel_pct':100*d.mean()/mm.mean(),'margen_rel_pct':100*delta_rel,
            'IC90_Fieller_inf_pct':100*lo,'IC90_Fieller_sup_pct':100*hi,
            'p_TOST_inferior':inferior.pvalue,'p_TOST_superior':superior.pvalue,'p_TOST':p})
    adj=multipletests([f['p_TOST'] for f in filas],method='holm')[1]
    for f,p in zip(filas,adj):f['p_TOST_Holm']=p;f['equivalencia_Holm_005']=bool(p<.05)
    equivalencias.extend(filas)
equivalencias=pd.DataFrame(equivalencias);display(equivalencias.round(5));equivalencias.to_csv(OUT/'equivalencia_TOST.csv',index=False)
p_equiv_arx_cnn=float(equivalencias.loc[equivalencias.comparacion=='CNN1D − ARX','p_TOST_Holm'].iloc[0])
relativas=[]
b=benchmark.set_index('familia');mejor_red=b.loc[neuronales].MASE.idxmin();peor_red=b.loc[neuronales].MASE.idxmax()
for m in neuronales:
    relativas.append({'modelo':etiquetas[m],'reduccion_MAE_vs_peor_pct':100*(1-b.loc[m,'MAE_kWh']/b.loc[peor_red,'MAE_kWh']),
        'reduccion_RMSE_vs_peor_pct':100*(1-b.loc[m,'RMSE_kWh']/b.loc[peor_red,'RMSE_kWh']),
        'reduccion_MAE_vs_mejor_pct':100*(1-b.loc[m,'MAE_kWh']/b.loc[mejor_red,'MAE_kWh']),
        'reduccion_RMSE_vs_mejor_pct':100*(1-b.loc[m,'RMSE_kWh']/b.loc[mejor_red,'RMSE_kWh'])})
relativas=pd.DataFrame(relativas);display(relativas.round(3));relativas.to_csv(OUT/'reducciones_relativas.csv',index=False)
umbral_mase=.97*float(mejor.MASE);umbral_mae=umbral_mase*escala
'''

REFERENCIAS=r'''observado=pd.read_parquet(ROOT/'data/processed/panel_horario.parquet',columns=['y_demanda_real_kwh'])['y_demanda_real_kwh']
tiempos=pd.DatetimeIndex([d+pd.Timedelta(hours=h) for d in fechas for h in range(24)])
for nombre,lag in [('ingenuo_24h',24),('ingenuo_168h',168)]:
    dias=lag//24
    # El rezago publicado de demanda impide usar t0-1h; retrasar referencias intradía no disponibles.
    base=np.asarray([observado.loc[t-pd.Timedelta(hours=lag if h<=lag-2 else 2*lag)] for d in fechas for h,t in enumerate(pd.date_range(d,periods=24,freq='h'))]).reshape(y.shape)
    abs_error=np.abs(base-y)/escala
    assert np.isclose(abs_error.mean(),referencias[nombre]['MASE'],rtol=2e-6),(nombre,abs_error.mean())
    loss_h[nombre]=abs_error;loss_d[nombre]=abs_error.mean(axis=1)
todos=modelos+['ingenuo_24h','ingenuo_168h'];perdidas_todas=np.stack([loss_d[m] for m in todos],axis=1)
boot_todas=bootstrap_temporal(perdidas_todas,14)
contrastes_globales=[];contrastes_ref=[]
familias=[('Redes',list(itertools.combinations(neuronales,2))),('Redes frente a ARX',[(m,'arx_lineal') for m in neuronales]),('Frente a ingenuos',[(m,r) for m in modelos for r in ['ingenuo_24h','ingenuo_168h']])]
for familia,lista in familias:
    filas=[];horas=[]
    for a,b in lista:
        d=loss_d[a]-loss_d[b];j=todos.index(a);l=todos.index(b);distrib=boot_todas[:,j]-boot_todas[:,l]
        lo,hi=np.quantile(distrib,[.025,.975]);t,p=dm_hln(d)
        p_boot=(1+np.sum(np.abs(distrib-distrib.mean())>=abs(d.mean())))/(len(distrib)+1)
        filas.append({'familia':familia,'comparacion':etiquetas[a]+' − '+etiquetas[b],'diferencia_MASE':d.mean(),'IC95_boot_inf':lo,'IC95_boot_sup':hi,'DM_HLN':t,'p_DM':p,'p_boot':p_boot,'reduccion_MAE_pct':100*(1-loss_d[a].mean()/loss_d[b].mean())})
        if familia=='Frente a ingenuos':
            for h in range(24):
                t,p=dm_hln(loss_h[a][:,h]-loss_h[b][:,h]);horas.append({'familia':familia,'a':a,'b':b,'hora':h,'diferencia_MASE':(loss_h[a][:,h]-loss_h[b][:,h]).mean(),'DM_HLN':t,'p':p})
    for campo in ['p_DM','p_boot']:
        ajustados=multipletests([f[campo] for f in filas],method='holm')[1]
        for f,p in zip(filas,ajustados):f[campo+'_Holm']=p
    contrastes_globales.extend(filas)
    if horas:
        adj=multipletests([f['p'] for f in horas],method='holm')[1]
        for f,p in zip(horas,adj):f['p_Holm']=p
        contrastes_ref.extend(horas)
globales=pd.DataFrame(contrastes_globales);display(globales[['comparacion','diferencia_MASE','IC95_boot_inf','IC95_boot_sup','p_DM_Holm','p_boot_Holm']].round(6))
globales.to_csv(OUT/'contrastes_temporales_globales.csv',index=False)
pd.concat([dm,pd.DataFrame(contrastes_ref)],ignore_index=True).to_csv(OUT/'DM_24_horizontes.csv',index=False)
intervalos.to_csv(OUT/'intervalos_temporales.csv',index=False)
'''

ATIPICOS=r'''serie=observado
desarrollo=serie.loc[serie.index<'2018-01-01'];log_razon=np.log(desarrollo/desarrollo.shift(168)).dropna()
q1,q3=log_razon.quantile([.25,.75]);iqr=q3-q1;limites=(q1-1.5*iqr,q3+1.5*iqr)
referencia_semana=observado.reindex(tiempos-pd.Timedelta(hours=168)).to_numpy().reshape(y.shape)
razon_objetivo=np.log(y/referencia_semana);atipico=(razon_objetivo<limites[0])|(razon_objetivo>limites[1])
impacto=[]
for m in modelos:
    errores=np.abs(pred[m]-y[None,:,:]);e=errores.mean(axis=0)
    impacto.append({'modelo':etiquetas[m],'horas_atipicas':int(atipico.sum()),'MAE_atipico_kWh':e[atipico].mean(),'MAE_resto_kWh':e[~atipico].mean(),
        'aporte_error_atipico_pct':100*e[atipico].sum()/e.sum(),'cambio_MAE_al_excluir_pct':100*(e[~atipico].mean()/e.mean()-1)})
impacto=pd.DataFrame(impacto);display(impacto.round(3));impacto.to_csv(OUT/'impacto_atipicos_error.csv',index=False)
porcentaje_atipico=100*atipico.mean()
digest=hashlib.sha256();digest.update(b'SIN-demand-test-v1\n');digest.update(np.asarray(origen,dtype='<i8').tobytes());digest.update(np.asarray(y,dtype='<f8').tobytes())
hash_prueba=digest.hexdigest()
(OUT/'huella_prueba.json').write_text(json.dumps({'sha256':hash_prueba,'commit':SHA,'origenes':1461,'horizonte':24,'inicio':str(fechas.min()),'fin':str(fechas.max()),
    'serializacion':'cabecera UTF-8 SIN-demand-test-v1 y LF; orígenes UTC int64 little-endian; etiquetas float64 little-endian, orden C','fecha_calculo':pd.Timestamp.now(tz='America/Bogota').isoformat()},ensure_ascii=False,indent=2),encoding='utf-8')
'''

DRIFT=r'''# Contraste de CDF sobre 19 cuantiles, con permutación de bloques completos.
pruebas_drift=[];rng=np.random.default_rng(11039);bloque=28;B=1999
for col in ['demanda',*covars]:
    a=ent[col].dropna().to_numpy();b=val[col].dropna().to_numpy();na=len(a)//bloque;nb=len(b)//bloque
    cortes=np.quantile(np.r_[a[:na*bloque],b[:nb*bloque]],np.linspace(.05,.95,19));aa=a[:na*bloque].reshape(na,bloque);bb=b[:nb*bloque].reshape(nb,bloque)
    cdf=np.concatenate([(aa[:,:,None]<=cortes).mean(axis=1),(bb[:,:,None]<=cortes).mean(axis=1)])
    estadistico=np.max(np.abs(cdf[:na].mean(axis=0)-cdf[na:].mean(axis=0)));superan=0
    for _ in range(B):
        ix=rng.permutation(na+nb);t=np.max(np.abs(cdf[ix[:na]].mean(axis=0)-cdf[ix[na:]].mean(axis=0)));superan+=t>=estadistico-1e-12
    pruebas_drift.append({'variable':etiquetas[col],'dias_entrenamiento':na*bloque,'dias_validacion':nb*bloque,'D_CDF19':estadistico,'p_permutacion':(superan+1)/(B+1)})
pruebas_drift=pd.DataFrame(pruebas_drift);from statsmodels.stats.multitest import multipletests
pruebas_drift['p_Holm']=multipletests(pruebas_drift.p_permutacion,method='holm')[1];display(pruebas_drift.round(5))
pruebas_drift.to_csv(ROOT/'results/jbook/desplazamiento_formal.csv',index=False)
'''


def corregir(books,md,code,note,root):
    # Diagnóstico del objetivo y sensibilidad de transformaciones sin cambiar entradas del modelo.
    books['01_datos'].insert(2,md('**Tipo de problema:** series de tiempo, con pronóstico multivariado directo de 24 pasos y rezagos del objetivo. **Pregunta de investigación:** ¿qué desempeño y estabilidad presentan MLP, LSTM y CNN1D para anticipar la demanda horaria del SIN colombiano durante las siguientes 24 horas, utilizando demanda histórica y calendario conocido? La etapa documental estudia la disponibilidad y representación de titulares para contrastar posteriormente su contribución incremental.'))
    books['01_datos'][1].source=books['01_datos'][1].source.replace('regresión temporal multivariada','pronóstico de series de tiempo multivariado')
    books['03_demanda'].extend([
        md('## Asimetría, curtosis y transformaciones del objetivo\n\nSe comparan escala original, logaritmo y Box-Cox. El parámetro de Box-Cox se estima exclusivamente con 2000–2017 y se aplica sin reajuste a 2018–2021. La curtosis se expresa como exceso respecto a la normal. El diagnóstico de forma no acredita normalidad ni decide por sí solo una transformación para pronosticar. [Hyndman y Athanasopoulos, §3.1](https://otexts.com/fpp3/decomposition.html#mathematical-transformations), relacionan estas transformaciones con la estabilidad de la variación y advierten de su interpretación al invertirlas.'),
        code("from scipy import stats\nfrom scipy.special import boxcox as aplicar_boxcox\nbase_y=gwh.loc[gwh.index<'2018-01-01'];externo_y=gwh.loc[gwh.index>='2018-01-01'];assert (base_y>0).all() and (externo_y>0).all()\n_,lambda_bc=stats.boxcox(base_y.to_numpy());formas=[]\nfor particion,s in [('Entrenamiento',base_y),('Validación',externo_y)]:\n    for transformacion,v in [('Original',s.to_numpy()),('Log',np.log(s.to_numpy())),('Box-Cox',aplicar_boxcox(s.to_numpy(),lambda_bc))]:\n        formas.append({'particion':particion,'transformacion':transformacion,'asimetria':stats.skew(v,bias=False),'exceso_curtosis':stats.kurtosis(v,fisher=True,bias=False)})\nformas=pd.DataFrame(formas);display(formas.round(4));formas.to_csv(ROOT/'results/jbook/transformaciones_objetivo.csv',index=False)\nasim_original=float(formas.loc[(formas.particion=='Entrenamiento')&(formas.transformacion=='Original'),'asimetria'].iloc[0]);curt_original=float(formas.loc[(formas.particion=='Entrenamiento')&(formas.transformacion=='Original'),'exceso_curtosis'].iloc[0])\n"),
        note('En entrenamiento, la asimetría es {asim_original:.3f} y el exceso de curtosis {curt_original:.3f}; Box-Cox estima λ={lambda_bc:.3f}. El logaritmo aumenta la asimetría en magnitud; Box-Cox la reduce en entrenamiento, pero no mantiene esa ventaja en validación. Reducir asimetría no garantiza menor MAE y exige considerar el sesgo al invertir una transformación. Se conserva la normalización por nivel de los modelos evaluados; estas transformaciones se comparan sólo como diagnóstico.')])
    books['02_calidad'].extend([
        md('## Sensibilidad descriptiva a observaciones extremas\n\nSe utiliza el criterio exploratorio ya definido: demanda inferior a la mitad de la registrada 168 horas antes. Se cuantifica su influencia sin modificar los datos originales.'),
        code("mascara=(y/y.shift(168)<.5).fillna(False);sin_extremos=y.loc[~mascara];dif=y.diff(168).abs();sensibilidad=pd.DataFrame([{'escenario':'Original','media_kWh':y.mean(),'DE_kWh':y.std(),'diferencia_semanal_absoluta_kWh':dif.mean()},{'escenario':'Sin horas marcadas','media_kWh':sin_extremos.mean(),'DE_kWh':sin_extremos.std(),'diferencia_semanal_absoluta_kWh':dif.loc[~mascara].mean()}]);display(sensibilidad.round(2));sensibilidad.to_csv(ROOT/'results/jbook/sensibilidad_atipicos_desarrollo.csv',index=False)\nn_marcadas=int(mascara.sum());cambio_dispersion=100*(sin_extremos.std()/y.std()-1)"),
        note('El criterio marca {n_marcadas} horas de desarrollo y excluirlas cambiaría la desviación estándar un {cambio_dispersion:.2f} %. El efecto sobre diferencias semanales caracteriza sensibilidad descriptiva, sin identificar corrupción de medición. El capítulo de evaluación cuantifica por separado su relación con el error de las predicciones congeladas.')])
    books['05_dependencia'].extend([
        md('## Lectura conjunta de ADF y KPSS\n\n| ADF: rechaza raíz unitaria | KPSS: rechaza estacionariedad | Lectura conjunta |\n|---|---|---|\n| Sí | No | Evidencia compatible con estacionariedad bajo las especificaciones usadas. |\n| No | Sí | Evidencia compatible con no estacionariedad. |\n| Sí | Sí | Resultados en conflicto: revisar rupturas, tendencia, especificación y tamaño de muestra. |\n| No | No | Evidencia insuficiente para resolver; no confirma ninguna de las dos hipótesis. |'),
        code("adf_nivel=float(registros[0]['ADF_p']);kpss_nivel=float(registros[0]['KPSS_p_tabulado']);adf_diferencia=float(registros[1]['ADF_p']);kpss_diferencia=float(registros[1]['KPSS_p_tabulado']);assert adf_nivel>.05 and kpss_nivel<=.05 and adf_diferencia<.05 and kpss_diferencia>.05"),
        note('El nivel diario corresponde al segundo escenario: ADF p={adf_nivel:.3f} no rechaza raíz unitaria y KPSS p≤{kpss_nivel:.2f} rechaza estacionariedad. La diferencia diaria corresponde al primero: ADF p={adf_diferencia:.3g} y KPSS p≥{kpss_diferencia:.2f}. La diferenciación es compatible con estacionariedad en este diagnóstico, sin descartar rupturas ni garantizar estabilidad futura.')])
    books['06_covariables'].extend([
        md('## Prueba de desplazamiento entre particiones\n\nSe contrasta igualdad de distribuciones mediante la máxima diferencia de CDF evaluada en 19 cuantiles del desarrollo agrupado, invariantes al intercambio de etiquetas de partición. Se permutan bloques completos de 28 días, con 1.999 permutaciones y Holm entre variables. Los bloques incompletos finales se excluyen del contraste. El procedimiento conserva dependencia dentro del bloque y supone intercambiabilidad aproximada entre bloques bajo la hipótesis nula; no utiliza horas como observaciones independientes.'),code(DRIFT),
        md('Los valores p ajustados complementan las distancias de Wasserstein: una distancia cuantifica magnitud y el contraste evalúa compatibilidad con igualdad bajo el esquema de permutación. La tendencia y los cambios de régimen pueden explicar un rechazo. No se deduce causalidad ni pérdida de precisión predictiva a partir de ese resultado. [Descargar resultados](_static/resultados/desplazamiento_formal.csv).')])
    for cells in books.values():
        for cell in cells:
            if cell.cell_type=='code' and 'from scipy import stats' in cell.source and 'calculadas=[]' in cell.source:
                cell.source+='\nformato_p=lambda x: "<1e-300" if x==0 else f"{x:.3g}"\n'
    _diseno(books,md,code,note,root)
    _benchmark(books,md,code,note,root)
    _literatura(books,md,code,note,root)
    _discusion_final(books,md,root)
    for cells in books.values():
        for cell in cells:
            if cell.cell_type=='code':
                cell.source=cell.source.replace("display(comparaciones[['comparacion','diferencia_MASE','IC95_t_inf','IC95_t_sup','IC95_boot_inf','IC95_boot_sup','Shapiro_p','p_t_Holm','p_Wilcoxon_Holm','Cohen_dz']].round(5))", "display(comparaciones[['comparacion','diferencia_MASE','IC95_t_inf','IC95_t_sup','IC95_boot_inf','IC95_boot_sup','Shapiro_p','p_t_Holm','p_Wilcoxon_Holm','Cohen_dz']].style.format({'Shapiro_p':formato_p,'p_t_Holm':formato_p,'p_Wilcoxon_Holm':formato_p},precision=5))")
                cell.source=cell.source.replace("display(globales[['comparacion','diferencia_MASE','IC95_boot_inf','IC95_boot_sup','p_DM_Holm','p_boot_Holm']].round(6))", "display(globales[['comparacion','diferencia_MASE','IC95_boot_inf','IC95_boot_sup','p_DM_Holm','p_boot_Holm']].style.format({'p_DM_Holm':formato_p,'p_boot_Holm':formato_p},precision=5))")
                cell.source=cell.source.replace("display(diagnosticos[['modelo','prueba','rezago','estadistico','p','p_Holm']].round(6))", "display(diagnosticos[['modelo','prueba','rezago','estadistico','p','p_Holm']].style.format({'p':formato_p,'p_Holm':formato_p},precision=5))")
    books['10_evaluacion'].append(md('## Fundamento metodológico\n\nLas implementaciones consultan la documentación de [Shapiro–Wilk](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.shapiro.html), [Wilcoxon](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html), [rango studentizado](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.studentized_range.html) y [TOST pareado](https://www.statsmodels.org/stable/generated/statsmodels.stats.weightstats.ttost_paired.html). [Demšar (2006)](https://jmlr.org/papers/v7/demsar06a.html) desarrolla comparaciones por rangos entre conjuntos de datos; aquí se explicita que los bloques son inicializaciones de un mismo conjunto, con inferencia condicional y no generalización entre sistemas eléctricos. Los p conservan notación científica; valores numéricamente subdesbordados se muestran como <1e-300.'))
    # Numeración de anexos y archivos coherente con el índice del libro.
    books['10_evaluacion'][0]=md('# 10.9. Evaluación temporal y diagnósticos')
    nuevos={ '08_validacion':'09_validacion','09_modelos':'10_modelos'}
    renombrados={nuevos.get(k,k):v for k,v in books.items()};books.clear();books.update(renombrados)
    for cells in books.values():
        for cell in cells:
            for antiguo,nuevo in nuevos.items():cell.source=cell.source.replace(antiguo+'.ipynb',nuevo+'.ipynb').replace(antiguo+' ',nuevo+' ')
            if cell.metadata.get('plantilla_resultados'):cell.metadata['plantilla_resultados']=cell.source
    p=root/'book/10_pipeline.md';s=p.read_text(encoding='utf-8');s=s.replace('# Pipeline de entrenamiento y evaluación','# 10.10. Pipeline de entrenamiento y evaluación');p.write_text(s,encoding='utf-8')


def _diseno(books,md,code,note,root):
    c=books['08_validacion']
    for cell in c:
        if cell.cell_type!='markdown':continue
        if cell.source.startswith('## Búsqueda y prevención'):
            cell.source='''## Búsqueda y prevención de fuga

La selección utiliza confirmación de validación con dos semillas y cuatro años externos. Los escaladores de características se ajustan exclusivamente con las filas de optimización de cada pliegue; parada interna y evaluación externa permanecen separadas. MLP y ARX ajustan además su estandarización tabular sobre el mismo tramo de optimización. El nivel histórico de cada ventana se calcula antes del origen. La escala fija de MASE de 2000–2017 es un denominador de evaluación y no una entrada del optimizador.

El análisis temporal emplea bloques de 14 días, con siete y 28 como sensibilidad; 10.000 réplicas e índices apareados. DM–HLN utiliza covarianza HAC de 14 rezagos diarios. Las familias de contrastes y sus ajustes de multiplicidad se especifican en el capítulo de evaluación. Los resultados son exploratorios y se distinguen de una confirmación independiente.'''
        if cell.source.startswith('## Unidad de evaluación'):
            cell.source=cell.source.replace('se estimaría','se estima').replace('se reportaría','se reporta')
            a=cell.source.index('El protocolo inferencial por completar')
            cell.source=cell.source[:a]+'''Se comparan las tres redes entre sí, cada red frente a ARX y cada familia frente a los ingenuos diario y semanal. Holm se aplica separadamente por familia de hipótesis; los horizontes se ajustan conjuntamente dentro de cada familia. Las diferencias, intervalos, tamaños de efecto y valores p se reportan junto con sus supuestos. TOST examina θ_AB=(μ_A−μ_B)/μ_MLP con un único margen relativo de ±3 %, sin inferir equivalencia por ausencia de significación.

El dimensionamiento usa diferencias de MASE entre semillas piloto, agregadas primero sobre los cuatro años externos. El resultado se interpreta como planificación aproximada condicionada al piloto; no convierte las fechas compartidas en réplicas independientes.'''
    for cell in c:
        if cell.cell_type=='code' and 'p=int(candidatos[100])' in cell.source:
            cell.source=cell.source.replace('p=int(candidatos[100])','p=int(val[0])')
            cell.source=cell.source.replace("train=panel.loc[panel.index<'2018-01-01']; media_cal=train[calendario].mean(); escala_cal=train[calendario].std().replace(0,1)","filas_ajuste=panel.iloc[int(tr.min())-L:int(tr.max())]; assert filas_ajuste.index.max()<indice[es[0]]; media_cal=filas_ajuste[calendario].mean(); escala_cal=filas_ajuste[calendario].std().replace(0,1)")
            cell.source=cell.source.replace('escala_mase=float(train.y_demanda_real_kwh.diff(168).abs().dropna().mean())',"escala_mase=float(panel.loc[panel.index<'2018-01-01','y_demanda_real_kwh'].diff(168).abs().dropna().mean())")
    sha=json.loads((root/'results/jbook/modelos_actualizados.json').read_text())['commit'];snap=root/'results/repositorio'/sha
    filas=[]
    for grupo,carpeta,plan in [('MLP, ARX y CNN1D','base_v3','PLAN_REENTRENAMIENTO_V3.md'),('LSTM','base_v4','PLAN_VARIANTES_V4.md')]:
        log=subprocess.check_output(['git','log','-1','--format=%H|%aI',sha,'--',plan],cwd=root,text=True).strip().split('|')
        close=json.loads((snap/'results'/carpeta/'seleccion_cerrada.json').read_text(encoding='utf-8'))
        done=json.loads((snap/'results'/carpeta/'ejecucion_completa.json').read_text(encoding='utf-8'))
        filas.append('| '+grupo+' | ['+log[1]+'](https://github.com/Fabian-11037/Prediccion-Demanda-Electrica/commit/'+log[0]+') | '+close['fecha']+' | '+done['fecha']+' |')
    c.extend([md('## Registro del diseño y cierre de selección\n\n| Familias seleccionadas | Fecha del commit del plan | Cierre de selección y piloto | Finalización de evaluación |\n|---|---|---|---|\n'+'\n'.join(filas)+'\n\nLas fechas se conservan con el huso de los registros. Los planes y el cierre de selección están documentados antes de finalizar la evaluación. El piloto existe en los artefactos; el cálculo de dimensionamiento que se presenta en el informe es retrospectivo y no acredita que determinara las cinco semillas ejecutadas. Se preserva la distinción entre evidencia de cronología y cumplimiento de planificación de potencia.'),
        md('## Identidad del conjunto de prueba\n\nLa huella visible en el capítulo de evaluación identifica 1.461 orígenes UTC y sus 24 etiquetas, mediante serialización canónica int64/float64 little-endian. Las etiquetas coinciden en las 20 realizaciones seleccionadas. La huella de contenido permite verificar identidad; por sí sola no acredita la fecha en que se congeló el conjunto. [Especificación y SHA-256](_static/resultados/huella_prueba.json).')])


def _benchmark(books,md,code,note,root):
    c=books['09_modelos'];idx=next(i for i,x in enumerate(c) if x.cell_type=='markdown' and x.source.startswith('## 10.4.'))
    c[idx:idx]=[md('### Piloto y dimensionamiento\n\nLa media por semilla agrega los cuatro años externos antes de estimar dispersión. Los 20 resultados semilla–año no se consideran 20 inicializaciones independientes. Se conserva δ relativo=0,03. Las diferencias de MASE y su DE se expresan respecto al MASE medio piloto de MLP; esta conversión usa únicamente validación y no fija un margen absoluto adicional. Para cada diferencia pareada se calcula R≈[(z₀,₉₇₅+z₀,₈)σ̂d/δ]², δ/σ̂d y la diferencia mínima detectable con cinco semillas. Se añade planificación conservadora con Bonferroni para tres comparaciones.'),code(PILOTO),
        note('El máximo requerido para el margen relativo del 3 % por la aproximación normal individual es {R_max} semillas y asciende a {R_familia} con Bonferroni. Son estimaciones sensibles a cinco diferencias piloto y no una garantía de potencia. Si R supera las cinco realizaciones ejecutadas, el diseño no acredita potencia del 80 % para ese contraste. [Resumen piloto](_static/resultados/piloto_resumen.csv), [detalle por año](_static/resultados/piloto_por_fold.csv) y [dimensionamiento](_static/resultados/dimensionamiento_piloto.csv).')]
    for cell in c:
        if cell.cell_type=='code' and 'conteos={' in cell.source and 'fig,axes=plt.subplots(1,2,figsize=(11,3.7))' in cell.source:
            cell.source=cell.source.split('fig,axes=plt.subplots(1,2,figsize=(11,3.7))')[0]
    # La figura muestra realizaciones; la tabla conserva los agregados exigidos por la pauta.
    for i,cell in enumerate(c):
        if cell.cell_type=='code' and 'fig,ax=plt.subplots(figsize=(10,4.8))' in cell.source:
            cell.source="fig,ax=plt.subplots(figsize=(10,4.8))\nfor i,m in enumerate(modelos):\n    v=calculadas.loc[calculadas.modelo==m].sort_values('semilla').MASE;ax.scatter(v,np.full(5,i)+np.linspace(-.1,.1,5),label=etiquetas[m])\nfor r in ['ingenuo_168h','ingenuo_24h']:ax.axvline(referencias[r]['MASE'],ls='--',label=etiquetas[r])\nax.set(yticks=range(4),yticklabels=[etiquetas[m] for m in modelos],xlabel='MASE de cada inicialización; menor es mejor');ax.invert_yaxis();ax.legend(fontsize=8);fig.tight_layout();display(fig);plt.close(fig)"
        if cell.cell_type=='code' and 'display(benchmark.set_index' in cell.source:
            cell.source="display(benchmark.set_index('Modelo')[['MASE','DE','IC95_inf','IC95_sup','MAE_kWh','RMSE_kWh','sMAPE_pct','R2']].round(5))\nlatencias=pd.read_csv(OUT/'latencia_inferencia.csv');costo=benchmark.loc[benchmark.familia.isin(modelos),['familia','Modelo','parametros','ajuste_s']].set_index('familia');costo=costo.join(latencias.groupby('modelo')[['mediana_ms','p95_ms']].median());display(costo.set_index('Modelo').round(4));costo.to_csv(OUT/'costo_modelos.csv')"
        if cell.cell_type=='markdown' and 'La tabla añade indicadores complementarios' in cell.source:
            cell.source='La figura muestra las cinco realizaciones individuales; la tabla presenta sus agregados, intervalos t95 % y métricas complementarias. [Benchmark completo](benchmark_modelos.csv). El costo se resume para el mismo equipo de ajuste; la latencia se mide por separado en CPU, con un origen y 24 salidas.'
    c.extend([md('## Discusión crítica y margen disponible\n\nLa MLP obtiene el menor MASE medio, coherente con una representación directa del perfil semanal y calendario. Esta asociación no demuestra que su sesgo inductivo cause la ventaja: arquitectura, regularización y presupuesto cambian conjuntamente. La LSTM incorpora memoria recurrente y la CNN filtros compartidos; su mayor estructura temporal no garantiza menor error con estas entradas. ARX ofrece un control lineal competitivo y CNN1D presenta una media cercana a él.\n\nLa generalización se valora mediante años externos, prueba y perfiles de error, no por la pérdida de ajuste. La complejidad debe contrastarse con parámetros, duración de ajuste y latencia medida en el mismo dispositivo de inferencia. Los tiempos de CPU y GPU corresponden a procedimientos distintos y no se convierten entre sí.\n\nLa evidencia estadística se presenta en el análisis temporal con alcance exploratorio. Una mejora práctica exige reducir al menos un 3 % el MASE de MLP; distinguirla estadísticamente requiere considerar su dispersión y la dependencia temporal. La referencia semanal contextualiza desempeño operativo, pero no estima ruido irreducible. No hay mediciones replicadas ni un modelo validado de error instrumental que permita identificar una cota de ruido para esta serie; el residuo del mejor modelo tampoco la identifica.\n\nLos estudios de demanda revisados muestran que ordenar arquitecturas depende del sistema, las entradas y el horizonte. Sus errores no se convierten en una mejora transferible al SIN. El espacio de mejora debe verificarse con covariables disponibles al origen, ablaciones y un protocolo prospectivo; el presupuesto desigual y cinco inicializaciones limitan atribuciones fuertes sobre familias.'),
        md('### Protocolo de medición de inferencia\n\nSe cargan los 20 checkpoints sin modificar sus pesos. CPU, un hilo, batch uno, 20 pasadas de calentamiento y 200 mediciones por checkpoint. Se cronometra sólo forward con entradas sintéticas finitas de dimensiones operativas y 24 salidas; se excluyen carga, preparación, transferencia y escritura. Las medianas y p95 por familia agregan los valores de sus cinco checkpoints. [Entorno medido](_static/resultados/entorno_inferencia.json) y [mediciones por checkpoint](_static/resultados/latencia_inferencia.csv).\n\n'+chr(96)*3+'python\n'+(root/'scripts/medir_inferencia.py').read_text(encoding='utf-8')+'\n'+chr(96)*3)])
    e=books['10_evaluacion'];idx=len(e)-1
    e[idx:idx]=[md('## Comparaciones pareadas entre inicializaciones\n\nSe comparan las mismas cinco semillas. Shapiro–Wilk y Q–Q describen diferencias; con n=5 no rechazar normalidad no la demuestra. Se presenta t pareada como aproximación paramétrica, Wilcoxon bilateral exacto como sensibilidad y bootstrap empírico exhaustivo de los 5⁵ remuestreos posibles. Holm corrige separadamente las tres comparaciones de cada método. Cohen dz utiliza la DE de diferencias pareadas.'),code(PARES),
        md('Los Q–Q permiten detectar desviaciones visibles, pero cinco puntos ofrecen poca resolución. La independencia relevante es entre inicializaciones condicionadas a las mismas fechas; no implica independencia temporal. Los intervalos bootstrap describen la muestra empírica de cinco pares y no compensan su tamaño reducido.'),
        code("display(comparaciones[['comparacion','diferencia_MASE','IC95_t_inf','IC95_t_sup','IC95_boot_inf','IC95_boot_sup','Shapiro_p','p_t_Holm','p_Wilcoxon_Holm','Cohen_dz']].round(5))"),
        md('Las tres t pareadas con Holm producen p entre 0,000582 y 0,006434, mientras Wilcoxon ajustado es 0,1875 en las tres parejas. La t aporta evidencia condicional bajo su aproximación; la diferencia entre métodos refleja su resolución y supuestos. Una diferencia negativa favorece al primer modelo. Wilcoxon bilateral exacto tiene p mínimo 0,0625 con cinco pares sin ceros ni empates: su falta de rechazo no es evidencia de igualdad. La t y el bootstrap deben leerse con sus supuestos y junto con la evaluación temporal. [Resultados completos, incluidos p sin ajustar](_static/resultados/comparaciones_pareadas.csv).'),
        md('## Contraste global y comparación de rangos\n\nFriedman se evalúa por enumeración de las 6⁵ permutaciones de etiquetas dentro de semilla, evitando depender sólo de su aproximación asintótica. Nemenyi se informa como post-hoc aproximado; el rango studentizado incorpora la comparación conjunta de las tres familias.'),code(FRIEDMAN),
        md('Friedman obtiene Q=10 y p exacto=0,000772. Nemenyi aproximado produce p=0,00446 para MLP–LSTM y 0,25378 para las otras dos parejas. Contrasta diferencias de rangos, no igualdad de MAE promedio. Un resultado global no identifica por sí solo qué parejas difieren. Nemenyi es aproximado con cinco bloques y no reemplaza las diferencias pareadas y sus intervalos. [Friedman](_static/resultados/friedman_exacto.csv), [Nemenyi](_static/resultados/nemenyi.csv).'),
        md('## Equivalencia y reducciones relativas\n\nTOST contrasta θ_AB=(μ_A−μ_B)/μ_MLP dentro de [−0,03;0,03], siendo μ_MLP positiva. Se aplican dos t unilaterales a las combinaciones pareadas d+0,03·MLP y d−0,03·MLP; así se incorpora la incertidumbre de la referencia, sin sustituirla por un margen absoluto estimado en prueba. Se reportan IC90 % de Fieller individuales y p con Holm, por separado entre redes y frente a ARX. La equivalencia requiere rechazar ambas hipótesis unilaterales; no se deriva de medias próximas. Las reducciones de MAE y RMSE se calculan respecto a la mejor y la peor red del benchmark.'),code(TOST),
        note('Para CNN1D frente a ARX, TOST produce p ajustado={p_equiv_arx_cnn:.4f}; no se establece equivalencia al nivel 0,05 bajo el margen relativo del 3 % y los supuestos paramétricos. El umbral práctico para un modelo avanzado es MASE < {umbral_mase:.6f}, equivalente a MAE < {umbral_mae:,.2f} kWh con la escala fija. Cruzar ese umbral no sustituye un intervalo o contraste de superioridad. [TOST](_static/resultados/equivalencia_TOST.csv), [reducciones relativas](_static/resultados/reducciones_relativas.csv).'),
        md('## Contrastes temporales globales y referencias estacionales\n\nSe reconstruyen ingenuos diario y semanal con la misma disponibilidad al origen y se verifican sus métricas publicadas. El bootstrap circular de 14 días utiliza índices comunes para todos los métodos. Se reportan DM–HLN global, bootstrap centrado y Holm en tres familias: redes, redes frente a ARX y modelos frente a los dos ingenuos. Los 192 contrastes horarios frente a ingenuos forman una familia separada.'),code(REFERENCIAS),
        md('Los intervalos describen diferencias temporales de pérdida promediada entre inicializaciones. Una reducción positiva favorece al primer método; una diferencia negativa de MASE también lo favorece. Los resultados frente a ingenuos evalúan su utilidad como referencia sobre los mismos días. [Contrastes globales](_static/resultados/contrastes_temporales_globales.csv), [DM de todas las horas](_static/resultados/DM_24_horizontes.csv), [sensibilidad al bloque](_static/resultados/intervalos_temporales.csv).'),
        md('## Impacto de valores atípicos sobre el error e identidad de prueba\n\nSe fijan umbrales de Tukey, Q1−1,5IQR y Q3+1,5IQR, sobre el logaritmo de la razón entre demanda y su valor 168 horas antes, usando sólo 2000–2017. Se aplican a las etiquetas de prueba como diagnóstico de segmentos. Los modelos y sus predicciones permanecen congelados; excluir horas en esta tabla es una sensibilidad del cálculo del error, no un ajuste del modelo.'),code(ATIPICOS),
        note('Los umbrales marcan el {porcentaje_atipico:.2f} % de las horas evaluadas. En MLP concentran el 8,63 % del error absoluto; excluirlas reduce su MAE un 1,56 %. Para CNN1D la reducción descriptiva es 3,35 %. El benchmark conserva todas las horas. Un valor extremo relativo puede corresponder a un festivo o cambio real de actividad, sin ser un error de medición. [Sensibilidad del error](_static/resultados/impacto_atipicos_error.csv).\n\n**SHA-256 de orígenes y etiquetas de prueba:** `{hash_prueba}`. La especificación canónica y la fecha de cálculo están en la [huella del conjunto](_static/resultados/huella_prueba.json).'),
        md('## Resultados numéricos y archivos\n\nAdemás de los contrastes, se conservan [intervalos temporales](_static/resultados/intervalos_temporales.csv), [errores por hora](_static/resultados/errores_24_horizontes.csv), [errores por segmento](_static/resultados/errores_segmentos.csv), [residuos y Holm](_static/resultados/diagnosticos_residuos.csv) y [componentes de variabilidad](_static/resultados/componentes_variacion.csv).'),
        code("display(intervalos.loc[intervalos.bloque==14].round(5));display(diagnosticos[['modelo','prueba','rezago','estadistico','p','p_Holm']].round(6))"),
        md('Los intervalos de 14 días son la estimación temporal principal; los otros bloques evalúan sensibilidad. Los diagnósticos de residuos examinan estructura remanente y no prueban superioridad predictiva. Los archivos completos preservan precisión numérica y valores p sin ajustar.')]
    for cell in e:
        if cell.cell_type=='markdown':cell.source=cell.source.replace('La tabla descargable de contrastes conserva','La [tabla de contrastes](_static/resultados/DM_24_horizontes.csv) conserva')


def _literatura(books,md,code,note,root):
    c=books['08_literatura']
    c.insert(2,md('### Protocolo de elegibilidad y alcance de la búsqueda\n\nLa ampliación dirigida de demanda considera publicaciones de 2018 al 8 de noviembre de 2026. Se incluyen estudios empíricos de demanda eléctrica con datos, horizonte y evaluación identificables, y métodos pertinentes para representación temporal o documental. Para contar comparadores se exige texto completo y evaluación explícita de una familia; las revisiones orientan contexto, los abstracts generan candidatos y los modelos propuestos o componentes se excluyen del conteo. Se excluyen duplicados, retiradas, trabajos sin resultados verificables y aplicaciones sin vínculo metodológico justificado. País de afiliación y sistema eléctrico se registran por separado.\n\nLas búsquedas públicas del 8 de noviembre de 2026 incluyeron: `(load forecasting OR electricity demand forecasting) AND (CNN OR convolutional OR LSTM OR MLP) AND (comparison OR benchmark)` y las consultas literales conservadas en el registro. No se dispone de las cadenas originales ni las fechas ejecutadas de Scopus y WoS; se mantienen como información no recuperada. Las decisiones actuales de elegibilidad no se presentan como un protocolo fechado antes de las búsquedas originales. El flujo entre bases sigue sin conciliación completa. [Registro de la ampliación pública](_static/literature/busqueda_demanda_20261108.json).'))
    idx=next(i for i,x in enumerate(c) if x.cell_type=='markdown' and x.source.startswith('## 8.3.'))
    c.insert(idx,md('### Comparadores adicionales de demanda revisados en texto completo\n\n| Estudio | Afiliación / datos | Comparadores y resultados | Alcance crítico |\n|---|---|---|---|\n| [Kathirgamanathan et al. (2022), DOI 10.1049/tje2.12132](https://doi.org/10.1049/tje2.12132) | Canadá / Nueva Inglaterra, EE. UU. | ANN de dos capas, CNN1D y LSTM independientes; MAPE medio 4,137 %, 3,911 % y 3,894 %, respectivamente, tabla 7. | Ajuste 2004–2007, prueba 2009. Tabla 4 identifica CNN1D; la descripción de convoluciones no coincide completamente con esa tabla. El recorte de épocas de §8.3 limita comparación con la configuración completa. |\n| [Laitsos et al. (2023), DOI 10.3390/math11132912](https://doi.org/10.3390/math11132912) | Grecia y EE. UU. / Thira, Grecia | MLP, LSTM, CNN1D, CNN-LSTM y TCN frente a ConvLSTM encoder–decoder. MAE 0,4477; 0,4275; 0,4936; 0,4412; 0,4576 y 0,4142 MW, tabla 4. | Datos horarios 2017–2019; últimos tres meses para prueba. Veinte iteraciones de optimización por modelo. No se acredita repetición multisemilla; el ajuste exclusivo del Min–Max en entrenamiento requiere verificación. |\n\nAmbos estudios documentan CNN1D como comparador de demanda. Sus configuraciones y escalas difieren del SIN; respaldan la familia, sin transferir cifras ni demostrar que sea universalmente superior.'))
    for cell in c:
        if cell.cell_type=='markdown':
            cell.source=cell.source.replace('27 estudios destacados','29 estudios destacados').replace('De los 27 estudios','De los 29 estudios').replace('cuatro textos completos sobre demanda y 23','seis textos completos sobre demanda y 23')
            if 'El denominador contiene tres estudios empíricos' in cell.source:
                cell.source=cell.source.replace('tres estudios empíricos','cinco estudios empíricos').replace('Pelekis et al.','Pelekis et al., Kathirgamanathan et al. y Laitsos et al.')
            if cell.source.startswith('LSTM aparece en dos de los tres estudios'):
                cell.source='LSTM aparece en cuatro de cinco estudios empíricos; MLP en tres; CNN1D y TCN, en dos cada una. CNN-LSTM, N-BEATS y TFT aparecen una vez. CNN1D queda empatada en la tercera frecuencia del corpus dirigido. Este recuento justifica su uso como comparador documentado de demanda, sin estimar prevalencia universal. Los dos artículos de Bai comparten línea de investigación y sistema eléctrico; cuentan como publicaciones, sin constituir replicaciones independientes entre países. Los ensambles y las ramas de compresión textual no se descomponen como baselines individuales.'
            cell.source=cell.source.replace('La ampliación es posterior a los entrenamientos y no modifica el diseño ni las configuraciones ejecutadas.','')
        elif cell.cell_type=='code':
            if "demanda=pd.DataFrame" in cell.source:
                cell.source=cell.source.replace("familias=['LSTM'", "demanda=pd.concat([demanda,pd.DataFrame([('Kathirgamanathan','MLP/DNN'),('Kathirgamanathan','LSTM'),('Kathirgamanathan','CNN'),('Laitsos','MLP/DNN'),('Laitsos','LSTM'),('Laitsos','CNN'),('Laitsos','TCN'),('Laitsos','CNN-LSTM')],columns=demanda.columns)],ignore_index=True)\nfamilias=['CNN-LSTM','LSTM'")
                cell.source=cell.source.replace("f'{v}/3'","f'{v}/5'").replace('xlim=(0,3.3),xticks=range(4)','xlim=(0,5.3),xticks=range(6)')
            if 'matriz=pd.concat([extra,matriz])' in cell.source:
                cell.source=cell.source.replace('matriz=pd.concat([extra,matriz])',"extra=pd.concat([extra,pd.DataFrame([[0,1,0,1,1],[0,1,0,1,1]],index=['D05 Kathirgamanathan','D06 Laitsos'],columns=matriz.columns)])\nmatriz=pd.concat([extra,matriz])")
    c.append(md('## Brecha de investigación y pregunta contrastable\n\nEn el corpus auditado, los estudios con titulares y demanda se concentran en Reino Unido y los benchmarks numéricos utilizan otros sistemas eléctricos. No se identificó en estos estudios una comparación de demanda del SIN colombiano a 24 horas que combine disponibilidad histórica verificable de titulares, control del cambio de cobertura editorial y ablación numérica/documental con los mismos orígenes. La ausencia en este corpus no acredita inexistencia en toda la literatura.\n\nLa brecha operativa del proyecto es determinar si el contexto documental disponible antes del origen añade capacidad predictiva al baseline numérico del SIN bajo cambios de régimen. La hipótesis contrastable es una reducción relativa de MASE de al menos el 3 %, acompañada de incertidumbre pareada por bloques, frente al baseline congelado y con presupuesto comparable. La siguiente etapa debe separar utilidad del contenido, cobertura de fuentes y selección de representación mediante ablaciones. Los resultados actuales establecen la referencia numérica y no prueban todavía esa contribución del texto.\n\nTime-MMD, NSW-EPNews y los estudios de precio se conservan como trabajos de otros autores con aportaciones metodológicas complementarias; no se cuentan como evidencia empírica de demanda. Los doce abstracts se mantienen como contexto, fuera del denominador de comparadores de texto completo.'))


def exportar(root):
    static=root/'book/_static/resultados';static.mkdir(parents=True,exist_ok=True)
    for folder in [root/'results/jbook/modelos_actuales',root/'results/jbook']:
        for p in folder.iterdir():
            if p.is_file() and p.suffix in {'.csv','.json'}:shutil.copyfile(p,static/p.name)
    shutil.copyfile(root/'literature/busqueda_demanda_20261108.json',root/'book/_static/literature/busqueda_demanda_20261108.json')
    for antiguo in ['08_validacion','09_modelos']:
        for p in [root/'book'/(antiguo+'.ipynb'),root/'book/_build/html'/(antiguo+'.html'),root/'docs'/(antiguo+'.html')]:
            if p.exists():p.unlink()
        for folder in ['book/_build/html/_sources','docs/_sources']:
            for ext in ['.ipynb','.ipynb.txt']:
                p=root/folder/(antiguo+ext)
                if p.exists():p.unlink()


def _discusion_final(books,md,root):
    for cells in books.values():
        for cell in cells:
            if cell.cell_type!='markdown':continue
            cell.source=cell.source.replace('la latencia de inferencia no fue medida.',
                'la latencia de inferencia se midió por separado en CPU, con el protocolo y los valores reportados en este capítulo.')
            cell.source=cell.source.replace('Los planes y el cierre de selección están documentados antes de finalizar la evaluación.',
                'Los planes preceden al cierre de selección de cada grupo, pero el plan de LSTM de las 12:15:50 se registró después de finalizar la evaluación de MLP, ARX y CNN1D a las 12:02:06. Sus resultados de prueba ya estaban disponibles; este orden impide asumir una planificación conjunta independiente de esa información. La cronología acredita disponibilidad, sin demostrar por sí sola qué resultados se usaron para cada decisión. Además, el bloque 2022–2025 había sido consultado en evaluaciones anteriores: no constituye prueba independiente ni prospectiva. Cerrar una configuración antes de su propia evaluación no elimina esa exposición.')
            cell.source=cell.source.replace('Se preserva la distinción entre evidencia de cronología y cumplimiento de planificación de potencia.',
                'No se cumple el orden piloto → dimensionamiento → número de semillas de §4.1: el dimensionamiento presentado no determinó las cinco realizaciones. Se conserva como limitación metodológica, junto con el presupuesto desigual y la potencia insuficiente para el margen del 3 %.')
    c=books['08_literatura'];idx=next(i for i,x in enumerate(c) if x.cell_type=='markdown' and x.source.startswith('### Extensiones'))
    c[idx:idx]=[
        md('### 8.6.1. MLP sobre ventana de rezagos\n\nLa MLP representa interacciones no lineales entre una ventana fija y calendario mediante capas densas. Su sesgo inductivo no impone recurrencia ni invariancia a traslaciones temporales; cada posición puede aprender una función distinta. Es adecuada cuando los perfiles y el calendario explican gran parte de la variación, pero puede depender de una selección suficiente de rezagos y de regularización. Su paralelización es directa y su costo crece con longitud de entrada, ancho y profundidad.\n\nPelekis et al., Kathirgamanathan et al. y Laitsos et al. incluyen modelos densos como comparadores. Los datasets, DOI, resultados y dispersión disponible están en §8.2; sus configuraciones no son idénticas a las del SIN. La arquitectura seleccionada y su pérdida, optimizador, parámetros y costo se detallan en §10.2. La referencia lineal ARX contrasta si la capacidad no lineal añade precisión bajo las entradas observadas.'),
        md('### 8.6.2. LSTM sobre secuencias\n\nLa LSTM utiliza puertas y estado recurrente para representar dependencias de una secuencia. El último estado comprime la historia, por lo que una memoria más larga no garantiza conservar toda la información relevante. Su capacidad permite relaciones temporales flexibles, pero depende de escalado, longitud de secuencia y regularización. El procesamiento recurrente introduce una dependencia de cálculo entre pasos y puede elevar el tiempo respecto a una red densa de tamaño comparable.\n\nLas comparaciones de demanda de §8.2 documentan LSTM numéricas y variantes con documentos. El uso de texto, los horizontes y el presupuesto difieren entre artículos. No se atribuye una mejora a la recurrencia aislada ni se equipara el error entre sistemas eléctricos. §10.2 reproduce la configuración ejecutada y §10.9 contrasta estabilidad por fechas e inicializaciones.'),
        md('### 8.6.3. CNN1D sobre historia temporal\n\nLa CNN1D comparte filtros entre posiciones y favorece patrones locales. La longitud del kernel, el agrupamiento y la cabeza de salida determinan cuánto contexto puede representar. El entrenamiento permite paralelizar posiciones; la cabeza aplanada del modelo ejecutado conserva dependencias con el tamaño de ventana y puede concentrar una parte importante de los parámetros. La arquitectura no incorpora por sí sola memoria recurrente.\n\nLos dos textos completos incorporados identifican CNN1D como comparador de demanda; sus configuraciones y límites se registran en §8.2. La frecuencia observada empata con TCN en el corpus dirigido, sin justificar que ambas arquitecturas sean intercambiables. TCN conserva su categoría porque añade estructura temporal propia. §10.2 contiene las fórmulas de capacidad y costo de la CNN seleccionada.'),
        md('### 8.6.4. Comparación crítica de robustez y escalabilidad\n\n| Aspecto | MLP | LSTM | CNN1D |\n|---|---|---|---|\n| Representación | Ventana fija con pesos por posición. | Estado recurrente que resume la secuencia. | Filtros locales compartidos y cabeza temporal. |\n| Sensibilidad a revisar | Rezagos, ancho y representación del calendario. | Escalado, longitud, compresión del estado y regularización. | Kernel, agrupamiento, campo receptivo y dimensión de la cabeza. |\n| Escalabilidad esperada | Paralelización densa; capacidad crece con entrada y ancho. | Costo dependiente de secuencia y recurrencia. | Convoluciones paralelizables; cabeza aplanada crece con ventana. |\n| Evidencia de robustez | Comparar años, semillas y entradas equivalentes. | Separar variabilidad de inicialización y cambio de régimen. | Examinar conservación de picos y sensibilidad a cambios locales. |\n\nLa robustez no puede ordenarse universalmente desde estudios con distintos datos, épocas y búsquedas. El corpus presenta resultados heterogéneos y reporte desigual de repeticiones; una media publicada sin dispersión limita la comparación. Los diagnósticos propios por año, hora, tipo de día, atípicos y permutación de demanda permiten contrastar mecanismos en el SIN, sin demostrar invariancia causal. Las complejidades teóricas tampoco sustituyen la latencia medida: la implementación, el dispositivo y el tamaño del lote condicionan el costo.'),
        md('### Alcance del cumplimiento bibliográfico\n\nEl corpus dirigido incluye cinco estudios empíricos de demanda para el recuento de comparadores y una revisión de contexto. Los doce abstracts son material contextual y no evidencia elegible para ese recuento. No se recuperaron las cadenas ni las fechas originales de Scopus y Web of Science, no existe un flujo PRISMA conciliado entre todas las bases y no hubo un segundo revisor independiente. Las secciones por modelo y la comparación crítica completan la presentación, pero no convierten la revisión dirigida en una revisión sistemática. Se conservan estas carencias como pendientes de §3.')]
    books['10_evaluacion'].append(md('## Margen relativo e interpretación de los contrastes\n\nEl preregistro confirma δ=3 % como diferencia relativa de interés. Para toda pareja se define θ_AB=(μ_A−μ_B)/μ_MLP: el piloto estima su dispersión relativa y TOST contrasta el mismo intervalo [−0,03;0,03]. No se declara un δ de 0,0163 o de 0,0226 MASE; esas cantidades eran conversiones de escala dependientes del periodo y no son el margen del contraste actual. El umbral operativo para un modelo avanzado es μ_nuevo<0,97·μ_MLP.\n\nLos IC90 % de Fieller incorporan covarianza entre numerador y referencia, bajo normalidad bivariada, conforme a la [formulación para muestras pareadas de NIST](https://www.itl.nist.gov/div898/software/dataplot/refman1/auxillar/ratimean.htm). La decisión tras Holm no se deduce sólo de un IC90 % individual. Con cinco semillas estos supuestos tienen verificación limitada.\n\nLas t pareadas con Holm y los contrastes temporales encuentran diferencias entre MLP, CNN1D y LSTM en los datos evaluados. Wilcoxon exacto no rechaza con cinco pares. ARX y CNN1D no presentan una diferencia temporal significativa; tampoco se acredita equivalencia mediante TOST. Detectar diferencias mayores no acredita potencia del 80 % para una mejora del 3 %, y los resultados mantienen alcance exploratorio por la selección adaptativa y la exposición al conjunto de prueba.'))

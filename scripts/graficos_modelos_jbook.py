"""Curvas y resultados de prueba en el cuerpo principal del capítulo."""


def integrar_graficos(books,md,code,note):
    c=books['09_modelos']
    inicios={int(x.source.split()[1].split('.')[1]):i for i,x in enumerate(c)
             if x.cell_type=='markdown' and x.source.startswith('## 10.')}
    prefijo=c[:inicios[5]]
    benchmark=c[inicios[5]:inicios[6]]
    prueba=c[inicios[6]:inicios[7]]
    curvas=c[inicios[7]:inicios[8]]
    resto=c[inicios[8]:]
    curvas[0]=md('## 10.5 Curvas de aprendizaje y parada temprana\n\nLas pérdidas por época se recuperan de los historiales completos de los ajustes finales. Se utiliza la semilla 1000, prefijada para las cuatro familias. La escala logarítmica permite observar la convergencia tras el descenso inicial; las funciones MAE, Huber y MSE se interpretan dentro de cada panel.')
    codigo_curvas=code("""indice_curvas=pd.read_csv(R/'indice_artefactos.csv')
fig,axes=plt.subplots(2,2,figsize=(12,7))
curvas_guardadas=[]
for ax,nombre in zip(axes.ravel(),modelos):
    seleccion=(indice_curvas.etapa=='final')&(indice_curvas.modelo==nombre)&(indice_curvas.semilla==1000)&indice_curvas.archivo.str.endswith('/historia.json')
    assert seleccion.sum()==1
    historia=pd.DataFrame(json.loads((R/indice_curvas.loc[seleccion,'archivo'].iloc[0]).read_text(encoding='utf-8')))
    corrida=runs.loc[(runs.modelo==nombre)&(runs.semilla==1000)].iloc[0]
    assert len(historia)==corrida.epocas
    assert historia.epoca.tolist()==list(range(1,len(historia)+1))
    ax.plot(historia.epoca,historia.train_loss,label='Entrenamiento')
    ax.plot(historia.epoca,historia.val_loss,label='Parada interna')
    ax.axvline(corrida.mejor_epoca,color='grey',ls='--',label='Mínimo reportado')
    ax.set(title=etiquetas[nombre],xlabel='Época',ylabel='Pérdida '+json.loads(corrida.params)['perdida'].upper(),yscale='log')
    ax.legend(fontsize=8)
    curvas_guardadas.append(historia.assign(modelo=nombre,semilla=1000))
fig.tight_layout();display(fig);plt.close(fig)
pd.concat(curvas_guardadas,ignore_index=True).to_csv(OUT/'curvas_aprendizaje_1000.csv',index=False)
""")
    curvas[1:1]=[
        codigo_curvas,
        md('MLP presenta curvas de entrenamiento y parada interna próximas después del descenso inicial. LSTM y CNN1D mantienen una separación mayor dentro de sus respectivas funciones de pérdida, compatible con una generalización interna más limitada. El mínimo reportado y la detención posterior muestran el efecto de la paciencia; no se comparan magnitudes absolutas entre pérdidas diferentes.'),
        md('### Variación de la parada entre inicializaciones\n\nLa distribución siguiente considera las 195 ejecuciones finales y complementa las curvas de la semilla prefijada.')]

    benchmark[0]=md('## 10.6 Resultados de prueba\n\nLa figura ordena los modelos por MASE sobre 2022–2025 e incluye las referencias deterministas. Las barras representan IC del 95 % entre 195 inicializaciones sobre los mismos días; la incertidumbre temporal se presenta por separado en la evaluación detallada.')
    calculo=next(x for x in benchmark if x.cell_type=='code')
    calculo.source=calculo.source.replace(
        "display(benchmark.set_index('Modelo')[['Orden','n','MASE','DE','IC95_inf','IC95_sup','MAE_kWh','RMSE_kWh','sMAPE_pct','Ajuste_s_mediana','Parámetros']].round(4))",'')
    calculo.source+="""
varianza_objetivo=float(np.var(y_ref))
r2_por_familia={n:float(1-np.mean(runs.loc[runs.modelo==n,'RMSE'].to_numpy()**2)/varianza_objetivo) for n in modelos}
r2_por_familia.update({n:v['R2'] for n,v in referencias.items()})
benchmark['R2']=benchmark.Familia.map(r2_por_familia)
benchmark.to_csv(OUT/'benchmark.csv',index=False)
benchmark.to_csv(ROOT/'book/benchmark_modelos.csv',index=False)
"""
    i=benchmark.index(calculo)+1
    benchmark.insert(i,code("""fig,ax=plt.subplots(figsize=(10,5))
ordenado=benchmark.sort_values('MASE').reset_index(drop=True)
for pos,fila in ordenado.iterrows():
    if fila.Familia in modelos:
        ax.errorbar(fila.MASE,pos,xerr=[[fila.MASE-fila.IC95_inf],[fila.IC95_sup-fila.MASE]],
                    fmt='o',capsize=4,color='#247ba0',ms=6)
    else:
        ax.scatter(fila.MASE,pos,marker='D',s=35,color='#666666')
    ax.annotate(f'{fila.MASE:.3f}',(fila.MASE,pos),xytext=(9,7),textcoords='offset points',fontsize=9)
ax.axvline(referencias['ingenuo_168h']['MASE'],color='grey',ls='--',lw=1)
ax.set(yticks=np.arange(len(ordenado)),yticklabels=ordenado.Modelo,
       xlabel='MASE de prueba (menor es mejor)',xlim=(0,ordenado.MASE.max()*1.12))
ax.invert_yaxis()
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([],[],marker='o',color='#247ba0',label='Modelos: IC 95 % entre semillas'),
                   Line2D([],[],marker='D',color='#666666',ls='',label='Referencia determinista'),
                   Line2D([],[],color='grey',ls='--',label='Ingenuo semanal')],fontsize=8,loc='upper right')
fig.tight_layout();display(fig);plt.close(fig)
"""))
    descarga='{download}'+chr(96)+'Descargar tabla completa <benchmark_modelos.csv>'+chr(96)
    benchmark.extend([
        md('### Indicadores complementarios\n\nLa figura presenta MASE e intervalos. La tabla siguiente añade dispersión entre semillas, errores en unidades originales, sMAPE, costo de ajuste y capacidad. '+descarga+'.'),
        code("""display(benchmark.set_index('Modelo')[['n','DE','MAE_kWh','RMSE_kWh','sMAPE_pct','Ajuste_s_mediana','Parámetros']].rename(columns={'DE':'DE de MASE'}).round(4))
"""),
        md('Los tiempos corresponden al entrenamiento, no a latencia de predicción. Las referencias deterministas no tienen variación de inicialización; sus intervalos temporales se desarrollan en el análisis detallado.')])

    prueba[0]=md('## 10.7 Predicciones, generalización y costo\n\nLa ventana del 1 al 7 de julio de 2023 se fijó antes de examinar trayectorias. Se muestran la demanda observada, la media de las 195 predicciones y su rango p10–p90 entre inicializaciones. La banda representa dispersión algorítmica, no un intervalo predictivo calibrado; las métricas del benchmark promedian errores individuales.')
    prueba[1:1]=[
        code("""fechas_vista=pd.to_datetime(orig_ref,unit='ns',utc=True).tz_convert('America/Bogota')
sel_vista=(fechas_vista>=pd.Timestamp('2023-07-01',tz='America/Bogota'))&(fechas_vista<pd.Timestamp('2023-07-08',tz='America/Bogota'))
tiempos_vista=pd.DatetimeIndex([f+pd.Timedelta(hours=h) for f in fechas_vista[sel_vista] for h in range(24)])
fig,axes=plt.subplots(2,2,figsize=(13,7),sharex=True,sharey=True)
for ax,nombre in zip(axes.ravel(),modelos):
    seleccion=(indice_curvas.etapa=='final')&(indice_curvas.modelo==nombre)&indice_curvas.archivo.str.endswith('/predicciones.npz')
    archivos_vista=indice_curvas.loc[seleccion].sort_values('semilla')
    assert archivos_vista.semilla.tolist()==list(range(1000,1195))
    valores=[]
    for archivo in archivos_vista.archivo:
        with np.load(R/archivo,allow_pickle=False) as z:
            assert np.array_equal(z['y'],y_ref) and np.array_equal(z['origen'],orig_ref)
            valores.append(z['yhat'][sel_vista].ravel())
    valores=np.stack(valores)
    ax.plot(tiempos_vista,y_ref[sel_vista].ravel()/1e6,color='black',lw=1,label='Observado')
    ax.plot(tiempos_vista,valores.mean(axis=0)/1e6,label='Media entre semillas')
    ax.fill_between(tiempos_vista,np.quantile(valores,.1,axis=0)/1e6,np.quantile(valores,.9,axis=0)/1e6,alpha=.2)
    ax.set(title=etiquetas[nombre],ylabel='Demanda (GWh por intervalo)')
    ax.tick_params(axis='x',rotation=30);ax.legend(fontsize=7)
fig.tight_layout();display(fig);plt.close(fig)
"""),
        md('MLP y ARX siguen mejor el perfil observado de esta semana. LSTM y CNN1D sobreestiman picos concretos y alteran la amplitud diaria, especialmente CNN1D. Las bandas muestran que variar la inicialización no corrige esas desviaciones sistemáticas; los resultados de todo el periodo permiten valorar su alcance.')]
    books['09_modelos']=prefijo+curvas+benchmark+prueba+resto

    # Las mismas curvas y la misma ventana se presentan una sola vez.
    d=books['10_evaluacion']
    a=next(i for i,x in enumerate(d) if x.cell_type=='markdown' and x.source.startswith('## Curvas de aprendizaje completas'))
    b=next(i for i,x in enumerate(d) if x.cell_type=='markdown' and x.source.startswith('## Incertidumbre temporal'))
    d[a:b]=[md('Las [curvas de aprendizaje y predicciones observadas](09_modelos.ipynb) se presentan en el capítulo principal. Las secciones siguientes desarrollan la incertidumbre temporal, los contrastes y los residuos.')]

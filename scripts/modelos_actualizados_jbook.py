"""Informe del registro consolidado, calculado desde artefactos guardados."""
from pathlib import Path
import ast,json,re,shutil,textwrap

SETUP=r'''from pathlib import Path
import json,hashlib,itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.multitest import multipletests
from IPython.display import display
ROOT=next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/'results/jbook/modelos_actualizados.json').exists())
procedencia=json.loads((ROOT/'results/jbook/modelos_actualizados.json').read_text(encoding='utf-8'))
SHA=procedencia['commit'];SNAP=ROOT/'results/repositorio'/SHA
C=SNAP/'results/consolidado';OUT=ROOT/'results/jbook/modelos_actuales';OUT.mkdir(parents=True,exist_ok=True)
registro=json.loads((C/'modelos.json').read_text(encoding='utf-8'))
modelos=['mlp','arx_lineal','cnn1d','lstm'];neuronales=['mlp','cnn1d','lstm']
etiquetas={'mlp':'MLP','arx_lineal':'ARX','cnn1d':'CNN1D','lstm':'LSTM','ingenuo_24h':'Ingenuo diario','ingenuo_168h':'Ingenuo semanal'}
plt.rcParams.update({'figure.figsize':(10,4),'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.15})
importacion=json.loads((SNAP/'IMPORTACION.json').read_text(encoding='utf-8'))
huellas={x['archivo']:x for x in importacion['archivos']}
indice=pd.read_csv(C/'indice_artefactos_v3_v4.csv')
assert len(indice)==2304 and not indice.archivo.duplicated().any()
for fila in indice.itertuples():
    p=SNAP/fila.archivo
    assert p.stat().st_size==fila.bytes
    assert hashlib.sha256(p.read_bytes()).hexdigest()==fila.sha256==huellas[fila.archivo]['sha256']
runs=[];pilotos=[];confirmaciones=[];filtros=[];pred={};carpetas={};escaladores={};historias={};y=None;origen=None
for nombre in modelos:
    r=registro[nombre];carpeta=SNAP/'results'/r['ejecucion']
    seleccion=r['configuracion']['id']
    pp=pd.read_csv(carpeta/'piloto.csv');pp=pp.loc[(pp.modelo==nombre)&(pp.configuracion==seleccion)]
    assert len(pp)==20 and not pp.duplicated(['semilla','fold']).any()
    pilotos.append(pp)
    pila=[]
    for a in sorted(r['artefactos'],key=lambda a:a['semilla']):
        met=json.loads((SNAP/a['metricas']).read_text(encoding='utf-8'))
        esc=json.loads((SNAP/a['escaladores']).read_text(encoding='utf-8'))
        historia=pd.DataFrame(json.loads((SNAP/a['historial']).read_text(encoding='utf-8')))
        assert met['semilla']==a['semilla'] and met['params']==r['configuracion']['params']
        assert met['configuracion']==seleccion and len(historia)==met['epocas']
        assert historia.epoca.tolist()==list(range(1,len(historia)+1))
        assert met['mejor_epoca']==int(historia.val_loss.idxmin()+1)
        assert pd.Timestamp(esc['fin_ajuste'])<pd.Timestamp('2022-01-01',tz='America/Bogota')
        with np.load(SNAP/a['predicciones'],allow_pickle=False) as z:
            if y is None:y=z['y'].astype(float);origen=z['origen'].copy()
            assert np.array_equal(y,z['y']) and np.array_equal(origen,z['origen'])
            assert z['yhat'].shape==y.shape and np.isfinite(z['yhat']).all()
            pila.append(z['yhat'].astype(float))
        runs.append(met);historias[(nombre,a['semilla'])]=historia
        carpetas[(nombre,a['semilla'])]=SNAP/a['pesos']
        escaladores[(nombre,a['semilla'])]=esc
    pred[nombre]=np.stack(pila)
piloto=pd.concat(pilotos,ignore_index=True);runs=pd.DataFrame(runs)
assert len(runs)==20 and not runs.duplicated(['modelo','semilla']).any()
assert all(set(d.semilla)==set(range(1000,1005)) for _,d in runs.groupby('modelo'))
fechas=pd.to_datetime(origen,unit='ns',utc=True).tz_convert('America/Bogota')
assert fechas.equals(pd.date_range('2022-01-01','2025-12-31',freq='D',tz='America/Bogota'))
assert y.shape==(1461,24) and np.isfinite(y).all()
escala=float(escaladores[('mlp',1000)]['escala_mase'])
loss_h={};loss_d={};residuos={};calculadas=[]
for nombre in modelos:
    errores=pred[nombre]-y[None,:,:]
    loss_h[nombre]=np.abs(errores).mean(axis=0)/escala
    loss_d[nombre]=loss_h[nombre].mean(axis=1)
    residuos[nombre]=errores[0]
    for j,semilla in enumerate(range(1000,1005)):
        e=errores[j];mae=np.abs(e).mean();rmse=np.sqrt(np.mean(e**2))
        m=dict(modelo=nombre,semilla=semilla,MASE=mae/escala,MAE=mae,RMSE=rmse,R2=1-np.mean(e**2)/np.var(y),sesgo=e.mean(),sMAPE=(200*np.abs(e)/(np.abs(pred[nombre][j])+np.abs(y))).mean())
        original=runs.loc[(runs.modelo==nombre)&(runs.semilla==semilla)].iloc[0]
        assert np.allclose([m[k] for k in ['MASE','MAE','RMSE','sMAPE']],[original[k] for k in ['MASE','MAE','RMSE','sMAPE']],rtol=2e-6)
        calculadas.append(m)
calculadas=pd.DataFrame(calculadas)
seleccion_publicada=pd.read_csv(C/'modelos_seleccionados.csv').set_index('modelo')
assert np.allclose(calculadas.groupby('modelo').MASE.mean().sort_index(),seleccion_publicada.MASE_prueba.sort_index())
referencias=json.loads((C/'referencias_test.json').read_text(encoding='utf-8'))
filas=[]
for nombre in modelos:
    d=calculadas.loc[calculadas.modelo==nombre];media=d.MASE.mean();sd=d.MASE.std(ddof=1);ancho=stats.t.ppf(.975,len(d)-1)*sd/np.sqrt(len(d))
    filas.append(dict(familia=nombre,Modelo=etiquetas[nombre],n=len(d),MASE=media,DE=sd,IC95_inf=media-ancho,IC95_sup=media+ancho,MAE_kWh=d.MAE.mean(),RMSE_kWh=d.RMSE.mean(),sMAPE_pct=d.sMAPE.mean(),R2=d.R2.mean(),sesgo_kWh=d.sesgo.mean(),parametros=int(runs.loc[runs.modelo==nombre,'n_parametros'].iloc[0]),ajuste_s=float(runs.loc[runs.modelo==nombre,'segundos'].median())))
for nombre,m in referencias.items():
    filas.append(dict(familia=nombre,Modelo=etiquetas[nombre],n=1,MASE=m['MASE'],DE=np.nan,IC95_inf=np.nan,IC95_sup=np.nan,MAE_kWh=m['MAE'],RMSE_kWh=m['RMSE'],sMAPE_pct=m['sMAPE'],R2=1-m['RMSE']**2/np.var(y),sesgo_kWh=np.nan,parametros=0,ajuste_s=np.nan))
benchmark=pd.DataFrame(filas).sort_values('MASE').reset_index(drop=True);benchmark.insert(0,'rango',np.arange(1,len(benchmark)+1))
benchmark.to_csv(OUT/'benchmark.csv',index=False);benchmark.to_csv(ROOT/'book/benchmark_modelos.csv',index=False)
calculadas.to_csv(OUT/'metricas_desde_predicciones.csv',index=False)
for fase,destino in [('filtro',filtros),('confirmacion',confirmaciones)]:
    for ruta in sorted(set(r['ejecucion'] for r in registro.values())):
        destino.append(pd.read_csv(SNAP/'results'/ruta/(fase+'.csv')))
filtro=pd.concat(filtros,ignore_index=True);confirmacion=pd.concat(confirmaciones,ignore_index=True)
assert len(filtro)==90 and len(confirmacion)==144
mejor=benchmark.loc[benchmark.familia=='mlp'].iloc[0]
reduccion_semanal=100*(1-mejor.MASE/referencias['ingenuo_168h']['MASE'])
mae_mlp=float(mejor.MAE_kWh);mase_mlp=float(mejor.MASE)
'''


def integrar_actualizacion(books,md,code,note,root):
    sha=json.loads((root/'results/jbook/modelos_actualizados.json').read_text())['commit']
    snap=root/'results/repositorio'/sha
    registry=json.loads((snap/'results/consolidado/modelos.json').read_text(encoding='utf-8'))
    def table(headers,rows):
        clean=lambda x:str(x).replace('|','/').replace('\n',' ')
        return '| '+' | '.join(headers)+' |\n|'+'|'.join(['---']*len(headers))+'|\n'+'\n'.join('| '+' | '.join(clean(x) for x in r)+' |' for r in rows)
    configuraciones=[]
    labels={'mlp':'MLP','lstm':'LSTM','cnn1d':'CNN1D','arx_lineal':'ARX'}
    for m in ['mlp','lstm','cnn1d','arx_lineal']:
        p=registry[m]['configuracion']['params']
        arquitectura={'mlp':f"{p.get('capas')} capas de {p.get('unidades')} unidades",'lstm':f"{p.get('capas')} capa de {p.get('unidades')} unidades",'cnn1d':f"{p.get('capas')} convolución, {p.get('filtros')} filtros, kernel {p.get('kernel')}",'arx_lineal':f"{p.get('penalizacion')}, {p.get('lags')} rezagos"}[m]
        configuraciones.append([labels[m],arquitectura,f"{p['lr']:.5g}",f"{p.get('dropout',0):.5g}",f"{p.get('weight_decay',p.get('lambda',0)):.5g}",p['batch_size'],p['perdida']])
    c=[md('# 10. Modelos y benchmark de demanda'),
       md('Se comparan MLP, LSTM y CNN1D como familias de aprendizaje profundo, con ARX regularizado y referencias estacionales como controles. Cada configuración se evalúa con cinco inicializaciones sobre los mismos 1.461 orígenes diarios de 2022–2025. La selección utiliza el MASE de validación; el análisis presenta resultados descriptivos y su variabilidad por inicialización y por fecha.'),
       code(SETUP),
       note('Las predicciones reconstruyen las métricas registradas para las 20 realizaciones seleccionadas. La escala fija de MASE es {escala:,.2f} kWh, calculada con diferencias semanales de 2000–2017. Se verifican las huellas de 2.304 artefactos, incluidas las alternativas de búsqueda; las 20 realizaciones del benchmark corresponden a las cuatro configuraciones seleccionadas.'),
       md('## 10.1. Entradas, escalado y partición\n\nEl origen diario de las 00:00 genera 24 pronósticos horarios. La historia contiene 168 horas de demanda disponible y 23 variables de calendario. MLP utiliza demanda histórica y calendario futuro aplanados; ARX conserva 48 rezagos; LSTM y CNN1D concatenan calendario histórico a la demanda y calendario futuro a la representación temporal. Las condiciones meteorológicas, hidrología y noticias caracterizadas en el EDA no forman parte de estas entradas.\n\nLas etiquetas y la demanda histórica se normalizan por el nivel medio conocido en la ventana. En las redes secuenciales, la demanda normalizada se centra en uno y se divide por su desviación estimada en entrenamiento. Los escaladores del calendario y las representaciones tabulares se ajustan con el tramo de optimización de cada pliegue; la parada interna y el año externo quedan separados. Las salidas se restituyen a kWh mediante el nivel del origen.\n\nLa validación expansiva comprende 2018–2021. El último 10 % del historial previo se reserva para parada temprana y se comprueba la separación de las 24 etiquetas entre tramos. El ajuste final utiliza el historial anterior a 2022 con la misma reserva interna. MASE conserva un denominador fijo de referencia para hacer comparables los errores entre pliegues; ese denominador no se utiliza para optimizar las redes.'),
       md('## 10.2. Arquitecturas y configuraciones\n\n'+table(['Modelo','Representación seleccionada','Tasa de aprendizaje','Dropout','Regularización','Lote','Pérdida'],configuraciones)+'''\n\nLa MLP aplica dos transformaciones densas ReLU y una salida directa de 24 valores. La LSTM utiliza el último estado oculto; la CNN1D aplica convolución, ReLU, Dropout1d, promedio local de factor dos y aplanamiento. Las dos redes secuenciales concatenan el calendario futuro antes de la salida lineal y aprenden una desviación alrededor de uno en la escala del nivel. ARX penaliza los coeficientes mediante lasso.

Para MLP, con z como vector de entrada, la predicción es g(z)=W₃ ReLU(W₂ ReLU(W₁z+b₁)+b₂)+b₃. Para LSTM, hₜ,cₜ=LSTM([xₜ,kₜ],hₜ₋₁,cₜ₋₁), y la salida normalizada es 1+W[hₗ,k_f]+b. En CNN1D, z=flatten(pool₂(ReLU(Convₖ([x,k_p])))) y la salida es 1+W[z,k_f]+b. El Dropout se activa durante el ajuste y se desactiva al evaluar.

Con L=168, H=24, K=23 y f=24, el costo dominante por ejemplo es O(du+u²+uH) para MLP, O(Lu(f+u)+(u+HK)H) para LSTM y O(Lkfc+(Lc/2+HK)H) para CNN1D. Los parámetros se verifican frente a los metadatos guardados; capacidad, tiempo de ajuste y latencia de inferencia son magnitudes distintas.'''),
       code("L,H,K,f=168,24,23,24;d=L+H*K;u=128\nconteos={'mlp':(d+1)*u+(u+1)*u+(u+1)*H,'lstm':4*48*(f+48+2)+(48+H*K+1)*H,'cnn1d':(5*f+1)*16+((L//2)*16+H*K+1)*H,'arx_lineal':(48+H*K+1)*H}\nassert all(int(runs.loc[runs.modelo==m,'n_parametros'].iloc[0])==n for m,n in conteos.items())\nfig,axes=plt.subplots(1,2,figsize=(11,3.7))\naxes[0].barh([etiquetas[m] for m in modelos],[conteos[m] for m in modelos],color='#247ba0');axes[0].set(xlabel='Parámetros entrenables',xscale='log')\naxes[1].barh([etiquetas[m] for m in modelos],[runs.loc[runs.modelo==m,'segundos'].median() for m in modelos],color='#2a9d8f');axes[1].set_xlabel('Tiempo mediano por ajuste final (s)')\nfig.tight_layout();display(fig);plt.close(fig)"),
       md('La MLP concentra la mayor cantidad de parámetros y la LSTM requiere más tiempo de ajuste en el dispositivo registrado. La CNN1D posee una cabeza densa sobre posiciones temporales agrupadas; menos filtros no implica necesariamente menos parámetros que una LSTM con último estado. Estos costos describen el ajuste completo, incluida la evaluación registrada, y no miden latencia de inferencia.'),
       md('## 10.3. Búsqueda, confirmación y piloto\n\nSe examinaron 20 candidatos de MLP, 20 de ARX, 24 de LSTM y 26 de CNN1D. El filtro empleó el año 2019, semilla 42 y un máximo de 30 épocas; los finalistas se compararon en 2018–2021 con semillas 43 y 44 y un máximo de 60 épocas. La configuración de cada familia se eligió por el MASE medio de confirmación. El piloto conserva cinco semillas, 100–104, por año externo; prueba conserva 1000–1004. El número de candidatos es desigual y debe considerarse al comparar familias.\n\nEl optimizador es AdamW, con recorte de gradiente a uno y precisión completa. La parada exige al menos diez épocas, paciencia de seis y mejora mayor que max(10⁻⁶, 0,002 × pérdida de referencia). La paciencia y la restauración cumplen funciones distintas: se guarda el mínimo absoluto de pérdida, aunque sólo una mejora suficiente reinicia la paciencia. El planificador plateau reduce la tasa a la mitad tras dos épocas sin mejora. Huber utiliza δ=0,05 en la escala normalizada; este parámetro de pérdida es distinto del margen relativo de mejora del 3 %.'),
       code("fig,axes=plt.subplots(1,2,figsize=(12,4))\nfor m in modelos:\n    p=piloto.loc[piloto.modelo==m];grupo=p.groupby('fold').MASE\n    axes[0].errorbar(grupo.mean().index,grupo.mean(),yerr=grupo.std(),marker='o',capsize=3,label=etiquetas[m])\n    q=p.groupby('fold')[['MASE','MASE_demanda_permutada']].mean()\n    axes[1].plot(q.index,q.MASE_demanda_permutada/q.MASE,marker='o',label=etiquetas[m])\naxes[0].set(xticks=[2018,2019,2020,2021],xlabel='Año de validación',ylabel='MASE medio ± DE de cinco semillas')\naxes[1].axhline(1,color='grey',ls='--');axes[1].set(xticks=[2018,2019,2020,2021],xlabel='Año de validación',ylabel='MASE con demanda permutada / MASE original')\nfor ax in axes:ax.legend(fontsize=8)\nfig.tight_layout();display(fig);plt.close(fig)\npeor_piloto=etiquetas[piloto.groupby('modelo').MASE.mean().idxmax()]\nrazon_permutacion=piloto.groupby('modelo')[['MASE','MASE_demanda_permutada']].mean().eval('MASE_demanda_permutada/MASE')\n"),
       note('La validación muestra diferencias entre años y sitúa a {peor_piloto} como la familia de mayor MASE medio del piloto. La permutación intercambia la trayectoria de demanda entre orígenes y conserva el nivel y calendario del ejemplo: el aumento de error mide sensibilidad condicional a esa trayectoria, no el aporte total de la demanda ni una ablación causal.'),
       md('## 10.4. Curvas de aprendizaje\n\nSe presentan pérdidas de ajuste y parada interna para la semilla 1000. Las funciones MAE y Huber se interpretan dentro de cada panel; su magnitud no constituye un ranking entre familias. El Dropout activo durante el ajuste también influye en la separación visual entre curvas.'),
       code("fig,axes=plt.subplots(2,2,figsize=(12,7));curvas=[]\nfor ax,m in zip(axes.ravel(),modelos):\n    h=historias[(m,1000)];r=runs.loc[(runs.modelo==m)&(runs.semilla==1000)].iloc[0]\n    ax.plot(h.epoca,h.train_loss,label='Entrenamiento');ax.plot(h.epoca,h.val_loss,label='Parada interna');ax.axvline(r.mejor_epoca,color='grey',ls='--',label='Mínimo restaurado')\n    ax.set(title=etiquetas[m],xlabel='Época',ylabel='Pérdida '+r.params['perdida'].upper(),yscale='log');ax.legend(fontsize=8)\n    curvas.append(h.assign(modelo=m,semilla=1000))\nfig.tight_layout();display(fig);plt.close(fig)\npd.concat(curvas).to_csv(OUT/'curvas_aprendizaje_1000.csv',index=False)\nepocas_min=int(runs.epocas.min());epocas_max=int(runs.epocas.max())"),
       note('El descenso más pronunciado se concentra en las primeras épocas y después aparecen mejoras más pequeñas. Las realizaciones finales se detienen entre {epocas_min} y {epocas_max} épocas. La línea vertical identifica el mínimo restaurado, que puede preceder a la última época. La separación entre pérdidas no equivale por sí sola a una medida de sobreajuste; el error de evaluación se calcula con Dropout desactivado.'),
       md('## 10.5. Benchmark de prueba\n\nEl ranking ordena el MASE medio de cinco realizaciones, no la mejor semilla. Los intervalos t del 95 % describen variación de inicialización condicionada a las fechas evaluadas; no representan incertidumbre temporal. Los dos ingenuos utilizan la misma disponibilidad histórica que los modelos.'),
       code("fig,ax=plt.subplots(figsize=(10,4.8))\nfor i,r in benchmark.iterrows():\n    if r.familia in modelos:ax.errorbar(r.MASE,i,xerr=[[r.MASE-r.IC95_inf],[r.IC95_sup-r.MASE]],fmt='o',capsize=4,color='#247ba0')\n    else:ax.scatter(r.MASE,i,marker='D',color='grey')\n    ax.annotate(f'{r.MASE:.3f}',(r.MASE,i),xytext=(8,7),textcoords='offset points')\nax.axvline(referencias['ingenuo_168h']['MASE'],color='grey',ls='--');ax.set(yticks=range(len(benchmark)),yticklabels=benchmark.Modelo,xlabel='MASE de prueba; menor es mejor');ax.invert_yaxis()\nfig.tight_layout();display(fig);plt.close(fig)"),
       note('La MLP presenta el menor error medio: MASE {mase_mlp:.3f}, equivalente a MAE {mae_mlp:,.0f} kWh y una reducción del {reduccion_semanal:.1f} % frente al ingenuo semanal. ARX y CNN1D quedan próximos; LSTM presenta mayor error. Las cuatro familias mejoran las dos referencias estacionales en el agregado. La escala MASE fija explica que el ingenuo semanal de prueba no tenga necesariamente valor uno.'),
       md('La tabla añade indicadores complementarios a la figura. La [tabla completa del benchmark](benchmark_modelos.csv) conserva ranking, MASE e intervalos junto con las demás métricas.'),
       code("display(benchmark.set_index('Modelo')[['DE','MAE_kWh','RMSE_kWh','sMAPE_pct','R2','sesgo_kWh']].rename(columns={'DE':'DE de MASE'}).round(4))"),
       md('RMSE supera MAE al dar mayor peso a errores grandes; R² compara el error cuadrático con la dispersión del objetivo en todo el periodo. El sesgo firmado permite distinguir tendencia a sobreestimar o subestimar. La DE resume diferencias entre inicializaciones y no la variación diaria.'),
       md('## 10.6. Validación y generalización\n\nSe comparan el MASE de confirmación, el piloto de cinco semillas y el periodo de prueba con el mismo denominador. La confirmación participa en selección; el piloto describe estabilidad de la configuración elegida. Los periodos tienen años y regímenes diferentes.'),
       code("generalizacion=pd.DataFrame([{'modelo':m,'Confirmación':registro[m]['MASE_cv'],'Piloto':piloto.loc[piloto.modelo==m,'MASE'].mean(),'Prueba':calculadas.loc[calculadas.modelo==m,'MASE'].mean()} for m in modelos]).set_index('modelo')\ngeneralizacion.index=generalizacion.index.map(etiquetas);fig,axes=plt.subplots(1,2,figsize=(12,4));generalizacion.plot.bar(ax=axes[0],rot=0);axes[0].set(ylabel='MASE',xlabel='')\nfor i,m in enumerate(modelos):\n    d=runs.loc[runs.modelo==m];axes[1].scatter(np.full(len(d),i)-.08,d.MAE_train_normalizado_eval,label='Ajuste' if i==0 else None,color='#247ba0');axes[1].scatter(np.full(len(d),i)+.08,d.MAE_es_normalizado_eval,label='Parada interna' if i==0 else None,color='#d18a2d')\naxes[1].set(xticks=range(4),xticklabels=[etiquetas[m] for m in modelos],ylabel='MAE normalizado; modo evaluación');axes[1].legend()\nfig.tight_layout();display(fig);plt.close(fig)\nbrecha_mlp=100*(generalizacion.loc['MLP','Prueba']/generalizacion.loc['MLP','Piloto']-1)"),
       note('El MASE de prueba de MLP supera al piloto un {brecha_mlp:.1f} %. El aumento es compatible con una generalización más difícil en un periodo posterior y no se interpreta como un efecto causal de una variable aislada. El panel derecho confirma mayor error en la reserva interna que en ajuste con Dropout desactivado; evita atribuir esa diferencia a modos de evaluación distintos.'),
       md('## 10.7. Trayectorias observadas y predichas\n\nLa ventana del 1 al 7 de julio de 2023 ilustra el perfil horario. Se muestra la media de cinco pronósticos y el rango p10–p90 entre inicializaciones. La banda describe dispersión algorítmica; las métricas del benchmark promedian errores de las cinco realizaciones individuales.'),
       code("sel=(fechas>=pd.Timestamp('2023-07-01',tz='America/Bogota'))&(fechas<pd.Timestamp('2023-07-08',tz='America/Bogota'));tiempo=pd.DatetimeIndex([d+pd.Timedelta(hours=h) for d in fechas[sel] for h in range(24)])\nfig,axes=plt.subplots(2,2,figsize=(13,7),sharex=True,sharey=True)\nfor ax,m in zip(axes.ravel(),modelos):\n    v=pred[m][:,sel,:].reshape(5,-1)/1e6;ax.plot(tiempo,y[sel].ravel()/1e6,color='black',lw=1,label='Observado');ax.plot(tiempo,v.mean(axis=0),label='Media de cinco semillas');ax.fill_between(tiempo,np.quantile(v,.1,axis=0),np.quantile(v,.9,axis=0),alpha=.2)\n    ax.set(title=etiquetas[m],ylabel='Demanda (GWh por intervalo)');ax.tick_params(axis='x',rotation=25);ax.legend(fontsize=7)\nfig.tight_layout();display(fig);plt.close(fig)\nventana_mae={m:float(np.abs(pred[m][:,sel,:]-y[None,sel,:]).mean()) for m in modelos};mejor_ventana=etiquetas[min(ventana_mae,key=ventana_mae.get)]"),
       note('{mejor_ventana} registra el menor MAE en la ventana ilustrada. Las discrepancias en picos, valles y amplitud muestran errores que la media global comprime. Una banda estrecha indica acuerdo entre inicializaciones, sin garantizar proximidad al observado ni cobertura predictiva calibrada; la valoración global utiliza todo 2022–2025.'),
       md('## 10.8. Alcance e implicaciones\n\nMLP establece el baseline empírico de estas configuraciones. La proximidad entre ARX y CNN1D aconseja examinar estabilidad temporal y costo antes de atribuir una ventaja a la no linealidad. El horizonte y los segmentos se desarrollan en el [análisis temporal](10_evaluacion.ipynb).\n\nEl margen práctico conserva una reducción relativa de MASE del 3 %. Cinco semillas permiten describir dispersión, pero no acreditan potencia del 80 % para detectar ese margen. Los candidatos y el presupuesto son desiguales entre familias. Los contrastes y remuestreos se interpretan como análisis exploratorios; no se afirma superioridad confirmatoria ni equivalencia. Las revisiones de las fuentes y los supuestos de disponibilidad limitan la reproducción histórica en tiempo real.\n\nLos checkpoints, escaladores, identidades, historiales y predicciones se vinculan por SHA-256. El [pipeline de entrenamiento y evaluación](10_pipeline.md) incluye el preprocesamiento por pliegue, las arquitecturas y el bucle de optimización. La ejecución registrada utiliza PyTorch 2.9.1 con ROCm 6.4 y GPU AMD con aproximadamente 12 GB; la latencia de inferencia no fue medida.')]
    books['09_modelos']=c
    e=[md('# Evaluación temporal y diagnósticos'),
       md('El análisis utiliza los mismos 1.461 días y las cinco realizaciones de cada configuración. La unidad temporal es el origen diario con sus 24 errores. Se distingue variación entre semillas, entre fechas y entre segmentos; ninguna se interpreta como una nueva muestra independiente de mercados.'),code(SETUP),
       md('## Errores por horizonte y segmentos\n\nLos errores absolutos se promedian primero entre inicializaciones y después entre días del segmento. Se conservan las 24 horas del pronóstico y se distinguen año, tipo de día y nivel realizado de demanda.'),
       code("panel=pd.read_parquet(ROOT/'data/processed/panel_horario.parquet',columns=['cal_es_festivo'])\nfestivo=panel.loc[fechas,'cal_es_festivo'].to_numpy().astype(bool)\ntipo=np.where((fechas.day_of_week==6)|festivo,'Domingo/festivo',np.where(fechas.day_of_week==5,'Sábado','Lun–vie'))\nsegmentos={'Año':fechas.year.astype(str),'Tipo de día':tipo,'Nivel de demanda':pd.qcut(y.mean(axis=1),4,labels=['Q1','Q2','Q3','Q4']).astype(str)}\nfilas=[];horas=[];fig,axes=plt.subplots(2,2,figsize=(13,8))\nfor m in modelos:\n    perfil=loss_h[m].mean(axis=0);axes[0,0].plot(range(24),perfil*escala/1e3,label=etiquetas[m])\n    horas.extend({'modelo':m,'hora':h,'MASE':v,'MAE_kWh':v*escala} for h,v in enumerate(perfil))\nfor ax,(nombre,grupos) in zip([axes[0,1],axes[1,0],axes[1,1]],segmentos.items()):\n    orden=sorted(set(grupos)) if nombre!='Tipo de día' else ['Lun–vie','Sábado','Domingo/festivo']\n    for m in modelos:\n        valores=[loss_d[m][np.asarray(grupos)==g].mean() for g in orden];ax.plot(orden,valores,marker='o',label=etiquetas[m])\n        filas.extend({'modelo':m,'segmento':nombre,'grupo':g,'n_dias':int((np.asarray(grupos)==g).sum()),'MASE':v} for g,v in zip(orden,valores))\n    ax.set(xlabel=nombre,ylabel='MASE')\naxes[0,0].set(xlabel='Hora objetivo',ylabel='MAE (miles de kWh)');axes[0,0].legend(fontsize=8)\nfig.tight_layout();display(fig);plt.close(fig)\npd.DataFrame(filas).to_csv(OUT/'errores_segmentos.csv',index=False);pd.DataFrame(horas).to_csv(OUT/'errores_24_horizontes.csv',index=False)\npico_hora=int(np.argmax(loss_h['mlp'].mean(axis=0)));anos_mlp=pd.Series(loss_d['mlp'],index=fechas).groupby(fechas.year).mean();peor_ano=int(anos_mlp.idxmax());mejor_ano=int(anos_mlp.idxmin())\n"),
       note('Para MLP, el mayor error horario agregado se presenta a las {pico_hora:02d}:00; {peor_ano} concentra el mayor MASE anual y {mejor_ano} el menor. La distancia entre curvas depende del horario y del régimen. Los cuartiles se definen con la demanda realizada y son segmentos descriptivos posteriores, no entradas ni reglas de selección del modelo.'),
       md('## Incertidumbre temporal y sensibilidad al bloque\n\nEl bootstrap circular estratifica por año y remuestrea bloques de días con sus 24 horas. Los modelos comparten índices de remuestreo. Se utilizan 10.000 réplicas y bloques de 14 días, con sensibilidad de siete y 28. Antes de remuestrear se promedia la pérdida de las cinco semillas; el estimando es su error esperado, no el error de una predicción ensemble.'),
       code("perdidas=np.stack([loss_d[m] for m in modelos],axis=1)\ndef bootstrap_temporal(valores,bloque,B=10000,semilla=11038):\n    rng=np.random.default_rng(semilla);n=len(valores);salida=np.zeros((B,valores.shape[1]))\n    for anio in sorted(set(fechas.year)):\n        v=valores[fechas.year==anio];a=len(v);cantidad=int(np.ceil(a/bloque))\n        for inicio in range(0,B,250):\n            b=min(250,B-inicio);starts=rng.integers(0,a,size=(b,cantidad));ix=(starts[:,:,None]+np.arange(bloque))%a;ix=ix.reshape(b,-1)[:,:a]\n            salida[inicio:inicio+b]+=v[ix].sum(axis=1)/n\n    return salida\nresumen=[];boot=None\nfor bloque in [7,14,28]:\n    b=bootstrap_temporal(perdidas,bloque)\n    if bloque==14:boot=b\n    for j,m in enumerate(modelos):\n        lo,hi=np.quantile(b[:,j],[.025,.975]);resumen.append({'modelo':m,'bloque':bloque,'MASE':perdidas[:,j].mean(),'IC95_inf':lo,'IC95_sup':hi})\nintervalos=pd.DataFrame(resumen);intervalos.to_csv(OUT/'intervalos_temporales.csv',index=False)\nfig,axes=plt.subplots(1,2,figsize=(12,4))\nfor i,m in enumerate(modelos):\n    q=intervalos.loc[intervalos.modelo==m];axes[0].errorbar(q.bloque+.3*i,q.MASE,yerr=[q.MASE-q.IC95_inf,q.IC95_sup-q.MASE],marker='o',capsize=3,label=etiquetas[m])\nfor i,m in enumerate(modelos[1:]):\n    diferencia=boot[:,modelos.index(m)]-boot[:,0];lo,hi=np.quantile(diferencia,[.025,.975]);centro=perdidas[:,modelos.index(m)].mean()-perdidas[:,0].mean();axes[1].errorbar(centro,i,xerr=[[centro-lo],[hi-centro]],fmt='o',capsize=4)\naxes[0].set(xticks=[7,14,28],xlabel='Longitud del bloque (días)',ylabel='MASE e intervalo temporal 95 %');axes[0].legend(fontsize=8)\naxes[1].axvline(0,color='grey',ls='--');axes[1].set(yticks=range(3),yticklabels=[etiquetas[m]+' − MLP' for m in modelos[1:]],xlabel='Diferencia de MASE; bloque 14 días')\nfig.tight_layout();display(fig);plt.close(fig)\n"),
       md('Los intervalos temporales incorporan días difíciles compartidos por los modelos y difieren de los intervalos entre semillas. Las diferencias positivas indican mayor pérdida que MLP bajo el remuestreo utilizado. La sensibilidad al tamaño de bloque permite valorar cuánto cambia la incertidumbre al conservar más dependencia; estos intervalos exploratorios no demuestran superioridad confirmatoria ni equivalencia.'),
       md('## Contrastes de pérdida por horizonte\n\nDiebold–Mariano utiliza diferencias diarias de pérdida absoluta para cada hora, con covarianza HAC Bartlett de 14 rezagos y corrección HLN con h=1 origen diario no solapado. Holm se aplica por separado a los 72 contrastes neuronales y los 72 contrastes contra ARX. Los valores p se conservan como diagnósticos exploratorios.'),
       code("def dm_hln(d,rezagos=14):\n    d=np.asarray(d,dtype=float);n=len(d);z=d-d.mean();long_run=np.dot(z,z)/n\n    for k in range(1,rezagos+1):long_run+=2*(1-k/(rezagos+1))*np.dot(z[k:],z[:-k])/n\n    estadistico=d.mean()/np.sqrt(max(long_run,1e-20)/n)*np.sqrt((n-1)/n)\n    return estadistico,2*stats.t.sf(abs(estadistico),n-1)\ncontrastes=[]\nfor familia,pares in [('Redes',list(itertools.combinations(neuronales,2))),('Redes frente a ARX',[(m,'arx_lineal') for m in neuronales])]:\n    temp=[]\n    for a,b in pares:\n        for h in range(24):\n            d=loss_h[a][:,h]-loss_h[b][:,h];t,p=dm_hln(d);temp.append(dict(familia=familia,a=a,b=b,hora=h,diferencia_MASE=d.mean(),DM_HLN=t,p=p))\n    corregidos=multipletests([r['p'] for r in temp],method='holm')[1]\n    for r,p in zip(temp,corregidos):r['p_Holm']=p\n    contrastes.extend(temp)\ndm=pd.DataFrame(contrastes);dm.to_csv(OUT/'DM_24_horizontes.csv',index=False)\nfig,ax=plt.subplots(figsize=(10,4))\nfor a,b in itertools.combinations(neuronales,2):\n    q=dm.loc[(dm.a==a)&(dm.b==b)];ax.plot(q.hora,q.diferencia_MASE,label=etiquetas[a]+' − '+etiquetas[b])\nax.axhline(0,color='grey',ls='--');ax.set(xlabel='Hora objetivo',ylabel='Diferencia de MASE');ax.legend();fig.tight_layout();display(fig);plt.close(fig)"),
       md('Las curvas permiten identificar horas en las que cambia la distancia de error entre redes; por debajo de cero la primera familia presenta menor pérdida. El promedio global puede ocultar diferencias localizadas. La tabla descargable de contrastes conserva estadísticos, valores p y corrección de multiplicidad, sin convertir el análisis exploratorio en una afirmación confirmatoria.'),
       md('## Rangos y variabilidad entre inicializaciones\n\nLos rangos se calculan dentro de cada semilla. La descomposición separa diferencias descriptivas entre fechas e inicializaciones; no supone que las semillas sean mercados o periodos independientes.'),
       code("matriz=calculadas.pivot(index='semilla',columns='modelo',values='MASE').reindex(columns=neuronales);rangos=matriz.rank(axis=1).mean()\ncomponentes=[]\nfor m in modelos:\n    valores=np.abs(pred[m]-y[None,:,:]).mean(axis=2)/escala;media=valores.mean();por_semilla=np.mean((valores.mean(axis=1)-media)**2);por_dia=np.mean((valores.mean(axis=0)-media)**2);interaccion=np.mean((valores-valores.mean(axis=1)[:,None]-valores.mean(axis=0)[None,:]+media)**2)\n    componentes.append({'modelo':m,'Inicialización':por_semilla,'Fecha':por_dia,'Interacción':interaccion})\nfig,axes=plt.subplots(1,2,figsize=(12,4));rangos.rename(index=etiquetas).plot.bar(ax=axes[0],rot=0,color='#247ba0');axes[0].set_ylabel('Rango medio; uno es mejor')\ncomp=pd.DataFrame(componentes).set_index('modelo');comp.index=comp.index.map(etiquetas);comp.div(comp.sum(axis=1),axis=0).plot.bar(stacked=True,ax=axes[1],rot=0);axes[1].set(ylabel='Fracción de variación descriptiva',xlabel='');axes[1].legend(fontsize=8)\nfig.tight_layout();display(fig);plt.close(fig)\npd.DataFrame(componentes).to_csv(OUT/'componentes_variacion.csv',index=False);rangos.to_csv(OUT/'rangos_semillas.csv')"),
       md('El rango resume consistencia ordinal entre las cinco inicializaciones y descarta la magnitud de las diferencias. La componente de fecha muestra variación común a los pronósticos de un mismo día; la interacción recoge sensibilidad de cada inicialización a días concretos. Estas componentes describen los errores observados y no equivalen a efectos causales o a una estimación de potencia.'),
       md('## Residuos y estructura remanente\n\nSe utiliza la semilla 1000 y se promedian residuos horarios dentro de cada día. Ljung–Box examina autocorrelación; ARCH, variación condicional; BDS, dependencia incompatible con una secuencia i.i.d. BDS no aísla no linealidad cuando existe memoria.'),
       code("from statsmodels.tsa.stattools import acf,bds\nfrom statsmodels.stats.diagnostic import acorr_ljungbox,acorr_lm\nfig,axes=plt.subplots(2,2,figsize=(12,7));diagnosticos=[]\nfor ax,m in zip(axes.ravel(),modelos):\n    r=residuos[m].mean(axis=1)/1e3;ax.plot(range(29),acf(r,nlags=28,fft=True));ax.axhline(1.96/np.sqrt(len(r)),color='grey',ls='--');ax.axhline(-1.96/np.sqrt(len(r)),color='grey',ls='--');ax.set(title=etiquetas[m],xlabel='Rezago diario',ylabel='ACF del residuo medio')\n    lb=acorr_ljungbox(r,lags=[7,14,28],return_df=True)\n    for k,row in lb.iterrows():diagnosticos.append({'modelo':m,'prueba':'Ljung–Box','rezago':int(k),'estadistico':row.lb_stat,'p':row.lb_pvalue})\n    a=acorr_lm(r**2,nlags=7,ddof=0,result_object=False);diagnosticos.append({'modelo':m,'prueba':'ARCH','rezago':7,'estadistico':a[0],'p':a[1]})\n    z,p=bds(r[-1000:],max_dim=2);diagnosticos.append({'modelo':m,'prueba':'BDS','rezago':2,'estadistico':float(z),'p':float(p)})\nfig.tight_layout();display(fig);plt.close(fig)\ndiagnosticos=pd.DataFrame(diagnosticos);diagnosticos['p_Holm']=multipletests(diagnosticos.p,method='holm')[1];diagnosticos.to_csv(OUT/'diagnosticos_residuos.csv',index=False)"),
       md('La autocorrelación residual identifica estructura temporal que las predicciones no absorben. Las bandas son una referencia descriptiva de ruido blanco; el diagnóstico formal conserva Ljung–Box, ARCH y BDS con Holm entre 20 contrastes. La persistencia de memoria justifica conservar bloques temporales al resumir incertidumbre y limita una interpretación i.i.d. de los errores.'),
       md('## Lectura conjunta\n\nLa MLP constituye la referencia empírica del benchmark por su MASE agregado. La evaluación temporal permite comprobar si esa posición se mantiene entre horas, años y tipos de día. La cercanía global entre ARX y CNN1D debe contrastarse con sus perfiles y costos, sin reducir la elección a una única diferencia puntual. Las cinco inicializaciones y los intervalos temporales describen fuentes de variación distintas.\n\nLos archivos derivados se conservan en results/jbook/modelos_actuales. El análisis reproduce métricas desde predicciones guardadas y no modifica pesos, escaladores ni configuraciones.')]
    books['10_evaluacion']=e
    _actualizar_diseno(books,md)
    _actualizar_literatura(books,md,code)
    _pipeline(root,snap,sha)
    # Mantener el inventario descargable del informe y su procedencia.
    static=root/'book/_static/modelos';static.mkdir(parents=True,exist_ok=True)
    for name in ['modelos.json','modelos_seleccionados.csv','indice_artefactos_v3_v4.csv']:
        shutil.copyfile(snap/'results/consolidado'/name,static/name)
    books['12_reproducibilidad']=[cell for cell in books['12_reproducibilidad'] if not (cell.cell_type=='markdown' and cell.source.startswith('## Resultados importados'))]
    books['12_reproducibilidad'].append(md('## Procedencia de modelos y reconstrucción\n\nLos modelos se identifican mediante el commit '+sha+' y el [registro de artefactos](_static/modelos/modelos.json). El [índice SHA-256](_static/modelos/indice_artefactos_v3_v4.csv) registra 2.304 archivos y la selección conserva cinco checkpoints por familia. Las métricas se reconstruyen desde etiquetas y predicciones con idénticos orígenes. El código está en scripts/modelos_actualizados_jbook.py; el informe usa results/jbook/modelos_actuales.\n\nPara reconstruir las secciones: python scripts/crear_jbook.py 08_literatura 08_validacion 09_modelos 10_evaluacion 11_sintesis 12_reproducibilidad; después, jupyter-book build book --warningiserror --keep-going y python scripts/verificar_jbook.py. Este procedimiento analiza artefactos guardados. El pipeline reproducible admite un dispositivo explícito y rutas con pathlib; la ejecución GPU registrada utiliza ROCm. La reproducción en Linux y Windows requiere instalar PyTorch compatible con el dispositivo elegido.'))


def _actualizar_diseno(books,md):
    for name in ['08_validacion','11_sintesis','12_reproducibilidad']:
        for cell in books[name]:
            if cell.cell_type!='markdown':continue
            s=cell.source
            s=re.sub(r'La ejecución v2[^\n]*','La evaluación utiliza cinco semillas por familia y resultados descriptivos.',s)
            s=s.replace('195','5')
            s=s.replace('la desigualdad del presupuesto LSTM y el diseño retrospectivo limitan la interpretación confirmatoria. La nueva instantánea contiene predicciones y curvas completas, utilizadas en los diagnósticos temporales.','el presupuesto desigual entre familias y el alcance exploratorio limitan la interpretación confirmatoria. Se conservan predicciones y curvas completas para los diagnósticos temporales.')
            s=s.replace('La inferencia requiere un preregistro específico:','El diseño de una inferencia confirmatoria requiere:')
            # Las secciones metodológicas se centran en el diseño vigente.
            s=s.replace('El análisis secundario de los resultados guardados se documenta en el capítulo de modelos y su evaluación temporal. Una evaluación confirmatoria futura requiere un preregistro específico:','El capítulo de modelos presenta resultados descriptivos y diagnósticos exploratorios. Una evaluación confirmatoria requiere:')
            s=s.replace('conforme al plan registrado antes de cargar las predicciones.','con índices de remuestreo apareados entre modelos.')
            s=s.replace('En una evaluación futura, los mismos índices se aplicarían conjuntamente a las pérdidas de ambos métodos.','En el análisis temporal, los mismos índices se aplican conjuntamente a las pérdidas de ambos métodos.')
            s=s.replace('Una instalación de PyTorch compatible con CUDA y el controlador es necesaria para GPU; el entorno documental no la exige.','La GPU requiere PyTorch compatible con CUDA o ROCm y con su controlador; el entorno documental no lo exige.')
            cell.source=s
            if cell.metadata.get('plantilla_resultados'):cell.metadata['plantilla_resultados']=s
    books['08_validacion'].append(md('## Ejecución y alcance estadístico\n\nLas configuraciones se seleccionan por confirmación de validación con dos semillas y cuatro años; el piloto evalúa cinco semillas por pliegue y la prueba cinco realizaciones por familia. El preprocesamiento se ajusta sólo al tramo de optimización de cada pliegue. El máximo es 30 épocas en filtro y 60 en confirmación, piloto y final; la parada temprana aplica mínimo diez épocas y paciencia seis, restaurando el mínimo absoluto.\n\nMASE es la métrica primaria, δ relativo es 3 %, α=0,05 y la potencia objetivo de diseño es 0,80. El tamaño ejecutado de cinco semillas no demuestra esa potencia. Los intervalos por inicialización son condicionales al mismo conjunto de fechas; el remuestreo temporal conserva días completos y bloques de siete, 14 y 28 días. Los contrastes tienen alcance exploratorio.'))
    books['11_sintesis']=[cell for cell in books['11_sintesis'] if not (cell.cell_type=='markdown' and cell.source.startswith('## Resultado del benchmark'))]
    books['11_sintesis'].append(md('## Resultado del benchmark\n\nLa MLP presenta el menor MASE medio del periodo de prueba; ARX y CNN1D quedan próximos y LSTM obtiene mayor error. Las cuatro familias superan las referencias estacionales en el agregado. La evaluación conserva cinco semillas por configuración y separa variabilidad algorítmica de incertidumbre temporal. Las diferencias se describen junto con perfiles horarios, segmentos, curvas de aprendizaje y costo de ajuste, sin afirmar superioridad confirmatoria.'))


def _actualizar_literatura(books,md,code):
    cells=books['08_literatura']
    a=next(i for i,c in enumerate(cells) if c.cell_type=='markdown' and c.source.startswith('## 8.6.'))
    b=next(i for i,c in enumerate(cells[a:],a) if c.cell_type=='markdown' and c.source.startswith('### Extensiones'))
    cells[a:b]=[
      md('## 8.6. Arquitecturas de interés y modelos numéricos\n\nMLP, LSTM y CNN1D contrastan representación densa, estado recurrente y filtros temporales; ARX aporta un control lineal regularizado. Pelekis et al. sustenta conservar una referencia densa frente a métodos más complejos. La frecuencia de comparadores complementa esta justificación sin constituir una regla universal de selección.\n\nLas configuraciones seleccionadas utilizan historia de 168 horas, salida de 24 y 23 variables de calendario. MLP contiene dos capas de 128 unidades; LSTM una capa de 48; CNN1D una convolución de 16 filtros y kernel cinco, seguida de promedio local de factor dos y una cabeza lineal. LSTM y CNN1D concatenan calendario futuro a la representación temporal y predicen residuos alrededor del nivel. ARX utiliza 48 rezagos.\n\nCon f=24, L=168, H=24, K=23 y d=L+HK, MLP tiene P=(d+1)u+(u+1)u+(u+1)H; LSTM, P=4u(f+u+2)+(u+HK+1)H; CNN1D, P=(kf+1)c+(Lc/2+HK+1)H. Los costos dominantes por ejemplo son O(du+u²+uH), O(Lu(f+u)+(u+HK)H) y O(Lkfc+(Lc/2+HK)H), respectivamente.'),
      code("L,H,K,f=168,24,23,24;d=L+H*K;u=128\nparametros=pd.Series({'ARX':(48+H*K+1)*H,'CNN1D':(5*f+1)*16+((L//2)*16+H*K+1)*H,'MLP':(d+1)*u+(u+1)*u+(u+1)*H,'LSTM':4*48*(f+48+2)+(48+H*K+1)*H})\nassert parametros.to_dict()=={'ARX':14424,'CNN1D':47464,'MLP':111896,'LSTM':28632}\nfig,ax=plt.subplots(figsize=(8,3.7));parametros.plot.barh(ax=ax,color=['#9ba5ae','#2a9d8f','#d18a2d','#247ba0']);ax.set_xscale('log');ax.set_xlabel('Parámetros / coeficientes; escala logarítmica');fig.tight_layout();display(fig);plt.close(fig)"),
      md('La MLP presenta la mayor capacidad parametrizada. La CNN1D conserva una salida densa sobre posiciones temporales agrupadas, por lo que su tamaño no depende sólo del número de filtros. Los parámetros no determinan por sí solos precisión ni latencia; el capítulo de modelos contrasta error, variabilidad y tiempo registrado.')]
    for cell in cells:
        if cell.cell_type=='markdown':
            cell.source=cell.source.replace(' La ampliación documental es posterior a los entrenamientos y no modifica el diseño ni las configuraciones ejecutadas.','')


def _pipeline(root,snap,sha):
    tick=chr(96);fence=tick*3
    parts=['# Pipeline de entrenamiento y evaluación\n\nEl pipeline utiliza historia anterior al origen, calendario conocido del horizonte, escalado por pliegue y parada interna. El código corresponde a los modelos identificados en el registro consolidado. Los bloques documentan el procedimiento y no se ejecutan al construir el libro.\n\nLas rutas se expresan con pathlib. El dispositivo se selecciona entre CPU y GPU compatible con PyTorch; el entorno y las dependencias se registran junto con los resultados. Las etiquetas de prueba requieren acceso explícito en una evaluación final.\n']
    files=[('Partición temporal y parada interna','src/demanda/particion.py',None),
           ('Datos, ventanas, escalas y referencias','src/demanda/modelos_base.py',{'ConfigDatos','Conjunto','_columnas','Datos','_tabular_base','tabular_lineal'}),
           ('Modelos densos y opciones de ejecución','src/demanda/modelos_base.py',{'Opciones','resolver_dispositivo','fijar_semillas','_definir_redes','ModeloTorch'}),
           ('Preprocesamiento por pliegue y bucle de optimización','src/demanda/reentrenamiento_v3.py',{'DatosFold','Secuencial','Entrenador'}),
           ('Codificador recurrente seleccionado','src/demanda/variantes_v4.py',{'Secuencial'}),
           ('Métricas','src/demanda/metricas.py',None)]
    for title,file,names in files:
        source=(snap/file).read_text(encoding='utf-8');tree=ast.parse(source)
        if names is not None:
            selected=[]
            for node in tree.body:
                if isinstance(node,ast.FunctionDef) and node.name=='_definir_redes' and node.name in names:
                    for inner in node.body:
                        if isinstance(inner,ast.ClassDef) and inner.name in {'Tabular','ARXLineal','MLP'}:
                            selected.append(textwrap.dedent('\n'.join(source.splitlines()[inner.lineno-1:inner.end_lineno])))
                elif isinstance(node,ast.ClassDef) and node.name=='ModeloTorch' and node.name in names:
                    methods=[inner for inner in node.body if isinstance(inner,ast.FunctionDef) and inner.name in {'__init__','_tensores','predecir'}]
                    selected.append('class ModeloTorch:\n'+'\n\n'.join('\n'.join(source.splitlines()[inner.lineno-1:inner.end_lineno]) for inner in methods))
                elif isinstance(node,(ast.Import,ast.ImportFrom)) or isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in names:
                    start=min([node.lineno]+[d.lineno for d in getattr(node,'decorator_list',[])])
                    selected.append('\n'.join(source.splitlines()[start-1:node.end_lineno]))
            source='\n\n'.join(selected)
        source=source.replace('como denominador de referencia de la ejecución original; no participa en optimización','como denominador de referencia; no participa en optimización')
        parts.append('## '+title+'\n\nFuente: '+tick+file+tick+'. Se muestran las clases y funciones que intervienen en el pipeline.\n\n'+fence+'python\n'+source+'\n'+fence+'\n')
    pipeline=r'''from pathlib import Path
import json,sys
import numpy as np
import pandas as pd

def ejecutar_configuracion(raiz, familia, semilla=1000, dispositivo='auto',
                          fase='validacion', anio=2021, permitir_prueba=False,
                          salida=None):
    raiz=Path(raiz).resolve()
    procedencia=json.loads((raiz/'results/jbook/modelos_actualizados.json').read_text())
    fuentes=raiz/'results/repositorio'/procedencia['commit']
    sys.path.insert(0,str(fuentes/'src'))
    from demanda import modelos_base as M, particion as P
    from demanda import reentrenamiento_v3 as motor
    from demanda.variantes_v4 import Secuencial as Recurrente
    from demanda.metricas import evaluar
    import torch
    registro=json.loads((fuentes/'results/consolidado/modelos.json').read_text())
    config=registro[familia]['configuracion']
    panel=pd.read_parquet(raiz/'data/processed/panel_horario.parquet')
    variables=json.loads((fuentes/'data/processed/feature_registry.json').read_text())
    cfg=M.ConfigDatos(ventana=config['ventana'])
    if fase=='prueba':
        if not permitir_prueba:raise PermissionError('La prueba requiere acceso explícito')
        tr,es=P.pretest(panel.index,ventana=cfg.ventana)
        ev=P.origenes(panel.index,'test',ventana=cfg.ventana,allow_test=True)
    else:
        fold=P.folds_forward_chaining(panel.index,[anio],ventana=cfg.ventana)[0]
        tr,es,ev=fold['tr'],fold['es'],fold['val']
    datos=motor.DatosFold(panel,variables,cfg,tr,secuencial=familia in ('lstm','cnn1d'))
    train,stop,evaluacion=[datos.conjunto_pos(p) for p in [tr,es,ev]]
    assert train.origen.max()+pd.Timedelta(hours=23)<stop.origen.min()
    assert stop.origen.max()+pd.Timedelta(hours=23)<evaluacion.origen.min()
    assert pd.Timestamp(datos.auditoria['fin_ajuste'])<stop.origen.min()
    fabrica=motor.Secuencial
    try:
        if familia=='lstm':motor.Secuencial=Recurrente
        modelo=motor.Entrenador(familia)
        info=modelo.ajustar(train,stop,datos,config['params'],semilla,
                           M.Opciones(max_epocas=60,dispositivo=dispositivo,hilos=2))
    finally:
        motor.Secuencial=fabrica
    pred=datos.invertir(modelo.predecir(evaluacion),evaluacion)
    metricas=evaluar(evaluacion.y,pred,datos.escala_mase)
    if salida is not None:
        destino=Path(salida)/familia/str(semilla)/str(anio if fase=='validacion' else fase)
        destino.mkdir(parents=True,exist_ok=False)
        escala=datos.escaladores()
        torch.save({'state_dict':{k:v.detach().cpu() for k,v in modelo.red.state_dict().items()},
                    'modelo':familia,'params':config['params'],'escaladores':escala},
                    destino/'modelo.pt')
        np.savez_compressed(destino/'predicciones.npz',y=evaluacion.y,yhat=pred,
                            origen=evaluacion.origen.as_unit('ns').asi8)
        for nombre,objeto in [('escaladores',escala),('historia',info['historia']),
                              ('metricas',metricas)]:
            (destino/(nombre+'.json')).write_text(json.dumps(objeto,ensure_ascii=False,
                                                indent=2,default=str),encoding='utf-8')
    return modelo,datos,evaluacion,pred,info,metricas

# Selección: comparar el MASE medio de confirmación, sin leer prueba.
# Piloto: repetir cada configuración seleccionada con semillas 100–104
# en los cuatro años externos. Final: semillas 1000–1004 con acceso explícito.
# El argumento salida conserva pesos, escaladores, curvas, métricas y predicciones.
'''
    parts.append('## Pipeline integrado\n\nEl siguiente bloque vincula las implementaciones mostradas y permite seleccionar CPU o GPU sin funciones exclusivas del sistema operativo. La inicialización del dispositivo se conserva en resolver_dispositivo.\n\n'+fence+'python\n'+pipeline+'\n'+fence+'\n')
    parts.append('## Reconstrucción del benchmark\n\nLas celdas de los capítulos de modelos y evaluación cargan exclusivamente predicciones, historiales y metadatos guardados. Verifican igualdad de etiquetas y orígenes, reconstruyen métricas y calculan los diagnósticos. El registro identifica el commit '+sha+'.\n')
    (root/'book/10_pipeline.md').write_text('\n'.join(parts),encoding='utf-8')

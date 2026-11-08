"""Reconstruye referencias deterministas desde observaciones ya guardadas."""
REFERENCIAS=r'''
# Demanda observada archivada en un pliegue de 2021 y en la prueba final.
indice_ref=pd.read_csv(R/'indice_artefactos.csv')
sel_2021=(indice_ref.etapa=='piloto')&(indice_ref.modelo=='mlp')&(indice_ref.semilla==100)&indice_ref.inicio_evaluacion.str.startswith('2021')&indice_ref.archivo.str.endswith('/predicciones.npz')
sel_test=(indice_ref.etapa=='final')&(indice_ref.modelo=='mlp')&(indice_ref.semilla==1000)&indice_ref.archivo.str.endswith('/predicciones.npz')
assert sel_2021.sum()==1 and sel_test.sum()==1
series=[]
for sel in [sel_2021,sel_test]:
    with np.load(R/indice_ref.loc[sel,'archivo'].iloc[0],allow_pickle=False) as z:
        origen_ref=pd.to_datetime(z['origen'],unit='ns',utc=True).tz_convert('America/Bogota')
        idx=pd.DatetimeIndex([o+pd.Timedelta(hours=h) for o in origen_ref for h in range(24)])
        series.append(pd.Series(z['y'].ravel(),index=idx))
        if sel is sel_test:
            y_ref=z['y'].copy();orig_ref=z['origen'].copy();fechas_ref=origen_ref
observado=pd.concat(series).sort_index()
assert observado.index.is_unique
ref_pred={}
lag_publicacion=1
for nombre,base in [('ingenuo_24h',24),('ingenuo_168h',168)]:
    p=np.empty_like(y_ref)
    for h in range(24):
        k=max(1,int(np.ceil((h+1+lag_publicacion)/base))) if base==24 else 1
        requeridas=fechas_ref+pd.Timedelta(hours=h-base*k)
        p[:,h]=observado.loc[requeridas].to_numpy()
    ref_pred[nombre]=p
# El último intervalo disponible al origen es el que empieza dos horas antes.
ref_pred['persistencia']=np.repeat(observado.loc[fechas_ref-pd.Timedelta(hours=2)].to_numpy()[:,None],24,axis=1)
for nombre,p in ref_pred.items():
    absoluto=np.abs(p-y_ref);e=p-y_ref
    res={'MAE':float(absoluto.mean()),'RMSE':float(np.sqrt(np.mean(e*e))),
         'MASE':float(absoluto.mean()/escala_mase),
         'sMAPE':float(np.mean(200*absoluto/(np.abs(p)+np.abs(y_ref)))),
         'MAPE':float(np.mean(100*absoluto/np.abs(y_ref))),
         'R2':float(1-np.mean(e*e)/np.var(y_ref))}
    for h in [1,12,24]:res[f'MASE_h{h}']=float(absoluto[:,h-1].mean()/escala_mase)
    if nombre in referencias:
        for metrica,valor in referencias[nombre].items():
            assert np.isclose(res[metrica],valor,rtol=1e-10),(nombre,metrica)
    referencias[nombre]=res
etiquetas['persistencia']='Persistencia disponible'
np.savez_compressed(OUT/'referencias_desde_observaciones.npz',y=y_ref,origen=orig_ref,**ref_pred)
(OUT/'referencias_reconstruidas.json').write_text(json.dumps(referencias,indent=2),encoding='utf-8')
'''


def completar_referencias(books,md,code,note,root):
    c=books['09_modelos']
    c[2].source+='\n'+REFERENCIAS
    c.insert(4,md('Las referencias deterministas se reconstruyen con demanda observada ya archivada: persistencia del último intervalo disponible e ingenuos diario y semanal. Se respeta el rezago de publicación de una hora. Los resultados de los dos ingenuos coinciden con los publicados; no se ejecutan redes ni se ajustan parámetros.'))
    e=books['10_evaluacion']
    for cell in e:
        if cell.cell_type=='code' and 'metricas=[]; loss_h={}' in cell.source:
            cell.source+="""
with np.load(OUT/'referencias_desde_observaciones.npz',allow_pickle=False) as z:
    assert np.array_equal(z['y'],y) and np.array_equal(z['origen'],origen)
    refs=['ingenuo_24h','ingenuo_168h','persistencia']
    for nombre in refs:loss_h[nombre]=np.abs(z[nombre]-y)/escala;loss_d[nombre]=loss_h[nombre].mean(axis=1)
etiquetas.update({'ingenuo_24h':'Ingenuo diario','ingenuo_168h':'Ingenuo semanal','persistencia':'Persistencia'})
"""
        if cell.cell_type=='code' and "boot_h={n:pesos@loss_h[n] for n in modelos}" in cell.source:
            cell.source=cell.source.replace("'Contexto ARX':[(n,'arx_lineal') for n in neuronales]}",
                "'Contexto ARX':[(n,'arx_lineal') for n in neuronales],\n          'Referencias ingenuas':[('mlp',n) for n in refs]}")
            cell.source=cell.source.replace("boot_h={n:pesos@loss_h[n] for n in modelos}","boot_h={n:pesos@loss_h[n] for n in [*modelos,*refs]}")
            cell.source=cell.source.replace("boot_d={n:boot_h[n].mean(axis=1) for n in modelos}","boot_d={n:boot_h[n].mean(axis=1) for n in [*modelos,*refs]}")
        if cell.cell_type=='code' and "bdict={n:pesos@loss_d[n] for n in modelos}" in cell.source:
            cell.source=cell.source.replace("bdict={n:pesos@loss_d[n] for n in modelos}","bdict={n:pesos@loss_d[n] for n in [*modelos,*refs]}")
        if cell.cell_type=='markdown':
            cell.source=cell.source.replace('Las comparaciones neuronales y las de contexto ARX forman familias separadas de tres hipótesis, con corrección de Holm.',
                'Las comparaciones neuronales, las de contexto ARX y las de MLP frente a tres ingenuos forman familias separadas de tres hipótesis, con corrección de Holm.')
            cell.source=cell.source.replace('Las 72 comparaciones horarias neuronales y las 72 de contexto se corrigen por Holm en familias separadas.',
                'Las comparaciones por horizonte se corrigen por Holm en tres familias de 72 hipótesis: redes entre sí, redes frente a ARX y MLP frente a ingenuos.')

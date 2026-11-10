"""Cobertura anual y recencia editorial de oficiales y prensa, sin entrenar."""
from pathlib import Path
import json,re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import PercentFormatter

ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'results/documentos_oficiales';OUT=BASE/'cobertura_anual_recencia';OUT.mkdir(exist_ok=True)
TZ='America/Bogota';H=72;NS=pd.Timedelta(hours=H).value
ORIGENES=pd.date_range('2000-01-01','2025-12-31',freq='D',tz=TZ);V=ORIGENES.tz_convert('UTC').as_unit('ns').asi8
PARTES={'Entrenamiento 2000–2017':('2000-01-01','2017-12-31'),'Validación 2018–2021':('2018-01-01','2021-12-31'),'Prueba 2022–2025':('2022-01-01','2025-12-31'),'Historial previo a prueba 2000–2021':('2000-01-01','2021-12-31')}
PALABRAS=re.compile(r"[^\W\d_]+(?:['’\-][^\W\d_]+)*",re.UNICODE)

def preparar():
    o=pd.read_parquet(ROOT/'data/raw/documentos_oficiales/documentos_oficiales.parquet')
    o=o.loc[~o.cuarentena].copy();o['bloque']='Oficiales'
    t=pd.read_parquet(ROOT/'data/raw/noticias/titulares.parquet');t=t.loc[~t.quarantined&t.source_kind.eq('news')].copy()
    ver=t.availability_status.eq('verified_archived_version')&t.availability_proof.eq('wayback_payload_digest_and_capture')
    p=pd.DataFrame({'id':t.id,'fuente':t.source,'titulo':t.titular,'texto':'','fecha_declarada':t.publication_date_claimed,'precision_fecha':'day','base_fecha':'publisher_declared_publication_date','disponible_declarado':pd.to_datetime(t.available_at_assumed,utc=True,errors='coerce',format='mixed'),'disponible_estricto':pd.to_datetime(t.available_at_verified.where(ver),utc=True,errors='coerce',format='mixed'),'bloque':'Prensa previa'})
    d=pd.concat([o[['id','fuente','titulo','texto','fecha_declarada','precision_fecha','base_fecha','disponible_declarado','disponible_estricto','bloque']],p],ignore_index=True)
    bases_editoriales={'publisher_declared_publication_date','publisher_listing_publication_date','publisher_api_date_published','official_gazette_publication_date_in_document','publisher_listing_publication_day_time_not_certified','document_header_edition_date','archive_member_filename_edition_date'}
    precisa=d.precision_fecha.eq('day')&d.fecha_declarada.fillna('').str.fullmatch(r'\d{4}-\d{2}-\d{2}')&d.base_fecha.isin(bases_editoriales)
    ed=pd.to_datetime(d.fecha_declarada.where(precisa),errors='coerce')
    d['edicion_inicio']=ed.dt.tz_localize(TZ).dt.tz_convert('UTC')
    def palabras(r):
        txt=r.texto if isinstance(r.texto,str) and r.texto.strip() else r.titulo
        txt=re.sub(r'\[(?:Página|Diapositiva)[^\]]*\]','',str(txt))
        return len(PALABRAS.findall(txt))
    d['palabras']=np.fromiter((palabras(r) for r in d.itertuples()),dtype='int64',count=len(d))
    return d

def intervalos(g,escenario,reciente):
    disp=pd.to_datetime(g['disponible_'+escenario],utc=True,errors='coerce')
    bien=disp.notna()
    if reciente:bien&=g.edicion_inicio.notna()
    z=g.loc[bien].copy();a=pd.DatetimeIndex(disp.loc[bien]).as_unit('ns').asi8
    if reciente:
        e=pd.DatetimeIndex(z.edicion_inicio).as_unit('ns').asi8
        inicio=np.maximum(a,e);fin=np.minimum(a+NS,e+NS)
    else:inicio=a;fin=a+NS
    lo=np.searchsorted(V,inicio,side='left');hi=np.searchsorted(V,fin,side='right')
    validos=lo<hi;z=z.loc[validos].copy();lo=lo[validos];hi=hi[validos]
    delta=np.zeros(len(V)+1,dtype='int64');np.add.at(delta,lo,1);np.add.at(delta,hi,-1)
    return np.cumsum(delta[:-1]),z,lo,hi

def main():
    d=preparar();res=[];anuales=[];series=[]
    for bloque in ['Oficiales','Prensa previa','Oficiales + prensa previa']:
        frame=d if bloque=='Oficiales + prensa previa' else d.loc[d.bloque.eq(bloque)]
        fuentes=['TODAS']+(sorted(frame.fuente.unique()) if bloque=='Oficiales' else [])
        for fuente in fuentes:
            g=frame if fuente=='TODAS' else frame.loc[frame.fuente.eq(fuente)]
            for escenario in ['declarado','estricto']:
                disponible,_,_,_=intervalos(g,escenario,False)
                reciente,docs,lo,hi=intervalos(g,escenario,True)
                assert np.all(reciente<=disponible)
                if fuente=='TODAS':
                    series.append(pd.DataFrame({'origen':ORIGENES,'bloque':bloque,'escenario':escenario,'documentos_disponibles_72h':disponible,'documentos_recientes_72h':reciente}))
                for periodo,(inicio,fin) in PARTES.items():
                    j=ORIGENES.searchsorted(pd.Timestamp(inicio,tz=TZ));k=ORIGENES.searchsorted(pd.Timestamp(fin,tz=TZ),side='right')
                    usados=(lo<k)&(hi>j);r={'bloque':bloque,'fuente':fuente,'escenario':escenario,'periodo':periodo,'origenes':k-j,'origenes_con_disponibles_72h':int((disponible[j:k]>0).sum()),'origenes_con_recientes_72h':int((reciente[j:k]>0).sum()),'documentos_recientes_distintos':int(usados.sum()),'palabras_recientes_distintas':int(docs.loc[usados,'palabras'].sum())}
                    r['cobertura_disponibles_72h']=r['origenes_con_disponibles_72h']/(k-j);r['cobertura_recientes_72h']=r['origenes_con_recientes_72h']/(k-j);res.append(r)
                if bloque=='Oficiales':
                    for anio in range(2000,2026):
                        j=ORIGENES.searchsorted(pd.Timestamp(f'{anio}-01-01',tz=TZ));k=ORIGENES.searchsorted(pd.Timestamp(f'{anio}-12-31',tz=TZ),side='right');usados=(lo<k)&(hi>j)
                        anuales.append({'anio':anio,'fuente':fuente,'escenario':escenario,'origenes':k-j,'origenes_con_recientes_72h':int((reciente[j:k]>0).sum()),'cobertura_recientes_72h':float((reciente[j:k]>0).mean()),'documentos_recientes_distintos':int(usados.sum()),'palabras_recientes_distintas':int(docs.loc[usados,'palabras'].sum())})
    resultados=pd.DataFrame(res);resultados.to_csv(OUT/'texto_reciente_particiones_fuente.csv',index=False)
    pd.DataFrame(anuales).to_csv(OUT/'texto_reciente_anual_fuente.csv',index=False)
    pd.concat(series,ignore_index=True).to_parquet(OUT/'texto_reciente_origenes.parquet',index=False,compression='zstd')
    anual=pd.read_csv(BASE/'oficiales_cobertura_anual_fuente.csv');counts=anual.loc[anual.fuente.ne('TODAS')]
    fuentes=['CREG','IDEAM','MinEnergia','UPME','XM'];etiquetas=['CREG','IDEAM','MinEnergía','UPME','XM'];maximo=int(counts.documentos_habilitados.max())
    fig,axes=plt.subplots(2,1,figsize=(17,6),sharex=True,layout='constrained')
    for ax,esc in zip(axes,['declarado','estricto']):
        matriz=counts.loc[counts.escenario.eq(esc)].pivot(index='fuente',columns='anio',values='documentos_habilitados').reindex(fuentes)
        datos=matriz.to_numpy();im=ax.imshow(np.ma.masked_equal(datos,0),aspect='auto',norm=LogNorm(vmin=1,vmax=maximo),cmap='YlGnBu')
        ax.set_facecolor('#f3f4f6');ax.set_title('Escenario '+esc+' · documentos habilitados por año',fontsize=12);ax.set_yticks(range(5),etiquetas);ax.set_xticks(range(26),range(2000,2026),rotation=45)
        ax.axvline(21.5,color='#c53030',ls='--',lw=1.3)
        for (i,j),n in np.ndenumerate(datos):ax.text(j,i,str(n),ha='center',va='center',fontsize=8,color='white' if n>=100 else '#1f2937')
    fig.colorbar(im,ax=axes,label='Documentos · color logarítmico',shrink=.8)
    fig.savefig(OUT/'documentos_anio_fuente.png',dpi=180);plt.close(fig)
    for esc in ['declarado','estricto']:
        counts.loc[counts.escenario.eq(esc)].pivot(index='anio',columns='fuente',values='documentos_habilitados').reindex(columns=fuentes).to_csv(OUT/('documentos_anio_fuente_'+esc+'.csv'))
    # En cada celda: documentos habilitados y porcentaje de orígenes cubiertos.
    fig,axes=plt.subplots(2,1,figsize=(17,7),sharex=True,layout='constrained')
    for ax,esc in zip(axes,['declarado','estricto']):
        gg=counts.loc[counts.escenario.eq(esc)];nums=gg.pivot(index='fuente',columns='anio',values='documentos_habilitados').reindex(fuentes);cov=gg.pivot(index='fuente',columns='anio',values='cobertura_72h').reindex(fuentes)
        im=ax.imshow(cov.to_numpy(),aspect='auto',vmin=0,vmax=1,cmap='YlGnBu');ax.set_title('Escenario '+esc+' · documentos y cobertura por año de disponibilidad',fontsize=12)
        ax.set_yticks(range(5),etiquetas);ax.set_xticks(range(26),range(2000,2026),rotation=45);ax.axvline(21.5,color='#c53030',ls='--',lw=1.2)
        for i in range(5):
            for j in range(26):
                v=float(cov.iloc[i,j]);texto=str(int(nums.iloc[i,j]))+'\n'+f'{100*v:.1f}'.replace('.',',')+'%';ax.text(j,i,texto,ha='center',va='center',fontsize=7,color='white' if v>.55 else '#1f2937')
    fig.colorbar(im,ax=axes,label='Orígenes con texto disponible en 72 h',format=PercentFormatter(1),shrink=.8)
    fig.savefig(OUT/'cobertura_anual_documentos_y_dias.png',dpi=180);plt.close(fig)
    # Contraste independiente mediante máscaras para límites y fechas con capturas recientes.
    controles=0;oficiales=d.loc[d.bloque.eq('Oficiales')]
    for esc in ['declarado','estricto']:
        calculado,_,_,_=intervalos(oficiales,esc,True);disp=pd.to_datetime(oficiales['disponible_'+esc],utc=True)
        for fecha in ['2017-12-31','2018-01-01','2020-11-01','2021-08-08','2021-12-31','2022-01-01','2022-02-01','2022-02-04','2025-12-31']:
            t0=pd.Timestamp(fecha,tz=TZ);m=disp.between(t0-pd.Timedelta(hours=72),t0)&oficiales.edicion_inicio.between(t0-pd.Timedelta(hours=72),t0)
            assert int(m.sum())==int(calculado[ORIGENES.get_loc(t0)]);controles+=1
    (OUT/'validacion_recencia.json').write_text(json.dumps({'controles_por_mascaras':controles,'resultado':'correcto','ventana_horas':72,'fechas_de_carga_excluidas_como_edicion':True,'entrenamiento_ejecutado':False,'publicacion_ejecutada':False},indent=2),encoding='utf-8')
    resumen=resultados.loc[resultados.fuente.eq('TODAS')]
    (OUT/'resumen.json').write_text(json.dumps(resumen.to_dict('records'),ensure_ascii=False,indent=2),encoding='utf-8')
    texto='''# Cobertura anual y texto reciente

La figura agrupa por año en que el documento queda disponible en cada escenario. Cada celda muestra documentos y porcentaje de orígenes diarios con algún documento en una ventana de 72 horas. Se excluyen cuarentena y disponibilidad posterior a 2025.

![Cobertura anual por fuente](cobertura_anual_documentos_y_dias.png)

La disponibilidad declarada aumenta desde 2013, especialmente por MinEnergía e IDEAM. La disponibilidad histórica corroborada se concentra en los años recientes y comprende documentos antiguos capturados después.

Para contar texto reciente se exige una fecha editorial con precisión diaria, edición en las últimas 72 horas y disponibilidad al origen. El origen es las 00:00 de Bogotá. Se excluyen fechas de creación/modificación de archivos de XM como sustituto de edición y fechas solo anuales. Estos documentos pueden conservar valor de contexto; no se eliminan del dataset.

Los conteos de documentos y palabras son distintos por periodo, sin repetir un documento por cada origen. Un documento puede cruzar un corte y aparecer en ambos periodos. Las palabras proceden del cuerpo extraído o, si falta, del titular; no son tokens de un modelo. Parte del cuerpo es OCR parcial.

El historial 2000–2021 incluye entrenamiento inicial y validación; no se identifica con el tramo efectivo de optimización de cada pliegue ni con su reserva interna de parada.

Archivos: documentos_anio_fuente_declarado.csv, documentos_anio_fuente_estricto.csv, texto_reciente_anual_fuente.csv y texto_reciente_particiones_fuente.csv. El bloque Prensa previa incluye los siete medios del corpus original; Oficiales corresponde al bloque de documentos incorporado. No se aplican filtros temáticos ni se ajustan modelos.
'''
    (OUT/'LECTURA_RESULTADOS.md').write_text(texto,encoding='utf-8')
    print(resumen.to_string(index=False),flush=True)

if __name__=='__main__':main()

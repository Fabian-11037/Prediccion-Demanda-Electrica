"""Exporta resultados conservados. No importa ni ejecuta redes."""
from pathlib import Path
import json,hashlib,shutil
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2];SITE=ROOT/'dashboard/site';OUT=SITE/'data';R=ROOT/'results/evaluacion_e2'
BOOK='https://fabian-11037.github.io/Prediccion-Demanda-Electrica/'
FAMILY={'mlp':'mlp','lstm':'lstm','cnn1d':'cnn1d','arx_lineal':'arx','repo_nhits_numerico_r1__N':'nhits',
 'adj_modelo3_20261115_0419__N':'tft','adj_modelo4_20261114_1224__N':'tcn','adj_modelo5_20261114_1327__N':'itransformer',
 'adj_modelo6_20261112_1035__N':'patchtst','adj_modelo6_20261112_1035__N+O':'tats',
 'adj_fusion_tardia_20261113_1728__N+O':'fusion','adj_modelo7_20261114_0430__N':'tgforecaster','adj_modelo7_20261114_0430__N+O':'tgforecaster'}
LABEL={'arx':'ARX','mlp':'MLP','lstm':'LSTM','cnn1d':'CNN1D','patchtst':'PatchTST','nhits':'N-HiTS',
 'tft':'TFT','tcn':'TCN','itransformer':'iTransformer','tats':'TaTS','fusion':'Fusión tardía','tgforecaster':'TGForecaster'}
DESCRIPTION={'arx':'Combinación lineal regularizada de 48 rezagos y del calendario futuro.',
 'mlp':'Dos capas densas de 128 unidades combinan historia y calendario.',
 'lstm':'Memoria recurrente de 48 unidades y lectura con calendario conocido.',
 'cnn1d':'Filtros locales de cinco horas, agrupamiento y salida directa de 24 horas.',
 'patchtst':'Parches de 16 horas, dos capas de atención y dos lecturas: demanda y contexto.',
 'nhits':'Tres bloques residuales explican la historia a distintas escalas y suman pronósticos interpolados.',
 'tft':'Selección de variables, recurrencia, compuertas y atención, con ancho interno de 160.',
 'tcn':'Tres bloques causales con dilataciones 1, 4 y 16; campo receptivo de 169 horas.',
 'itransformer':'Atención entre 56 tokens de variables y calendario, de dimensión 128.',
 'tats':'La relevancia documental aprendida se convierte en canales temporales auxiliares de PatchTST.',
 'fusion':'Una rama numérica y otra documental aprenden conjuntamente y suman sus salidas.',
 'tgforecaster':'Descripciones de variables y calendario guían consultas sobre parches históricos.'}

def write(name,obj):
 p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,ensure_ascii=False,allow_nan=False,separators=(',',':')),encoding='utf-8')
def rows(frame):return json.loads(frame.to_json(orient='records',double_precision=12))

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 b=pd.read_csv(R/'benchmark.csv');keys=b.loc[b.semillas.eq(5),'clave'].tolist();selected=set(keys)
 assert len(keys)==13 and b.modelo.str.startswith('PatchTST').sum()==1
 meta=pd.read_csv(R/'metadatos_procedimientos.csv');lat=pd.read_csv(ROOT/'results/diagnosticos_ampliados/latencia_resumen.csv').set_index('clave')
 models=[]
 for v in b.loc[b.semillas.eq(5)].itertuples():
  f=FAMILY[v.clave];m=meta[meta.clave.eq(v.clave)]
  models.append(dict(id=v.clave,name=LABEL[f]+(' · N+O' if v.clave.endswith('N+O') else ' · N' if f=='tgforecaster' else ''),original=v.modelo,family=f,
   group='Comparador numérico' if f in ['patchtst','nhits'] else 'Modelo principal',
   inputs='Numéricas + documentos oficiales' if v.clave.endswith('N+O') else 'Demanda + calendario' if v.entradas=='demanda_calendario' else 'Numéricas ampliadas',
   description=DESCRIPTION[f],params=int(m.parametros.iloc[0]),latency=float(lat.loc[v.clave,'mediana_ms']),epochs=float(m.epocas.median()),training_seconds=float(m.segundos.median()),book=BOOK+'arquitecturas/'+f+'.html'))
 tables={}
 for key,file in {'benchmark':'benchmark','temporal':'comparaciones_temporales','seeds':'comparaciones_semillas','text':'contrastes_texto',
  'horizons':'resultados_horizonte','residuals':'diagnostico_residuos','ranks':'rangos','mcs':'mcs','curves':'curvas_aprendizaje',
  'selection':'diseno_seleccion','precision':'precision_desarrollo_resumen'}.items():tables[key]=rows(pd.read_csv(R/(file+'.csv')))
 imp=pd.read_csv(ROOT/'results/diagnosticos_ampliados/importancia_resumen.csv');imp=imp[imp.clave.isin(selected)]
 baseimp=pd.read_csv(ROOT/'results/jbook/importancia_variables/resumen.csv').rename(columns={'modelo':'clave','cambio_MAE_pct':'cambio_pct','p10_perm_pct':'p10','p90_perm_pct':'p90'})
 sens=pd.read_csv(ROOT/'results/diagnosticos_ampliados/sensibilidad_resumen.csv')
 write('study.json',dict(models=models,tables=tables,importance=rows(imp)+rows(baseimp),sensitivity=rows(sens),
  protocol=json.loads((R/'protocolo.json').read_text(encoding='utf-8')),authors=['Fabian José Manrique Morelos','Gloria Bastidas Rodriguez'],book=BOOK,
  dates=pd.date_range('2022-01-01','2025-12-31',freq='D').strftime('%Y-%m-%d').tolist()))
 # Se utilizan etiquetas canónicas y las cinco predicciones originales de cada origen.
 df=pd.read_parquet(R/'entradas/consolidado/predicciones_finales.parquet');pred={};truth=None
 for (exp,var),g in df.groupby(['experimento','variante'],sort=False):
  key=exp+'__'+var
  if key not in selected:continue
  arrays=[]
  for seed,part in g.groupby('semilla',sort=True):
   part=part.sort_values(['origen','horizonte']);y=part.observado_kwh.to_numpy().reshape(1461,24)
   if truth is None:truth=y
   else:assert np.array_equal(y,truth)
   arrays.append(part.predicho_kwh.to_numpy().reshape(1461,24))
  pred[key]=np.stack(arrays)
 snap=ROOT/'results/repositorio/d1ae9a4e09dd1dead4af1329f16aa3e8d0f2c3d3';reg=json.loads((snap/'results/consolidado/modelos.json').read_text(encoding='utf-8'))
 for key in ['mlp','lstm','cnn1d','arx_lineal']:
  arrays=[]
  for a in sorted(reg[key]['artefactos'],key=lambda a:a['semilla']):
   with np.load(snap/a['predicciones'],allow_pickle=False) as z:arrays.append(z['yhat'])
  pred[key]=np.stack(arrays)
 assert set(pred)==selected
 write('observed.json',dict(values=np.round(truth/1e6,8).tolist()))
 for key,v in pred.items():
  e=v-truth;assert np.isclose(np.abs(e).mean(),b.loc[b.clave.eq(key),'MAE'].iloc[0],rtol=1e-10)
  write('predictions/'+key+'.json',dict(mean=np.round(v.mean(0)/1e6,8).tolist(),p10=np.round(np.quantile(v,.1,axis=0)/1e6,8).tolist(),
   p90=np.round(np.quantile(v,.9,axis=0)/1e6,8).tolist(),mae_daily=np.round(np.abs(e).mean(axis=(0,2)),6).tolist(),
   mse_daily=np.round((e**2).mean(2),3).tolist(),mae_hourly=np.round(np.abs(e).mean(0),3).tolist()))
 panel=pd.read_parquet(ROOT/'data/processed/panel_horario.parquet');daily=panel.y_demanda_real_kwh.resample('D').mean()/1e6
 cols=[dict(id='demand',name='Demanda real',unit='GWh por intervalo',group='Demanda',values=np.round(daily,5).tolist())]
 spec=[('x_precio_bolsa_cop_kwh','Precio de bolsa','COP/kWh','Precio',1),('x_hid_llenado_valor_original','Llenado de embalses','%','Hidrología',100),
  ('x_hid_reserva_util_kwh','Reserva útil','GWh','Hidrología',1e-6),('x_hid_capacidad_util_kwh','Capacidad útil','GWh','Hidrología',1e-6),('x_hid_aportes_kwh_dia','Aportes hídricos','GWh/día','Hidrología',1e-6)]
 cities={'bogota':'Bogotá','medellin':'Medellín','cali':'Cali','barranquilla':'Barranquilla','cartagena':'Cartagena','bucaramanga':'Bucaramanga','pereira':'Pereira','cucuta':'Cúcuta'}
 for part,title,unit in [('temperature_2m_c','Temperatura','°C'),('relative_humidity_2m_pct','Humedad','%'),('precipitation_preceding_hour_mm','Precipitación','mm/h')]:
  for city,name in cities.items():spec.append(('x_obs_'+part+'_'+city,title+' · '+name,unit,'Clima',1))
 for col,name,unit,group,factor in spec:
  s=panel[col].resample('D').mean()*factor
  cols.append(dict(id=col,name=name,unit=unit,group=group,values=[round(float(v),5) if pd.notna(v) else None for v in s]))
 calendars={}
 for year,g in panel.groupby(panel.index.year):
  profile=(g.y_demanda_real_kwh/1e6).groupby([g.index.day_of_week,g.index.hour]).agg(['mean','count'])
  calendars[str(year)]={str(day):dict(mean=np.round(profile.loc[day,'mean'],5).tolist(),count=profile.loc[day,'count'].astype(int).tolist()) for day in range(7)}
 write('series.json',dict(dates=daily.index.strftime('%Y-%m-%d').tolist(),columns=cols,calendar=calendars))
 corpus=pd.read_parquet(R/'entradas/texto/corpus_oficial.parquet');corpus['year']=pd.to_datetime(corpus.disponible_desde,utc=True).dt.year
 cov=corpus.groupby(['year','fuente']).size().rename('documents').reset_index();orig=pd.read_csv(R/'entradas/texto/cobertura_original.csv')
 orig['year']=pd.to_datetime(orig.dia).dt.year;orig=orig[orig.year.le(2025)]
 annual=orig.groupby('year').agg(days=('dia','size'),covered=('documentos_usados',lambda a:int(a.gt(0).sum())),mean_used=('documentos_usados','mean')).reset_index()
 write('corpus.json',dict(coverage=rows(cov),windows=rows(annual),documents=len(corpus),unique_content=int(pd.read_parquet(R/'entradas/texto/indice_encoder.parquet').sha_contenido.nunique()),
  headline_only=int(corpus.texto.fillna('').str.strip().eq('').sum()),window_hours=72,max_documents=64))
 assets=SITE/'architectures';assets.mkdir(exist_ok=True)
 for f in set(FAMILY.values()):
  for kind in ['global','detalle']:shutil.copy2(ROOT/'book/_static/arquitecturas_presentacion'/f'{f}_{kind}.png',assets/f'{f}_{kind}.png')
 downloads=SITE/'downloads';downloads.mkdir(exist_ok=True)
 for file in ['benchmark','comparaciones_temporales','comparaciones_semillas','contrastes_texto','resultados_horizonte','diseno_seleccion']:shutil.copy2(R/(file+'.csv'),downloads/(file+'.csv'))
 write('manifest.json',dict(procedures=13,seeds=65,origins=1461,horizon=24,history=168,training_executed=False,
  source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [R/'benchmark.csv',R/'perdidas_y_bootstrap.npz',ROOT/'data/processed/panel_horario.parquet']}))
 print('13 procedimientos y 65 realizaciones exportados; métricas verificadas; sin entrenamiento.')

if __name__=='__main__':main()

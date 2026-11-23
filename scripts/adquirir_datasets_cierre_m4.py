"""Adquisición reproducible; selección por fuentes/tiempo, nunca por desempeño."""
from pathlib import Path
import hashlib, json, urllib.request, zipfile, io, shutil
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/external/m4_cierre_20261123'
FUENTES={
 'uci':('https://archive.ics.uci.edu/static/public/321/electricityloaddiagrams20112014.zip','uci.zip'),
 'gefcom':('https://www.dropbox.com/scl/fi/zwnlfwhds3k2xz0/GEFCom2014.zip?rlkey=oz7unehbwtglp1pbgbnb4wne2&dl=1','gefcom.zip'),
 'opsd':('https://data.open-power-system-data.org/time_series/2020-10-06/time_series_60min_singleindex.csv','opsd.csv'),
 'opsd_metadata':('https://data.open-power-system-data.org/time_series/2020-10-06/datapackage.json','opsd_datapackage.json')}

def descargar():
 OUT.mkdir(parents=True,exist_ok=True);filas=[]
 for key,(url,nombre) in FUENTES.items():
  p=OUT/nombre
  if not p.exists():
   tmp=p.with_suffix('.part')
   with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'Research reproducibility/1.0'}),timeout=120) as r,tmp.open('wb') as f:shutil.copyfileobj(r,f)
   tmp.replace(p)
  with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
  filas.append(dict(dataset=key,url=url,archivo=nombre,bytes=p.stat().st_size,sha256=h))
  (OUT/'FUENTES.json').write_text(json.dumps(filas,indent=2))
  print(key,p.stat().st_size,flush=True)
 return filas

def preparar():
 import pandas as pd, numpy as np
 # Etiquetas UCI al final de cada intervalo de 15 minutos: convertir a inicio,
 # sumar kWh de los 370 clientes y cuatro cuartos por hora. Mantener reloj
 # portugués de 96 muestras/día tal como lo entrega UCI, con su limitación DST.
 with zipfile.ZipFile(OUT/'uci.zip') as z:
  nombre=next(n for n in z.namelist() if n.endswith('LD2011_2014.txt') and not n.startswith('__MACOSX'))
  partes=[]
  for c in pd.read_csv(z.open(nombre),sep=';',decimal=',',chunksize=10000,index_col=0):
   partes.append(pd.Series(c.sum(axis=1).to_numpy()/4,index=pd.to_datetime(c.index)-pd.Timedelta(minutes=15)))
  u=pd.concat(partes).resample('h').sum(min_count=4).loc['2012-01-01':'2014-12-31']
  assert len(u)==26304 and u.notna().all()
  pd.DataFrame({'y':u}).to_parquet(OUT/'uci.parquet')
 # GEFCom2014-L, no la variante E procedente de ISO-NE.
 with zipfile.ZipFile(OUT/'gefcom.zip') as z:
  nombre=next(n for n in z.namelist() if n.endswith('GEFCom2014-L_V2.zip'))
  with zipfile.ZipFile(io.BytesIO(z.read(nombre))) as zz:
   partes=[]
   nombres=[next(n for n in zz.namelist() if n.endswith(f'/L{i}-train.csv')) for i in range(1,16)]
   nombres+=[next(n for n in zz.namelist() if n.endswith('/solution15_L.csv'))]
   siguiente=pd.Timestamp('2001-01-01 01:00')
   for n in nombres:
    df=pd.read_csv(zz.open(n));df.columns=df.columns.str.lower()
    fechas=pd.date_range(siguiente,periods=len(df),freq='h')
    codificados=[f'{d.month}{d.day}{d.year} {d.hour}:00' for d in fechas]
    assert codificados==df.timestamp.tolist(),'La cronología no reproduce cada marca original: '+n
    siguiente=fechas[-1]+pd.Timedelta(hours=1)
    df['timestamp']=fechas-pd.Timedelta(hours=1)
    partes.append(df[['timestamp','load']].dropna())
   g=pd.concat(partes);g=g.drop_duplicates('timestamp').set_index('timestamp').sort_index()['load']
   assert not g.index.duplicated().any();g=g.asfreq('h')
   assert g.notna().all();pd.DataFrame({'y':g}).to_parquet(OUT/'gefcom.parquet')
 op=pd.read_csv(OUT/'opsd.csv',usecols=['utc_timestamp','DE_load_actual_entsoe_transparency'])
 op.index=pd.to_datetime(op.pop('utc_timestamp'),utc=True).dt.tz_localize(None)
 d=op.iloc[:,0].loc['2015-01-01':'2019-12-31'].asfreq('h')
 # No rellenar etiquetas de prueba/validación. Se descartan los días con huecos
 # y ventanas afectadas en el cargador. Las entradas ausentes usan máscaras.
 pd.DataFrame({'y':d}).to_parquet(OUT/'opsd.parquet')
 met=[]
 for key in ['uci','gefcom','opsd']:
  t=pd.read_parquet(OUT/f'{key}.parquet')
  with (OUT/f'{key}.parquet').open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
  met.append(dict(dataset=key,inicio=str(t.index.min()),fin=str(t.index.max()),filas=len(t),ausentes=int(t.y.isna().sum()),sha256=h))
 (OUT/'PANELES.json').write_text(json.dumps(met,indent=2));print(json.dumps(met),flush=True)

if __name__=='__main__':descargar();preparar()

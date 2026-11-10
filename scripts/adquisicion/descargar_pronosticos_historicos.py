"""Descarga reproducible de Previous Runs; sus fechas de emisión son proxies."""
from pathlib import Path
import gzip
import hashlib
import json
import time
import numpy as np
import pandas as pd
import requests

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'data/external/pronosticos_temperatura'
URL='https://previous-runs-api.open-meteo.com/v1/forecast'
FIN='2026-11-09'
VARIABLES={'temperature_2m':'temperature_2m_c',
    'relative_humidity_2m':'relative_humidity_2m_pct','precipitation':'precipitation_mm'}
INICIOS={'gfs_global':'2021-03-01','jma_gsm':'2018-01-01'}


def pedir(params,ruta):
    meta=ruta.with_suffix('.meta.json')
    if ruta.exists() and meta.exists():
        raw=gzip.decompress(ruta.read_bytes())
        m=json.loads(meta.read_text(encoding='utf-8'))
        assert hashlib.sha256(raw).hexdigest()==m['sha256_respuesta']
        assert m['params']==params
        return json.loads(raw),m
    for intento in range(5):
        r=requests.get(URL,params=params,timeout=100)
        if r.status_code==200:break
        if r.status_code==429 or r.status_code>=500:
            time.sleep(min(10*(intento+1),40));continue
        raise RuntimeError(f'{r.status_code}: {r.text[:350]}')
    else:raise RuntimeError(f'Descarga agotada: {r.status_code}')
    j=r.json()
    assert 'hourly' in j
    ruta.parent.mkdir(parents=True,exist_ok=True)
    ruta.write_bytes(gzip.compress(r.content,mtime=0))
    m=dict(url=r.url,params=params,recuperado_utc=pd.Timestamp.now(tz='UTC').isoformat(),
        sha256_respuesta=hashlib.sha256(r.content).hexdigest(),bytes_respuesta=len(r.content),
        sha256_archivo=hashlib.sha256(ruta.read_bytes()).hexdigest(),
        archivo=str(ruta.relative_to(ROOT)).replace('\\','/'),
        lat_malla=j['latitude'],lon_malla=j['longitude'],elevacion_m=j['elevation'])
    meta.write_text(json.dumps(m,ensure_ascii=False,indent=2),encoding='utf-8')
    return j,m


def main():
    OUT.mkdir(exist_ok=True,parents=True)
    ciudades=pd.read_csv(ROOT/'data/raw/numeric/ciudades.csv')
    partes=[];manifiesto=[]
    hourly=','.join(f'{v}_previous_day{n}' for v in VARIABLES for n in (1,2,3))
    for modelo,inicio in INICIOS.items():
        for ciudad in ciudades.itertuples():
            for anio in range(pd.Timestamp(inicio).year,2027):
                ini=max(pd.Timestamp(inicio),pd.Timestamp(f'{anio}-01-01'))
                fin=min(pd.Timestamp(FIN),pd.Timestamp(f'{anio}-12-31'))
                params=dict(latitude=float(ciudad.lat_solicitada),longitude=float(ciudad.lon_solicitada),
                    hourly=hourly,models=modelo,start_date=f'{ini:%Y-%m-%d}',end_date=f'{fin:%Y-%m-%d}',timezone='America/Bogota')
                ruta=OUT/'originales'/f'{ciudad.ciudad}_{modelo}_{anio}.json.gz'
                cache=ruta.exists()
                j,m=pedir(params,ruta);manifiesto.append(m)
                h=j['hourly'];fechas=pd.to_datetime(h['time']).tz_localize('America/Bogota')
                esperado=pd.date_range(ini,fin+pd.Timedelta(hours=23),freq='h',tz='America/Bogota')
                assert fechas.equals(esperado)
                for lead in (1,2,3):
                    d=pd.DataFrame({'valido':fechas,'ciudad':ciudad.ciudad,'modelo':modelo,'lead_dias':lead})
                    for var,col in VARIABLES.items():
                        d[col]=pd.Series(h[f'{var}_previous_day{lead}'],dtype='float32')
                    d['emision_estimada']=d.valido-pd.Timedelta(days=lead)
                    d['disponible_estimado']=d.emision_estimada+pd.Timedelta(hours=12)
                    partes.append(d)
                print(f'{modelo} {ciudad.ciudad} {anio}: {len(fechas)} horas',flush=True)
                if not cache:time.sleep(3)
    d=pd.concat(partes,ignore_index=True).sort_values(['modelo','ciudad','lead_dias','valido'])
    claves=['modelo','ciudad','lead_dias','valido']
    assert not d.duplicated(claves).any()
    negativos=d.precipitation_mm.lt(0)
    d.loc[negativos,claves+['precipitation_mm']].to_csv(OUT/'precipitacion_negativa_proveedor.csv',index=False)
    d['flag_precipitation_negativa']=negativos
    # Los originales mantienen el valor recibido; el dato físicamente inválido
    # queda ausente en la tabla de entradas, sin reemplazarlo por lluvia cero.
    d.loc[negativos,'precipitation_mm']=np.nan
    d.to_parquet(OUT/'pronosticos_meteorologicos.parquet',index=False,compression='zstd')
    cols=['valido','ciudad','lead_dias','temperature_2m_c','modelo','emision_estimada','disponible_estimado']
    d.loc[d.temperature_2m_c.notna(),cols].to_parquet(OUT/'pronosticos_temperatura.parquet',index=False,compression='zstd')
    cobertura=[]
    for (modelo,ciudad,lead),g in d.groupby(['modelo','ciudad','lead_dias']):
        for var,col in VARIABLES.items():
            valido=g[col].notna();v=g.loc[valido,'valido']
            cobertura.append(dict(modelo=modelo,ciudad=ciudad,lead_dias=lead,variable=col,
                inicio=str(v.min()) if len(v) else None,fin=str(v.max()) if len(v) else None,
                horas_solicitadas=len(g),horas_validas=int(valido.sum()),faltantes=int((~valido).sum()),
                faltantes_entre_extremos=int(g.loc[g.valido.between(v.min(),v.max()),col].isna().sum()) if len(v) else None))
    pd.DataFrame(cobertura).to_csv(OUT/'cobertura.csv',index=False)
    diferencias=[]
    for (modelo,ciudad),g in d.groupby(['modelo','ciudad']):
        for col in VARIABLES.values():
            p=g.pivot(index='valido',columns='lead_dias',values=col)
            comun=p[[1,2]].dropna()
            diferencias.append(dict(modelo=modelo,ciudad=ciudad,variable=col,
                horas_comparables=len(comun),fraccion_leads_distintos=float((comun[1]!=comun[2]).mean()) if len(comun) else None))
    pd.DataFrame(diferencias).to_csv(OUT/'diferencias_entre_leads.csv',index=False)
    reporte=dict(filas=len(d),claves_duplicadas=0,ciudades=len(ciudades),modelos=list(INICIOS),leads=[1,2,3],
        fin_solicitado=FIN,fecha_descarga='2026-11-10',emision_exacta=False,
        regla='emision_estimada=valido-lead*24h; disponibilidad_proxy=emision_estimada+12h',
        advertencia='Previous Runs no identifica el ciclo exacto ni acredita su publicación. No completar faltantes con ERA5 futuro.',
        precipitaciones_negativas_proveedor=int(negativos.sum()),
        tratamiento_precipitacion_negativa='Conservar original y CSV de anomalías; NaN con bandera en tabla procesada, sin imputación',
        documentacion='https://open-meteo.com/en/docs/previous-runs-api',peticiones=manifiesto,
        archivos={p.name:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
            for p in OUT.glob('*.parquet')})
    (OUT/'manifest.json').write_text(json.dumps(reporte,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in reporte.items() if k not in ('peticiones',)},ensure_ascii=False),flush=True)


if __name__=='__main__':main()

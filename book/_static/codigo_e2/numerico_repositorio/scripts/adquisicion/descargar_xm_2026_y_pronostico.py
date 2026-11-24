"""Extensión separada de XM2026 y pronóstico oficial público de SIMEM."""
from pathlib import Path
import gzip,hashlib,json,time
import pandas as pd
import requests

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'data/external/actualizacion_2026'
FORE=ROOT/'data/external/pronostico_demanda_xm'
FIN=pd.Timestamp('2026-11-09')


def pedir(url,archivo,params=None,body=None):
    archivo.parent.mkdir(parents=True,exist_ok=True)
    meta=archivo.with_suffix('.meta.json')
    if archivo.exists() and meta.exists():
        raw=gzip.decompress(archivo.read_bytes());m=json.loads(meta.read_text(encoding='utf-8'))
        assert hashlib.sha256(raw).hexdigest()==m['sha256_respuesta']
        return json.loads(raw),m
    for i in range(4):
        r=requests.post(url,json=body,timeout=80) if body is not None else requests.get(url,params=params,timeout=80)
        if r.status_code==200:break
        if r.status_code==429 or r.status_code>=500:time.sleep(5*(i+1))
        else:r.raise_for_status()
    r.raise_for_status()
    archivo.write_bytes(gzip.compress(r.content,mtime=0))
    m=dict(url=r.url,params=params,body=body,recuperado_utc=pd.Timestamp.now(tz='UTC').isoformat(),
        sha256_respuesta=hashlib.sha256(r.content).hexdigest(),sha256_archivo=hashlib.sha256(archivo.read_bytes()).hexdigest(),
        archivo=str(archivo.relative_to(ROOT)).replace('\\','/'))
    meta.write_text(json.dumps(m,ensure_ascii=False,indent=2),encoding='utf-8')
    return r.json(),m


def meses(inicio):
    for ini in pd.date_range(inicio,FIN,freq='MS'):
        yield ini,min(ini+pd.offsets.MonthEnd(0),FIN)


def descargar_xm():
    variables={'DemaReal':('hourly','demanda_real_kwh'),'DemaCome':('hourly','demanda_comercial_kwh'),
        'PrecBolsNaci':('hourly','precio_bolsa_cop_kwh'),'VoluUtilDiarEner':('daily','reserva_util_kwh'),
        'CapaUtilDiarEner':('daily','capacidad_util_kwh'),'AporEner':('daily','aportes_kwh_dia')}
    horarias=[];diarias=[];manifest=[]
    for metrica,(tipo,col) in variables.items():
        filas=[]
        for ini,fin in meses('2026-01-01'):
            body=dict(MetricId=metrica,Entity='Sistema',StartDate=str(ini.date()),EndDate=str(fin.date()),Filter=[])
            j,m=pedir('https://servapibi.xm.com.co/'+tipo,OUT/'xm/originales'/f'{metrica}_{ini:%Y%m}.json.gz',body=body)
            manifest.append(m)
            for item in j['Items']:
                fecha=pd.Timestamp(item['Date']).tz_localize('America/Bogota')
                for ent in item['HourlyEntities' if tipo=='hourly' else 'DailyEntities']:
                    assert ent['Id']=='Sistema'
                    if tipo=='hourly':
                        for h in range(1,25):
                            valor=ent['Values'].get(f'Hour{h:02d}')
                            filas.append({'inicio':fecha+pd.Timedelta(hours=h-1),col:pd.to_numeric(valor,errors='coerce')})
                    else:filas.append({'fecha':fecha,col:pd.to_numeric(ent.get('Value'),errors='coerce')})
            time.sleep(.3)
        d=pd.DataFrame(filas).set_index('inicio' if tipo=='hourly' else 'fecha')
        assert d.index.is_unique
        (horarias if tipo=='hourly' else diarias).append(d)
        print('XM',metrica,len(d),flush=True)
    h=pd.concat(horarias,axis=1).reindex(pd.date_range('2026-01-01',FIN+pd.Timedelta(hours=23),freq='h',tz='America/Bogota'))
    h.index.name='inicio'
    d=pd.concat(diarias,axis=1).reindex(pd.date_range('2026-01-01',FIN,freq='D',tz='America/Bogota'))
    d.index.name='fecha';d['llenado_pct']=100*d.reserva_util_kwh/d.capacidad_util_kwh
    h.to_parquet(OUT/'xm/horario_2026.parquet',compression='zstd')
    d.to_parquet(OUT/'xm/hidrologia_2026.parquet',compression='zstd')
    (OUT/'xm/manifest.json').write_text(json.dumps(dict(filas_horarias=len(h),filas_diarias=len(d),
        nulos_horarios=h.isna().sum().to_dict(),nulos_diarios=d.isna().sum().to_dict(),
        ultimos_validos={c:str(h[c].last_valid_index()) for c in h},originales=manifest,
        nota='Extensión recuperada el10nov2026, retrospectiva. No reemplaza el panel congelado; cifras revisables y horas de publicación no certificadas.'),ensure_ascii=False,indent=2),encoding='utf-8')


def descargar_pronostico():
    FORE.mkdir(parents=True,exist_ok=True)
    partes=[];manifest=[]
    for ini,fin in meses('2022-01-01'):
        params=dict(datasetId='63d90d',startDate=str(ini.date()),endDate=str(fin.date()))
        j,m=pedir('https://www.simem.co/backend-files/api/PublicData',FORE/'originales'/f'oficial_{ini:%Y%m}.json.gz',params=params)
        manifest.append(m)
        records=j.get('result',{}).get('records',[])
        if isinstance(records,str):records=json.loads(records)
        if records:
            d=pd.DataFrame(records);partes.append(d)
        print('Pronóstico XM',ini.strftime('%Y-%m'),len(records),flush=True)
        time.sleep(.3)
    d=pd.concat(partes,ignore_index=True)
    d['FechaHora']=pd.to_datetime(d.FechaHora).dt.tz_localize('America/Bogota')
    d['FechaPublicacion']=pd.to_datetime(d.FechaPublicacion).dt.tz_localize('America/Bogota')
    claves=['FechaHora','CodigoSubAreaOperativa','FechaPublicacion','CodigoDuracion']
    duplicados=int(d.duplicated(claves).sum())
    assert not duplicados
    d['disponible_declarado_simem']=d.FechaPublicacion+pd.Timedelta(days=1)
    d.to_parquet(FORE/'pronostico_oficial_subareas_2022_2026.parquet',index=False,compression='zstd')
    cobertura=d.groupby([d.FechaHora.dt.year,'CodigoSubAreaOperativa']).agg(
        filas=('PronDem','size'),inicio=('FechaHora','min'),fin=('FechaHora','max'))
    cobertura.to_csv(FORE/'cobertura_anual_subarea.csv')
    publicable=(d.disponible_declarado_simem<=d.FechaHora)
    (FORE/'manifest.json').write_text(json.dumps(dict(filas=len(d),duplicados=duplicados,
        inicio=str(d.FechaHora.min()),fin=str(d.FechaHora.max()),subareas=int(d.CodigoSubAreaOperativa.nunique()),
        fecha_emision_original_identificada=False,publicaciones_no_posteriores_al_objetivo=int(publicable.sum()),
        advertencia='FechaPublicacion es la publicación en SIMEM, no la emisión del pronóstico. No habilitar como entrada o comparador operativo sin verificar emisión y versiones anteriores al origen.',
        originales=manifest),ensure_ascii=False,indent=2),encoding='utf-8')


def descargar_clima_2026():
    ciudades=pd.read_csv(ROOT/'data/raw/numeric/ciudades.csv');partes=[];manifest=[]
    for c in ciudades.itertuples():
        params=dict(latitude=c.lat_solicitada,longitude=c.lon_solicitada,models='era5',
            hourly='temperature_2m,relative_humidity_2m,precipitation',start_date='2026-01-01',end_date=str(FIN.date()),timezone='America/Bogota')
        j,m=pedir('https://archive-api.open-meteo.com/v1/archive',OUT/'clima/originales'/f'{c.ciudad}_era5_2026.json.gz',params=params)
        manifest.append(m);h=j['hourly']
        d=pd.DataFrame(h).rename(columns={'time':'inicio','temperature_2m':'temperature_2m_c',
            'relative_humidity_2m':'relative_humidity_2m_pct','precipitation':'precipitation_mm'})
        d.inicio=pd.to_datetime(d.inicio).dt.tz_localize('America/Bogota');d['ciudad']=c.ciudad
        partes.append(d)
        time.sleep(1)
    d=pd.concat(partes,ignore_index=True)
    assert not d.duplicated(['inicio','ciudad']).any()
    d.to_parquet(OUT/'clima/clima_observado_2026.parquet',index=False,compression='zstd')
    cols=['temperature_2m_c','relative_humidity_2m_pct','precipitation_mm']
    cob=d.groupby('ciudad').agg(filas=('inicio','size'),inicio=('inicio','min'),fin=('inicio','max'))
    for col in cols:cob[col+'_nulos']=d.groupby('ciudad')[col].apply(lambda x:int(x.isna().sum()))
    cob.to_csv(OUT/'clima/cobertura.csv')
    (OUT/'clima/manifest.json').write_text(json.dumps(dict(filas=len(d),originales=manifest,
        ultimos_validos={c:str(d.loc[d.ciudad.eq(c)&d[cols].notna().all(axis=1),'inicio'].max()) for c in ciudades.ciudad},
        advertencia='Reanálisis ERA5; las horas recientes aún no publicadas permanecen nulas. No usar como pronóstico futuro.'),ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':
    descargar_xm()
    descargar_pronostico()
    descargar_clima_2026()

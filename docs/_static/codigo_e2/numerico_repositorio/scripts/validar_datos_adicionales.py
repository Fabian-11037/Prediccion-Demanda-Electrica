"""Audita los nuevos datos sin construir ventanas de etiquetas ni ajustar modelos."""
from pathlib import Path
import gzip,hashlib,json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
EX=ROOT/'data/external'
OUT=ROOT/'results/datos_adicionales_20261110'


def hash_archivo(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    OUT.mkdir(exist_ok=True,parents=True)
    ciu=pd.read_csv(ROOT/'data/raw/numeric/ciudades.csv').ciudad.tolist()
    f=pd.read_parquet(EX/'pronosticos_temperatura/pronosticos_meteorologicos.parquet')
    assert len(f)==3061440 and not f.duplicated(['modelo','ciudad','lead_dias','valido']).any()
    variables=['temperature_2m_c','relative_humidity_2m_pct','precipitation_mm']
    for (modelo,ciudad,lead),g in f.groupby(['modelo','ciudad','lead_dias']):
        assert pd.DatetimeIndex(g.valido).equals(pd.date_range(g.valido.iloc[0],g.valido.iloc[-1],freq='h'))
    assert f.relative_humidity_2m_pct.dropna().between(0,100).all()
    assert f.precipitation_mm.dropna().ge(0).all()
    assert np.isfinite(f[variables].dropna().to_numpy()).all()
    diff=pd.read_csv(EX/'pronosticos_temperatura/diferencias_entre_leads.csv')
    assert diff.loc[diff.variable.eq('temperature_2m_c'),'fraccion_leads_distintos'].gt(.01).all()
    cobertura=[]
    cortes=[('entrenamiento_referencia','2000-01-01','2017-12-31'),('validacion_externa','2018-01-01','2021-12-31'),
            ('prueba_retrospectiva','2022-01-01','2025-12-31'),('extension_retrospectiva','2026-01-01','2026-11-09')]
    for modelo,d in f.groupby('modelo'):
        horas=pd.date_range(d.valido.min(),d.valido.max(),freq='h')
        origenes=pd.date_range(horas.min().normalize(),horas.max().normalize(),freq='D')
        for etiqueta,cols in [('temperatura',['temperature_2m_c']),('temperatura_humedad_precipitacion',variables)]:
            util=np.zeros((len(origenes),24,8),dtype=bool)
            for lead,g in d.groupby('lead_dias'):
                g=g.assign(completo=g[cols].notna().all(axis=1))
                m=g.pivot(index='valido',columns='ciudad',values='completo').reindex(index=horas,columns=ciu).fillna(False)
                mascara=m.to_numpy(dtype=bool).reshape(len(origenes),24,8)
                util |= mascara & ((np.arange(24)+12)<=lead*24)[None,:,None]
            completo=util.all(axis=(1,2))
            for particion,inicio,fin in cortes:
                esperado=pd.date_range(inicio,fin,freq='D',tz='America/Bogota')
                disponible=pd.Series(completo,index=origenes).reindex(esperado,fill_value=False)
                fechas=disponible.index[disponible]
                cobertura.append(dict(modelo=modelo,variables=etiqueta,particion=particion,
                    origenes=len(esperado),origenes_24h_8ciudades_completos=int(disponible.sum()),
                    cobertura_pct=float(100*disponible.mean()),primer_origen_completo=str(fechas.min()) if len(fechas) else None,
                    ultimo_origen_completo=str(fechas.max()) if len(fechas) else None,
                    regla='Ultimo lead1..3 no nulo que cumple disponibilidad proxy=valido-lead*24h+12h<=origen'))
    pd.DataFrame(cobertura).to_csv(OUT/'cobertura_pronosticos_por_particion.csv',index=False)
    ise=pd.read_csv(EX/'actividad_economica/ise_dane.csv')
    assert len(ise)==260 and not ise.mes.duplicated().any()
    assert ise.mes.tolist()==pd.period_range('2005-01','2026-08',freq='M').astype(str).tolist()
    assert not ise[['ise_original','ise_desestacionalizado','ise_tendencia_ciclo']].isna().any().any()
    iv=pd.read_parquet(EX/'actividad_economica/ise_vintages.parquet')
    assert not iv.duplicated(['archivo_fuente','mes']).any()
    im=json.loads((EX/'actividad_economica/manifest.json').read_text(encoding='utf-8'))
    assert im['anexos_descargados']==im['anexos_identificados']==168 and not im['errores']
    pop=pd.read_csv(EX/'poblacion_dane/poblacion_ciudades_2000_2026.csv')
    assert len(pop)==216 and not pop.duplicated(['ciudad','anio']).any()
    assert np.allclose(pop.groupby('anio').peso_total_8_ciudades.sum(),1)
    assert (pop.total==pop.cabecera+pop.rural).all()
    xm=pd.read_parquet(EX/'actualizacion_2026/xm/horario_2026.parquet')
    hid=pd.read_parquet(EX/'actualizacion_2026/xm/hidrologia_2026.parquet')
    assert len(xm)==7512 and xm.index.is_unique and len(hid)==313 and hid.index.is_unique
    assert not hid.isna().any().any() and hid.llenado_pct.between(0,100).all()
    clima=pd.read_parquet(EX/'actualizacion_2026/clima/clima_observado_2026.parquet')
    assert len(clima)==60096 and not clima.duplicated(['ciudad','inicio']).any()
    pr=pd.read_parquet(EX/'pronostico_demanda_xm/pronostico_oficial_subareas_2022_2026.parquet')
    assert len(pr)==762480 and not pr.duplicated(['FechaHora','CodigoSubAreaOperativa','FechaPublicacion','CodigoDuracion']).any()
    assert pr.CodigoDuracion.eq('PT1H').all()
    assert pr.disponible_declarado_simem.gt(pr.FechaHora).all()
    comprobados=0
    for carpeta in ['pronosticos_temperatura','actividad_economica','poblacion_dane','actualizacion_2026','pronostico_demanda_xm']:
        for p in (EX/carpeta).rglob('*.json'):
            if p.name=='manifest.json':continue
            m=json.loads(p.read_text(encoding='utf-8'))
            if not isinstance(m,dict) or 'archivo' not in m:continue
            raw=ROOT/m['archivo']
            if 'sha256_archivo' in m:assert hash_archivo(raw)==m['sha256_archivo']
            if 'sha256_respuesta' in m:assert hashlib.sha256(gzip.decompress(raw.read_bytes())).hexdigest()==m['sha256_respuesta']
            if 'sha256' in m:assert hash_archivo(raw)==m['sha256']
            comprobados+=1
    previo=json.loads((ROOT/'results/documentos_oficiales/estado_previo.json').read_text(encoding='utf-8'))
    conservados=[]
    for nombre,valor in previo.items():
        if nombre.startswith('data/'):
            sha=valor['sha256'] if isinstance(valor,dict) else valor
            assert hash_archivo(ROOT/nombre)==sha,nombre
            conservados.append(nombre)
    archivos=[EX/'pronosticos_temperatura/pronosticos_meteorologicos.parquet',EX/'pronosticos_temperatura/pronosticos_temperatura.parquet',
        EX/'actividad_economica/ise_dane.csv',EX/'actividad_economica/ise_vintages.parquet',
        EX/'poblacion_dane/poblacion_ciudades_2000_2026.csv',EX/'actualizacion_2026/xm/horario_2026.parquet',
        EX/'actualizacion_2026/xm/hidrologia_2026.parquet',EX/'actualizacion_2026/clima/clima_observado_2026.parquet',
        EX/'pronostico_demanda_xm/pronostico_oficial_subareas_2022_2026.parquet',ROOT/'data/processed/corpus_texto.parquet']
    manifest=dict(version='datos_adicionales_20261110',fecha='2026-11-10',
        panel_numerico_congelado='data/processed/panel_horario.parquet',
        archivos=[dict(ruta=str(p.relative_to(ROOT)).replace('\\','/'),bytes=p.stat().st_size,sha256=hash_archivo(p)) for p in archivos],
        entrenamiento_ejecutado=False,configuracion_modelos_modificada=False,
        disponibilidad=dict(meteorologia='proxy explicito; emision exacta no certificada',
            ise='vintage declarado+dia siguiente y rezago60dias; no mezclar bases',
            poblacion='proyecciones actuales revisadas; fechas2023/2025',
            pronostico_xm='solo auditoria; publicacion posterior al objetivo, sin emision original',
            extension2026='retrospectiva; no es bloque prospectivo'),
        selectores='src/demanda/datos_adicionales.py')
    (ROOT/'data/processed/dataset_ampliado_20261110.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    report=dict(correcto=True,originales_con_sha256_verificado=comprobados,datos_congelados_intactos=conservados,
        filas_pronosticos=len(f),versiones_ise=im['versiones_interpretadas'],anexos_ise=168,
        anexos_ise_solo_original=im['versiones_sin_parser'],meses_ise_base2015=260,
        inicio_ise_versiones=iv.mes.min(),filas_poblacion=len(pop),filas_pronostico_demanda=len(pr),
        claves_duplicadas=0,entrenamiento=False,publicacion_jbook=False)
    (OUT/'validacion.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False))
    print(pd.DataFrame(cobertura).to_string(index=False))


if __name__=='__main__':main()

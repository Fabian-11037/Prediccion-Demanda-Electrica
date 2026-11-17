"""Cobertura, denominadores y separación de los controles de inferencia congelada."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import nbformat

ROOT=Path(__file__).resolve().parents[1]
O=ROOT/'results/diagnosticos_ampliados'

def test_importancia_conserva_semillas_fechas_y_repeticiones():
    p=json.loads((O/'protocolo.json').read_text(encoding='utf-8'))
    dates=pd.DatetimeIndex(p['fechas_importancia'])
    assert len(dates)==192 and dates.is_unique and set(dates.day)=={1,8,15,22}
    assert pd.Series(dates.to_period('M')).value_counts().eq(4).all()
    d=pd.read_csv(O/'importancia_semillas.csv')
    assert d.clave.nunique()==10 and len(d)==1350
    assert d.groupby(['clave','grupo']).size().eq(15).all()
    assert d.origenes.eq(192).all() and set(d.semilla)==set(range(1000,1005))
    assert set(d.grupo)==set(p['grupos'])
    assert np.allclose(d.delta_MASE,(d.MAE_perturbado-d.MAE_base)/p['q168'])
    assert np.allclose(d.cambio_pct,100*(d.MAE_perturbado/d.MAE_base-1))
    assert p['entrenamiento'] is False and p['seleccion'] is False and p['publicacion'] is False

def test_reproduccion_de_todos_los_pesos_y_latencia_comun():
    d=pd.read_csv(O/'reproduccion_semillas.csv');lat=pd.read_csv(O/'latencia_semillas.csv')
    assert len(d)==50 and d.groupby('clave').size().eq(5).all() and d.origenes.eq(1461).all()
    assert d.reproduccion_numerica.all() and d.max_diferencia_relativa.lt(.001).all() and d.cambio_relativo_MAE.lt(.005).all()
    assert d.checkpoint_sha256.str.len().eq(64).all()
    assert len(lat)==210 and lat.clave.nunique()==14 and lat.groupby(['clave','semilla']).size().eq(3).all()
    assert lat.repeticiones.eq(100).all() and lat.calentamiento.eq(20).all() and lat.hilos.eq(1).all()
    assert lat.mediana_ms.gt(0).all() and (lat.p95_ms>=lat.mediana_ms).all()
    assert lat.loc[~lat.clave.str.endswith('__N+O'),'documentos'].eq(0).all()
    tats=lat.clave.eq('adj_modelo6_20261112_1035__N+O')
    assert lat.loc[tats,'cierres_textuales'].eq(7).all()
    assert lat.loc[lat.clave.str.endswith('__N+O')&~tats,'cierres_textuales'].eq(1).all()

def test_placebo_textual_respeta_fuente_y_disponibilidad_historica():
    c=pd.read_parquet(ROOT/'results/evaluacion_e2/entradas/texto/corpus_oficial.parquet')
    dates=pd.to_datetime(c.disponible_desde,utc=True)
    for r in range(3):
        donor=np.load(O/f'donantes_texto_{r}.npy',allow_pickle=False)
        assert len(donor)==len(c) and (dates.iloc[donor]<pd.Timestamp('2022-01-01',tz='America/Bogota').tz_convert('UTC')).all()
        assert np.array_equal(c.fuente.to_numpy()[donor],c.fuente.to_numpy())
    s=pd.read_csv(O/'sensibilidad_semillas.csv');summary=pd.read_csv(O/'sensibilidad_resumen.csv')
    assert len(s)==115 and s.origenes.eq(1461).all() and len(summary)==13
    assert s.groupby(['clave','condicion']).semilla.nunique().eq(5).all()
    assert (summary.IC95_inf<=summary.IC95_sup).all()

def test_el_libro_presenta_el_alcance_actual_sin_ausencias_ficticias():
    n=nbformat.read(ROOT/'book/10_evaluacion.ipynb',4)
    text='\n'.join(c.source for c in n.cells if c.cell_type=='markdown')
    assert '## Importancia de variables en los modelos ampliados' in text
    assert '## Sensibilidad al contenido de los documentos oficiales' in text
    assert '### Descripciones estáticas de los canales en TGForecaster' in text
    assert 'conservan el alcance de las cuatro familias medidas' not in text
    coverage=pd.read_csv(ROOT/'results/evaluacion_e2/cobertura_diagnosticos.csv')
    assert coverage['Importancia y latencia CPU'].eq('Medidas').sum()==13
    assert coverage['Sensibilidad textual'].str.startswith('No aplica').sum()==9

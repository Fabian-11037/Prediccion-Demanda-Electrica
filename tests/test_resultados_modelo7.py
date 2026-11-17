"""TGForecaster entra con dos variantes reales y sus controles correctos."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/evaluacion_e2'
EID='adj_modelo7_20261114_0430'


def test_predicciones_completas_y_metricas_reproducidas():
    df=pd.read_parquet(OUT/'entradas/consolidado/predicciones_finales.parquet',filters=[('experimento','==',EID)])
    assert len(df)==350640 and not df.duplicated(['variante','semilla','origen','horizonte']).any()
    benchmark=pd.read_csv(OUT/'benchmark.csv').set_index('clave')
    for variant,group in df.groupby('variante'):
        assert sorted(group.semilla.unique())==list(range(1000,1005))
        for _,seed in group.groupby('semilla'):
            assert seed.origen.nunique()==1461 and seed.groupby('origen').size().eq(24).all()
        mase=np.abs(group.predicho_kwh-group.observado_kwh).mean()/215617.90361589016
        assert np.isclose(mase,benchmark.loc[EID+'__'+variant,'MASE'],rtol=1e-10)
    assert set(df.variante)=={'N','N+O'}


def test_familias_completas_y_control_textual_propio():
    protocol=json.loads((OUT/'protocolo.json').read_text(encoding='utf-8'))
    assert protocol['n_modelos']==13 and protocol['comparaciones_semillas']==78
    assert protocol['comparaciones_temporales']==105 and protocol['n_contrastes_textuales']==3
    text=pd.read_csv(OUT/'contrastes_texto.csv')
    row=text[text.A.eq(EID+'__N+O')].iloc[0]
    assert row.B==EID+'__N' and row.relativa_pct>0
    assert row.p_DM_Holm_texto<.05 and row.p_bootstrap_Holm_texto<.05
    with np.load(OUT/'perdidas_y_bootstrap.npz',allow_pickle=False) as z:
        assert z['semillas'].shape==(5,13) and z['loss'].shape==(1461,15)


def test_pesos_y_fuentes_conservados_sin_cargar_checkpoints():
    audit=json.loads((ROOT/'results/modelo7_integracion/importacion.json').read_text(encoding='utf-8'))
    assert audit['finales']==10 and not audit['checkpoints_cargados'] and not audit['entrenamiento']
    weights=[row for row in audit['archivos'] if row['ruta'].endswith('/modelo.pt')]
    assert len(weights)==10 and len(audit['fuentes_verificadas'])>=17
    consolidation=ROOT.parent/'Consolidado-Entrenamientos-20261115'
    for row in weights:
        assert hashlib.sha256((consolidation/row['ruta']).read_bytes()).hexdigest()==row['sha256']
    assert {tuple(x['shape']) for x in audit['vectores']}=={(14162,768),(33,768)}

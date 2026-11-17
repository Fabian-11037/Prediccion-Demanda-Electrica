"""Contratos de documentación e inferencia E2, sin entrenar modelos."""
import ast,json,importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
ROOT=Path(__file__).resolve().parents[1]

def test_todas_las_arquitecturas_tienen_esquema_y_pipeline():
    j=json.loads((ROOT/'results/arquitecturas_e2/indice_documentacion.json').read_text(encoding='utf-8'))
    assert len(j)==12
    for m in j:
        texto=(ROOT/'book/arquitecturas'/f"{m['key']}.md").read_text(encoding='utf-8')
        assert 'Pipeline de entrenamiento y prueba' in texto and 'literalinclude' in texto
        assert (ROOT/'book/_static/arquitecturas'/f"{m['key']}.svg").exists()
        ast.parse((ROOT/'book/_static/codigo_e2'/f"{m['key']}_red.py").read_text(encoding='utf-8'))

def test_resultados_tgforecaster_verificados():
    b=pd.read_csv(ROOT/'results/evaluacion_e2/benchmark.csv')
    assert b.clave.str.contains('modelo7').sum()==2
    assert len(b)==15 and sum(b.semillas.eq(5))==13
    assert 'repo_patchtst_numerico_r1__N' not in set(b.clave)
    retained=b.loc[b.clave.eq('adj_modelo6_20261112_1035__N')]
    assert len(retained)==1 and retained.MASE.iloc[0]==pytest.approx(0.6725016176243331)
    assert b.modelo.str.startswith('PatchTST').sum()==1

def test_referencias_y_familias_de_comparacion():
    b=pd.read_csv(ROOT/'results/evaluacion_e2/benchmark.csv').set_index('clave')
    assert b.loc['ingenuo_24h','MASE']==pytest.approx(2.0767245486389916)
    assert b.loc['ingenuo_168h','MASE']==pytest.approx(1.354906444254724)
    p=pd.read_csv(ROOT/'results/evaluacion_e2/comparaciones_temporales.csv')
    assert len(p)==105 and np.all(p.IC95_inf<=p.IC95_sup)
    assert np.all(p.p_DM_Holm>=p.p_DM)

def test_prueba_portable_requiere_autorizacion_antes_de_importar_torch():
    p=ROOT/'scripts/pipeline_ampliado_portable.py'
    spec=importlib.util.spec_from_file_location('portable_bajo_prueba',p)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    with pytest.raises(PermissionError):mod.ejecutar(ROOT,'patchtst',fase='prueba')

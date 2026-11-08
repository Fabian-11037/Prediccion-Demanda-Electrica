"""Contratos de resultados y redacción del informe, sin ajuste de modelos."""
import csv,json,re
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
RESULTADOS=ROOT/'results/jbook/modelos_actuales'
pytestmark=pytest.mark.skipif(not (RESULTADOS/'benchmark.csv').exists(),reason='Requiere artefactos del JBook')


def leer_csv(nombre):
    with (RESULTADOS/nombre).open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))


def narrativa(stem):
    n=json.loads((ROOT/'book'/(stem+'.ipynb')).read_text(encoding='utf-8'))
    return '\n'.join(''.join(c['source']) for c in n['cells'] if c['cell_type']=='markdown')


def test_se_conservan_cinco_semillas_por_familia():
    filas=leer_csv('metricas_desde_predicciones.csv')
    assert len(filas)==20
    for m in ['mlp','arx_lineal','cnn1d','lstm']:
        semillas=[int(r['semilla']) for r in filas if r['modelo']==m]
        assert sorted(semillas)==list(range(1000,1005))


def test_tost_usa_un_margen_relativo_y_referencia_comun():
    nombres={'MLP':'mlp','ARX':'arx_lineal','CNN1D':'cnn1d','LSTM':'lstm'}
    b={r['familia']:float(r['MASE']) for r in leer_csv('benchmark.csv')}
    for fila in leer_csv('equivalencia_TOST.csv'):
        assert float(fila['margen_rel_pct'])==3
        a,c=fila['comparacion'].split(' − ')
        esperado=100*(b[nombres[a]]-b[nombres[c]])/b['mlp']
        assert float(fila['diferencia_rel_pct'])==pytest.approx(esperado)
        assert 'margen_MASE' not in fila


def test_latencia_reportada_y_medida_en_mismos_modelos():
    assert 'la latencia de inferencia no fue medida' not in narrativa('10_modelos').lower()
    filas=leer_csv('latencia_inferencia.csv')
    assert len(filas)==20 and all(r['dispositivo']=='cpu' for r in filas)
    assert all(float(r['mediana_ms'])>0 and float(r['p95_ms'])>=float(r['mediana_ms']) for r in filas)


def test_numeracion_y_archivos_canonicos():
    assert narrativa('09_validacion').startswith('# 9.')
    assert narrativa('10_modelos').startswith('# 10.')
    assert narrativa('10_evaluacion').startswith('# 10.9.')
    assert not (ROOT/'book/08_validacion.ipynb').exists()
    assert not (ROOT/'book/09_modelos.ipynb').exists()


def test_no_se_reintroduce_margen_absoluto_en_documentacion():
    for stem in ['09_validacion','10_modelos','10_evaluacion']:
        texto=narrativa(stem)
        assert 'Se fija δ absoluto' not in texto
        assert 'margen absoluto de MASE' not in texto


def test_no_se_reporta_cero_como_probabilidad_adf_o_ljungbox():
    texto=narrativa('05_dependencia')
    assert not re.search(r'(?:ADF|Ljung[^\n]*?)\s+p\s*=\s*0(?![\d.])',texto)

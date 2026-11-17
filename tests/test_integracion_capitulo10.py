"""Integración documental sin resultados ni código duplicados."""
from pathlib import Path
import nbformat
ROOT=Path(__file__).resolve().parents[1]

def test_arquitecturas_antes_de_resultados_y_sin_capitulo_redundante():
    toc=(ROOT/'book/_toc.yml').read_text(encoding='utf-8')
    assert toc.index('arquitecturas/fundamentos')<toc.index('arquitecturas/arx')<toc.index('arquitecturas/fusion')<toc.index('file: 10_evaluacion')
    assert 'file: 10_pipeline' not in toc and 'file: 10_estadistica_e2' not in toc
    assert not (ROOT/'book/10_pipeline.md').exists()
    assert not (ROOT/'book/10_estadistica_e2.ipynb').exists()
    for p in (ROOT/'book/arquitecturas').glob('*.md'):
        assert '../10_pipeline' not in p.read_text(encoding='utf-8')

def test_evaluacion_preserva_analisis_complementarios_y_unifica_inferencia():
    n=nbformat.read(ROOT/'book/10_evaluacion.ipynb',4)
    t='\n'.join(c.source for c in n.cells if c.cell_type=='markdown')
    for s in ['Piloto','Demanda observada','Segmentos','Importancia de variables','atípicos','Friedman','confianza','Coste de inferencia']:
        assert s.casefold() in t.casefold()
    assert sum('## Desempeño y precisión temporal' in c.source for c in n.cells)==1
    assert '(observado-prediccion)=' in t and '(importancia-variables)=' in t
    assert 'p ajustado=0.8590' not in t

def test_codigo_compartido_preservado_al_eliminar_pipeline():
    t=(ROOT/'book/arquitecturas/fundamentos.md').read_text(encoding='utf-8')
    for s in ['Partición temporal y parada interna','Datos, ventanas, escalas y referencias','Métricas']:
        assert s in t
    assert '# 10.1.' in t

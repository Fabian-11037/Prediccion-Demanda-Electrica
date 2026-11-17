"""Evita que los resultados generales vuelvan a filtrar solo las variantes textuales."""
from pathlib import Path
from itertools import combinations
import nbformat
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/evaluacion_e2'


def test_parejas_completas_y_cinco_semillas_por_procedimiento():
    benchmark=pd.read_csv(OUT/'benchmark.csv')
    keys=set(benchmark.loc[benchmark.semillas.eq(5),'clave'])
    metadata=pd.read_csv(OUT/'metadatos_procedimientos.csv')
    assert set(metadata.clave)==keys and len(metadata)==5*len(keys)
    for _,group in metadata.groupby('clave'):
        assert sorted(group.semilla)==list(range(1000,1005))
    expected={frozenset(pair) for pair in combinations(keys,2)}
    for name in ['comparaciones_semillas.csv','nemenyi.csv']:
        pairs=pd.read_csv(OUT/name)
        assert len(pairs)==len(expected)
        assert {frozenset(pair) for pair in zip(pairs.A,pairs.B)}==expected


def test_tablas_visibles_no_filtran_semillas_ni_mcs():
    notebook=nbformat.read(ROOT/'book/10_evaluacion.ipynb',4)
    paircell=next(c for c in notebook.cells if c.cell_type=='code' and "g_pairs = pd.read_csv" in c.source)
    assert "objetivos =" not in paircell.source and len(paircell.outputs)>0
    html='\n'.join(o.get('data',{}).get('text/html','') for o in paircell.outputs)
    assert html.count('<tr>')==len(pd.read_csv(OUT/'comparaciones_semillas.csv'))+1
    mcscell=next(c for c in notebook.cells if c.cell_type=='code' and 'g_mcs=pd.read_csv' in c.source)
    html='\n'.join(o.get('data',{}).get('text/html','') for o in mcscell.outputs)
    assert html.count('<tr>')==len(pd.read_csv(OUT/'mcs.csv'))+1


def test_ausencias_se_documentan_sin_inventar_resultados():
    coverage=pd.read_csv(OUT/'cobertura_diagnosticos.csv').set_index('Procedimiento')
    assert len(coverage)==13
    assert coverage.index.str.contains('TGForecaster').sum()==2
    assert coverage['Importancia y latencia CPU'].eq('Medidas').sum()==13
    assert coverage['Piloto independiente'].eq('Disponible').sum()==5
    metadata=pd.read_csv(OUT/'metadatos_procedimientos.csv')
    repo=metadata[metadata.clave.str.startswith('repo_')]
    assert repo[['MASE_ajuste','MASE_parada','MAE_ajuste_normalizado','MAE_parada_normalizado']].isna().all().all()


def test_seleccion_y_dimensionamiento_incluyen_los_procedimientos_ampliados():
    design=pd.read_csv(OUT/'diseno_seleccion.csv')
    benchmark=pd.read_csv(OUT/'benchmark.csv')
    assert set(design.clave)==set(benchmark.loc[benchmark.semillas.eq(5),'clave'])
    summary=pd.read_csv(OUT/'resumen_piloto_integrado.csv')
    assert len(summary)==13
    absent=summary.Disponibilidad.eq('Sin piloto independiente')
    assert absent.sum()==8
    assert summary.loc[absent,['MASE medio','DE','CV (%)']].isna().all().all()
    pilots=pd.read_csv(OUT/'pilotos_disponibles.csv')
    expected={frozenset(p) for p in combinations(set(pilots.clave),2)}
    dim=pd.read_csv(OUT/'dimensionamiento_piloto_integrado.csv')
    assert len(dim)==10 and {frozenset(p) for p in zip(dim.A,dim.B)}==expected
    assert (dim['R Bonferroni(10)']>=dim['R individual']).all()
    assert (dim['R individual']>=5).all()
    notebook=nbformat.read(ROOT/'book/10_evaluacion.ipynb',4)
    text='\n'.join(c.source for c in notebook.cells if c.cell_type=='markdown')
    assert 'Dimensionamiento del grupo demanda/calendario' not in text
    assert 'diez parejas' in text

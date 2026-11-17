"""Contrasta el texto publicado y sus tablas con las familias realmente calculadas."""
from pathlib import Path
import importlib.util,json,re
import nbformat
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('revision_integral',ROOT/'scripts/revision_integral_jbook.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_contratos_numericos_de_todas_las_tablas_de_evaluacion():
    report=module.verificar_contratos(ROOT)
    assert report['parejas_temporales']==105 and report['parejas_semillas']==78
    assert report['contrastes_texto']==3 and report['horizontes_texto']==72


def test_denominadores_del_texto_coinciden_con_las_tablas():
    n=nbformat.read(ROOT/'book/10_evaluacion.ipynb',4)
    temporal=pd.read_csv(ROOT/'results/evaluacion_e2/comparaciones_temporales.csv')
    seeds=pd.read_csv(ROOT/'results/evaluacion_e2/comparaciones_semillas.csv')
    intro=next(c.source for c in n.cells if c.cell_type=='markdown' and c.source.startswith('## Comparaciones temporales'))
    assert f'Holm({len(temporal)})' in intro
    result=next(c.source for c in n.cells if c.cell_type=='markdown' and c.source.startswith('DM–HLN con Holm rechaza'))
    count,total=map(int,re.search(r'rechaza en (\d+) de las (\d+) parejas',result).groups())
    assert count<=total and count==temporal.p_DM_Holm.lt(.05).sum() and total==len(temporal)
    intro=next(c.source for c in n.cells if c.cell_type=='markdown' and c.source.startswith('## Comparaciones entre semillas'))
    assert f'las {len(seeds)} parejas' in intro and f'Holm({len(seeds)})' in intro


def test_tablas_textuales_distinguen_correspondencia_de_corpus_y_codificacion():
    root=ROOT/'results/evaluacion_e2/entradas/texto'
    corpus=pd.read_parquet(root/'corpus_oficial.parquet')
    index=pd.read_parquet(root/'indice_encoder.parquet')
    assert corpus.id.tolist()==index.id.tolist() and len(corpus)==14162
    assert corpus.id.nunique()==14162 and index.sha_contenido.nunique()==12330
    assert index.fragmentos_usados.le(8).all()
    facts=json.loads((ROOT/'results/revision_integral/corpus_modelos_auditoria.json').read_text(encoding='utf-8'))
    assert facts['descarte_tokens_sin_doble_conteo_pct']>facts['descarte_tokens_reportado_pct']
    n=nbformat.read(ROOT/'book/07_texto.ipynb',4)
    text='\n'.join(c.source for c in n.cells if c.cell_type=='markdown')
    assert '33.36 %' in text and '36.16 %' in text


def test_el_inventario_no_omite_capitulos_con_tablas_o_figuras():
    inventory=module.inventariar(ROOT)
    assert set(inventory.loc[inventory.tipo.eq('figura'),'archivo'])=={
        '02_calidad.ipynb','03_demanda.ipynb','04_estacionalidad.ipynb','05_dependencia.ipynb',
        '06_covariables.ipynb','07_texto.ipynb','08_literatura.ipynb','09_validacion.ipynb','10_evaluacion.ipynb'}
    assert inventory.tipo.eq('esquema').sum()==24
    assert inventory.tipo.eq('figura').sum()==51

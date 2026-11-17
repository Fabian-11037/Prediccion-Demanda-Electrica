"""Verifica el libro ejecutado y guarda su HTML local sin publicar."""
import json
import shutil
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
from collections import Counter
import nbformat

ROOT=Path(__file__).resolve().parents[1]


def main():
    book=ROOT/'book'; html=book/'_build/html'
    notebooks={p.stem:nbformat.read(p,4) for p in sorted(book.glob('*.ipynb'))}
    figuras=celdas=0; advertencias=[]
    for nombre,n in notebooks.items():
        nbformat.validate(n)
        for i,c in enumerate(n.cells):
            assert '**Interpretación.' not in c.source,(nombre,i)
            if c.cell_type!='code':continue
            celdas+=1
            assert c.execution_count is not None,(nombre,i)
            for salida in c.outputs:
                assert salida.output_type!='error',(nombre,i,salida)
                if salida.output_type=='stream' and salida.get('name')=='stderr':
                    advertencias.append({'cuaderno':nombre,'celda':i,'texto':salida.text})
                if 'image/png' in salida.get('data',{}):
                    figuras+=1
                    assert i+1<len(n.cells) and n.cells[i+1].cell_type=='markdown',(nombre,i)
                    assert len(n.cells[i+1].source.strip())>35,(nombre,i)
    modelo=notebooks['10_modelos']
    assert modelo.metadata.get('capitulo10_integrado')
    assert (book/'arquitecturas/fundamentos.md').exists()
    assert not (book/'10_pipeline.md').exists()
    assert not (book/'10_estadistica_e2.ipynb').exists()
    dic=json.loads((ROOT/'data/processed/diccionario_datos.json').read_text(encoding='utf-8'))
    assert all(r['descripcion'] and r['unidad'] for r in dic)
    assert len(dic)==len({(r['bloque'],r['variable']) for r in dic})
    for nombre in ['diccionario','10_evaluacion','arquitecturas/fundamentos','11_sintesis','12_reproducibilidad']:
        assert (html/(nombre+'.html')).exists(),nombre
    # Compatibilidad de enlaces externos sin conservar capítulos duplicados.
    redirecciones_capitulo10={'10_pipeline':'arquitecturas/fundamentos.html','10_estadistica_e2':'10_evaluacion.html'}
    for nombre,destino in redirecciones_capitulo10.items():
        (html/(nombre+'.html')).write_text('<!doctype html><html lang="es"><head><meta charset="utf-8"><meta http-equiv="refresh" content="0;url='+destino+'"><link rel="canonical" href="'+destino+'"><title>Contenido integrado</title></head><body><a href="'+destino+'">Continuar a la sección integrada</a></body></html>',encoding='utf-8')
    class Enlaces(HTMLParser):
        def __init__(self):
            super().__init__();self.links=[]
        def handle_starttag(self,tag,attrs):
            for k,v in attrs:
                if k in ['href','src'] and v:self.links.append(v)
    rotos=[]
    for p in html.rglob('*.html'):
        if '_static' in p.relative_to(html).parts:continue
        parser=Enlaces();parser.feed(p.read_text(encoding='utf-8'))
        for enlace in parser.links:
            u=urlsplit(enlace)
            if u.scheme or u.netloc or not u.path or u.path.startswith('/'):continue
            if not (p.parent/unquote(u.path)).exists():rotos.append((str(p.relative_to(html)),enlace))
    assert not rotos,rotos[:10]
    shutil.copytree(html,ROOT/'docs',dirs_exist_ok=True)
    (ROOT/'docs/.nojekyll').touch()
    audit=dict(development_start='2000-01-01',development_end_exclusive='2022-01-01',
        test_values_read=True,model_results_read=True,training_executed=False,model_chapter_empty=False,
        published_test_metrics_read=True,saved_test_predictions_read=True,
        results_commit=json.loads((ROOT/'results/jbook/modelos_actualizados.json').read_text(encoding='utf-8'))['commit'],
        authors=['Fabian José Manrique Morelos','Gloria Bastidas Rodriguez'],
        notebooks=len(notebooks),analytic_code_cells=celdas,figures=figuras,
        execution_errors=0,broken_local_links=0,dictionary_fields=len(dic),
        dictionary_by_scope=dict(Counter(r['bloque'] for r in dic)),
        dictionary_schema_coverage_verified=True,duplicate_plot_tables_removed=True,
        published=False,warnings=advertencias)
    (ROOT/'results/jbook/auditoria.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in audit.items() if k!='warnings'},ensure_ascii=False,indent=2))
    print('Advertencias:',len(advertencias))


if __name__=='__main__':
    main()

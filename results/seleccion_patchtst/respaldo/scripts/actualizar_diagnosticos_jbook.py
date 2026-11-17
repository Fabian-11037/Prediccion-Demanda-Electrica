"""Reconstruye la presentación a partir de diagnósticos ya calculados."""
from pathlib import Path
import base64,shutil
import nbformat as nb
from reorganizar_capitulo10 import construir,ejecutar
from diagnosticos_ampliados_jbook import listo,armonizar
from revision_integral_jbook import auditar

def main():
    root=Path(__file__).resolve().parents[1]
    assert listo(root),'Primero deben finalizar inferencia, controles y latencia'
    path=root/'book/10_evaluacion.ipynb'
    n=nb.read(path,4);n.cells=construir(root)
    count=ejecutar(n.cells);nb.write(n,path)
    qa=root/'results/diagnosticos_ampliados/qa';qa.mkdir(exist_ok=True)
    for c in n.cells:
        if c.cell_type!='code':continue
        tag='importancia' if 'd_matrix=' in c.source else 'texto_documental' if 'd_textkeys=' in c.source else 'descripciones' if 'd_statkeys=' in c.source else None
        if tag:
            for output in c.outputs:
                if 'image/png' in output.get('data',{}):
                    (qa/(tag+'.png')).write_bytes(base64.b64decode(output.data['image/png']))
    armonizar(root)
    src=root/'results/evaluacion_e2/cobertura_diagnosticos.csv'
    shutil.copy2(src,root/'book/_static/resultados/evaluacion_e2'/src.name)
    auditar(root)
    print('Capítulo actualizado:',count,'celdas analíticas; sin entrenar ni publicar',flush=True)

if __name__=='__main__':main()

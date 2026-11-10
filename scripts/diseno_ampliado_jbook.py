# Integra el diseño ampliado sin recalcular el benchmark ni entrenar.
from pathlib import Path
import json
import shutil
import nbformat

TITULO='## Contrato numérico de la comparación con texto'
INTRO='''## Contrato numérico de la comparación con texto

Las variantes numérica, numérica con documentos oficiales y numérica con documentos oficiales más prensa comparten demanda, calendario, clima de ocho ciudades, hidrología y precio. El contrato contiene 30 variables históricas y 23 de calendario. La relevancia documental se aprende en dos canales independientes, con sus indicadores de ausencia.

El clima pasado se desplaza hasta un rezago efectivo de 168 horas; se fija una sensibilidad de 90 días. ERA5 es un reanálisis recuperado posteriormente: esta regla no verifica la disponibilidad histórica de cada versión. El precio tiene un rezago de 24 horas; la hidrología diaria se habilita 24 horas después de cerrar el día. Ninguna de estas variables se observa dentro del horizonte pronosticado. El calendario sí se conoce a futuro.

El {ref}`diseño de entrenamiento ampliado <diseno-ampliado>` detalla entradas, presupuesto, controles, inferencia y garantías matemáticas. La siguiente preparación selecciona únicamente desarrollo, conserva ausencias y no estima imputaciones ni escalas. Los resultados del capítulo de modelos corresponden a sus entradas efectivamente entrenadas, demanda y calendario.'''

CIERRE='''La preparación conserva la malla horaria y una máscara para cada entrada ausente. Las tres variantes utilizarán las mismas columnas y fechas; las diferencias se limitarán a los canales textuales. Las medianas, escalas y ponderaciones de relevancia se estimarán dentro del tramo de ajuste de cada pliegue, excluyendo su reserva de parada y su validación externa.'''


def fuente_codigo(root):
    return (root/'src/demanda/diseno_numerico.py').read_text(encoding='utf-8')+r'''

from pathlib import Path
import json
import pyarrow.parquet as pq

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents]
            if (p/'config/diseno_entrenamiento_texto.json').exists())
contrato_comun = json.loads((ROOT/'config/diseno_entrenamiento_texto.json').read_text(encoding='utf-8'))
registro_comun = json.loads((ROOT/'data/processed/feature_registry.json').read_text(encoding='utf-8'))
columnas_comunes = contrato_comun['numericas']['historicas'] + contrato_comun['numericas']['calendario']
# La preparación documental lee desarrollo; no necesita etiquetas de prueba.
panel_desarrollo = pq.read_table(
    ROOT/contrato_comun['panel'], columns=['inicio']+columnas_comunes,
    filters=[('inicio','<',__import__('pandas').Timestamp('2022-01-01',tz='America/Bogota'))]
).to_pandas()
numericas_comunes, ausencia_comun = preparar_numericas_comunes(
    panel_desarrollo, registro_comun, contrato_comun)
print(f"Entradas históricas: {len(contrato_comun['numericas']['historicas'])}; calendario: {len(contrato_comun['numericas']['calendario'])}")
print(f"Filas de desarrollo: {len(numericas_comunes):,}; columnas numéricas: {numericas_comunes.shape[1]}")
print(f"Máscara de ausencia: {ausencia_comun.shape}; rezago climático: {contrato_comun['numericas']['rezago_clima_h']} h")
'''


def integrar_ampliado(books,md,code,root):
    cells=books['09_validacion']
    pos=next((i for i,c in enumerate(cells) if c.source.startswith(TITULO)),None)
    if pos is not None:
        del cells[pos:pos+3]
    cells.extend([md(INTRO),code(fuente_codigo(root)),md(CIERRE)])
    for stem in ['07_texto','06_covariables']:
        for c in books.get(stem,[]):
            if c.cell_type!='markdown':continue
            if stem=='07_texto':
                c.source=c.source.replace('La segmentación, la ventana temporal y los límites de documentos, fragmentos y tokens por canal se fijarán con desarrollo antes de la siguiente evaluación textual.',
                    'Se fijan una ventana principal de 72 horas, hasta 64 documentos por canal y ocho fragmentos por documento; cada fragmento admite 512 tokens incluyendo prefijo y tokens especiales, con solapamiento de 64 tokens. Las sensibilidades de ventana son 24 y 168 horas.')
            if c.metadata.get('plantilla_resultados'):
                # Las sustituciones solo afectan texto fijo, no expresiones numéricas.
                template=c.metadata.plantilla_resultados
                if stem=='07_texto':
                    template=template.replace('La segmentación, la ventana temporal y los límites de documentos, fragmentos y tokens por canal se fijarán con desarrollo antes de la siguiente evaluación textual.',
                        'Se fijan una ventana principal de 72 horas, hasta 64 documentos por canal y ocho fragmentos por documento; cada fragmento admite 512 tokens incluyendo prefijo y tokens especiales, con solapamiento de 64 tokens. Las sensibilidades de ventana son 24 y 168 horas.')
                c.metadata.plantilla_resultados=template
        marker='## Entradas comunes del diseño predictivo'
        if not any(c.source.startswith(marker) for c in books.get(stem,[])):
            books[stem].append(md(marker+'\n\nLa comparación con texto empleará el mismo conjunto numérico en todas sus variantes: demanda, calendario, clima de ocho ciudades, hidrología y precio. La especificación de disponibilidad, las máscaras y el presupuesto se presentan en el {ref}`diseño de entrenamiento ampliado <diseno-ampliado>`. La información realizada durante el horizonte queda excluida de las entradas.'))


def escribir_pagina(root):
    texto=(root/'literature/Diseño de entrenamiento pronóstico de demanda eléctrica con texto.md').read_text(encoding='utf-8')
    texto=texto.replace('# Diseño de entrenamiento: pronóstico de demanda eléctrica con texto',
                        '(diseno-ampliado)=\n# 9.1. Diseño de entrenamiento ampliado',1)
    (root/'book/09_diseno_ampliado.md').write_text(texto,encoding='utf-8')
    static=root/'book/_static/diseno';static.mkdir(parents=True,exist_ok=True)
    shutil.copy2(root/'config/diseno_entrenamiento_texto.json',static/'diseno_entrenamiento_texto.json')
    shutil.copy2(root/'literature/Diseño de entrenamiento pronóstico de demanda eléctrica con texto.md',static/'diseno_entrenamiento_texto.md')
    p=root/'book/09_diseno_ampliado.md'
    p.write_text(texto+'\n\n[Contrato ejecutable JSON](_static/diseno/diseno_entrenamiento_texto.json) · {download}`Documento de diseño <_static/diseno/diseno_entrenamiento_texto.md>`\n',encoding='utf-8')
    toc=root/'book/_toc.yml'
    t=toc.read_text(encoding='utf-8')
    if 'file: 09_diseno_ampliado' not in t:
        t=t.replace('  - file: 09_validacion\n','  - file: 09_validacion\n    sections:\n      - file: 09_diseno_ampliado\n')
    toc.write_text(t,encoding='utf-8')


def actualizar_local(root):
    from IPython.core.interactiveshell import InteractiveShell
    from IPython.utils.capture import capture_output
    books={stem:nbformat.read(root/'book'/f'{stem}.ipynb',4) for stem in ['06_covariables','07_texto','09_validacion']}
    listas={k:v.cells for k,v in books.items()}
    integrar_ampliado(listas,nbformat.v4.new_markdown_cell,nbformat.v4.new_code_cell,root)
    shell=InteractiveShell.instance()
    for stem,n in books.items():
        for c in n.cells:
            if c.cell_type!='code' or c.execution_count is not None:continue
            with capture_output() as cap:result=shell.run_cell(c.source,store_history=False)
            if result.error_in_exec or result.error_before_exec:raise RuntimeError(result.error_in_exec or result.error_before_exec)
            c.execution_count=max([x.execution_count or 0 for x in n.cells if x.cell_type=='code'])+1
            c.outputs=[]
            if cap.stdout:c.outputs.append(nbformat.v4.new_output('stream',name='stdout',text=cap.stdout))
            if cap.stderr:c.outputs.append(nbformat.v4.new_output('stream',name='stderr',text=cap.stderr))
            assert not cap.outputs
        nbformat.validate(n)
        nbformat.write(n,root/'book'/f'{stem}.ipynb')
    escribir_pagina(root)


if __name__=='__main__':
    actualizar_local(Path(__file__).resolve().parents[1])

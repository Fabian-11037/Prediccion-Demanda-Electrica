"""Orden estable: fundamentos, arquitecturas/pipelines y evaluación integrada."""
from pathlib import Path
from reorganizar_capitulo10 import portada,escribir_toc,reorganizar

def integrar_e2(books,md,root):
    books['10_modelos'][:]=portada(root)
    cells=books.get('11_sintesis',[])
    titulo='## Evaluación conjunta de arquitecturas'
    cells[:]=[c for c in cells if not c.source.startswith(titulo)]
    for c in cells:
        if c.cell_type=='markdown' and c.source.startswith('## Resultado del benchmark'):
            c.source=c.source.replace('## Resultado del benchmark','## Referencia con demanda y calendario').replace('La MLP presenta el menor MASE','Entre las cuatro familias con demanda y calendario, la MLP presenta el menor MASE')
    cells.append(md(titulo+"""

La {ref}`evaluación integrada <evaluacion-e2>` compara los procedimientos disponibles y conserva los diagnósticos complementarios de las cuatro familias base. El conjunto de confianza retiene TaTS y su control numérico. El aporte textual de TaTS es equivalente dentro del margen temporal del 3 % condicionado a las inicializaciones ejecutadas, sin mejora demostrada; la fusión tardía aumenta el error. Las entradas, máscaras y presupuestos distintos limitan la atribución exclusiva a la arquitectura. TGForecaster conserva ficha y pipeline con resultados pendientes.
"""))

def escribir_toc_e2(root):escribir_toc(root)
def actualizar_local(root):reorganizar(root)
if __name__=='__main__':actualizar_local(Path(__file__).resolve().parents[1])

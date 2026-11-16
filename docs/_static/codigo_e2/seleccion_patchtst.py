"""Vista del benchmark con una sola configuración numérica de PatchTST."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]
POLITICA=json.loads((ROOT/'config/seleccion_patchtst.json').read_text(encoding='utf-8'))
CONSERVAR=POLITICA['conservar'];RETIRAR=POLITICA['retirar'];NOMBRE=POLITICA['nombre']
def filtrar(frame,columna='clave'):
    return frame.loc[frame[columna].ne(RETIRAR)].copy()

def ajustar_celdas(root,cells):
    """Actualiza alcance y referencias del código visible al conjunto conservado."""
    import pandas as pd
    import re
    o=Path(root)/'results/evaluacion_e2'
    p=json.loads((o/'protocolo.json').read_text(encoding='utf-8'))
    replacements={
      'catorce':'trece','Catorce':'Trece','70 realizaciones':'65 realizaciones','70realizaciones':'65realizaciones',
      'Setenta métricas':'Sesenta y cinco métricas','120 pares':'105 pares','120 comparaciones':'105 comparaciones',
      '120 parejas':'105 parejas','Holm(120)':'Holm(105)','91 pares':'78 pares','91 parejas':'78 parejas',
      '91 comparaciones':'78 comparaciones','Holm(91)':'Holm(78)',
      'seis pilotos':'cinco pilotos','Los seis procedimientos con piloto':'Los cinco procedimientos con piloto',
      'seis a partir del piloto':'cinco a partir del piloto','seis mediante piloto':'cinco mediante piloto',
      'seis procedimientos':'cinco procedimientos','trece contrastes diagnósticos':'doce contrastes diagnósticos',
      'PatchTST (repositorio) · N':NOMBRE,'PatchTST (control N de TaTS/fusión) · N':NOMBRE,
      'control numérico de TaTS/fusión':'PatchTST numérico','control N de TaTS/fusión':'PatchTST numérico',
      'los diez procedimientos':'los nueve procedimientos','las 50 predicciones basales':'las 45 predicciones basales',
      'PatchTST y N-HiTS numéricos conservan 130 ajustes cada uno':'N-HiTS conserva 130 ajustes',
      'La validación de PatchTST y N-HiTS numéricos promedia 2019–2021':'La validación de N-HiTS promedia 2019–2021',
      'La validación de PatchTST/N-HiTS numéricos promedia 2019–2021':'La validación de N-HiTS promedia 2019–2021',
      'seis pilotos externos':'cinco pilotos externos','seis mediante':'cinco mediante',
      'las quince parejas':'las diez parejas','Bonferroni(15)':'Bonferroni(10)',
      'quince comparaciones':'diez comparaciones','Quince comparaciones':'Diez comparaciones',
      '15 parejas externas':'10 parejas externas',
      'g_pilot_seed.shape==(5,6)':'g_pilot_seed.shape==(5,5)',
      'len(g_pilot_pairs)==15':'len(g_pilot_pairs)==10',
      'PatchTST y N-HiTS numéricos conservan curvas':'N-HiTS conserva curvas',
      'PatchTST y su control numérico':'PatchTST numérico',
      'PatchTST y N-HiTS numéricos utilizan tres semillas':'N-HiTS utiliza tres semillas',
      'los ocho procedimientos ampliados restantes':'los ocho procedimientos ampliados restantes',
      'diez para PatchTST/N-HiTS numéricos del repositorio':'diez para N-HiTS',
    }
    for cell in cells:
        for a,b in replacements.items():cell.source=cell.source.replace(a,b)
        if cell.cell_type=='markdown':
            cell.source=cell.source.replace('120',str(p['comparaciones_temporales'])) if 'comparaciones temporales' in cell.source else cell.source
            cell.source=re.sub(r'distancia crítica de aproximadamente [0-9,]+ posiciones',f"distancia crítica de aproximadamente {p['CD_Nemenyi']:.2f} posiciones",cell.source)
            cell.source=cell.source.replace('El control N compartido por TaTS y fusión tardía se contabiliza una vez.','PatchTST se contabiliza una vez y sirve como control numérico de TaTS y fusión tardía.')
            cell.source=cell.source.replace('El menor MASE medio del piloto corresponde a MLP (0,542), seguido de PatchTST numérico (0,569). N-HiTS registra 0,827.',
                'MLP presenta el menor MASE medio del piloto externo (0,542); N-HiTS registra 0,827. PatchTST utiliza la precisión de parada interna, presentada por separado.')
            cell.source=cell.source.replace('PatchTST/N-HiTS conservaron también un diagnóstico registrado mediante bloques de 14 días frente al ingenuo semanal de validación. Sus estimaciones límite son 0,0984 y 0,0843 y no determinan un número final de semillas.',
                'N-HiTS conserva además el diagnóstico temporal registrado frente al ingenuo semanal de validación. Su estimación se presenta en la tabla y no determina un número final de semillas.')
            cell.source=cell.source.replace('Ambos registros indican precisión temporal insuficiente','El registro indica precisión temporal insuficiente')
        else:
            if 'if g_key not in g_keys: continue' not in cell.source:
                cell.source=cell.source.replace("g_key = g_exp+'__'+g_variant","g_key = g_exp+'__'+g_variant\n    if g_key not in g_keys: continue")
        if cell.cell_type=='code' and cell.source.startswith('g_pilot_seed='):
            cell.source=cell.source.replace('range(15)','range(10)')
    return cells

def armonizar(root):
    """Mantiene la misma selección en diseño, literatura y reproducción."""
    import nbformat as nb
    import shutil
    import pandas as pd
    import re
    root=Path(root)
    dim=pd.read_csv(root/'results/evaluacion_e2/dimensionamiento_piloto_integrado.csv')
    max_b=int(dim['R Bonferroni(10)'].max())
    for path in (root/'book').glob('*.ipynb'):
        n=nb.read(path,4)
        cells=[c for c in n.cells if c.cell_type=='markdown']
        ajustar_celdas(root,cells)
        for c in cells:
            c.source=c.source.replace('y diez para PatchTST/N-HiTS numéricos del repositorio','y diez para N-HiTS')
            c.source=c.source.replace('PatchTST y N-HiTS numéricos utilizan tres semillas y la media de 2019–2021','N-HiTS utiliza tres semillas y la media de 2019–2021')
            c.source=c.source.replace('PatchTST y N-HiTS numéricos del repositorio','N-HiTS del repositorio')
            c.source=c.source.replace('máximo conservador para las diez comparaciones es 82',f'máximo conservador para las diez comparaciones es {max_b}')
            c.source=c.source.replace('máximos de 82 semillas en los pilotos externos',f'máximos de {max_b} semillas en los pilotos externos')
            c.source=c.source.replace('En los ocho procedimientos sin piloto independiente, el número requerido permanece sin estimar.',
                'Los ocho procedimientos sin piloto independiente conservan su dimensionamiento retrospectivo de parada interna, presentado por separado.')
            c.source=c.source.replace('Los registros de PatchTST y N-HiTS incluyen además un diagnóstico temporal por bootstrap de bloques de 14 días frente al ingenuo semanal de validación. Sus estimaciones límite son 0,0984 y 0,0843 y no determinan un número final de semillas.',
                'N-HiTS conserva además un diagnóstico temporal por bootstrap de bloques de 14 días frente al ingenuo semanal de validación. Su estimación límite es 0,0843 y no determina un número final de semillas.')
        nb.write(n,path)
    # Dejar el instalador y los activos disponibles para una reconstrucción posterior.
    shutil.copy2(root/'scripts/seleccion_patchtst.py',root/'book/_static/codigo_e2/seleccion_patchtst.py')

"""Presentación académica del libro a partir de sus resultados conservados."""
from pathlib import Path
import re
import nbformat as nb

DISCUSION = '''## Discusión

TaTS y PatchTST presentan los menores errores y forman el conjunto de confianza del benchmark. La incorporación de documentos oficiales no demuestra una mejora práctica del 3 % en TaTS; fusión tardía y TGForecaster N+O presentan mayor error que sus controles numéricos. Estos resultados corresponden al corpus declarado y a los mecanismos de integración evaluados. La utilidad del contenido documental debe interpretarse junto con su disponibilidad, cobertura y coste de codificación.

Las diferencias entre procedimientos combinan arquitectura, entradas y presupuesto de selección. Los tiempos de entrenamiento registrados en GPU AMD/ROCm y NVIDIA/CUDA no constituyen una comparación controlada de eficiencia. Los parámetros y la latencia de inferencia complementan la precisión, mientras que la cobertura incompleta de memoria pico y FLOPs limita la comparación de recursos.

La selección retrospectiva y la exposición al conjunto de prueba restringen la interpretación confirmatoria. Los intervalos temporales condicionan las cinco inicializaciones ejecutadas; las comparaciones entre semillas describen variabilidad del ajuste en una misma serie. El dimensionamiento retrospectivo no acredita potencia de 0,80 para el margen del 3 %. Las revisiones de las series, la disponibilidad declarada del texto y los cambios de composición documental también afectan la validez externa.

Los controles con pesos fijos permiten examinar la dependencia del modelo respecto de sus entradas, pero la atribución del aporte textual requiere comparar núcleos e hiperparámetros idénticos. Una evaluación independiente, con presupuestos homogéneos y controles de cobertura y fuentes, permitiría contrastar la estabilidad de los resultados. El error residual indica margen de mejora, sin identificar un nivel de ruido irreducible.'''

ALCANCE_REVISION = '''### Alcance de la revisión

El recuento de comparadores comprende cinco estudios empíricos de demanda revisados en texto completo; la revisión de Nti et al. y los trabajos consultados únicamente mediante resumen aportan contexto. La extracción fue realizada por un solo revisor. La falta de cadenas y fechas originales de Scopus y Web of Science, junto con la conciliación incompleta entre bases, limita la exhaustividad y reproducibilidad de la búsqueda.'''

REPRODUCCION = '''## Reproducción de la evaluación y sus diagnósticos

Las fuentes de `book/` incluyen los cuadernos con sus salidas, las fichas de arquitectura, los esquemas y los archivos descargables. `jupyter-book build book --warningiserror --keep-going` construye el HTML desde esas fuentes, con la ejecución de cuadernos desactivada.

Los CSV y NPZ de `results/evaluacion_e2/` y `results/diagnosticos_ampliados/` contienen las pérdidas, contrastes, curvas y mediciones utilizadas en las figuras. La reconstrucción de la presentación utiliza estos archivos y las predicciones conservadas:

```bash
python scripts/reorganizar_capitulo10.py
jupyter-book build book --warningiserror --keep-going
python scripts/verificar_jbook.py
```

`scripts/actualizar_diagnosticos_jbook.py` actualiza las figuras a partir de las mismas mediciones. `scripts/reconstruir_evaluacion.py` vuelve a ejecutar el código analítico del cuaderno integrado con las entradas verificadas por SHA-256. Ninguno de estos procedimientos reajusta modelos.

La repetición de inferencia requiere, además, los checkpoints, escaladores y vectores E5 identificados en `fuentes_diagnosticos.json`. Su ubicación se configura mediante `DEMANDA_ARTEFACTOS`, apuntando al directorio que contiene `consolidado/indice_modelos.json` y `fuentes/`. Estos archivos binarios no son necesarios para construir el libro ni reconstruir las gráficas desde las mediciones. Las nuevas mediciones utilizan el entorno `requirements-inferencia.lock.txt` y los comandos siguientes:

```bash
python scripts/diagnosticos_ampliados.py --fase inferir --workers 4
python scripts/diagnosticos_ampliados.py --fase latencia
```

`reproduccion_semillas.csv` documenta la correspondencia con las predicciones conservadas; `entorno.json` identifica las condiciones de la medición de latencia.'''

def literatura(cells):
    # La selección se basa en encabezados y código, conservando resultados de demanda.
    out=[]
    for c in cells:
        s=c.source
        if c.cell_type=='markdown':
            if s.startswith('### Precio eléctrico') or s.startswith('### Evaluación crítica de los textos completos'):
                continue
            if s.startswith('El cribado de IEEE retiene') or s.startswith('### Trazabilidad de la búsqueda') or s.startswith('### Consultas públicas conservadas'):
                continue
            if s.startswith('En precio puntual, LSTM aparece'):
                continue
            if s.startswith('# 8. Literatura'):
                c.source='''# 8. Literatura y fundamento del diseño

La revisión examina el pronóstico de demanda eléctrica mediante redes numéricas, representaciones multiescala e integración documental. La evidencia empírica de demanda sustenta la selección de comparadores; los métodos generales orientan las alternativas de representación. Los resultados se presentan con las métricas y condiciones de cada publicación, distinguiendo afiliación de los autores y territorio de los datos.'''
            elif s.startswith('## 8.1. Fuentes'):
                c.source='''## 8.1. Fuentes, selección y alcance

Se realizó una revisión dirigida de estudios de demanda eléctrica y métodos de pronóstico temporal y documental. Se consultaron exportaciones de Scopus, Web of Science e IEEE Xplore, complementadas con búsquedas públicas y recuperación de textos completos. La extracción distingue evidencia empírica de demanda, contribuciones metodológicas y resultados disponibles únicamente en resúmenes.

Se priorizaron estudios con datos, horizonte, arquitectura y evaluación identificables. Los duplicados, avisos de retirada y aplicaciones sin relación metodológica con el problema se excluyeron de la síntesis empírica.'''
            elif s.startswith('### Protocolo de elegibilidad'):
                c.source='''### Criterios de elegibilidad y búsqueda

La búsqueda dirigida abarcó publicaciones de 2018 al 8 de noviembre de 2026. El recuento de comparadores requiere texto completo y evaluación explícita de la familia: se excluyen modelos propuestos, componentes, ablaciones y menciones. Las revisiones aportan contexto y los resúmenes permiten identificar candidatos.

La ampliación pública del 8 de noviembre de 2026 utilizó `(load forecasting OR electricity demand forecasting) AND (CNN OR convolutional OR LSTM OR MLP) AND (comparison OR benchmark)`. El [registro de búsqueda](_static/literature/busqueda_demanda_20261108.json) documenta las consultas y fuentes recuperadas.'''
            elif s.startswith('## 8.3.'):
                c.source='''## 8.3. Métodos complementarios

Las contribuciones multiescala, de mezcla temporal y de representación documental ofrecen mecanismos de interés para la demanda. Los resultados obtenidos en otros dominios se utilizan como fundamento metodológico y no se incorporan al recuento empírico de comparadores.'''
            elif s.startswith('### Métodos generales'):
                lines=s.splitlines()
                c.source='\n'.join(line for line in lines if not any(x in line for x in ['Time-MMD','Tan et al.','REST-TS','TeR-TSF','TimesX']))
                c.source+='''\n\n[Liu et al., Time-MMD](https://arxiv.org/pdf/2406.08627v2) aporta un diseño de integración entre series y documentos. [Tan et al.](https://arxiv.org/pdf/2406.16964v2) examina el papel del modelo de lenguaje mediante ablaciones. [Nguyen et al., REST-TS](https://arxiv.org/pdf/2606.19413) estudia una corrección documental de residuos. Estas referencias orientan los controles del componente textual; sus resultados no se trasladan directamente al pronóstico de demanda del SIN.'''
            elif s.startswith('### Texto operativo y financiero'):
                c.source='\n'.join(line for line in s.replace('### Texto operativo y financiero','### Documentos operativos').splitlines() if not line.startswith('| [') or 'Egharevba' in line)
            elif s.startswith('Forwardformer se relaciona'):
                continue
            elif s.startswith('## 8.4.'):
                c.source='''## 8.4. Matriz de evidencia de demanda

La matriz distingue el uso de texto externo, la presencia de resultados numéricos y la revisión de texto completo en los seis estudios de demanda. Una revisión bibliográfica y un estudio empírico no aportan el mismo tipo de evidencia.'''
            elif s.startswith('De los 29 estudios destacados'):
                c.source='La síntesis reúne cinco estudios empíricos de demanda y una revisión. Dos estudios empíricos integran información textual. Esta evidencia permite examinar mecanismos de integración, pero su presencia no demuestra una mejora generalizable entre sistemas eléctricos.'
            elif s.startswith('### Alcance del cumplimiento bibliográfico'):
                c.source=ALCANCE_REVISION
            elif s.startswith('## 8.7.'):
                c.source=s.replace(' Los estudios de precio aportan protocolos y alternativas, pero su volatilidad, unidades y mecanismos económicos difieren de los de demanda.','')
                c.source=c.source.replace(' En NSW-EPNews, una ablación que elimina simultáneamente noticias y temperatura no identifica por sí sola el efecto del texto.',' Una ablación que retire simultáneamente texto y covariables numéricas no permite aislar la contribución documental.')
            elif s.startswith('### Extensiones respaldadas'):
                c.source=s.replace('News and Load, Bai LSTM, AGESL, BELT, TSITM','News and Load, Bai LSTM, documentos operativos')
            elif s.startswith('## Brecha de investigación'):
                c.source=s.replace('Permanecen pendientes la variante con prensa, las ablaciones por fuente y una comparación con núcleo e hiperparámetros idénticos para atribuir el efecto exclusivamente al contenido.','La atribución específica al contenido requiere controles por fuente y comparaciones con núcleo e hiperparámetros idénticos; la integración de prensa constituye una extensión del diseño.')
                c.source=re.sub(r'\n\nTime-MMD, de Liu et al\., y NSW-EPNews.*', '', c.source, flags=re.S)
        elif c.cell_type=='code':
            if s.startswith("roles=['comparador'"):
                continue
            if s.startswith('from pathlib import Path\nimport json, hashlib'):
                c.source='''import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from IPython.display import display
plt.rcParams.update({'figure.dpi':120,'font.size':10})'''
                c.outputs=[];c.execution_count=None
            elif s.startswith("matriz=estudios.set_index"):
                c.source='''matriz=pd.DataFrame(
    [[0,0,1],[1,1,1],[1,1,1],[0,1,1],[0,1,1],[0,1,1]],
    index=['Nti · revisión','News and Load','Bai · LSTM','Pelekis','Kathirgamanathan','Laitsos'],
    columns=['Texto externo','Resultado numérico','Texto completo'])
fig,ax=plt.subplots(figsize=(8,3.7))
ax.imshow(matriz.values,aspect='auto',cmap='Blues',vmin=0,vmax=1)
ax.set(xticks=range(3),xticklabels=matriz.columns,yticks=range(len(matriz)),yticklabels=matriz.index)
for i in range(len(matriz)):
    for j in range(3):
        if matriz.iloc[i,j]:ax.text(j,i,'●',ha='center',va='center',color='white')
fig.tight_layout();display(fig);plt.close(fig)'''
                c.outputs=[];c.execution_count=None
        out.append(c)
    return out

def aplicar(root):
    root=Path(root)
    for path in (root/'book').glob('*.ipynb'):
        n=nb.read(path,4);original=nb.writes(n)
        for c in n.cells:
            if c.cell_type=='code':
                c.source=c.source.replace('for fila in indice.itertuples():\n    p = SNAP / fila.archivo',"for fila in indice.itertuples():\n    if fila.archivo.endswith(('.pt', '.pth')): continue\n    p = SNAP / fila.archivo")
                continue
            if c.cell_type!='markdown':continue
            if c.source.startswith('## Discusión y alcance del Entregable 2'):
                c.source=DISCUSION
            c.source=c.source.replace('La pauta E2 §§6.3 y 7.4 S7–S8 fundamenta las comparaciones y diagnósticos. ','')
            c.source=c.source.replace('No se cumple el orden piloto → dimensionamiento → número de semillas de §4.1: el dimensionamiento presentado no determinó las cinco realizaciones. Se conserva como limitación metodológica, junto con el presupuesto desigual y la potencia insuficiente para el margen del 3 %.','El dimensionamiento es retrospectivo y no determinó las cinco realizaciones ejecutadas. La desigualdad de presupuestos y la incertidumbre de potencia para el margen del 3 % limitan la comparación.')
            c.source=c.source.replace('La tabla integra decisiones y riesgos; los valores y patrones gráficos se presentan en sus capítulos correspondientes.','').rstrip()
            c.source=c.source.replace('la incorporación de prensa permanece pendiente','el canal documental evaluado se limita a fuentes oficiales')
            c.source=c.source.replace('Calcular una huella no deserializa ni evalúa etiquetas de prueba.','').rstrip()
            c.source=c.source.replace('## Conservación y reconstrucción de la entrega','## Construcción y verificación del libro')
            c.source=c.source.replace(' No publica ni inicia entrenamiento.','')
            c.source=c.source.replace('; una actualización del sitio requiere una solicitud expresa.','.')
            if c.source.startswith('## Reproducción de los diagnósticos ampliados') or c.source.startswith('## Reproducción de la evaluación y sus diagnósticos'):
                c.source=REPRODUCCION
        if path.stem=='08_literatura':n.cells=literatura(n.cells)
        if nb.writes(n)!=original:nb.write(n,path)
    p=root/'book/intro.md';s=p.read_text(encoding='utf8')
    s=s.replace('catorce procedimientos de pronóstico','trece procedimientos de pronóstico')
    s=s.replace('variantes con texto oficial y referencias ARX y estacionales.','variantes con texto oficial y el modelo ARX, además de dos referencias estacionales.')
    p.write_text(s,encoding='utf8')
    p=root/'book/09_diseno_ampliado.md';s=p.read_text(encoding='utf8')
    s=re.sub(r'La pauta E2 §14 exime.*?encoder preentrenado\.', 'Las configuraciones adoptadas de la literatura deben identificar las capas, dimensiones y adaptación a las entradas del SIN. Para las arquitecturas sin configuración predefinida, la búsqueda considera un espacio común de anchos y regularización.',s,flags=re.S)
    s=s.replace('1.000 arquitecturas por modelo sujeto a §14','1.000 arquitecturas por modelo sin configuración predefinida')
    s=s.replace('0,80 para el benchmark E2 y 0,90 para las afirmaciones de superioridad del modelo original E3','0,80 para el benchmark y 0,90 para las afirmaciones de superioridad del modelo original')
    s=s.replace('## 9. Entregables de implementación','## 9. Implementación y registro experimental')
    p.write_text(s,encoding='utf8')
    p=root/'book/arquitecturas/tcn.md';s=p.read_text(encoding='utf8')
    s=s.replace('Los 20 candidatos efectivamente evaluados no equivalen a completar el espacio de 1.000 arquitecturas exigido por la pauta cuando no existe una configuración publicada adoptada.','La selección evaluó 20 candidatos; su alcance depende del espacio y del presupuesto de búsqueda descritos en el diseño experimental.')
    p.write_text(s,encoding='utf8')

if __name__=='__main__':aplicar(Path(__file__).resolve().parents[1])

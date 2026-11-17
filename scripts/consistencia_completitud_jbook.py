"""Armonización documental con resultados conservados; no ejecuta redes."""
from pathlib import Path
import json,re
import nbformat as nb


def aplicar(root):
    root=Path(root)
    protocolo=json.loads((root/'results/evaluacion_e2/protocolo.json').read_text(encoding='utf-8'))
    for path in (root/'book').glob('*.ipynb'):
        n=nb.read(path,4)
        for c in n.cells:
            if c.cell_type!='markdown':continue
            s=c.source
            if path.stem=='01_datos':
                s=s.replace('documentos oficiales disponibles antes del origen','documentos oficiales disponibles al emitir el pronóstico')
                s=s.replace('texto oficial disponible antes del origen','texto oficial disponible al emitir el pronóstico')
            elif path.stem=='08_literatura':
                if s.startswith('### 8.6.1.'):
                    s=s.replace('en §10.2','en la {ref}`ficha de MLP <e2-mlp>`')
                elif s.startswith('### 8.6.2.'):
                    s=s.replace('§10.2 reproduce','La {ref}`ficha de LSTM <e2-lstm>` reproduce')
                elif s.startswith('### 8.6.3.'):
                    s=s.replace('§10.2 contiene','La {ref}`ficha de CNN1D <e2-cnn1d>` contiene')
                if s.startswith('## Brecha de investigación'):
                    s=s.replace('La siguiente etapa debe separar utilidad del contenido, cobertura de fuentes y selección de representación mediante ablaciones. La evaluación disponible contrasta el texto oficial en TaTS y fusión tardía frente a un control numérico compartido; la variante adicional con prensa y las ablaciones de contenido siguen pendientes.',
                        'La evaluación contrasta TaTS y fusión tardía frente a PatchTST numérico, y TGForecaster con documentos frente a su variante numérica. Los controles de retirada, contenido cero y contenido histórico de la misma fuente examinan la dependencia del texto con pesos fijos. Permanecen pendientes la variante con prensa, las ablaciones por fuente y una comparación con núcleo e hiperparámetros idénticos para atribuir el efecto exclusivamente al contenido.')
            elif path.stem=='07_texto':
                s=s.replace('La comparación distingue el aporte de los documentos oficiales y el de la prensa mediante canales separados, con relevancia aprendida dentro del modelo.',
                    'La comparación ejecutada evalúa documentos oficiales con relevancia aprendida dentro del modelo. El diseño contempla incorporar prensa en un canal separado.')
                s=s.replace('Las sensibilidades de ventana son 24 y 168 horas.','El diseño prevé sensibilidades de ventana de 24 y 168 horas.')
                s=s.replace('el estricto se utiliza para sensibilidad de disponibilidad.','el estricto se caracteriza mediante cobertura y se reserva para una sensibilidad predictiva de disponibilidad.')
                s=s.replace('cuya longitud se fijará antes del experimento textual.','cuya longitud efectiva es de 72 horas en los modelos evaluados; TaTS aplica esa ventana en siete cierres diarios.')
                s=s.replace('La segmentación y la ponderación aprendida se incorporarán al pipeline predictivo; este bloque únicamente prepara sus entradas.',
                    'Este bloque prepara ambos canales. Los modelos evaluados utilizan embeddings E5 y ponderación aprendida para el canal oficial; la rama de prensa pertenece al diseño de ampliación.')
                s=s.replace('El protocolo textual se versionará antes de ejecutar las comparaciones.','Las nuevas comparaciones se registrarán antes de su ejecución.')
                if s.startswith('## Corpus oficial utilizado en los modelos') and 'extremos inclusivos' not in s:
                    s+='\n\nLas listas admiten disponibilidad declarada en [cierre − 72 h, cierre], con extremos inclusivos. En TaTS los cierres son o−6 días,…,o; fusión tardía y TGForecaster consultan el origen o. La regla coincide con el código conservado y no acredita la versión histórica de cada documento.'
            elif path.stem=='09_validacion':
                s=s.replace('MAE, RMSE y sMAPE como complementarias','MAE y RMSE como complementarias del benchmark integrado')
                s=s.replace(' sMAPE promedia $200|y-\\widehat y|/(|y|+|\\widehat y|)$, con contribución cero cuando ambos son cero.','')
                if s.startswith('## Métricas y comparación') and 'RMSE del pronóstico promedio' not in s:
                    s+='\n\nLas métricas se calculan por semilla y se promedian entre las cinco realizaciones. En particular, el RMSE medio entre semillas no es el RMSE del pronóstico promedio. Las dos referencias deterministas conservan una sola realización.'
                if s.startswith('## Contrato numérico de la comparación con texto'):
                    s=s.replace('Las variantes numérica, numérica con documentos oficiales y numérica con documentos oficiales más prensa comparten',
                        'El diseño de las variantes numérica, con documentos oficiales y con documentos oficiales más prensa establece como entradas comunes')
                    s=s.replace('La relevancia documental se aprende en dos canales independientes, con sus indicadores de ausencia.',
                        'Los modelos ejecutados incorporan el canal oficial; la separación entre oficial y prensa corresponde a la ampliación propuesta.')
                    s=s.replace('se fija una sensibilidad de 90 días.','el diseño prevé una sensibilidad de 90 días.')
            elif path.stem=='10_evaluacion':
                s=re.sub(r'Levene centrado en mediana rechaza igualdad de varianzas \(p≈[0-9.,]+\)',
                    f"Levene centrado en mediana rechaza igualdad de varianzas (p≈{protocolo['p_Levene']:.6f})",s)
                if s.startswith('## Discusión y alcance del Entregable 2'):
                    anterior='La comparación reúne ocho familias ampliadas, representadas por nueve procedimientos, además de MLP, LSTM, CNN1D y ARX. La diversidad cubre más de cuatro familias, pero no completa el Top 10 adicional exigido en §3.2 ni su tabla reproducible de priorización.'
                    actual='El conjunto principal comprende diez modelos entrenados: ARX, MLP, LSTM, CNN1D, TFT, TCN, iTransformer, TaTS, fusión tardía y TGForecaster. PatchTST y N-HiTS se conservan como comparadores numéricos; las dos variantes de TGForecaster llevan el total a trece procedimientos, además de las dos referencias deterministas.'
                    s=s.replace(anterior,actual)
                    if actual not in s:
                        s+='\n\n'+actual+' Tampoco se documenta una réplica del paper guía bajo su dataset, partición y métrica originales (§5). Los parámetros y la latencia están registrados para todos los procedimientos; la memoria pico y los FLOPs no tienen cobertura completa (§4.2).'
                    s=s.replace('la memoria pico y los FLOPs no tienen cobertura completa (§4.2).',
                        'solo N-HiTS conserva un registro de memoria pico y no se reportan FLOPs para los trece procedimientos (§4.2).')
            elif path.stem=='12_reproducibilidad':
                if s.startswith('### Windows') and '##' in s:
                    s='''### Windows

```powershell
python -m venv .bookenv
.bookenv\\Scripts\\python.exe -m pip install -r requirements-book.lock.txt
.bookenv\\Scripts\\jupyter-book.exe build book --warningiserror --keep-going
.bookenv\\Scripts\\python.exe scripts\\verificar_jbook.py
```

### Linux

```bash
python3 -m venv .bookenv
.bookenv/bin/python -m pip install -r requirements-book.lock.txt
.bookenv/bin/jupyter-book build book --warningiserror --keep-going
.bookenv/bin/python scripts/verificar_jbook.py
```

Estos comandos construyen el libro con las salidas conservadas. `execute_notebooks: off` evita reevaluar las celdas durante esa construcción. Para recalcular los análisis se utiliza `python scripts/crear_jbook.py`; para reconstruir solo la evaluación, `python scripts/reorganizar_capitulo10.py`. La recalculación requiere las entradas e instantáneas indicadas en la procedencia; los diagnósticos de redes conservan su entorno separado.'''
            c.source=s
        nb.write(n,path)
    p=root/'book/09_diseno_ampliado.md'
    if p.is_file():
        s=p.read_text(encoding='utf-8')
        marker='Versión de diseño: 10 de noviembre de 2026.'
        if 'La especificación siguiente corresponde a la ampliación' not in s:
            s=s.replace(marker,marker+'\n\nLa especificación siguiente corresponde a la ampliación con canales oficial y prensa, comparaciones con núcleo común y una propuesta de fusión acotada. Los procedimientos evaluados y sus controles con pesos fijos se describen en el capítulo 10; la rama de prensa y la compuerta acotada no forman parte de esos resultados.')
        s=s.replace('Las métricas actuales de demanda y calendario no representan el rendimiento del conjunto ampliado.',
            'La evaluación distingue las cuatro familias de demanda/calendario de las arquitecturas que utilizan el conjunto numérico ampliado.')
        p.write_text(s,encoding='utf-8')
    for p in [root/'literature/Diseño de entrenamiento pronóstico de demanda eléctrica con texto.md',
              root/'book/_static/diseno/diseno_entrenamiento_texto.md']:
        if not p.exists():continue
        s=p.read_text(encoding='utf-8');marker='Versión de diseño: 10 de noviembre de 2026.'
        if 'La especificación siguiente corresponde a la ampliación' not in s:
            s=s.replace(marker,marker+'\n\nLa especificación siguiente corresponde a la ampliación con canales oficial y prensa, comparaciones con núcleo común y una propuesta de fusión acotada. Los procedimientos evaluados y sus controles con pesos fijos se describen en el capítulo 10; la rama de prensa y la compuerta acotada no forman parte de esos resultados.')
        s=s.replace('Las métricas actuales de demanda y calendario no representan el rendimiento del conjunto ampliado.',
            'La evaluación distingue las cuatro familias de demanda/calendario de las arquitecturas que utilizan el conjunto numérico ampliado.')
        p.write_text(s,encoding='utf-8')


if __name__=='__main__':aplicar(Path(__file__).resolve().parents[1])

"""Párrafos de cobertura y normalización editorial derivados de los resultados."""
from pathlib import Path
import re
import pandas as pd


def sin_repeticiones(texto):
    """Elimina párrafos idénticos dentro de una celda, preservando código y tablas."""
    vistos=set();partes=re.split(r'(```.*?```)',texto,flags=re.S)
    for i,parte in enumerate(partes):
        if parte.startswith('```'):continue
        conservar=[]
        for parrafo in parte.split('\n\n'):
            clave=parrafo.strip()
            repetible=len(clave)>90 and not clave.startswith(('#','|'))
            if repetible and clave in vistos:continue
            if repetible:vistos.add(clave)
            conservar.append(parrafo)
        partes[i]='\n\n'.join(conservar)
    return ''.join(partes)


def textos(s):
    from narrativa_resultados import palabras as w
    n=s['procedimientos'];sem=s['semillas'];meta=s['_tablas']['metadatos_procedimientos']
    d=pd.read_csv(Path(s['_root'])/'results/evaluacion_e2/diagnostico_residuos.csv')
    cuenta=lambda c:int(d[c].lt(.05).sum())
    ajustados=meta[['MASE_ajuste','MASE_parada']].dropna()
    normalizados=meta[['MAE_ajuste_normalizado','MAE_parada_normalizado']].dropna()
    comparaciones=len(ajustados)+len(normalizados)
    mayores=int((ajustados.MASE_parada>ajustados.MASE_ajuste).sum()+(normalizados.MAE_parada_normalizado>normalizados.MAE_ajuste_normalizado).sum())
    faltan=n-meta.loc[meta[['MASE_ajuste','MASE_parada']].notna().all(axis=1)|meta[['MAE_ajuste_normalizado','MAE_parada_normalizado']].notna().all(axis=1),'clave'].nunique()
    ampliadas=meta.loc[meta[['MASE_ajuste','MASE_parada']].notna().all(axis=1),'clave'].nunique()
    r={}
    hor=s['_tablas']['resultados_horizonte']
    ganadores=hor.loc[hor.groupby('horizonte').MASE.idxmin()].groupby('clave').size().sort_values(ascending=False)
    detalle='; '.join(f"{s['nombres'][k]} en {int(v)}" for k,v in ganadores.items())
    r['horizonte_interpret']=f'El menor MASE medio por horizonte corresponde a {detalle}, de las {s["horizonte"]} horas evaluadas. El orden cambia con la hora; este recuento descriptivo no equivale a un contraste de diferencias. Los contrastes textuales conservan {s["horizonte"]*s["contrastes_textuales"]} pruebas DM–HLN, ajustadas conjuntamente por Holm, en la tabla descargable.'
    r['datos_pregunta']=f'**Tipo de problema:** series de tiempo, con pronóstico directo de 24 pasos, covariables y rezagos del objetivo. **Pregunta de investigación:** ¿qué precisión y estabilidad presentan las arquitecturas numéricas para anticipar la demanda horaria del SIN colombiano durante las siguientes 24 horas, y qué contribución incremental aporta el texto oficial disponible al emitir el pronóstico? La evaluación compara {w(n)} procedimientos completados y {w(s["referencias"])} referencias estacionales; TGForecaster participa con sus variantes N y N+O.'
    r['ajuste_interno']=f'### Ajuste y parada interna\n\nLos registros conservan MAE normalizado en modo evaluación para las {w(s["demanda_calendario"])} familias de demanda/calendario y MASE de ajuste/parada para {w(ampliadas)} variantes ampliadas. Se muestran por separado para preservar sus unidades. N-HiTS conserva curvas de pérdida, pero no estas métricas de evaluación de ambos tramos; no se reemplazan por pérdidas de entrenamiento calculadas con dropout.'
    r['ajuste_conclusion']=f'La parada interna presenta mayor error que el ajuste en {mayores} de las {comparaciones} realizaciones con ambas métricas. La separación describe generalización dentro del periodo de desarrollo; su magnitud no se compara entre paneles porque las escalas difieren. Las curvas de aprendizaje incluyen también {('un procedimiento' if faltan==1 else w(faltan)+' procedimientos')} sin estas métricas desagregadas.'
    r['semana']=f'### Perfil horario en una semana de prueba\n\nSe utiliza la misma ventana del 1 al 7 de julio de 2023 para los {w(n)} procedimientos. La banda p10–p90 representa dispersión entre {w(sem)} semillas y no un intervalo predictivo calibrado. El MAE semanal se calcula con los errores individuales, mientras la predicción media se utiliza para visualizar el perfil.'
    r['segmentos']=f'## Segmentos de evaluación\n\nSe conserva la evaluación por año, tipo de día y cuartil de demanda para los {w(n)} procedimientos. Cada panel muestra pérdidas promediadas entre semillas dentro del segmento. Los cortes describen heterogeneidad de prueba y no se utilizan para seleccionar configuraciones.'
    r['residuos_intro']=f'## Diagnóstico de residuos y calibración\n\nPara el diagnóstico se utiliza el residuo diario promedio de la predicción media de las {w(sem)} semillas. Este promedio se restringe al diagnóstico: no reemplaza la definición del benchmark. Se evalúan Ljung–Box a 7, 14 y 28 días, Ljung–Box de residuos cuadrados, ARCH-LM con 14 rezagos, BDS con dimensión 2 y radio 1,5 desviaciones, Jarque–Bera y calibración de Mincer–Zarnowitz. Cada columna de diagnósticos se ajusta por Holm entre los {w(len(d))} procedimientos.'
    r['residuos_resultado']=f'Ljung–Box a 14 días detecta dependencia residual en {cuenta("p_LB14_Holm")} de los {len(d)} procedimientos; Mincer–Zarnowitz rechaza la calibración conjunta de intercepto cero y pendiente uno en {cuenta("p_Mincer_Zarnowitz_Holm")}. BDS rechaza independencia en {cuenta("p_BDS_Holm")}; con autocorrelación remanente, ese rechazo no demuestra por sí solo no linealidad. ARCH-LM rechaza ausencia de heterocedasticidad dinámica en {cuenta("p_ARCH_Holm")} procedimientos. La dependencia remanente justifica emplear inferencia temporal robusta.\n\nMincer–Zarnowitz se ajusta sobre demanda diaria media expresada en unidades q168 para evitar una matriz de restricciones mal condicionada; el contraste nulo sigue siendo intercepto cero y pendiente uno. La covarianza es HAC de 13 días. No se calcula PIT ni cobertura probabilística a partir del rango entre semillas, porque estos modelos entregan pronósticos puntuales.'
    r['varianza']=f'## Variación entre fechas e inicializaciones\n\nLa pérdida diaria de las {w(sem)} semillas se descompone, para los {w(n)} procedimientos, en efectos descriptivos de fecha, inicialización e interacción. No se supone independencia entre días ni se interpreta la descomposición como ANOVA inferencial.'
    r['atipicos_intro']=f'## Impacto de valores atípicos sobre el error e identidad de prueba\n\nLos umbrales de Tukey se fijan sobre log(y/y_lag168) en 2000–2017 y se aplican sin modificación a prueba. Se comparan errores en horas señaladas y restantes para los {w(n)} procedimientos; ninguna hora se elimina del benchmark.'
    r['repro_procedencia']=f'La evaluación integra {s["realizaciones"]} realizaciones finales de {w(n)} procedimientos, con {w(sem)} semillas cada uno, y {w(s["referencias"])} referencias deterministas. Las métricas se reconstruyen con predicciones conservadas y orígenes alineados. [Metadatos de ajuste y validación](_static/resultados/evaluacion_e2/metadatos_procedimientos.csv) y [cobertura de diagnósticos](_static/resultados/evaluacion_e2/cobertura_diagnosticos.csv) identifican el alcance de cada análisis. TGForecaster conserva cinco realizaciones por variante.'
    r['repro_entorno']='Los análisis documentales se ejecutan por CPU. El entrenamiento de las configuraciones seleccionadas utiliza `requirements-entrenamiento.txt` y `scripts/entrenar_arquitectura.py`, con `--dispositivo auto`, `cpu` o `cuda`. La instalación de PyTorch debe ser compatible con el dispositivo y su controlador. El código completo y los comandos por modelo se muestran en el {ref}`pipeline compartido <entrenamiento-portable>`. La comprobación funcional de las 15 configuraciones se ejecutó en Windows con CPU; Linux y GPU requieren comprobación en sus propios entornos.'
    return r


PREFIJOS={
 'El error depende de la hora:':'horizonte_interpret',
 'El menor MASE medio por horizonte corresponde a':'horizonte_interpret',
 '### Ajuste y parada interna':'ajuste_interno',
 'La parada interna presenta mayor error que el ajuste':'ajuste_conclusion',
 '### Perfil horario en una semana de prueba':'semana',
 '## Segmentos de evaluación':'segmentos',
 '## Diagnóstico de residuos y calibración':'residuos_intro',
 'Ljung–Box detecta dependencia residual':'residuos_resultado',
 'Ljung–Box a 14 días detecta dependencia residual':'residuos_resultado',
 '## Variación entre fechas e inicializaciones':'varianza',
 '## Impacto de valores atípicos sobre el error e identidad de prueba':'atipicos_intro'}


def actualizar_descargas(texto,s):
    etiquetas={
      'evaluacion_e2/metadatos_procedimientos.csv':f'Metadatos de los {s["procedimientos"]} procedimientos',
      'evaluacion_e2/precision_desarrollo_resumen.csv':f'Precisión de desarrollo de {s["procedimientos"]} procedimientos',
      'evaluacion_e2/precision_desarrollo_semillas.csv':f'{s["realizaciones"]} métricas de desarrollo',
      'diagnosticos_ampliados/latencia_semillas.csv':f'Latencia de {s["latencia_realizaciones"]} realizaciones'}
    for ruta,label in etiquetas.items():
        patron=r'\[[^\]]+\]\((_static/resultados/'+re.escape(ruta)+r')\)'
        texto=re.sub(patron,lambda m:'['+label+']('+m[1]+')',texto)
    return texto


def actualizar_codigo(texto):
    if 'e2_grupos' not in texto or 'resultados_horizonte.csv' not in texto:return texto
    grupos="""e2_keys=e2_benchmark.loc[e2_benchmark.semillas.gt(1),'clave'].tolist()
e2_grupos=[
    (e2_benchmark.loc[e2_benchmark.clave.isin(e2_keys)&e2_benchmark.entradas.eq('demanda_calendario'),'clave'].tolist(),'Demanda y calendario'),
    (e2_benchmark.loc[e2_benchmark.clave.isin(e2_keys)&e2_benchmark.entradas.eq('numerico_ampliado')&~e2_benchmark.clave.str.endswith('__N+O'),'clave'].tolist(),'Numéricos ampliados'),
    (e2_benchmark.loc[e2_benchmark.clave.isin(e2_keys)&e2_benchmark.clave.str.endswith('__N+O'),'clave'].tolist(),'Contrastes textuales')]
assert {k for ks,_ in e2_grupos for k in ks}==set(e2_keys)
assert sum(len(ks) for ks,_ in e2_grupos)==len(e2_keys)
"""
    import ast
    tree=ast.parse(texto);lines=texto.splitlines(keepends=True)
    # Regenerar sólo asignaciones de grupos y sus comprobaciones, conservando el trazado.
    inicio=next(n.lineno for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ['e2_keys','e2_grupos'] for t in n.targets))-1
    fin=next(n.lineno for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Tuple) for t in n.targets))-1
    return ''.join(lines[:inicio])+grupos+'\n'+''.join(lines[fin:])


def resumen_revision(s):
    return f"""
## Comprobaciones de completitud

- La selección contiene {s['procedimientos']} procedimientos aprendidos y {s['realizaciones']} realizaciones, además de {s['referencias']} referencias. Se conservan los diez modelos adicionales, tres baselines neuronales, ARX y el control numérico de TGForecaster.
- Curvas, observado–predicho, semana ilustrativa, horizontes, segmentos, residuos, calibración, variación y atípicos utilizan las configuraciones seleccionadas. Los grupos por horizonte se derivan del benchmark y verifican cobertura y ausencia de repeticiones.
- La precisión separa {s['pilotos']} pilotos externos de {s['procedimientos_parada']} reservas internas. Ajuste y parada tienen métricas en modo evaluación para catorce procedimientos; N-HiTS conserva sus curvas de pérdida. Son medidas con distintos alcances y unidades.
- La importancia ampliada cubre {s['importancia_ampliada']} procedimientos; la referencia de demanda/calendario se presenta por separado. La latencia cubre {s['latencia_procedimientos']} procedimientos y {s['latencia_realizaciones']} realizaciones. Los tres mecanismos documentales conservan sensibilidad al contenido.
- MASE es la métrica primaria; MAE y RMSE son complementarias. El margen es relativo, del 3 %. Los intervalos temporales, la dispersión entre semillas y las bandas visuales conservan interpretaciones distintas.
- Los capítulos 1 y 12 y las etiquetas de descarga se regeneran desde los resultados. Se eliminan párrafos repetidos dentro de una celda y se comprueba la cobertura de N-BEATS y TimesNet en el error por horizonte. Los CSV y las métricas del dashboard se contrastan con sus fuentes.
- El pipeline seleccionado y sus comprobaciones funcionales están documentados. La verificación física conservada corresponde a Windows con CPU; no demuestra ejecución en Linux o GPU.

## Pendientes metodológicos que la revisión documental no resuelve

| Aspecto | Evidencia disponible y trabajo pendiente |
|---|---|
| Preguntas e hipótesis | Existe una pregunta central y una hipótesis de mejora del texto. Falta articular al menos tres RQ con H0/H1, prueba, tamaño de efecto y criterio de decisión, como solicita E2 §1. Su formulación actual debe conservar alcance exploratorio. |
| Independencia de prueba y cronología | La exposición retrospectiva y el orden de registro están documentados. Un hash y la congelación actual no restauran independencia ni constituyen un preregistro anterior. |
| Potencia y presupuesto | Hay cinco semillas y dimensionamiento retrospectivo. No se acredita potencia 0,80 para todas las parejas; los presupuestos de selección son desiguales. |
| Revisión sistemática y priorización | Faltan cadenas y fechas originales de Scopus/WoS, conciliación completa, flujo PRISMA y revisión independiente. El denominador de comparadores contiene cinco estudios empíricos de demanda; parte de las referencias se caracteriza desde resúmenes. La priorización del Top 10 requiere su matriz reproducible. |
| Réplica del paper guía | Las adaptaciones arquitectónicas no constituyen por sí mismas una réplica bajo datos, partición y métrica originales. Falta esa evidencia y el criterio de éxito. |
| Búsqueda de arquitectura | E2 §14 aplica las 1000 configuraciones a modelos sin arquitectura predefinida. El espacio, las dos fases, los finalistas y sus análisis deben acreditarse para las familias a las que corresponda; no se exigen indiscriminadamente a todas las arquitecturas publicadas. |
| Robustez, ablación e incertidumbre | Desplazamientos y controles con pesos fijos no sustituyen una evaluación a tres severidades ni ablaciones por componentes. Faltan cobertura predictiva y techo de desempeño; las bandas entre semillas no los reemplazan. |
| Costo y atribución | Hay parámetros y latencia para todos; memoria pico y FLOPs no tienen cobertura completa. Entradas y presupuestos diferentes limitan atribuciones causales a arquitectura o texto. |
| Comparabilidad de TimesNet | Los artefactos conservan desarrollo con lote {s['timesnet_lote_desarrollo']} y prueba con lote {s['timesnet_lote_prueba']}. El lanzador de futuros ajustes homogeneiza la evaluación por origen; eso no cambia las métricas de desarrollo ya guardadas. |

Las prioridades son cerrar la trazabilidad bibliográfica y formular las preguntas explícitas, definir una comparación futura con protocolo homogéneo y completar después los experimentos que requieren nueva evidencia. Esta revisión no ejecuta entrenamiento, inferencia de redes ni publicación.
"""

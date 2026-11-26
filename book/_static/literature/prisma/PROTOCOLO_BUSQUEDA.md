# Protocolo de búsqueda sistemática y flujo PRISMA 2020

El anexo recibido el 26 de noviembre registra una búsqueda declarada el 25 de noviembre de 2026. Se conserva su versión original por separado. No se dispone de una etiqueta Git o de las exportaciones de ese día que permitan acreditar el registro previo o confirmar las cadenas efectivamente ejecutadas; las cifras siguientes se concilian desde el inventario adjunto.

## 1. Pregunta y alcance

- **Pregunta:** ¿qué arquitecturas de aprendizaje profundo se han evaluado como comparadores en el pronóstico de demanda eléctrica de corto plazo (horario, día siguiente) y con qué resultados?
- **Población:** series de demanda o carga eléctrica agregada (sistema, región o red de distribución).
- **Intervención y comparación:** modelos neuronales (MLP, LSTM, CNN, TCN, Transformer y variantes) evaluados frente a otros modelos.
- **Resultado:** métricas de error de pronóstico (MAE, RMSE, MAPE, MASE u otras).
- **Periodo:** publicaciones de 2018 a 2026.

## 2. Criterios de elegibilidad

| Código | Criterio de inclusión | Motivo de exclusión asociado |
| --- | --- | --- |
| I1 | El objetivo es demanda o carga eléctrica | E1 · otro dominio u otra variable (precio, solar, eólica, consumo de un edificio aislado) |
| I2 | Horizonte de corto plazo (≤ 7 días) | E2 · horizonte mediano o largo |
| I3 | Al menos un modelo neuronal evaluado | E3 · sin modelo neuronal |
| I4 | Compara dos o más modelos con métricas numéricas | E4 · sin comparación o sin métricas |
| I5 | Estudio primario | E5 · revisión, editorial o resumen de congreso sin resultados |
| I6 | Texto completo en inglés o español recuperable | E6 · texto completo no recuperado |
| — | — | E7 · retractado |

## 3. Cadenas de búsqueda

Ejecutar las tres el **mismo día**. Anotar fecha, hora y número de resultados en la tabla de la sección 4.

### Scopus (Advanced search)

```
TITLE ( "load forecasting" OR "load forecast" OR "electricity demand forecasting" OR "electricity demand forecast" OR "electric load forecasting" OR "power demand forecasting" OR "power load forecasting" )
AND TITLE-ABS-KEY ( "deep learning" OR "neural network*" OR lstm OR cnn OR "convolutional" OR transformer OR "temporal convolutional" OR mlp )
AND TITLE-ABS-KEY ( "short-term" OR "day-ahead" OR "day ahead" OR hourly OR "24-hour" )
AND TITLE-ABS-KEY ( comparison OR comparative OR benchmark OR "compared with" OR baseline* )
AND PUBYEAR > 2017 AND PUBYEAR < 2027
AND ( LIMIT-TO ( DOCTYPE , "ar" ) OR LIMIT-TO ( DOCTYPE , "cp" ) )
AND ( LIMIT-TO ( LANGUAGE , "English" ) OR LIMIT-TO ( LANGUAGE , "Spanish" ) )
```

### Web of Science Core Collection (Advanced search)

```
TI=("load forecasting" OR "load forecast" OR "electricity demand forecasting" OR "electricity demand forecast" OR "electric load forecasting" OR "power demand forecasting" OR "power load forecasting")
AND TS=("deep learning" OR "neural network*" OR LSTM OR CNN OR convolutional OR transformer OR "temporal convolutional" OR MLP)
AND TS=("short-term" OR "day-ahead" OR "day ahead" OR hourly OR "24-hour")
AND TS=(comparison OR comparative OR benchmark OR "compared with" OR baseline*)
AND PY=(2018-2026)
```

Refinar después en la interfaz: *Document Types* = Article y Proceedings Paper; *Languages* = English y Spanish.

### IEEE Xplore (Command search)

```
("Document Title":"load forecasting" OR "Document Title":"load forecast" OR "Document Title":"electricity demand forecasting" OR "Document Title":"power demand forecasting")
AND ("All Metadata":"deep learning" OR "All Metadata":"neural network" OR "All Metadata":LSTM OR "All Metadata":CNN OR "All Metadata":transformer)
AND ("All Metadata":"short-term" OR "All Metadata":"day-ahead" OR "All Metadata":hourly)
AND ("All Metadata":comparison OR "All Metadata":benchmark OR "All Metadata":comparative)
```

Filtro de años en la interfaz: 2018–2026.

**Si una base devuelve más de 1.500 registros**, añadir al bloque de modelos la condición de que aparezcan en el título (TITLE en Scopus, TI en WoS, "Document Title" en IEEE) y anotar el cambio en la sección 4.

## 4. Registro de ejecución (llenar al buscar)

| Base | Fecha y hora | Cadena usada (sin cambios / con cambio) | Resultados | Archivo exportado |
| --- | --- | --- | --- | --- |
| Scopus | 2026-11-25, archivo de las 22:07 | Por confirmar | 1.619 | scopus.csv |
| Web of Science | 2026-11-25, archivo de las 22:13 | Por confirmar | 873 | Web of Science.xls |
| IEEE Xplore | 2026-11-25, archivo de las 22:16 | Sin filtro de años en la interfaz; el script retira 132 registros anteriores a 2018 | 684 | IEE.csv |
| Otras fuentes (búsqueda pública, referencias citadas) | 2026-11-07 y 2026-11-08 | Consultas del registro `busqueda_demanda_20261108.json` | 6 | otras_fuentes.csv |

Scopus superó el umbral de 1.500 registros y no se acotó; la decisión queda registrada aquí. WoS asigna el año 2027 a 11 artículos de acceso anticipado publicados en línea en 2026; se conservan.

## 5. Procedimiento de cribado con dos revisores

1. `py -3.14 prisma.py unir` junta las exportaciones, elimina duplicados por DOI y por título normalizado y crea `cribado.csv`.
2. Cada revisor (A = Gloria, B = Fabián) llena **por separado** sus columnas de título y resumen (`incluir` / `excluir` + código E1–E7), sin ver las del otro.
3. Los desacuerdos se resuelven por consenso en `consenso_titulo`.
4. Los incluidos pasan a texto completo: se anota si se recuperó (`recuperado` = si/no) y cada revisor decide de nuevo (`tc_A`, `tc_B`, `consenso_tc`, `motivo_tc`).
5. `py -3.14 prisma.py diagrama` cuenta cada etapa, calcula el kappa de Cohen entre revisores y dibuja el diagrama PRISMA 2020.

## 5 bis. Cribado asistido por IA (declaración)

El 2026-11-25 se aplicó un cribado asistido por IA (Claude, de Anthropic) a los 1.910 registros de bases. Sus decisiones están en las columnas `ia_decision`, `ia_motivo` e `ia_regla` de `cribado.csv`, separadas de las columnas de los revisores humanos (A y B).

1. **Reglas explícitas** (`cribado_ia.py`) sobre el título y el tipo de documento: retractados (E7), revisiones (E5), edificios, hogares, consumidores individuales, vehículos eléctricos y otros dominios (E1), y horizontes mediano o largo (E2).
2. **Lectura de títulos** de los registros que las reglas no cubrieron. Las 62 exclusiones y las 5 correcciones de falsos positivos de la regla de revisiones quedan en `cribado_ia_manual.csv`.
3. **Ante la duda se incluye.** Lo no excluido pasa a texto completo, con `ia_regla = lectura_titulo`.

Resultado: 401 registros excluidos (E1 327, E2 52, E5 20, E7 2) y 1.509 incluidos.

### Enmienda 1 (2026-11-25): segunda pasada por resumen, criterio I4b

Con 1.509 registros incluidos por título, la fase de texto completo no era viable. Se añadió una segunda pasada del cribado sobre el resumen con un criterio más estricto, decidido **antes** de leer textos completos:

- **I4b:** las oraciones del resumen que comparan modelos nombran al menos **dos familias neuronales distintas** como comparadores (MLP/ANN, RNN, LSTM, GRU, CNN, TCN, Transformer, N-BEATS/N-HiTS, GNN). Los nombres híbridos del modelo propuesto, como CNN-LSTM, no cuentan como comparadores. Además, el resumen debe mencionar métricas de error. Si no cumple, se excluye con el código E4.
- También se excluye con E1 cuando el resumen declara un único edificio, hogar o consumidor.
- Los registros sin resumen útil pasan a texto completo.

Implementación: `cribado_resumen.py`, con las columnas `ia_resumen_decision`, `ia_resumen_motivo`, `ia_resumen_regla` e `ia_familias`. Resultado: 248 incluidos y 1.261 excluidos (E4 1.250, E1 11).

Verificación por lectura de una muestra de 28 oraciones comparativas (14 incluidos y 14 excluidos): unas 24 de 28 decisiones fueron correctas. Tras esa lectura se corrigieron los patrones FFNN y *multilayer*. Esta precisión es aproximada y debe confirmarla un revisor humano.

### Desviación respecto de la sección 5

El procedimiento con dos revisores humanos independientes (columnas A y B) no se ejecutó. El cribado definitivo es el asistido por IA de esta sección, con supervisión del procedimiento a cargo de Gloria Bastidas Rodriguez y sin verificación humana decisión por decisión. No se calcula kappa entre revisores. La fase de texto completo se limitó a los registros de otros métodos, ya revisados en texto completo en el capítulo 8. Los 248 registros elegibles de las bases quedan como elegibles por resumen. Para el diagrama PRISMA, cada revisor decide de forma independiente o, como mínimo, verifica las decisiones de la IA en una muestra aleatoria, y se informa el kappa entre revisor e IA.

## 6. Referencia

Page, M. J., et al. (2021). The PRISMA 2020 statement: an updated guideline for reporting systematic reviews. *BMJ*, 372, n71. https://doi.org/10.1136/bmj.n71


## 7. Ajustes de integración del 26 de noviembre de 2026

- Se distingue el flujo aritmético comprobado de la ejecución de cadenas declarada en el anexo. La carpeta indicada por el usuario contiene SCOPUS_TM_NLP.txt y WOS_TM_NLP.txt del3de noviembre y cinco archivos IEEE del6de noviembre; no corresponden a scopus.csv, Web of Science.xls e IEE.csv de los3176registros. No se sustituyen por exportaciones de otra búsqueda.
- Los248registros de bases son candidatos seleccionados por el resumen; no se presentan como248textos completos ni como extracciones verificadas de uso como baseline. El estudio adicional de Laitsos se conserva como extracción de texto completo.
- Las frecuencias originales son señales automáticas en frases comparativas. El grupo denominadoMLP incluye ANN,DNN,ELM yRBF: se rotula «Redes densas y afines» y no se utiliza como frecuencia verificada deMLP. CNN no acredita dimensión1D.
- cribado_resumen_corregido.py identifica las cláusulas que nombran referencias, conserva híbridos y separaMLP,ANN/DNN,ELM,RBF,N-BEATS yN-HiTS. Los casos ambiguos quedan pendientes, sin excluir por omisión del resumen. No se ha reaplicado al corpus del25de noviembre porque sus resúmenes no están incluidos.
- La exclusión de News and Load de esta síntesis de dos familias se codificaE4, noE3: el motivo es no acreditar dos familias neuronales como comparadores, no ausencia de una arquitectura neuronal.
- Los28casos de control se informan como comprobación interna declarada en el anexo; no constituyen una muestra aleatoria documentada ni validación independiente. No se calcula kappa con revisores inexistentes.
- La búsqueda complementaria y esta enmienda no reconstruyen una selección prospectiva anterior a los experimentos.

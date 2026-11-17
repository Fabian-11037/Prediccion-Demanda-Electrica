# Revisión de consistencia y completitud del JBook

16 de noviembre de 2026. Se contrastaron los capítulos, sus tablas y figuras, las fichas de arquitectura, el código mostrado, los resultados guardados, los enlaces internos y las pautas E1/E2. La revisión usa artefactos existentes; no inicia entrenamiento, inferencia de redes ni publicación.

## Resultado

La presentación y los resultados son consistentes después de las correcciones. El cumplimiento experimental de E1/E2 sigue siendo parcial: una entrega sin errores de construcción no acredita el cumplimiento de todos los requisitos metodológicos.

Inventario vigente: 13 cuadernos, 97 celdas analíticas, 53 figuras, 49 tablas (25 calculadas y 24 Markdown), 12 fichas y 24 esquemas. Benchmark: 13 procedimientos, 65 realizaciones, dos referencias, 105 parejas temporales, 78 entre semillas, tres contrastes textuales y 72 contrastes textuales por horizonte. PatchTST figura una sola vez con MASE 0,6725016176.

## Correcciones aplicadas

- El p de Levene del texto se actualiza de 0,000834 a 0,001526 según el protocolo vigente. La corrección se deriva del archivo de resultados para evitar otra cifra desactualizada al regenerar.
- Las referencias de literatura a MLP, LSTM y CNN1D enlazan sus fichas propias; antes apuntaban a §10.2, correspondiente a ARX.
- La síntesis bibliográfica incorpora TGForecaster y los controles de contenido ya calculados. La prensa, las ablaciones por fuente y la comparación con núcleo e hiperparámetros idénticos permanecen pendientes.
- Se distingue lo ejecutado de la ampliación propuesta: canal oficial frente a prensa, ventana principal frente a sensibilidades previstas, y fusión tardía evaluada frente a la propuesta de corrección acotada. El escenario estricto tiene auditoría de cobertura, no resultados predictivos equivalentes a los del declarado.
- Se explicita la ventana documental inclusiva [cierre − 72 h, cierre] y los siete cierres de TaTS. Se conserva el procedimiento real; no se retroajustan listas ni pesos.
- El diseño de métricas coincide con el benchmark integrado: MASE primario, MAE y RMSE complementarios. Se aclara que las métricas se calculan por semilla y después se promedian, incluyendo RMSE.
- La reproducción distingue construir HTML con salidas conservadas de recalcular los análisis. La revisión no certifica una ejecución nueva en Linux ni el entrenamiento en todas las combinaciones de hardware.
- El alcance del E2 explicita el conjunto principal de diez modelos entrenados, la réplica del paper guía y los registros de recursos faltantes.

Las correcciones persisten en scripts/consistencia_completitud_jbook.py, integrado en la armonización de la selección y en la reconstrucción de diagnósticos. Las salidas numéricas del benchmark, las configuraciones y las predicciones se conservan.

## Consistencia y completitud por componente

| Componente | Resultado | Evidencia o pendiente |
|---|---|---|
| Formulación, unidades y entradas | Consistente | Demanda nacional en kWh por intervalo, 168 h de historia, 24 h de horizonte. Covariables ampliadas y entradas de las cuatro familias base diferenciadas. |
| Datos, EDA y diccionario | Presentes | Calidad, distribución, transformaciones, ADF/KPSS, estacionalidad, clima e hidrología, atípicos y desplazamiento formal; código e interpretación junto a las figuras. 84 campos persistidos definidos. |
| Selección y prueba | Documentadas con limitaciones |4 años externos disponibles, confirmaciones distintas por procedimiento, prueba retrospectiva 2022–2025. La selección de PatchTST por error de prueba no se presenta como prospectiva. |
| Semillas, métricas y contrastes | Consistentes |5 semillas por procedimiento; MASE, MAE, RMSE, intervalos temporales y entre semillas; Holm, Friedman, Nemenyi, MCS, DM–HLN ybootstrap concordantes con los archivos. |
| Potencia y equidad | Parcial |No se acredita potencia 0,80; los presupuestos difieren. El dimensionamiento retrospectivo no determina cuántas semillas deban ejecutarse automáticamente. |
| Arquitecturas y pipelines | Presentes |12 fichas, 24 diagramas y 60 referencias a 52 archivos distintos; se comprueban existencia y sintaxis Python. No se ejecutan los pipelines de entrenamiento en esta revisión. |
| Corpus y disponibilidad | Consistente bajo escenario declarado |14.162 registros del caché, escenario y reglas identificados;9.497 cierres auditados sin documentos adelantados respecto a su disponibilidad declarada. La publicación histórica de esas versiones sigue sin certificarse. |
| Sensibilidad al texto | Presente con alcance condicionado |Retirada, contenido cero y contenido histórico con pesos fijos para 3 procedimientos; descripciones estáticas de TGForecaster separadas. No sustituye una ablación con reentrenamiento y núcleo idéntico. |
| Conjunto principal del benchmark | Diez modelos entrenados | ARX, MLP, LSTM, CNN1D, TFT, TCN, iTransformer, TaTS, fusión tardía y TGForecaster. PatchTST y N-HiTS son comparadores numéricos; TGForecaster tiene dos variantes. Se conservan trece procedimientos y dos referencias deterministas. |
| Referencias y paper guía | Parcial |ARX e ingenuos presentes; falta ARIMA o ETS según ruta temporal §7.4 S5. No se documenta réplica bajo dataset, partición y métrica originales del paper guía §5. |
| Robustez y ablación §4.6 | Parcial |Hay perturbaciones e intervenciones de contenido, pero no una batería de severidad de al menos 3 niveles por perturbación ni ablación de 3 componentes del mejor modelo con el diseño exigido. |
| Recursos §4.2 | Parcial |Parámetros, tiempos y latencia presentes. Solo N-HiTS conserva memoria pico: 416.907.776 bytes en cada semilla. Sin FLOPs registrados para 13 procedimientos. No confundir latencia del forward con costo del encoder o servicio completo. |
| Literatura §3 | Parcial |Revisión dirigida con fuentes y recuentos; faltan cadenas y fechas originales de Scopus/WoS, conciliación PRISMA y revisión independiente. 12 estudios derivados de abstracts permanecen fuera del denominador de comparadores de texto completo. |
| Reproducibilidad documental | Verificada localmente |86 pruebas correctas, 5 omitidas; construcción sin advertencias, errores ni enlaces rotos. 902 fragmentos HTML verificados en 35 páginas. No se ejecuta una instalación limpia en Linux durante esta revisión. |

## Recomendaciones priorizadas

1. **Aislar el aporte del texto.** Comparar una base PatchTST común con y sin documentos manteniendo núcleo, hiperparámetros, entradas, semillas y presupuesto. Conservar los controles de presencia, contenido cero e histórico y añadir ablaciones por fuente y disponibilidad. El resultado de TaTS es cercano al control; la dependencia de un canal no demuestra utilidad de noticias recientes.
2. **Completar las obligaciones experimentales del E2 antes de ampliar complejidad.** Documentar la selección de los diez modelos existentes, incorporar ARIMA o ETS como referencia y seleccionar un paper guía con dataset y código accesibles para la réplica exigida. Preparar robustez a 3 severidades y ablación de 3 componentes, evitando contar cambios de arquitectura o presupuesto como efecto puro del texto.
3. **Planificar prospectivamente potencia y prueba.** Fijar hipótesis primarias, δ = 3 %, familias de contrastes y presupuesto antes de nuevas corridas. Estimar precisión con un piloto adecuado; no adoptar mecánicamente las mayores estimaciones retrospectivas. Reservar una prueba con fechas posteriores al congelamiento efectivo y registrar entradas/pronósticos en tiempo real.
4. **Cerrar la evidencia bibliográfica.** Recuperar estrategias y fechas de búsqueda, conciliar duplicados y exclusiones, documentar PRISMA y someter la extracción a revisión independiente. Los abstracts sirven para localizar candidatos, no para respaldar comparaciones experimentales no verificadas.
5. **Completar costos y reproducción.** Medir memoria pico y FLOPs con entrada, batch y hardware definidos, e incluir el costo del encoder cuando se evalúe operación. Probar una reconstrucción desde copia limpia en Windows y Linux con dependencias fijadas y un inventario mínimo de archivos.

Estas recomendaciones quedan como pendientes. No se programan entrenamientos ni publicación.

## Evidencias

- results/revision_consistencia_20261116/auditoria.json: contratos, conteos, código, esquemas, ventanas y enlaces.
- results/revision_consistencia_20261116/recursos_por_semilla.csv: 65 registros de disponibilidad de memoria/FLOPs y sus fuentes.
- results/revision_consistencia_20261116/pruebas.log y construccion.log.
- results/revision_consistencia_20261116/pauta_E1.txt y pauta_E2.txt: extracción de 34 y 65 páginas, con límites de página.
- results/revision_consistencia_20261116/respaldo/: documentación antes de las correcciones.

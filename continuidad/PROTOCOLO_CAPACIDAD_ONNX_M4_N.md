# M4_N: capacidad igualada y exportación ONNX

Fecha: 23 de noviembre de 2026. Especificación para ejecución futura; no se han entrenado controles ni exportado modelos en esta actualización.

## 1. Propuesta y referencia histórica

**M4_N es la propuesta retrospectiva del proyecto:** memoria numérica de residuos de M1 fuera de muestra, recuperación aprendida y compuerta, sin canales de texto activos. Su designación se realiza después de conocer los resultados de 2022–2025. El registro futuro cubrirá los controles y conjuntos adicionales aún no evaluados, no convertirá esa elección en una selección a priori.

M4_OP conserva su condición de variante principal en la selección histórica de ocho configuraciones. M4_N hereda la configuración 6: 16 vecinos, 365 días, dimensión 32 y compuerta de ancho 64. No buscar ahora una configuración exclusiva de M4_N ni cambiar los pesos existentes. M4_O y M4_OP permanecen como comparaciones de sensibilidad al texto; no se atribuye al texto el rendimiento de M4_N.

Fuentes: `results/m4_gpu_20261122_194500/fuentes/scripts/e3_modelo4.py`, `final/M4_N/semilla_101/escaladores.json` y `final/M4_N/semilla_101/TERMINADO.json`. Las cifras de parámetros siguientes se derivan de esas fuentes y metadatos, sin ejecutar una red.

## 2. Qué se iguala

Comparar la **capacidad paramétrica activa del corrector** añadido a la misma base M1 congelada, no el número bruto de parámetros del contenedor. Mantener iguales entradas numéricas, normalización, horizonte, escala de pérdida, periodos, semillas, política temporal y presupuesto. Contabilizar aparte los parámetros y el coste de M1, compartidos entre procedimientos.

La implementación histórica de M4_N instancia módulos textuales que no utiliza y concatena 64 ceros a las variables numéricas. No contar como capacidad activa esos módulos ni las columnas de proyección multiplicadas siempre por cero. Registrar, por modelo: parámetros almacenados, entrenables declarados, activos en el camino de salida, congelados y excluidos con motivo. Un gradiente cero en un solo lote no prueba que un parámetro sea estructuralmente inactivo; revisar el grafo y verificar gradientes sobre varios casos, tras la inicialización si corresponde.

Para n variables numéricas, d=32 y ancho de compuerta w=64, la capacidad activa de las dos proyecciones con LayerNorm y la compuerta es:

\[
P_{M4_N}=2(32n+32+64)+(36\cdot64+64+64+1)=64n+2625.
\]

Con n=912, P_M4_N=60.993. El contenedor histórico registra 144.225 parámetros totales. Los 83.232 excluidos no intervienen en la salida numérica, incluidos 4.096 pesos de las columnas de texto constantes. Confirmar esta contabilidad con el grafo antes de un ajuste nuevo; no cambiar el checkpoint para hacer coincidir un conteo.

## 3. Controles concretos

| Procedimiento futuro | Composición del corrector y entrada | Capacidad activa para Colombia |
|---|---|---:|
| M4_N | Dos proyecciones numéricas de dimensión 32 con LayerNorm; atención sobre residuos y compuerta de ancho 64 | 60.993 |
| MLP_CE | Linear(912,65) → GELU → Dropout(0,1) → Linear(65,24); solo contexto numérico normalizado | 60.929; diferencia 0,105 % |
| EWMA_MLP_CE | Linear(936,63) → GELU → Dropout(0,1) → Linear(63,24); contexto numérico y 24 medias residuales exponenciales observables | 60.567; diferencia 0,698 % |

Ambos controles predicen una corrección normalizada de 24 componentes: y_hat=b_M1+q*c(x). Para EWMA_MLP_CE, cada componente adicional es la media exponencial del residuo correspondiente al horizonte, usando el mismo banco admisible de M4_N, retraso de dos días y ventana de 365 días. Fijar semivida de 30 días; no ajustarla en prueba. Si no hay episodios admisibles, aplicar identidad con M1 en los tres métodos. Este control tiene acceso a las mismas observaciones reveladas; difiere en cómo las resume y utiliza. MLP_CE, sin ese acceso, aísla el efecto de añadir parámetros, pero por sí solo no separa la arquitectura de la actualización informativa.

**Tolerancia:** |P_control−P_M4_N|/P_M4_N ≤ 0,05 para cada control, dataset y especificación. Los valores de Colombia anteriores son propuestas de diseño calculadas, no controles entrenados. En datasets con distinto n, fijar el ancho por conteo: para una MLP de entrada p, P(h)=h(p+25)+24; elegir h entero de 1 a 4.096 que minimice la diferencia absoluta, desempate por menor h, sin consultar pérdidas. Si no alcanza 5 %, registrar otro control antes de abrir métricas o marcar que la comparación no cumple; nunca añadir parámetros desconectados para simular igualdad.

No igualar parámetros de todas las arquitecturas del Top 10 alterando sus versiones canónicas. Reportarlas en su capacidad nativa, y presentar este grupo adicional para responder específicamente al efecto del mecanismo de M4_N. RATL/FreshCast conservarán versiones documentadas y controles de capacidad cuando su implementación lo permita; no etiquetarlos como igualados si no cumplen el criterio.

## 4. Entrenamiento futuro y comparación

Antes de cualquier ajuste, registrar fuentes y hashes, n, conteos por módulo, anchos, acceso al banco, semillas y presupuesto. M1, sus escaladores y sus predicciones OOF anidadas serán idénticos para los tres métodos. Usar pérdida MAE/q, AdamW con lr 0,001 y weight_decay 0,0001, lote 64, máximo 100 épocas, mínimo 10, paciencia 10 y min_delta 0,0001, como referencia de M4. La parada observa solo el periodo interno registrado. Mantener Dropout 0,1 en las capas indicadas; la capa final de los controles se inicializa a cero para comenzar con la base, sin modificar la inicialización histórica de M4_N. Registrar esta diferencia de parametrización.

No buscar anchos por desempeño ni asignar épocas extra al control que pierde. Documentar ajustes, actualizaciones, épocas efectivas, tiempo GPU, FLOPs y RAM/VRAM: igualdad de parámetros no significa igualdad de cómputo ni de capacidad estadística. El banco no paramétrico de M4_N se contabiliza aparte por número de episodios, bytes y coste de búsqueda.

Los controles nuevos y las réplicas de M4_N requieren autorización expresa. Planificar al menos diez semillas independientes o más según potencia 0,90, con el protocolo común y tres datasets adicionales del plan principal. Añadir los controles a la familia de hipótesis antes de evaluarlos; aplicar las pruebas temporales y Holm del plan. Conservar las familias históricas sin recalcularlas selectivamente ni sobrescribir los resultados anteriores.

## 5. Qué se exportará a ONNX

La exportación es una tarea futura sobre **pesos congelados**, sin ajuste ni cuantización. Debe separar el cálculo neuronal sin estado del controlador cronológico de memoria. ONNX no sustituye el control de disponibilidad ni mantiene por sí solo un historial creciente entre llamadas.

Paquete propuesto por semilla:

- `m4_n_encoder.onnx`: contexto numérico normalizado → consulta y clave de dimensión 32. Eliminar del grafo solo módulos textuales inactivos y columnas de entrada constantes, copiando exactamente los pesos activos; no reentrenar. La eliminación exige equivalencia posterior con el checkpoint original.
- `m4_n_corrector.onnx`: consulta, claves del banco, residuos normalizados de 24 horas, fechas, indicador de filas válidas, origen, predicción de M1 y escala q → predicción de 24 horas, compuerta, índices y pesos de vecinos. Incluir máscara causal antes de top-k, edad máxima, normalización segura y caso sin vecinos.
- Subgrafos ONNX de los expertos y router de M1, con contrato y transformaciones explícitos, si se entrega inferencia completa de extremo a extremo. Exportar solo el corrector no equivale a exportar todo M4_N: hasta validar M1, etiquetar el alcance como **corrector ONNX con base externa**.
- `pipeline_inferencia.py` futuro: lectura de variables y escaladores, ejecución de M1/encoder/corrector, banco con llegada de residuos observable, orden cronológico y persistencia de su estado con hash. Se probará en Windows y Linux sin rutas absolutas de otro equipo.

Para el contrato inicial, usar FP32, lote uno, 24 salidas y banco de 365 posiciones con padding y máscara explícita; máximo un episodio diario. El tamaño de lote y el banco se fijan por protocolo, sin suponer dimensiones dinámicas no verificadas. Las filas de padding tienen claves/residuos nulos y jamás son vecinos válidos. La máscara comprueba diferencia de fechas de al menos dos y no más de 365 días. El controlador solo ingresa residuos cuando se conoce toda su trayectoria y se cumple el retraso; el origen actual y sus objetivos no son entradas.

Refactorizar en un wrapper puramente tensorial para exportación: no capturar como constantes decisiones dependientes de `.item()`, índices Python o el objeto Banco. No cambiar el algoritmo por dificultades del exportador. Conservar la regla de empate de top-k y documentar casos próximos al umbral; si otra plataforma selecciona vecinos con predicciones fuera de tolerancia, no aprobar esa exportación ni corregirlo agregando ruido a los puntajes.

Usar el [exportador oficial de PyTorch](https://docs.pytorch.org/docs/stable/onnx_export.html) y ONNX Runtime. Fijar versiones de PyTorch, ONNX, runtime y opset compatible en el manifiesto antes de exportar; emplear `torch.onnx.export` con `dynamo=True`, `report=True` y `verify=True` cuando lo soporte el entorno fijado. El reporte automático no sustituye el recorrido temporal completo. Consultar los [proveedores de ejecución de ONNX Runtime](https://onnxruntime.ai/docs/execution-providers/): CPU obligatorio y GPU solo si el proveedor está disponible y su combinación de controlador/librerías está validada. Registrar dónde se ejecuta cada parte y cualquier fallback.

## 6. Criterios de aceptación de la exportación

Registrar antes de inferir los casos de equivalencia; no reajustar tolerancias después de ver errores. Evaluar cada semilla exportada frente a su propio checkpoint original en modo eval, FP32 y los mismos insumos. El conjunto de comprobación se define por fechas/condiciones, no buscando los casos donde ONNX funciona.

| Verificación futura | Criterio |
|---|---|
| Grafo y artefactos | Comprobación ONNX válida; inputs/outputs, tipos, nombres, dimensiones y hashes documentados |
| Valores numéricos | Sin NaN/Inf; para predicción y corrección normalizadas por q, abs(a−b) ≤ 1e−5 + 1e−5·abs(b) por componente; igual criterio en consultas, claves, compuertas y pesos |
| Caso sin vecinos | Salida exactamente igual al tensor de base suministrado; compuerta y pesos de vecinos válidos nulos |
| Causalidad | Padding, filas futuras y episodios de ayer no afectan la predicción; solo vecinos admisibles; misma secuencia de actualizaciones del banco |
| Extremos | Memoria vacía, menos de 16 vecinos, al menos 16, límite de dos/365 días, empates y fechas faltantes |
| Recorrido temporal | Equivalencia diaria de predicciones, memoria y escala en desarrollo; comparación retrospectiva guardada solo si se autoriza esa inferencia |
| Métricas | Diferencia absoluta de MASE ≤ 1e−5 sobre el recorrido autorizado; este criterio no sustituye la equivalencia por observación |
| Portabilidad | CPU en Windows y Linux; GPU en los proveedores que se declaren compatibles, sin asumir equivalencia por haber pasado CPU |

Si un criterio falla, guardar el caso y bloquear la etiqueta de exportación validada; no cambiar pesos, sustituir vecinos ni ajustar tolerancias para aprobar. La exportación no es una ablación ni cuenta como otra semilla o arquitectura. Medir latencia por separado, con calentamiento, sincronización cuando corresponda, mediana y p95, incluyendo o separando explícitamente base, búsqueda y actualización del banco.

## 7. Artefactos y estado

Guardar en una carpeta nueva por semilla los ONNX y pesos externos que utilicen, wrapper, controlador, escaladores, contrato JSON, requisitos fijados, informes de exportación, pruebas de equivalencia, latencia y manifiesto SHA-256. Conservar checkpoint original y fuente exacta que lo produjo. Un `.onnx` aislado sin la base, estado y transformaciones no es un paquete ejecutable completo.

El preregistro futuro incluirá la designación retrospectiva de M4_N, control de capacidad, matrices de confirmación y contrato ONNX. Hoy solo se actualiza documentación: los controles no están entrenados, los archivos ONNX no se han generado y no se han ejecutado inferencia, prueba ni despliegue del JBook/dashboard.

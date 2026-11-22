# Cuatro modelos propuestos: diseño y continuidad en GPU

Fecha: 22 de noviembre de 2026. Esta guía amplía `ENTRENAMIENTO_GPU_E3.md`; los diseños están especificados en `DISENO_CUATRO_MODELOS_GPU.json`. **Documentación de diseño; se autoriza su carga al repositorio, sin entrenar, evaluar prueba ni publicar JBook o dashboard.** Una ejecución futura requiere una solicitud expresa. Los cuatro modelos aún no tienen implementación validada; no confundir este documento con un paquete ejecutable terminado.

## Evidencia que motiva los diseños

La exploración numérica evaluó 24 configuraciones en 2018–2019 con cinco semillas. PatchTST alcanzó MASE 0,4107; el apilamiento de todos los expertos, 0,3720 (9,41 % menos), y el apilamiento de controles exportables, 0,3883 (5,44 % menos). Son resultados de desarrollo. La propuesta de agregación anclada no superó a las referencias convencionales.

El corrector se evaluó en 2019–2021 con tres semillas. C+R alcanzó MASE 0,54671; corrector numérico, 0,54186; texto oficial reciente, 0,54169. La diferencia descriptiva entre los últimos dos fue 0,03 %. Las variantes de anuncios alineados y oficial más prensa no superaron al control numérico. Esto motiva explotar complementariedad y exigir evidencia incremental del contenido, sin presuponer que los nuevos diseños mejorarán.

Los ensambles y la modulación con texto tienen antecedentes. Las contribuciones propuestas deben contrastarse con [MoLE](https://proceedings.mlr.press/v238/ni24a.html), [MoME](https://arxiv.org/abs/2601.21547), [RAFT](https://proceedings.mlr.press/v267/han25d.html) y [FreshCast](https://arxiv.org/abs/2610.07834). La originalidad no queda demostrada por combinar sus componentes. Registrar la diferencia metodológica y las ablaciones que la sostienen.

Los contextos, las dos guías y las configuraciones se distribuyen en el repositorio bajo `continuidad/`. Leer primero `README_GPU.md`. Las referencias a los resultados originales de la exploración identifican su procedencia; esos paquetes no se incluyen en esta subida documental. Las configuraciones completas de los expertos sí se incluyen en los JSON.

## 1. M1: ensamble supervisado por complementariedad de errores

**Hipótesis:** anticipar cuándo se compensan los errores de dos expertos aporta más que escoger el de menor error individual.

Usar tres expertos numéricos congelados por partición: PatchTST·control, E·control y periodica·control. Sus arquitecturas y parámetros proceden de `CONFIGURACIONES_PRUEBA_E3_GPU.json`, no de una búsqueda nueva. Cada experto recibe las mismas entradas y produce 24 pronósticos.

La red de selección recibe calendario conocido, resúmenes de demanda, clima, hidrología y precio disponibles, pronósticos de los expertos, su desacuerdo y errores observados con retraso de dos días. Una MLP genera logits por hora y experto. Aplicar softmax entre expertos:

\[
\hat y_{d,h}=\sum_k w_{d,h,k}p_{d,h,k},\qquad w_{d,h,k}\geq0,\quad\sum_k w_{d,h,k}=1.
\]

Una cabeza auxiliar estima complementariedad por pareja. Con errores fuera de muestra \(r_i=y-p_i\), la etiqueta de ajuste es

\[
c_{ij}=\frac{|r_i|+|r_j|-|r_i+r_j|}{q_{168}}.
\]

Es una señal de compensación de signos; no garantiza que la mezcla futura mejore. La etiqueta real solo se utiliza para ajustar la cabeza; al predecir se usa su estimación, nunca el error del día objetivo. Pasar esa estimación al selector. Minimizar MAE escalado más Huber auxiliar y una penalización fija de suavidad entre pesos de horas contiguas, de 0,001. Para evitar realimentación ficticia, todas las predicciones de entrenamiento de los expertos son fuera de muestra.

Rejilla de ocho candidatos: ancho 32/64, dropout 0/0,1 y coeficiente auxiliar 0,01/0,1. Comparar con NNLS por hora, mezcla convexa fija, media simple, EWA con retraso dos y la misma red sin supervisión de complementariedad. NNLS no exige suma de pesos igual a uno: conservarlo como referencia y declarar esa diferencia de clase funcional. No prometer que un selector aprendido nunca empeora.

## 2. M2: energía diaria y distribución horaria

**Hipótesis:** separar error de nivel diario y error de distribución horaria facilita una adaptación más precisa que corregir conjuntamente 24 salidas.

Usar el codificador de PatchTST·control congelado y adaptar dos cabezas pequeñas. La representación concatena estado numérico, calendario y errores ya observables. La primera cabeza estima energía diaria positiva mediante softplus, con escala estimada únicamente en ajuste. La segunda modifica el perfil horario ancla de PatchTST:

\[
\hat E_d=s_E\operatorname{softplus}(f_E(z_d)),\qquad
s_{d,h}=\operatorname{softmax}_h\bigl(\log(s^0_{d,h}+10^{-8})+f_s(z_d)_h\bigr),\qquad
\hat y_{d,h}=\hat E_d s_{d,h}.
\]

Construir \(s^0\) normalizando la predicción positiva de PatchTST; cualquier recorte numérico y su epsilon deben fijarse antes de validar. La suma de las 24 salidas coincide por construcción con la energía estimada. Esta identidad garantiza consistencia interna, no precisión.

Pérdida: MAE horario dividido por q168, más lambda por el error absoluto de energía dividido por 24·q168. Rejilla de ocho candidatos: ancho 32/64, dropout 0/0,1 y lambda 0/0,1. Ablaciones: solo energía, solo distribución, ancla intacta y cabeza directa de 24 salidas con capacidad comparable. Separar en los diagnósticos el error de energía diaria y el error del perfil, además del MASE horario principal.

## 3. M3: selección de expertos condicionada por texto

**Hipótesis:** el texto puede anticipar qué combinación de expertos funcionará mejor, aun cuando no explique directamente el residuo horario.

Extender M1 manteniendo sus expertos y selector numérico congelados por partición y semilla. Crear un agregador de fragmentos oficiales y otro de prensa; ambos usan E5 congelado y atención condicionada por el contexto numérico. Su relevancia se aprende con el objetivo de pronóstico. No añadir listas de palabras, clases temáticas ni fuentes seleccionadas por su desempeño en prueba.

La nueva cabeza produce una perturbación acotada de los logits numéricos:

\[
w^T=\operatorname{softmax}\!\left(l^N+I_{\mathrm{texto}}g\varepsilon\tanh(\Delta l)\right),\qquad
\hat y^T_{d,h}=\sum_k w^T_{d,h,k}p_{d,h,k}.
\]

Si no hay candidatos válidos, la salida debe ser exactamente la de M1. La compuerta es aprendida y no constituye una garantía de no empeorar. Penalizar la magnitud de la perturbación para favorecer cambios justificados por la pérdida. Conservar la misma supervisión predictiva por complementariedad del diseño numérico; el texto solo modifica el selector.

Rejilla de ocho candidatos, elegida solo con la variante oficial+prensa: dimensión textual 32/64, penalización de delta 0,001/0,01 y epsilon de logits 0,25/0,5; dropout fijo 0,1. Después congelar esos parámetros para oficial, oficial+prensa, metadatos y los cinco desplazamientos del texto. El control de metadatos usa canal, fuente, antigüedad y volumen, sin embeddings ni intervalos extraídos semánticamente. Comparar además con una adaptación numérica de capacidad equivalente y con eliminación de cada canal.

La base numérica de cada fila de ajuste debe ser una predicción de M1 fuera de muestra. En el fold Y, M1 y su configuración solo pueden depender de años anteriores a Y; no reutilizar el ganador global elegido con 2018–2021 para generar retrospectivamente esas filas.

## 4. M4: memoria de episodios numéricos y textuales

**Hipótesis:** situaciones pasadas con texto y estado numérico similares pueden aportar una trayectoria residual útil, con menos necesidad de memorizar acontecimientos poco frecuentes en los pesos.

Mantener M1 como base. Para cada episodio guardar clave numérica, representaciones oficiales y de prensa por separado, disponibilidad y residuo horario fuera de muestra. Proyectar claves y consultas a dimensión 32. Recuperar k episodios anteriores dentro de una edad máxima; aprender atención y una compuerta según similitud, dispersión de los residuos, cantidad de respaldo y contexto actual:

\[
\hat y_{d,h}=p^{M1}_{d,h}+q_{168}g_d\sum_{j\in\mathcal M_d}\alpha_{d,j}v_{j,h},\qquad
v_{j,h}=\frac{y_{j,h}-p^{M1,OOF}_{j,h}}{q_{168}}.
\]

Entrenar con MAE escalado. La recuperación y la compuerta se supervisan por utilidad para el pronóstico; la semejanza textual no se trata como evidencia automática de que la respuesta energética será igual. Si no hay vecinos elegibles, devolver M1. Evitar acceso al propio episodio, a resultados futuros y a memoria construida según el orden aleatorio de minibatches: la elegibilidad siempre se calcula por fecha real.

Rejilla de ocho candidatos: k=8/16, edad máxima=365/730 días y ancho de compuerta=32/64. Dropout fijo 0,1. Seleccionar con oficial+prensa y aplicar la misma configuración al control exclusivamente numérico, oficial, metadatos y texto desplazado. Todos comparten fechas, cupos de memoria y regla de disponibilidad. Ablaciones: memoria fija en 2021, sin compuerta y sin contexto numérico.

Política principal futura: memoria creciente con episodios completos disponibles en d−2 o antes. Durante prueba los pesos, escalas y métrica de recuperación permanecen congelados; incorporar una observación ya disponible a la memoria no autoriza a actualizar el optimizador. Identificar esta evaluación como secuencial. La memoria congelada en 2021 permite separar el efecto de recibir observaciones recientes del aporte del texto.

## 5. Datos, GPU y preparación

Aplican las instrucciones de entorno y traslado de `ENTRENAMIENTO_GPU_E3.md`. Recuperar el corpus mediante Git LFS y verificar hashes. Usar el panel y las 30 variables históricas y 23 de calendario del contrato, con máscaras de ausencia y rezagos. El clima es reanálisis y nunca se trata como clima futuro conocido. Ajustar imputación, escalas, mapas de fuentes y cualquier proyección de datos solo dentro de cada tramo de ajuste.

Instalar PyTorch compatible con NVIDIA/CUDA o AMD/ROCm desde la documentación oficial enlazada en la guía general. Comprobar operación y retropropagación real en GPU; si falla, detenerse, sin traslado silencioso a CPU. Registrar GPU, VRAM, sistema operativo, versiones, precisión y dependencias. Mantener FP32 como referencia; cualquier uso de AMP necesita verificación numérica y quedar registrado antes de la corrida comparativa. No cambiar el lote silenciosamente por un error de memoria.

Para M3 y M4, E5: `intfloat/multilingual-e5-base`, revisión `d128750597153bb5987e10b1c3493a34e5a4502a`, prefijo `query: `, pooling con máscara y normalización L2, máximo 512 tokens. Conservar fragmentos de 1.500 caracteres, solape 200, máximo 16 por documento y candidatos del contrato del corrector: hasta 32 alineados, 48 oficiales y 48 de prensa, ventanas de 72 horas y alineados hasta 120 días. No concatenar URL, rutas o fechas de recuperación al contenido. Registrar tokens, fragmentos truncados, ausencia y cobertura por canal.

Crear cachés versionados por hash de texto, codificador, reglas y escenario. No sustituir estos vectores por los del corpus congelado de 14.162 documentos. Construir primero los datos anteriores a 2022. Congelar vocabulario y mapas de fuentes; reservar el identificador desconocido para fuentes posteriores. En la prueba aplicar exactamente las reglas congeladas, sin ajustar E5 ni cambiar el número de candidatos. El escenario principal sigue siendo declarado; documentar la incertidumbre de versión histórica y mantener la sensibilidad estricta separada.

## 6. Predicciones fuera de muestra y selección

Antes de entrenar una red superior, generar o verificar OOF de los tres expertos para años anteriores a su evaluación. Se propone historia anual 2011–2021 y cinco semillas 101–105. Reutilizar predicciones existentes solo cuando coincidan configuración, semilla, año, contrato, disponibilidad, purga y hashes. Los datos 2018–2019 existentes no completan por sí solos los folds de 2020–2021. No hay checkpoints de la exploración recibida.

Para cada año Y, los expertos ajustan solo con años anteriores a Y y purga de 24 horas, usando la reserva cronológica del contrato para early stopping. Ninguna etiqueta de Y aparece en sus escalas o pesos. Para M1, crear también una segunda capa OOF: sus filas históricas y residuos se producen con una red que no vio sus etiquetas. El primer año posible de esta capa es 2013 si se dispone de ajuste 2011 y parada 2012. Registrar el arranque real y las filas no elegibles.

La configuración de M1 utilizada por otros modelos se selecciona exclusivamente con el prefijo temporal disponible. Por ejemplo, para pronosticar Y, se pueden seleccionar candidatos en Y−1 con early stopping en Y−2 y ajuste hasta Y−3; luego reajustar el elegido con datos hasta Y−2, parar en Y−1 y predecir Y. Nunca importar a ese prefijo una configuración elegida con años posteriores. Guardar las decisiones por prefijo. La configuración final, para 2022–2025, sí puede elegirse usando todo el desarrollo anterior a 2022.

Desarrollo común: folds 2018, 2019, 2020 y 2021; ajuste de las redes superiores con OOF hasta Y−2, parada en Y−1 y evaluación en Y. Usar los mismos orígenes de evaluación; la ausencia de texto o memoria produce retorno a la base, no eliminación de días. Las métricas de desarrollo empleadas en la selección se reportan como tales, sin presentarlas como una evaluación independiente del proceso de búsqueda.

Cada diseño tiene ocho configuraciones de su módulo nuevo. Usar cinco semillas comunes y registrar presupuesto y tiempo completo, distinguiendo cómputo reutilizado de expertos. Seleccionar por media MASE de folds y semillas, con igual peso para cada fold; en empate exacto, menor número de parámetros y luego menor ID. Las ramas de texto seleccionan una sola configuración con oficial+prensa; los controles y ablaciones conservan esos parámetros y no se afinan por separado.

Receta propuesta común para módulos nuevos: AdamW, lr=0,001, weight_decay=0,0001, lote 64, recorte de gradiente 1, hasta 100 épocas, mínimo 10 y paciencia 10. Monitorizar exclusivamente MASE de parada, con mejora mínima absoluta 0,0001, y restaurar el mejor checkpoint. Mantener la receta original de los expertos; no reemplazarla por la de las cabezas. Estos son parámetros de diseño por validar en desarrollo, no hiperparámetros ya seleccionados ni ejecutados.

## 7. Ajuste final y prueba

Antes de abrir nuevas métricas, fijar las configuraciones, comparadores, política de memoria, semillas, contrastes, presupuesto y hashes en un registro fechado. La fecha es la real: no retrofechar. Los modelos nuevos se proponen después de conocer resultados del benchmark 2022–2025; documentar esa selección informada y no denominar al bloque prueba nunca consultada.

Ajustar expertos finales con `ajuste_final()` y su reserva previa a 2022. Ajustar los módulos nuevos sobre sus filas OOF elegibles hasta 2020 y detener con 2021. Registrar los rangos exactos. Guardar todos los pesos y transformaciones antes de predecir 2022–2025. No repetir la selección después de ver prueba.

Prueba: 1.461 orígenes diarios entre 2022-01-01 y 2025-12-31, 00:00 Colombia, historia 168 horas y horizonte 24. Las entradas numéricas son anteriores al origen; documentos solo con disponibilidad menor o igual al origen. Los errores usados como rasgos y los valores incorporados a memoria deben ser observables con retraso de dos días. La política secuencial no permite reentrenamiento con etiquetas de prueba.

TimesNet no es experto del diseño principal exportable. Si se agrega como comparación secundaria, registrar esa extensión y usar lote uno en parada, inferencia y prueba; no decidirlo por el resultado obtenido.

## 8. Métricas, atribución e inferencia

Métrica principal: MASE con q168=215.617,90361589016 kWh. Informar también MAE y RMSE en kWh y sus conversiones a GWh cuando se utilicen, por año, horizonte y segmentos definidos antes de evaluar. Calcular las métricas de cada semilla primero y promediar después; informar dispersión entre semillas y por separado el resultado del promedio de pronósticos si se muestra un ensamble de inicializaciones.

La unidad temporal es el origen diario. No contar 24 horas ni cinco semillas como observaciones diarias independientes. Bootstrap pareado de 9.999 réplicas, bloques de 14 días dentro de cada año, con sensibilidad a 7 y 28 días. Los IC temporales condicionados a las cinco semillas no sustituyen una inferencia sobre nuevas inicializaciones; describir su alcance.

El JSON enumera 14 parejas primarias: M1 frente a NNLS y mejor individual de desarrollo; M2 frente al mejor individual y M1; M3 oficial y oficial+prensa frente a M1, y oficial+prensa frente a metadatos y desplazados; las comparaciones equivalentes de M4 contra su memoria numérica; y el incremento de prensa frente a oficial en ambos diseños. No redefinir el mejor individual con prueba. Los desplazados se resumen como media de pérdidas de las cinco versiones, no como promedio arbitrario de sus predicciones.

Para cada pareja registrar dos hipótesis unilaterales, ganancia relativa menor o igual a cero y menor o igual a 3 %. Ajustar Holm conjuntamente sobre las 28 hipótesis; conservar efectos, IC y p sin ajustar y ajustados. La superioridad estadística y superar el margen del 3 % son conclusiones diferentes. Implementar el contraste de margen con remuestreo pareado centrado bajo ese margen; no deducir significación al 3 % de un porcentaje puntual mayor a 3. Si se evalúa equivalencia, registrar TOST con margen relativo ±3 %, dependencia temporal y su familia de multiplicidad antes de la prueba. p mayor a 0,05 no demuestra equivalencia.

La atribución semántica exige superar el control numérico correspondiente, los metadatos y el texto desplazado; una mejora frente a una base más débil no basta. M4 necesita comparar recuperación numérica y textual bajo idéntica política de memoria. Usar las mismas configuraciones y presupuesto en controles. Los análisis adicionales se identifican como secundarios y su multiplicidad se fija antes de mirar resultados.

## 9. Implementación y entregables pendientes

No hay un comando validado que entrene estos cuatro modelos. Implementar adaptadores separados y un lanzador GPU, sin modificar los paquetes de resultados ni sobrescribir su código. La secuencia futura es: verificar insumos y GPU; preparar OOF de expertos; preparar OOF anidado de M1; entrenar y seleccionar en desarrollo; fijar configuraciones y variantes; realizar ajuste final; congelar checkpoints; predecir secuencialmente; evaluar y redactar el informe.

Antes de una ejecución extensa comprobar con datos anteriores a 2022: finitud y gradientes, identidad de M3 sin texto, identidad de M4 sin vecinos, suma de salidas de M2 igual a su energía, invariancia al modificar información posterior al origen y restauración exacta de predicciones desde checkpoints sin recalcular escalas. Son validaciones de implementación pendientes; no afirmar que se ejecutaron. Incluir verificaciones de índices de memoria y ausencia de acceso al propio objetivo.

Guardar en `results/modelos_originales_e3_gpu_<fecha_hora>/`: registro, configuración elegida por modelo y por prefijo OOF, historias, checkpoints completos, escalas, mapa de fuentes, matriz completa de pesos NNLS por hora, caché e índices de texto, memoria inicial y log de actualizaciones, predicciones por origen/hora/semilla, métricas, contrastes, selección y manifiestos SHA-256. Guardar optimizador y generadores aleatorios para reanudar. Las predicciones archivadas permiten calcular nuevos diagnósticos sin repetir el entrenamiento.

Registrar latencia por origen, parámetros, memoria máxima y tiempos de ajuste, preparación textual e inferencia, distinguiendo caché frío/caliente. Comprobar exportación ONNX con varias entradas diferentes de la usada para trazar y documentar el tratamiento de recuperación, actualización de memoria y tokenización fuera del grafo. No afirmar exportabilidad completa de M4 porque se exporte únicamente su red de atención.

Mantener corpus y benchmark anteriores intactos. Compartir los resultados con el usuario antes de modificar JBook o dashboard. No publicar ni ejecutar el trabajo futuro como consecuencia de leer esta guía.

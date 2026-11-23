# M4: originalidad y evaluación en GPU

Fecha: 23 de noviembre de 2026. Estado: plan de continuidad; ejecución pendiente.

## 1. Objetivo y alcance

Determinar si la memoria de episodios de M4 constituye una contribución metodológica para el pronóstico de demanda eléctrica y completar su evaluación conforme al Entregable 3. Una mejora de error no demuestra por sí sola originalidad; ambas hipótesis requieren evidencia propia.

La solicitud actual autoriza actualizar y subir documentación. **No autoriza entrenar, ejecutar inferencia, ampliar la evaluación ni modificar o publicar JBook o dashboard.** El agente del PC con GPU deberá obtener una solicitud expresa antes de iniciar ajustes nuevos. Esta guía no es un preregistro de experimentos ya ejecutados ni activa una cola de entrenamiento.

## 2. Punto de partida que debe conservarse

Los pesos, configuraciones, predicciones y fuentes ya están publicados en `results/m4_gpu_20261122_194500/`. Consultar `INFORME.md`, `fuentes/scripts/e3_modelo4.py`, `fuentes/scripts/test_m4_gpu.py` y los manifiestos de `results/publicacion_m2_m3_m4_20261122_194500/`. La base procede de M1 con OOF anidado; M1 y el codificador E5 permanecen congelados.

M4_OP fue la variante principal de selección: oficial más prensa, escenario declarado y canales separados. La configuración elegida es 6: 16 vecinos, edad máxima de 365 días y ancho de compuerta 64; proyección de consulta/clave de dimensión 32. Se cierra la búsqueda amplia de variantes y se conserva esta configuración como referencia histórica.

Las cinco semillas existentes son 101–105. La prueba comprende 1.461 orígenes diarios de 2022–2025 y 24 horas por origen. MASE: M4_OP 0,578898; M4_N 0,573144; M4_O 0,571659; sin compuerta 0,572099; memoria fija 0,630091; TimesNet con lote uno 0,642928. Las cifras proceden de los resultados guardados, no de ajustes ejecutados en esta actualización.

M4_OP mejora aproximadamente 9,96 % respecto de TimesNet en la comparación temporal retrospectiva disponible. El texto oficial no demuestra una mejora incremental frente al control numérico: reducción de 0,259 % y p Holm 1 en la familia registrada. Oficial más prensa y la compuerta tampoco acreditan un beneficio incremental. No convertir M4_O, M4_N o un placebo en la variante principal por su menor error de prueba.

Los contrastes actuales usan bootstrap temporal y cinco semillas; no satisfacen por sí solos las tres pruebas distintas, el mínimo de diez semillas ni la potencia del E3. La prueba 2022–2025 ya fue examinada y seguirá identificada como retrospectiva. Añadir semillas no la convierte en una prueba ciega.

## 3. Primera decisión: contribución frente a métodos cercanos

Antes de gastar GPU, completar una matriz de cinco a diez trabajos cercanos con texto completo, ecuaciones, código, objeto recuperado, construcción de memoria, disponibilidad, combinación, entrenamiento y aplicación. El archivo `MATRIZ_NOVEDAD_M4.md` inicia esa revisión; distingue lecturas completas de evidencia limitada al resumen.

Prioridad: [RATL](https://arxiv.org/abs/2609.03937), que recupera residuos de una base congelada y aprende a combinarlos con restricciones de disponibilidad, y [FreshCast](https://arxiv.org/abs/2610.07834), que actualiza memoria con observaciones reveladas después del entrenamiento y calibra su contribución. Recuperar residuos, actualizar memoria o incluir una compuerta no bastan como afirmaciones de novedad. Revisar también [RAFT](https://proceedings.mlr.press/v267/han25d.html), [MoME](https://arxiv.org/abs/2601.21547) y [MoLE](https://proceedings.mlr.press/v238/ni24a.html).

La contribución candidata es la **construcción temporal anidada y fuera de muestra de una memoria de errores, combinada con recuperación aprendida y actualización sujeta al retraso de observación**. Hay que establecer qué parte difiere efectivamente de esos antecedentes y qué efecto tiene cada componente. El uso de datos colombianos, otro horizonte o hiperparámetros distintos no constituye por sí solo una contribución metodológica.

Registrar fecha real de búsqueda, bases consultadas, cadenas booleanas, versiones de los artículos, criterios de inclusión y decisiones de cribado. Usar, al menos, búsquedas equivalentes a:

```text
("load forecasting" OR "electricity demand forecasting") AND
("residual memory" OR "retrieval augmented" OR "episodic memory" OR "residual correction")

("time series forecasting") AND
("retrieved residuals" OR "online memory" OR "out-of-fold" OR "cross-fitted")
```

Esas son cadenas propuestas, no búsquedas retrospectivamente atribuidas a Scopus o Web of Science. Si se dispone de un segundo revisor, documentar la revisión independiente y resolver discrepancias. Actualizar la búsqueda antes de la entrega final.

**Criterio de avance:** una diferencia algorítmica verificable, formulada con precisión y pertinente para la tarea. Si la revisión no la establece, presentar M4 como adaptación o combinación evaluada, sin afirmar originalidad demostrada. No abrir una búsqueda extensa de arquitecturas para obtener un resultado favorable.

## 4. Formulación y propiedad teórica

Para un origen t, sean b_t la predicción congelada de M1, q la escala de error semanal calculada solo con entrenamiento y r_i=(y_i-b_i)/q una trayectoria residual histórica fuera de muestra. Con vecinos admisibles A_t:

\[
\alpha_{ti}=\operatorname{softmax}_{i\in A_t}(s_\theta(x_t,x_i)),\qquad
\widehat y_t=b_t+q\,g_\theta(x_t,A_t)\sum_{i\in A_t}\alpha_{ti}r_i.
\]

La compuerta está en [0,1], las ponderaciones suman uno y la máscara se aplica antes de top-k. Solo ingresan episodios d−2 o anteriores y dentro de la edad máxima. Sin vecinos, la salida es exactamente b_t. No hay garantía general de reducción del error ni de positividad de la predicción.

Desarrollar una demostración de estabilidad **local** de la corrección con memoria fija. Suponer residuos acotados por R, escala q fija, inferencia determinista y margen entre los puntajes del vecino k y k+1 suficiente para conservar el conjunto de vecinos bajo la perturbación. Para dos consultas con ese mismo conjunto:

\[
\|\Delta(x)-\Delta(x')\|_\infty
\le qR\bigl(|g(x)-g(x')|+\|\alpha(x)-\alpha(x')\|_1\bigr).
\]

Si g y alpha admiten constantes locales L_g y L_alpha y la base L_b, se obtiene la cota local L_b+qR(L_g+L_alpha). Derivar las constantes para las capas y normalizaciones realmente implementadas y verificar numéricamente las hipótesis. Top-k puede cambiar discontinuamente sin un margen; la incorporación de un episodio modifica la memoria y requiere un análisis separado. Esta propuesta de demostración no se considera todavía un resultado probado del proyecto. Las pruebas de causalidad y de identidad sin vecinos complementan, pero no sustituyen, la propiedad exigida.

## 5. Controles para identificar el mecanismo

Mantener iguales base, variables, normalización, capacidad, horizonte, presupuesto, semillas y política de acceso a observaciones. Comparar:

| Eje | Comparación controlada | Pregunta |
|---|---|---|
| Construcción de residuos | En muestra, OOF temporal y OOF temporal anidado | ¿La procedencia fuera de muestra aporta generalización? |
| Recuperación | Vecinos uniformes, ponderación por similitud y recuperación aprendida | ¿Qué aporta aprender la selección? |
| Memoria | Fija al cierre de entrenamiento y creciente observable | ¿La mejora procede de nuevas observaciones? |
| Combinación | Sin corrección, corrección sin compuerta y compuerta aprendida | ¿La compuerta aporta precisión o estabilidad? |
| Texto | N, O y OP; metadatos y desplazamientos temporales | ¿Existe información semántica incremental? |

Ejecutar al menos tres ablaciones aisladas del mecanismo central. El control en muestra es un diagnóstico de construcción del banco: nunca permite valores futuros al predecir validación o prueba. No confundir retirar OOF con retirar la máscara causal. La variante sin texto mantiene dimensión/capacidad comparable o informa el cambio de parámetros.

Incluir los diez modelos canónicos, los ingenuos y controles fuertes existentes —M1 y NNLS entre ellos—, más corrección por media móvil/EWMA de residuos, regresión residual, kNN y réplicas o adaptaciones documentadas de RATL y FreshCast. Resolver la pertinencia de RAFT y otros antecedentes en la revisión inicial y cerrar la lista completa antes de confirmar resultados. No retirar rivales por su rendimiento.

Todas las políticas que actualicen memoria reciben las mismas observaciones con el mismo retraso. La comparación con modelos de pesos y memoria fijos debe identificarse; incluir controles adaptativos permite separar actualización de información y arquitectura. Verificar que ninguna memoria utilice predicciones en muestra donde se exige OOF anidado.

## 6. Registro y datos antes de la evaluación adicional

Crear `experiments/estandar.md` y un registro específico con fecha real y etiqueta Git antes de la nueva ejecución. Incluir pregunta, contribución candidata, variante principal, conjuntos, hashes SHA-256, particiones, escala MASE, variables disponibles, baselines, presupuesto, semillas, regla de parada, familia de hipótesis, análisis de potencia y criterios de éxito. No reconstruir un preregistro para los resultados ya conocidos.

Conservar el margen **relativo del 3 % de reducción del MASE del rival**, alpha familiar 0,05 y escala calculada únicamente en entrenamiento. No alternar márgenes absolutos derivados del piloto y de prueba. Registrar pérdidas diarias de 24 horas, además de métricas por horizonte, semilla, año y segmento.

Para Colombia, mantener desarrollo 2018–2021 y prueba histórica 2022–2025 en el informe retrospectivo. Antes de abrir nuevas métricas, reservar un periodo realmente no inspeccionado o una evaluación prospectiva; documentar qué datos ya fueron consultados. No afirmar independencia de un periodo solo por cambiar su nombre. Clima de reanálisis, hidrología y precio entran únicamente con rezagos y retrasos de publicación verificables; nunca como clima observado del día futuro.

Seleccionar **al menos tres datasets adicionales**, independientes del colombiano y adecuados al mismo problema de pronóstico de demanda, antes de mirar su desempeño. Candidatos de adquisición: [ElectricityLoadDiagrams20112014 de UCI](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014), [GEFCom2014 Load](https://blog.drhongtao.com/2017/03/gefcom2014-load-forecasting-data.html) e [informes de demanda de ISO New England](https://www.iso-ne.com/isoexpress/web/reports/load-and-demand). La selección sigue pendiente de comprobar acceso, licencia, cobertura, independencia y diversidad de escala, calendario o frecuencia; no se han descargado ni evaluado en esta actualización. Tres zonas de una misma tabla no se contarán automáticamente como tres datasets. No sustituir GEFCom2014 Load por GEFCom2014-E y contarlo como independiente de ISO New England: el proveedor advierte que la variante E procede de ISO New England.

Fijar versiones, agregación de unidades, cambios de hora, horizonte, particiones temporales, MASE estacional y covariables realmente disponibles de cada dataset. En ausencia de texto, evaluar el núcleo numérico registrado; esos resultados no prueban generalización del componente semántico. Aplicar a cada dataset la familia completa de comparadores pertinentes fijada en el protocolo. No promediar MAE de unidades diferentes.

## 7. Semillas, potencia y presupuesto GPU

El E3 requiere al menos diez semillas independientes para los modelos y comparadores estocásticos, y más si la potencia lo exige: **objetivo mínimo de potencia 0,90 por contraste**, informando también potencia conjunta. Los controles deterministas no adquieren replicación independiente por repetir la misma salida.

Partir de 101–110 como lista candidata común, ampliable por dimensionamiento previo. Los resultados 101–105 se conservan como evidencia histórica; solo pueden reutilizarse en un análisis homogéneo si coinciden exactamente datos, fuentes, configuración, política temporal y hashes. Las semillas del Top 10 no se emparejan por posición con 101–105. Si faltan réplicas comparables, serán necesarios ajustes nuevos autorizados; no inventar semillas a partir de épocas, folds u horas.

Antes de esos ajustes, estimar potencia mediante simulación de pérdidas OOF de desarrollo con dependencia temporal y variabilidad entre semillas. Incluir el tamaño de efecto relevante, el número final de hipótesis y el ajuste Holm. Dimensionar para las pruebas efectivamente registradas y su criterio conjunto; guardar curvas de potencia y supuestos. No usar p-valores de prueba para calcular una supuesta potencia a priori ni asumir que diez semillas garantizan 0,90. Si el coste impide alcanzar el tamaño requerido, declarar el alcance sin certificar el requisito.

No ampliar ahora la rejilla de ocho configuraciones de M4. Los controles usan la configuración fijada o un presupuesto de selección comparable, decidido en desarrollo. Para datasets adicionales, registrar la misma herramienta y presupuesto de búsqueda para métodos comparados; no transferir arbitrariamente la configuración ganadora solo a uno. Guardar ajustes/épocas/tiempo GPU por procedimiento y separar selección, ablaciones y replicación final.

## 8. Contrastes y criterio de cierre

Para cada rival, definir d_t=L_rival,t−L_M4,t y la ganancia relativa 1−MASE_M4/MASE_rival. La unidad temporal es el origen diario, con pérdida media de sus 24 horas; las horas y las semillas no son miles de observaciones independientes. Conservar por separado incertidumbre entre inicializaciones y entre días.

La familia F2 del Cuadro 4 del E3 admite los siguientes tres procedimientos propuestos:

1. **Bootstrap estacionario pareado** de pérdidas diarias, conservando las mismas fechas para todos los métodos. Fijar longitud esperada de bloque solo con desarrollo; 14 días es la referencia histórica y 7/28 las sensibilidades, no tres pruebas distintas.
2. **Diebold–Mariano con corrección HLN** y varianza de largo plazo adecuada a la autocorrelación diaria. Para la pérdida agregada de trayectorias no solapadas de un día, justificar h efectivo de un origen; no introducir h=24 automáticamente porque hay 24 salidas. Registrar por separado los contrastes por horizonte si se incluyen.
3. **Giacomini–White**, con instrumentos y covarianza HAC prefijados, para evaluar capacidad predictiva condicional. Verificar sus supuestos y la compatibilidad de la regla de pronóstico/memoria con el procedimiento; no sustituirla sin justificación por Clark–West, cuya corrección habitual corresponde a otros supuestos y pérdida cuadrática.

Registrar alfa, direccionalidad y restricciones de cada prueba. Holm controlará la familia conjunta declarada de rivales, hipótesis, datasets y pruebas utilizada para las afirmaciones de superioridad. Distinguir diferencia estadística, dirección favorable y relevancia práctica >3 %. Para esta última, contrastar pérdidas ajustadas por el margen cuando sea válido y obtener intervalos relativos; no interpretar el rechazo de una hipótesis de igualdad como prueba de mejora >3 %. Un GW bilateral tampoco establece por sí solo la dirección del efecto.

El cierre de superioridad exige tres pruebas admisibles con resultados favorables por contraste conforme al E3, además del criterio de magnitud y la potencia planificada; los diagnósticos de residuos no cuentan como pruebas de superioridad. Si falla un supuesto, revisar el protocolo antes de ejecutar confirmación y documentar una alternativa admisible. La ampliación no sustituye ni mezcla las familias históricas de 28 y 220 hipótesis.

Complementar con Model Confidence Set y comparación entre datasets mediante Friedman–Iman–Davenport y post hoc Holm cuando las unidades y supuestos lo permitan. Reportar todos los rivales, incluidos resultados nulos o desfavorables; no deducir equivalencia de p>0,05.

## 9. Verificaciones y ejecución futura en GPU

1. Sincronizar el commit del plan, instalar Git LFS y comprobar archivos reales, pesos y hashes. Consultar los manifiestos publicados: no suponer que embeddings, cachés o estados reconstruibles están todos en GitHub. Reconstruir solo según su receta registrada. Usar rutas con `pathlib` y un entorno aislado en Windows o Linux.
2. Instalar PyTorch compatible con la GPU real siguiendo [la guía oficial](https://pytorch.org/get-started/locally/). Comprobar dispositivo, memoria y una operación con gradiente antes de una cola. Registrar SO, Python, PyTorch, controlador, CUDA/ROCm, GPU, precisión y determinismo. CUDA no es una opción válida para una GPU Intel; la máquina de esta actualización dispone de Torch CPU y no ejecutó redes.
3. Revisar y fortalecer las pruebas del código: máscara antes de top-k, ausencia de efecto de filas futuras, retraso de dos días, OOF por prefijo, identidad sin vecinos, restauración exacta, formas, entradas inválidas, determinismo, sobreajuste de lote mínimo y gradientes finitos/no nulos en **cada parámetro activo**. Distinguir parámetros congelados y canales inactivos; comprobar que alguno tenga gradiente no basta. Verificar la propiedad teórica y estabilidad frente a cambios del conjunto de vecinos.
4. Con autorización expresa, ejecutar primero el piloto/dimensionamiento pendiente con desarrollo y el protocolo registrado. Fijar tamaño final de réplicas sin consultar prueba. Respetar selección temporal y early stopping interno; nunca detener por error del conjunto de prueba.
5. Ejecutar los ajustes finales autorizados manteniendo pesos seleccionados y presupuestos. Guardar optimizador, escaladores, generadores aleatorios y criterio de parada; reanudar solo con igualdad de configuración. Las réplicas nuevas se guardan en una carpeta nueva, sin sobrescribir los resultados históricos.
6. Evaluar en orden cronológico con pesos congelados y memoria disponible. Verificar invariancia al tamaño de lote o usar lote uno cuando existan decisiones entre ejemplos. Exportar vecinos, fechas, residuos usados, atenciones y compuertas para auditar causalidad.
7. Generar tablas, figuras e interpretaciones desde predicciones canónicas: observado/predicho, curvas, error por horizonte/segmento, ganancias, IC, tres pruebas, correcciones, potencia, coste y robustez. Medir latencia mediana/p95 con calentamiento y sincronización GPU, tamaño de lote y hardware declarados. Las atenciones no se presentarán automáticamente como importancia causal.

No hay un lanzador nuevo validado para este cierre: adaptar y probar los scripts publicados es trabajo pendiente, sujeto a autorización. No usar comandos de ajuste sin verificar sus argumentos y el protocolo final.

## 10. Entregables y decisiones finales

| Fase | Artefactos requeridos | Estado al 23-11-2026 |
|---|---|---|
| Novedad | Matriz de 5–10 textos completos, búsquedas y diferencia formal | Matriz inicial; revisión pendiente |
| Teoría | Derivación, supuestos y verificación numérica | Propiedad candidata; pendiente |
| Diseño | `experiments/estandar.md`, registro y etiqueta con fecha real | Pendiente antes de nuevas ejecuciones |
| Potencia | Simulación de desarrollo, tamaño requerido y presupuesto | Pendiente |
| GPU | Pruebas, réplicas comparables y ≥3 datasets adicionales | Pendiente; requiere autorización |
| Evaluación | Predicciones, métricas, tres pruebas y Holm, ablaciones y robustez | Evidencia histórica disponible; ampliación pendiente |
| Informe | Conclusiones separadas para mecanismo, texto y superioridad | Pendiente de la evidencia ampliada |

Guardar el trabajo futuro, por ejemplo, en `experiments/m4_originalidad/` y `results/m4_cierre_gpu_<fecha>/`, con fuentes, registro, manifiestos y hashes. Los conteos y conclusiones se generarán desde resultados para evitar divergencias documentales. No cambiar los hashes ni el contenido de los artefactos históricos.

**Decisión actual:** conservar M4 y cerrar su búsqueda amplia; invertir primero en demostrar la diferencia metodológica y aislar el mecanismo. El texto sigue siendo una hipótesis sin mejora incremental demostrada. Si las comparaciones cercanas explican toda la contribución o la evaluación no supera a todos los rivales, informar ese resultado sin forzar una conclusión de originalidad o superioridad. El JBook y el dashboard se actualizarán o publicarán únicamente cuando el usuario lo solicite.

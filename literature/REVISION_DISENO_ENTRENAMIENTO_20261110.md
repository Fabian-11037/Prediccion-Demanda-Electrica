# Revisión del diseño de entrenamiento con texto

10 de noviembre de 2026. Documento revisado: «Diseño de entrenamiento pronóstico de demanda eléctrica con texto.md». La revisión inicial identificó los puntos siguientes. Las correcciones se aplicaron al documento de diseño y al contrato numérico común el 10 de noviembre de 2026, por solicitud del usuario. No se ejecutaron entrenamientos. Este archivo conserva la motivación de los cambios; la especificación vigente está en el diseño corregido y en `config/diseno_entrenamiento_texto.json`.

La propuesta es útil para organizar el Top 10: distingue familias numéricas, dos mecanismos de fusión, controles de contenido y una contribución propia con propiedades verificables. Sin embargo, requiere conciliación documental, correcciones matemáticas y un protocolo de datos común antes de implementarse. No debe trasladarse literalmente al preregistro.

## 1. Decisiones que deben conciliarse

- El diseño confirmado utiliza el escenario declarado, tres variantes comparables (numérica, numérica + oficial y numérica + oficial + prensa), canales oficial/prensa separados y relevancia aprendida. El filtrado por términos y la división principal energía/IDEAM del documento no sustituyen esas decisiones. La fuente puede utilizarse para controles o subgrupos internos.
- Conservar embeddings por documento o fragmento para aprender relevancia. Un promedio diario calculado antes del modelo elimina la posibilidad de ponderar posteriormente los documentos que lo componen. La agregación aprendida puede producir después los canales temporales que requiere TaTS.
- La ausencia de texto se representa con cero y un indicador propio de cada canal. La frase «sin información disponible» es una alternativa experimental que no está adoptada como sustitución de ese contrato.
- Fijar las covariables numéricas comunes a todas las variantes. Demanda y calendario son las entradas del benchmark ya ejecutado; clima pasado, hidrología y precio están disponibles y su contribución aún no se evaluó. No convertir la elección de covariables en una diferencia silenciosa entre arquitecturas.
- Las cantidades del corpus y sus fechas deben proceder del manifiesto real y de la elegibilidad por origen. El nombre `corpus_oficial_20261109` no debe presentarse como etiqueta Git verificada sin crear y registrar efectivamente esa versión. Las cifras de cobertura de 2008–2019 no describen la partición de referencia vigente 2000–2017.
- La escala MASE vigente es fija: q168 = 215617,90361589016 kWh. No cambia entre validación y prueba, pese a lo indicado en una tabla del documento. El criterio de selección es **menor** MASE, no mayor.
- El bloque 2022–2025 ya fue examinado. Ningún nuevo preregistro puede convertirlo en una prueba históricamente no consultada. Enero de 2026 tampoco es ahora una fecha prospectiva: los datos descargados el 10 nov 2026 son retrospectivos. La confirmación prospectiva comienza después de congelar protocolo, modelo e información de entrada.
- El entrenador activo antiguo no coincide con el que produjo los checkpoints consolidados. La implementación futura debe partir del contrato validado de escalado por pliegue y parada interna conservado en la instantánea v3/v4, sin modificar esa instantánea ni presentar los presupuestos históricos como iguales.

## 2. La cota de la fusión no garantiza «no empeorar»

La propiedad de la idea A es válida como cota de desviación si la base es la misma y está congelada. Pero de |ŷ−ŷ_num|≤ρ solo se obtiene |MAE−MAE_num|≤ρ en la escala normalizada. El nuevo error puede aumentar: si la base acierta exactamente y la corrección vale ρ, el nuevo error es ρ.

La propiedad demostrable es **deterioro acotado**, y equivalencia exacta con la base cuando la máscara es cero. No es dominancia del error con texto. En kWh, el radio del origen t es ρ·nivel_t; no debe confundirse ρ con un porcentaje del MAE ni con el margen relativo del 3 %.

Congelar gradientes no fija el modo de evaluación: `fusion.train()` activa recursivamente el modo entrenamiento de la base. Se debe mantener expresamente `base.eval()` durante el ajuste de la rama adicional y verificar que la salida sin texto coincida con la referencia para todo lote. `no_grad()` no desactiva Dropout.

## 3. La proposición de la pérdida con placebo requiere otra hipótesis

En la idea C, si la base congelada b(X) no es la media condicional y la pérdida es cuadrática, la función que ignora el texto minimiza

    E[(y−f(X))²|X] + λ(f(X)−b(X))²

en f*(X) = (E[y|X] + λ·b(X))/(1+λ). En general no es E[y|X]. La independencia del texto permite justificar que no añada información bajo las hipótesis descritas; no demuestra que ambos términos se minimicen simultáneamente en la media condicional salvo que la base ya sea óptima. La restricción de capacidad y el radio de corrección requieren además considerar la clase de funciones admisible.

Un placebo permutado dentro de una partición puede introducir documentos futuros respecto de un origen. Debe usar donantes anteriores al origen, o ruido sintético independiente, conservando volumen y máscara. «No mezclar particiones» no es suficiente para acreditar disponibilidad temporal.

## 4. Campo receptivo y restricciones de representación

Con convoluciones causales de paso uno, el campo receptivo es R=1+Σ_l(k_l−1)d_l. Para tres convoluciones con dilataciones 1,4,16 y k=5, R=85; con k=7, R=127. No alcanza 168. Dos convoluciones por bloque dan R=169 y R=253 respectivamente; esa estructura debe figurar expresamente y probarse para cada arquitectura, contando las convoluciones reales y cualquier agrupamiento.

La idea F conserva la equivarianza positiva indicada, pero también impone media(ŷ)=ℓ(x). Esa restricción puede impedir adaptar el nivel del día siguiente a un festivo o cambio operativo. Un factor positivo dependiente del perfil normalizado y del calendario podría permitir variar el nivel conservando la equivarianza. Su originalidad sigue requiriendo búsqueda específica.

## 5. Evaluación y presupuesto

Clark–West se formula para diferencias ajustadas de errores cuadrados en modelos anidados bajo determinadas hipótesis; no se traslada automáticamente al MAE/MASE ni a cualquier red que contenga una compuerta. El análisis primario debe conservar la pérdida absoluta y una inferencia pareada compatible con dependencia temporal. Cualquier contraste secundario con error cuadrático debe identificarse como tal y justificar sus supuestos.

No conviene escoger la prueba principal según si Shapiro–Wilk rechaza con cinco semillas. Se debe fijar el procedimiento principal y sus sensibilidades antes de comparar. La probabilidad conjunta 0,9^12 solo vale bajo independencia de los doce rechazos, poco plausible cuando comparten modelos y fechas. La fórmula de potencia debe expresar δ y la desviación de las diferencias en la misma escala.

El presupuesto de 20 configuraciones×3 semillas propuesto no es el presupuesto histórico ejecutado de todos los baselines. La búsqueda de 1000 arquitecturas, la selección del núcleo de texto y las dos etapas de la fusión deben contabilizarse por separado. Las comparaciones del aporte del texto necesitan sus controles numéricos con las mismas covariables y un presupuesto declarado; no basta con heredar un resultado de otro procedimiento.

## 6. Codificación y disponibilidad

La ficha oficial de [multilingual-E5-base](https://huggingface.co/intfloat/multilingual-e5-base) documenta 768 dimensiones, pooling enmascarado, normalización y prefijos incluso para texto no inglés. Se debe fijar el prefijo apropiado para la tarea, el commit del modelo y del tokenizer, la política de pooling y el máximo de tokens incluyendo prefijos y tokens especiales. El caché debe identificarlos junto al hash del texto y de la fragmentación.

Conservar solo los primeros ocho fragmentos limita memoria, pero no garantiza igual información ni igual peso efectivo entre documentos. Reportar la proporción truncada y mantener las representaciones necesarias para la selección aprendida. El enmascarado no corrige revisiones históricas; no debe borrar indiscriminadamente magnitudes útiles (MW, precipitaciones o porcentajes) ni horizontes anunciados.

La disponibilidad inclusiva al origen no es fuga por sí misma. Cualquier cambio a `<` o cualquier latencia debe declararse de forma uniforme, sin introducir accidentalmente un día extra cuando la fecha diaria ya se habilita a medianoche del día siguiente. Las fechas de publicación de catálogos, creación de repositorio y emisión de un pronóstico son conceptos distintos.

## 7. Valoración y siguiente decisión

Mantendría la diversidad del Top 10 y el contraste entre fusión temprana y atención cruzada. Usaría como punto de partida para el modelo propio una corrección residual acotada con relevancia aprendida y canales separados, formulando su garantía como estabilidad y ausencia exacta de corrección cuando falta texto. Antes de entrenar: fijar el conjunto numérico ampliado, conciliar el documento con el contexto, confirmar formulaciones y presupuesto, y registrar una evaluación temporal que reconozca la exposición retrospectiva.

Las arquitecturas publicadas constituyen referencias de mecanismo: [TaTS, versión 3](https://arxiv.org/html/2502.08942v3) y [TGForecaster, versión 2](https://arxiv.org/html/2405.13522v2). La fecha de la versión 3 de TaTS es 7 mar 2026; debe distinguirse de su prepublicación inicial de 2025. Las adaptaciones al español, al SIN y a información exclusivamente anterior al origen requieren identificarse como adaptaciones y no como reproducción exacta.

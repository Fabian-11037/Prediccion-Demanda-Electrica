(e2-tats)=
# 10.11. TaTS: texto como variables temporales auxiliares

La {ref}`notación y las operaciones compartidas <fundamentos-arquitecturas>` definen los tensores, el nivel de demanda, la atención, las activaciones y la optimización.

## Arquitectura y fundamentos

TaTS convierte el texto en canales adicionales de una serie. En lugar de modificar el pronóstico al final, permite que el núcleo temporal procese conjuntamente la información numérica y una representación textual diaria. Los embeddings de documentos son constantes; la relevancia que los pondera se aprende con la pérdida de pronóstico.

Para cada uno de los siete cierres diarios de la historia se reúnen documentos oficiales disponibles en las 72 horas anteriores, hasta 64. La consulta $q_t$ se obtiene de los promedios numéricos de las últimas 24 y 168 horas. Cada embedding E5 $e_j\in\mathbb R^{768}$ recibe un peso:

$$\alpha_{d,j}=\operatorname{softmax}_j\!\left(\frac{q_t^TK e_{d,j}}{\sqrt{64}}\right),\quad
z_d=\operatorname{MLP}_{64\rightarrow64\rightarrow32\rightarrow12}\!\left(\sum_j\alpha_{d,j}Ve_{d,j}\right).$$

La representación diaria y su indicador de presencia se repiten en las 24 horas del bloque y se concatenan a las entradas históricas. Sin documentos, ambos quedan en cero. Se conservan únicamente documentos habilitados al cierre correspondiente; la ventana textual no lee artículos futuros.

La variante oficial seleccionada utiliza PatchTST de dimensión 128, tres capas y 16 cabezas, con 21 parches por canal. Una cabeza lineal opera sobre los parches de demanda y otra MLP de contexto combina resúmenes de todos los canales y calendario. Esta segunda vía permite que el texto y las exógenas influyan en el objetivo aun con un codificador independiente por canal. El total es de 1.428.380 parámetros.

La configuración numérica seleccionada usa un núcleo más pequeño: dimensión 64, dos capas y cuatro cabezas. Por tanto, el contraste de resultados mide el procedimiento completo de selección con y sin texto, no un efecto puro de contenido con arquitectura idéntica. La agregación aprendida y la cabeza de contexto son adaptaciones al mecanismo de TaTS.

### Esquema global

[![Arquitectura global de tats](../_static/arquitecturas_presentacion/tats_global.png)](../_static/arquitecturas_presentacion/tats_global.png)

El texto oficial se resume mediante relevancia aprendida en siete cierres diarios. Doce atributos y un indicador de presencia por día se repiten sobre las horas correspondientes y se añaden a los 33 canales numéricos. PatchTST procesa los 46 canales y combina la lectura de demanda con una lectura de contexto.

### Detalle de las operaciones

[![Detalle de la arquitectura de tats](../_static/arquitecturas_presentacion/tats_detalle.png)](../_static/arquitecturas_presentacion/tats_detalle.png)

Los embeddings E5 permanecen congelados; las proyecciones de consulta, clave y valor aprenden a ponderar documentos disponibles. Las dos lecturas producen 24 correcciones: una utiliza el canal de demanda y otra reúne los demás canales con el calendario. Sin documentos, los canales textuales son cero.

El {ref}`código de exportación de los esquemas <exportacion-esquemas>` forma parte del preprocesamiento compartido.

### Código de la arquitectura

```{literalinclude} ../_static/codigo_e2/tats_red.py
:language: python
:linenos:
```

## Pipeline de entrenamiento y prueba

El pipeline fija el plan y las huellas de entrada, genera ventanas cronológicas, ajusta las transformaciones por pliegue y evalúa 20 candidatos en 2018. Confirma cinco finalistas en 2019–2021, selecciona por MASE medio y reajusta con cinco semillas finales usando la reserva interna anterior a 2022. La lectura de la prueba solo ocurre después de cerrar la selección. El libro muestra las implementaciones y no llama al entrenamiento.

### Datos, referencias y escaladores por pliegue

```{literalinclude} ../_static/codigo_e2/Modelo6_TaTS/src/modelo6/datos.py
:language: python
:linenos:
```

### Bucle de ajuste y predicción

```{literalinclude} ../_static/codigo_e2/Modelo6_TaTS/src/modelo6/motor.py
:language: python
:linenos:
```

### Búsqueda, confirmación y evaluación final

```{literalinclude} ../_static/codigo_e2/Modelo6_TaTS/src/modelo6/protocolo.py
:language: python
:linenos:
```

### Disponibilidad, embeddings y preparación del texto

```{literalinclude} ../_static/codigo_e2/Modelo6_TaTS/src/modelo6/texto.py
:language: python
:linenos:
```

### Entrada Python y registro de la ejecución

```{literalinclude} ../_static/codigo_e2/Modelo6_TaTS/ejecutar.py
:language: python
:linenos:
```

La configuración exacta del paquete se conserva a continuación. Identifica variantes, candidatos, semillas, parada y presupuesto; se mantiene separada de las métricas calculadas a partir de artefactos.

```{literalinclude} ../_static/codigo_e2/Modelo6_TaTS/config/modelo6.json
:language: json
```


## Evaluación

Los resultados de esta familia se integran en la {ref}`evaluación estadística conjunta <evaluacion-e2>`, con métricas por semilla, horizonte y origen. La comparación distingue las entradas demanda/calendario de las entradas numéricas ampliadas y el aporte de texto oficial.

## Fuentes

[Li et al., TaTS, versión 3](https://arxiv.org/html/2502.08942v3). La descripción corresponde a la implementación del experimento; las adaptaciones se identifican en el texto.

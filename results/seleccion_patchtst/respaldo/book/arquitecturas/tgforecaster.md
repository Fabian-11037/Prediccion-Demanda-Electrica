(e2-tgforecaster)=
# 10.12. TGForecaster: consultas del horizonte guiadas por texto

La {ref}`notación y las operaciones compartidas <fundamentos-arquitecturas>` definen los tensores, el nivel de demanda, la atención, las activaciones y la optimización.

## Arquitectura y fundamentos

TGForecaster utiliza descripciones textuales de los canales para orientar la relación entre información documental y señales numéricas. La atención cruzada distingue qué se pregunta —tokens del horizonte— de dónde se busca la respuesta —la memoria de la historia—. La variante N conserva las descripciones estáticas de canales y no incorpora documentos que cambien por origen; N+O añade el contexto documental oficial disponible.

Cada uno de los 33 canales históricos se divide en 21 parches de 16 horas con paso ocho. Sus embeddings se codifican y se suman a una proyección de la descripción E5 del canal; siete tokens adicionales representan el calendario diario pasado. La memoria reúne así $33\times21+7=700$ tokens. No se aplica RevIN dentro de este codificador; la normalización externa del contrato permanece.

El horizonte de 24 horas se divide en cinco parches de ocho con paso cuatro. La consulta de cada parche combina descripción de demanda, calendario futuro conocido, posición y resumen de documentos oficiales disponibles en el origen:

$$q_k=W_DE_{dem}+W_FF_k+p_k+m_tW_Tz_t,\quad
H=\operatorname{softmax}\!\left(\frac{QK_M^\top}{\sqrt{d_k}}\right)V_M.$$

El texto participa como parte de la consulta; las claves y valores proceden de la memoria numérica. Cada token decodifica ocho correcciones. Para las horas cubiertas por varios parches se aprende una combinación convexa: $\omega_{k,h}=\operatorname{softmax}_{k:h\in k}(a_{k,h})$, de modo que $\sum_k\omega_{k,h}=1$. La salida es $\ell_t[1+\sum_k\omega_{k,h}o_{k,h}]$.

El paquete adapta el artículo a una sola serie objetivo y a documentos disponibles al emitir el pronóstico; no incorpora texto realizado durante el horizonte. Las descripciones y los embeddings son extractores congelados; la selección documental y la fusión son entrenables. La variante N selecciona dimensión256, tres capas temporales, dos de mezcla y ocho cabezas; N+O selecciona dimensión64, dos capas temporales, una de mezcla y cuatro cabezas. Sus configuraciones distintas hacen que el contraste estime el procedimiento completo de selección, además de la incorporación de documentos.

### Esquema global

[![Arquitectura global de tgforecaster](../_static/arquitecturas_presentacion/tgforecaster_global.png)](../_static/arquitecturas_presentacion/tgforecaster_global.png)

La historia se divide en parches y se combina con descripciones estáticas de las 33 variables. El calendario pasado completa una memoria de 700 tokens. Cinco consultas incorporan el calendario del horizonte, la descripción de demanda y el resumen documental; la atención cruzada genera parches de pronóstico superpuestos.

### Detalle de las operaciones

[![Detalle de la arquitectura de tgforecaster](../_static/arquitecturas_presentacion/tgforecaster_detalle.png)](../_static/arquitecturas_presentacion/tgforecaster_detalle.png)

La identidad de cada variable se proyecta sobre sus parches históricos. Las consultas del horizonte atienden esa memoria y producen cinco parches de ocho horas, desplazados cuatro horas. Pesos aprendidos, normalizados entre los parches que cubren cada hora, reconstruyen las 24 salidas. La variante numérica conserva las descripciones estáticas y omite los documentos.

El {ref}`código de exportación de los esquemas <exportacion-esquemas>` forma parte del preprocesamiento compartido.

### Código de la arquitectura

```{literalinclude} ../_static/codigo_e2/tgforecaster_red.py
:language: python
:linenos:
```

## Pipeline de entrenamiento y prueba

El pipeline fija el plan y las huellas de entrada, genera ventanas cronológicas, ajusta las transformaciones por pliegue y evalúa 20 candidatos en 2018. Confirma cinco finalistas en 2019–2021, selecciona por MASE medio y reajusta con cinco semillas finales usando la reserva interna anterior a 2022. La lectura de la prueba solo ocurre después de cerrar la selección. El libro muestra las implementaciones y no llama al entrenamiento.

### Datos, referencias y escaladores por pliegue

```{literalinclude} ../_static/codigo_e2/Modelo7_TGForecaster/src/entrenamiento/datos.py
:language: python
:linenos:
```

### Bucle de ajuste y predicción

```{literalinclude} ../_static/codigo_e2/Modelo7_TGForecaster/src/entrenamiento/motor.py
:language: python
:linenos:
```

### Búsqueda, confirmación y evaluación final

```{literalinclude} ../_static/codigo_e2/Modelo7_TGForecaster/src/entrenamiento/protocolo.py
:language: python
:linenos:
```

### Disponibilidad, embeddings y preparación del texto

```{literalinclude} ../_static/codigo_e2/Modelo7_TGForecaster/src/entrenamiento/texto.py
:language: python
:linenos:
```

### Entrada Python y registro de la ejecución

```{literalinclude} ../_static/codigo_e2/Modelo7_TGForecaster/ejecutar.py
:language: python
:linenos:
```

La configuración exacta del paquete se conserva a continuación. Identifica variantes, candidatos, semillas, parada y presupuesto; se mantiene separada de las métricas calculadas a partir de artefactos.

```{literalinclude} ../_static/codigo_e2/Modelo7_TGForecaster/config/modelo.json
:language: json
```


## Evaluación

Las variantes N y N+O conservan cinco semillas finales cada una, con métricas y predicciones para 2022–2025. Su contraste documental y la inferencia conjunta se presentan en la {ref}`evaluación integrada <evaluacion-e2>`. Las descripciones estáticas están presentes en ambas variantes; la comparación aísla el procedimiento con y sin documentos oficiales variables en el tiempo, con hiperparámetros seleccionados por separado.

## Fuentes

[Xu et al., TGForecaster, versión 2](https://arxiv.org/html/2405.13522v2). La descripción corresponde a la implementación del experimento; las adaptaciones se identifican en el texto.

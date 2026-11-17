(e2-tft)=
# 10.8. TFT: selección de variables y atención interpretable

La {ref}`notación y las operaciones compartidas <fundamentos-arquitecturas>` definen los tensores, el nivel de demanda, la atención, las activaciones y la optimización.

## Arquitectura y fundamentos

TFT combina selección de variables, memoria recurrente y atención. Primero decide qué variables ponderar en cada hora; después resume la secuencia y permite que cada hora del horizonte consulte los pasos relevantes de la historia. Las compuertas residuales regulan cuánto de cada transformación se incorpora.

La configuración seleccionada tiene ancho 160 y cuatro cabezas. Recibe 56 variables pasadas: 30 históricas, tres indicadores de ausencia por bloque y 23 de calendario. El futuro contiene únicamente 23 variables de calendario. Cada variable continua tiene un embedding lineal y una GRN propia. Una segunda GRN produce pesos softmax $v_{s,j}$; su suma es uno y la representación seleccionada es $z_s=\sum_jv_{s,j}\operatorname{GRN}_j(e_{s,j})$.

Una GRN utiliza una transformación no lineal, una compuerta y un atajo:

$$\operatorname{GRN}(a)=\operatorname{LayerNorm}\!\left(Sa+\operatorname{GLU}(W_1\operatorname{ELU}(W_2a+b_2)+b_1)\right),\quad
\operatorname{GLU}(u)=\sigma(W_gu+b_g)\odot(W_vu+b_v).$$

El codificador LSTM procesa las 168 horas y su estado inicializa al decodificador que recibe el calendario futuro. El enriquecimiento GRN precede a la atención interpretable: las cabezas comparten la proyección de valores y se promedian. La máscara impide que la consulta de la hora $h$ utilice representaciones del decodificador posteriores a ella. Las GRN por posición y las compuertas finales producen una salida escalar por hora. El total es de 11.715.596 parámetros.

Se omiten codificadores estáticos porque se estudia una sola serie. Se entrena un pronóstico puntual con MAE normalizado, no todos los cuantiles del artículo. Los pesos de selección describen el mecanismo del modelo; no son efectos causales de las variables. Esta documentación y los resultados usan la configuración TFT seleccionada del paquete, identificada por sus fuentes y checkpoints.

### Esquema global

[![Arquitectura global de tft](../_static/arquitecturas_presentacion/tft_global.png)](../_static/arquitecturas_presentacion/tft_global.png)

La selección de variables transforma las 56 entradas históricas y las 23 de calendario futuro a representaciones de dimensión 160. La recurrencia, las compuertas y la atención de cuatro cabezas producen una salida por hora.

### Detalle de las operaciones

[![Detalle de la arquitectura de tft](../_static/arquitecturas_presentacion/tft_detalle.png)](../_static/arquitecturas_presentacion/tft_detalle.png)

Las GRN combinan transformaciones no lineales con conexiones residuales y compuertas. La selección softmax pondera las representaciones de las variables en cada instante; esos pesos describen el mecanismo interno y no efectos causales.

El {ref}`código de exportación de los esquemas <exportacion-esquemas>` forma parte del preprocesamiento compartido.

### Código de la arquitectura

```{literalinclude} ../_static/codigo_e2/tft_red.py
:language: python
:linenos:
```

## Pipeline de entrenamiento y prueba

El pipeline fija el plan y las huellas de entrada, genera ventanas cronológicas, ajusta las transformaciones por pliegue y evalúa 20 candidatos en 2018. Confirma cinco finalistas en 2019–2021, selecciona por MASE medio y reajusta con cinco semillas finales usando la reserva interna anterior a 2022. La lectura de la prueba solo ocurre después de cerrar la selección. El libro muestra las implementaciones y no llama al entrenamiento.

### Datos, referencias y escaladores por pliegue

```{literalinclude} ../_static/codigo_e2/Modelo3_TFT/src/entrenamiento/datos.py
:language: python
:linenos:
```

### Bucle de ajuste y predicción

```{literalinclude} ../_static/codigo_e2/Modelo3_TFT/src/entrenamiento/motor.py
:language: python
:linenos:
```

### Búsqueda, confirmación y evaluación final

```{literalinclude} ../_static/codigo_e2/Modelo3_TFT/src/entrenamiento/protocolo.py
:language: python
:linenos:
```

### Entrada Python y registro de la ejecución

```{literalinclude} ../_static/codigo_e2/Modelo3_TFT/ejecutar.py
:language: python
:linenos:
```

La configuración exacta del paquete se conserva a continuación. Identifica variantes, candidatos, semillas, parada y presupuesto; se mantiene separada de las métricas calculadas a partir de artefactos.

```{literalinclude} ../_static/codigo_e2/Modelo3_TFT/config/modelo.json
:language: json
```


## Evaluación

Los resultados de esta familia se integran en la {ref}`evaluación estadística conjunta <evaluacion-e2>`, con métricas por semilla, horizonte y origen. La comparación distingue las entradas demanda/calendario de las entradas numéricas ampliadas y el aporte de texto oficial.

## Fuentes

[Lim et al., TFT](https://arxiv.org/abs/1912.09363). La descripción corresponde a la implementación del experimento; las adaptaciones se identifican en el texto.

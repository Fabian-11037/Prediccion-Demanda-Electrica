(e2-tcn)=
# 10.9. TCN: convoluciones causales dilatadas

La {ref}`notación y las operaciones compartidas <fundamentos-arquitecturas>` definen los tensores, el nivel de demanda, la atención, las activaciones y la optimización.

## Arquitectura y fundamentos

Una TCN aumenta su alcance temporal separando las posiciones que consulta cada filtro. Con dilatación cuatro, por ejemplo, un núcleo de cinco pesos lee posiciones separadas cuatro horas. El apilamiento permite integrar semanas de historia sin usar recurrencia. La causalidad se impone recortando el relleno derecho de cada convolución.

La red seleccionada usa tres bloques de 64 filtros, dilataciones 1, 4 y 16 y núcleo cinco. Cada bloque contiene dos convoluciones, normalización de pesos, ReLU y una conexión residual; una proyección $1\times1$ adapta el atajo si cambia el número de canales. El dropout de 0,2 está activo al final del bloque central, de acuerdo con el patrón (0,1,0).

$$\operatorname{Conv}_{d}(x)_s=\sum_{i=0}^{k-1}W_i x_{s-di}+b,\qquad
h_{b+1}=\operatorname{ReLU}(S_bh_b+F_b(h_b)).$$

Con dos convoluciones por bloque, el campo receptivo es $R=1+2(k-1)\sum_bd_b=169$ horas, suficiente para las 168 de entrada. El resultado depende de contar ambas convoluciones: una sola por bloque no cubriría la ventana. La representación de la última hora se concatena con una proyección del calendario futuro y una cabeza densa produce las 24 correcciones normalizadas. Hay 169.944 parámetros.

La implementación utiliza únicamente datos anteriores al origen en las convoluciones. El calendario futuro entra por una rama separada porque sí se conoce de antemano. La selección evaluó 20 candidatos; su alcance depende del espacio y del presupuesto de búsqueda descritos en el diseño experimental.

### Esquema global

[![Arquitectura global de tcn](../_static/arquitecturas_presentacion/tcn_global.png)](../_static/arquitecturas_presentacion/tcn_global.png)

Los 56 canales históricos atraviesan tres bloques causales con dilataciones 1, 4 y 16. Su campo receptivo de 169 horas cubre la ventana de 168 horas; el último estado se combina con el calendario futuro para producir las 24 salidas.

### Detalle de las operaciones

[![Detalle de la arquitectura de tcn](../_static/arquitecturas_presentacion/tcn_detalle.png)](../_static/arquitecturas_presentacion/tcn_detalle.png)

Cada bloque contiene dos convoluciones de cinco posiciones, recorte del relleno derecho y una conexión residual. El bloque central aplica dropout de 0,2; el atajo proyecta los canales cuando cambia su dimensión. La lectura combina 64 atributos temporales y 64 de calendario.

El {ref}`código de exportación de los esquemas <exportacion-esquemas>` forma parte del preprocesamiento compartido.

### Código de la arquitectura

```{literalinclude} ../_static/codigo_e2/tcn_red.py
:language: python
:linenos:
```

## Pipeline de entrenamiento y prueba

El pipeline fija el plan y las huellas de entrada, genera ventanas cronológicas, ajusta las transformaciones por pliegue y evalúa 20 candidatos en 2018. Confirma cinco finalistas en 2019–2021, selecciona por MASE medio y reajusta con cinco semillas finales usando la reserva interna anterior a 2022. La lectura de la prueba solo ocurre después de cerrar la selección. El libro muestra las implementaciones y no llama al entrenamiento.

### Datos, referencias y escaladores por pliegue

```{literalinclude} ../_static/codigo_e2/Modelo4_TCN/src/entrenamiento/datos.py
:language: python
:linenos:
```

### Bucle de ajuste y predicción

```{literalinclude} ../_static/codigo_e2/Modelo4_TCN/src/entrenamiento/motor.py
:language: python
:linenos:
```

### Búsqueda, confirmación y evaluación final

```{literalinclude} ../_static/codigo_e2/Modelo4_TCN/src/entrenamiento/protocolo.py
:language: python
:linenos:
```

### Entrada Python y registro de la ejecución

```{literalinclude} ../_static/codigo_e2/Modelo4_TCN/ejecutar.py
:language: python
:linenos:
```

La configuración exacta del paquete se conserva a continuación. Identifica variantes, candidatos, semillas, parada y presupuesto; se mantiene separada de las métricas calculadas a partir de artefactos.

```{literalinclude} ../_static/codigo_e2/Modelo4_TCN/config/modelo.json
:language: json
```


## Evaluación

Los resultados de esta familia se integran en la {ref}`evaluación estadística conjunta <evaluacion-e2>`, con métricas por semilla, horizonte y origen. La comparación distingue las entradas demanda/calendario de las entradas numéricas ampliadas y el aporte de texto oficial.

## Fuentes

[Bai et al., convoluciones causales dilatadas](https://arxiv.org/abs/1803.01271). La descripción corresponde a la implementación del experimento; las adaptaciones se identifican en el texto.

(e2-itransformer)=
# 10.10. iTransformer: atención entre variables

La {ref}`notación y las operaciones compartidas <fundamentos-arquitecturas>` definen los tensores, el nivel de demanda, la atención, las activaciones y la optimización.

## Arquitectura y fundamentos

iTransformer invierte el significado habitual de los tokens: cada ficha representa una variable completa, no una hora. Así, la atención relaciona la historia de demanda con la del precio, el clima y la hidrología. La estructura temporal de cada variable se comprime mediante una proyección lineal antes de la atención.

Las 30 variables históricas y tres indicadores producen 33 tokens de longitud 168. Cada variable de calendario usa historia y futuro conocidos, 192 valores, y genera uno de los 23 tokens adicionales. Las proyecciones llevan ambos tipos a dimensión 128 y se suman embeddings de identidad de variable. Las mismas matrices se comparten dentro de cada tipo; $p_j$ distingue los canales.

$$e_j=W_Hx_{1:168,j}+p_j,\quad
e_c=W_C[F_{pas,c};F_{fut,c}]+p_c,\quad
Z=\operatorname{TransformerEncoder}([e_1,\ldots,e_{56}]).$$

El codificador tiene dos capas, ocho cabezas, red interna de ancho 256, GELU y normalización posterior con LayerNorm final. La atención compara consultas y claves de variables: $A=\operatorname{softmax}(QK^\top/\sqrt{d_k})$, seguido de $AV$. Solo el token de demanda alimenta la cabeza de 24 salidas: $\widehat{\mathbf y}=\ell_t[\mathbf1+W_OZ_0+b_O]$. La configuración 1 contiene 321.816 parámetros.

El calendario futuro puede relacionarse con las variables históricas dentro de la atención, pero no se crean tokens de clima o precio realizados durante el horizonte. La identidad por variable y la integración del calendario constituyen adaptaciones explícitas del experimento.

### Esquema global

[![Arquitectura global de itransformer](../_static/arquitecturas_presentacion/itransformer_global.png)](../_static/arquitecturas_presentacion/itransformer_global.png)

Cada variable histórica se representa mediante un token de dimensión 128. El calendario aporta otros 23 tokens construidos con sus 168 horas pasadas y 24 futuras. Dos capas de atención relacionan los 56 tokens; la lectura del token de demanda genera el horizonte.

### Detalle de las operaciones

[![Detalle de la arquitectura de itransformer](../_static/arquitecturas_presentacion/itransformer_detalle.png)](../_static/arquitecturas_presentacion/itransformer_detalle.png)

La proyección se comparte entre variables del mismo tipo, mientras una identidad aprendida distingue cada canal. Cada capa combina atención de ocho cabezas, conexiones residuales, normalización y una red interna de dimensión 256. La atención opera entre variables.

El {ref}`código de exportación de los esquemas <exportacion-esquemas>` forma parte del preprocesamiento compartido.

### Código de la arquitectura

```{literalinclude} ../_static/codigo_e2/itransformer_red.py
:language: python
:linenos:
```

## Pipeline de entrenamiento y prueba

<!-- entrenamiento-portable:inicio -->
### Comandos de entrenamiento y evaluación

Validación temporal de la configuración seleccionada:

```bash
python scripts/entrenar_arquitectura.py --modelo itransformer --anio 2021 --semilla 1000 --dispositivo auto --salida salidas/itransformer_validacion_1000
```

Comprobación funcional breve:

```bash
python scripts/entrenar_arquitectura.py --modelo itransformer --smoke --dispositivo cpu --salida salidas/itransformer_funcional
```

Evaluación final, con ajuste y parada anteriores a 2022:

```bash
python scripts/entrenar_arquitectura.py --modelo itransformer --fase prueba --allow-test --semilla 1000 --dispositivo auto --salida salidas/itransformer_prueba_1000
```

El código completo de preparación, optimización, guardado y restauración está en el {ref}`pipeline ejecutable compartido <entrenamiento-portable>`. Cada salida debe ser una carpeta nueva.
<!-- entrenamiento-portable:fin -->

El pipeline fija el plan y las huellas de entrada, genera ventanas cronológicas, ajusta las transformaciones por pliegue y evalúa 20 candidatos en 2018. Confirma cinco finalistas en 2019–2021, selecciona por MASE medio y reajusta con cinco semillas finales usando la reserva interna anterior a 2022. La lectura de la prueba solo ocurre después de cerrar la selección. El libro muestra las implementaciones y no llama al entrenamiento.

### Datos, referencias y escaladores por pliegue

```{literalinclude} ../_static/codigo_e2/Modelo5_iTransformer/src/entrenamiento/datos.py
:language: python
:linenos:
```

### Bucle de ajuste y predicción

```{literalinclude} ../_static/codigo_e2/Modelo5_iTransformer/src/entrenamiento/motor.py
:language: python
:linenos:
```

### Búsqueda, confirmación y evaluación final

```{literalinclude} ../_static/codigo_e2/Modelo5_iTransformer/src/entrenamiento/protocolo.py
:language: python
:linenos:
```

### Ejecución

El {ref}`pipeline común <entrenamiento-portable>` utiliza las fuentes de esta ficha y la configuración seleccionada. Los siguientes comandos ejecutan el ajuste y la evaluación.

La configuración exacta del paquete se conserva a continuación. Identifica variantes, candidatos, semillas, parada y presupuesto; se mantiene separada de las métricas calculadas a partir de artefactos.

```{literalinclude} ../_static/codigo_e2/Modelo5_iTransformer/config/modelo.json
:language: json
```


## Evaluación

Los resultados de esta familia se integran en la {ref}`evaluación estadística conjunta <evaluacion-e2>`, con métricas por semilla, horizonte y origen. La comparación distingue las entradas demanda/calendario de las entradas numéricas ampliadas y el aporte de texto oficial.

## Fuentes

[Liu et al., iTransformer](https://arxiv.org/abs/2310.06625). La descripción corresponde a la implementación del experimento; las adaptaciones se identifican en el texto.

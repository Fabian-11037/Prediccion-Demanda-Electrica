(e2-patchtst)=
# 10.6. PatchTST numérico

La {ref}`notación y las operaciones compartidas <fundamentos-arquitecturas>` definen los tensores, el nivel de demanda, la atención, las activaciones y la optimización.

## Arquitectura y fundamentos

PatchTST divide cada serie en pequeños segmentos superpuestos y trata cada segmento como un token. La atención conecta segmentos distantes sin recorrerlos secuencialmente. El codificador comparte parámetros entre canales, de modo que la demanda, el precio y cada covariable se procesan con la misma transformación temporal.

La configuración numérica usa 30 variables históricas, 23 de calendario pasado y 30 máscaras de ausencia: 83 canales. Cada historia de 168 horas se normaliza reversiblemente con RevIN, se rellena con ocho copias del último valor y se divide en parches de 16 con paso 8: $\lfloor(168+8-16)/8\rfloor+1=21$ tokens. Se proyectan a dimensión 16 y se añaden posiciones aprendidas. Tres bloques de atención, cuatro cabezas y una red interna de ancho 128 producen la representación.

$$P_c=\operatorname{patch}_{16,8}(\tilde x_c),\quad E_c=P_cW_P+E_{pos},\qquad
\operatorname{Att}(Q,K,V)=\operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V.$$

RevIN resta media y divide por dispersión de cada ventana y canal; su inversión devuelve la salida a la escala del canal. Estas estadísticas de entrada no incluyen etiquetas futuras. La cabeza aplanada genera 24 valores por canal. Una lectura afín explícita reúne esos valores y el calendario futuro para pronosticar únicamente demanda; así las covariables sí tienen un camino hacia el objetivo. Sin esa lectura, el procesamiento independiente no transferiría sus salidas a demanda. Hay 24.982 parámetros en la configuración seleccionada P20.

Esta es una adaptación del código oficial, no una identidad con el experimento del artículo. La normalización externa por nivel y las máscaras son parte del contrato numérico; RevIN opera además dentro del núcleo. El motor utiliza MAE normalizado, AdamW, hasta 200 épocas y paciencia efectiva de diez desde un mínimo de diez épocas.

### Esquema global

[![Arquitectura global de patchtst](../_static/arquitecturas_presentacion/patchtst_global.png)](../_static/arquitecturas_presentacion/patchtst_global.png)

Cada canal se divide en parches y pasa por el mismo codificador de tres capas. La cabeza genera 24 valores por canal; una lectura afín reúne esos valores con el calendario futuro para pronosticar demanda.

### Detalle de las operaciones

[![Detalle de la arquitectura de patchtst](../_static/arquitecturas_presentacion/patchtst_detalle.png)](../_static/arquitecturas_presentacion/patchtst_detalle.png)

La atención y la red interna se combinan con conexiones residuales y BatchNorm. La lectura de cada hora utiliza 83 pronósticos de canal y 23 atributos de calendario, con un total de 106 entradas.

El {ref}`código de exportación de los esquemas <exportacion-esquemas>` forma parte del preprocesamiento compartido.

### Código de la arquitectura

```{literalinclude} ../_static/codigo_e2/patchtst_red.py
:language: python
:linenos:
```

## Pipeline de entrenamiento y prueba

La preparación conserva 30 variables históricas, 23 de calendario y 30 máscaras. Se usan las mismas ventanas para ajuste, parada y prueba, con imputaciones y escalas estimadas solo en ajuste. El motor recibe el dispositivo explícitamente, guarda el mínimo de parada interna y convierte la salida normalizada a kWh. El código de la arquitectura incluye el motor para PatchTST; N-HiTS comparte ese motor sin compartir su red.

### Preparación, ajuste y predicción numérica

```{literalinclude} ../_static/codigo_e2/numerico_repositorio/src/demanda/patchtst_numerico.py
:language: python
:linenos:
```

### Núcleo PatchTST utilizado

```{literalinclude} ../_static/codigo_e2/numerico_repositorio/src/demanda/vendor/patchtst/PatchTST_backbone.py
:language: python
:linenos:
```

### Selección, piloto, ajuste final y prueba

```{literalinclude} ../_static/codigo_e2/numerico_repositorio/scripts/entrenar_patchtst_numerico.py
:language: python
:linenos:
```

El orquestador preservado utiliza bloqueo `fcntl` y exige GPU en su entrada principal. El siguiente adaptador conserva red, configuración, preparación, parada y predicción, y permite escoger CPU o GPU sin ese controlador de Linux. Su inclusión documental no ejecuta un nuevo ajuste.

### Pipeline explícito compatible con Windows y Linux

```{literalinclude} ../_static/codigo_e2/pipeline_ampliado_portable.py
:language: python
:linenos:
```


## Evaluación

Los resultados de esta familia se integran en la {ref}`evaluación estadística conjunta <evaluacion-e2>`, con métricas por semilla, horizonte y origen. La comparación distingue las entradas demanda/calendario de las entradas numéricas ampliadas y el aporte de texto oficial.

## Fuentes

[Nie et al., PatchTST](https://arxiv.org/abs/2211.14730). La descripción corresponde a la implementación del experimento; las adaptaciones se identifican en el texto.

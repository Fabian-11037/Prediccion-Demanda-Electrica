(e2-fusion)=
# 10.13. Fusión tardía de pronósticos numéricos y textuales

La {ref}`notación y las operaciones compartidas <fundamentos-arquitecturas>` definen los tensores, el nivel de demanda, la atención, las activaciones y la optimización.

## Arquitectura y fundamentos

La fusión tardía mantiene dos caminos hasta la salida: uno produce un pronóstico con la serie y el calendario; el otro propone un aporte basado en documentos. La combinación se realiza después de ambos cálculos. Esto contrasta con TaTS, que introduce el texto dentro de la secuencia que procesa el núcleo.

El núcleo seleccionado es PatchTST de dimensión 64, dos capas y cuatro cabezas, con cabeza lineal de demanda y MLP de contexto. Los documentos oficiales disponibles en las 72 horas anteriores al origen se ponderan por relevancia aprendida. El resumen de dimensión 64 pasa por una capa de 128, GELU, dropout y una salida de 24. Un escalar $\alpha$ aprendido combina las ramas:

$$\widehat{\mathbf r}=b(X,F)+\alpha\,m_t\,g(z_t),\qquad
\widehat{\mathbf y}=\ell_t\widehat{\mathbf r}.$$

La última capa textual se inicializa en cero y $\alpha$ en 0,5; el modelo empieza con el pronóstico numérico. Sin documentos, $m_t=0$ y la contribución textual es exactamente cero para los mismos pesos del núcleo. Ambas ramas se entrenan conjuntamente: no se congela la base. Por ello, esa identidad no garantiza igualdad con un checkpoint numérico entrenado por separado.

La suma no es una mezcla convexa: $\alpha$ no está restringido a [0,1] y la rama textual puede tener cualquier signo. Tampoco es la corrección residual acotada del diseño de investigación: aquí no hay radio $\rho$ ni garantía de deterioro acotado. La configuración contiene 584.649 parámetros. Su control sin texto coincide en las cinco predicciones con el control numérico de TaTS y se identifica como una única referencia en el benchmark, conservando la trazabilidad de ambos paquetes.

### Esquema global

[![Arquitectura global de fusion](../_static/arquitecturas_presentacion/fusion_global.png)](../_static/arquitecturas_presentacion/fusion_global.png)

Una rama PatchTST procesa la historia numérica y el calendario, mientras otra transforma un resumen de documentos oficiales en 24 correcciones. Ambas ramas se ajustan conjuntamente; un escalar aprendido pondera la contribución textual antes de sumar las salidas.

### Detalle de las operaciones

[![Detalle de la arquitectura de fusion](../_static/arquitecturas_presentacion/fusion_detalle.png)](../_static/arquitecturas_presentacion/fusion_detalle.png)

La rama documental utiliza una consulta numérica para ponderar los embeddings disponibles y un MLP de dimensiones 64, 128 y 24. Su última capa inicia en cero; el escalar comienza en 0,5 y se aprende sin acotación. El indicador de presencia anula exactamente la contribución documental cuando falta texto.

El {ref}`código de exportación de los esquemas <exportacion-esquemas>` forma parte del preprocesamiento compartido.

### Código de la arquitectura

```{literalinclude} ../_static/codigo_e2/fusion_red.py
:language: python
:linenos:
```

## Pipeline de entrenamiento y prueba

<!-- entrenamiento-portable:inicio -->
### Comandos de entrenamiento y evaluación

Validación temporal de la configuración seleccionada:

```bash
python scripts/entrenar_arquitectura.py --modelo fusion_tardia --anio 2021 --semilla 1000 --dispositivo auto --salida salidas/fusion_tardia_validacion_1000
```

Comprobación funcional breve:

```bash
python scripts/entrenar_arquitectura.py --modelo fusion_tardia --smoke --dispositivo cpu --salida salidas/fusion_tardia_funcional
```

Evaluación final autorizada, con ajuste y parada anteriores a 2022:

```bash
python scripts/entrenar_arquitectura.py --modelo fusion_tardia --fase prueba --allow-test --semilla 1000 --dispositivo auto --salida salidas/fusion_tardia_prueba_1000
```

El código completo de preparación, optimización, guardado y restauración está en el {ref}`pipeline ejecutable compartido <entrenamiento-portable>`. Cada salida debe ser una carpeta nueva.
<!-- entrenamiento-portable:fin -->

El pipeline fija el plan y las huellas de entrada, genera ventanas cronológicas, ajusta las transformaciones por pliegue y evalúa 20 candidatos en 2018. Confirma cinco finalistas en 2019–2021, selecciona por MASE medio y reajusta con cinco semillas finales usando la reserva interna anterior a 2022. La lectura de la prueba solo ocurre después de cerrar la selección. El libro muestra las implementaciones y no llama al entrenamiento.

### Datos, referencias y escaladores por pliegue

```{literalinclude} ../_static/codigo_e2/FusionTardia/src/entrenamiento/datos.py
:language: python
:linenos:
```

### Bucle de ajuste y predicción

```{literalinclude} ../_static/codigo_e2/FusionTardia/src/entrenamiento/motor.py
:language: python
:linenos:
```

### Búsqueda, confirmación y evaluación final

```{literalinclude} ../_static/codigo_e2/FusionTardia/src/entrenamiento/protocolo.py
:language: python
:linenos:
```

### Disponibilidad, embeddings y preparación del texto

```{literalinclude} ../_static/codigo_e2/FusionTardia/src/entrenamiento/texto.py
:language: python
:linenos:
```

### Ejecución

El {ref}`pipeline común <entrenamiento-portable>` utiliza las fuentes de esta ficha y la configuración seleccionada. Los siguientes comandos ejecutan el ajuste y la evaluación.

La configuración exacta del paquete se conserva a continuación. Identifica variantes, candidatos, semillas, parada y presupuesto; se mantiene separada de las métricas calculadas a partir de artefactos.

```{literalinclude} ../_static/codigo_e2/FusionTardia/config/modelo.json
:language: json
```


## Evaluación

Los resultados de esta familia se integran en la {ref}`evaluación estadística conjunta <evaluacion-e2>`, con métricas por semilla, horizonte y origen. La comparación distingue las entradas demanda/calendario de las entradas numéricas ampliadas y el aporte de texto oficial.

## Fuentes

La formulación y las dimensiones se verifican en las clases del código mostrado.. La descripción corresponde a la implementación del experimento; las adaptaciones se identifican en el texto.

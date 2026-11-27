(e2-nbeats)=
# 10.14. N-BEATS: reconstrucción y pronóstico residual

## Arquitectura y fundamentos

N-BEATS divide el pronóstico en contribuciones sucesivas. Cada bloque recibe el residuo de la demanda, reconstruye una parte de la historia (*backcast*) y produce un pronóstico parcial. La reconstrucción se resta de la entrada del bloque siguiente; los pronósticos se suman. Así, los bloques pueden representar estructuras temporales complementarias sin una descomposición estacional impuesta.

La configuración seleccionada usa 30 bloques independientes, cada uno con cuatro capas densas de 512 unidades, ReLU y dropout 0,1. El residuo contiene las 168 horas de demanda en orden inverso. Cada bloque incorpora además las últimas 24 horas de las 29 covariables y tres máscaras de ausencia, y las 23 variables de calendario para las 24 horas futuras: la entrada tiene dimensión $168+24\times32+24\times23=1488$.

$$
u_b=[r_b,\operatorname{vec}(X_{exo,24}),\operatorname{vec}(C_f)],\quad
h_b=\operatorname{MLP}_b(u_b),\quad [\widehat b_b,\widehat f_b]=W_bh_b+a_b,
$$ (eq-arquitecturas-nbeats-001)
$$
r_{b+1}=r_b-\widehat b_b,\qquad
\widehat{\mathbf y}=\ell_t\left(\mathbf1+\sum_{b=1}^{30}\widehat f_b\right).
$$ (eq-arquitecturas-nbeats-002)

La base genérica convierte directamente los coeficientes aprendidos en reconstrucción y pronóstico; no corresponde a las bases interpretables de tendencia y estacionalidad de otra variante de N-BEATS. Las covariables se mantienen en todos los bloques y solo se actualiza el residuo de demanda. El modelo contiene 49.464.960 parámetros. Su coste denso crece aproximadamente como $O(B[d_{in}w+3w^2+w(L+H)])$, con $B=30$ bloques y ancho $w=512$.

La arquitectura sigue el mecanismo de doble residuo de [Oreshkin et al., N-BEATS, ICLR 2020](https://arxiv.org/abs/1905.10437), ampliado con las covariables numéricas y el calendario del proyecto.

## Esquema global

[![Arquitectura global de nbeats](../_static/arquitecturas_presentacion/nbeats_global.png)](../_static/arquitecturas_presentacion/nbeats_global.png)

Los bloques reducen sucesivamente el residuo de la demanda y suman sus contribuciones a las siguientes 24 horas.

## Detalle de las operaciones

[![Operaciones de nbeats](../_static/arquitecturas_presentacion/nbeats_detalle.png)](../_static/arquitecturas_presentacion/nbeats_detalle.png)

La reconstrucción de 168 valores y el pronóstico de 24 valores son dos salidas del mismo bloque; solo la reconstrucción modifica la entrada del siguiente.

El {ref}`código de generación de estos esquemas <esquemas-top10-detallados>` conserva sus dimensiones y conexiones.


## Código de la arquitectura

```{literalinclude} ../_static/codigo_e2/nbeats_red.py
:language: python
:linenos:
```

<!-- diseno-entrenamiento:inicio -->

## Diseño de entrenamiento y prueba

(diseno-entrenamiento-nbeats)=

### Selección en validación

Se evaluaron **20 configuraciones** y 5 llegaron a la comparación de selección. La **configuración 16** obtuvo el menor MASE de validación (0,6908) sobre 2018–2021. La configuración fijada para prueba es **16**, con 5 semillas.

#### Configuración conservada
| Parámetro | Valor |
|---|---|
| Pilas | 30 |
| Bloques por pila | 1 |
| Capas densas por bloque | 4 |
| Ancho | 512 |
| Dropout | 0,1 |
| Tasa de aprendizaje | 0,000357243 |
| Decaimiento de pesos | 0 |
| Lote de ajuste | 64 |

#### Etapas del entrenamiento y prueba

| Etapa | Especificación |
|---|---|
| Filtro inicial | 20 configuraciones; semilla 43; validación 2018 |
| Confirmación y criterio | Cinco finalistas; semilla 43; evaluación adicional 2019–2021; media de los cuatro años 2018–2021. Menor MASE medio de validación; cada año tiene el mismo peso. |
| Ajuste final | 2000-01-09 00:00:00-05:00 a 2019-10-20 00:00:00-05:00 |
| Parada interna final | 2019-10-21 00:00:00-05:00 a 2021-12-31 00:00:00-05:00 |
| Prueba | 2022-01-01 a 2025-12-31; 1.461 orígenes diarios |
| Semillas finales | 1000, 1001, 1002, 1003, 1004 |
| Historia y horizonte | 168 y 24 horas, respectivamente |
| Pérdida y optimizador | MAE de demanda dividida por el nivel histórico; AdamW |
| Épocas y parada | Máximo 200; mínimo 10; paciencia efectiva 6; restauración del menor valor interno |
| Tasa de aprendizaje | ReduceLROnPlateau: factor 0,5; paciencia 2 |
| Lotes | Ajuste: 64; evaluación de prueba: hasta 256 orígenes por lote |
| Épocas ejecutadas | 17–25; épocas de los checkpoints: 14, 13, 17, 11, 19 |


#### Configuraciones evaluadas en desarrollo
El filtro y la selección corresponden a periodos y cantidades de repeticiones distintos. El guion identifica las configuraciones que no avanzaron a confirmación; no representa un resultado nulo.
| Configuración | Hiperparámetros | MASE del filtro | MASE de validación | Estado |
|---|---|---:|---:|---|
| **16** | Pilas: 30; Bloques por pila: 1; Capas densas por bloque: 4; Ancho: 512; Dropout: 0,1; Tasa de aprendizaje: 0,000357243; Decaimiento de pesos: 0; Lote de ajuste: 64 | 0,4891 | 0,6908 | **Seleccionada para prueba** |
| 9 | Pilas: 30; Bloques por pila: 1; Capas densas por bloque: 4; Ancho: 512; Dropout: 0; Tasa de aprendizaje: 0,000167913; Decaimiento de pesos: 0,0001; Lote de ajuste: 64 | 0,5255 | 0,6933 | Finalista |
| 10 | Pilas: 30; Bloques por pila: 1; Capas densas por bloque: 4; Ancho: 512; Dropout: 0,1; Tasa de aprendizaje: 0,000570889; Decaimiento de pesos: 0,0001; Lote de ajuste: 128 | 0,4951 | 0,7019 | Finalista |
| 6 | Pilas: 30; Bloques por pila: 1; Capas densas por bloque: 4; Ancho: 512; Dropout: 0,1; Tasa de aprendizaje: 0,000554015; Decaimiento de pesos: 0; Lote de ajuste: 64 | 0,4999 | 0,7069 | Finalista |
| 18 | Pilas: 30; Bloques por pila: 1; Capas densas por bloque: 4; Ancho: 512; Dropout: 0,1; Tasa de aprendizaje: 0,000240778; Decaimiento de pesos: 0; Lote de ajuste: 128 | 0,5124 | 0,7187 | Finalista |
| 3 | Pilas: 30; Bloques por pila: 1; Capas densas por bloque: 4; Ancho: 512; Dropout: 0,1; Tasa de aprendizaje: 0,000264241; Decaimiento de pesos: 0; Lote de ajuste: 128 | 0,5259 | — | Filtro inicial |
| 11 | Pilas: 30; Bloques por pila: 1; Capas densas por bloque: 4; Ancho: 512; Dropout: 0,1; Tasa de aprendizaje: 0,000439967; Decaimiento de pesos: 1e-05; Lote de ajuste: 64 | 0,5273 | — | Filtro inicial |
| 19 | Pilas: 30; Bloques por pila: 1; Capas densas por bloque: 4; Ancho: 512; Dropout: 0; Tasa de aprendizaje: 0,000408621; Decaimiento de pesos: 0; Lote de ajuste: 64 | 0,5294 | — | Filtro inicial |
| 5 | Pilas: 30; Bloques por pila: 1; Capas densas por bloque: 4; Ancho: 512; Dropout: 0; Tasa de aprendizaje: 0,000263148; Decaimiento de pesos: 0; Lote de ajuste: 64 | 0,5348 | — | Filtro inicial |
| 15 | Pilas: 30; Bloques por pila: 1; Capas densas por bloque: 4; Ancho: 512; Dropout: 0; Tasa de aprendizaje: 0,000393724; Decaimiento de pesos: 0; Lote de ajuste: 128 | 0,5438 | — | Filtro inicial |
| 14 | Pilas: 30; Bloques por pila: 1; Capas densas por bloque: 4; Ancho: 512; Dropout: 0,1; Tasa de aprendizaje: 0,00036962; Decaimiento de pesos: 0; Lote de ajuste: 256 | 0,5450 | — | Filtro inicial |
| 17 | Pilas: 30; Bloques por pila: 1; Capas densas por bloque: 4; Ancho: 512; Dropout: 0,1; Tasa de aprendizaje: 0,000105918; Decaimiento de pesos: 1e-05; Lote de ajuste: 128 | 0,5462 | — | Filtro inicial |
| 1 | Pilas: 30; Bloques por pila: 1; Capas densas por bloque: 4; Ancho: 512; Dropout: 0,1; Tasa de aprendizaje: 0,000613748; Decaimiento de pesos: 0,0001; Lote de ajuste: 256 | 0,5483 | — | Filtro inicial |
| 20 | Pilas: 30; Bloques por pila: 1; Capas densas por bloque: 4; Ancho: 512; Dropout: 0,1; Tasa de aprendizaje: 0,000242018; Decaimiento de pesos: 0,0001; Lote de ajuste: 256 | 0,5514 | — | Filtro inicial |
| 8 | Pilas: 30; Bloques por pila: 1; Capas densas por bloque: 4; Ancho: 512; Dropout: 0; Tasa de aprendizaje: 0,000194616; Decaimiento de pesos: 0; Lote de ajuste: 128 | 0,5544 | — | Filtro inicial |
| 4 | Pilas: 30; Bloques por pila: 1; Capas densas por bloque: 4; Ancho: 512; Dropout: 0; Tasa de aprendizaje: 0,000191509; Decaimiento de pesos: 0,0001; Lote de ajuste: 128 | 0,5607 | — | Filtro inicial |
| 7 | Pilas: 30; Bloques por pila: 1; Capas densas por bloque: 4; Ancho: 512; Dropout: 0,1; Tasa de aprendizaje: 0,000109654; Decaimiento de pesos: 1e-05; Lote de ajuste: 256 | 0,5648 | — | Filtro inicial |
| 13 | Pilas: 30; Bloques por pila: 1; Capas densas por bloque: 4; Ancho: 512; Dropout: 0; Tasa de aprendizaje: 0,000154096; Decaimiento de pesos: 0,0001; Lote de ajuste: 128 | 0,5660 | — | Filtro inicial |
| 12 | Pilas: 30; Bloques por pila: 1; Capas densas por bloque: 4; Ancho: 512; Dropout: 0; Tasa de aprendizaje: 0,000299397; Decaimiento de pesos: 0; Lote de ajuste: 256 | 0,5704 | — | Filtro inicial |
| 2 | Pilas: 30; Bloques por pila: 1; Capas densas por bloque: 4; Ancho: 512; Dropout: 0; Tasa de aprendizaje: 0,000144134; Decaimiento de pesos: 0,0001; Lote de ajuste: 256 | 0,6194 | — | Filtro inicial |

[Resultados de desarrollo por año y semilla](../_static/resultados/diseno_entrenamiento/nuevo_nbeats__N.csv) · [Configuraciones completas](../_static/resultados/diseno_entrenamiento/nuevo_nbeats__N_configuraciones.csv).

<!-- diseno-entrenamiento:fin -->

## Pipeline de entrenamiento y prueba

<!-- entrenamiento-portable:inicio -->
### Comandos de entrenamiento y evaluación

Validación temporal de la configuración seleccionada:

```bash
python scripts/entrenar_arquitectura.py --modelo nbeats --anio 2021 --semilla 1000 --dispositivo auto --salida salidas/nbeats_validacion_1000
```

Comprobación funcional breve:

```bash
python scripts/entrenar_arquitectura.py --modelo nbeats --smoke --dispositivo cpu --salida salidas/nbeats_funcional
```

Evaluación final, con ajuste y parada anteriores a 2022:

```bash
python scripts/entrenar_arquitectura.py --modelo nbeats --fase prueba --allow-test --semilla 1000 --dispositivo auto --salida salidas/nbeats_prueba_1000
```

El código completo de preparación, optimización, guardado y restauración está en el {ref}`pipeline ejecutable compartido <entrenamiento-portable>`. Cada salida debe ser una carpeta nueva.
<!-- entrenamiento-portable:fin -->

Se comparan 20 configuraciones en 2018; cinco finalistas se evalúan en 2019–2021 y se selecciona la menor media de los cuatro años. El ajuste final utiliza cinco semillas (1000–1004), máximo de 200 épocas, mínimo de 10 y paciencia de 6. El checkpoint se restaura en el mínimo de parada interna. La reserva contiene 803 orígenes de 2019-10-21 a 2021-12-31; la prueba comprende 1.461 orígenes de 2022–2025. Las imputaciones y escalas se ajustan con el tramo de entrenamiento. Precio, hidrología y clima se usan como historia con sus rezagos de disponibilidad.

### Preparación de datos

```{literalinclude} ../_static/codigo_e2/top10_adicionales/src/entrenamiento/datos.py
:language: python
:linenos:
```

### Selección y ajuste con parada temprana

```{literalinclude} ../_static/codigo_e2/top10_adicionales/src/entrenamiento/protocolo_e2.py
:language: python
:pyobject: modelo_estandar
:linenos:
```

```{literalinclude} ../_static/codigo_e2/top10_adicionales/src/entrenamiento/protocolo.py
:language: python
:pyobject: corrida
:linenos:
```

### Optimización y restauración del checkpoint

```{literalinclude} ../_static/codigo_e2/top10_adicionales/src/entrenamiento/motor.py
:language: python
:pyobject: entrenar
:linenos:
```

### Prueba por origen con pesos congelados

El siguiente código restaura la configuración y los escaladores del checkpoint y obtiene las métricas sin modificar los pesos. TimesNet se evalúa con lote uno.

```{literalinclude} ../_static/codigo_e2/top10_adicionales/pipeline_prueba.py
:language: python
:linenos:
```

Los resultados, curvas de cinco semillas, métricas por horizonte y comparaciones están en la {ref}`evaluación integrada <evaluacion-e2>`.

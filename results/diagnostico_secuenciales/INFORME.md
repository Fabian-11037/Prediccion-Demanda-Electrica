# Diagnóstico exploratorio de LSTM y CNN1D

Se inspeccionó el código y se reconstruyeron los checkpoints del piloto para validación 2018. Se usaron las cinco semillas 100–104 de las configuraciones seleccionadas. No se accedió al conjunto de prueba ni se modificaron modelos o resultados de la ejecución base. Este análisis se realizó después de conocer los resultados finales; es exploratorio.

## Sensibilidad a entradas

Se permutó exclusivamente la demanda histórica normalizada entre días, conservando calendario, etiquetas y nivel de la ventana original utilizado para invertir la normalización. Por tanto, esta intervención mide el uso de la forma temporal de la demanda; no elimina el nivel histórico que aporta la normalización. En una intervención separada se permutaron conjuntamente el calendario pasado y futuro entre días; se conserva la distribución marginal de los canales y las constantes por hora.

| Modelo | MASE original | Demanda permutada | Cambio | Calendario permutado |
|---|---:|---:|---:|---:|
| mlp | 0.3966 | 1.4574 | +267.44% | 2.2426 |
| lstm | 1.3430 | 1.3433 | +0.03% | 2.9049 |
| cnn1d | 1.3591 | 1.3594 | +0.02% | 2.8759 |

LSTM y CNN1D muestran muy poca sensibilidad a la forma histórica de demanda y fuerte dependencia del calendario. MLP sí utiliza la demanda histórica. Esto identifica una limitación de las soluciones aprendidas, no prueba por sí solo un error del optimizador. La evidencia se limita al pliegue 2018.

## Reconstrucción y prueba de ajuste pequeño

Las predicciones guardadas de la semilla 100 se reconstruyeron con error máximo de 0 kWh para los tres modelos. Se verificó la ruta común de normalización, inversión, actualización del optimizador y restauración del mejor estado; no se identificó un fallo evidente en estas operaciones.

Se realizaron 400 actualizaciones sobre 16 ejemplos tomados uniformemente del entrenamiento del primer pliegue, con semilla 42 y lote completo, sin validación ni parada temprana. El control conserva la arquitectura y pérdida, usa lr=0.001 y elimina dropout y weight_decay. Es una prueba de capacidad/optimización, no una estimación de generalización. El scheduler de las configuraciones originales no se ejecuta en esta prueba de pasos; no replica el entrenamiento original. `diagnostico.json` contiene las pérdidas cada 100 pasos. MLP y LSTM alcanzaron errores normalizados cercanos a cero en controles intermedios, pero repuntaron posteriormente: capacidad de ajuste y sensibilidad a optimización. CNN mejoró sin alcanzar interpolación en este presupuesto; no se concluye incapacidad definitiva. La ablación inicial con calendario cero se excluye de la interpretación porque produce entradas fuera de distribución, especialmente en las columnas constantes que MLP vuelve a estandarizar.

## Factores candidatos y recomendación

- La demanda entra a secuenciales como cociente respecto al nivel, centrada alrededor de uno, junto a muchos canales de calendario estandarizados. MLP añade una estandarización tabular; la diferencia puede favorecer desequilibrios entre entradas, pero no demuestra causalidad.
- La rama directa de calendario ofrece un atajo y las perturbaciones muestran que predomina en las soluciones inspeccionadas.
- CNN seleccionada: una convolución de kernel 7 y promedios de bloques de 24 horas. Esa compresión puede dificultar representar perfiles intradiarios y rezagos específicos.
- LSTM resume la historia mediante el último estado; su búsqueda quedó limitada a cinco configuraciones. CNN usa lr≈0.00838; LSTM≈0.00321, ambos con dropout. Su papel causal requiere variar un factor a la vez.

No repetir la búsqueda original sin cambios. Realizar una fase exploratoria acotada y separada, exclusivamente en entrenamiento/validación: controlar lr, escala de demanda y contribución de calendario; para CNN, comparar pooling con una representación que conserve la posición horaria. Mantener semillas y particiones, variar un factor por comparación, y exigir que cambie la sensibilidad a demanda además de reducir MASE. No presentar nuevas mejoras sobre el test ya observado como confirmatorias.

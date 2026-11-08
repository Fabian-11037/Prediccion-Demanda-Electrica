# Adenda: incorporación de CNN1D a la ejecución v2

Fecha: 7 de noviembre de 2026, antes de entrenar CNN1D. El espacio de arquitectura y la búsqueda de regularización fueron confirmados antes de esta ejecución. Se incorpora un cuarto modelo sin modificar las opciones de ARX lineal, MLP o LSTM ni sus resultados completados. El preregistro original v1/v2 se conserva sin cambios.

## Arquitectura y espacio

Historia de 168 horas: demanda real normalizada por el nivel de su ventana y calendario histórico. La CNN recibe canales por tiempo, con 1, 2 o 3 bloques Conv1d–ReLU–Dropout1d; cada bloque usa el mismo ancho (32, 64 o 128 filtros), kernel 3, 5 o 7, paso 1, dilatación 1 y padding simétrico kernel//2. Todas las posiciones pertenecen a la historia anterior al origen.

Se aplica promedio adaptativo a siete posiciones y una proyección de 7*filtros a filtros, con ReLU y Dropout. Una rama separada proyecta el calendario conocido de las 24 horas futuras a filtros unidades. Las dos representaciones se concatenan y se procesan mediante Dense(2*filtros,filtros)–ReLU–Dropout–Dense(filtros,24). Salida lineal, conforme al pipeline de normalización y los demás modelos. No se incorporan nuevas covariables ni clima futuro realizado.

Espacio confirmado: capas {1,2,3}; filtros {32,64,128}; kernel {3,5,7}; dropout uniforme[0,0.5]; weight_decay log-uniforme[1e-6,1e-2]; batch_size {32,64,128,256}. Se mantienen los parámetros comunes: lr log-uniforme[1e-4,1e-2], scheduler {none,cosine,plateau}, pérdida {mse,mae,huber}, paciencia {8,15,25}; AdamW, recorte 1, máximo200épocas y min_delta=1e-6. Weight decay es el de AdamW; no se añade una segunda penalización L2 explícita.

## Presupuesto, selección y etapas

20configuraciones aleatorias para CNN1D, semillas1/2/3, cuatro pliegues anuales2018–2021:240ajustes adicionales. La búsqueda total pasa a960ajustes (80configuraciones entre cuatro modelos), manteniendo las opciones de los tres modelos existentes. Selección por MASE medio entre semillas y pliegues; parada temprana con el último10% cronológico del entrenamiento, con purga, no con el bloque anual de evaluación.

El piloto incorpora cinco semillas deCNN1D en cuatro pliegues:20ajustes adicionales,80en total. El número de semillas finales común se determina con los pilotos de los cuatro modelos, mínimo5por modelo y el máximo requerido por la aproximación de potencia ya preregistrada. La etapa final incorporaCNN1D bajo las reglas originales:configuración fijada antes de prueba y resultados guardados por semilla. No se aplican pruebas estadísticas nuevas sin su preregistro.

Los valores de prueba ya fueron consultados para modelos de la v1. Esta ampliación es posterior a esa consulta y no constituye diseño confirmatorio independiente. No se utilizó el desempeño de prueba para escoger el espacio deCNN1D; las decisiones se basan en la especificación de arquitectura y el presupuesto común.

## Continuidad y trazabilidad

Se interrumpe de forma controlada la ejecución para incorporar la nueva familia. Los ajustes completos, curvas, predicciones y pesos se conservan; un ajuste parcial se repite desde el inicio. El plan de tres modelos y sus fuentes se archivan en `results/base_v2/plan_gpu_inicial_tres_modelos.json` y `fuentes_tres_modelos/`. La migración requiere el argumento explícito `--incorporar-cnn`, verifica que los datos y opciones comunes no cambien y queda registrada en `incorporacion_cnn.json`. La ejecución reanuda con cuatro modelos y mantiene el mismo panel local de progreso.

Nota de implementación previa a entrenarCNN1D: la retropropagación de AdaptiveAvgPool1d no dispone de implementación determinista en la GPU utilizada. ParaL=168,se implementa mediante reshape a7bloques de24horas y media dentro de cada bloque. Es matemáticamente equivalente al pooling adaptativo a7posiciones con esta longitud;se comprueba equivalencia de valores y gradientes en pruebas. Se conserva el determinismo estricto y no se cambia el espacio confirmado.

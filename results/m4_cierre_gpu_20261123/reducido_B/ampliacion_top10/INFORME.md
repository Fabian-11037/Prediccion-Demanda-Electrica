# Evaluación temporal final y Top10 histórico

Se integran TimesNet y N-BEATS con las pérdidas de las predicciones individuales restauradas de sus cinco checkpoints originales; si faltan, se usan las pérdidas diarias publicadas y se identifica la procedencia. Son errores individuales promediados, distintos del error de la predicción ensemble. No se inventan pronósticos ni réplicas.

El CSV final utiliza49999 réplicas, tres procedimientos y Holm de la familia máxima1536. Sustituye únicamente la resolución Monte Carlo de la nueva evaluación, conservando el paso preliminar y las familias históricas28/220. Colombia sigue siendo retrospectiva. El Top10 histórico conserva cinco semillas, frente a diez de los ajustes nuevos. No se certifica potencia0,90, originalidad ni validez confirmatoria de pruebas cuyos supuestos no estén establecidos.

GW utiliza la pérdida observable de dos días antes y excluye los dos orígenes iniciales. El GW preliminar de rezago1 no se interpreta; se conserva como procedencia. Consultar la adenda registrada antes de nuevas pruebas.

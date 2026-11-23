# M1 · evaluación retrospectiva 2022–2025

Evaluación en GPU de 1.461 orígenes diarios, horizonte de 24 horas y cinco semillas (101–105). Configuración 8: ancho 64, dropout 0,1 y coeficiente auxiliar 0,1. Pesos y escaladores congelados; no hubo entrenamiento ni selección adicional.

| Modelo | MASE medio | DE entre semillas | MAE (kWh) | RMSE (kWh) |
|---|---:|---:|---:|---:|
| NNLS | 0.620904 | 0.008338 | 133,877.92 | 176,527.32 |
| M1 | 0.634923 | 0.010050 | 136,900.69 | 180,108.41 |
| media | 0.643461 | 0.011127 | 138,741.69 | 181,864.70 |
| convexa_fija | 0.647729 | 0.006769 | 139,662.05 | 182,922.53 |
| periodica | 0.677992 | 0.017774 | 146,187.32 | 191,033.73 |
| patchtst | 0.683149 | 0.029009 | 147,299.20 | 196,124.79 |
| E | 0.738865 | 0.030648 | 159,312.42 | 205,586.91 |

M1 reduce el MASE un 7,06 % frente a PatchTST, el mejor experto seleccionado en desarrollo. El IC95 puntual de esa reducción es [5,71 %, 8,45 %]. El contraste unilateral supera tanto el margen de 0 % como el de 3 %: p sin ajuste = 0,0001; cota superior conservadora de p ajustado global = 0,0028.

Frente a NNLS, M1 incrementa el MASE un 2,26 %; el IC95 puntual de la reducción relativa es [−2,93 %, −1,58 %]. NNLS obtiene el menor error entre los procedimientos evaluados. Esta comparación no prueba equivalencia y no se utilizó para modificar el modelo.

## Protocolo e interpretación

La unidad estadística es el origen diario. Se promedian los errores de las 24 horas y las cinco semillas antes de remuestrear. Se emplean 9.999 réplicas de bootstrap pareado con bloques circulares de 14 días dentro de cada año; las sensibilidades de 7 y 28 días conservan la conclusión. La desviación entre semillas describe variación por inicialización, no incertidumbre temporal. El RMSE informado es la media del RMSE de cada semilla.

Se conserva la familia prevista de 28 hipótesis. Las otras 24 aún no se han evaluado: se informa una cota superior conservadora de Holm al completarlas con unos, no sus p-valores observados ni el ajuste final del estudio completo. Los IC95 son puntuales, sin ajuste simultáneo.

Los errores usados como entradas tienen dos días de retraso. Su historia se inicializa con predicciones OOF anteriores a 2022; durante prueba se incorporan únicamente errores ya disponibles, sin actualizar pesos ni escaladores. Los rasgos reproducen exactamente la receta del entrenamiento y permanecen invariantes a la perturbación de objetivos contemporáneos y futuros.

El benchmark de estos años ya era conocido durante el diseño: esta evaluación es retrospectiva e informada. El control sin pérdida auxiliar carece de checkpoint final y no se evaluó ni reentrenó. No se publicaron resultados automáticamente.

## Artefactos

Predicciones y pesos de mezcla por fecha, hora y semilla: cinco NPZ y cinco CSV comprimidos. Métricas por semilla, año y horizonte; contrastes principales y sensibilidades; registro previo; verificación de causalidad y conservación de originales; fuentes de la ejecución y manifiesto SHA-256.

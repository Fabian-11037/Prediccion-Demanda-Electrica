# M3 · selector de expertos condicionado por texto

Prueba retrospectiva 2022–2025:1.461 orígenes diarios,24 horas y cinco semillas. M1 y E5 congelados. Selección exclusiva de M3_OP con desarrollo2018–2021 y aplicación de su configuración a variantes, controles y sensibilidad estricta.

| Procedimiento | MASE | DE semillas | MAE kWh | RMSE kWh |
|---|---:|---:|---:|---:|
| NNLS | 0.620904 | 0.008338 | 133877.92 | 176527.32 |
| M3_OP_estricto | 0.634256 | 0.010404 | 136756.99 | 179813.21 |
| M1 | 0.634923 | 0.010050 | 136900.69 | 180108.41 |
| M3_desalineado_301 | 0.636132 | 0.009260 | 137161.38 | 180475.75 |
| M3_O | 0.636160 | 0.011708 | 137167.45 | 180477.99 |
| M3_desalineado_49 | 0.636221 | 0.011573 | 137180.70 | 180557.39 |
| M3_desalineado_217 | 0.636500 | 0.008298 | 137240.82 | 180580.94 |
| M3_desalineado_91 | 0.636916 | 0.010947 | 137330.58 | 180615.68 |
| M3_N | 0.637496 | 0.009615 | 137455.48 | 180858.57 |
| M3_OP | 0.637692 | 0.011461 | 137497.80 | 180908.38 |
| M3_P | 0.637728 | 0.010179 | 137505.60 | 180853.19 |
| M3_desalineado_154 | 0.637735 | 0.010488 | 137507.10 | 180994.92 |
| M3_metadatos | 0.638337 | 0.012143 | 137636.86 | 181080.24 |
| patchtst | 0.683149 | 0.029009 | 147299.20 | 196124.79 |

La base de entrenamiento se obtiene por OOF anidado de M1 desde2013: cada configuración se elige solo con folds anteriores a la fila predicha. El primer prefijo usa cfg1 predefinida al no existir un fold previo. No se usa retrospectivamente el ganador global2018–2021.

Se adopta el contrato textual específico de §5 del diseño Markdown: fragmentos1500 caracteres, solape200, máximo16; reservas32 alineados,48 oficiales y48 prensa, con ventanas72h y alineados120 días. Este detalle difiere del bloque genérico encoder del JSON; la decisión queda registrada antes de entrenar. E5 y vectores existentes se reutilizan con SHA-256. Las fuentes se mapean solo en ajuste de cada fold; las nuevas se tratan como desconocidas. No se ajusta E5.

Dos agregadores separados con atención condicionada por entradas numéricas. Perturbación de logits acotada por epsilon y compuerta aprendida; sin texto, identidad exacta con M1. Coeficiente de complementariedad0,1 y suavidad0,001 fijos; penalización de delta según rejilla. El control numérico aproxima el número de parámetros activos mediante una MLP independiente. Las variantes desplazadas usan días anteriores reales, nunca texto futuro.

Bootstrap por origen,9999 réplicas, bloques14 con sensibilidad7/28 dentro del año. Las semillas se promedian en pérdidas antes de remuestrear. Se conservan18 hipótesis observadas de M1/M2/M3; quedan10 de M4 pendientes. Cota conservadora de Holm global completada con unos; IC95 puntuales. El control desplazado se agrega promediando pérdidas, no predicciones.

Los resultados del benchmark ya se conocían al diseñar: evaluación retrospectiva informada. No se afirma aporte semántico por superar únicamente una base débil. La sensibilidad estricta se mantiene separada. No se publica ni despliega automáticamente.

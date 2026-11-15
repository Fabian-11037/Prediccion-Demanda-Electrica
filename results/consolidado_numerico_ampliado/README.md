# Resultados numéricos ampliados · noviembre de2026

Prueba retrospectiva 2022–2025, historia de 168 h, horizonte de 24 h y origen diario a las 00:00 de Colombia. Los modelos usan las mismas 30 variables numéricas históricas, 23 de calendario y 30 máscaras, con escalado por tramo de optimización. Se evaluaron cinco semillas finales (1000–1004) por modelo.

| Modelo | MASE validación | MASE prueba | DE entre semillas | MAE (kWh) | sMAPE (%) | Mejora frente a ingenuo semanal |
|---|---:|---:|---:|---:|---:|---:|
| PatchTST_adaptado | 0.6216 | 0.7256 | 0.0189 | 156463 | 1.70 | 46.44% |
| NHITS_adaptado | 0.9304 | 1.0236 | 0.0571 | 220713 | 2.40 | 24.45% |
| TFT_adaptado | 0.8882 | 1.1469 | 0.0464 | 247296 | 2.70 | 15.35% |

## Presupuesto y alcance

- PatchTST: P20, 130 ajustes; N-HiTS: N14, 130 ajustes. Cada corrida incluye 105 de búsqueda, 20 de piloto y cinco finales. La selección usa confirmación 2019–2021 con tres semillas.
- TFT: T02, 62 ajustes tras una reducción por coste: 51 de filtro, seis de confirmación y cinco finales. La confirmación usa una semilla (43) en 2019–2021; se omitió el piloto.
- Los tres usan una arquitectura fija adaptada; no se afirma réplica literal de los artículos ni exención automática del requisito académico de 1.000 arquitecturas. Se conservan los planes, adendas y fuentes.
- El piloto no permitió dimensionar potencia suficiente para el margen relativo del 3 % en PatchTST y N-HiTS. Las cinco semillas de prueba son un presupuesto exploratorio; TFT tampoco acredita potencia. La desviación estándar entre semillas no es un intervalo de confianza temporal. No se afirma significancia, equivalencia ni superioridad confirmatoria.
- La prueba se observó repetidamente en el proyecto. Las diferencias entre modelos son descriptivas y los presupuestos de selección no son iguales. Los baselines v3/v4 conservados usan demanda y calendario; no constituyen una comparación de arquitectura con información idéntica.
- Referencias comunes: ingenuo semanal, MASE 1,3549064443; ingenuo diario, MASE 2,0767245486. El denominador MASE fijo es 215.617,90361589016 kWh.

## Artefactos publicados

`modelos_finales.json` referencia los 15 checkpoints finales, escaladores, historiales, predicciones y métricas. Cada modelo se reconstruye con su clase en `demanda.patchtst_numerico`, `demanda.nhits_numerico` o `demanda.tft_numerico`, usando la configuración y sigma_demanda del checkpoint. Las dependencias incluidas y sus licencias acompañan al código.

Se publican predicciones NPZ por origen UTC y semilla de todos los ajustes completos, historiales completos por época, escaladores, identidades y métricas. Los CSV comprimidos de esta carpeta exportan las predicciones finales con origen y fecha objetivo en America/Bogota, semilla y horizonte.

Los pesos intermedios de búsqueda y piloto se conservan localmente y se excluyen de Git por volumen (cerca de 9 GB). Los 15 pesos finales sí están publicados. Los estados provisionales, bloqueos y servicios locales no forman parte de los resultados científicos publicados. `indice_publicado.csv` contiene tamaños y SHA-256 de los artefactos incluidos en esta publicación.

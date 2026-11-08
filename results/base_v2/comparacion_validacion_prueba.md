# Comparación de validación y prueba

Validación: promedio de cinco semillas piloto sobre cuatro pliegues anuales (2018–2021). Prueba: promedio de 195 semillas finales en 2022–2025. MASE utiliza la misma escala fijada en entrenamiento. La brecha porcentual es 100 × (MASE_prueba / MASE_validación − 1).

| Modelo | MASE validación | MASE prueba | Brecha |
|---|---:|---:|---:|
| arx_lineal | 0.628 | 0.831 | +32.4 % |
| mlp | 0.521 | 0.692 | +32.9 % |
| lstm | 1.513 | 1.748 | +15.5 % |
| cnn1d | 1.604 | 2.070 | +29.0 % |

La brecha mide deterioro de generalización temporal; no identifica por sí sola sobreajuste. Cambian los periodos, el tamaño de entrenamiento y las semillas. La validación también intervino en la selección de hiperparámetros. Para diagnosticar sobreajuste se requieren curvas de entrenamiento y validación dentro de cada ajuste. No se ejecutaron nuevos entrenamientos ni evaluaciones sobre prueba.

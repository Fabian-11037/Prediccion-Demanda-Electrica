# Resultados de entrenamiento GPU v2

Ejecución completada el 8 de noviembre de 2026. Demanda horaria del SIN; historia de 168 horas y horizonte de 24 horas. Validación forward chaining: 2018–2021; prueba: 2022–2025. GPU AMD Radeon RX 6700 XT, PyTorch con ROCm, determinismo estricto.

## Archivos publicados

- `busqueda.csv`, `mejor_config.json`: búsqueda y configuración seleccionada por familia.
- `piloto.csv`, `piloto_resumen.json`: cinco semillas por familia sobre cuatro pliegues; dimensionamiento de 195 semillas finales por familia.
- `final_runs.csv`, `resumen_final.csv`: métricas individuales y agregadas de prueba.
- `comparacion_validacion_prueba.md` y `.csv`: brecha de generalización temporal.
- `resumen_ajustes.csv`: métricas y pérdidas resumidas por ajuste completado.
- `referencias_cv.json`, `referencias_test.json`: baselines estacionales.
- Planes, entornos y registros de modificación: trazabilidad del entrenamiento.

Los pesos, predicciones por origen e historiales completos permanecen en el almacenamiento local, en `ajustes/`, `preds/` e `hist/`. No se incluyen en Git por su volumen; los hashes de pesos y predicciones figuran en `resultados_guardados.json`. Los archivos publicados permiten consultar resultados y configuración, pero no sustituyen esos artefactos para reproducir las predicciones exactas.

## Interpretación y limitaciones

MLP obtuvo el menor MASE de prueba (0,692), seguido de ARX (0,831), ambos inferiores al baseline semanal (1,355). LSTM (1,748) y CNN1D (2,070) no superaron esa referencia. No se han realizado pruebas de significancia. Los intervalos de `resumen_final.csv` reflejan variación entre semillas, no incertidumbre de muestreo temporal.

La búsqueda LSTM se redujo durante su ejecución a cinco configuraciones por costo computacional; las demás familias evaluaron veinte. Véase `ADENDA_LIMITE_LSTM.md` en la raíz. La evaluación final utilizó 195 semillas por familia. La brecha validación–prueba no identifica por sí sola sobreajuste.

# Selección de PatchTST · 16 de noviembre de 2026

Se conserva únicamente adj_modelo6_20261112_1035__N, presentado como PatchTST · N: MASE 0.672502, MAE 145003.39 kWh y RMSE 194117.80 kWh. Su MAE es 7.32 % menor que el del otro resultado numérico disponible. La elección fue solicitada después de observar la prueba y queda registrada como decisión retrospectiva; no cambia los pesos ni las métricas individuales.

La evaluación utiliza 13 procedimientos, 65 realizaciones y dos referencias. Se recalculan 105 comparaciones temporales, 78 entre semillas, Holm, rangos, Friedman, Nemenyi y MCS. Friedman Q=54.382418, p=2.33475e-07; diferencia crítica Nemenyi=8.159475. DM–HLN Holm rechaza 98 de105 parejas. Los tres contrastes textuales mantienen sus controles y resultados.

Se actualizan curvas, precisión, observado–predicho, semana, horizontes, segmentos, residuos, importancia, latencia y las tablas descargables. Se usan cinco pilotos externos y ocho reservas internas (65 métricas de desarrollo); el dimensionamiento externo tiene10 parejas y máximo conservador76. Los diagnósticos crudos conservan el inventario completo en results/diagnosticos_ampliados; la presentación y las descargas seleccionan los procedimientos vigentes.

La ficha y ambos esquemas de PatchTST describen configuración18: 33canales, dimensión64, dos capas, cuatro cabezas, dos lecturas y466480parámetros, sin RevIN interno. El código conservado es Modelo6_TaTS/src/modelo6/redes.py, varianteN; las configuracionesN+O permanecen identificadas en TaTS. Se revisan visualmente las21 figuras de evaluación y los dos diagramas nuevos.

Política persistente: config/seleccion_patchtst.json y scripts/seleccion_patchtst.py. Respaldo de tablas, arrays y fuentes documentales antes del cambio: results/seleccion_patchtst/respaldo. Los pesos y las fuentes de ambos experimentos permanecen disponibles. Sin entrenamiento, inferencia de redes ni publicación.

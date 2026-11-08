# Revisión de consistencia del Jupyter Book

Fecha: 8 de noviembre de 2026. Alcance: trece cuadernos, introducción, fuentes, diccionario, pipeline, resultados derivados y contexto. Se contrastaron lenguaje, conceptos, cifras y enlaces; no se ajustaron modelos ni se modificaron sus predicciones.

| Aspecto | Resultado de la revisión |
|---|---|
| Problema y variables | Una serie objetivo, demanda real horaria del SIN, con covariables. Horizonte 24, historia 168; entradas ejecutadas: demanda y 23 variables de calendario. |
| Unidades y calendario | Demanda expresada como energía por intervalo, no mezclada con precios o potencia de otros estudios. Domingo/festivo se distingue del resto y de sábado no festivo. |
| Periodos y escalado | EDA2000–2021; validación expansiva 2018–2021; prueba 2022–2025. Estadísticas de características por tramo de optimización; denominador MASE fijo de referencia 2000–2017. |
| Margen | δ relativo único 3 %. Planificación sobre diferencias relativas a MLP piloto; TOST sobre θ_AB=(μ_A−μ_B)/μ_MLP. Umbral operativo 0,97×MASE de MLP. |
| Semillas y potencia | Cinco semillas finales 1000–1004 por familia. El cálculo aproximado exige 25–45 y hasta 60 con ajuste conservador; cinco no acreditan potencia 0,80 para 3 %. |
| Conclusiones estadísticas | t pareadas con Holm y DM detectan diferencias entre las tres redes. Wilcoxon exacto no rechaza con cinco pares; Nemenyi no distingue todas las parejas. Las conclusiones se atribuyen a cada método. |
| ARX frente a CNN1D | No se detecta diferencia temporal significativa ni se establece equivalencia. TOST relativo con Holm: p≈0,859. |
| Latencia | Medida en CPU y separada del ajuste GPU. Veinte checkpoints, calentamiento 20, mediciones 200, batch 1 y un hilo; sólo forward. |
| Cronología | Se declara que el plan LSTM fue posterior a la disponibilidad de otros resultados de prueba y que el conjunto no constituye confirmación independiente. El dimensionamiento es retrospectivo. |
| Literatura | Se separan demanda, precio y métodos generales. Cinco estudios empíricos de demanda en el denominador; resúmenes sólo como contexto. Se incorporan secciones propias por modelo y comparación crítica. |
| Matriz bibliográfica | Se corrigió la clasificación de irradiancia como variable meteorológica; los estudios de otros autores mantienen atribución y alcance. |
| Redacción y estructura | Se consolidaron alcance y discusión, tiempos verbales y términos; se eliminaron rutas repetidas de exportación en tablas. Capítulos 09_validacion y 10_modelos; evaluación 10.9 y pipeline 10.10. |
| Reproducibilidad | Código analítico visible, narrativa nativa, descargas de resultados, hash canónico de etiquetas y referencias cruzadas verificables. |

## Limitaciones que permanecen

La potencia, el presupuesto desigual, la selección adaptativa y la exposición a prueba limitan una conclusión confirmatoria. La cronología no prueba por sí sola qué información influyó en las decisiones. No se recuperaron las cadenas originales ni las fechas de ejecución de Scopus y WoS; no hay un PRISMA conciliado ni revisión independiente duplicada. No se identifica una cota empírica de ruido irreducible.

La coherencia interna del libro no equivale a cumplimiento integral de la pauta. Los resultados cuantitativos se conservan con alcance exploratorio. La medición CPU no se presenta como medición GPU ni como latencia de extremo a extremo.

## Validación

La construcción y la verificación local comprueban ejecución, enlaces, presencia de interpretaciones, cobertura del diccionario y contratos entre resultados. Los resultados de esas comprobaciones quedan en results/jbook/auditoria.json y results/jbook/revision_consistencia.json. No se publica el sitio durante esta revisión.

En ADF, el valor cero del software corresponde al límite de su aproximación de MacKinnon y se reporta como p<0,001; no se interpreta como probabilidad exactamente nula ni como una cota de 1e-300.

Validación final: 43 pruebas correctas y 5 omitidas; las advertencias de pruebas corresponden a APIs de pandas de módulos existentes. El JBook tiene cero errores, advertencias de construcción y enlaces locales rotos. Se conservan cinco semillas por familia, sin entrenamiento ni publicación adicionales.

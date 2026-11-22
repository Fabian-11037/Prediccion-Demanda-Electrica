# Revisión de consistencia y completitud del JBook

Estado local verificado el 2026-11-20. Este informe sustituye el inventario de la revisión del 16 de noviembre de 2026. Los resultados proceden de artefactos conservados; no se ejecuta entrenamiento ni publicación.

## Inventario vigente

13 cuadernos, 96 celdas analíticas, 51 gráficas, 47 tablas y 28 esquemas. La evaluación comprende 15 procedimientos aprendidos, 75 realizaciones, 2 referencias, 136 parejas temporales y 105 parejas entre semillas.

La precisión distingue 5 pilotos externos y 10 reservas internas, con 50 registros y 45 parejas. La importancia ampliada comprende 11 procedimientos; la latencia, 15.

## Conclusiones y procedencia

El MCS al 90 % retiene TimesNet · N. El conjunto está condicionado al periodo de prueba, a las cinco inicializaciones y al protocolo de remuestreo. El menor MASE medio corresponde a TimesNet · N: 0,64292782.

Conteos, orden descriptivo, decisiones de contrastes, MCS y conclusiones textuales se generan mediante `scripts/narrativa_resultados.py` a partir de CSV y JSON. Las celdas Markdown identifican su función con `resultado_documental` y se regeneran al reconstruir el libro. Las fuentes y sus huellas están en `results/revision_documental_20261120/resumen_documental.json`.

## Alcance metodológico

TimesNet conserva desarrollo con lote 256 y prueba con lote 1; esa diferencia se declara en la evaluación. Los resultados mantienen alcance retrospectivo. La equiparación de presupuestos, la potencia objetivo, la priorización reproducible del Top 10, la revisión sistemática, la réplica del paper guía, la robustez por severidad, las ablaciones, la cobertura predictiva y el techo de desempeño requieren su evidencia específica. La corrección documental no los presenta como experimentos completados.

## Comprobaciones de completitud

- La selección contiene 15 procedimientos aprendidos y 75 realizaciones, además de 2 referencias. Se conservan los diez modelos adicionales, tres baselines neuronales, ARX y el control numérico de TGForecaster.
- Curvas, observado–predicho, semana ilustrativa, horizontes, segmentos, residuos, calibración, variación y atípicos utilizan las configuraciones seleccionadas. Los grupos por horizonte se derivan del benchmark y verifican cobertura y ausencia de repeticiones.
- La precisión separa 5 pilotos externos de 10 reservas internas. Ajuste y parada tienen métricas en modo evaluación para catorce procedimientos; N-HiTS conserva sus curvas de pérdida. Son medidas con distintos alcances y unidades.
- La importancia ampliada cubre 11 procedimientos; la referencia de demanda/calendario se presenta por separado. La latencia cubre 15 procedimientos y 75 realizaciones. Los tres mecanismos documentales conservan sensibilidad al contenido.
- MASE es la métrica primaria; MAE y RMSE son complementarias. El margen es relativo, del 3 %. Los intervalos temporales, la dispersión entre semillas y las bandas visuales conservan interpretaciones distintas.
- Los capítulos 1 y 12 y las etiquetas de descarga se regeneran desde los resultados. Se eliminan párrafos repetidos dentro de una celda y se comprueba la cobertura de N-BEATS y TimesNet en el error por horizonte. Los CSV y las métricas del dashboard se contrastan con sus fuentes.
- El pipeline seleccionado y sus comprobaciones funcionales están documentados. La verificación física conservada corresponde a Windows con CPU; no demuestra ejecución en Linux o GPU.

## Pendientes metodológicos que la revisión documental no resuelve

| Aspecto | Evidencia disponible y trabajo pendiente |
|---|---|
| Preguntas e hipótesis | Existe una pregunta central y una hipótesis de mejora del texto. Falta articular al menos tres RQ con H0/H1, prueba, tamaño de efecto y criterio de decisión, como solicita E2 §1. Su formulación actual debe conservar alcance exploratorio. |
| Independencia de prueba y cronología | La exposición retrospectiva y el orden de registro están documentados. Un hash y la congelación actual no restauran independencia ni constituyen un preregistro anterior. |
| Potencia y presupuesto | Hay cinco semillas y dimensionamiento retrospectivo. No se acredita potencia 0,80 para todas las parejas; los presupuestos de selección son desiguales. |
| Revisión sistemática y priorización | Faltan cadenas y fechas originales de Scopus/WoS, conciliación completa, flujo PRISMA y revisión independiente. El denominador de comparadores contiene cinco estudios empíricos de demanda; parte de las referencias se caracteriza desde resúmenes. La priorización del Top 10 requiere su matriz reproducible. |
| Réplica del paper guía | Las adaptaciones arquitectónicas no constituyen por sí mismas una réplica bajo datos, partición y métrica originales. Falta esa evidencia y el criterio de éxito. |
| Búsqueda de arquitectura | E2 §14 aplica las 1000 configuraciones a modelos sin arquitectura predefinida. El espacio, las dos fases, los finalistas y sus análisis deben acreditarse para las familias a las que corresponda; no se exigen indiscriminadamente a todas las arquitecturas publicadas. |
| Robustez, ablación e incertidumbre | Desplazamientos y controles con pesos fijos no sustituyen una evaluación a tres severidades ni ablaciones por componentes. Faltan cobertura predictiva y techo de desempeño; las bandas entre semillas no los reemplazan. |
| Costo y atribución | Hay parámetros y latencia para todos; memoria pico y FLOPs no tienen cobertura completa. Entradas y presupuestos diferentes limitan atribuciones causales a arquitectura o texto. |
| Comparabilidad de TimesNet | Los artefactos conservan desarrollo con lote 256 y prueba con lote 1. El lanzador de futuros ajustes homogeneiza la evaluación por origen; eso no cambia las métricas de desarrollo ya guardadas. |

Las prioridades son cerrar la trazabilidad bibliográfica y formular las preguntas explícitas, definir una comparación futura con protocolo homogéneo y completar después los experimentos que requieren nueva evidencia. Esta revisión no ejecuta entrenamiento, inferencia de redes ni publicación.

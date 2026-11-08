# Fuentes y fundamento metodológico

| Fuente | Uso en el estudio | Alcance |
|---|---|---|
| XM, archivos numéricos y registro de variables | Demanda e información del SIN | Series retrospectivas con rezagos de disponibilidad supuestos |
| ERA5 vía Open-Meteo | Condiciones meteorológicas observadas | Reanálisis; no equivale a un pronóstico con emisión previa |
| Censo 2018 citado en el inventario de ciudades | Ponderación de observaciones urbanas | Pesos estáticos y fuente secundaria |
| Archivos de titulares y cobertura | Diagnóstico documental | Disponibilidad declarada o versión archivada verificada |
| Preregistro y configuración del repositorio | Horizonte, partición, métricas y control temporal | Versiones experimentales conservadas |

[Hyndman y Athanasopoulos, *Forecasting: Principles and Practice*, capítulo 5](https://otexts.com/fpp3/toolbox.html) fundamentan las referencias estacionales, los errores escalados y la validación temporal. MASE divide el error absoluto del pronóstico por la escala de una referencia estacional calculada en entrenamiento. La comparación entre particiones requiere considerar el cambio del nivel de demanda. La validación temporal evalúa observaciones posteriores a la historia de ajuste.

La síntesis de textos completos sobre demanda y noticias se presenta en el capítulo de literatura. Los enlaces identifican las versiones consultadas; las cifras reproducidas se atribuyen a sus tablas originales.

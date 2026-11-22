# Dashboard de demanda eléctrica del SIN

Datos y documentación sincronizados con las fuentes actuales del JBook el 21 de noviembre de 2026.

[Abrir el dashboard público](https://demanda-electrica-dashboard.onrender.com/index.html).

Aplicación estática con Contexto, EDA y Modelos. Presenta diez familias Top 10, tres baselines E1, ARX y el control numérico de TGForecaster: quince procedimientos, cinco semillas por procedimiento y dos referencias ingenuas. N-BEATS y TimesNet están incluidos; TimesNet usa inferencia de lote uno.

Los pronósticos corresponden a la evaluación histórica 2022–2025. El selector de fechas explora resultados guardados; no produce predicciones para fechas nuevas. No necesita pesos, GPU, credenciales ni servidor Python en producción. Los datos y Plotly se sirven desde el mismo sitio, sin servicios externos de analítica.

## Ejecución local

Desde la raíz del repositorio:

```console
python -m http.server 8068 --bind 127.0.0.1 --directory dashboard/site
```

Abrir `http://127.0.0.1:8068`. Para comprobar agregaciones con Node.js:

```console
cd dashboard
npm test
```

No hay dependencias JavaScript que instalar. `site/vendor/plotly.min.js` conserva su licencia MIT.

## Datos y métricas

`scripts/export_data.py` transforma los artefactos consolidados del proyecto en `site/data/`, sin entrenamiento ni inferencia de redes. Requiere el entorno científico y los archivos originales del repositorio. El directorio `site/` ya contiene los datos exportados y funciona de forma autónoma. `site/data/manifest.json` conserva SHA-256 de las fuentes. Los CSV descargables reproducen los contrastes y resultados consolidados.

MAE y RMSE se evalúan por semilla antes de promediar. Para una ventana seleccionada, RMSE es la media de las raíces del error cuadrático medio de cada semilla. MASE utiliza el denominador fijo 215617,90361589016 kWh. Las curvas están en GWh por intervalo; la banda p10–p90 representa dispersión entre semillas, no un intervalo de predicción. La última ventana se recorta al 31 de diciembre de 2025.

Los contrastes conservan la orientación A−B de los resultados originales y sus familias de ajuste por multiplicidad. Se distinguen diferencia, equivalencia y ausencia de evidencia. La comparación entre arquitecturas con entradas y presupuestos distintos no identifica por sí sola el efecto causal del texto.

El EDA incluye 30 variables numéricas del panel observado 2000–2025, con agregación diaria. El clima es reanálisis histórico. Las series exploratorias no constituyen información conocida del futuro. La cobertura del corpus representa la caché oficial y las ventanas efectivamente admitidas por disponibilidad declarada, separadas de la cobertura de publicaciones.

## Render

Crear **Static Site** desde este repositorio, rama `main`, directorio raíz `dashboard`, comando de construcción `echo Dashboard preparado` y directorio de publicación `site`. Configurar `SKIP_INSTALL_DEPS=true`. No requiere un servicio de pago. `render.yaml` documenta los mismos parámetros y restringe los cambios que activan despliegues a `dashboard/**`.

No regenerar datos durante el despliegue: la exportación se realiza explícitamente cuando se actualiza el benchmark. No modificar entrenamientos ni publicar el JBook al actualizar esta aplicación.

## Fichas y código de los modelos

Cada arquitectura tiene una ficha Markdown descargable con sus ecuaciones, los esquemas actuales y el código Python que muestra el JBook. Las inclusiones de código se expanden durante la exportación. Las interpretaciones de los esquemas y la fuente de precisión de desarrollo se obtienen de los archivos actuales del libro y del benchmark.

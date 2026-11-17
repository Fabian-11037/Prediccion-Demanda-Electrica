# Contexto del dashboard

Fecha: 17 de noviembre de 2026. Autores: Fabian José Manrique Morelos y Gloria Bastidas Rodriguez.

## Alcance

Aplicación pública de demanda eléctrica del SIN colombiano. Tres secciones: Contexto, EDA y Modelos. El EDA permite seleccionar periodos y 30 series numéricas, explorar ciclos y revisar la cobertura del corpus oficial. Modelos ofrece pronósticos guardados por fecha, benchmark, contrastes temporales, rangos, MCS, importancia, aprendizaje, sensibilidad documental, residuos y esquemas con enlaces a los pipelines del JBook.

Se conservan diez modelos principales, dos comparadores numéricos y las variantes N/N+O de TGForecaster: 13 procedimientos, 65 realizaciones y dos referencias ingenuas. PatchTST aparece una vez. Se utilizan 1461 orígenes diarios de 2022–2025 y 24 horizontes. No se ejecutan entrenamiento ni inferencia de redes en esta aplicación.

## Reproducción

La implementación es HTML/CSS/JavaScript sin compilación. Plotly 3.7.0 se distribuye localmente con licencia MIT. `scripts/export_data.py` transforma artefactos congelados en JSON y copia esquemas y CSV; no importa PyTorch. `site/data/manifest.json` conserva hashes de las fuentes. No regenerar resultados durante un despliegue.

Los cálculos de `site/core.js` reproducen MAE, RMSE y MASE desde pérdidas guardadas, agregando métricas después de evaluar cada semilla. No reemplazar RMSE por el error de la predicción media. La banda p10–p90 representa inicializaciones, no cobertura probabilística. Los horarios se presentan como horas locales de Colombia. Las fechas seleccionables se limitan al histórico disponible.

Los contrastes conservan el orden A−B del archivo fuente y Holm sobre 105 parejas. Los tres contrastes documentales conservan su ajuste Holm(3). No intercambiar esos valores p ni presentar ausencia de significación como equivalencia. El EDA no atribuye efectos causales; la comparación N/N+O incluye diferencias de configuraciones.

## Despliegue

Render Static Site: `demanda-electrica-dashboard`. Servicio `srv-dbuc2rec6qcc73da0mdg`. Repositorio público `Fabian-11037/Prediccion-Demanda-Electrica`, rama `main`, raíz `dashboard`, publicación `site`, construcción `echo Dashboard preparado`, variable `SKIP_INSTALL_DEPS=true`.

URL: https://demanda-electrica-dashboard.onrender.com/index.html. Primer despliegue `dep-dbuc2rmc6qcc73da0nc0`, commit `1a4fffe5fddf8adeef7773fd9fc488c1eafed494`. Render confirma Live y el acceso público devuelve HTTP 200. La publicación inicial demoró 2 min 12 s.

Se conecta como repositorio público sin ampliar permisos de GitHub. Los siguientes despliegues se activan manualmente desde Render; no asumir que un push publica automáticamente. `render.yaml` documenta una alternativa mediante Blueprint. No se necesita un servicio de pago ni claves API. No publicar el JBook al cambiar la aplicación.

Verificaciones de esta entrega: 86 pruebas del proyecto correctas y 5 omitidas; tres pruebas del dashboard correctas, incluyendo equivalencia de las métricas de los 13 procedimientos con el benchmark. Se revisan navegación, filtros, límites de fechas, diagramas, diagnósticos y diseño de escritorio/móvil. No hay errores de consola en la revisión local.

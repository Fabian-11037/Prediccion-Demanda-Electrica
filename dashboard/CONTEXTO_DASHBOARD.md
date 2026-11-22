# Contexto del dashboard

Fecha: 21 de noviembre de 2026. Autores: Fabian José Manrique Morelos y Gloria Bastidas Rodriguez.

## Alcance

Aplicación pública de demanda eléctrica del SIN colombiano. Tres secciones: Contexto, EDA y Modelos. El EDA permite seleccionar periodos y 30 series numéricas, explorar ciclos y revisar la cobertura del corpus oficial. Modelos ofrece pronósticos guardados por fecha, benchmark, contrastes temporales, rangos, MCS, importancia, aprendizaje, sensibilidad documental, residuos y esquemas con enlaces a los pipelines del JBook.

El Top 10 reúne PatchTST, N-HiTS, TFT, TCN, iTransformer, TaTS, fusión tardía, TGForecaster, N-BEATS y TimesNet. Con tres baselines E1, ARX y el control numérico de TGForecaster son 15 procedimientos, 75 realizaciones y dos referencias ingenuas. PatchTST aparece una vez. Se utilizan 1461 orígenes diarios de 2022–2025 y 24 horizontes. No se ejecutan entrenamiento ni inferencia de redes en esta aplicación.

## Reproducción

La implementación es HTML/CSS/JavaScript sin compilación. Plotly 3.7.0 se distribuye localmente con licencia MIT. `scripts/export_data.py` transforma artefactos congelados en JSON y copia esquemas y CSV; no importa PyTorch. `site/data/manifest.json` conserva hashes de las fuentes. No regenerar resultados durante un despliegue.

Los cálculos de `site/core.js` reproducen MAE, RMSE y MASE desde pérdidas guardadas, agregando métricas después de evaluar cada semilla. No reemplazar RMSE por el error de la predicción media. La banda p10–p90 representa inicializaciones, no cobertura probabilística. Los horarios se presentan como horas locales de Colombia. Las fechas seleccionables se limitan al histórico disponible.

Los contrastes conservan el orden A−B del archivo fuente y Holm sobre 136 parejas. Los tres contrastes documentales conservan su ajuste Holm(3). No intercambiar esos valores p ni presentar ausencia de significación como equivalencia. El EDA no atribuye efectos causales; la comparación N/N+O incluye diferencias de configuraciones.

## Despliegue

Render Static Site: `demanda-electrica-dashboard`. Servicio `srv-dbuc2rec6qcc73da0mdg`. Repositorio público `Fabian-11037/Prediccion-Demanda-Electrica`, rama `main`, raíz `dashboard`, publicación `site`, construcción `echo Dashboard preparado`, variable `SKIP_INSTALL_DEPS=true`.

URL: https://demanda-electrica-dashboard.onrender.com/index.html. Primer despliegue `dep-dbuc2rmc6qcc73da0nc0`, commit `1a4fffe5fddf8adeef7773fd9fc488c1eafed494`. Render confirma Live y el acceso público devuelve HTTP 200. La publicación inicial demoró 2 min 12 s.

Se conecta como repositorio público sin ampliar permisos de GitHub. Los siguientes despliegues se activan manualmente desde Render; no asumir que un push publica automáticamente. `render.yaml` documenta una alternativa mediante Blueprint. No se necesita un servicio de pago ni claves API. No publicar el JBook al cambiar la aplicación.

Verificaciones de la actualización local: 132 pruebas del proyecto correctas y 6 omitidas; cinco pruebas del dashboard correctas, incluyendo equivalencia de las métricas de los 15 procedimientos con el benchmark. Se revisan navegación, filtros, límites de fechas, diagramas, diagnósticos y diseño de escritorio/móvil. No hay errores de consola en la revisión local.


## Actualización local · 20 de noviembre de 2026

N-BEATS y TimesNet están disponibles en pronósticos, aprendizaje, benchmark, contrastes, rangos, MCS, residuos, importancia, latencia y esquemas. TimesNet utiliza exclusivamente la prueba con lote uno (MASE 0,642927818); N-BEATS, sus predicciones conservadas (MASE 0,953342617). La validación y parada de TimesNet mantienen los registros de desarrollo originales. El MCS al 90 % retiene TimesNet. Los contrastes entre semillas incluyen 105 parejas y los tres contrastes textuales mantienen Holm(3).

El selector distingue diez familias Top 10, tres baselines E1, ARX y el control numérico de TGForecaster. TimesNet aparece inicialmente en la vista de modelos. Los límites del gráfico de rangos utilizan el número efectivo de procedimientos. La aplicación carga los CSV y JSON exportados, sin pesos ni inferencia. La publicación manual utiliza el contenido de dashboard/site; el registro del despliegue conserva su commit y su estado.

## Sincronización con el JBook · 21 de noviembre de 2026

Las once tablas exportadas coinciden con las fuentes del benchmark y los 28 esquemas coinciden byte a byte con el JBook. Los 15 procedimientos conservan sus 75 realizaciones, curvas, contrastes, importancia, latencia y predicciones. Se añade la descarga de 14 fichas con ecuaciones y código Python expandido, sin directivas literalinclude pendientes. La navegación desde el pronóstico abre la ficha interna y evita enlaces a páginas de arquitecturas todavía no publicadas en GitHub Pages. Cada esquema muestra la interpretación de su fuente Markdown; el diseño distingue piloto externo de parada interna y presenta el MASE de desarrollo disponible.

Los conteos del resumen y las parejas temporales se obtienen de los resultados. El manifiesto incorpora las huellas de las tablas, protocolos, corpus congelado y fuentes de arquitectura del JBook. La actualización del dashboard no requiere publicar el libro, entrenar redes ni cambiar sus artefactos.

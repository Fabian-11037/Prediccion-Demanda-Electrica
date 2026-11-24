# Contexto del dashboard

Fecha de actualización local: 23 de noviembre de 2026. Autores: Fabian José Manrique Morelos y Gloria Bastidas Rodriguez.

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

El selector distingue diez familias Top 10, tres baselines E1, ARX y el control numérico de TGForecaster. M4ₙ aparece inicialmente en la vista de modelos con sus resultados incorporados. Los límites del gráfico de rangos utilizan el número efectivo de procedimientos. La aplicación carga los CSV y JSON exportados, sin pesos ni inferencia. La publicación manual utiliza el contenido de dashboard/site; el registro del despliegue conserva su commit y su estado.

## Sincronización con el JBook · 21 de noviembre de 2026

Las once tablas exportadas coinciden con las fuentes del benchmark y los 28 esquemas coinciden byte a byte con el JBook. Los 15 procedimientos conservan sus 75 realizaciones, curvas, contrastes, importancia, latencia y predicciones. Se añade la descarga de 14 fichas con ecuaciones y código Python expandido, sin directivas literalinclude pendientes. La navegación desde el pronóstico abre la ficha interna y evita enlaces a páginas de arquitecturas todavía no publicadas en GitHub Pages. Cada esquema muestra la interpretación de su fuente Markdown; el diseño distingue piloto externo de parada interna y presenta el MASE de desarrollo disponible.

Los conteos del resumen y las parejas temporales se obtienen de los resultados. El manifiesto incorpora las huellas de las tablas, protocolos, corpus congelado y fuentes de arquitectura del JBook. La actualización del dashboard no requiere publicar el libro, entrenar redes ni cambiar sus artefactos.

## Resumen al final de Modelos · revisión local

La sección Modelos termina con una tabla de los 15 procedimientos: periodo de validación, MASE de selección y MASE medio de prueba 2022–2025. Los datos se exportan desde metadatos_procedimientos.csv y benchmark.csv; el resumen y su orden se generan sin cifras fijadas manualmente. Incluye el enlace a 10_evaluacion.html del JBook. La publicación conjunta del 22 de noviembre incorpora este resumen en el dashboard y las fuentes actuales del JBook.


Publicación autorizada del 22 de noviembre: commit `bc2b8b382bcb8641cb27e3bca5f2269a7b45afa7`, despliegue Render `dep-dc17gabl550s73dm0gu0`. Los recursos públicos coinciden con la versión preparada. Evidencia: `results/publicacion/publicacion_20261122.json`.


## M4_N · teoría de la propuesta retrospectiva

Modelos → Procedimiento → M4ₙ → Arquitectura y pipeline presenta arquitectura, fundamentos, cuatro esquemas —global, memoria, E y periódico— y fórmulas locales y código preservado de preparación, entrenamiento y evaluación. La ficha procede de book/arquitecturas/m4_n.md y conserva sus ecuaciones y fuentes; el manifiesto m4_teoria_manifest.json registra hashes. Las métricas, curvas y predicciones actuales se incorporan según la revisión de resultados siguiente. Los quince procedimientos de referencia conservan sus artefactos originales. Actualización local, sin publicación ni inferencia.


## Revisión documental sin resultados · 23 de noviembre de 2026

La portada y el recorrido metodológico se alinean con la propuesta numérica escogida y definen M1 en su primera mención. Se añade acceso directo a M4_N desde Contexto, se justifica la prosa y se conserva el formato matemático y los esquemas de la ficha. La teoría ya no depende de cargar el archivo de observados del benchmark. La nota sobre llenado superior al 100 % aparece únicamente al seleccionar esa variable. El esquema del experto periódico se sincroniza con el JBook y su código de generación. La revisión no interpreta ni modifica resultados, datos o predicciones; no publica Render o Pages.


M4ₙ forma parte del selector Procedimiento del resto de modelos, en el grupo Propuesta. Su ficha se abre en Arquitectura y pipeline; no utiliza una pestaña de navegación independiente.


## Resultados de M4_N · revisión local

M4ₙ permanece en el selector común de Modelos y abre por defecto los pronósticos conservados. Se incorporan diez semillas 101–110, 1.461 orígenes diarios, curvas de entrenamiento y parada interna, observado frente a predicho, errores por horizonte y año, y comparación interactiva con las diez arquitecturas Top 10, MLP, LSTM, CNN1D y ARX. Las figuras incluyen interpretaciones procedentes del JBook. El resumen presenta el MASE de parada interna de M4 por separado de la validación externa de selección del benchmark.

MASE de M4_N=0,5770369087419158; MAE=0,12441948857192563 GWh y RMSE=0,1660419417545531 GWh. La ganancia frente a TimesNet es 10,25 %, con IC95 % puntual [8,90; 11,60] %. Se integran los 60 contrastes conservados del Top 10: bootstrap estacionario, DM–HLN y GW bilateral, para márgenes 0 y 3 %. Se preservan las cotas de Holm sobre 1.536 hipótesis, los 49.999 remuestreos y el bloque de 14 días. Frente a los cuatro baselines se muestran métricas y ganancias descriptivas; no se inventan contrastes ausentes. El benchmark de referencia mantiene sus quince procedimientos y sus diagnósticos originales: no extender su Friedman, MCS, latencia o importancia a M4 sin artefactos correspondientes.

dashboard/scripts/export_m4_results.py lee los NPZ e historias guardados, verifica fechas, observados, escalas y métricas, exporta m4_results.json y predictions/m4_n.json y sincroniza cinco figuras y las descargas del JBook. export_data.py invoca esta integración al terminar; m4_views.js implementa las vistas específicas dentro de la navegación común. m4_results_manifest.json registra hashes y el alcance de la exportación. Los resultados originales de los otros procedimientos se conservan. Ocho pruebas del dashboard verifican métricas, curvas y contrastes. Se revisan selectores, gráficos y enlaces en la vista local. No se ejecutan entrenamientos, inferencia neuronal, subida al repositorio ni publicación en Render o Pages.


## Inferencia real · 24 de noviembre de 2026

 se añade inferencia real local en dashboard/app.py y dashboard/runtime_inferencia.py, con Dash 3.4.0 y Flask 3.1.3. El selector común de los dieciséis procedimientos incluye Inferencia en vivo. Se usa CPU y checkpoints prefijados —1000 para comparadores y 101 para M4_N—, sin selección de semilla por prueba. TimesNet recibe un único origen. M4 recalcula PatchTST, E, periódico y M1, y solo materializa los valores del banco admisibles d−2…d−365. El servidor acepta origen, perturbación ±20 % de demanda y CSV opcional de 168 horas con las treinta columnas históricas del contrato, alineadas por disponibilidad. Sin CSV, la última historia local permite emitir 2026-01-01; no hay adquisición automática de datos recientes.

La interfaz muestra pronóstico de 24 horas, energía diaria, pico, tiempo de respuesta, semilla y hashes. Incluye sensibilidad a +1 % de demanda reciente y advertencia de extrapolación respecto de rangos de 2000–2021. M4 añade base M1, compuerta y episodios recuperados. Esta vista usa una inicialización y no sustituye el estimando de promedio de semillas del benchmark ni ofrece un intervalo calibrado. La API POST /api/inferencia no entrena ni guarda entradas del usuario. El servidor se limita a 127.0.0.1 y rutas estáticas del dashboard/libro.

Se comprueba paridad en un origen de los dieciséis procedimientos, con tolerancia relativa 0,001; el máximo observado es 0,00006222. TimesNet coincide en el punto comprobado y M4 difiere 2,087 kWh por redondeo de motor. Los dieciséis emiten 24 valores finitos para 2026-01-01 sin leer pronósticos precalculados. Se verifica invariancia de M4 al corromper observaciones posteriores al corte, respuesta a cambios de historia, inferencia desde CSV, sensibilidad local, rechazo de entradas inválidas y restricción de rutas. Las pruebas no cambian ni actualizan métricas del benchmark.

Inicio Windows: .inferenv/Scripts/python.exe dashboard/app.py o dashboard/iniciar_inferencia.cmd. Linux: .inferenv/bin/python dashboard/app.py. URL local http://127.0.0.1:8815/dashboard/site/index.html; panel directo /inferencia/. requirements-live.lock.txt fija las dependencias directas. Se requieren los checkpoints, fuentes y artefactos consolidados; DEMANDA_ARTEFACTOS admite una ruta alternativa al consolidado. Falta verificar un paquete autónomo y contenedor limpio antes de su distribución o despliegue. No tratar el servidor de desarrollo como configuración de producción. La implementación y código se documentan en el capítulo 12. La versión nueva no se publica en Render ni en Pages, y no se entrenan modelos. El usuario solicita apagar el PC al terminar.

## Significancia en la tabla resumen · 24 de noviembre de 2026

El resumen de desarrollo y prueba añade el p ajustado de M4_N frente a cada modelo del Top 10. Usa el máximo de los tres contrastes conservados (bootstrap estacionario, DM–HLN y GW bilateral), margen cero, y preserva las cotas Holm de la familia de 1.536 hipótesis. No recalcula ajustes sobre las diez filas visibles. La regla requiere que los tres valores sean menores que 0,05; no añade una prueba conjunta nueva. M4_N y comparadores sin contraste guardado muestran «—». La métrica primaria sigue siendo MASE. Cambio local, sin entrenar ni publicar.

## Evaluación complementaria de M4_N · 24 de noviembre de 2026

El dashboard incorpora m4_closure.json: familia complementaria102, contrastes de baselines/controles, MCS conM4, intervalos, importancia, fidelidad, estabilidad, latencia/RSS y paqueteONNX. Diagnósticos añade cuatro vistas. La tabla resumen conserva las cotas históricasTop10 y usa las nuevas para baselines; no sustituye familias. JBook y dashboard comparten figuras, datos y descargas. Sin entrenamiento, publicación ni apagado. Estado detallado y límites en CONTEXTO_PROYECTO.md §67.


## Auditoría integral del E3 · 24 de noviembre de2026

Resultados y familias coinciden con el JBook. La interfaz principal sigue siendo HTML/JS/Plotly con un panel Dash de inferencia. Para el cumplimiento literal de la pauta faltan la migración íntegra a Dash, una tabla de líderes ordenable/filtrable con todos los campos, incertidumbre en vivo y despliegue verificado. Las fuentes y la nueva matriz de remanentes están en REVISION_ENTREGABLE_3.md. No se entrena ni publica.


## Sincronización editorial con el JBook · 24 de noviembre de 2026

Se incorporan MASE nativo y ONNX del panel prefijado de cuatro fechas y 96 intervalos, diferencia y descarga por fecha. La interpretación de fidelidad distingue la consulta de memoria del pipeline completo y cuantifica la eliminación de dieciséis residuos. Los conteos proceden de CSV; las métricas, imágenes y paquetes se contrastan con el JBook. Exportación y figura de arquitectura se mantienen en el selector común de M4_N. Sin entrenar ni publicar. Registro en results/auditoria_e3_final_20261124/sincronizacion_dashboard.json.

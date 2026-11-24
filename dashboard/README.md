# Dashboard de demanda eléctrica del SIN

Documentación local revisada el 24 de noviembre de 2026. La ficha de la propuesta seleccionada comparte sus fuentes con el JBook; los resultados conservan sus artefactos de referencia.

[Abrir el dashboard público](https://demanda-electrica-dashboard.onrender.com/index.html).

Dashboard con Contexto, EDA y Modelos, y un servicio local Dash para inferencia CPU. Presenta diez familias Top 10, tres baselines E1, ARX y el control numérico de TGForecaster: quince procedimientos, cinco semillas por procedimiento y dos referencias ingenuas. N-BEATS y TimesNet están incluidos; TimesNet usa inferencia de lote uno.

Los pronósticos corresponden a la evaluación histórica 2022–2025. El selector de Pronósticos explora resultados guardados. Inferencia en vivo calcula nuevas trayectorias desde los pesos, con origen, historia CSV y perturbación de demanda. Los resultados históricos pueden servirse de forma estática; la inferencia necesita el servicio Python y los artefactos. Los datos y Plotly se sirven desde el mismo sitio, sin servicios externos de analítica.

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


## Propuesta retrospectiva M4_N

Modelos → Procedimiento → M4ₙ → Arquitectura y pipeline presenta fundamentos matemáticos, esquemas global, memoria, experto E y experto periódico, y el código preservado de preparación, ajuste y prueba. No altera el benchmark E2. La ficha y las fórmulas SVG funcionan sin servicios externos. Para regenerarla, ejecutar `python scripts/integrar_m4_teoria.py` desde la raíz con el entorno del JBook; el exportador principal también realiza esta sincronización al terminar. El generador no importa redes ni utiliza pesos. Las fuentes Python descargables conservan sus dependencias de entrenamiento y GPU; no constituyen un entorno autónomo.


M4ₙ forma parte del selector Procedimiento del resto de modelos, en el grupo Propuesta. Su ficha se abre en Arquitectura y pipeline; no utiliza una pestaña de navegación independiente.


## Resultados de M4_N

M4ₙ comparte el selector y las vistas de pronósticos, benchmark, diagnósticos y arquitectura. Se incluyen diez semillas, los mismos 1.461 días de prueba, curvas de aprendizaje, observado frente a predicción y perfiles de error por horizonte y año. El benchmark permite compararlo con el Top 10 y cuatro baselines; muestra los contrastes guardados del Top 10 para márgenes de 0 y 3 %, conservando su ajuste original por multiplicidad. La parada interna de 2021 se identifica en el resumen de desarrollo.

Para actualizar exclusivamente esta exportación desde las predicciones e historias conservadas, ejecutar `python dashboard/scripts/export_m4_results.py` desde la raíz. No entrena ni utiliza pesos. El sitio exportado funciona con su servidor estático habitual. `data/m4_results_manifest.json` conserva las huellas de las fuentes y `downloads/` contiene métricas, curvas y contrastes. No se publica al exportar.


## Inferencia en vivo

Ejecutar desde la raíz con el entorno CPU preparado:

```console
.inferenv\Scripts\python.exe dashboard\app.py
```

En Linux, usar `.inferenv/bin/python dashboard/app.py`. Abrir `http://127.0.0.1:8815/dashboard/site/index.html`, seleccionar el procedimiento y **Inferencia en vivo**. `dashboard/iniciar_inferencia.cmd` inicia el servidor en Windows. El panel también abre en `http://127.0.0.1:8815/inferencia/`.

`requirements-live.lock.txt` fija las dependencias directas; PyTorch CPU se instala desde su índice oficial. Se requieren el panel, los códigos y checkpoints de los resultados, los vectores oficiales y los artefactos consolidados. `DEMANDA_ARTEFACTOS` permite indicar la carpeta de estos últimos; el valor predeterminado es la carpeta de consolidado contigua al repositorio. La vista estática no incorpora estos pesos. La preparación de un paquete autónomo y su comprobación en entorno limpio siguen siendo necesarias para distribuir la inferencia a otro PC.

Los dieciséis procedimientos ejecutan pesos prefijados: semilla 1000 en los comparadores y 101 en M4_N. Se utilizan todas las variables de cada checkpoint y TimesNet recibe un solo origen por llamada. M4_N calcula sus tres expertos y M1, y recupera únicamente residuos disponibles con retraso de dos días. El escalado procede de los checkpoints. El corpus documental utiliza la disponibilidad declarada. El servicio no actualiza pesos ni datos.

Sin CSV, la última ventana permite emitir 2026-01-01. Un CSV propio debe contener `fecha` y las 30 variables del contrato en 168 horas consecutivas anteriores al origen, ya alineadas por disponibilidad. La plantilla se descarga en el panel. La fecha no activa descargas de XM ni fuentes meteorológicas. El calendario se calcula para el horizonte; el objetivo futuro no se utiliza como entrada. Se validan forma, continuidad, demanda positiva, valores finitos, límites de perturbación y máscaras compatibles.

Se muestran la trayectoria de 24 horas, energía diaria, pico, tiempo de respuesta, semilla y hashes. Una perturbación de +1 % en la demanda de las últimas 24 horas cuantifica sensibilidad local, manteniendo las demás entradas y el estado residual. M4_N añade compuerta, base M1 y episodios recuperados. El aviso de extrapolación compara las entradas con los rangos de 2000–2021. Los pronósticos se descargan en CSV. La vista utiliza una inicialización; no representa el promedio de semillas del benchmark ni un intervalo calibrado.

La API `POST /api/inferencia` recibe `model`, `origin`, `demand_pct`, `csv` opcional y `explain`. `GET /healthz` comprueba disponibilidad y `GET /api/plantilla.csv` entrega el esquema. Solo se sirven las rutas del dashboard y el HTML del libro. El servidor local escucha en 127.0.0.1. Para producción se requiere un servidor WSGI y revisar recursos; no se cambia la configuración ni se publica en Render.

Verificación: paridad de un origen conservado en los dieciséis procedimientos, dentro de tolerancia relativa 0,001, y dieciséis pronósticos para un origen sin resultados guardados. También se comprueban CSV, rechazo de entradas inválidas, respuesta a cambios de historia, exclusión de información posterior al corte y rutas de archivos. Evidencia en `results/revision_e3_integral_20261123/`.

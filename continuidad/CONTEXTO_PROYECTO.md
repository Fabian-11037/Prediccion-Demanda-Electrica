# Contexto del proyecto

Actualizado el 10 de noviembre de 2026. Idioma: español. Autores: Fabian José Manrique Morelos y Gloria Bastidas Rodriguez. Maestría en Estadística Aplicada, Universidad del Norte.

## 0. Redacción y alcance

El Jupyter Book se redacta como un artículo de maestría sobre demanda eléctrica del SIN colombiano. Presentar procedimiento, modelos y resultados vigentes, sin comparaciones históricas innecesarias. Se documentan la exposición al conjunto de prueba y la cronología cuando afectan la validez del análisis. No mencionar otros proyectos del autor. Se permite citar Time-MMD, NSW-EPNews y publicaciones externas con atribución explícita a sus autores. Integrar las interpretaciones naturalmente después de cada figura. Mostrar el código analítico, de gráficas y pipelines; usar Markdown nativo para narrativa y evitar tablas que repitan gráficos. No publicar ni iniciar entrenamientos salvo solicitud expresa.

## 1. Objetivo
Proyecto final del curso de Deep Learning. Meta global: un **modelo original y novedoso que combine modelos de series de tiempo con NLP** (variables temporales + texto) para mejorar el pronóstico de la **demanda real de energía eléctrica del Sistema Interconectado Nacional (SIN) de Colombia**. Hay tres entregables encadenados; el conjunto de datos, la partición, las métricas y los márgenes de equivalencia que se fijen en el primero se heredan en los siguientes.

| Entregable | Qué pide (resumen) |
|---|---|
| **E1** Estado del arte, EDA y 3 baselines | EDA exhaustivo, extracción de 3 modelos base de la literatura, piloto, ≥5 semillas, benchmark, baseline e inferencia, Jupyter Book. La pauta exige tres modelos de Deep Learning; las referencias clásicas se reportan como contexto. |
| **E2** Benchmark de 10 modelos | Top-10 (≥4 familias) + baselines, replicar un paper guía e identificar su limitación, preregistro, auditoría de fuga, búsqueda de 1000 arquitecturas en dos fases, ablación (≥3 componentes), robustez (≥3 niveles), potencia, Friedman/Holm/MCS/DM. |
| **E3** Modelo original | Mecanismo, pérdida o entrenamiento nuevos; implementación propia; superioridad significativa frente al Top-10 y baselines fuertes en el conjunto principal y en ≥2 de 3 conjuntos adicionales de naturaleza distinta (elegidos antes de ver resultados); demostración formal de una propiedad; dashboard público; informe; sustentación. |

Las pautas oficiales están en el proyecto de claude.ai «DL Project» (`Entregable{1,2,3}DeepLearningProject.pdf`).

## 2. Problema y parámetros

Objetivo: demanda real horaria nacional del SIN, en kWh por intervalo. Pronóstico directo de 24 horas desde las 00:00 de Colombia, con 168 horas de historia. Entradas ejecutadas: demanda histórica y 23 variables de calendario; ARX utiliza 48 rezagos de demanda. Clima, hidrología, precio, ISE y titulares están disponibles para caracterización y extensiones, sin atribuirles resultados de modelos que no los utilizan.

MASE es la métrica primaria, con denominador fijo de diferencias semanales de 2000–2017: 215617,90361589016 kWh. MAE, RMSE, sMAPE, R², sesgo y perfiles horarios complementan el informe. Margen relativo de interés: 3 %; α=0,05 y potencia objetivo 0,80. Cinco semillas finales por familia no acreditan esa potencia. La unidad temporal es el origen diario con sus 24 errores; las semillas no vuelven independientes los días compartidos.

## 3. Datos (carpeta `data/`)
Inventario y detalle en `data/raw/README.md` y `data/external/README.md`.

| Bloque | Cobertura | Estado |
|---|---|---|
| Demanda real, demanda comercial, precio de bolsa (XM) | horario 2000-01-01 → 2025-12-31 23:00, 227 928 filas, sin huecos ni NaN | incluido (`data/raw/numeric/hourly`) |
| Clima observado: temperatura, humedad, precipitación en 8 ciudades (ERA5 vía Open-Meteo) | igual | incluido; **reanálisis, no pronóstico** |
| Habitantes por ciudad (censo 2018 del DANE ajustado por omisión censal) | estático | incluido en `data/raw/numeric/ciudades.csv`; cifras de **fuente secundaria** (Portafolio, 17-oct-2019, citando al DANE); sustituir por el archivo oficial |
| Hidrología diaria (llenado, reserva útil, capacidad útil, aportes) | 2000-01-01 → 2025-12-31 | incluido |
| Titulares de prensa (7 medios) y boletines IDEAM | 568 295 titulares, 2004-2025 | incluido (`data/raw/noticias/titulares.parquet`) |
| Corpus integrado de prensa y documentos oficiales | 575473 filas; 7907 documentos oficiales únicos | data/processed/corpus_texto.parquet; disponibilidad y alcance en §13 |
| Festivos de Colombia | 2000-2035 | generado (`data/external/festivos_colombia.csv`) |
| ISE (DANE), actividad económica | Base2015:2005–2026-08; ediciones antiguas desde2000 | Descargado,168anexos y102ediciones normalizadas; ver§17 |
| Pronósticos meteorológicos | JMA2018 y GFS2021, ocho ciudades, hasta2026-11-09 | Descargado; temperatura/humedad/precipitación y disponibilidad estimada; ver§17 |

La extensión numérica2026 se descarga por separado y no modifica el panel congelado; ver§17.

### Tabla consolidada
`python scripts/construir_panel.py` genera en `data/processed/`: `panel_horario.parquet` (índice = inicio del intervalo, hora de Colombia), `feature_registry.json` (rol, clase de disponibilidad, rezago y fuente de cada columna), `calidad_datos.json` y `cobertura_titulares_{estricto, declarado}.csv`.

Prefijos de columnas:
- `y_demanda_real_kwh`: objetivo; solo se lee como etiqueta del horizonte.
- `x_*`: entradas **as-of** (solo pasado, con rezago de publicación ya aplicado): demanda real rezagada, precio de bolsa, clima observado por ciudad, clima ponderado por habitantes (`x_obs_*_ponderada`, promedio de las 8 ciudades con pesos del censo 2018; pesos estáticos, declarar que el censo es posterior a parte del periodo), hidrología.
- `cal_*`: calendario conocido a futuro (hora, día de la semana, festivos, víspera, puentes, Semana Santa, temporada navideña, codificaciones cíclicas).
- `aux_*`: solo auditoría; no usar como entrada.

## 4. Control de fuga de información (reglas que NO se deben romper)
1. **Clases de disponibilidad:** `known_future` (calendario, festivos), `past_only` (demanda, precio, clima observado, hidrología, ISE), `oracle_only` (clima ERA5 de la hora objetivo: prohibido como entrada), `excluded_input` (`demanda_comercial_kwh`: se deriva de la demanda real más pérdidas y se liquida después; redundante y con riesgo de fuga).
2. **Rezagos de disponibilidad** (supuestos conservadores en `src/demanda/config.py`, a declarar en el preregistro): demanda 1 h, precio de bolsa 24 h, clima observado 1 h, hidrología 24 h tras terminar el día, ISE fin de mes + 60 días, pronósticos meteorológicos emisión + margen 6 h + latencia 6 h. El panel ya trae estos rezagos aplicados: una ventana que termina en la fila `p-1` es usable en el origen `t0 = índice[p]`.
3. **Ventanas por origen** (`src/demanda/ventanas.py`): las columnas `x_*` nunca se leen en filas `>= p`; solo `cal_*` y pronósticos con emisión previa se leen en el horizonte; `y_*` solo como etiqueta. Prueba que lo verifica corrompiendo el futuro.
4. **Partición cronológica con purga** (`src/demanda/particion.py`): cada intervalo-objetivo queda completo dentro de una partición (purga = H). **El conjunto de prueba está congelado**: `origenes(..., 'test')` exige `allow_test=True` y solo debe invocarse en la corrida final.
5. **Escalas, imputación, vocabulario y climatología se ajustan solo con entrenamiento** (`AjustadorEscala`, `ClimatologiaTemperatura`; hay pruebas que alteran validación/prueba y verifican que no cambia nada).
6. **Texto:** el **escenario declarado es la versión oficial del dataset textual** y se aplica a documentos institucionales y prensa en todas las particiones. Fecha diaria sin hora: día siguiente 00:00 de Colombia; año de publicación sin mes: año siguiente. Conservar advertencias de versión histórica no verificada y excluir cuarentena/fechas inutilizables. El escenario estricto queda para sensibilidad y auditoría de disponibilidad; no supone una alternativa predictiva equivalente cuando falta entrenamiento histórico. La relevancia se aprende dentro del modelo, sin filtro por términos ni categorías manuales. Nunca concatenar URL, fechas de recuperación ni pruebas de procedencia al texto. El diseño de canales y las tres variantes comparables se especifican en §14.
7. **Clima:** el clima de `data/raw` es reanálisis. Como covariable del futuro solo valen pronósticos con emisión conocida (cobertura corta) o la climatología de entrenamiento. No interpolar pronósticos hacia atrás con ERA5.
8. Los preregistros se escriben **antes** de buscar, hacer piloto o ejecutar, y se versionan.
9. Mismo esfuerzo de búsqueda para todos los modelos. «No significativo» no es equivalencia (usar TOST/ROPE). Reportar p exactos, ajustados y sin ajustar, IC y tamaño de efecto.
10. Antes de afirmar algo sobre literatura, verificarlo en el texto completo. No inventar referencias ni DOI. Las cifras de los textos deben salir de celdas ejecutadas.

## 5. Diseño y modelos seleccionados

Forward chaining propio, sin timeseries-cv: años externos 2018–2021, historia expansiva y último 10 % de orígenes previos para parada interna. Purga de etiquetas de 24 horas entre tramos. Prueba: 2022–2025, con 1461 orígenes diarios y sin faltantes ni duplicados. El ajuste final utiliza el historial anterior a 2022 y su reserva interna. Los escaladores se ajustan sólo con el tramo de optimización; la escala MASE es un denominador de evaluación separado del preprocesamiento.

Configuraciones consolidadas: MLP de dos capas de 128 unidades, 111896 parámetros; LSTM de una capa de 48, 28632 parámetros; CNN1D de una convolución con 16 filtros y kernel cinco, 47464 parámetros; ARX lasso de 48 rezagos, 14424 parámetros. Las secuenciales concatenan calendario histórico y demanda; el calendario futuro se concatena a la representación antes de una cabeza lineal, con salida residual alrededor de uno en escala de nivel. CNN aplica promedio local de factor dos. MLP y ARX estandarizan además su entrada tabular con entrenamiento.

Candidatos: MLP: 20, ARX: 20, LSTM: 24, CNN: 26. Filtro en 2019 y semilla 42, máximo30 épocas. Confirmación con semillas 43/44 y cuatro años externos, máximo60 épocas. Selección por menor MASE de confirmación. Piloto: semillas 100–104 en los cuatro pliegues de la configuración seleccionada. Final: semillas 1000–1004 por familia. El registro de artefactos incluye384 ajustes:90 filtro,144 confirmación,120 piloto y30 final; el benchmark usa20 realizaciones seleccionadas. El presupuesto desigual limita comparaciones de familias.

Entrenador: AdamW, precisión completa, determinismo estricto, clip 1; mínimo10 épocas y paciencia6. Mejora significativa max(1e-6,0,002×pérdida de referencia), con restauración del mínimo absoluto. Plateau reduce lr a la mitad tras dos épocas sin mejora. Huberδ=0,05 es parámetro de pérdida, distinto del margen relativo3 %. Las opciones exactas están en el registro de modelos y el código congelado; no modificarlas al reconstruir el libro.

## 6. Resultados y lectura estadística

| Modelo | MASE medio | MAE (kWh) | RMSE (kWh) | R² medio |
|---|---:|---:|---:|---:|
| MLP | 0.752022 | 162,149.39 | 209,153.21 | 0.967130 |
| ARX | 0.851675 | 183,636.41 | 245,952.61 | 0.954621 |
| CNN1D | 0.857347 | 184,859.31 | 238,815.15 | 0.957063 |
| LSTM | 1.000502 | 215,726.19 | 270,164.36 | 0.945162 |
| Ingenuo semanal | 1.354906 | 292,142.09 | 444,631.87 | 0.851736 |
| Ingenuo diario | 2.076725 | 447,778.99 | 659,593.46 | 0.673723 |

MLP es el baseline empírico del benchmark y reduce el error un44,5 % frente al ingenuo semanal. ARX y CNN1D tienen errores agregados próximos. Se calculan medias, DE e intervalos t95 % entre cinco inicializaciones, condicionados a las mismas fechas. No confundir esos intervalos con incertidumbre temporal ni la banda de semillas con un intervalo predictivo calibrado.

La evaluación temporal promedia pérdidas individuales entre semillas antes de remuestrear; no evalúa una predicción ensemble. Bootstrap circular estratificado por año,10000 réplicas, bloques 7/14/28 días e índices apareados. DM–HLN sobre diferencias diarias por hora, HAC Bartlett14rezagos, h=1origen diario no solapado; Holm por separado entre72 contrastes neuronales y72contraARX. Residuos de semilla 1000: Ljung–Box 7/14/28, ARCH 7, BDSdimensión2 sobre últimos1000 días, Holm entre20diagnósticos. BDS no aísla no linealidad si existe memoria.

Los resultados y contrastes tienen alcance exploratorio y descriptivo: no afirmar superioridad confirmatoria, equivalencia ni potencia alcanzada. La revisión de fuentes y los supuestos de disponibilidad limitan la validez histórica. No presentar el solapamiento de IC como prueba de igualdad.

## 7. Pendientes y limitaciones de fuentes

### Datos que faltan
| Dato | Por qué falta | Qué hacer |
|---|---|---|
| ISE (DANE), actividad económica | Descargado;102ediciones normalizadas | Completar parser de66formatos si se requieren; no mezclar bases ni vintages |
| Pronósticos meteorológicos | Descargados JMA/GFS con adelantos distintos verificados | Emisión exacta y publicación no certificadas; tratar disponibilidad proxy y subperiodo |
| Datos de2026 | Extensión numérica descargada por separado | Texto exhaustivo pendiente; no llamar prospectivo a un periodo ya observado |
| Habitantes oficiales DANE | Descargadas proyecciones/retroproyecciones de ocho ciudades2000–2026 | Decidir uso y vintage sin reemplazar los pesos congelados |
| Pronóstico oficial XM | Histórico SIMEM descargado2022–2026 | Publicación posterior al objetivo; falta emisión original, continúa excluido de entradas |
| Horas reales de publicación (demanda, precio, hidrología) | La fuente no las certifica | Los rezagos de `config.py` son supuestos; documentarlos o medirlos |
| PREREGISTRO.md y adendas | Conservar documentos versionados | Preservar identidad y trazabilidad |



Completar trazabilidad de búsqueda bibliográfica, cadenas institucionales con fecha, conciliación entre bases y revisión independiente. Los recuentos dirigidos de comparadores no equivalen a prevalencia en toda la literatura. Evaluar extensiones con covariables y texto mediante ablaciones, mismos orígenes y presupuesto comparable; no alterar modelos ni resultados guardados durante una actualización documental. La latencia CPU está medida y su protocolo se documenta en §11. Conservar el margen y la métrica definidos.

### Limitaciones conocidas
- **Cobertura de texto desigual:** 2008-2019 es casi solo El Colombiano (sitemap, fecha declarada sin hora); desde 2020 hay otras fuentes con volúmenes muy distintos por año (Semana: ≈17.6 mil en 2022 y ≈0.1 mil en 2025); los boletines IDEAM suben en 2024-2025. El cambio de composición de fuentes entre entrenamiento y prueba es un riesgo de sesgo de distribución, no de fuga.
- El escenario estricto de texto (62 572 titulares) empieza en 2021-04-25.
- Las horas de publicación históricas de demanda, precio e hidrología no están certificadas; los rezagos son supuestos. XM puede revisar cifras históricas.
- El ISE se revisa; una serie descargada hoy tiene sesgo de *vintage*.
- ERA5 es reanálisis y no representa lo que un operador habría tenido como pronóstico.

## 8. Artefactos y procedencia

Commit de modelos: d1ae9a4e09dd1dead4af1329f16aa3e8d0f2c3d3. Descarga local: results/repositorio/d1ae9a4e09dd1dead4af1329f16aa3e8d0f2c3d3. IMPORTACION.json conserva2404 archivos y177897923 bytes. Se descargaron pesos, escaladores, curvas, predicciones, identidades, métricas, código yplanes. El registro results/consolidado/modelos.json identifica cinco checkpoints por familia; el índice conserva2304 artefactos con SHA-256 de384 ajustes. Preservar archivos y rutas originales como evidencia técnica; no editar pesos ni metadatos para modificar resultados.

Fuentes: src/demanda/modelos_base.py, reentrenamiento_v3.py yvariantes_v4.py dentro de la instantánea; configuración de datos, particion.py ymetricas.py completan el pipeline. El checkpoint LSTM usa la fábrica variantes_v4; CNN, DatosFold yEntrenador usan reentrenamiento_v3. No confundir módulos del checkout activo con las fuentes de la instantánea. El pipeline del libro muestra las implementaciones y una función integrada portable que permite CPU o GPU, con prueba protegida por acceso explícito. No se invoca durante la construcción del informe.

Hardware registrado: PyTorch 2.9.1+rocm6.4, GPU AMD Radeon Graphics,12868124672 bytes. Las rutas del informe usan pathlib y soportan Windows/Linux; cada plataforma requiere PyTorch compatible con CPU, CUDA oROCm. No afirmar ejecución verificada en Linux por compilar rutas en Windows. Los tiempos de ajuste incluyen evaluación registrada y no miden latencia; ésta se mide por separado en CPU.

## 9. Jupyter Book

Generador: scripts/crear_jbook.py. scripts/modelos_actualizados_jbook.py construye los capítulos de modelos y evaluación; scripts/correcciones_e1_jbook.py es la capa final de conciliación estadística y documental. El registro activo está en results/jbook/modelos_actualizados.json. Los resultados derivados actuales se escriben exclusivamente en results/jbook/modelos_actuales. La reconstrucción no modifica configuraciones, entrena ni genera pronósticos neuronales.

Capítulos: problema/datos, calidad, demanda, estacionalidad, dependencia, covariables, texto, literatura, diseño, modelos, evaluación, síntesis, reproducibilidad. book/10_pipeline.md incorpora clases yfunciones de datos, partición, arquitecturas, optimizador, parada, predicción, métricas ypersistencia. Los cuadernos contienen el código para curvas, benchmark, perfiles, segmentos, incertidumbre ydiagnósticos. La tabla complementa la figura en lugar de repetir métricas visualizadas. La ventana ilustrativa es1–7julio2023 ylas curvas usan semilla 1000; no atribuirles representatividad universal.

EDA numérico: exclusivamente desarrollo2000–2021. Diccionario84 campos:59panel,18 titulares,7ciudades. Verifica rangos, finitud, coherencia hidráulica, distancia de malla ypesos poblacionales; no corrige fuentes. Rupturas retrospectivas ycentrado climático descriptivo no se usan como transformaciones de entrenamiento. Los diagnósticos de diferencias de entrenamiento no son residuos predictivos.

Diagnóstico léxico del archivo original:484289 titulares elegibles de desarrollo; cobertura, duplicados, longitud, bigramas, diversidad léxica ytransferencia. TF-IDF máximo6000 términos yNMF6 componentes, ajustados en muestra determinista de30000 titulares anteriores a2018; proyección con representación congelada. Sensibilidad de inicialización con semilla 11038 yemparejamiento óptimo. No clasificar temas manualmente ni afirmar utilidad predictiva a partir del EDA. Literatura:27estudios destacados,15 textos completos y12 abstracts; ochoabstracts IEEEde contexto; registro complementario de20 textos completos con frecuencias por denominadores16/14/19, separado del corpus empírico de demanda de cinco estudios. Scripts/literatura_integral_jbook.py y literature/analisis contienen extracción, código yhuellas.

Reconstrucción parcial: python scripts/crear_jbook.py 08_literatura 09_validacion 10_modelos 10_evaluacion 11_sintesis 12_reproducibilidad. Construir con jupyter-book build book --warningiserror --keep-going y verificar con python scripts/verificar_jbook.py. El verificador revisa celdas, figuras, diccionario, enlaces y copia HTML a docs; no publica. Entorno documental:.bookenv, requirements-book.lock.txt. Antes de cualquier commit ejecutar pytest -q tests. No publicar salvo solicitud expresa.

## 10. Estado de esta actualización

Se descargaron los artefactos completos y se reconstruyeron las métricas desde predicciones. El JBook presenta únicamente la descripción de los modelos, el procedimiento ylos resultados actuales, con código visible einterpretaciones integradas. Se eliminan referencias narrativas aotras ejecuciones; los identificadores técnicos originales se conservan para trazabilidad. HTML local en docs. No se ejecuta entrenamiento ni se publica el sitio durante esta actualización.


Construcción local completada: cero errores, advertencias y enlaces locales rotos. Sin publicación.


## 11. Correcciones del E1 y evidencia estadística

La última capa del generador es scripts/correcciones_e1_jbook.py. Se ejecuta después de modelos_actualizados_jbook.py y concilia todo el contenido. Los archivos canónicos son book/09_validacion.ipynb y book/10_modelos.ipynb. La evaluación temporal es §10.9 y el pipeline §10.10; el índice mantiene esos anexos bajo modelos. No reintroducir nombres ni descripciones de escalado global en la documentación vigente.

EDA: asimetría y exceso de curtosis, logaritmo y Box-Cox con lambda ajustada en 2000–2017 y aplicada a validación; sensibilidad descriptiva de anomalías; cuatro escenarios de ADF/KPSS y conclusión observada: nivel no estacionario, diferencia diaria compatible con estacionariedad bajo las pruebas usadas. Desplazamiento: contraste de diferencias de CDF en 19 cuantiles del desarrollo agrupado, permutación de bloques de 28 días, 1999 permutaciones y Holm entre siete variables. El supuesto es intercambiabilidad aproximada de bloques bajo la nula; no tratar horas como independientes ni convertir un rechazo en evidencia causal.

Piloto: promediar primero los cuatro años dentro de cada semilla (cinco realizaciones por familia). Mostrar media, DE, CV y detalle por año. Dimensionamiento normal pareado: R=25–45 según contraste; hasta 60 con Bonferroni conservador. MDD con cinco semillas: aproximadamente 6,68–8,96 % del MASE piloto de MLP, mayor que el margen del 3 %. El dimensionamiento es retrospectivo y no acredita haber determinado las cinco semillas ejecutadas. Mantener esta limitación sin iniciar entrenamientos adicionales. Se muestran las fechas de commits de planes y los cierres de selección existentes, sin inventar fechas ni reescribir protocolos.

Inferencia condicional por semillas: t pareada, Shapiro–Wilk, Q–Q, Wilcoxon exacto, bootstrap exhaustivo de 5^5 remuestreos, Cohen dz y Holm. Las t ajustadas producen p 0,000582–0,006434; Wilcoxon ajustado es 0,1875 en las tres parejas. No rechazar normalidad con cinco diferencias no la acredita. Friedman exacto enumera 6^5=7776 permutaciones: Q=10, p=0,000771605. Nemenyi aproximado distingue MLP–LSTM con p=0,00446; las otras parejas dan p=0,25378. No extrapolar estas pruebas a independencia temporal ni entre datasets.

TOST: margen relativo único ±3 % para θ_AB=(μ_A−μ_B)/μ_MLP. Dos t unilaterales sobre d±0,03·MLP incorporan incertidumbre de la referencia; IC90 % pareados de Fieller. CNN1D–ARX tiene p ajustado=0,858967: no se establece equivalencia. El umbral práctico es MASE<0,72946128, que equivale a MAE<157284,91 kWh con escala fija; cruzarlo no demuestra por sí solo superioridad estadística. Las reducciones MAE/RMSE frente a mejor y peor red están en reducciones_relativas.csv.

El análisis temporal incluye 14 contrastes globales: tres entre redes, tres frente a ARX y ocho frente a ingenuos. DM–HLN por hora conserva 72+72+192 contrastes, con Holm separado por familia. Bootstrap centrado y pareado de 10000 réplicas en bloques de 14 días e intervalos temporales con sensibilidad 7/28. Los CSV completos y tablas numéricas están enlazados desde el libro mediante book/_static/resultados y se exportan automáticamente después de ejecutar los cuadernos.

Atípicos de evaluación: límites de Tukey sobre log(y/y_lag168) ajustados en 2000–2017. Marcan 2520 horas (aprox.7,19 %). En MLP concentran el 8,63 % del error absoluto; excluirlas del cálculo reduciría MAE un 1,56 %. No se excluyen del benchmark ni se cambian predicciones. El hash canónico de orígenes UTC y etiquetas float64 little-endian es 595f00dd186280542cfd7fed9a580ebe1b19c8cc4e0e742ef12e2e6d2eb0f49a. Está visible en el texto y definido en huella_prueba.json; identificar contenido no acredita por sí solo cuándo se congeló.

Latencia: scripts/medir_inferencia.py carga los 20 checkpoints con weights_only=True y strict=True, sin optimizador, ajuste ni etiquetas. Entorno aislado .inferenv, PyTorch 2.9.1+cpu, un hilo, batch 1, 20 calentamientos y 200 mediciones por checkpoint. Medianas aproximadas: ARX 0,0209ms, MLP 0,0482ms, CNN1D 0,0656ms, LSTM 0,2858ms por origen de 24 salidas. Sólo forward con entradas sintéticas finitas; no incluye carga, preparación, transferencia ni escritura. Los valores p95 por familia son medianas de los p95 de sus checkpoints. No comparar directamente con tiempos de ajuste GPU. Guardar entorno_inferencia.json y latencia_inferencia.csv; el código completo está visible en el JBook.

Literatura: se agregan textos completos de Kathirgamanathan et al. (2022), DOI 10.1049/tje2.12132, y Laitsos et al. (2023), DOI 10.3390/math11132912. Se verificaron arquitectura, métricas, países, datos y límites; no se equipara automáticamente su horizonte con salida directa24. Corpus empírico de demanda: LSTM 4/5, MLP 3/5, CNN1D 2/5, TCN 2/5, CNN-LSTM 1/5, N-BEATS 1/5 y TFT 1/5. Los dos trabajos de Bai son publicaciones de una línea compartida, no replicaciones independientes entre sistemas. Time-MMD y estudios de precio son contexto de otros autores, fuera del denominador de demanda. Se explicitan elegibilidad, ventana 2018–8noviembre 2026, búsqueda pública realizada, brecha circunscrita al corpus e hipótesis de contribución documental al SIN. Las cadenas y fechas originales de Scopus/WoS no se recuperaron; sigue pendiente reconciliación entre bases y doble evaluación independiente. Los 12 abstracts son contexto, no evidencia de comparadores de texto completo.

El test de higiene permite referencias a publicaciones de otros autores y sigue prohibiendo referencias a proyectos del autor en la entrega. Su ámbito son archivos de entrega propios, excluyendo dependencias y archivos de terceros. Se verificaron37pruebas, con 5 omitidas por dependencias opcionales; el código activo de datos y entrenamiento no se modificó. Las advertencias de esas pruebas proceden de APIs de pandas en módulos existentes. La reconstrucción documental no entrena modelos ni publica. Mantener las interpretaciones junto a figuras, tablas complementarias y código analítico visible.

Verificación final: 13 cuadernos, 75 celdas analíticas y 42 figuras; cero errores, advertencias del libro y enlaces locales rotos. Los CSV y el hash de prueba están publicados sólo en el HTML local. Sin entrenamiento ni despliegue.


## 12. Revisión global de coherencia de lenguaje y contenido

Se revisaron los trece cuadernos, introducción, fuentes, diccionario y pipeline. La capa final es scripts/consistencia_jbook.py, invocada después de correcciones_e1_jbook.py; armonizar_archivos se ejecuta después de escribir las páginas Markdown. Se homogeneizaron las unidades de demanda como energía por intervalo horario, los nombres geográficos, la lectura del calendario y el uso de «resumen» en español, preservando títulos, enlaces e identificadores originales. Los p numéricamente subdesbordados se muestran como <1e-300. Las secciones de alcance y discusión se consolidan en §10.8.

El margen confirmado es relativo (δ=3 %), no un número absoluto de MASE fijado con prueba. La planificación normal utiliza DE de diferencias divididas por la media piloto de MLP; conserva R25–45 y hasta60 con Bonferroni. TOST usa el mismo parámetro relativo θ_AB=(μ_A−μ_B)/μ_MLP y dos contrastes unilaterales de combinaciones lineales pareadas. Fieller incorpora la covarianza con la referencia y requiere normalidad bivariada; con cinco semillas ese supuesto no se acredita por falta de rechazo de Shapiro. La implementación verifica coherencia entre IC90 % individual y TOST sin ajuste, y aplica Holm por familia. Ninguna pareja demuestra equivalencia; CNN1D–ARX tiene p ajustado0,858967. No se alteraron las cinco semillas, predicciones ni pesos.

El plan de LSTM de12:15:50 se registró después de finalizar los otros modelos a12:02:06. Documentar que esos resultados de prueba estaban disponibles y que el bloque2022–2025 había sido consultado. La cronología por sí sola no demuestra qué métricas influyeron en cada decisión, pero impide asumir una comparación confirmatoria independiente. El orden de dimensionamiento no cumple §4.1 y la potencia para3 % sigue sin acreditarse. No reescribir protocolos originales ni ocultar esta limitación mediante estilo editorial.

La revisión bibliográfica continúa siendo dirigida. Las secciones8.6.1–8.6.4 desarrollan MLP, LSTM, CNN1D y comparación crítica de robustez/escalabilidad. Faltan cadenas originales y fechas de Scopus/WoS, un flujo PRISMA conciliado y evaluación independiente por segundo revisor. Time-MMD y NSW-EPNews se citan como publicaciones externas. La irradiancia de RSAM es meteorológica: se corrigió G10.other_electricity_variable=0 y se actualizó su manifiesto; no se alteraron resultados de estudios ni fuentes numéricas de modelos.

Los contratos de coherencia se verifican en tests/test_consistencia_jbook.py: cinco semillas por familia, margen relativo, referencia MLP común, latencia CPU registrada, numeración canónica y presentación de p. La revisión se entrega en REVISION_CONSISTENCIA_JBOOK.md. El estado de coherencia interna no equivale a cumplimiento integral del E1. No publicar ni iniciar entrenamiento adicional sin solicitud expresa.

Cierre de la revisión global: trece cuadernos, introducción, fuentes, diccionario y pipeline revisados; 75 celdas analíticas, 42 figuras, cero errores, advertencias del libro y enlaces locales rotos. Evidencia: results/jbook/revision_consistencia.json y REVISION_CONSISTENCIA_JBOOK.md. Sin nuevos entrenamientos ni publicación.

En ADF, el valor cero del software corresponde al límite de su aproximación de MacKinnon y se reporta como p<0,001; no se interpreta como probabilidad exactamente nula ni como una cota de 1e-300.

Validación final: 43 pruebas correctas y 5 omitidas; las advertencias de pruebas corresponden a APIs de pandas de módulos existentes. El JBook tiene cero errores, advertencias de construcción y enlaces locales rotos. Se conservan cinco semillas por familia, sin entrenamiento ni publicación adicionales.

## 13. Corpus oficial y disponibilidad temporal · 9 de noviembre de 2026

Adquisición local, sin entrenamiento ni publicación. Fuente operativa: results/documentos_oficiales/INFORME_COBERTURA.md y auditoria_integracion.json. Se descargaron 7219 entradas de 7223 del inventario; 4 fallos HTTP permanecen documentados. Los 47 paquetes contienen 770 PDF individuales. El conjunto oficial conserva 7907 versiones únicas; 729 titulares se enriquecen y el corpus consolidado tiene 575473 filas, con ids únicos.

| fuente | documentos | con_cuerpo | con_texto_parcial | cuarentena | habilitados_declarado_2000_2025 | habilitados_estricto_2000_2025 |
|---|---|---|---|---|---|---|
| CREG | 909 | 748 | 470 | 183 | 726 | 252 |
| IDEAM | 1857 | 1855 | 22 | 47 | 1809 | 878 |
| MinEnergia | 2722 | 2722 | 0 | 0 | 2721 | 0 |
| UPME | 207 | 207 | 8 | 0 | 206 | 121 |
| XM | 2212 | 2148 | 646 | 9 | 2203 | 4 |

Tablas: data/processed/corpus_texto.parquet y data/raw/documentos_oficiales/documentos_oficiales.parquet. Manifiesto: data/processed/dataset_texto_oficial_manifest.json. Diccionario: data/processed/diccionario_corpus_texto.csv. Los originales numéricos y titulares, cuadernos, HTML y modelos del E1 no se modifican. El corpus ampliado se integra mediante src/demanda/corpus.py: cargar_corpus requiere un escenario explícito y documentos_en_origen selecciona origen−ventana <= disponibilidad <= origen, con zona de Colombia y ventanas auditadas de 24/72/168 horas.

La disponibilidad estricta de un PDF solo se adelanta a una captura cuando su SHA-1 Base32 coincide exactamente con el digest de una respuesta 200 en CDX. Se conservan el índice completo, timestamp, URL y SHA-256 local. No se declara una recuperación del payload archivado cuando la reproducción no fue accesible. Son 1255 documentos oficiales históricos utilizables por esta regla; la primera captura utilizable es 2004-06-30 06:09:07+00:00. El título editorial actual no queda verificado por ese digest: el cargador estricto utiliza el encabezado derivado del propio PDF. Una captura de un titular no verifica un cuerpo recuperado después. Sin prueba histórica, disponible_estricto es la recuperación en 2026 y el registro no entra antes en los orígenes de 2000–2025.

Antigüedad al verificar: 1016 documentos corroborados tienen un campo de fecha diaria y 11 presentan una captura en las primeras 72 horas desde su inicio; desfase mediano 1659.7 días. La auditoría editorial posterior excluye uno de esos 11 porque su fecha procede del repositorio XM y no de una edición verificable: quedan diez documentos oficiales recientes con disponibilidad estricta (nueve antes de 2022 y uno en prueba). Detalle: antiguedad_edicion_al_verificar.csv. Un boletín antiguo capturado después no constituye un pronóstico reciente para ese origen. Antes de evaluar señal anticipatoria, comprobar su vigencia y fecha de edición; las ventanas de captura por sí solas no acreditan frescura. No se aplicó un filtro temático ni se entrenó una compuerta.

El escenario declarado mantiene la advertencia de versión histórica no verificada. Fecha diaria sin hora: día siguiente 00:00 de Bogotá; año de publicación sin mes: año siguiente. Un año de referencia del archivo no acredita publicación. XM conserva el supuesto de fecha de creación/modificación del repositorio como tal, sin llamarlo hora oficial de publicación. Los miembros de ZIP usan su encabezado de edición, no la fecha del contenedor. Fechas desconocidas o incompatibles quedan en cuarentena, salvo evidencia de disponibilidad de esa misma versión por captura.

Cobertura de la ampliación oficial en ventana de 72 horas:
- Entrenamiento 2000–2017, declarado: 41.1% de orígenes con texto oficial en 72 horas.
- Validación 2018–2021, declarado: 98.1% de orígenes con texto oficial en 72 horas.
- Prueba 2022–2025, declarado: 98.0% de orígenes con texto oficial en 72 horas.
- Entrenamiento 2000–2017, estricto: 0.3% de orígenes con texto oficial en 72 horas.
- Validación 2018–2021, estricto: 8.9% de orígenes con texto oficial en 72 horas.
- Prueba 2022–2025, estricto: 30.2% de orígenes con texto oficial en 72 horas.

La cobertura mensual incluye meses con cero documentos y se guarda por fuente y escenario. cobertura_particiones.csv conserva los cortes vigentes y continuidad_fuente.csv identifica días sin texto. Estos gaps no se rellenan ni se interpretan automáticamente como publicaciones perdidas: la periodicidad de las series es distinta. Los orígenes y métricas numéricas permanecen congelados.

OCR local Windows.Media.Ocr, idioma es-MX, render máximo 1800 píxeles y hasta cuatro páginas por escaneo. El campo texto_alcance distingue cuerpo, fragmento, texto de diapositivas y titular; las páginas restantes de documentos largos permanecen en el PDF original. No se asignan temas ni se filtra por relevancia manual. Toda selección aprendida, vocabulario o ajuste posterior debe usar únicamente entrenamiento. La adquisición no inició entrenamiento, selección de arquitecturas ni evaluación predictiva. La posterior actualización documental y su publicación se registran en §14.

Reproducción: comandos en data/raw/documentos_oficiales/README.md; scripts de adquisición, expansión segura de ZIP, extracción, verificación CDX e integración incluidos. requirements-oficiales.txt y requirements-ocr-windows.txt separan el procesamiento general del OCR. Los JSON OCR permiten reutilizar el texto sin el motor Windows. No confundir el corpus nuevo con el archivo original de titulares al continuar E2/E3; utilizar el manifiesto y el cargador por origen.

## 14. Diseño de integración textual acordado · 9 de noviembre de 2026

### Dataset de referencia y variantes

El **escenario declarado es la versión oficial del dataset textual**. No confundir la versión oficial del dataset con el canal de documentos oficiales: el mismo escenario temporal rige también para la prensa. Mantener las advertencias históricas, reglas de precisión de fecha, cuarentena y trazabilidad de §13. El escenario estricto se reserva para análisis de sensibilidad y auditoría de disponibilidad, con su cobertura limitada explícita.

Comparar tres variantes: **(1) variables numéricas sin texto; (2) las mismas variables numéricas y texto oficial; (3) las mismas variables numéricas, texto oficial y prensa**. La comparación principal estima el aporte del texto oficial frente a la referencia numérica; la tercera estima el aporte adicional de la prensa. Conservar orígenes, horizonte24, calendario y conjunto numérico comunes; no descartar días sin texto. Mantener particiones vigentes (referencia2000–2017, validación externa2018–2021, prueba2022–2025), semillas, MASE, margen relativo3 % y presupuesto de búsqueda comparable. El ajuste final conserva la historia anterior a2022 y su reserva interna. Las cinco semillas y artefactos numéricos actuales no se modifican.

### Canales independientes y relevancia aprendida

Separar por `tipo_fuente`: `official_report` para XM, IDEAM, CREG, UPME y MinEnergia; `news` para los siete medios conservados (El Colombiano, El Tiempo, El Espectador, La República, Portafolio, Semana y Dinero). Los boletines IDEAM previos siguen en el canal oficial aunque solo tengan titular; distinguir su alcance respecto de los nuevos cuerpos. Usar `data/processed/corpus_texto.parquet`, cargar con `cargar_corpus('declarado')` y seleccionar con `documentos_en_origen`.

Cada canal produce su representación agregada y un indicador propio de ausencia. Sin documentos, su representación es cero con indicador activo; ausencia en el archivo no equivale a ausencia de eventos. La ponderación de fragmentos/documentos se aprende dentro del modelo mediante la pérdida de pronóstico, condicionada cuando corresponda a la historia numérica disponible. No filtrar prensa por términos ni asignar relevancia o temas manuales. Los tópicos NMF del EDA describen el archivo de titulares y no constituyen etiquetas ni embeddings predictivos del corpus ampliado. Los embeddings de cada canal se fusionan conservando su identidad y la posibilidad de reducir el peso de la prensa.

Fijar con desarrollo y versionar antes de la siguiente comparación: ventana temporal, segmentación determinista, límites de documentos/fragmentos/tokens por canal, política de selección temporal previa al límite, encoder y política de ajuste, controles por fuente/volumen y tratamiento de cuerpos parciales. La ventana de72h del informe de recencia es descriptiva, no un hiperparámetro ya seleccionado. Conservar originales y alcance de extracción; registrar descartes por presupuesto. Un aumento de prensa no puede desplazar documentos del presupuesto oficial. Los identificadores, URL y pruebas de disponibilidad son auditoría, no contenido del encoder. Vocabulario, escalas, relevancia y cualquier ajuste se aprenden solo con el tramo permitido de cada pliegue; no optimizar filtros ni composición contra prueba.

### Reporte y alcance

Reportar diferencias apareadas sobre los mismos orígenes, cobertura, días sin texto, volumen por canal, composición por fuente, parámetros adicionales y coste de inferencia. Definir controles de volumen y ablaciones por fuente con desarrollo. Un resultado sin mejora de prensa queda circunscrito al corpus y al procedimiento. Documentar versión y corte temporal conocido de cualquier encoder preentrenado; no presumir que sea históricamente independiente.

El bloque2022–2025 ya fue examinado en el análisis numérico; las nuevas comparaciones sobre ese bloque tendrán alcance retrospectivo. El protocolo textual se versionará antes de ejecutarse, sin presentarlo como previo a exposiciones ya ocurridas. Un bloque prospectivo genuino debe comenzar después de congelar protocolo/modelo y registrar entradas y pronósticos en tiempo real.

### Documentación y publicación

El capítulo `book/07_texto.ipynb` se titula «Corpus documental y procesamiento del texto». Describe variantes, disponibilidad, canales, relevancia aprendida, controles y código de preparación por origen; conserva el EDA léxico de titulares con su alcance identificado. `scripts/diseno_texto_jbook.py` integra estos contenidos al generador `scripts/crear_jbook.py` para evitar perderlos en reconstrucciones. Los inventarios anuales por escenario y la recencia por partición/fuente se enlazan en el capítulo desde `_static/resultados/texto_oficial/`.

Actualización documental autorizada para publicar en GitHub Pages. Sin nuevos entrenamientos, embeddings ajustados, métricas predictivas ni cambios de datos o modelos. La continuidad específica del libro se mantiene en `continuidad/CONTEXTO_PROYECTO.md` al publicar; el contexto raíz remoto conserva además sus antecedentes técnicos.

## 15. Comparación gráfica observado–predicho · 10 de noviembre de 2026

El §10.7 del capítulo de modelos presenta las35.064horas de prueba2022–2025 mediante densidad observado–predicho de las cinco inicializaciones por familia. Diagonal de identidad, escalas y colores comunes; cada panel anota MAE y RMSE medios entre semillas, comprobados contra el benchmark. Los175.320pares hora–semilla de cada panel no son horas independientes. Se conserva el detalle horario del1–7julio2023 con observado, media de pronósticos, rango p10–p90 y MAE de los errores individuales de la semana. La media de pronósticos es visual; no sustituye el estimando del benchmark. Ambas figuras incluyen interpretaciones integradas y su código analítico completo.

El §10.5 enlaza directamente esta comparación. `scripts/observado_prediccion_jbook.py` integra la sección al generador; los PNG y `resumen_observado_predicho.csv` se conservan en `results/jbook/modelos_actuales/`. La actualización utiliza exclusivamente predicciones guardadas, verifica2304huellas y reproduce las métricas existentes; no altera pesos, configuraciones, datos, semillas ni benchmark. Se actualiza el libro local sin entrenamiento ni publicación. No copiar automáticamente esta edición a GitHub Pages.

## 16. Importancia agrupada de variables · 10 de noviembre de 2026

Se incorpora a `book/10_evaluacion.ipynb` el análisis de importancia de las entradas efectivas de los cuatro modelos congelados, enlazado desde §10.5. `scripts/importancia_variables.py` reproduce en CPU los20checkpoints con sus escaladores y fuentes guardadas, verifica sus SHA-256 y calcula13grupos×10desplazamientos×5semillas×4familias=2600registros. Los donantes trasladan circularmente orígenes completos dentro de cada año2022–2025, con separación circular mínima28días y semilla11037+r. No se ajustan modelos, escaladores ni opciones de búsqueda. La mayor diferencia frente a los pronósticos guardados es26,915kWh en un punto y0,127kWh de MAE; las comparaciones de perturbación usan un baseline CPU común y el benchmark conserva su hash.

Grupos: demanda completa (perfil y nivel); perfil normalizado con nivel fijo; nivel de ventana; calendario completo; semana/festivos; año/temporadas; siete bloques de24horas de demanda. Los grupos se solapan y no son una descomposición aditiva. Se preservan conjuntamente las codificaciones del grupo en pasado y futuro. Los bloques3–7 de ARX quedan marcados como no utilizados porque solo recibe48rezagos. La hora es invariante entre orígenes diarios a medianoche y su importancia no es identificable mediante esta permutación; no se informa como cero. El clima, hidrología y texto no forman parte de estos modelos y no reciben puntuación.

La diferencia de MAE dividida por q168 expresa importancia en MASE. Para las figuras se promedian primero las semillas de cada desplazamiento; el cambio relativo usa el MAE basal medio de la familia. Los segmentos p10–p90 describen solo dispersión entre diez desplazamientos tras promediar semillas; no son IC temporales. La perturbación rompe dependencias entre grupos y puede crear combinaciones artificiales; mide sensibilidad del modelo, sin efectos causales ni el resultado de retirar una entrada y reentrenar. El diagnóstico sobre prueba es retrospectivo y no se usa para selección.

Resultados y protocolo: `results/jbook/importancia_variables/`, con reproduccion_baseline.csv, permutaciones.csv, resumen.csv, donantes.npz, protocolo.json, ejecucion.json y dos PNG. Los CSV y registros se enlazan desde `_static/resultados/importancia_variables/`. `scripts/importancia_jbook.py` integra la sección al generador y mantiene visibles las gráficas, análisis, interpretaciones y código completo de inferencia/perturbación. Se mantienen las gráficas observado–predicho. Actualización local sin entrenamiento ni publicación.

## 17. Datos adicionales y diseño recibido · 10 de noviembre de 2026

### Descargas verificadas e incorporación

Manifiesto: `data/processed/dataset_ampliado_20261110.json`. Informe reproducible: `results/datos_adicionales_20261110/INFORME_DESCARGAS.md`. Validación:572 originales con SHA-256, sin claves duplicadas; panel numérico, feature_registry y titulares originales conservan sus hashes. No se ajustan modelos, escaladores ni embeddings y no se publica el JBook por esta actualización.

- Pronósticos JMA GSM y GFS Global:3.061.440 filas, ocho ciudades, adelantos 1/2/3 días, temperatura/humedad/precipitación hasta 9 nov 2026. JMA permite cubrir todos los orígenes 2018–2025 con 24 h y ocho ciudades bajo el proxy de disponibilidad y respaldo a un adelanto anterior; GFS tiene cobertura menor. No promediar proveedores.186 precipitaciones negativas del proveedor quedan NaN con bandera, conservando originales y CSV de anomalías. No imputar con ERA 5 futuro.
- ISE:168 anexos descargados y sus boletines;102 ediciones normalizadas en `ise_vintages.parquet`,66 conservadas aún solo como Excel y12 sin fecha de publicación identificada. El archivo `ise_dane.csv` contiene 260 meses de la base 2015 (enero 2005–agosto 2026), edición 19 oct 2026. Las ediciones antiguas incluyen 2000, con otras bases/metodologías; no empalmarlas de forma automática ni sustituir vintages desconocidos por el final. La primera publicación identificada entre las ediciones normalizadas es 21 sep 2018; la existencia de valores anteriores no prueba su disponibilidad histórica.
- Población DANE:216 filas, ocho ciudades×27 años,2000–2026; total=cabecera+rural y pesos anuales suman 1. Son proyecciones/retroproyecciones revisadas con publicaciones 2023/2025. `ciudades.csv` y los pesos congelados no se reemplazan.
- Extensión XM 2026: demanda real/comercial y precio válidos hasta 7 nov 23:00; hidrología hasta 9 nov. Las 48 horas pendientes se conservan nulas. ERA 5 de las ocho ciudades tiene datos efectivos hasta 4 nov 18:00; la malla posterior queda nula. Archivos separados en `data/external/actualizacion_2026/`; no se modifica el panel 2000–2025.
- Pronóstico oficial XM SIMEM:762.480 filas,18 subáreas, enero 2022–octubre 2026. Todas las publicaciones SIMEM descargadas son posteriores a la hora objetivo. `FechaPublicacion` no es emisión original; conservar para auditoría, sin habilitarlo como entrada ni referencia operativa. La antigua URL UUID fue sustituida por el identificador público de conjunto 63 d90 d del catálogo oficial.
- El corpus declarado se conserva en `data/processed/corpus_texto.parquet`; el repositorio debe incluirlo junto con diccionario y manifiesto para continuar sin reconstruir los 7 GBde originales documentales. Los binarios originales permanecen locales y la procedencia queda registrada.

### Reglas para futuros entrenamientos

Mantener congelados resultados, arquitecturas y particiones actuales. En la nueva comparación, fijar un conjunto numérico común para todas las variantes antes de buscar: demanda, calendario y las covariables que se adopten (clima pasado, hidrología y precio), con disponibilidad definida. La ausencia actual de esas covariables de los modelos no constituye evidencia de falta de utilidad.

No hay pronósticos meteorológicos para el entrenamiento de referencia 2000–2017. La rama de pronósticos debe evaluarse en un subperiodo con historia real de ajuste y un protocolo específico; no introducirla en los pliegues originales como si estuviera completa. La climatología del horizonte puede ajustarse solo con entrenamiento; no sustituir pronósticos ausentes por el ERA 5 de la hora objetivo.

Previous Runs no identifica el ciclo exacto. `emision_estimada=valido−lead×24h` y `disponible_estimado=emision_estimada+12h` son proxies. `src/demanda/datos_adicionales.py` exige aceptación explícita de esta suposición, proveedor identificado, conserva máscara y selecciona el último adelanto completo permitido. La publicación real no queda certificada. No confundir esta suposición meteorológica con la aceptación del escenario declarado de texto.

La latencia de 1 h del clima ERA 5 en la configuración existente no queda certificada por estos archivos: la descarga 2026 muestra varios días de retraso. Para una evaluación operativa nueva, definir una latencia compatible con el producto, usar observaciones publicadas oportunamente o declarar el análisis retrospectivo. No cambiar silenciosamente los parámetros del benchmark congelado. Las horas originales de publicación de demanda/precio/hidrología también siguen siendo supuestos pendientes de verificación.

ISE por origen: seleccionar edición ya publicada (día siguiente si solo tiene fecha) y mes con rezago conservador 60 días; excluir fechas desconocidas y bases distintas. El nuevo selector no utiliza la revisión final para todos los años. Las poblaciones actuales revisadas requieren igual tratamiento de vintage si se usan para ponderar covariables.

El nuevo periodo 2026 es retrospectivo y no un test prospectivo. La cobertura de texto 2026 aún no está consolidada exhaustivamente. Un bloque prospectivo real empieza después de congelar protocolo/modelo y registrar entradas y pronósticos. No se autoriza entrenamiento mediante la incorporación de estos datos.

### Archivo de diseño y revisión

Se conserva sin modificar `literature/Diseño de entrenamiento pronóstico de demanda eléctrica con texto.md`. Es una propuesta recibida, no un preregistro adoptado. Su revisión está en `literature/REVISION_DISENO_ENTRENAMIENTO_20261110.md`. Mantener las decisiones confirmadas de §14: declarado principal; tres variantes; canales oficial/prensa y relevancia aprendida, sin filtrado por palabras clave.

Antes de implementar: corregir que una cota de corrección garantiza deterioro acotado, no «no empeorar»; revisar la proposición de pérdida placebo cuando la base no es óptima; fijar campo receptivo real de TCN; mantener la base congelada en eval; no aplicar Clark–West directamente a MAE; conservar MASE fijo y selección por menor valor; reconocer exposición de prueba y conciliar presupuestos y fuente de entrenador con los artefactos v3/v4. No copiar automáticamente fórmulas de memoria, exenciones de búsqueda ni la fecha prospectivaenero 2026 al preregistro.

Reproducción y paquetes: `requirements-datos.txt` y `requirements-datos.lock.txt`. Nuevos scripts de descarga en `scripts/adquisicion/`; `scripts/validar_datos_adicionales.py` verifica cobertura y hashes. Las reglas as-of se prueban en `tests/test_datos_adicionales.py`. No sustituir los controles de texto por un placebo que pueda introducir documentos posteriores al origen aunque estén en la misma partición.

## 18. Diseño corregido y contrato numérico común · 10 de noviembre de 2026

Se adopta el diseño corregido de `literature/Diseño de entrenamiento pronóstico de demanda eléctrica con texto.md` y su contrato `config/diseno_entrenamiento_texto.json`. Esta decisión sustituye el estado de propuesta sin modificar descrito en §17. El diseño está listo para implementar; no constituye una nueva ejecución ni un preregistro retroactivo.

Las tres variantes N, N+O y N+O+P comparten 30 variables históricas (demanda, precio, cuatro hidrológicas y 24 climáticas de ocho ciudades) y 23 de calendario. `src/demanda/diseno_numerico.py` selecciona las mismas columnas para todas, conserva NaN/máscaras y añade 167 h al clima ya desplazado 1 h, para un rezago efectivo de 168 h. Sensibilidad fijada: 90 días. El panel y registro congelados no se sobrescriben. La disponibilidad histórica de versiones ERA5 no queda certificada: ERA5T tiene unos cinco días de retraso y ERA5 final dos a tres meses; el análisis con el archivo recuperado sigue siendo retrospectivo. No usar clima, precio o hidrología realizados en el horizonte. Imputación y escalado se ajustan solo en el tramo de optimización de cada pliegue. No introducir ISE, pronósticos incompletos ni pesos poblacionales revisados como entradas comunes sin otro protocolo.

El escenario declarado es el dataset oficial; estricto es sensibilidad. Oficial/prensa mantienen embeddings y máscaras independientes, con relevancia aprendida y sin filtro temático. Ventana principal72h (sensibilidades24/168),64 documentos por canal,8 fragmentos/documento,512 tokens incluyendo prefijo/especiales y solapamiento64. Encoder congelado multilingual-E5-base, revisión d128750597153bb5987e10b1c3493a34e5a4502a, prefijo query:, promedio enmascarado y norma L2. Las representaciones por fragmento se conservan; la agregación se aprende dentro del modelo. Estos límites se auditan por fuente y truncación.

Se corrigen selección por menor MASE, escala fija q168=215617,90361589016kWh, margen relativo3%, inferencia pareada temporal y denominador de referencia uniforme por contraste. Se conservan cuatro pliegues2018–2021, ventana168, horizonte24, reserva interna10% y purga24h. La prueba2022–2025 y los datos2026 son retrospectivos. El prospectivo comienza después de una congelación efectiva y registros en tiempo real.

La fusión garantiza corrección y deterioro acotados, identidad exacta sin texto y base congelada siempre en eval; no garantiza mejorar. Se corrige el óptimo de la penalización placebo, se exigen donantes anteriores al origen, dos convoluciones por bloque TCN y campo receptivo verificado. Clark–West no se traslada al MAE. El presupuesto nuevo no se confunde con el histórico. E2§14 exime solo configuraciones concretas publicadas que realmente se adopten; en los demás casos se enumeran1000 arquitecturas, fase1 de50/200épocas, fase2 con al menos diez finalistas más control y cinco semillas. La precisión definitiva se dimensiona con un piloto, con objetivo de potencia 0,80 en E2 y 0,90 en E3; cinco semillas no acreditan esos objetivos.

Pendientes antes del siguiente entrenamiento: fichas exactas de arquitecturas publicadas, motor versionado basado en el contrato v3/v4, adaptadores con covariables, causalidad, piloto y número definitivo de semillas, registro fechado de configuración y hashes. Los resultados actuales utilizan demanda/calendario y permanecen intactos; no atribuirles el conjunto ampliado. La revisión de literatura y las garantías de novedad del modelo propio siguen requiriendo verificación específica.

Esta actualización autoriza publicación del JBook con el diseño corregido y las gráficas locales observado–predicho e importancia agrupada. No ejecuta entrenamiento, búsqueda, ajuste de embeddings ni nuevas métricas predictivas. Mantener los checkpoints, predicciones y benchmark conservados.

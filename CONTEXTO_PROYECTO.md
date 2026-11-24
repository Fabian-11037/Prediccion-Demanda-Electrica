# Contexto del proyecto

Actualizado el 2026-11-21. Idioma: español. Autores: Fabian José Manrique Morelos y Gloria Bastidas Rodriguez. Maestría en Estadística Aplicada, Universidad del Norte.

## Estado vigente · 20 de noviembre de 2026

La evaluación conserva 15 procedimientos aprendidos, 75 realizaciones y 2 referencias estacionales. El menor MASE medio corresponde a TimesNet · N: 0,6429278. El MCS al 90 % retiene TimesNet · N. Hay 51 gráficas, 47 tablas y 28 esquemas. Los conteos y las conclusiones se regeneran desde resultados; ver §37. Esta actualización es local; no entrenar ni publicar sin solicitud expresa.

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

Objetivo: demanda real horaria nacional del SIN, en kWh por intervalo. Pronóstico directo de 24 horas desde las 00:00 de Colombia, con 168 horas de historia. Las cuatro familias de demanda/calendario utilizan demanda histórica y 23 variables de calendario; ARX utiliza 48 rezagos. Los procedimientos ampliados incorporan 30 variables históricas: demanda, precio, cuatro hidrológicas y 24 climáticas, además de calendario y máscaras de ausencia. TaTS, fusión tardía y TGForecaster N+O incorporan documentos oficiales. ISE y prensa no se incluyen en los resultados disponibles.

MASE es la métrica primaria, con denominador fijo de diferencias semanales de 2000–2017: 215617,90361589016 kWh. MAE y RMSE complementan el benchmark integrado; los diagnósticos incluyen sesgo y perfiles horarios. sMAPE y R² de análisis parciales no se presentan como métricas comunes de todos los procedimientos. Margen relativo de interés: 3 %; α=0,05 y potencia objetivo 0,80. Cinco semillas finales por familia no acreditan esa potencia. La unidad temporal es el origen diario con sus 24 errores; las semillas no vuelven independientes los días compartidos.

## 3. Datos (carpeta `data/`)
Inventario y detalle en `data/raw/README.md` y `data/external/README.md`.

| Bloque | Cobertura | Estado |
|---|---|---|
| Demanda real, demanda comercial, precio de bolsa (XM) | horario 2000-01-01 → 2025-12-31 23:00, 227 928 filas, sin huecos ni NaN | incluido (`data/raw/numeric/hourly`) |
| Clima observado: temperatura, humedad, precipitación en 8 ciudades (ERA5 vía Open-Meteo) | igual | incluido; **reanálisis, no pronóstico** |
| Habitantes por ciudad (censo 2018 del DANE ajustado por omisión censal) | estático | incluido en `data/raw/numeric/ciudades.csv`; cifras de **fuente secundaria** (Portafolio, 17-oct-2019, citando al DANE); sustituir por el archivo oficial |
| Hidrología diaria (llenado, reserva útil, capacidad útil, aportes) | 2000-01-01 → 2025-12-31 | incluido |
| Titulares de prensa (7 medios) y boletines IDEAM | 568 295 titulares, 2004-2025 | incluido (`data/raw/noticias/titulares.parquet`) |
| Corpus integrado de prensa y documentos oficiales | 625577 filas; ampliación oficial versionada en §41 | data/processed/corpus_texto.parquet; manifiesto activo y cobertura en §41 |
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

Entre las cuatro familias de demanda/calendario, MLP es el baseline empírico y reduce el error un 44,5 % frente al ingenuo semanal. El benchmark ampliado y su ranking vigente están en §36–40. ARX y CNN1D tienen errores agregados próximos. Se calculan medias, DE e intervalos t95 % entre cinco inicializaciones, condicionados a las mismas fechas. No confundir esos intervalos con incertidumbre temporal ni la banda de semillas con un intervalo predictivo calibrado.

La evaluación integrada promedia pérdidas individuales entre semillas antes de remuestrear; no evalúa una predicción ensemble. Utiliza bootstrap circular pareado con 10.000 réplicas y bloques de 7/14/28 días. DM–HLN utiliza diferencias diarias, HAC Bartlett de 13 rezagos y horizonte efectivo de un origen no solapado. Holm considera 136 parejas temporales, 105 entre semillas y una familia específica de tres contrastes textuales. Los 72 contrastes textuales por horizonte se ajustan conjuntamente.

Los diagnósticos integrados utilizan el residuo diario medio de la predicción media de cinco semillas, exclusivamente para diagnóstico. Incluyen Ljung–Box 7/14/28, ARCH-LM 14, BDS dimensión 2, Jarque–Bera y Mincer–Zarnowitz con HAC 13; cada familia se ajusta por Holm entre 15 procedimientos. BDS no aísla no linealidad cuando existe memoria. Las condiciones vigentes están en `results/evaluacion_e2/protocolo.json`; los análisis parciales de las cuatro familias base conservan su propio alcance.

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

Separar por `tipo_fuente`: `official_report` para XM, IDEAM, CREG, UPME, MinEnergia y Presidencia; `news` para los siete medios conservados (El Colombiano, El Tiempo, El Espectador, La República, Portafolio, Semana y Dinero). Los boletines IDEAM previos siguen en el canal oficial aunque solo tengan titular; distinguir su alcance respecto de los nuevos cuerpos. Usar `data/processed/corpus_texto.parquet`, cargar con `cargar_corpus('declarado')` y seleccionar con `documentos_en_origen`.

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

## 19. Arquitecturas y evaluación estadística E2 · 15 de noviembre de 2026

Se incorporan doce fichas bajo `book/arquitecturas/`: ARX, MLP, LSTM, CNN1D, PatchTST, N-HiTS, TFT seleccionado, TCN, iTransformer, TaTS, TGForecaster y fusión tardía. Cada una describe entradas, dimensiones, operaciones, fórmulas y adaptaciones reales; contiene SVG/PNG, código del esquema, red y pipeline de datos, optimización y prueba. Las fuentes de los seis paquetes recibidos se conservan en `results/arquitecturas_e2/fuentes/`; las huellas de las fuentes de los cinco paquetes con resultados coinciden con REGISTRO.json. La limitación de fuentes faltantes de la consolidación inicial queda resuelta por estos adjuntos. Los bloques de código se muestran mediante literalinclude y no ejecutan entrenamientos al construir el libro.

TFT principal: paquete Modelo3_TFT, ancho160, cuatro cabezas, selección GRN por variable, LSTM, atención interpretable causal; la versión compacta del repositorio permanece alternativa. TGForecaster sólo tiene fuente/configuración, sin métricas ni rangos inventados. Las cuatro familias base conservan pesos/configuraciones/entradas originales. Las variantes ampliadas reciben clima, hidrología, precio, demanda y calendario; se declaran 30 máscaras en el repositorio frente a tres por bloque en los paquetes.

`scripts/estadistica_e2.py` calcula resultados desde predicciones guardadas; entradas de consolidación en `results/evaluacion_e2/entradas/`. Doce procedimientos con cinco semillas, dos ingenuos, 1461 orígenes diarios y 35064 horas por semilla. El control N de TaTS/fusión no se duplica. q168 fijo215617,90361589016; margen relativo3%; bootstrap circular pareado10000, bloques14d con sensibilidad7/28; HAC 13d y HLN h efectivo1 para las emisiones diarias no solapadas. El ingenuo diario respeta publicación1h y usa48h en el último horizonte. Se calculan 91 comparaciones temporales, 66 entre semillas, Holm por prueba/familia, t pareada, Wilcoxon exacto, Shapiro, Levene, Cohen dz con IC no central, Friedman/Nemenyi/CD, MCS90%, TOST, error por horizonte, Ljung–Box, ARCH-LM, BDS y Mincer–Zarnowitz HAC. Los diagnósticos usan residuo diario de la predicción media sólo para diagnóstico; el benchmark promedia pérdidas, no predicciones.

Resultados principales: MCS retiene TaTS N+O y control N. TaTS: cambio relativo−0,605%, DM pHolm(2)=0,4403, bootstrap pHolm(2)=0,4356; no mejora demostrada. TOST pHolm(2)=0,00120 acredita equivalencia temporal ±3% condicionada a las cinco semillas. Fusión tardía: cambio+4,654%, DM pHolm(2)=4,35e−12, bootstrap pHolm(2)=0,000200; empeora. No se interpreta esa equivalencia temporal como generalización a nuevas semillas. Friedman Q51,5538 y p3,28e−7; Wilcoxon con cinco pares no rechaza tras Holm. BDS no demuestra no linealidad por sí solo si existe autocorrelación. Los contrastes son retrospectivos y no convierten el bloque examinado en confirmatorio.

La pauta E2 se revisó completa (65 páginas). Persisten pendientes de cumplimiento: TGForecaster evaluado, ARIMA/ETS, presupuesto homogéneo, potencia acreditada, búsquedas arquitectónicas §14 cuando apliquen y ablaciones/robustez controladas. No se aplica Clark–West a MAE ni se inventan intervalos predictivos/PIT a partir de la dispersión entre semillas. Los pvalores y limitaciones se muestran junto a las conclusiones.

Reproducción: `scripts/crear_arquitecturas_e2.py`, `scripts/estadistica_e2.py`, `scripts/crear_capitulo_e2.py` y `scripts/arquitecturas_e2_jbook.py`; luego jupyter-book build y scripts/verificar_jbook.py. El generador principal conserva índice/alcance con el integrador E2. `scripts/pipeline_ampliado_portable.py` aporta ajuste/validación o ajuste/prueba explícitos en CPU/GPU para PatchTST/N-HiTS, conserva motor y parámetros y exige permiso de prueba; sólo se compiló y verificó su guarda, sin entrenar. Se añaden cinco curvas por procedimiento a la evaluación. Actualización local: no se publica ni se modifica un checkpoint, una predicción conservada o una semilla.

## 20. Integración y orden del capítulo 10 · 15 de noviembre de 2026

Se unifican las secciones de modelos/resultados y las dos evaluaciones. `10_modelos.ipynb` queda como introducción; sigue `arquitecturas/fundamentos.md` (§10.1), doce fichas con código (§10.2–10.13) y una sola evaluación `10_evaluacion.ipynb` (§10.14). Se retiran del libro `10_pipeline.md` y `10_estadistica_e2.ipynb`. El código único de partición, datos/escalado y métricas pasa a fundamentos; los pipelines particulares permanecen en cada ficha. Las páginas antiguas tendrán redirección local sin contenido duplicado.

La evaluación común utiliza la inferencia global disponible, sin repetir los contrastes parciales de tres redes con otra familia de Holm como si fueran conclusiones globales. Conserva selección y piloto, generalización, observado–predicho, segmentos, variación fecha/semilla, atípicos y huella de prueba, coste e importancia de variables. Los diagnósticos complementarios de cuatro familias mantienen ese alcance explícito. Se eliminan la curva de una sola semilla, los benchmarks/rangos/residuos formales parciales duplicados y el perfil horario repetido entre segmentos. Las curvas de cinco semillas, las comparaciones globales y sus hipótesis son las de la evaluación integrada. Las tablas originales permanecen en results; no se alteran predicciones, pesos o semillas.

`scripts/reorganizar_capitulo10.py` preserva las fuentes antes de la integración en `results/reorganizacion_capitulo10/fuentes/`, separa los nombres del análisis general y los del grupo base, ejecuta sólo análisis conservados y mantiene el orden. `crear_jbook.py` lo invoca como último paso; `crear_capitulo_e2.py` genera su cuaderno fuente fuera del libro y lo integra, en vez de recrear una evaluación independiente. `crear_arquitecturas_e2.py` utiliza el pipeline archivado para las fichas y respeta la numeración. El integrador E2, los enlaces, la síntesis y el verificador se ajustan al nuevo orden. Actualización local, sin entrenar ni publicar.


## 21. Auditoría integral de cobertura gráfica y estadística · 15 de noviembre de 2026

La evaluación integrada de book/10_evaluacion.ipynb utiliza doce procedimientos completados, cinco semillas cada uno (60 realizaciones), 1.461 orígenes diarios y 35.064 horas de prueba por realización. Añade dos referencias deterministas. El control N de TaTS/fusión se cuenta una vez. TGForecaster conserva arquitectura y pipeline, sin resultados; no se incorpora artificialmente a contrastes ni gráficas de rendimiento.

Se corrigieron las selecciones parciales de presentación: la tabla entre semillas muestra las 66 parejas completas, diferencia MASE, Shapiro, t pareada/Wilcoxon con Holm y Cohen dz con IC95 %. Incluye matriz completa del efecto y Q-Q de los once procedimientos frente al control N. Friedman/rangos incluye doce procedimientos; Nemenyi presenta las 66 parejas (siete rechazos), y el MCS muestra los catorce participantes, con paso, p y estado retenido/excluido. Retiene solo TaTS y N: es el resultado del procedimiento, no un filtro de entrada. La t pareada ajustada rechaza en 27/66 como sensibilidad; Wilcoxon en ninguna. Se muestran las 91 parejas temporales mediante matriz DM–HLN Holm, con 84 rechazos; las diferencias e intervalos siguen descargables. No confundir incertidumbre de inicialización con temporal ni rechazo global con todas las parejas significativas.

Se ampliaron a los doce procedimientos generalización validación/prueba, densidad observado–predicho, perfil del 1–7julio2023, segmentos por año/tipo de día/cuartil de demanda, descomposición descriptiva fecha/semilla/interacción y sensibilidad a atípicos. Curvas e indicadores por horizonte ya abarcan los doce. Las figuras usan las cinco predicciones individuales para las métricas; la media de pronósticos es solo visual. Escalas comunes en observado–predicho y semana, banda p10–p90 sin atribuirle cobertura probabilística. Las predicciones se alinean por semilla, origen y horizonte y reproducen MAE/MASE/RMSE del benchmark.

El piloto gráfico incorpora los seis procedimientos con piloto independiente (120 resultados semilla–año). Las seis variantes ampliadas restantes no conservan piloto; no se reemplaza por prueba ni se dibuja como cero. El dimensionamiento integrado se amplía en §22 a los seis procedimientos con piloto, con disponibilidad explícita para los doce. La validación externa de PatchTST/N-HiTS del repositorio promedia2019–2021; las demás2018–2021. El coste registrado y los parámetros incluyen los doce procedimientos. La latencia CPU y la importancia por perturbación permanecen medidas solo en las cuatro familias de demanda/calendario: las ocho ampliadas figuran pendientes. No inferir importancia nula de clima/hidrología/precio/texto a partir de esa ausencia. Las métricas ajuste/parada se muestran en paneles separados: MAE normalizado para cuatro familias y MASE para seis variantes ampliadas; los dos modelos numéricos del repositorio solo conservan curvas en ese diagnóstico. No convertir Huber ni pérdida online con dropout en MAE de evaluación.

Se armonizaron introducción, pregunta de investigación, alcance bibliográfico, diseño y reproducción. Diseño declara presupuestos distintos, 60 realizaciones, máscaras30/3 y HAC 13 en la inferencia integrada. El margen relativo sigue3 %; en la familia textual el denominador es el control N. El código de análisis y las interpretaciones están en las celdas del libro; el alcance y las ausencias se resumen en cobertura_diagnosticos.csv. Importancia/latencia adicional, piloto no ejecutado, TGForecaster, potencia y requisitos experimentales pendientes no se reconstruyen a partir de predicciones finales.

Implementación: scripts/completar_evaluacion_jbook.py, invocado por scripts/reorganizar_capitulo10.py y por el generador completo. metadatos_procedimientos.csv conserva60 filas; pilotos_disponibles.csv120. Se pueden reutilizar en otro PC sin la carpeta hermana de consolidación. La armonización es idempotente para evitar párrafos repetidos. Se preservan código visible, Markdown nativo y orden arquitecturas antes de resultados. No se modifican pesos, configuraciones, semillas, panel ni resultados estadísticos guardados. Sin entrenamiento, nuevas predicciones de redes ni publicación. Verificación: 69 pruebas correctas y cinco omitidas por dependencias opcionales; auditoría de construcción y cobertura en results/evaluacion_e2/.

Cierre de auditoría §21:13cuadernos,84celdas analíticas,48figuras analíticas y12esquemas. Construcción sin advertencias, cero errores y enlaces locales rotos; hashes del panel y benchmark preservados. Se comprobó la reconstrucción de metadatos sin carpeta externa. Informe:results/evaluacion_e2/REVISION_COBERTURA.md y auditoria_cobertura.json.


## 22. Selección y dimensionamiento integrados · 15 de noviembre de 2026

La sección «Entrenamiento, selección y precisión del piloto» presenta ahora los doce procedimientos completados mediante diseno_seleccion.csv: candidatos, ajustes y semillas de filtro/confirmación, años externos, piloto independiente, cinco semillas finales, máximos de épocas y paciencia efectiva. Los presupuestos de las cuatro familias de demanda/calendario se obtienen de todos los CSV de filtro/confirmación conservados (90 y144ajustes), sin omitir sus rondas. PatchTST/N-HiTS conservan60ajustes de filtro y45confirmación cada uno; los seis procedimientos ampliados restantes20y15por variante. Sus criterios de selección promedian2019–2021y2018–2021,respectivamente. TGForecaster no tiene resultados.

«Dimensionamiento y precisión del piloto» sustituye el bloque limitado al grupo demanda/calendario. El resumen incluye doce filas: seis pilotos con cinco semillas100–104, promediadas sobre cuatro años2018–2021, y seis procedimientos sin piloto con métricas sin estimar. Se dimensionan las15parejas de los seis pilotos, incluyendo ARX,MLP,LSTM,CNN1D,PatchTSTyN-HiTS. Se conserva δ relativo0,03 y como escala de las diferencias la media piloto de MLP0,5415568537267458; no se utiliza prueba. Aproximación normal bilateral α0,05,potencia0,80, mínimo5semillas; se informa DE relativa,δ/DE,Rindividual,RBonferroni(15)yMDDcon5. Rindividual5–45y máximoBonferroni82. Es planificación retrospectiva entre inicializaciones, no el diseño que determinó los cinco ajustes finales ni una garantía de potencia temporal. Los resultados guardados de prueba, pesos y semillas permanecen intactos.

Se incorporan también los registros dimensionamiento.json de PatchTST/N-HiTS: diagnóstico temporal bootstrap14d, referencia ingenuo semanal de validación MASE1,1419350261674317, estimación límite0,0984/0,0843y n_finalno determinado. Se distingue del cálculo normal entre semillas; no se transforma en un R inventado ni se usa la variación de prueba como piloto.

Código visible en book/10_evaluacion.ipynb. Datos y cálculos descargables:results/evaluacion_e2/{diseno_seleccion,resumen_piloto_integrado,dimensionamiento_piloto_integrado,dimensionamiento_temporal_registrado}.csv,copiados a book/_static/resultados/evaluacion_e2/. scripts/completar_evaluacion_jbook.py conserva metadatos de selección y genera las celdas; scripts/reorganizar_capitulo10.py ejecuta y copia las tablas. Prueba de cobertura adicional en tests/test_cobertura_evaluacion.py: doce procedimientos,seis pilotos,quince parejas y ausencia de métricas fabricadas. Actualización local,sin entrenar ni publicar.

Verificación de §22:13pruebas de cobertura/integración/consistencia correctas;13cuadernos,87celdas analíticas,48figuras, cero errores,advertencias y enlaces rotos. HTML local en docs/10_evaluacion.html actualizado,sin publicación.


## 23. Precisión retrospectiva con parada interna · 15 de noviembre de 2026

Por solicitud expresa, se completa la precisión de los seis procedimientos sin piloto con las métricas MASE_parada de sus cinco entrenamientos finales (1000–1004):TFT,TCN,iTransformer,controlN,TaTSN+Oyfusión tardíaN+O. Se verifican30registros,803orígenes comunes2019-10-21–2021-12-31y q168=215617,90361589016. Son métricas de evaluación del checkpoint restaurado, no pérdidas online ni mínimos de otra escala. La extracción conserva ruta y SHA-256 de cada resultado y coteja las métricas con metadatos_procedimientos.csv. No se cargan pesos ni se generan nuevos pronósticos para este cálculo.

El resumen incorpora media,DE,CV e IC95%t entre cinco inicializaciones. Se calculan quince diferencias pareadas,DE e IC95% de la diferencia,DE relativa,δ/DE,Rindividual,RBonferroni(15)yMDDcon5. Se conserva margen3%,α0,05,potencia objetivo0,80 y mínimo5semillas. La referencia de escala de este grupo es el MASE medio de parada del controlN:0,5709805536589401. La fórmula utiliza la DE y el margen fijado; no la diferencia observada como alternativa para estimar potencia. Máximos:124semillas en aproximación individual y226conBonferroni. No se ha aumentado el número ejecutado ni se acredita potencia externa.

MASE de parada medio:TaTS0,564763;controlN0,570981;fusión tardía0,587055;iTransformer0,595955;TCN0,728790;TFT0,799907. El CV menor es iTransformer2,735%y el mayor,fusión tardía7,484%. Estos resultados describen la reserva de desarrollo y no reemplazan el ranking de prueba. Los IC son condicionales al periodo compartido y dependen de la aproximación t con n5.

La reserva también seleccionó la época de parada:el análisis es retrospectivo y no equivale al piloto externo de cuatro años ni a un preregistro anterior a entrenar. Se mantienen por separado los seis pilotos externos y los seis análisis de parada,sin mezclar sus medias o semillas. La tabla piloto visible muestra solo sus seis procedimientos; su inventario histórico sigue conservando doce filas y las ausencias reales. La cobertura añade «Precisión de desarrollo»:seis pilotos externos y seis reservas internas retrospectivas;TGForecaster sigue sin resultados. El diseño del capítulo9 incorpora ese alcance.

Implementación:scripts/precision_parada_jbook.py,integrado al final de scripts/completar_evaluacion_jbook.py. El JBook muestra íntegro calcular_precision_parada y sus llamadas para resumen y diferencias. Entradas/salidas:results/evaluacion_e2/{entradas_precision_parada.csv,precision_parada_resumen.csv,precision_parada_comparaciones.csv,precision_parada_protocolo.json},con copias descargables enbook/_static/resultados/evaluacion_e2/. Las métricas de prueba no intervienen en la planificación. Pruebas verifican cobertura,semillas,tramo común,invariancia ante métricas de prueba y que desplazar una diferencia media sin alterar su DE no cambie R. Se preservan panel,benchmark,pesos,escaladores,predicciones y semillas. Actualización local,sin entrenamiento ni publicación.

Verificación de §23:17pruebas de precisión/cobertura/integración/consistencia correctas. Libro:13cuadernos,89celdas analíticas,48figuras;sin errores,advertencias ni enlaces rotos. Métricas de prueba y panel preservados. HTML local actualizado en docs/10_evaluacion.html,sin publicación.


## 24. Presentación integral de precisión en §10.14 · 15 de noviembre de 2026

Se actualizan las tablas y figuras de desarrollo para incorporar las métricas de parada de los seis procedimientos sin piloto. Un único resumen visible incluye los doce procedimientos,con fuente,periodo,cinco semillas,MASE,DE,CV e IC95%t. Conserva dos fuentes:seis pilotos externos(media de cuatro años2018–2021,semillas100–104)y seis reservas internas(803días2019-10-21–2021-12-31,semillas1000–1004). Las tablas parciales duplicadas de seis procedimientos se sustituyen por el resumen;los CSV de detalle y los cálculos separados se conservan. No se agregan las dos fuentes para producir contrastes de66parejas de desarrollo ni un ranking común.

La figura de selección muestra piloto anual,puntos de cinco semillas piloto,puntos de las cinco semillas de parada y el control de permutación disponible para cuatro familias. La figura de generalización tiene las doce estimaciones de desarrollo,sin barras vacías:azul para piloto y naranja tramado para parada,con periodos explícitos;junto a validación externa y prueba. La tabla de selección añade fuente de precisión y la tabla de métricas de prueba identifica esa procedencia. Una nueva figura muestra por separado las quince parejas de planificación de cada fuente,con sus referenciasMLP/controlN y máximosBonferroni82/226.

Las figuras y tablas de observado–predicho,semana,horizontes,segmentos,residuos,atípicos,contrastes,Friedman/Nemenyi/MCS y las métricas de prueba se reconstruyen a partir de sus artefactos conservados. La parada interna no reemplaza sus pérdidas de prueba. La importancia y latencia siguen limitadas a las cuatro familias medidas;TGForecaster continúa sin resultados. El texto y la discusión aclaran que tener una estimación retrospectiva de precisión no acredita potencia ni un piloto externo.

Implementación:scripts/presentacion_precision_jbook.py,última capa de scripts/precision_parada_jbook.py. El cálculo de consolidación,las gráficas y las tablas se muestran como código analítico ejecutado en el libro. Nuevos archivos:results/evaluacion_e2/{precision_desarrollo_resumen.csv(12filas),precision_desarrollo_semillas.csv(60filas),generalizacion_integrada.csv(12filas)},con copias descargables. Gráficas:precision_desarrollo_fuentes.png,generalizacion_completa.png,dimensionamiento_fuentes.png. Se verifica cobertura de los doce,cinco realizaciones por procedimiento,ausencia de huecos y coincidencia de las métricas de prueba con el benchmark original. Sin entrenamiento,inferencia adicional,selección de configuraciones ni publicación.

Verificación de §24:75pruebas correctas y cinco omitidas;13cuadernos,91celdas analíticas,49figuras;sin errores,advertencias ni enlaces rotos. Inventario de tablas y figuras de10.14enresults/evaluacion_e2/auditoria_presentacion_precision.json. HTML local actualizado,sin publicación.


## 25. Incorporación de TGForecaster · 15 de noviembre de 2026

Se incorporan los ZIP ENVIAR_resultados_modelo7_20261114_0430.zip y ENVIAR_pesos_y_vectores_modelo7_20261114_0430.zip,con sus SHA-256 en results/modelo7_integracion/importacion.json. Importador idempotente:scripts/importar_modelo7.py. Los282archivos se extraen con rutas comprobadas a la carpeta hermana Consolidado-Entrenamientos-20261115/fuentes/adjuntos/modelo7_20261114_0430/resultados_modelo7/. Se verifican17archivos de código/configuración frente a REGISTRO.json,admitiendo equivalencia CRLF/LF sin modificar originales. Se conservan diez checkpoints (cinco N y cinco N+O),sin deserializarlos ni ejecutar redes. Las predicciones tienen1461orígenes y24h por semilla,fechas completas,sin duplicados,con350640filas nuevas. Las fechas NPY de tipo objeto se leen mediante análisis estático de cadenas ISO,sin ejecutar pickle.

Clave del experimento:adj_modelo7_20261114_0430. N:configuración19,d_model256,tres capas temporales,dos de mezcla,ocho cabezas,2899072parámetros. N+O:configuración1,d_model64,dos capas temporales,una de mezcla,cuatro cabezas,326464parámetros. Cada variante conserva20candidatos en2018,cinco finalistas en2019–2021,selección por media de cuatro años y cinco semillas1000–1004. Son80ajustes (70búsqueda+10finales),sin piloto externo independiente. La precisión de desarrollo se calcula con sus métricas de parada sobre los803días comunes anteriores a2022. No se vuelven a entrenar modelos ni se seleccionan configuraciones a partir de prueba.

TGForecaster N ya utiliza embeddings de descripciones estáticas de33canales;no debe describirse como ausencia absoluta de representaciones lingüísticas. N+O añade documentos oficiales variables por origen. Se verifica el caché oficial14162×768float16,idéntico por SHA-256 al de TaTS/fusión,por lo que no cuenta como un corpus nuevo;las33descripciones×768 constituyen otro recurso. Se actualizan índice de modelos,vectores,inventario global,manifiesto y resumen de la consolidación. El conjunto ampliado tiene50realizaciones principales,además de10alternativas/alias. Con las20realizaciones de las cuatro familias base,el JBook compara14procedimientos y70realizaciones finales,más dos ingenuos.

Resultados de prueba reproducidos con etiquetas canónicas:TGForecaster N MASE1,001992 y MAE216047,3903kWh;N+O MASE1,122419 y MAE242013,7271kWh. El texto incrementa el error12,0188%,IC95%relativo[10,2740%;13,7437%]. DM–HLN Holm(3)=1,3620e-55,bootstrap Holm(3)=0,000300;TOSTno acredita equivalencia. Los hiperparámetros de N y N+O difieren:el contraste corresponde al procedimiento completo,no una ablación de contenido con núcleo fijo.

Se recalcula la inferencia desde las predicciones conservadas:120parejas temporales,91parejas entre semillas y tres contrastes textuales,con ajustes Holm por familia. Las columnas textuales se llaman p_DM_Holm_texto,p_bootstrap_Holm_texto,p_TOST_Holm_texto;no reutilizar el nombre Holm2. El MCS90% mantiene TaTS y su controlN. FriedmanQ59,3314,p6,916e-8;Levene centrado en mediana ahora rechaza igualdad de varianzas(p0,000834). NemenyiCD8,87284. Se actualizan las sensibilidades7/14/28d y72contrastes textuales por horizonte. Las métricas originales de prueba de los otros modelos no cambian,pero sus valores p ajustados y rangos se recalculan al ampliar la familia.

La precisión de desarrollo abarca seis pilotos externos y ocho reservas internas:70métricas en total. El grupo de parada conserva40métricas y28parejas (Bonferroni28),referencia común controlN de parada0,5709805536589401;Raproximado máximo2112individual y4231Bonferroni. La gran variación entre semillas de TGForecaster aumenta esas estimaciones;son planificación retrospectiva sensible a n5,no potencia alcanzada ni un número de entrenamientos autorizado. No mezclar los periodos ni las referenciasMLP/controlN de los dos grupos.

Los generadores usan matrices/cuadrículas dimensionadas con el inventario y ocultan paneles vacíos. Todas las curvas,observado–predicho,semana,horizontes,segmentos,generalización,precisión,estadística,atípicos y costes incluyen TGForecaster. Se retiran las declaraciones de resultados pendientes en su ficha,introducción,literatura,síntesis,diseño y reproducción. La importancia por perturbación y la latencia CPU siguen disponibles solo para las cuatro familias medidas;no se inventan para los diez procedimientos ampliados. Se mantienen pesos,escaladores,datos y semillas. La actualización es local,sin entrenamiento,inferencia de redes ni publicación.

Reproducción:importar_modelo7.py;estadistica_e2.py;crear_arquitecturas_e2.py;crear_capitulo_e2.py;construcción y verificar_jbook.py. Las entradas de resultados y metadatos normalizados permanecen en results/evaluacion_e2 para reconstrucción documental sin la carpeta hermana. No usar el consolidador base antiguo sin reaplicar después el importador de modelo7. Los respaldos previos están en results/modelo7_integracion/respaldo/ y los cambios de capacidad del generador,en results/modelo7_integracion/actualizar_generadores.py.

Verificación de §25:78pruebas correctas y5omitidas. Libro local:13cuadernos,91celdas analíticas,49figuras y12esquemas de arquitectura;sin errores,advertencias ni enlaces locales rotos. Modelo7incorporado a tablas,gráficas y contrastes;HTML en docs/10_evaluacion.html actualizado,sin publicar.


## 26. Revisión integral de tablas y gráficas · 15 de noviembre de 2026

Estado vigente: 13 cuadernos, 93 celdas analíticas, 50 gráficas, 25 tablas calculadas, 24 tablas Markdown y 12 esquemas. Se revisaron todas las figuras visualmente, sus interpretaciones, periodos y unidades; las tablas mantienen columnas alineadas y no presentan valores no finitos accidentales. Se cotejaron resultados contra benchmark, pérdidas, semillas, curvas, metadatos y correcciones Holm. El capítulo 10.14 incluye 14 procedimientos, 70 realizaciones y dos referencias: 120 parejas temporales, 91 entre semillas, tres contrastes textuales y 72 contrastes por horizonte. Se corrigen denominadores antiguos (111/120 rechazos DM), etiquetas de 28 parejas de parada y máximos de dimensionamiento 82/4231. Las secciones anteriores de este contexto documentan estados históricos; los inventarios vigentes son los de esta sección y §25.

Se incorpora al capítulo 7 el corpus oficial efectivamente codificado (14.162 registros, 12.330 contenidos distintos, 6.574 solo con titular, 1.175 truncados), preservando identidad y orden del índice del encoder y verificando los hashes de contenido. Entradas autocontenidas en results/evaluacion_e2/entradas/texto/: corpus_oficial.parquet, indice_encoder.parquet, ficha_encoder.json y cobertura_original.csv. El caché incluye documentos que no estaban disponibles al cierre de 2025; la tabla distingue esa elegibilidad. La ficha informa 33,36 % de descarte por capacidad nominal, mientras que la cuenta sin duplicar el solapamiento de fragmentos produce 36,16 %. Los vectores permanecen intactos; el promedio normalizado de fragmentos precede a la relevancia aprendida entre documentos. Se añade cobertura anual de disponibilidad y volumen admitido en ventanas de 72 horas, cotejada con el archivo conservado. TaTS conserva su agregación mediante cierres diarios históricos.

El gráfico del escenario estricto ahora representa los años sin registros de desarrollo en un eje 2000–2021, evitando un panel aparentemente vacío. STL utiliza etiquetas españolas y unidades correctas de GWh por intervalo. Se precisan nombres de ciudades y precipitación acumulada en la hora previa. Diseño y texto distinguen las variables efectivamente entrenadas, los presupuestos desiguales y la variante de prensa sin resultados. Importancia por perturbación y latencia permanecen medidas en cuatro familias, ausentes para diez procedimientos ampliados; no se les atribuye importancia nula.

Implementación: scripts/revision_integral_jbook.py, última capa de completar_evaluacion_jbook.py y armonización final de reorganizar_capitulo10.py; código analítico y Markdown nativo en el libro. Informe e inventario: results/revision_integral/REVISION_TABLAS_GRAFICAS.md, inventario_tablas_graficas.csv y auditoria.json. Se actualizan también los informes de cobertura existentes. Verificación: 82 pruebas correctas y cinco omitidas; construcción sin errores ni advertencias, cero enlaces rotos. Hashes del panel y benchmark base preservados. HTML local en docs actualizado, sin entrenar, ejecutar redes ni publicar.

La tabla de «Desempeño y precisión temporal» incluye MASE de prueba con cuatro decimales, junto con MAE, RMSE, realizaciones y fuente de precisión. Los valores proceden del benchmark conservado y abarcan los 14 procedimientos y las dos referencias.


## 27. Diagnósticos ampliados con pesos congelados · 16 de noviembre de 2026

Se completaron los diez procedimientos ampliados y las cinco semillas de cada uno: 50 checkpoints restaurados con weights_only=True y carga estricta. Se conservaron los escaladores de ajuste y las configuraciones elegidas; no se inició optimización ni se sustituyó el benchmark. Las reproducciones basales cubren los 1.461 días de 2022–2025. Máxima diferencia frente a las predicciones conservadas: 778,4835 kWh por pronóstico; máximo cambio relativo de MAE: 0,005962 %. Los diagnósticos usan como referencia el basal CPU de cada checkpoint.

Importancia: 192 fechas fijas (1,8,15,22 de cada mes), cinco semillas, nueve grupos y tres desplazamientos circulares por año con separación mínima de 28 días. Se obtienen 1.350 mediciones y 90 resúmenes. Grupos: perfil de demanda, nivel de restitución, precio, reservas (llenado/reserva/capacidad), aportes, temperatura de ocho ciudades, humedad de ocho ciudades, precipitación de ocho ciudades y calendario. La hora es invariante y no queda identificada. El p10–p90 entre desplazamientos no es IC temporal; no comparar esta muestra como si fuera idéntica a las 1.461 fechas y diez desplazamientos de las cuatro familias base. El calendario domina en siete procedimientos y el perfil en tres; máximo de una covariable externa: precio en N-HiTS, +1,4768 % de MAE. No inferir efectos causales ni sumarlos.

Sensibilidad: 1.461 días y cinco semillas, manteniendo historia numérica, calendario y pesos. Controles: sin documentos; vectores a cero con presencia conservada; contenido histórico de la misma fuente con disponibilidad declarada anterior a 2022 (tres remuestreos con reemplazo, no permutación biyectiva). No se agrupa por tema ni se aprende un filtro externo de relevancia. TaTS: retirada +17,0654 %, cero +15,0910 %, contenido histórico +0,0152 % (divergencia 476,8 kWh). Fusión tardía: +11,6111 %, +3,1296 % y +1,7576 %, respectivamente. TGForecaster N+O: +30,3923 %, +28,4251 % y −1,6524 %. La sensibilidad dentro de un checkpoint no es la comparación de procedimientos N/N+O ajustados por separado. El reemplazo histórico apenas afecta TaTS y mejora descriptivamente TGForecaster; no demuestra una ventaja predictiva de las noticias recientes.

Las descripciones estáticas TG se evalúan aparte, manteniendo fija la descripción del objetivo en la relevancia documental: N a cero +167,2748 %, permutada +4,1954 %; N+O a cero +69,9672 %, permutada +9,4964 %. Son intervenciones sobre codificación de canales y consultas del horizonte, susceptibles de alejar la entrada de entrenamiento. Los 13 resúmenes textuales tienen IC95 % puntuales por bootstrap circular pareado de 14 días, B10000, después de promediar semillas y repeticiones. No convierten cinco inicializaciones en potencia acreditada ni equivalencia confirmatoria.

Latencia: las 70 realizaciones de los 14 procedimientos se miden de nuevo en la misma CPU con un hilo, batch uno, tres orígenes comunes, 20 calentamientos y 100 mediciones por corrida. Son 210 corridas. Se muestran mediana de medianas y mediana de p95 por corrida; no p95 agrupado. ARX 0,02395 ms; TFT 54,32780 ms, extremos de las medianas. Se excluyen carga, preparación y encoder E5, usando vectores congelados en memoria. Los contadores documentales distinguen siete cierres para TaTS, uno para las otras ramas documentales y cero para variantes sin documentos. La importancia agrupada y la latencia ya no están pendientes para los diez ampliados.

Entradas: carpeta hermana Consolidado-Entrenamientos-20261115 y su consolidado/indice_modelos.json; corpus e índice del encoder ya documentados en §26. Resultados autocontenidos para el libro en results/diagnosticos_ampliados; el cálculo de redes necesita además la consolidación con pesos y vectores. scripts/inferencia_ampliada.py restaura redes/entradas; scripts/diagnosticos_ampliados.py ejecuta --fase inferir (cuatro procesos de un hilo; reanuda realizaciones completas), --fase resumir y --fase latencia (serial, después de terminar perturbaciones). scripts/diagnosticos_ampliados_jbook.py integra resultados como última capa de revision_integral_jbook.corregir_capitulo10; scripts/actualizar_diagnosticos_jbook.py reconstruye sólo el capítulo con CSV guardados. La generación completa también armoniza la sección de reproducción. La construcción documental no ejecuta redes. Los resultados, las figuras y el código completo de inferencia/diagnóstico aparecen en 10.14; el capítulo 12 explica la reproducción.

Informe: results/diagnosticos_ampliados/RESUMEN_DIAGNOSTICOS.md. Trazabilidad: fuentes_diagnosticos.json (50 pesos y sus escaladores/predicciones/resultados, hashes del código, panel y benchmark base), protocolo.json, entorno.json y reproduccion_semillas.csv. Los CSV se copian a book/_static/resultados/diagnosticos_ampliados. Revisión: 86 pruebas correctas y cinco omitidas; 13 cuadernos, 97 celdas analíticas, 53 gráficas, 49 tablas y 12 esquemas; sin errores, advertencias ni enlaces rotos. Tres figuras nuevas inspeccionadas visualmente. HTML local actualizado en docs. Sin entrenamiento ni publicación. El usuario solicita apagar el equipo después del cierre y guardado; el apagado se programa como última acción.


## 28. Esquemas de arquitectura procedentes del PowerPoint · 16 de noviembre de 2026

Se reemplazan los esquemas de ARX, MLP, CNN1D, LSTM, PatchTST, N-HiTS y TFT por sus diagramas globales y detalles de celdas de Esquemas_arquitectura_modelos_Entregable2.pptx. Son 14 PNG de 3200×1800. Se conservan los cinco esquemas de TCN, iTransformer, TaTS, fusión tardía y TGForecaster, que no aparecen en ese archivo. Estado vigente: 12 páginas de modelos y 19 esquemas; 13 cuadernos, 97 celdas analíticas, 53 gráficas y 49 tablas.

El PPTX suministrado permanece intacto y se conserva una copia en results/esquemas_presentacion/fuente_original.pptx, con SHA-256 en manifest.json. Las diapositivas 15–16 describían TFT compacto de 16 unidades, 83 entradas y 281443 parámetros. Sólo su adaptación para las imágenes del JBook usa el TFT seleccionado: configuración1 (etiqueta T01), ancho160, 56 variables pasadas (30 numéricas+3máscaras+23calendario), 23 de calendario futuro, cuatro cabezas, 11715596 parámetros y MASE1,0510256114. La fórmula del embedding usa R^160. Se usó edición dirigida con artifact-tool; se restauró el contenedor del PPTX original para conservar la fidelidad de las formas y fórmulas durante la exportación. No se modifican pesos, hiperparámetros ni resultados. La adaptación es documental y está en results/esquemas_presentacion/esquemas_para_jbook.pptx, no es un experimento nuevo.

Fuente, diapositivas, imágenes y hashes: results/esquemas_presentacion/manifest.json. PNG de extracción: renderizados/; copia de uso en book/_static/arquitecturas_presentacion/. Las imágenes del JBook abren la versión de alta resolución al pulsarlas. Se integran interpretaciones específicas tras el esquema global y el detalle. Se retira en esas siete páginas el código Matplotlib que correspondía a las imágenes sustituidas; el código real de arquitectura, entrenamiento y prueba se conserva. El procedimiento de extracción se muestra una vez en fundamentos, con scripts/exportar_esquemas_presentacion.ps1. PowerPoint sólo se necesita para volver a exportar; la construcción de libro en Windows/Linux usa los PNG incluidos.

Persistencia: scripts/esquemas_presentacion.py instala y valida los activos sin llamar a PowerPoint, redes ni entrenamiento. Se invoca al finalizar crear_arquitecturas_e2.py y reorganizar_capitulo10.py, evitando que el generador reintroduzca diagramas sustituidos. Las cinco páginas sin reemplazo conservan su código de esquema. La revisión visual cubre las 14 diapositivas relevantes y la corrección final de TFT; el inventario se actualiza a 19 esquemas. Construcción local sin advertencias, cero errores y enlaces rotos; 86 pruebas correctas y cinco omitidas. HTML local en docs actualizado; sin entrenamiento ni publicación. El apagado de §27 correspondió al cierre de esa tarea; esta actualización no programa otro apagado.


## 29. Esquemas de las cinco familias restantes · 16 de noviembre de 2026

Estado vigente: doce familias con dos diagramas por familia, 24 esquemas en total. TCN, iTransformer, TaTS, fusión tardía y TGForecaster sustituyen sus cinco SVG genéricos por diez PNG de 3200×1800 con el estilo de la presentación: Cambria/Calibri, colores por operación, flujo global, detalle de celdas y cuatro paneles de ecuaciones. Cada página conserva las implementaciones de arquitectura y los pipelines de entrenamiento y prueba, e integra explicaciones tras ambos esquemas. Las imágenes abren su versión de alta resolución.

Los diagramas nuevos se construyen con formas, conectores y texto editables mediante scripts/crear_esquemas_ampliados.mjs; el PPTX resultante está en results/esquemas_presentacion/ampliados/esquemas_ampliados.pptx. La ficha de dimensiones y variantes está en ampliados/ficha_modelos.json, obtenida del índice principal de la consolidación, las configuraciones guardadas y results/evaluacion_e2/benchmark.csv. El código de redes contrastado se conserva en results/arquitecturas_e2/fuentes. La presentación suministrada y las catorce imágenes incorporadas en §28 permanecen intactas. El manifiesto registra fuente, diapositiva, dimensiones y SHA-256 de los 24 PNG y del generador nuevo.

TCN: tres bloques de dos convoluciones causales, kernel5, dilataciones1/4/16, 64 filtros, campo receptivo169h y lectura con calendario futuro. iTransformer: 33 tokens históricos y23 de calendario, dimensión128, dos capas y ocho cabezas. TaTS N+O: siete cierres diarios, resumen documental aprendido de12 atributos más presencia, 46 canales, PatchTST128/tres capas/16cabezas y dos lecturas; su controlN de64/dos/cuatro se identifica aparte. Fusión tardía: núcleo64/dos/cuatro, rama documental64→128→24, combinación aditiva con presencia y escalar aprendido sin acotación; ambas ramas se entrenan conjuntamente. TGForecaster N+O: dimensión64, dos capas temporales, una de mezcla y cuatro cabezas; memoria700tokens y cinco parches futuros de ocho horas con paso cuatro y pesos de solape. Su varianteN tiene dimensión 256, tres capas temporales y dos de mezcla, ocho cabezas; conserva descripciones estáticas de variables. Las cifras de parámetros y MASE de los subtítulos corresponden a esas variantes seleccionadas.

Reproducción documental: generar los diez diagramas con artifact-tool en Node, exportar mediante scripts/exportar_esquemas_presentacion.ps1 e instalar con scripts/esquemas_presentacion.py. El generador JS y la extracción PowerShell se muestran en fundamentos; la construcción del libro usa los PNG incluidos y no ejecuta redes ni requiere PowerPoint. El instalador ya está integrado en los dos generadores del capítulo10. No reaplicar los SVG genéricos sobre las páginas instaladas.

Se revisan visualmente los diez diagramas nuevos, sus conexiones, ecuaciones y dimensiones; el PPTX supera las verificaciones de integridad, geometría, tipografía e importación. Inventarios actualizados: 13cuadernos, 97celdas analíticas, 53gráficas, 49tablas y24esquemas. Se conservan pesos, semillas, configuraciones, datos y resultados. HTML local en docs actualizado, sin entrenamiento ni publicación. Esta tarea no programa un apagado.

Verificación de §29: 86 pruebas correctas y cinco omitidas; construcción sin advertencias, cero errores y enlaces locales rotos. Se cotejan las diez imágenes servidas por el HTML con sus SHA-256. Las advertencias de deprecación de pandas en las pruebas de calendario y fuentes externas preexistían y no afectan esta actualización documental.


## 30. Un único resultado numérico de PatchTST · 16 de noviembre de 2026

Se conserva adj_modelo6_20261112_1035__N como «PatchTST · N», con cinco semillas, MASE0,6725016176243331, MAE145003,38897045367kWh y RMSE194117,79554852837kWh. repo_patchtst_numerico_r1__N (MASE0,7256488817261028) deja de participar en la presentación y en la familia de contrastes; sus pesos, predicciones y fuentes no se eliminan. La elección por mejor rendimiento de prueba fue solicitada por el usuario después de observar los resultados y se documenta como retrospectiva. No describirla como selección a priori ni como resultado nuevo de entrenamiento. La configuración conservada sigue siendo el control numérico común de TaTS y fusión tardía, contabilizado una vez.

Estado vigente: 13procedimientos, 65realizaciones y dos referencias; 105parejas temporales y78entre semillas, tres contrastes textuales y72por horizonte. Se reconstruyen bootstrap, Holm, Friedman, Nemenyi, MCS y residuos desde predicciones guardadas, conservando exactamente las métricas individuales de los modelos restantes. FriedmanQ54,3824175824,p2,334748e-7 y diferencia críticaNemenyi8,159475334. Precisión de desarrollo: cinco pilotos externos (25métricas) y ocho reservas internas (40), dimensionados por separado. Son10parejas externas y máximo conservador76; el grupo interno mantiene28parejas y máximo4231. La ficha, los pipelines y ambos esquemas de PatchTST ahora describen la configuración18 del paqueteModelo6: 33canales, dimensión64, dos capas, cuatro cabezas, dropout0,1, contexto2825→128→24 y466480parámetros, sinRevIN interno.

Política: config/seleccion_patchtst.json; scripts/seleccion_patchtst.py la aplica en la reconstrucción estadística y documental. Se filtran metadatos, diseño, pilotos y curvas; se actualizan todas las figuras, tablas y descargas de la evaluación y se armonizan introducción, literatura, diseño y reproducción. El libro continúa con12fichas y24esquemas; los modelos con documentos siguen siendo procedimientos separados. Los diagnósticos crudos conservan sus50checkpoints y70realizaciones medidos; la vista vigente incluye45checkpoints ampliados, 65realizaciones y195corridas de latencia. No ejecutar redes para rehacer estas figuras: usar los CSV y predicciones existentes.

Respaldo previo de derivados: results/seleccion_patchtst/respaldo. Auditoría y resumen: results/seleccion_patchtst/auditoria.json y RESUMEN.md. Reproducción: estadistica_e2.py; crear_arquitecturas_e2.py; crear_capitulo_e2.py; construcción y verificar_jbook.py. Diagramas nativos de PatchTST en diapositivas11–12 del PPTX ampliado, exportados a ampliado_11/12.png y registrados porSHA-256. La presentación suministrada permanece intacta. Sin entrenamiento, inferencia de redes, publicación ni apagado.

Verificación de §30: 86 pruebas correctas y cinco omitidas; 13 cuadernos, 97 celdas analíticas, 53 gráficas, 49 tablas y24 esquemas. Construcción sin advertencias, cero errores y enlaces locales rotos. El HTML y la descarga del benchmark coinciden con la selección de13 procedimientos. Se confirma porSHA-256 que el panel y el benchmark base permanecen intactos. HTML local actualizado en docs; sin publicación.


## 31. Revisión de consistencia y completitud · 16 de noviembre de 2026

Se contrastan las pautas locales E1/E2 (34/65 páginas), narrativa, código y artefactos del JBook. Se corrigen Levene (p≈0,001526), los enlaces de literatura a MLP/LSTM/CNN1D, la descripción de controles textuales ya ejecutados y el alcance de diseños futuros, prensa y escenario estricto. Se explicitan extremos documentales inclusivos y los siete cierres de TaTS. MASE permanece primario, MAE/RMSE son las métricas complementarias del benchmark; el RMSE se calcula por semilla y después se promedia. Se distingue construir HTML con salidas conservadas de recalcular los análisis.

Persistencia: scripts/consistencia_completitud_jbook.py es la última armonización de scripts/seleccion_patchtst.py; actualizar_diagnosticos_jbook.py aplica la misma capa. revision_integral_jbook.py obtiene el p de Levene del protocolo vigente. El documento de diseño y su descarga identifican la ampliación propuesta; los resultados conservados no ejecutan la variante de prensa ni una compuerta acotada.

Estado comprobado: 13 procedimientos,65 realizaciones,105 parejas temporales,78 entre semillas, tres contrastes textuales y72 por horizonte;13 cuadernos,97 celdas,53 gráficas,49 tablas y24 esquemas. Se verifican60 referencias de código (52 archivos distintos),24 imágenes porSHA-256 y902 enlaces con fragmento en35 páginas. Se auditan9497 cierres de72h con límite64 y extremos inclusivos; no hay adelanto respecto de la disponibilidad declarada. Esto no certifica versiones publicadas históricamente. Memoria pico: soloN-HiTS tiene416907776bytes por semilla;FLOPs ausentes en13 procedimientos. Las predicciones y los pesos se conservan.

Completitud parcial E1/E2. Pendientes prioritarios: documentar la selección de los diez modelos principales existentes,ARIMA oETS según§7.4S5,réplica del paper guía con condiciones originales§5,robustez de3severidades y ablación de3componentes§4.6,potencia y presupuesto prospectivos,memoria/FLOPs,conciliaciónPRISMA y revisión independiente. Los controles de contenido con pesos fijos ya se completaron; no deben volver a describirse como pendientes, pero no sustituyen una ablación con núcleo e hiperparámetros idénticos. No iniciar automáticamente estos experimentos.

Recomendaciones: priorizar atribución controlada del texto con PatchTST común; reservar evaluación posterior al congelamiento efectivo; no usar datos2026 ya inspeccionados como prospectivos; cerrar trazabilidad bibliográfica y verificar una reconstrucción limpia enWindows/Linux. El mayor número retrospectivo de semillas no es una orden de entrenamiento ni potencia acreditada.

Informe vigente: REVISION_CONSISTENCIA_JBOOK.md y results/revision_consistencia_20261116/REVISION.md. Evidencias en auditoria.json,recursos_por_semilla.csv,pruebas.log,construccion.log yqa/. Verificación:86pruebas correctas y5omitidas;construcción sin errores ni advertencias,enlaces y anclas locales íntegros. HTML local actualizado;sin entrenamiento,inferencia de redes,publicación ni apagado.


## 32. Conjunto principal de diez modelos · 16 de noviembre de 2026

El usuario aclara que los diez modelos son los ya entrenados: ARX, MLP, LSTM, CNN1D, TFT, TCN, iTransformer, TaTS, fusión tardía y TGForecaster. Este es el conjunto principal acordado. PatchTST y N-HiTS se conservan como comparadores numéricos; TGForecaster tiene variantes N/N+O. De ahí los trece procedimientos y65realizaciones de las tablas, además de los dos ingenuos. No recomendar incorporar nuevas arquitecturas para completar un supuesto conjunto de diez adicionales. Se retira ese pendiente del JBook y del informe de revisión. Los restantes pendientes metodológicos se mantienen. Sin cambios de pesos, predicciones ni métricas, sin entrenamiento ni publicación.


## 33. Publicación del estado actual · 16 de noviembre de 2026

A solicitud del usuario se publica el HTML generado en main:/docs del repositorio Prediccion-Demanda-Electrica. Commit 95c492c65bccc2ceecdfb9da176d9506c5986f5a. GitHub Pages confirma estado built y respuesta HTTP200 en https://fabian-11037.github.io/Prediccion-Demanda-Electrica/intro.html. Se conserva el contenido vigente, con trece procedimientos y diez modelos principales. No se recalculan modelos ni se modifican pesos. La publicación solo actualiza docs; el checkout de publicación es la carpeta hermana Publicacion-Demanda-JBook-20261116. Recibo local: results/publicacion/jbook_20261116.json. No volver a publicar cambios posteriores sin una nueva solicitud.

## 34. Dashboard público en Render · 17 de noviembre de 2026

Se crea dashboard/ con aplicación estática independiente: Contexto, EDA y Modelos. URL: https://demanda-electrica-dashboard.onrender.com/index.html. Render confirma Live; servicio srv-dbuc2rec6qcc73da0mdg, despliegue dep-dbuc2rmc6qcc73da0nc0, código 1a4fffe5fddf8adeef7773fd9fc488c1eafed494. El acceso HTTP devuelve 200.

Utiliza los resultados congelados: diez modelos principales, trece procedimientos,65realizaciones,1461orígenes,105parejas temporales y78entre semillas. EDA de30series2000–2025, cobertura del canal oficial, pronósticos por fecha, benchmark, rangos/MCS, diagnósticos,24esquemas y enlaces a los pipelines del JBook. MAE/RMSE se agregan después de evaluar cada semilla; banda p10–p90 entre inicializaciones. No ofrece pronósticos para fechas nuevas.

El despliegue no requiere Python, pesos, GPU, claves ni servicio de pago. Raíz dashboard, publicación site, comando echo Dashboard preparado, SKIP_INSTALL_DEPS=true. Conexión por repositorio público y despliegue manual; no ampliar permisos GitHub. La exportación se ejecuta explícitamente con dashboard/scripts/export_data.py, nunca durante el despliegue. Trazabilidad en site/data/manifest.json. Contexto de operación en dashboard/CONTEXTO_DASHBOARD.md.

Comprobaciones:86pruebas correctas y5omitidas del proyecto;3pruebas específicas del dashboard correctas, incluida reproducción de las métricas de los13procedimientos. Revisión de navegación, filtros, límites de fecha, diagnósticos y diseño móvil/escritorio. No se entrenan modelos ni se modifican sus pesos, predicciones o configuraciones. El JBook conserva el contenido publicado.

## 35. Redacción académica y fuentes reproducibles · 17 de noviembre de 2026

El usuario solicita prosa propia de investigación, sin autoevaluaciones del cumplimiento del entregable ni aclaraciones editoriales innecesarias. La discusión conserva resultados, incertidumbre y amenazas metodológicas. La literatura se centra en demanda: se retiran las tablas y los contrastes empíricos de precio, y Time-MMD se cita como referencia metodológica externa. Cinco estudios empíricos de demanda sustentan el recuento de comparadores; la revisión y los resúmenes mantienen su función contextual. La introducción identifica trece procedimientos y dos referencias estacionales. Inventario vigente:51figuras,47tablas y24esquemas.

Se sincronizan book/, sus arquitecturas, TOC, recursos y scripts con el análisis actual. La sección estadística está integrada en10_evaluacion.ipynb; su fuente histórica se conserva en results/reorganizacion_capitulo10/fuentes, sin duplicar páginas. scripts/redaccion_investigacion.py aplica el criterio editorial al finalizar las reconstrucciones. No restaurar dictámenes sobre la entrega ni las tablas de precio retiradas.

scripts/reconstruir_evaluacion.py reejecuta las40celdas analíticas desde predicciones y mediciones conservadas. La instantánea d1ae9a4 conserva2019archivos de análisis no neuronales,57,5MB; se verifican hashes de metadatos y predicciones. Los checkpoints no se requieren para reconstruir figuras. Se incluyen results/evaluacion_e2, results/diagnosticos_ampliados y las fuentes y recursos de los generadores. .gitattributes conserva los bytes de las fuentes con hash al mover entre Windows/Linux.

scripts/reorganizar_capitulo10.py reconstruye la presentación integrada; la actualización completa del EDA es opcional con --actualizar-eda. La inferencia nueva requiere PyTorch y los binarios identificados en fuentes_diagnosticos.json; su raíz se configura mediante DEMANDA_ARTEFACTOS. La construcción documental utiliza las salidas incluidas y no requiere la carpeta hermana de consolidación. No presentar estos dos procesos como si tuvieran las mismas dependencias.

No se cambian pesos, arquitecturas, predicciones ni métricas de prueba; no se entrena ni se publica el JBook. El HTML público conserva su última versión hasta una solicitud de publicación.


## 36. N-BEATS y TimesNet · integración local del 20 de noviembre de 2026

### Composición vigente

El Top 10 comprende PatchTST, N-HiTS, TFT, TCN, iTransformer, TaTS, fusión tardía, TGForecaster, N-BEATS y TimesNet. Las tres redes de E1 (MLP, LSTM y CNN1D) y ARX se presentan aparte. La variante N de TGForecaster es un control adicional de su variante N+O. Son quince procedimientos aprendidos con cinco semillas (1000–1004), más los ingenuos diario y semanal. Esta composición sustituye los conteos de estados anteriores, incluido §32.

### Predicciones y métricas

N-BEATS conserva la configuración 16, 30 bloques, cuatro capas densas de 512 unidades por bloque, dropout 0,1 y 49.464.960 parámetros. TimesNet conserva la configuración 19, dos TimesBlock, cinco periodos, seis núcleos, dimensiones internas 16 y 328.266 parámetros. No se modifican checkpoints, escaladores, selección ni entrenamientos.

| Modelo | MASE | DE MASE entre semillas | MAE (kWh) | RMSE (kWh) |
|---|---:|---:|---:|---:|
| N-BEATS | 0.953342617 | 0.032285941 | 205557.736528 | 261699.235572 |
| TimesNet · lote uno | 0.642927818 | 0.013146216 | 138626.748294 | 181637.303953 |

Las métricas se calculan por semilla y después se promedian; el RMSE no es el de la predicción media. TimesNet es el menor error descriptivo y el único retenido por el MCS al 90 % bajo este protocolo. La evaluación temporal y la incertidumbre entre inicializaciones mantienen sus unidades separadas.

**TimesNet: fuente canónica obligatoria.** Para prueba se utiliza `results/reevaluacion_timesnet_lote1_20261120/predicciones_{semilla}.npz`, campo `yhat_lote1_kwh`. Su copia normalizada es `results/evaluacion_e2/entradas/adicionales/nuevo_timesnet/final/N/semilla_{semilla}/predicciones_canonicas.npz`. No utilizar el `pred_prueba.npz` nativo del paquete para el benchmark vigente: corresponde al lote agrupado. TimesBlock selecciona frecuencias promediando sobre el lote; lote uno evita que las historias de otros orígenes intervengan en el pronóstico. La validación, las curvas y el MASE de parada conservan los registros originales de desarrollo; no se reevaluó la selección.

N-BEATS utiliza las predicciones originales con etiquetas canónicas de doble precisión. Sus exógenas incorporan las últimas 24 horas; la demanda utiliza 168 horas. Ambos modelos comparten las 30 variables históricas del contrato numérico (clima con su rezago de diseño, precio e hidrología), 23 de calendario y tres máscaras de ausencia. Las escalas proceden exclusivamente del ajuste.

### Estadística, precisión y diagnósticos

Se recalculan 136 parejas temporales entre 17 métodos y 105 parejas entre semillas de los 15 procedimientos aprendidos. DM–HLN usa HAC de 13 rezagos y bootstrap circular de 14 días con 10.000 remuestras; Holm se aplica a cada familia completa. Friedman, Nemenyi, MCS, TOST, residuos y resultados por horizonte incluyen ambos modelos. Los tres contrastes textuales conservan su familia Holm(3) y sus 72 contrastes por horizonte. El margen relativo sigue siendo 3 %, α=0,05 y la potencia objetivo 0,80.

Cinco procedimientos conservan piloto externo; los otros diez incluyen precisión retrospectiva de la parada interna, ahora con 50 registros y 45 parejas. Las curvas contienen 75 realizaciones. No se presenta la parada interna como un piloto independiente ni se altera el número de semillas.

La importancia de los once procedimientos ampliados utiliza 192 fechas (días 1, 8, 15 y 22 de cada mes de 2022–2025), nueve grupos y tres desplazamientos circulares por año, cinco semillas y donantes de los 1.461 días. N-BEATS y TimesNet se incorporan mediante los pesos congelados; TimesNet usa lote uno en cada perturbación. Las cuatro referencias de demanda/calendario conservan su diagnóstico de importancia con 1.461 fechas y diez repeticiones. La sensibilidad documental corresponde a los tres modelos con documentos y no se aplica a las dos nuevas redes numéricas.

La latencia incluye los quince procedimientos: un hilo CPU, lote uno, 20 calentamientos y 100 mediciones en tres orígenes por semilla. Se conservan 225 resúmenes de medición. Se agregan la mediana de las 15 medianas y la mediana de sus p95; no es el p95 conjunto. Los archivos y el entorno están en `results/diagnosticos_ampliados/` y `results/integracion_top10_20261120/protocolo_diagnosticos.json`.

### Archivos y reproducción

- Entradas comunes: `results/evaluacion_e2/entradas/consolidado/predicciones_finales.parquet` y `resumen_principal.csv`, claves `nuevo_nbeats__N` y `nuevo_timesnet__N`.
- Configuraciones, escaladores, curvas y metadatos: `results/evaluacion_e2/entradas/adicionales/nuevo_{nbeats,timesnet}/`.
- Pesos originales: `results/integracion_top10_20261120/pesos/ENVIAR_pesos_NBEATS_TimesNet_20261119_2125.zip`; SHA-256 en su manifest y diez checkpoints en `auditoria_pesos.csv`. Conservar el ZIP como artefacto binario; no es necesario para construir el libro ni ejecutar el dashboard.
- `scripts/integrar_top10.py` normaliza entradas y agrega metadatos, precisión y diagnósticos. Mantiene copia previa en `results/integracion_top10_20261120/respaldo/`.
- `scripts/estadistica_e2.py` recalcula estadísticas de las predicciones conservadas; `scripts/reorganizar_capitulo10.py` reconstruye la evaluación e interpretaciones; `scripts/arquitecturas_top10.py` genera las fichas y los cuatro esquemas nuevos.
- `scripts/diagnosticos_top10.py` mide importancia y latencia mediante inferencia, sin ajuste. No ejecutarlo al reconstruir el libro: sus CSV ya están guardados.
- Las fichas `book/arquitecturas/nbeats.md` y `timesnet.md` incluyen fundamentos, esquemas y código de preparación, selección, optimización, parada temprana y prueba por origen. La evaluación integrada pasa a §10.16.
- `dashboard/scripts/export_data.py` exporta datos guardados y verifica sus métricas; el selector separa Top 10, baselines E1, referencia clásica y control numérico. `dashboard/site/` funciona de forma autónoma.

Para regenerar estadísticas y presentación, sin redes ni entrenamiento:

```console
python scripts/estadistica_e2.py
python scripts/reorganizar_capitulo10.py
python dashboard/scripts/export_data.py
jupyter-book build book --warningiserror --keep-going
python scripts/verificar_jbook.py
```

La construcción usa salidas guardadas (`execute_notebooks: off`). Las versiones locales de `book/`, `docs/` y `dashboard/site/` se actualizan; GitHub Pages y Render permanecen sin desplegar. No hacer push ni publicar sin solicitud expresa.

Verificación de la integración de §36: 89 pruebas correctas y 5 omitidas; tres pruebas del dashboard correctas. El inventario mantiene 51 figuras, 47 tablas y 28 esquemas. La construcción estricta del JBook finaliza sin advertencias y el verificador comprueba los enlaces locales. La revisión visual confirma TimesNet y N-BEATS en el selector, pronósticos, benchmark e importancia, sin errores de consola. No se publicaron cambios.

## 37. Consistencia documental y generación desde resultados · 20 de noviembre de 2026

Las inconsistencias de conteo, precisión, importancia y MCS se corrigen en los capítulos 9, 10 y 11. La revisión vigente se genera en `REVISION_CONSISTENCIA_JBOOK.md`; el capítulo 12 enlaza esa revisión y el resumen de procedencia.

`scripts/narrativa_resultados.py` es la fuente de presentación de conteos y conclusiones. Lee el benchmark, contrastes, MCS, metadatos, pilotos, parada, importancia y latencia. Las celdas Markdown se identifican con `resultado_documental` y no contienen código de generación de texto. `integrar_top10.ajustar_celdas`, `reorganizar_capitulo10.py`, `reconstruir_evaluacion.py`, `actualizar_diagnosticos_jbook.py` y la armonización final de `redaccion_investigacion.py` aplican esa fuente. No volver a corregir los números mediante sustituciones de palabras o cifras antiguas.

Estado derivado: 136 parejas temporales, 105 entre semillas, 5 pilotos externos y 10 procedimientos con precisión de parada (50 registros; 45 parejas). La importancia ampliada comprende 11 procedimientos; sus basales se distinguen de las 45 comprobaciones de reproducción completa ya conservadas. La latencia comprende 15 procedimientos y 225 registros de medición.

TimesNet mantiene validación y parada con lote 256, y prueba y perturbaciones con lote 1. La comparación desarrollo–prueba declara esa diferencia; los checkpoints, predicciones y métricas se conservan. Los vacíos metodológicos identificados en la revisión no se presentan como corregidos por esta actualización documental.

El verificador rechaza las celdas gestionadas si su contenido no coincide con los resultados, incluso si el libro construye correctamente. Las pruebas incluyen idempotencia, detección de párrafos obsoletos y cambios simulados de conteos, ranking y MCS. `results/revision_documental_20261120/resumen_documental.json` conserva los valores y huellas de origen. Regenerar con `python scripts/reorganizar_capitulo10.py`; construir con `jupyter-book build book --warningiserror --keep-going` y revisar con `python scripts/verificar_jbook.py`. Ninguno de estos pasos entrena ni despliega.

## 38. Esquemas detallados de N-BEATS y TimesNet · 20 de noviembre de 2026

Los cuatro esquemas (global y detalle de cada modelo) utilizan el estilo de las demás arquitecturas: leyenda por tipo de elemento, marcos de módulos, entradas y dimensiones explícitas, conexiones residuales y cuatro tarjetas matemáticas. Se generan PNG de 3200×1800 y SVG vectoriales.

`scripts/esquemas_top10_detallados.py` dibuja la arquitectura congelada y obtiene parámetros y MASE del benchmark y sus metadatos. N-BEATS representa las cuatro capas densas, las dos salidas de cada bloque, la resta de backcast y la suma de pronósticos. TimesNet representa normalización, embeddings, dos TimesBlock, cinco periodos por FFT, dos bancos Inception de seis núcleos, agregación, residuo y lectura con calendario.

`arquitecturas_top10.py` utiliza este generador; `reorganizar_capitulo10.fundamentos` conserva su código visible en el JBook. Las cuatro imágenes se sincronizan con `dashboard/site/architectures/`. La procedencia y comprobación de ajuste de textos están en `results/esquemas_presentacion/top10_detallados/manifest.json`; las versiones sustituidas se conservan en su respaldo. Los resultados numéricos y pesos permanecen intactos. No se publica esta actualización.


## 39. Revisión funcional de los pipelines · 20 de noviembre de 2026

A solicitud del usuario, se verificó el funcionamiento de entrenamiento de las 15 configuraciones seleccionadas mediante ajustes mínimos aislados: dos épocas, ocho orígenes de optimización, dos de parada y dos de validación de agosto–octubre de 2020. Las arquitecturas completas coincidieron con sus conteos de parámetros, recibieron gradientes finitos y actualizaron sus pesos. Se verificó restauración exacta de predicciones y escaladores en procesos independientes. TaTS, fusión tardía y TGForecaster N+O tienen gradientes no nulos en sus módulos de relevancia. Estas comprobaciones no son nuevos resultados del benchmark ni habilitan reentrenar la evaluación por iniciativa propia.

El punto de entrada vigente para futuros ajustes es `scripts/entrenar_arquitectura.py`, con dependencias en `requirements-entrenamiento.txt` y configuraciones seleccionadas en `config/entrenamiento_congelado.json`. Acepta `--modelo`, `--semilla`, `--anio` (2018–2021), `--dispositivo cpu|cuda|auto`, `--salida` y `--smoke`. La fase predeterminada es validación; prueba requiere `--fase prueba --allow-test` y autorización de evaluación final. Nunca sobrescribe carpetas. Ejecutar cada arquitectura en un proceso nuevo. `inferir` restaura el modelo y las transformaciones persistidas sin volver a estimarlas.

PatchTST utiliza Modelo6 N, configuración 18 y 466.480 parámetros. El adaptador `pipeline_ampliado_portable.py` delega en esta selección; no usar la configuración retirada P20. N-HiTS conserva N14 y sus 83 canales. Los vectores oficiales y descripciones congelados se copiaron dentro del repositorio a `data/processed/entrenamiento_texto/`; sus huellas están en `manifiesto.json` y la relación con el corpus en `procedencia.json`. Deben convertirse a float32. Los checkpoints del lanzador incluyen estos buffers, no persistentes en los state_dict originales.

Para futuras ejecuciones del lanzador, TimesNet conserva su lote de ajuste y usa un origen por llamada en parada, validación y predicción. No se alteraron las pérdidas, pesos o resultados de los artefactos históricos; su protocolo sigue registrado en el benchmark. Los lotes finales unitarios se unen al anterior, sin descartar etiquetas. Los demás hiperparámetros y arquitecturas seleccionados se conservan.

El código ejecutable y los comandos están incluidos en fundamentos (§10.1) y las catorce fichas. `scripts/documentar_entrenamiento.py` sincroniza las fuentes y lo llaman los generadores de arquitecturas y fundamentos. La construcción del libro no entrena. Informe y evidencias: `results/auditoria_entrenamiento_20261120/INFORME.md`, `verificacion_final/resumen.json`, `restauracion_sin_reajuste.json`, `fuentes_verificadas.json` y `cierre.json`. Los checkpoints diagnósticos se eliminan tras verificar la restauración; no se mezclan con los pesos del proyecto.

Validación física realizada en Windows y CPU (PyTorch 2.9.1+cpu). CUDA y Linux quedan disponibles por el diseño portable del lanzador, pero no fueron ejecutados en esta máquina. No declarar validación de GPU o Linux como realizada. Se conservaron los 28 archivos numéricos de evaluación verificados por SHA-256. No se entrenó con prueba, no se sobrescribieron pesos originales y no se publicó.

Cierre de verificación: 106 pruebas del proyecto correctas (6 omitidas en el entorno documental), 12 pruebas del módulo con Torch correctas, construcción estricta del JBook sin advertencias y sin enlaces locales rotos.


## 40. Revisión final de completitud y consistencia · 20 de noviembre de 2026

Se contrastaron capítulos, figuras, tablas, CSV descargables, resultados, dashboard y código con las pautas E 1/E 2. Se corrigieron conteos obsoletos en los capítulos 1 y 12, las etiquetas de descarga y las secciones de segmentos, residuos, variación y atípicos. La precisión de ajuste/parada comprende cuatro familias base y diez ampliadas: 69 de 70 realizaciones tienen mayor error de parada; N-HiTS conserva curvas sin esas métricas desagregadas. Se eliminaron párrafos repetidos en literatura y validación.

La gráfica de error por horizonte omitía N-BEATS y TimesNet aunque sus filas existían en el CSV. Ahora construye sus grupos desde el benchmark y verifica cobertura exhaustiva sin duplicados. Se recalculó únicamente esa figura con resultados guardados. El menor MASE por horizonte corresponde a TimesNet en 16 horas, PatchTST en 6 y TaTS en 2; es un recuento descriptivo. Las demás figuras y tablas conservan los subconjuntos que corresponden a sus entradas o fuentes de precisión.

`scripts/coherencia_documental.py` complementa la narrativa canónica con coberturas, interpretaciones y deduplicación. Los capítulos 1 y 12 se incluyen en la verificación de celdas gestionadas. `scripts/auditar_completitud.py` comprueba resultados, alcance de horizontes, CSV, dashboard y configuraciones; no ejecuta redes. Los conteos del motor de diagnósticos se derivan del registro en lugar de constantes fijas. La reconstrucción sincroniza su código visible.

Las evidencias están en `results/revision_final_20261120/auditoria.json` y `pautas_fuentes.json`; `REVISION_CONSISTENCIA_JBOOK.md` incluye comprobaciones y pendientes metodológicos. Estos últimos siguen siendo preguntas/H 0/H 1 explícitas, trazabilidad de revisión sistemática y priorización, preregistro e independencia de prueba, potencia y presupuesto, réplica del paper guía, búsqueda aplicable a diseños no predefinidos, robustez por severidad, ablaciones, cobertura predictiva, techo de desempeño y costos incompletos. La diferencia de lotes de TimesNet en los resultados conservados sigue documentada.

La revisión conserva 15 procedimientos,75 realizaciones,2 referencias,136 parejas temporales, 105 entre semillas,51 gráficas,47 tablas y 28 esquemas. No se modificaron los 28 archivos numéricos contrastados por SHA-256. No se entrenó, no se ejecutó inferencia neuronal y no se publicó.

Cierre de §40: 14 controles de cobertura correctos, 111 pruebas del proyecto correctas (6 omitidas en el entorno documental), 3 pruebas del dashboard correctas y 11 pruebas finales de narrativa/cobertura correctas. Construcción estricta sin advertencias ni enlaces locales rotos. El informe completo se conserva en `results/revision_final_20261120/INFORME.md`.



## 41. Ampliación documental oficial · 2026-11-21

El corpus activo es `corpus_oficial_ampliado_20261120`: 625,577 registros, con 50,104 documentos nuevos de IDEAM, CREG, XM y Presidencia. Se conservan sin cambios todas las filas anteriores, la prensa y el panel numérico. La instantánea anterior se encuentra en `data/processed/versiones/corpus_oficial_20261109/`. Esta sección sustituye los conteos operativos de §13 para futuras adquisiciones, sin alterar la trazabilidad histórica.

| Fuente | Nuevos | Con cuerpo | Disponibles en declarado 2000–2025 | Cuarentena |
|---|---:|---:|---:|---:|
| CREG | 3706 | 3544 | 2768 | 938 |
| IDEAM | 45199 | 45003 | 40201 | 4990 |
| Presidencia | 773 | 772 | 753 | 20 |
| XM | 426 | 425 | 143 | 283 |

### Cobertura y disponibilidad

Los 312 meses de 2000–2025 están auditados con ceros explícitos por fuente y escenario. El informe completo y los CSV están en `results/ampliacion_documental_20261120/INFORME.md`. Los archivos `cobertura_mensual_fuente.csv`, `cobertura_anual_fuente.csv`, `cobertura_particiones.csv` y `cobertura_oficial_anual_cuerpo.csv` distinguen disponibilidad, cuerpos, titulares, días y ventanas de24/72/168horas. `inventario_por_periodo_referido.csv` describe el periodo del informe y no su disponibilidad.

IDEAM incorpora paquetes históricos de ITD, BCH y alertas oficiales BADT, BAICV y BAH, con la fecha de edición identificada por documento y su base registrada. Esas familias mezclan observaciones, alertas y pronósticos; no equivalen al boletín específico de pronóstico a tres días. No se considera cerrado el hueco2000–2009 ni el pronóstico a tres días anterior a2024 por recuperar otros tipos de boletín. Los fallos del archivo histórico y Wayback quedan registrados.

CREG incluye resoluciones definitivas y la numeración por prefijos. Se verifica la correspondencia de número y año entre el acto y su PDF enlazado; si no se corrobora, se conserva el PDF como evidencia y se utiliza el HTML depurado. Se priorizan PDF originales; los HTML jurídicos sin original excluyen notas posteriores y artículos sustituidos y se marcan parciales. Los decretos provienen de catálogos oficiales CREG y Cancillería: no son un inventario exhaustivo de toda norma nacional o local. Los actos replicados se deduplican por número y año. El OCR se limita a cuatro primeras páginas, con alcance y alertas explícitas.

XM añade informes mensuales Sinergox y anuales del SIN y mercado. La modificación por migración en agosto de2026, el mes referido y la creación del PDF no se utilizan como fechas de publicación. Los documentos sin fecha editorial sustentada quedan en cuarentena, conservando archivo y procedencia. No presentar la recuperación física como cobertura temporal utilizable.

El declarado sigue siendo el dataset oficial. Una fecha sin hora se incorpora desde el día siguiente a las00:00 de Colombia. Para actos jurídicos se usa la publicación del Diario Oficial o la fecha posterior de disponibilidad web expresamente declarada. Cuando los gestores discrepan se utiliza la más tardía y se conserva la alerta. No se usa la expedición del acto como publicación. Los PDF de paquetes no heredan la fecha del ZIP. Las fechas de encabezado y nombre se conservan separadas; si discrepan en un PDF histórico, queda en cuarentena hasta revisión. Una discrepancia de año con la carpeta del paquete sin correspondencia entre encabezado y nombre también activa cuarentena; la carpeta nunca se usa para redatar el documento. Se eliminan caracteres de control de fuentes PDF mal mapeadas, identificando el alcance parcial; si no queda texto legible, se aplica OCR.

La disponibilidad estricta nueva es la recuperación efectiva en2026; no se retrodata, por lo que no aumenta el estricto2000–2025. La versión histórica de los bytes permanece sin verificación para el escenario declarado. Los registros en cuarentena se conservan para auditoría y el cargador los excluye.

### Contrato para futuros entrenamientos

Se mantienen tres variantes comparables: numérica; numérica+oficial; numérica+oficial+prensa. Los canales oficial y prensa siguen separados, con indicadores de ausencia y relevancia aprendida por el modelo. No se introduce clasificación temática ni selección manual de relevancia. Las variables numéricas comunes conservan clima, hidrología y precio con sus rezagos; el reanálisis no se utiliza como clima conocido del futuro.

**La evaluación vigente conserva su corpus congelado de14.162 documentos** en `results/evaluacion_e2/entradas/texto/corpus_oficial.parquet`, sus vectores y los resultados de los15procedimientos. Los registros nuevos no corresponden a filas de los vectores congelados. Un entrenamiento posterior autorizado debe construir un nuevo caché y vectores versionados desde el manifiesto actualizado, auditar disponibilidad y ventanas y conservar la correspondencia entre id y vector. No volver a codificar ni entrenar automáticamente. Antes del caché nuevo, revisar la densidad y el truncamiento a64documentos de72horas en cobertura_particiones.csv, fijar el presupuesto común y auditar la selección sin temas ni relevancia manual.

Reproducción de esta adquisición: `scripts/adquisicion/ampliar_corpus_20261120.py`, `integrar_ampliacion_20261120.py`, `cerrar_ampliacion_20261120.py` y `completar_ampliacion_20261120.py`. Extracción nativa con PyMuPDF1.26.5; hojas con openpyxl; OCR español con Windows.Media.Ocr y `.ocr_env`. Los paquetes validan CRC y los archivos conservan SHA-256. El manifiesto activo y el diccionario incluyen procedencia, alcance, versiones y reglas. Se preservan originales locales y fallos de acceso; no se afirma exhaustividad donde quedan meses o familias sin documentos.

Verificación final del proyecto: 128 passed, 6 skipped; registro `results/ampliacion_documental_20261120/pruebas_finales.log`. Los controles están en `conservacion_resultados.json`, `validacion_cobertura.json`, `calidad_extraccion.csv`, `documentos_cuarentena.csv` y `cierre.json`. No se entrenan modelos, no se regeneran embeddings ni se evalúa demanda. No se modifica ni publica el JBook o dashboard en esta adquisición.


## 42. Diagnóstico temporal del texto · 2026-11-21

El procedimiento vigente para este diagnóstico es `scripts/diagnostico_texto_temporal.py`; sustituye la selección interna LOO de los scripts conservados en `docs/Claude outputs/`. Estos últimos y sus CSV se mantienen como entradas históricas de comparación. No usar sus etiquetas de «puntos de MASE» para las diferencias de porcentajes.

Se reutilizan ocho NPZ de validación archivados de E cfg07 y TimesNet cfg19, realización identificada como semilla 43. Historial inicial 2018; evaluación externa 2019–2021 (1096 orígenes diarios). Cuatro particiones internas expansivas, escalado por partición, purga de dos días interna y externa, 25 valores de alpha y selección por MSE del residuo relativo. Vocabulario y SVD anteriores a 2018; representaciones, truncamiento y factor de corrección diario conservados. No se cargan redes ni se usan etiquetas o predicciones de prueba 2022–2025.

La reducción textual se calcula como 100·(MASE_C+R−MASE_texto)/MASE_C+R. Las diferencias anteriores frente al porcentaje de mejora sobre la base original se etiquetan como puntos porcentuales y se convierten al denominador común en `comparacion_anterior.csv`. IC95 puntuales: 9.999 remuestreos pareados por origen, bloques de 14 días dentro de cada año. Holm global sobre 110 contrastes: 54 de desempeño, 54 de magnitud del error y 2 correcciones numéricas. Los 296 desplazamientos placebo se ajustan con Holm por separado; su resolución mínima no permite rechazo con 54 comparaciones, por lo que no se interpreta ese resultado como ausencia de señal.

| base | MASE_sin_correccion | MASE_C_R | reduccion_relativa_pct | ic95_inferior | ic95_superior | p_bootstrap_holm |
| --- | --- | --- | --- | --- | --- | --- |
| E | 0.5400 | 0.5356 | 0.8085 | -0.8161 | 2.3414 | 1.0000 |
| TimesNet | 0.5566 | 0.5360 | 3.6889 | 2.4187 | 4.9209 | 0.0110 |

Resultados textuales: 9 de 54 mejoras puntuales; 0 tras Holm; 0 alcanzan 3 %. Mayor mejora puntual: oficial_congelado · E · AN, 0.3752 %, p ajustado 0.2725. El incremento máximo de R² de magnitud es 0.059461; 0 incrementos significativos tras Holm. Estas conclusiones se limitan a las representaciones y la corrección diaria examinadas, con una realización archivada de cada base.

Informe y resultados: `results/diagnostico_texto_temporal_20261121/INFORME.md`, `metricas.csv`, `correccion_numerica.csv`, `correccion_numerica_anual.csv`, `magnitud_error.csv`, `particiones_y_alpha.csv`, `placebos.csv`, `verificacion.json` y `manifiesto.json`. `scripts/informar_diagnostico_texto.py` genera el informe y esta sección desde los CSV; no fijar manualmente sus conteos. Los ZIP compactos `entradas_E.zip` y `entradas_TimesNet.zip` conservan exactamente los ocho NPZ usados, sin depender de Descargas para repetir el cálculo.

Verificación: 132 pruebas correctas y 6 omitidas; registro `pruebas.log` en la carpeta del diagnóstico. Se conservan 37 archivos de entrada y resultados previos por SHA-256. El corpus activo, el corpus congelado de 14.162 documentos, sus vectores y el benchmark de 15 procedimientos permanecen intactos. No se reentrenan las redes, no se modifica el JBook ni el dashboard y no se publica.


## 43. Dashboard sincronizado · revisión del usuario · 21 de noviembre de 2026

El dashboard local coincide con las once tablas del benchmark, sus 15 procedimientos, 75 realizaciones y 28 esquemas del JBook. Se añaden 14 fichas Markdown descargables con las inclusiones de código Python resueltas, interpretaciones tomadas de las fichas y precisión de desarrollo por fuente. Los enlaces de pronóstico abren la ficha interna, evitando las páginas de N-BEATS y TimesNet todavía ausentes en GitHub Pages. Se verificaron 132 pruebas del proyecto (6 omitidas) y 5 pruebas del dashboard. Los artefactos previos, pesos y resultados se conservaron; no se modificó ni publicó el JBook.

La publicación en Render fue autorizada inicialmente. El commit `ba891c0c8c307f466b6dd875dc539e9826629493`, limitado a `dashboard/`, ya se había subido a GitHub cuando el usuario indicó «no publiques aún» y pidió revisar. **No se activó el despliegue manual de Render. No desplegar hasta nueva autorización.** Se comprobó que Render conserva un único despliegue Live del commit `1a4fffe5fddf8adeef7773fd9fc488c1eafed494`.

Vista para revisión: `http://127.0.0.1:8068/`, servidor local de `dashboard/site`. Copia aislada de publicación: `../Publicacion-Dashboard-20261121/`. Evidencias, pruebas y estado: `results/sincronizacion_dashboard_20261121/`; `estado_publicacion.json` registra el commit subido y el despliegue detenido.


Resumen final de Modelos solicitado durante la revisión: se añade una tabla comparativa de los 15 procedimientos con periodo y MASE de validación de selección, MASE de prueba y enlace a la evaluación del JBook. La exportación obtiene validación de metadatos_procedimientos.csv y prueba de benchmark.csv. El resumen permanece al final de todas las vistas de Modelos. Cambios únicamente locales, posteriores al commit ba891c0; no hacer push ni desplegar hasta nueva autorización.


## 44. Publicación conjunta y paquete de ejecución · 22 de noviembre de 2026

El usuario autorizó publicar el JBook y el dashboard, sustituyendo la pausa de revisión de la sección anterior. Commit publicado: `bc2b8b382bcb8641cb27e3bca5f2269a7b45afa7`; Render: `dep-dc17gabl550s73dm0gu0`. El JBook se sirve desde `main:/docs`; el dashboard se despliega manualmente desde `dashboard/site`. Se publicaron también las fuentes actuales de `book/`, la guía `EJECUCION_LOCAL.md`, `ejecutar.py` y `requirements-jbook.txt`. El dashboard incluye el resumen de validación y prueba al final de Modelos y el enlace al JBook.

Se comprobó que intro, N-BEATS, TimesNet y evaluación publicados coinciden con los HTML preparados; app.js, core.js, index.html y study.json públicos coinciden con el dashboard actualizado, normalizando únicamente finales de línea. El JBook se construye sin ejecutar cuadernos, con sus salidas guardadas. Las pruebas del proyecto dieron 132 correctas y 6 omitidas; las cinco del dashboard son correctas.

Paquete mínimo: `results/publicacion/Demanda_Electrica_JBook_Dashboard_minimo_20261122.zip` (33.274.128 bytes). Incluye fuentes y recursos necesarios para reconstruir el libro con sus salidas guardadas, dashboard estático con datos exportados, instrucciones Windows/Linux, lanzador y manifiesto SHA-256. No incluye datos de investigación completos ni pesos. Se extrajo y reconstruyó en una carpeta limpia: sin enlaces locales rotos, 254 archivos con integridad comprobada y dashboard accesible por HTTP con los 15 procedimientos. En Windows conviene extraer en una ruta corta para evitar el límite de longitud de archivos.

Registro de comprobaciones, URLs y hash del ZIP: `results/publicacion/publicacion_20261122.json`. No se entrenaron modelos ni se alteraron resultados científicos.


## 45. Corpus activo disponible en el repositorio · 22 de noviembre de 2026

Se actualiza `data/processed/corpus_texto.parquet` con la versión `corpus_oficial_ampliado_20261120`: 625.577 registros de 13 fuentes, 50.104 incorporaciones y sin identificadores duplicados. El Parquet se conserva byte a byte (159.086.185 bytes; SHA-256 `e82f1b0fccc09ffb4deb9bd8c2dfebb7bedcc3cabc2c0fd355b2ae1d2413898c`) y se publica mediante Git LFS. Para recuperar sus bytes tras clonar: `git lfs install` y `git lfs pull --include="data/processed/corpus_texto.parquet"`. Un puntero LFS no es el dataset completo.

Se sincronizan el manifiesto activo, el diccionario y las auditorías de cobertura en `results/ampliacion_documental_20261120/`. `data/processed/README_CORPUS.md` documenta descarga, integridad y uso. Las fechas y alertas se conservan; declarado continúa como escenario principal y estricto como sensibilidad. El campo de publicación del manifiesto original corresponde al cierre de la adquisición; esta sección registra su posterior carga al repositorio.

No cambian el panel numérico, corpus congelado, pesos, vectores ni resultados. No se entrenan modelos ni se despliega Render; el JBook publicado conserva su contenido. Los próximos entrenamientos autorizados deben construir y versionar nuevos vectores desde este corpus ampliado.


## 46. Corrector semántico E3 en GPU · 22 de noviembre de 2026

Se ejecuta el paquete Corrector_Semantico_E3 con el corpus ampliado publicado mediante Git LFS (SHA-256 e82f1b0f…), en GPU AMD y PyTorch/ROCm. Se completan 156 ajustes: 33 anuales de E y 123 del corrector, semillas 101–103. La selección utiliza 2018 y la evaluación de desarrollo 2019–2021 (1.096 días). La prueba 2022–2025 no se ejecuta ni se presenta como evaluada en esta corrida.

Resultados en `results/corrector_semantico_e3_20261122/`: informe, contrastes con bootstrap por bloques y Holm, métricas por año y hora, predicciones, historiales, correcciones por fecha y semilla, plan, fuentes y manifiesto SHA-256. El MASE de C+R es 0,54671; corrector numérico 0,54186; oficial reciente 0,54169; oficial alineado 0,54397; oficial y prensa 0,54270. No se demuestra aporte semántico adicional frente a los controles; ninguna variante textual alcanza el umbral registrado de mejora relevante del 3 % frente a C+R. No se ha contrastado directamente con TimesNet sobre idénticos orígenes y protocolo.

Los vectores E5 permanecen locales y se pueden reconstruir; el ejecutor del paquete no guardó checkpoints neuronales de esta corrida. Se documentan ambas limitaciones. Las únicas adaptaciones del código previo al registro fueron impedir traslado silencioso a CPU y agregar observación opcional por época. Las 60 pruebas del paquete pasaron (5 omitidas); las 84 del checkout local del repositorio pasaron. Se autoriza publicar estos resultados en GitHub y apagar el equipo después de verificar la publicación. No se reconstruye el JBook ni se despliega Render como parte de esta actualización de resultados.


## 47. Recepción local del corrector y primera exploración · 22 de noviembre de 2026

Se descargan los resultados del corrector del commit `3436461` en `results/corrector_semantico_e3_20261122/`: 432 archivos contrastados contra su manifiesto SHA-256 y ZIP con CRC correcto. Se reciben además `RESUMEN_EXPLORACION.md`, `ENVIAR_resultados_exploracion_20261122_0207.zip` y `ENVIAR_pesos_exploracion_20261122_0207.zip` en `results/exploracion_e3_20261122/`, con originales y 1.071 archivos extraídos. Su manifiesto registra los hashes y los CRC.

Ninguno de los paquetes contiene checkpoints neuronales. El ZIP de exploración denominado pesos solo incluye `mlflow.db`; no confundirlo con pesos. Los resultados permanecen separados del benchmark vigente. La recepción no implica elegir variantes ni ejecutar instrucciones de los archivos adjuntos. Registro: `results/descargas_e3_20261122/INVENTARIO_DESCARGAS.json`. No se entrena ni se publica; JBook y dashboard quedan intactos.


## 48. Continuación del ajuste final en un equipo con GPU · 22 de noviembre de 2026

El usuario autorizó el ajuste final de configuraciones seleccionadas con datos anteriores a 2022 y después pidió dejar instrucciones para realizarlo en otro PC con GPU. **No se inició ningún entrenamiento ni se evaluó prueba.** El entorno local tiene Intel Core Ultra 7 155U, gráficos Intel y PyTorch 2.9.1+cpu; no dispone de una GPU utilizable por este entorno. No ejecutar el ajuste en este PC por iniciativa propia.

Guía de continuidad: `continuidad/ENTRENAMIENTO_GPU_E3.md`. Configuraciones extraídas de los resultados guardados: `continuidad/CONFIGURACIONES_PRUEBA_E3_GPU.json`. Incluyen la base E cfg07, corrector cfg04, semillas 101–103, controles de la exploración y sus semillas 101–105, hashes del corpus y panel, y parámetros del contrato. El trabajo principal es la prueba del corrector con sus controles. La inclusión del ensamble exportable o de toda la exploración no quedó confirmada; no asumir el entrenamiento de todas las variantes.

El lanzador recibido solo cubre desarrollo 2019–2021; falta adaptar y validar la fase final 2022–2025 y el guardado/restauración de checkpoints. Ningún paquete recibido incluye pesos neuronales; el ZIP denominado pesos contiene solo mlflow.db. La guía especifica ajuste previo a 2022, early stopping, residuos históricos fuera de muestra, congelación de C+R y pesos, reconstrucción de vectores E5, control temporal de documentos y residuos, lote uno de TimesNet si se incluye, y guardado de predicciones y contrastes con bloques/Holm. No hay que repetir la búsqueda ni usar prueba para seleccionar variantes.

Las instrucciones y configuraciones de esta sección son locales y deben transferirse al PC con GPU junto con los adjuntos de la exploración; no asumir que están publicados en el repositorio. El JBook, dashboard, corpus y resultados previos permanecen intactos. Compartir resultados antes de modificar o publicar los sitios.


## 49. Diseño de cuatro modelos numéricos y con texto para GPU · 22 de noviembre de 2026

A solicitud del usuario se documentan cuatro diseños, sin entrenar ni publicar: **M1**, ensamble de PatchTST, E y codificador periódico supervisado por complementariedad de errores; **M2**, separación de energía diaria y distribución horaria; **M3**, extensión de M1 con selección de expertos condicionada por canales oficial y prensa; **M4**, memoria de episodios numéricos y textuales con residuos horarios fuera de muestra. Son propuestas de investigación pendientes de implementación y contraste de originalidad, no modelos ya entrenados ni ganadores seleccionados.

Especificación canónica: `continuidad/DISENO_CUATRO_MODELOS_GPU.md` y `continuidad/DISENO_CUATRO_MODELOS_GPU.json`. Complementan la guía del corrector de §48, sin sustituir las configuraciones históricas de `CONFIGURACIONES_PRUEBA_E3_GPU.json`. El JSON contiene arquitecturas, fórmulas, ocho candidatos por diseño, controles y ablaciones, configuraciones de expertos extraídas de resultados, cinco semillas 101–105, early stopping, hashes, políticas temporales y memoria, y 14 parejas estadísticas con 28 hipótesis globales para Holm.

El procedimiento futuro exige OOF anual de expertos y OOF anidado de M1, con selección por prefijo histórico, antes de ajustar las capas superiores. No usar una configuración elegida con años futuros para producir retrospectivamente filas supuestamente independientes. Desarrollo 2018–2021; ajuste final de las capas nuevas hasta 2020 y parada 2021; prueba 2022–2025 solo con configuraciones y pesos congelados. MASE y escala semanal del contrato, margen relativo 3 %, bloques de 14 días y separación de incertidumbre temporal e inicialización. La memoria y los rasgos de errores solo reciben observaciones ya disponibles, con retraso de dos días; no actualizan pesos durante prueba. El texto conserva escenario declarado, canales separados y relevancia aprendida.

La autorización de esta actualización es exclusivamente documental. No se implementó ni ejecutó un lanzador nuevo, no se inició entrenamiento, no se abrió una evaluación de prueba y no se modificaron JBook, dashboard, corpus ni resultados anteriores. Las instrucciones son locales y deben transferirse al PC con GPU; no asumir que ya están en GitHub. Una ejecución futura requiere nueva solicitud expresa. Presentar primero los resultados antes de modificar o publicar los sitios.


## 50. Documentación GPU incorporada al repositorio · 22 de noviembre de 2026

El usuario autoriza subir al repositorio los contextos y los archivos de diseño para continuar en otro PC con GPU, **sin entrenar ni publicar JBook o dashboard**. La entrada es `continuidad/README_GPU.md`; se incluyen `DISENO_CUATRO_MODELOS_GPU.md`, `DISENO_CUATRO_MODELOS_GPU.json`, `ENTRENAMIENTO_GPU_E3.md`, `CONFIGURACIONES_PRUEBA_E3_GPU.json` y la comprobación documental, además de ambos contextos. Esta sección actualiza las indicaciones de §48–§49 sobre la ubicación exclusivamente local de esas guías: ahora forman parte de la entrega en GitHub.

Las configuraciones completas de los expertos están incluidas en los JSON. Los adjuntos originales de la exploración no forman parte de esta subida: transferir `results/exploracion_e3_20261122/` por separado si se requieren sus predicciones, historias o archivos de procedencia. El corrector y sus fuentes ya están en el repositorio. Los cuatro modelos propuestos siguen pendientes de implementación y validación; no se ha creado ni ejecutado su lanzador. Los nuevos vectores y checkpoints no se incluyen.

No se actualizan carpetas book/, docs/ o dashboard/, no se entrena ni se ejecuta prueba y no se activa despliegue manual alguno. Los datos y resultados protegidos permanecen intactos. Para continuar, leer la entrada GPU y solicitar la ejecución futura antes de iniciarla.


## 51. Cola GPU del corrector final y M1 · 22 de noviembre de 2026

El usuario solicita ejecutar la evaluación final del corrector y después entrenar M1; autoriza apagar al finalizar. Lanzador local: `scripts/ejecutar_corrector_final_y_m1.py`; adaptadores separados `e3_gpu_comun.py`, `e3_corrector_final.py` y `e3_modelo1.py`. La instancia activa y su salida se consultan en `results/COLA_CORRECTOR_M1_ACTIVA.json`; no iniciar duplicados. Barra local: http://127.0.0.1:39976/progreso.html. Entorno mlenv, PyTorch 2.9.1+ROCm6.4, GPU AMD de 12 GB; se verifica retropropagación real.

Corrector: E cfg07, cfg04, tres semillas101–103; ajuste del corrector2012–2020, parada2021 y prueba2022–2025. Se guardan y restauran36checkpoints antes de la predicción de prueba; C+R se congela con alpha2021 por semilla, y los errores usados como rasgos llevan retraso de dos días. Se conservan variantes, cinco desplazamientos y sensibilidad estricta. El texto ampliado reutiliza los vectores anteriores a2022, congela fuentes y comprueba invariancia del histórico y disponibilidad temporal.

M1: expertos PatchTST·control, E·control y periódico con las configuraciones publicadas; OOF anual2011–2021 y cinco semillas101–105. Se registran160ajustes de búsqueda (8×5×4),20del control sin auxiliar y ajustes finales previos a2022. La cola resuelve401tareas neuronales, incluidas las reutilizaciones verificadas; no equivale a401arquitecturas. M1 selecciona solo con desarrollo2018–2021 y guarda pesos finales con parada2021. No se predice prueba de M1 en esta solicitud. El OOF anidado de M1 necesario para M3/M4 permanece pendiente; no usar predicciones en muestra en su lugar.

El registro previo guarda fuentes, planes, semillas y hashes. El hash Windows del registro de variables coincide al convertir exclusivamente LF aCRLF; se registra la equivalencia y se conservan los bytes locales. El primer inicio se detuvo antes de entrenar por esa diferencia. Una corrección de instrumentación homogeneiza `no_grad` en la comprobación de restauración; se conserva y reanuda el estado de la última época completa, sin alterar arquitectura, pérdida, semillas ni presupuesto. Los checkpoints guardan estado del optimizador y generadores aleatorios.

La autorización de apagado solo se aplica al éxito de ambas etapas y al guardado del manifiesto final. En caso de error se conserva el equipo encendido. Esta ejecución no publica GitHub, JBook ni dashboard.

## 52. Prueba final retrospectiva de M1 · 22 de noviembre de 2026

Tras completar la cola de §51, el usuario solicita expresamente la prueba de M1. Se ejecuta `scripts/probar_m1_gpu.py` con los checkpoints finales de `results/corrector_final_y_m1_gpu_20261122_150345/M1/`, sin reentrenamiento, búsqueda, ajuste de escaladores ni actualización de parámetros. Configuración elegida en desarrollo: 8, ancho64, dropout0,1, lambda de complementariedad0,1. Se restauran en GPU los tres expertos y el router para las cinco semillas101–105.

Evaluación2022–2025: 1.461 orígenes diarios y24 horas por origen; q168=215617,90361589016kWh. MASE medio: NNLS0,620904; M10,634923; media0,643461; convexa fija0,647729; periódico0,677992; PatchTST0,683149; E0,738865. M1 mejora7,0594% frente a PatchTST, seleccionado como mejor individual en desarrollo; IC95 puntual[5,7079%,8,4460%]. Frente a NNLS, su error aumenta2,2579%; IC95 de reducción[−2,9294%,−1,5787%]. NNLS obtiene el menor MASE de esta evaluación.

Bootstrap pareado por origen,9.999 réplicas, bloques14 días dentro de cada año y sensibilidades7/28. Se promedian errores entre semillas antes del remuestreo. Se conserva la familia global de28 hipótesis: cuatro evaluadas para M1 y24 pendientes. El informe proporciona una cota superior conservadora de Holm completando las pendientes con unos; no los interpreta como p-valores observados. Para M1 frente a PatchTST, tanto mejora>0 como>3% tienen p unilateral0,0001 y cota ajustada0,0028 con bloques14. No se afirma equivalencia por ausencia de significancia.

Los rasgos replican bit a bit la receta del entrenamiento y verifican invariancia a objetivos presentes y futuros. Los errores entran con retraso de dos días, inicializados con OOF2011–2021; solo se incorporan resultados de prueba ya disponibles. Se conserva por SHA-256 la totalidad de los artefactos de entrada usados. Las35 métricas MASE por procedimiento y semilla se comprobaron contra las predicciones guardadas. El diseño fue informado por el benchmark conocido: la prueba es retrospectiva, no un holdout nunca consultado. La ablación sin auxiliar no tiene checkpoint final y no se reentrena aquí.

Resultados locales: `results/prueba_m1_gpu_20261122_185842/`; entrada resumida `results/PRUEBA_M1_FINAL.json`. Se guardan predicciones por fecha/hora/semilla, pesos del router, métricas por semilla/año/horizonte, contrastes, registro previo, fuentes, verificaciones y manifiesto. El informe es `INFORME.md`. No se publica GitHub ni se modifica JBook/dashboard. Esta solicitud no programa un nuevo apagado.


## Cierre GPU de M2, M3 y M4 · 2026-11-23

Entrenamiento y prueba finalizados en GPU con cinco semillas101–105,1.461 días2022–2025 y24 horas. M2: MASE 5.326852; M3_OP: MASE 0.637692; M4_OP: MASE 0.578898. Se publican resultados favorables y desfavorables, predicciones, historiales, pesos, escaladores, fuentes y registros. La base M1 de M3/M4 es OOF anidada con selección por prefijo. M4 reúne las28 hipótesis para Holm conjunto. Evaluación retrospectiva informada por un benchmark conocido.

Entrada de resultados: `results/publicacion_m2_m3_m4_20261122_194500/README.md`. Su alcance y manifiesto distinguen archivos publicados de embeddings/estados reconstruibles conservados localmente. Se preservan los manifiestos originales. No se modifica ni despliega JBook o dashboard.


## 53. Recepción y análisis de prueba de E3 y exploración E · 23 de noviembre de 2026

Se descargan del commit `b71c3670665f476da2ca1afb32fc0f70b643f4c2` los resultados, fuentes y pesos del corrector final, M1, M2, M3 y M4: 8,644 archivos efectivamente publicados, incluidos 1,691 archivos .pt, comprobados por SHA-256. Tres .log figuran en los manifiestos pero no están en GitHub; la discrepancia queda registrada y no afecta a las predicciones o checkpoints. Las fuentes adicionales se guardan en results/recepcion_e3_20261123/fuentes_repositorio/, sin reemplazar scripts activos.

Carpetas canónicas: results/corrector_final_y_m1_gpu_20261122_150345/, results/prueba_m1_gpu_20261122_185842/, results/m2_gpu_20261122_192625/, results/m3_gpu_20261122_193240/ y results/m4_gpu_20261122_194500/. Los adjuntos se conservan con originales y 639 archivos extraídos en results/exploracion_E_20261123/; su ZIP denominado pesos solo contiene mlflow.db. No confundir esta segunda exploración con resultados/exploracion_e3_20261122/.

Informe canónico de esta recepción: results/recepcion_e3_20261123/INFORME.md. comparar_top10.py recalcula métricas y contrasta pérdidas guardadas sin redes. Se verificaron 43 filas de tablas originales, fechas, 1.461 orígenes, 24 horas y objetivos contra el panel. Las diez familias del Top 10 usan sus pérdidas canónicas; TimesNet conserva lote uno. Corrector: tres semillas; nuevos modelos: cinco. No se emparejan semillas distintas: la inferencia temporal condiciona a las realizaciones ejecutadas.

| Procedimiento | MASE recalculado |
|---|---:|
| Top10_TimesNet | 0.642928 |
| NNLS | 0.620904 |
| M1 | 0.634923 |
| M2 | 5.326852 |
| M3_OP | 0.637692 |
| M4_OP | 0.578898 |
| M4_N | 0.573144 |
| M4_O | 0.571659 |
| Corrector_oficial | 0.677038 |

Se añade una comparación exploratoria retrospectiva de once procedimientos con las diez familias: 110 parejas y 220 hipótesis (diferencia bilateral y mejora superior al 3 %), Holm conjunto. Bootstrap pareado por origen, 9.999 réplicas, bloques de 14 días dentro de año y sensibilidad 7/28. IC95 puntuales; incertidumbre condicionada a las semillas guardadas. Las comparaciones originales de M1–M4 mantienen su familia registrada de 28 hipótesis, distinta de esta ampliación con el Top 10.

M4_OP mejora a TimesNet 9,959 %; M4_N, 10,854 %, y M4_O, 11,085 %. Las tres superan significativamente las diez familias y el margen de 3 % con p Holm 0,022 en este análisis. M1 y M3_OP mejoran a nueve familias pero no significativamente a TimesNet. NNLS mejora significativamente a TimesNet 3,426 %, sin demostrar que supere el margen de 3 %. Los efectos se mantienen frente a TimesNet al cambiar el tamaño de bloque.

No se demuestra aporte semántico incremental: en M4, oficial frente a memoria numérica mejora 0,259 % con p Holm 1 en los contrastes registrados; oficial+prensa tiene más error que el control numérico. M3_OP empeora frente a M1. El corrector oficial mejora a C+R, pero frente al corrector numérico la ganancia es 0,214 % con p Holm 0,5292; además es significativamente peor que TimesNet en esta ampliación. M4 es una política con memoria creciente observable: su mejora frente al Top 10 no debe atribuirse solo a arquitectura o texto.

M2 presenta MASE 2,5114 en desarrollo y 5,3269 en prueba, con subestimación de energía diaria media 12,535 %. La ablación solo distribución obtiene 0,657655; solo energía, 5,331279. Auditar la estimación de energía y la coherencia de representaciones anuales/finales antes de extraer una conclusión general sobre el diseño; no se corrigió ni reentrenó.

La exploración E adjunta revisada contiene 14 celdas y 140 ajustes en 2018–2019, cinco semillas. E·recencia obtiene 0,410328 frente a 0,410656 de PatchTST control: mejora 0,0799 %, t pareada p 0,93798, sin evidencia de mejora y sin alcanzar 3 %. No incluye prueba 2022–2025. Sus p publicados son exploratorios, con pares año/semilla; no constituyen evidencia temporal confirmatoria.

Los pesos publicados ahora disponibles corresponden a los nuevos ajustes de GPU; la ausencia de pesos descrita en §47 sigue siendo cierta para los paquetes anteriores. En esta recepción no se entrenaron modelos, no se ejecutó inferencia neuronal y no se modificaron ni publicaron JBook o dashboard. El benchmark anterior permanece intacto.

## 54. Plan de originalidad y cierre de evaluación de M4 en GPU · 23 de noviembre de 2026

Se documenta el plan en `continuidad/PLAN_ORIGINALIDAD_EVALUACION_M4_GPU.md`, con especificación JSON y matriz inicial `continuidad/MATRIZ_NOVEDAD_M4.md`. `continuidad/README_GPU.md` se actualiza al estado real de M1–M4 ya entrenados y con pesos/resultados publicados. Los diseños iniciales del 22 de noviembre conservan sus bytes y hashes como procedencia histórica; no describen el trabajo restante.

Decisión inicial del plan, sustituida en cuanto a la propuesta vigente por §55: conservar la configuración 6 elegida con M4_OP y cerrar su búsqueda amplia. M4_OP mantiene su papel en la selección histórica; la nueva presentación de M4_N como propuesta retrospectiva no modifica esa procedencia ni implica selección a priori. La mejora del texto oficial frente al control numérico es 0,259 % con p Holm 1; no hay aporte semántico incremental demostrado. La ganancia de M4 frente al Top 10 no debe atribuirse únicamente al texto o a la arquitectura porque la memoria incorpora observaciones nuevas con retraso.

Primero revisar entre cinco y diez textos completos, especialmente RATL (arXiv:2609.03937) y FreshCast (arXiv:2610.07834), que ya proponen recuperación residual o actualización de memoria con base congelada. La matriz inicial distingue lecturas completas de resúmenes. La contribución candidata es la construcción OOF temporal anidada de la memoria y su integración con recuperación aprendida y actualización observable; deberá establecerse una diferencia algorítmica real. Aplicación colombiana, compuerta o cambio de hiperparámetros no bastan. Si la revisión no sustenta esa diferencia, describir una adaptación evaluada sin afirmar originalidad demostrada.

Completar formulación, pseudocódigo, complejidad y una demostración con verificación numérica. Se propone estabilidad local de la corrección bajo memoria fija, residuos acotados y margen que preserve top-k; no se afirma continuidad global ni garantía de no empeorar. Fortalecer las pruebas para verificar cada parámetro activo, causalidad, OOF por prefijo, retraso de dos días, máscara antes de top-k, identidad sin vecinos y restauración.

El plan GPU incluye controles de residuos en muestra/OOF/OOF anidado, recuperación uniforme/similitud/aprendida, memoria fija/creciente, compuerta y canales N/O/OP. Comparar con todo el Top 10 y controles fuertes, además de métodos próximos como RATL y FreshCast, con información observable y presupuestos equiparados. No alterar ni sobreescribir resultados históricos.

Antes de nuevas ejecuciones, registrar con fecha real `experiments/estandar.md`, protocolo, versiones/hashes, familias de hipótesis y etiqueta Git. Conservar MASE y margen relativo 3 %, alfa familiar 0,05; dimensionar con OOF de desarrollo para potencia mínima 0,90 por contraste e informar potencia conjunta. El E3 exige al menos diez semillas independientes —o más según potencia— para modelos estocásticos y comparaciones homogéneas. Cinco semillas existentes no acreditan ese cierre; no crear réplicas a partir de horas, épocas o copias de controles deterministas. Proponer 101–110 sin ejecutar y verificar igualdad de protocolo antes de reutilizar ajustes.

Seleccionar al menos tres datasets adicionales adecuados al pronóstico de demanda antes de mirar sus métricas; candidatos de adquisición UCI ElectricityLoadDiagrams20112014, GEFCom2014 Load e ISO New England, sujetos a comprobar cobertura, independencia y licencia. La prueba colombiana 2022–2025 ya examinada conserva carácter retrospectivo; reservar una confirmación no inspeccionada o prospectiva. Las pruebas F2 propuestas son bootstrap estacionario, DM–HLN y Giacomini–White, con supuestos verificados, Holm de la familia completa y dirección/magnitud de efecto explícitas. Tres tamaños de bloque no equivalen a tres pruebas distintas. Guardar incertidumbre entre días y semillas por separado.

Esta solicitud autoriza únicamente documentación y subida de contexto/plan al repositorio. No se entrenan redes, no se ejecuta inferencia ni se amplía evaluación. La lectura del plan no autoriza automáticamente una cola en GPU: requiere solicitud expresa. No se modifica ni publica JBook o dashboard y no se programa apagado. Los artefactos futuros se guardarán en carpetas nuevas con manifiestos; las conclusiones se generarán desde resultados.


## 55. M4_N como propuesta retrospectiva, capacidad igualada y ONNX · 23 de noviembre de 2026

Por solicitud del usuario, M4_N pasa a ser la propuesta retrospectiva del proyecto. La decisión ocurre después de conocer la prueba 2022–2025; no se presenta como preregistrada ni confirmatoria. Se mantienen configuración 6, pesos existentes y procedencia de la selección realizada con M4_OP. M4_O y M4_OP se conservan como sensibilidad al texto. La búsqueda amplia sigue cerrada. La originalidad metodológica y la confirmación con conjuntos adicionales continúan pendientes según el plan.

Se actualizan `continuidad/PLAN_ORIGINALIDAD_EVALUACION_M4_GPU.md` y su JSON v2 y se añade `continuidad/PROTOCOLO_CAPACIDAD_ONNX_M4_N.md`. La propuesta numérica será el centro de la evaluación adicional y de las ablaciones; los contrastes históricos se preservan sin reinterpretar la selección.

Control de capacidad: comparar correctores sobre la misma M1 congelada y las mismas variables, escala, horizonte, periodos y presupuesto. Excluir del conteo efectivo módulos textuales inactivos y columnas multiplicadas siempre por cero. Para n=912, la arquitectura M4_N activa tiene 64n+2625=60.993 parámetros, frente a 144.225 almacenados en su contenedor histórico. Se diseña MLP_CE de 912→65→24, con GELU y dropout 0,1: 60.929 parámetros, diferencia 0,105 %. Añadir EWMA_MLP_CE de 936→63→24: 60.567 parámetros, diferencia 0,698 %, con 24 resúmenes residuales del mismo banco observable y semivida fija de 30 días. Son conteos de diseño obtenidos de fuentes y metadatos, no controles entrenados. La tolerancia máxima será ±5 % de parámetros activos; en otros datasets elegir anchos solo por conteo, sin consultar error. No añadir pesos desconectados para igualar capacidad. Registrar memoria no paramétrica, cómputo y latencia aparte.

Los controles tendrán pérdida MAE/q, AdamW lr 0,001 y weight_decay 0,0001, lote 64, máximo 100 épocas, mínimo 10, paciencia 10 y min_delta 0,0001; parada solo interna temporal. Todas las actualizaciones del banco respetan episodios d−2 o anteriores y edad máxima de 365 días. El control sin memoria aísla parámetros; el control EWMA recibe la misma información revelada para no atribuir a arquitectura una ventaja informativa. El Top 10 conserva sus versiones nativas, sin presentarlo como grupo de capacidad igualada. Registrar estos controles en la familia de hipótesis antes de evaluarlos, junto con el dimensionamiento, diez semillas mínimas o más por potencia y al menos tres datasets adicionales del plan.

ONNX se incorpora como tarea futura con pesos congelados: encoder numérico y corrector tensoriales sin estado; controlador cronológico externo; base M1 explícita y sus subgrafos necesarios si se declara un paquete completo. Si solo se exporta el corrector, identificar base externa. Contrato inicial FP32, lote uno, 24 horas, banco con 365 posiciones y padding enmascarado, top-k 16 y máscara temporal antes de top-k. No fijar por trazado decisiones Python que dependan de entradas ni cambiar pesos o regla de vecinos para exportar.

Validar cada semilla contra su checkpoint original: CPU Windows/Linux y GPU compatible declarada; ausencia de NaN/Inf, memoria vacía, menos de k vecinos, límites de dos/365 días, empates, causalidad y recorrido temporal. Tolerancia propuesta en escala normalizada por q: atol=rtol=1e−5 y diferencia absoluta de MASE ≤1e−5; sin vecinos, identidad exacta con la base suministrada. Fijar versiones/opset, casos y tolerancias antes de ejecutar; no relajarlas después. Guardar grafos/pesos externos, fuente, escaladores, controlador, contrato, reportes de equivalencia y manifiesto SHA-256 en carpeta nueva. ONNX no almacena automáticamente el historial ni sustituye la auditoría de disponibilidad.

Esta actualización solo modifica y sube documentación. No se han entrenado los controles, generado archivos ONNX ni ejecutado inferencia o nueva evaluación. Los ajustes y comprobaciones futuras requieren solicitud expresa. No se modifica ni publica el JBook ni el dashboard.


## Adenda estadística durante la ejecución GPU · 23 de noviembre de 2026

La cola registrada continúa sin cambiar sus fuentes ni pesos. Se registra `experiments/m4_originalidad/ADENDA_RESOLUCION_F2.md` antes de las nuevas pruebas: resolución final de 49.999 réplicas por multiplicidad y GW con instrumento observable de dos días antes. La etapa final publica y verifica su cierre antes del apagado; el paso preliminar se conserva.

Se extraen los pronósticos agregados históricos de TimesNet y N-BEATS en `results/m4_cierre_gpu_20261123/predicciones_historicas_disponibles/`: 35.064 intervalos por modelo, promedio y percentiles entre cinco semillas, con fecha y horizonte. No se reconstruyen realizaciones individuales. Las pérdidas diarias publicadas permiten el contraste temporal con esos procedimientos; su incertidumbre entre semillas no queda reconstruida. Las pruebas del proyecto registran 84 aprobadas. El entrenamiento adicional y la publicación final siguen pendientes; esta adenda no certifica E3, originalidad ni portabilidad.


## Ampliación del cierre y recuperación GPU · 23 de noviembre de 2026

Se registra `experiments/m4_originalidad/ADENDA_COMPLETAR_PLAN.md` y `continuidad/AMPLIACION_PLAN_GPU_20261123.json`, antes de abrir las nuevas pruebas. La cola complementaria añade procedencia de residuos en muestra/OOF/OOF anidado, desarrollo de comparadores, diez semillas de canales/placebos documentales, dimensionamiento OOF con presupuestos 10/15/20/30, diagnósticos y auditoría de portabilidad. La familia final máxima pasa a 1.536 hipótesis, conservando las tablas y familias anteriores. El registro operativo está en `results/m4_cierre_gpu_20261123/complemento/REGISTRO_PREVIO.json`. Las fases están programadas, no se declaran completadas.

Se reciben los diez pesos originales de N-BEATS/TimesNet, con SHA y rutas en `results/m4_cierre_gpu_20261123/historicos_nbeats_timesnet/RECEPCION.json`. El ZIP no aporta escaladores; su reconstrucción determinista solo utiliza entrenamiento. La inferencia original, TimesNet con lote uno, debe reproducir el benchmark antes de integrar sus cinco realizaciones individuales.

Un timeout del controlador gráfico a las 13:37 recuperó la GPU y detuvo los procesos; no se perdió el progreso guardado por época. Se reanuda con la lista original de fuentes y checkpoints, sin cambiar arquitecturas ni hiperparámetros. La ejecución GPU será secuencial y se gestiona mediante servicios temporales de usuario. El cierre realiza inferencia histórica, complemento, publicación y verificación estadística antes de apagar. Windows, proveedor ONNX GPU no disponible y confirmación prospectiva permanecen sujetos a ejecución material; no se certifican potencia F2 ni originalidad por programar estas fases. Se aprobaron 84 pruebas del proyecto.


## Evaluación adicional M4_N en GPU · cierre20261123

El usuario sustituye la cola extensa por la opción B: solo Colombia, diez semillas 101–110, controles numéricos y ablaciones finales de residuos y texto. La selección y la prueba colombianas conservan carácter retrospectivo. Se reutilizan los ajustes completos; no se realiza nueva búsqueda ni validación externa ampliada de las nuevas ablaciones. UCI, GEFCom, OPSD y los nuevos ajustes del Top10 quedan diferidos. Los checkpoints y ajustes anteriores se conservan. TimesNet y N-BEATS se infieren desde cinco pesos originales sin reentrenar, con comprobación contra resultados canónicos. Se conservan 49.999 réplicas, GW con rezago dos y Holm con cota familiar 1.536. No se acredita potencia 0,90 ni cierre completo del E3. Protocolo: `experiments/m4_originalidad/REDUCCION_B_COLOMBIA_20261123.md`; resultados de la nueva cola: `results/m4_cierre_gpu_20261123/reducido_B/`. Publicación verificada antes del apagado. No se modifica JBook ni dashboard.

La cola B se reanuda con registro v2 y una corrección de compatibilidad numpy.str_→str en los dos límites Timestamp de la ablación en muestra. Los límites temporales, la purga y el procedimiento de ajuste se conservan; ver `experiments/m4_originalidad/ADENDA_B_FECHAS_NUMPY_20261123.md`. La inferencia histórica y los controles numéricos completados se reutilizan.

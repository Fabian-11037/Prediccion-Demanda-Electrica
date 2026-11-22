# Ampliación del corpus oficial · versión 20261120

Cierre de adquisición: 2026-11-21T22:38:36.051403+00:00 (UTC).

Se incorporan documentos de IDEAM, CREG, XM y Presidencia, sin modificar la prensa ni el panel numérico. El escenario declarado conserva su condición de principal; el estricto representa disponibilidad comprobada. No se ejecutaron entrenamientos, embeddings, nuevas evaluaciones de demanda ni publicaciones.

## Incorporación

| fuente | nuevos | con_cuerpo | declarado_2000_2025 | estricto_2000_2025 | cuarentena |
| --- | --- | --- | --- | --- | --- |
| CREG | 3706 | 3544 | 2768 | 0 | 938 |
| IDEAM | 45199 | 45003 | 40201 | 0 | 4990 |
| Presidencia | 773 | 772 | 753 | 0 | 20 |
| XM | 426 | 425 | 143 | 0 | 283 |

El corpus activo contiene 625,577 registros, de los cuales 50,104 son nuevos. Se conservan intactas las 575,473 filas previas. Las discrepancias entre la fecha del encabezado y la del nombre de un PDF histórico activan cuarentena hasta su revisión. Se conservan ambos campos; la fecha candidata más tardía no habilita su uso mientras exista el conflicto. Las discrepancias de año con la carpeta del paquete, sin concordancia de encabezado y nombre, también activan cuarentena: el año de la carpeta no se usa para redatar. Un archivo personal incluido en los paquetes se excluye del uso como boletín oficial. Los registros en cuarentena se conservan para auditoría y el cargador los excluye. `con_cuerpo` describe la presencia de más de 100 caracteres extraídos; no implica extracción completa ni disponibilidad histórica comprobada.

## Fuentes y alcance

- **IDEAM:** catálogos de ITD, BCH, pronóstico a tres días, predicción climática, ENOS y alertas oficiales BADT, BAICV y BAH, conciliados con el archivo público JSON:API. Se recuperaron 816 paquetes ZIP; cada PDF se extrajo con validación CRC, sin adoptar la fecha del paquete como fecha del boletín. ITD y BCH incluyen observaciones, alertas y pronósticos; no se presentan como equivalentes al pronóstico a tres días.
- **CREG:** índices anuales 2000–2025 de resoluciones definitivas y proyectos recientes, incluida la numeración por prefijos. Se prioriza el PDF original cuya identidad corresponde al acto. Los enlaces con identificador no corroborado se conservan en la auditoría; su contenido no se asigna a esa norma y se recurre al HTML depurado cuando existe. Los proyectos sin fecha sustentada permanecen en cuarentena.
- **XM:** informes mensuales del catálogo Sinergox y anuales del SIN y mercado. La migración de agosto de 2026 no acredita publicación en el año informado. Los documentos sin fecha editorial identificable permanecen en cuarentena aunque el archivo se haya recuperado.
- **Decretos:** compilaciones oficiales de Presidencia en CREG y Cancillería, 2000–2025. El inventario corresponde a esos catálogos y no certifica exhaustividad de todos los decretos nacionales o territoriales. Los actos replicados se deduplican por número y año.

## Disponibilidad temporal

Una fecha sin hora entra desde el día siguiente a las 00:00 de Colombia. En los actos jurídicos se toma la fecha de publicación del Diario Oficial y, cuando figura una fecha posterior de publicación web, esta última. La fecha de expedición no sustituye la publicación. La discrepancia del BADT cuyo archivo termina en `05_ABR_2010.pdf` se confirmó visualmente: el encabezado dice 5 de marzo de 2010. El registro queda en cuarentena; la imagen y la muestra auditada están en `encabezado_badt_20100405.png` y `muestra_fechas_alertas_historicas.json`.

La fecha estricta nueva es la recuperación efectiva en 2026, por lo que esta ampliación no incrementa la cobertura estricta en 2000–2025.

Los boletines históricos se datan por su encabezado individual o por una fecha de edición identificable en el nombre del PDF; la base queda registrada por documento. Un informe retrospectivo solo puede incorporarse una vez publicado; su periodo de referencia no es información disponible en ese periodo. Los metadatos `CreationDate`, `Last-Modified` y la fecha del ZIP se conservan como diagnósticos, no como prueba de disponibilidad histórica.

El escenario declarado no acredita que los bytes descargados en 2026 sean idénticos a los publicados originalmente. Esa advertencia se conserva por registro. Las versiones HTML jurídicas se limitan a secciones sin sustitución editorial, excluyen notas y artículos modificados y se identifican como parciales. Los caracteres de control no imprimibles se retiran y se identifican como pérdida parcial de extracción. El OCR se limita a cuatro primeras páginas y se identifica como parcial cuando corresponde.

## Cobertura anual de cuerpos · escenario declarado

| anio | CREG | IDEAM | MinEnergia | Presidencia | UPME | XM |
| --- | --- | --- | --- | --- | --- | --- |
| 2000 | 85 | 0 | 0 | 12 | 0 | 0 |
| 2001 | 58 | 0 | 0 | 12 | 1 | 0 |
| 2002 | 39 | 0 | 0 | 20 | 12 | 0 |
| 2003 | 76 | 0 | 0 | 13 | 12 | 0 |
| 2004 | 75 | 0 | 0 | 27 | 13 | 0 |
| 2005 | 91 | 0 | 0 | 21 | 15 | 0 |
| 2006 | 93 | 0 | 0 | 13 | 10 | 0 |
| 2007 | 100 | 0 | 0 | 16 | 12 | 0 |
| 2008 | 136 | 0 | 0 | 18 | 6 | 0 |
| 2009 | 179 | 0 | 0 | 23 | 6 | 0 |
| 2010 | 160 | 495 | 0 | 31 | 3 | 165 |
| 2011 | 149 | 2931 | 0 | 24 | 8 | 87 |
| 2012 | 115 | 2782 | 0 | 24 | 7 | 167 |
| 2013 | 164 | 2669 | 102 | 27 | 6 | 89 |
| 2014 | 133 | 1830 | 76 | 24 | 7 | 99 |
| 2015 | 187 | 3335 | 223 | 67 | 15 | 104 |
| 2016 | 210 | 3647 | 373 | 35 | 13 | 100 |
| 2017 | 156 | 3595 | 269 | 52 | 10 | 249 |
| 2018 | 146 | 3671 | 270 | 23 | 6 | 157 |
| 2019 | 129 | 3213 | 205 | 16 | 6 | 156 |
| 2020 | 158 | 2364 | 201 | 79 | 7 | 162 |
| 2021 | 171 | 2784 | 151 | 51 | 6 | 145 |
| 2022 | 146 | 1940 | 157 | 34 | 5 | 190 |
| 2023 | 93 | 837 | 134 | 26 | 6 | 144 |
| 2024 | 108 | 2666 | 147 | 31 | 7 | 138 |
| 2025 | 186 | 3250 | 369 | 34 | 7 | 121 |

## Cobertura anual de cuerpos · escenario estricto

| anio | CREG | IDEAM | MinEnergia | Presidencia | UPME | XM |
| --- | --- | --- | --- | --- | --- | --- |
| 2000 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2001 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2002 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2003 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2004 | 0 | 0 | 0 | 0 | 19 | 0 |
| 2005 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2006 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2007 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2008 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2009 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2010 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2011 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2012 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2013 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2014 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2015 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2016 | 0 | 0 | 0 | 0 | 0 | 1 |
| 2017 | 2 | 1 | 0 | 0 | 0 | 0 |
| 2018 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2019 | 1 | 4 | 0 | 0 | 0 | 1 |
| 2020 | 11 | 14 | 0 | 0 | 0 | 1 |
| 2021 | 35 | 0 | 0 | 0 | 1 | 1 |
| 2022 | 103 | 16 | 0 | 0 | 4 | 0 |
| 2023 | 25 | 23 | 0 | 0 | 67 | 0 |
| 2024 | 11 | 303 | 0 | 0 | 22 | 0 |
| 2025 | 64 | 517 | 0 | 0 | 8 | 0 |

Las cifras se agrupan por el año de disponibilidad, no por el periodo descrito. Los CSV mensuales contienen todos los 312 meses, incluidos los ceros. `inventario_por_periodo_referido.csv` e `inventario_por_mes_referido.csv` muestran por separado el periodo físico de los informes; no debe confundirse con disponibilidad.

## Cobertura, calidad y conservación

- `cobertura_mensual_fuente.csv`, `cobertura_anual_fuente.csv` y `cobertura_particiones.csv`: documentos y orígenes con texto a 24/72/168 horas, en ambos escenarios, con agregados de todas las fuentes y de solo oficiales.
- `cobertura_oficial_mensual_cuerpo.csv` y `cobertura_oficial_anual_cuerpo.csv`: cuerpos y días por fuente, separando titulares.
- `calidad_extraccion.csv`, `documentos_cuarentena.csv`, `alias_documentos.csv` y `fallos_descarga_final.json`: extracción parcial, exclusiones, duplicados evitados y fallos de acceso (0 URL pendientes).
- `auditoria_consistencia_fechas_paquetes.csv`: discrepancias entre las fechas del documento, su nombre y la carpeta histórica.
- `conservacion_resultados.json`: huellas de los archivos previos. Se conservan los resultados, vectores, noticias originales y panel numérico.

La consulta de los catálogos no garantiza recuperar cada documento histórico. El archivo antiguo de IDEAM y las consultas históricas de Wayback no permitieron cerrar los huecos más antiguos. La cobertura efectiva de 2000–2009, del pronóstico específico a tres días anterior a 2024 y de los informes XM sin fecha debe leerse en las tablas; no se considera resuelta por la mera descarga de paquetes o informes.

## Uso en futuros entrenamientos

El dataset activo es `data/processed/corpus_texto.parquet`, versión `corpus_oficial_ampliado_20261120`. Se mantienen las variantes numérica, numérica + oficial y numérica + oficial + prensa, con canales separados e indicadores de ausencia y relevancia aprendida por el modelo. No se añadieron agrupamientos temáticos ni filtros manuales.

El corpus congelado de la evaluación actual conserva 14.162 documentos en `results/evaluacion_e2/entradas/texto/corpus_oficial.parquet`; sus vectores y resultados no cambian. Antes de un entrenamiento autorizado con el corpus ampliado se debe generar un nuevo caché versionado y volver a auditar sus ventanas temporales. No mezclar los índices de documentos nuevos con los vectores congelados.

Antes de construir el caché nuevo se debe revisar `origenes_superan64_72h`, `p95_documentos_72h` y `max_documentos_72h` en `cobertura_particiones.csv`. La ampliación puede cambiar la cantidad de documentos que caben en el límite documental; fijar y auditar ese presupuesto sin filtros temáticos manuales ni selección por resultados de prueba.

La instantánea previa está en `data/processed/versiones/corpus_oficial_20261109/`. El manifiesto activo registra SHA-256, versiones, reglas y rutas. Las fases reproducibles son `ampliar_corpus_20261120.py`, `integrar_ampliacion_20261120.py` y `cerrar_ampliacion_20261120.py`; requieren requests, BeautifulSoup, pandas, numpy, openpyxl y PyMuPDF. El entorno exacto se registra en `entorno.json`. La extracción nativa es portable; el OCR empleado requiere Windows y el entorno `.ocr_env` existente.

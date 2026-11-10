# Datos adicionales para futuros entrenamientos

Versión: `datos_adicionales_20261110`. Recuperación: 10 de noviembre de 2026. El panel numérico 2000–2025, el archivo original de titulares, los modelos y el benchmark se conservan. La ampliación se descarga y valida por separado; no ejecuta entrenamientos ni reconstruye/publica el JBook.

| Fuente | Archivo de entrada | Cobertura y alcance |
|---|---|---|
| Open-Meteo, JMA GSM y GFS Global | `data/external/pronosticos_temperatura/pronosticos_meteorologicos.parquet` |3.061.440 filas de hora×ciudad×modelo×adelanto; ocho ciudades y adelantos 1/2/3 días. Incluye temperatura, humedad y precipitación con faltantes explícitos. |
| DANE ISE | `data/external/actividad_economica/ise_dane.csv` |260 meses de la base 2015, enero 2005–agosto 2026, vintage de publicación 19 oct 2026. Las 168 ediciones descargadas se conservan;102 se normalizan en `ise_vintages.parquet`. |
| DANE población | `data/external/poblacion_dane/poblacion_ciudades_2000_2026.csv` |216 filas, ocho municipios por 27 años; población total, cabecera, rural y peso entre las ocho ciudades. Proyecciones actuales con versiones 2023/2025. |
| XM actualización 2026 | `data/external/actualizacion_2026/xm/` |Demanda y precio hasta 7 nov 2026; hidrología hasta 9 nov 2026. Las 48 horas finales aún no publicadas permanecen nulas en la malla horaria solicitada. |
| ERA 5 actualización 2026 | `data/external/actualizacion_2026/clima/` |Ocho ciudades, enero–9 nov solicitado; valores efectivos hasta 4 nov 2026 a las 18:00. Las horas posteriores permanecen nulas. |
| XM SIMEM pronóstico oficial | `data/external/pronostico_demanda_xm/` |762.480 filas,18 subáreas, enero 2022–octubre 2026. La publicación SIMEM es posterior al objetivo en todos los registros: solo auditoría, sin emisión original acreditada. |
| Corpus textual declarado | `data/processed/corpus_texto.parquet` |575.473 registros de prensa y documentos institucionales; mismos datos y advertencias del corpus integrado. |

## Pronósticos meteorológicos

El proveedor `jma_gsm` tiene temperatura y otras variables desde 2018; `gfs_global` conserva temperatura desde marzo 2021 y humedad/precipitación desde enero 2024. No usar `gfs_seamless` como sustituto: las pruebas de ese identificador devolvieron valores nulos en fechas donde `gfs_global` sí tiene datos. No promediar proveedores en silencio.

Previous Runs alinea adelantos fijos; no identifica el ciclo exacto de inicialización ni prueba su publicación. Se guarda `emision_estimada=valido−lead×24h` y `disponible_estimado=emision_estimada+12h`. Son un proxy declarado, no horas certificadas. Ver [documentación del proveedor](https://open-meteo.com/en/docs/previous-runs-api). Las respuestas originales comprimidas, parámetros y SHA-256 se conservan en `originales/`.

Hay 186 valores de precipitación negativa (−0,1 mm) devueltos por JMA. El original y un CSV conservan esos valores; la tabla procesada los marca con bandera y NaN, sin sustituirlos por lluvia cero. Los huecos de archivo tampoco se completan con ERA 5 futuro.

Para el origen diario y un horizonte de 24 h, la auditoría busca el último adelanto 1..3 disponible bajo el proxy, con las tres variables completas y ocho ciudades. JMA cubre todos los 1461 orígenes de validación 2018–2021 y todos los 1461 de prueba 2022–2025. En 2026 cubre 299 de 313. GFS con temperatura cubre 1439 de 1461 orígenes de prueba; con las tres variables,712 de 1461. La cobertura se documenta en `results/datos_adicionales_20261110/cobertura_pronosticos_por_particion.csv`.

**No hay pronósticos en entrenamiento de referencia 2000–2017.** No introducir esta rama en los cuatro pliegues originales como si hubiera tenido historia de ajuste completa. Para el benchmark numérico del periodo completo pueden estudiarse clima pasado con disponibilidad adecuada y climatología ajustada solo con entrenamiento; una evaluación de pronósticos requiere un protocolo específico para su subperiodo y sus máscaras, sin alterar los resultados congelados.

## Versiones de ISE y población

Las ediciones antiguas del ISE conservan datos desde 2000, pero tienen bases o metodologías distintas y algunas fechas desconocidas. No unirlas automáticamente a la base 2015 desde 2005. Las 66 ediciones aún sin parser permanecen completas en sus Excel; las 12 fechas de publicación no identificadas quedan sin fecha, no se infieren del mes de referencia. Una fecha de boletín no verifica el contenido histórico exacto del Excel recuperado en 2026.

Para cada origen, seleccionar edición declarada ya publicada y mes conocido con rezago conservador 60 días; no emplear retrospectivamente el vintage final. El selector excluye bases distintas de la solicitada y fechas desconocidas. Las ponderaciones poblacionales nuevas tampoco sustituyen los pesos congelados de 2018: las retroproyecciones y proyecciones son versiones actuales, publicadas en 2023/2025.

## Uso y reproducción

`src/demanda/datos_adicionales.py` prepara un proveedor explícito, selecciona un pronóstico antes del origen con aceptación explícita del proxy, conserva máscaras y elige ISE por edición. No contiene optimización ni se integra automáticamente a los modelos guardados.

```bash
python scripts/adquisicion/descargar_pronosticos_historicos.py
python scripts/adquisicion/descargar_dane_pendientes.py
python scripts/adquisicion/descargar_xm_2026_y_pronostico.py
python scripts/validar_datos_adicionales.py
```

Instalar `requirements-datos.txt`; `requirements-datos.lock.txt` registra las versiones usadas. Las descargas se reanudan desde caché con hashes. Revisar la cobertura, el escenario temporal y el conjunto numérico común antes de ajustar nuevos modelos. El manifiesto de entradas es `data/processed/dataset_ampliado_20261110.json`.

La extensión 2026 es retrospectiva. Un bloque prospectivo debe comenzar después de congelar protocolo/modelo y registrar pronósticos en tiempo real. Siguen pendientes la continuidad exhaustiva del texto 2026, las emisiones originales de XM, horas certificadas de publicación y los metadatos bibliográficos institucionales que no se han recuperado.

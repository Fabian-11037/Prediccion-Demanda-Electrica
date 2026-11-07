# Datos crudos

| Ruta | Contenido | Fuente | Notas |
|---|---|---|---|
| `numeric/hourly/AAAA.csv.gz` | Serie horaria 2000-01-01 00:00 a 2025-12-31 23:00 (hora de Colombia, 227 928 filas, sin huecos): demanda real, demanda comercial, precio de bolsa, temperatura, humedad relativa y precipitación de la hora previa en 8 ciudades | XM (demanda, precio); ERA5 vía Open-Meteo (clima) | `timestamp_start_colombia` es el inicio del intervalo; `xm_hour` 1..24 (Hora 1 = 00:00-01:00). El clima es **reanálisis observado**, no pronóstico. |
| `numeric/hydrology/AAAA.csv.gz` | Hidrología diaria del SIN: llenado, reserva útil, capacidad útil, aportes | XM | Dato diario: no replicar en horas del mismo día sin regla de publicación (ver `src/demanda/consolidar.py`). |
| `numeric/ciudades.csv` | Coordenadas solicitadas, celda de malla y elevación de las 8 ciudades | Open-Meteo | |
| `noticias/titulares.parquet` | 568 295 titulares (prensa colombiana 2004-2025 y boletines IDEAM 2020-2025) con campos de fecha y de disponibilidad | Archivos web, feeds archivados, sitemaps y catálogos públicos | Solo titulares; `url` únicamente en registros con versión archivada verificada. Ver `src/demanda/noticias.py`. |

## Cobertura de titulares (importante)
La cobertura por fuente cambia mucho con el año: de 2008 a 2019 casi todo es El Colombiano (sitemap), y a partir de 2020 aparecen otras fuentes con volúmenes muy desiguales. El escenario estricto (versión archivada verificada) solo cubre desde 2021-04-25. Ver `data/processed/cobertura_titulares_*.csv`.

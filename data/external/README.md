# Datos externos

| Elemento | Estado | Cómo se obtiene |
|---|---|---|
| `festivos_colombia.csv` | Generado (2000-2035) | `python scripts/generar_festivos.py` (implementación de la Ley 51/1983 y leyes posteriores; verificada contra el paquete `holidays` en las pruebas) |
| `actividad_economica/ise_dane.csv` | **Pendiente de descarga** | Descargar el anexo del ISE del DANE y ejecutar `python scripts/adquisicion/importar_ise.py <xlsx> --vintage AAAA-MM-DD` |
| `pronosticos_temperatura/pronosticos_temperatura.parquet` | **Pendiente de descarga** | `python scripts/adquisicion/descargar_pronosticos_temperatura.py --inicio 2021-03-01 --fin 2025-12-31` |

## Pronósticos de temperatura
El clima de `data/raw` es reanálisis (ERA5) y **no puede usarse como pronóstico** de la hora objetivo. Los pronósticos con emisión conocida se obtienen de la API *Previous Runs* de Open-Meteo (variables `temperature_2m_previous_dayN`). La cobertura es corta (GFS desde 2021-03; la mayoría de modelos desde 2024-01): solo permite experimentos en un subperiodo. Regla de uso: `valido - N*24h + margen + latencia <= origen` (`src/demanda/pronosticos.py`). Mientras no exista el archivo, el sustituto sin fuga es la climatología de entrenamiento (`src/demanda/climatologia.py`).

## ISE (actividad económica)
Indicador mensual (base 2015 = 100; serie original, desestacionalizada y tendencia-ciclo). El DANE lo publica unos 45-49 días después de terminar el mes de referencia; se asume disponibilidad = fin de mes + 60 días (`ISE_REZAGO_DIAS`). Las cifras se revisan en cada publicación, por lo que una serie descargada hoy contiene valores que no existían en la fecha original (sesgo de *vintage*): debe declararse como limitación.

## Festivos
Todos los festivos son `known_future` (la ley los fija con antelación). Desde 2026 existe un festivo adicional (Nuestra Señora del Rosario de Chiquinquirá, 9 de julio, trasladable al lunes; Ley 2578 de 2026).

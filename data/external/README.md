# Datos externos

| Elemento | Estado | Cómo se obtiene |
|---|---|---|
| `festivos_colombia.csv` | Generado (2000-2035) | `python scripts/generar_festivos.py` (implementación de la Ley 51/1983 y leyes posteriores; verificada contra el paquete `holidays` en las pruebas) |
| `actividad_economica/ise_dane.csv` | Descargado:260meses,2005–2026-08, base2015 |168anexos originales;102versiones normalizadas, con advertencias de vintage |
| `pronosticos_temperatura/pronosticos_temperatura.parquet` | Descargado: JMA2018 y GFS2021 hasta2026-11-09 |Usar el nuevo script `descargar_pronosticos_historicos.py`; proveedor explícito y disponibilidad proxy |
| `poblacion_dane/poblacion_ciudades_2000_2026.csv` | Descargado: ocho municipios,2000–2026 |Proyecciones revisadas; no sustituir pesos congelados automáticamente |
| `actualizacion_2026/` | Extensión retrospectiva descargada |Los últimos días no publicados permanecen nulos |
| `pronostico_demanda_xm/` | Histórico SIMEM2022–2026 descargado |Sin emisión original: solo auditoría temporal |

## Pronósticos meteorológicos, ISE y población

Inventario actualizado, cobertura, límites de disponibilidad y comandos en [README_DATOS_ADICIONALES_20261110.md](README_DATOS_ADICIONALES_20261110.md). El clima de `data/raw` es ERA5 y no representa un pronóstico del horizonte. Los pronósticos descargados son adelantos fijos con fecha de emisión estimada; no certificar una hora de publicación que la API no proporciona.

## Festivos
Todos los festivos son `known_future` (la ley los fija con antelación). Desde 2026 existe un festivo adicional (Nuestra Señora del Rosario de Chiquinquirá, 9 de julio, trasladable al lunes; Ley 2578 de 2026).

# Resultados consolidados

Actualización: 8 de noviembre de 2026. Demanda horaria del SIN; pronóstico diario de 24 horas. Entradas: demanda histórica y calendario. Validación expansiva 2018–2021, prueba fija 2022–2025.

## Modelos de referencia con preprocesamiento corregido

Selección por el menor MASE de confirmación de validación entre las configuraciones elegidas en v3 y v4; no por error de prueba ni por la mejor semilla. Se conservan las cinco semillas finales de cada modelo.

| Familia | Ejecución | MASE validación piloto | MASE prueba | MAE prueba (kWh) | Reducción frente al ingenuo semanal |
|---|---|---:|---:|---:|---:|
| mlp | base_v3 | 0.542 | 0.752 | 162,149 | 44.5% |
| arx_lineal | base_v3 | 0.681 | 0.852 | 183,636 | 37.1% |
| cnn1d | base_v3 | 0.681 | 0.857 | 184,859 | 36.7% |
| lstm | base_v4 | 0.771 | 1.001 | 215,726 | 26.2% |

Referencias de prueba: ingenuo semanal MASE=1.355; ingenuo diario MASE=2.077.

## Trazabilidad y alcance

- `metricas_ejecuciones.csv`: todas las métricas agregadas de v2, v3 y v4; la ejecución original no se mezcla con las corregidas.
- `modelos.json`: configuración y rutas de pesos, escaladores, predicciones, curvas y métricas de las cinco semillas por familia. Las rutas parten de la raíz del repositorio; no se duplican checkpoints.
- `indice_artefactos_v3_v4.csv`: SHA-256 de 2304 archivos, correspondientes a 384 ajustes completos. Los artefactos de v2 conservan su índice en `results/base_v2/indice_artefactos.csv`.
- `referencias_test.json`: métricas de ambos ingenuos. Las versiones originales y sus fuentes permanecen disponibles.
- Predicciones: NPZ con `y`, `yhat` y `origen` (nanosegundos UTC); `scripts/exportar_predicciones.py` exporta fechas y semillas a CSV. Los historiales contienen pérdidas de entrenamiento y parada temprana por época.
- Checkpoints: estado PyTorch y metadatos; en v3/v4 el escalador también está disponible como JSON independiente. Para reconstruir v3 se usan `DatosFold` y `Secuencial` de `demanda.reentrenamiento_v3`; para secuenciales v4, `demanda.variantes_v4.Secuencial`. ARX/MLP usan los constructores de `modelos_base`. La clase Entrenador v4 reutiliza el bucle v3 con la fábrica v4.

## Limitaciones

La prueba fue observada repetidamente: todas las comparaciones entre ejecuciones son exploratorias y descriptivas. No se afirma significancia, equivalencia ni confirmación independiente. La selección consolidada también es retrospectiva. V2 usó 195 semillas finales; v3/v4, cinco. Cambian presupuesto, escalado y arquitectura entre versiones, por lo que las diferencias no identifican efectos causales aislados. Las desviaciones entre semillas no representan incertidumbre temporal. El escalado fijo de v2 incluía parte de la parada temprana en los pliegues tempranos; v3/v4 lo corrigen sin modificar las ejecuciones antiguas. El denominador MASE de referencia 2000–2017 permanece fijo para comparabilidad y no participa en la optimización.

## Auditoría y diagnósticos

La auditoría de ventanas y la salvedad del escalado están en `results/auditoria_dataset/`; la sensibilidad a demanda y los controles exploratorios iniciales están en `results/diagnostico_secuenciales/`. Sus hallazgos no deben confundirse con un experimento causal o con resultados de la fase corregida.

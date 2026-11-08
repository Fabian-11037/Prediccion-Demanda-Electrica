# Resultados de entrenamiento GPU v2

Ejecución completada el 8 de noviembre de 2026. Demanda horaria del SIN; historia de 168 horas y horizonte de 24 horas. Validación forward chaining: 2018–2021; prueba: 2022–2025. GPU AMD Radeon RX 6700 XT, PyTorch con ROCm, determinismo estricto.

## Archivos publicados

- `busqueda.csv`, `mejor_config.json`: búsqueda y configuración seleccionada por familia.
- `piloto.csv`, `piloto_resumen.json`: cinco semillas por familia sobre cuatro pliegues; dimensionamiento de 195 semillas finales por familia.
- `final_runs.csv`, `resumen_final.csv`: métricas individuales y agregadas de prueba.
- `comparacion_validacion_prueba.md` y `.csv`: brecha de generalización temporal.
- `resumen_ajustes.csv`: métricas y pérdidas resumidas por ajuste completado.
- `referencias_cv.json`, `referencias_test.json`: baselines estacionales.
- Planes, entornos y registros de modificación: trazabilidad del entrenamiento.

Los artefactos completos se incluyen en `ajustes/{etapa}/{id}/`: `modelo.pt` (pesos, escaladores y configuración), `escaladores.json`, `predicciones.npz`, `historia.json`, `identidad.json` y `resultado.json`. `indice_artefactos.csv` identifica etapa, modelo, semilla, periodo, dimensiones y SHA-256 de cada archivo. Los historiales agregados originales están en `hist/`. La carpeta local `preds/` contiene copias redundantes y no se publica.

Las predicciones NPZ contienen `y`, `yhat` (orígenes × 24 horas) y `origen` (nanosegundos UTC). El horizonte 1 corresponde al propio origen, con intervalos horarios posteriores hasta horizonte 24. Para obtener fechas en hora de Colombia y semilla en cada fila:

```bash
python scripts/exportar_predicciones.py results/base_v2/ajustes/final/ID salida.csv
```

Los escaladores se ajustaron con los datos de entrenamiento correspondientes. Además de las medias y desviaciones, el checkpoint conserva columnas de entrada, configuración de ventanas y escala MASE. La normalización por nivel utiliza la media de demanda de cada ventana histórica al pronosticar. Los `.pt` son checkpoints PyTorch producidos localmente; para leer sus metadatos completos se utiliza `torch.load(..., map_location="cpu", weights_only=False)` con estos archivos de confianza.


## Interpretación y limitaciones

MLP obtuvo el menor MASE de prueba (0,692), seguido de ARX (0,831), ambos inferiores al baseline semanal (1,355). LSTM (1,748) y CNN1D (2,070) no superaron esa referencia. No se han realizado pruebas de significancia. Los intervalos de `resumen_final.csv` reflejan variación entre semillas, no incertidumbre de muestreo temporal.

La búsqueda LSTM se redujo durante su ejecución a cinco configuraciones por costo computacional; las demás familias evaluaron veinte. Véase `ADENDA_LIMITE_LSTM.md` en la raíz. La evaluación final utilizó 195 semillas por familia. La brecha validación–prueba no identifica por sí sola sobreajuste.

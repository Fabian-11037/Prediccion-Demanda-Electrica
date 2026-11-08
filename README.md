# Predicción de demanda eléctrica (SIN, Colombia) con series de tiempo y texto

Datos consolidados con reglas explícitas de disponibilidad temporal para entrenar modelos que combinen series de tiempo y NLP en el pronóstico de la **demanda real horaria** del Sistema Interconectado Nacional.

Lee primero [`CONTEXTO_PROYECTO.md`](CONTEXTO_PROYECTO.md): define el problema, las reglas anti-fuga, las decisiones abiertas y las limitaciones de los datos.

## Contenido
- `data/raw`: serie horaria 2000-2025 (demanda, precio de bolsa, clima en 8 ciudades), hidrología diaria y 568 295 titulares con metadatos de disponibilidad.
- `data/external`: festivos de Colombia (generados), y scripts/plantillas para ISE (DANE) y pronósticos de temperatura (pendientes de descarga).
- `data/processed`: panel horario *as-of* (`panel_horario.parquet`), registro de variables con su clase de disponibilidad, informe de calidad y cobertura de titulares.
- `src/demanda`: calendario, consolidación, particiones, ventanas por origen, titulares por escenario, ISE, pronósticos y climatología (todo ajustado solo con entrenamiento).
- `scripts/entrenar_modelos_base.py`: búsqueda, piloto y corrida final de los tres modelos base en PyTorch (ARX lineal ridge/lasso, MLP, LSTM), con GPU automática y opciones de hiperparámetros (ver `CONTEXTO_PROYECTO.md` §9).
- `tests`: pruebas de fuga (ventanas, rezagos, escalas, particiones, pronósticos, ISE, titulares) y de calendario.

## Uso rápido
```bash
pip install -r requirements.txt
python scripts/construir_panel.py
pytest -q tests
```

## Diseño anti-fuga en una línea
Cada columna `x_*` ya trae su rezago de publicación y solo se lee antes del origen; el futuro solo expone calendario (`cal_*`) y pronósticos con emisión previa; las escalas se ajustan con entrenamiento; la partición purga el horizonte y la prueba está congelada.

## Resultados consolidados

Las ejecuciones GPU v2, v3 y v4 están completas. La consolidación de métricas, limitaciones y modelos se encuentra en [`results/consolidado/README.md`](results/consolidado/README.md). El registro [`modelos.json`](results/consolidado/modelos.json) referencia los pesos, escaladores, predicciones y curvas de las cinco semillas de cada configuración seleccionada por validación entre las versiones corregidas.

| Familia | Ejecución | MASE de prueba ↓ |
|---|---|---:|
| MLP | v3 | 0,752 |
| ARX | v3 | 0,852 |
| CNN1D | v3 | 0,857 |
| LSTM | v4 | 1,001 |
| Ingenuo semanal | Referencia | 1,355 |
| Ingenuo diario | Referencia | 2,077 |

La prueba 2022–2025 ya fue observada: son resultados exploratorios, sin confirmación independiente ni pruebas de significancia. La selección entre ejecuciones es retrospectiva. Las versiones v3/v4 corrigen el ajuste del escalador para excluir la parada temprana; v2 se conserva como antecedente, con su salvedad documentada. No se deben mezclar sus presupuestos ni interpretar sus diferencias como efectos causales aislados.

Se conservan los artefactos completos en `results/base_v2/`, `results/base_v3/` y `results/base_v4/`: pesos, escaladores, predicciones con fecha y semilla e historiales por época. Los planes de cada fase documentan arquitectura, presupuesto y particiones. Pendientes de datos: ISE y pronósticos meteorológicos con emisión verificable; no se incorporaron como entradas a estas ejecuciones.

## Jupyter Book

[Consultar el libro](https://fabian-11037.github.io/Prediccion-Demanda-Electrica/intro.html).

Autores: Fabian José Manrique Morelos y Gloria Bastidas Rodriguez. El libro integra datos, EDA, literatura, diseño, benchmark, curvas y diagnósticos con código visible e interpretaciones. El código completo de entrenamiento y prueba se conserva en el apéndice.

Las fuentes del libro están en book y scripts; docs contiene el HTML de publicación. La reproducción documental y la continuidad se describen en [el contexto del libro](continuidad/CONTEXTO_PROYECTO.md).

El libro publicado presenta el análisis y benchmark documentados en los cuadernos actuales. Los artefactos de los modelos se consultan en [la consolidación](results/consolidado/README.md).

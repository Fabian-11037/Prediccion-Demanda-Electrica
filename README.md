# Pronóstico de demanda eléctrica del SIN colombiano

**Fabian José Manrique Morelos y Gloria Bastidas Rodriguez**  
Maestría en Estadística Aplicada · Universidad del Norte.

Pronóstico directo de demanda real horaria a 24 horas, con 168 horas de historia. El estudio compara diez modelos principales, dos comparadores numéricos y las variantes N/N+O de TGForecaster: trece procedimientos con cinco semillas cada uno, además de dos referencias estacionales. MASE es la métrica primaria; MAE y RMSE complementan la evaluación.

- [Jupyter Book](https://fabian-11037.github.io/Prediccion-Demanda-Electrica/intro.html).
- [Dashboard interactivo](https://demanda-electrica-dashboard.onrender.com/index.html).
- [Contexto del proyecto](CONTEXTO_PROYECTO.md).

## Contenido

`book/` contiene las fuentes del libro, los cuadernos ejecutados, las fichas de arquitectura, sus esquemas y los pipelines. `docs/` contiene la versión publicada. `data/processed/panel_horario.parquet` integra demanda, precio, clima e hidrología con sus reglas de disponibilidad. `results/evaluacion_e2/` conserva el benchmark y los contrastes; `results/diagnosticos_ampliados/` conserva importancia, latencia y sensibilidad documental. `scripts/` y `src/` incluyen las implementaciones; `tests/` verifica datos, disponibilidad y evaluación.

## Construcción del libro

Con Python y las dependencias documentales:

```bash
python -m pip install -r requirements-book.lock.txt
jupyter-book build book --warningiserror --keep-going
```

El resultado se genera en `book/_build/html/intro.html`. Los cuadernos incluyen sus salidas y la construcción no los reejecuta. En Windows, los ejecutables se encuentran en `Scripts/` del entorno virtual; en Linux, en `bin/`.

Para reconstruir las gráficas de evaluación a partir de predicciones y mediciones conservadas:

```bash
python scripts/reconstruir_evaluacion.py
jupyter-book build book --warningiserror --keep-going
```

`scripts/reorganizar_capitulo10.py` reconstruye la organización y presentación integrada; `scripts/actualizar_diagnosticos_jbook.py` actualiza los diagnósticos desde los CSV. La sección estadística forma parte de `10_evaluacion.ipynb`; la fuente inicial se conserva en `results/reorganizacion_capitulo10/fuentes/`. Estos procesos no entrenan modelos.

La repetición de inferencia requiere los checkpoints, escaladores y vectores identificados en `results/diagnosticos_ampliados/fuentes_diagnosticos.json`, el entorno `requirements-inferencia.lock.txt` y la ubicación `DEMANDA_ARTEFACTOS`. La construcción documental y la regeneración de figuras desde resultados guardados no requieren esos pesos.

## Datos y evaluación

Las entradas históricas se leen antes del origen, el calendario puede conocerse para el horizonte y las transformaciones se ajustan exclusivamente con entrenamiento. El clima ERA5 es reanálisis. El texto utiliza disponibilidad declarada y conserva las advertencias de versión histórica. La validación avanza cronológicamente; la evaluación retrospectiva utiliza los mismos 1461 orígenes de 2022–2025 para todos los procedimientos.

```bash
python -m pytest -q tests
```

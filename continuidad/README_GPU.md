# Continuidad en un PC con GPU

Documentación actualizada el 22 de noviembre de 2026. Esta entrega publica contextos y diseños; no ejecuta entrenamiento, evaluación de prueba ni despliegue de JBook o dashboard.

## Orden de lectura

1. `../AGENTS.md` y `../CONTEXTO_PROYECTO.md`, especialmente §48–§50.
2. `DISENO_CUATRO_MODELOS_GPU.md`: arquitecturas M1–M4, controles, entrenamiento y evaluación.
3. `DISENO_CUATRO_MODELOS_GPU.json`: parámetros propuestos, configuraciones de expertos, rejillas y familia de contrastes.
4. `ENTRENAMIENTO_GPU_E3.md`: entorno GPU, datos y trabajo pendiente para la prueba del corrector ya estudiado.
5. `CONFIGURACIONES_PRUEBA_E3_GPU.json`: configuraciones históricas seleccionadas; no confundirlas con los hiperparámetros propuestos para M1–M4.

Los cuatro diseños son M1 (ensamble por complementariedad), M2 (energía diaria y forma horaria), M3 (selección de expertos con texto) y M4 (memoria de episodios numéricos y textuales). No hay un lanzador validado para estos cuatro modelos; su implementación es una tarea futura.

## Obtener datos y fuentes

Desde la raíz del repositorio, con Git LFS instalado:

```console
git pull --ff-only
git lfs install
git lfs pull --include="data/processed/corpus_texto.parquet"
```

Verificar los hashes del corpus y panel indicados en los JSON antes de preparar datos. El archivo LFS debe contener el Parquet de 159.086.185 bytes; un puntero de texto no es el dataset. Los vectores E5 de esta ampliación y los nuevos checkpoints no están incluidos y deben recuperarse o reconstruirse según las guías.

El código y resultados del corrector están en `results/corrector_semantico_e3_20261122/`. Los adjuntos de la primera exploración, `results/exploracion_e3_20261122/`, permanecen fuera de esta subida documental: trasladarlos por separado si se necesitan predicciones, historiales o comprobación de los archivos originales citados. Las configuraciones completas de los expertos se incluyen en los JSON; las rutas de procedencia locales no impiden consultarlas.

## Alcance de la continuación

La instrucción vigente es preparar la continuidad documental. Una ejecución futura requiere solicitud expresa. Después de implementarlos y verificar disponibilidad temporal y restauración de checkpoints, seleccionar solo con desarrollo 2018–2021, congelar configuraciones y evaluar 2022–2025 según el protocolo. No utilizar prueba para elegir diseños ni sustituir modelos por pesos de otra ejecución.

El MASE es la métrica principal; el margen relativo es 3 %. Se mantienen los controles semánticos, cinco semillas, early stopping, OOF temporal y anidado, memoria observable con retraso y contraste por bloques/Holm. Leer los detalles antes de iniciar un ajuste. No modificar ni publicar JBook o dashboard sin solicitud del usuario.

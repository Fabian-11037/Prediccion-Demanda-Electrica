# Continuidad en un PC con GPU

Actualizado el 23 de noviembre de 2026.

## Orden de lectura

1. [Instrucciones](../AGENTS.md) y [contexto del proyecto](../CONTEXTO_PROYECTO.md), especialmente §53–§54.
2. [Plan de originalidad y evaluación de M4](PLAN_ORIGINALIDAD_EVALUACION_M4_GPU.md): prioridades, controles, propiedad teórica, dimensionamiento y cierre estadístico del E3.
3. [Especificación del plan](PLAN_ORIGINALIDAD_EVALUACION_M4_GPU.json) y [matriz inicial de antecedentes](MATRIZ_NOVEDAD_M4.md).
4. Informes y manifiestos de resultados publicados indicados abajo.

## Estado real de los modelos

M1, M2, M3 y M4 ya fueron implementados, entrenados y evaluados con cinco semillas. Los resultados y checkpoints están publicados; los manifiestos distinguen artefactos incluidos de embeddings y estados reconstruibles conservados fuera del repositorio.

- Corrector y entrenamiento de M1: `results/corrector_final_y_m1_gpu_20261122_150345/`.
- Prueba de M1: `results/prueba_m1_gpu_20261122_185842/`.
- M2: `results/m2_gpu_20261122_192625/`.
- M3: `results/m3_gpu_20261122_193240/`.
- M4: `results/m4_gpu_20261122_194500/`.
- Procedencia y alcance: `results/publicacion_e3_20261122/` y `results/publicacion_m2_m3_m4_20261122_194500/`.

Los análisis complementarios de recepción citados en §53 se conservan localmente y no forman parte de esta subida documental. Para continuar en otro PC, usar los resultados y fuentes canónicos de las carpetas anteriores; no depender de una ruta local de recepción como si estuviera publicada.

La prioridad es revisar la novedad de M4 frente a RATL y FreshCast y aislar la contribución de su memoria. No se amplía su búsqueda general de configuraciones. El texto no tiene mejora incremental demostrada. El cierre del E3 requiere nuevas tareas —entre ellas potencia, réplicas comparables y datasets adicionales— que no se resuelven solo con los cinco ajustes disponibles.

## Preparación de archivos

Desde la raíz, con Git LFS instalado:

```console
git pull --ff-only
git lfs install
git lfs pull --include="data/processed/corpus_texto.parquet"
```

Comprobar los hashes de corpus, panel y artefactos con los manifiestos vigentes. Un puntero LFS no es el Parquet. No asumir que todas las cachés o embeddings están publicados; seguir las recetas documentadas antes de reconstruirlos.

## Documentos históricos

[Diseño de los cuatro modelos](DISENO_CUATRO_MODELOS_GPU.md), su [JSON](DISENO_CUATRO_MODELOS_GPU.json), [guía del ajuste inicial](ENTRENAMIENTO_GPU_E3.md) y [configuraciones iniciales](CONFIGURACIONES_PRUEBA_E3_GPU.json) conservan el estado del diseño del 22 de noviembre, anterior a los resultados de cierre. Sus menciones a implementación o pesos pendientes deben leerse en esa cronología. Se preservan sus bytes y hashes como procedencia; para el trabajo pendiente prevalecen el contexto actualizado y el plan de M4.

## Alcance autorizado

Esta actualización es documental. No iniciar entrenamiento, inferencia ni evaluación automáticamente al leer las guías. Los ajustes nuevos requieren solicitud expresa; no sustituir ni sobrescribir pesos o resultados históricos. No modificar ni publicar el JBook o el dashboard sin solicitud del usuario.

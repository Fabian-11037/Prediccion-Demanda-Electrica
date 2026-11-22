# Corrector semántico E3 · resultados del 22 de noviembre de 2026

Evaluación de un corrector horario sobre la base E congelada y la corrección numérica de calendario y residuos (C+R). Se completaron 33 ajustes anuales de E y 123 del corrector, con las semillas 101, 102 y 103. El entrenamiento neuronal y la codificación E5 se ejecutaron en GPU AMD mediante PyTorch 2.9.1 y ROCm 6.4.

## Resultados y alcance

Evaluación de desarrollo: 1.096 orígenes diarios de 2019–2021; 878 con anuncio oficial alineado. La prueba 2022–2025 no se evalúa en esta ejecución. El MASE promedia los errores de las semillas, no las predicciones de un ensamble.

| Procedimiento | MASE | Reducción frente a C+R |
|---|---:|---:|
| E | 0,55609 | — |
| E + C+R | 0,54671 | Referencia |
| Corrector numérico horario | 0,54186 | 0,89 % |
| Texto oficial reciente | 0,54169 | 0,92 % |
| Texto oficial y anuncios alineados | 0,54397 | 0,50 % |
| Texto oficial y prensa | 0,54270 | 0,73 % |
| Metadatos | 0,54242 | 0,78 % |

El corrector numérico mejora ligeramente a C+R: IC 95 % [0,52; 1,27] % y p de Holm = 0,001. No se demuestra un aporte del contenido textual frente a los controles numérico, de metadatos y de desplazamiento temporal, según la regla registrada. Las ganancias agregadas se concentran en 2021. La sensibilidad estricta tiene cobertura textual histórica muy reducida. Los resultados no establecen superioridad frente a TimesNet: no se ha realizado un contraste sobre los mismos orígenes y protocolo.

La inferencia utiliza 9.999 remuestreos pareados con bloques circulares de 14 días dentro de cada año y ajuste de Holm por familia. Los intervalos están condicionados a las semillas ejecutadas; no acreditan superioridad universal ni potencia sobre nuevas inicializaciones.

## Archivos

- [Informe y contrastes](resultados_corrector_e3/RESUMEN_CORRECTOR.md).
- [Contrastes estadísticos](resultados_corrector_e3/evaluacion/contrastes.csv), [MASE anual](resultados_corrector_e3/evaluacion/mase_por_anio.csv) y [resultados horarios](resultados_corrector_e3/evaluacion/por_hora.csv).
- `resultados_corrector_e3/residuos/`: resultados por año y semilla, predicciones, historiales y métricas de E.
- `resultados_corrector_e3/cr/`: objetivos, predicciones y correcciones por semilla. Los archivos del corrector conservan `delta.npy`; su predicción se reconstruye como `yhat_cr * (1 + delta)`, respetando el orden de los años y semillas del plan.
- `resultados_corrector_e3/corrector/`: configuraciones, historias por época, métricas y correcciones de búsqueda, variantes, placebos y sensibilidad estricta.
- `resultados_corrector_e3/texto*/`: resúmenes de cobertura, candidatos, intervalos y fragmentos usados.
- [Paquete de resultados](ENVIAR_resultados_corrector_20261122_0212.zip), [metadatos de ejecución](EJECUCION.json) y [manifiesto de integridad](MANIFIESTO_SHA256.json).
- `codigo/`: entrada científica, plan, fuentes, pruebas y dependencias de la ejecución.

Los vectores E5 se conservan localmente y se excluyen de esta publicación; pueden reconstruirse con el corpus y revisión del codificador fijados en el plan. El ejecutor original no exporta checkpoints neuronales de esta corrida: se publican las correcciones y los resultados efectivamente guardados.

## Reproducción

Preparar un entorno PyTorch compatible con la GPU, instalar las dependencias y copiar a `codigo/data/processed/` el panel y registro de variables indicados por el contrato. Sus huellas están en `resultados_corrector_e3/REGISTRO.json`. Recuperar el corpus activo mediante Git LFS desde `data/processed/corpus_texto.parquet` de este repositorio y verificar SHA-256 `e82f1b0fccc09ffb4deb9bd8c2dfebb7bedcc3cabc2c0fd355b2ae1d2413898c`.

Desde `codigo/`, ejecutar `python ejecutar_corrector.py --dispositivo cuda --corpus /ruta/al/corpus_texto.parquet`. La interfaz CUDA de PyTorch se utiliza también con ROCm. En esta GPU se utilizó `HSA_OVERRIDE_GFX_VERSION=10.3.0`. Las salidas se generan en una carpeta nueva; no sobrescribir los resultados publicados.

Adaptaciones locales previas al registro de huellas: desactivación del traslado automático de E a CPU por falta de memoria y observador opcional de época del corrector. No se cambiaron las configuraciones, semillas, pérdidas ni particiones del ZIP. Los hashes registrados corresponden al código ejecutado.

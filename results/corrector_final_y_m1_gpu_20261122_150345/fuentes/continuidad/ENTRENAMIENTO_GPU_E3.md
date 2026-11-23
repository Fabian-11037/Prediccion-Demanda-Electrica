# Continuación en GPU: ajuste final y prueba de E3

> Ampliación del 22 de noviembre: el diseño de los cuatro modelos propuestos está en `DISENO_CUATRO_MODELOS_GPU.md` y su JSON. Esta guía conserva el trabajo del corrector existente. La instrucción vigente es documental: no iniciar entrenamientos ni publicar sin una nueva solicitud.

Actualizado: 22 de noviembre de 2026. Leer primero `CONTEXTO_PROYECTO.md` y `AGENTS.md`. Las rutas de esta guía son relativas a la raíz del repositorio.

## Estado y objetivo

El usuario autorizó ajustar las configuraciones seleccionadas con datos anteriores a 2022 y evaluar después 2022–2025. A continuación pidió dejar las instrucciones para ejecutar ese trabajo en otro PC con GPU. **No se inició ningún entrenamiento ni evaluación de prueba en este PC.** El equipo revisado tiene Intel Core Ultra 7 155U, gráficos Intel y PyTorch 2.9.1+cpu; `torch.cuda.is_available()` devuelve falso.

El trabajo principal pendiente es la prueba del corrector semántico E3 con sus controles. La inclusión del ensamble exportable o de todas las variantes exploradas sigue sin confirmarse: no convertir esa extensión en un entrenamiento de las 24 configuraciones por iniciativa propia. La autorización no incluye publicar ni modificar JBook o dashboard.

Los paquetes recibidos no contienen pesos neuronales. `ENVIAR_pesos_exploracion_20261122_0207.zip` solo contiene `mlflow.db`. Los archivos `delta.npy` y las predicciones guardadas corresponden al desarrollo; no predicen 2022–2025. Tampoco se pueden sustituir estos modelos por checkpoints de otros protocolos solo porque compartan arquitectura.

## Material que debe llevarse al equipo con GPU

1. Clonar el repositorio `https://github.com/Fabian-11037/Prediccion-Demanda-Electrica` y recuperar el corpus con Git LFS:

```console
git lfs install
git lfs pull --include="data/processed/corpus_texto.parquet"
```

2. Recuperar las guías, configuraciones y contextos versionados en `continuidad/` mediante `git pull`. Leer primero `continuidad/README_GPU.md`. La autorización actual permite subir documentación, sin ejecutar entrenamientos ni publicar los sitios.
3. Copiar `results/exploracion_e3_20261122/` si se va a trabajar con la exploración. Sus adjuntos y configuraciones aún son locales. El corrector publicado está en `results/corrector_semantico_e3_20261122/`, commit de procedencia `3436461`.
4. Conservar el panel `data/processed/panel_horario.parquet`, `feature_registry.json`, el contrato numérico y el corpus. Verificar sus hashes antes de crear los datos. El Parquet del corpus tiene 625.577 filas, 159.086.185 bytes y SHA-256 `e82f1b0fccc09ffb4deb9bd8c2dfebb7bedcc3cabc2c0fd355b2ae1d2413898c`; el panel tiene SHA-256 `bf81ea9bce7581da75f8da8a9eccf3e0929b4117b5ef509f35bb6bf2216cab15`.
5. Validar los 432 archivos del corrector contra su `MANIFIESTO_SHA256.json` y la exploración contra `MANIFIESTO_DESCARGA.json`. Los originales se mantienen intactos.

## Entorno GPU

Crear un entorno nuevo. Instalar PyTorch con la distribución compatible con la GPU y el controlador desde el [selector oficial de PyTorch](https://pytorch.org/get-started/locally/); las versiones anteriores están en [Previous Versions](https://pytorch.org/get-started/previous-versions/). El corrector documentado se ejecutó con PyTorch 2.9.1+ROCm 6.4 en una GPU AMD de aproximadamente 12 GB. Ese es un registro de ejecución, no una garantía de compatibilidad con cualquier AMD.

Para NVIDIA, elegir CUDA compatible con el equipo; para AMD, comprobar soporte de GPU, sistema operativo y ROCm. La interfaz `torch.cuda` también se utiliza con PyTorch/ROCm. No copiar el ajuste `HSA_OVERRIDE_GFX_VERSION` de otro equipo sin verificar su compatibilidad. Instalar después las dependencias de `results/corrector_semantico_e3_20261122/codigo/requirements.txt`; registrar versiones resueltas y `pip freeze`.

Antes de la corrida, comprobar disponibilidad y una operación real en GPU:

```python
import torch
assert torch.cuda.is_available(), "No se permite continuar silenciosamente en CPU"
x = torch.ones(8, device="cuda", requires_grad=True)
x.square().sum().backward()
assert x.grad is not None and x.grad.is_cuda
print(torch.__version__, torch.version.cuda, torch.version.hip)
print(torch.cuda.get_device_name(0))
```

Medir tiempo por época y por lote de embeddings al comenzar la ejecución real, guardar progreso reanudable y actualizar la estimación con esas mediciones. Las 2–4 horas de GPU mencionadas en conversación son orientativas; no son una medición del ajuste final. Los embeddings no están incluidos y pueden dominar el tiempo.

## Configuraciones congeladas

`CONFIGURACIONES_PRUEBA_E3_GPU.json` extrae las configuraciones de los resultados efectivamente recibidos, sin una nueva búsqueda.

- Base del corrector: E, configuración 7, núcleo PatchTST, d_model 32, tres capas y cuatro cabezas. Conservar sus restantes parámetros del plan.
- Corrector: configuración 4; d=32, cuatro cabezas, dropout=0,3, lote 64, lr=0,0003 y weight_decay=0,0001. Semillas 101–103, hasta 60 épocas y paciencia 8, con restauración del mejor estado de parada.
- Variantes: num_horario, oficial_reciente, oficial, oficial_prensa y metadatos. Para atribuir un efecto al contenido, conservar los placebos de 49, 91, 154, 217 y 301 días del plan, con sus respectivos ajustes previos a 2022. La sensibilidad estricta se reporta separada.
- Exploración, si se incluye: semillas 101–105 y configuraciones de control de PatchTST, E, TimesNet y codificador periódico tomadas de sus `resultado.json`. No usar los parámetros de E2 como sustitutos.

La exploración no seleccionó una propuesta nueva: conserva E. El apilamiento exportable es una referencia prometedora, no una nueva ganadora confirmada. Sus expertos son PatchTST·control, E·control y periodica·control. El apilamiento de todas las variantes tiene otros expertos y no se reproduce entrenando solo esos tres.

## Adaptaciones pendientes antes de ejecutar

**No hay todavía un lanzador final E3 validado para 2022–2025.** `codigo/ejecutar_corrector.py` ejecuta desarrollo 2019–2021; llamarlo sin cambios repetiría esa fase. `scripts/entrenar_arquitectura.py` corresponde a otras configuraciones del benchmark y no constituye por sí solo el lanzador de este corrector.

Crear un adaptador en una carpeta nueva, conservando el código publicado sin modificar. Separar explícitamente preparación, ajuste final y predicción; registrar plan, entradas, configuraciones y huellas antes de consultar métricas de prueba. Añadir guardado y restauración de pesos, escalas y estado del optimizador al motor del corrector: actualmente solo exporta `delta.npy` y métricas. Verificar que restaurar el checkpoint reproduce las predicciones sin volver a ajustar transformaciones.

Preparar rutas portables: los datos del paquete deben apuntar al panel y contrato correctos; no reutilizar rutas absolutas del PC original. El código `DatosNumericos` espera un directorio con `config/diseno_entrenamiento_texto.json` y `data/processed/{panel_horario.parquet,feature_registry.json}`. Conservar clima, hidrología, precio, demanda y calendario del contrato común, con sus rezagos. El reanálisis no es clima conocido del futuro.

## Diseño de la prueba final que debe registrarse

El bloque es **2022-01-01 a 2025-12-31**, origen diario a las 00:00 de Colombia, historia de 168 horas y salida de 24 horas: 1.461 orígenes esperados. Verificar su cobertura y duplicados sin elegir subconjuntos por su error.

Mantener los hiperparámetros congelados. Para las bases numéricas, usar `DatosNumericos.ajuste_final()` y su reserva cronológica del 10 % para parada, con la purga del contrato. Ajustar imputación y escalas solo con el tramo de optimización. Guardar los rangos exactos. Para el corrector, registrar ajuste 2012–2020 y parada en 2021; todas sus etiquetas y estadísticas de ajuste deben ser anteriores a 2022. Esta extensión final debe quedar escrita antes de ejecutar; no afirmar que ya figuraba en el plan de desarrollo.

Reutilizar los residuos anuales fuera de muestra de E de 2011–2021, por semilla, como supervisión histórica del corrector. Ajustar C+R con los datos anteriores a 2022 y el alpha ya seleccionado por semilla para 2021 en `cr/resumen.json`; congelarlo para la prueba. No recalcular etiquetas históricas con un modelo final que entrenó sobre esos mismos días.

Congelar los pesos de E, C+R y cada corrector antes de la predicción de prueba. No reajustar anualmente usando 2022–2025. Los rasgos de errores pasados pueden actualizarse secuencialmente cuando sean observables: usar d−2 y d−8..d−2 según el código; esto no autoriza a usar errores del día objetivo ni a actualizar pesos con prueba. Mantener el arranque con historia real anterior a enero de 2022.

En TimesNet, si se incluye, usar **lote uno** en parada, predicción y prueba para evitar que la FFT mezcle orígenes. No seleccionar esa opción después de mirar las nuevas métricas.

## Texto y ensambles

Reutilizar o reconstruir vectores del codificador E5 congelado con la revisión, prefijo, pooling y normalización del contrato. No sustituirlos por hashes de prueba rápida ni por vectores del corpus congelado de 14.162 documentos: no corresponden a estos fragmentos.

Crear primero el caché de entrenamiento y parada hasta 2021; congelar mapa de fuentes, tokenización, reglas y parámetros. Ampliar después los candidatos hasta el último origen de 2025. La preparación actual se detiene antes de 2022 y hay que adaptarla. Usar un índice explícito de fuente desconocida para fuentes nuevas, dentro de la capacidad reservada de la arquitectura, sin reordenar los identificadores aprendidos.

Conservar escenario declarado, cuarentenas, ventana reciente de 72 horas, reservas de hasta 32 alineados, 48 oficiales y 48 de prensa, anuncios hasta 120 días, fragmentos de 1.500 caracteres con solape 200 y máximo 16 por documento. Solo admitir documentos cuya disponibilidad sea anterior o igual al origen. Una fecha sin hora entra al día siguiente. Los intervalos anunciados se extraen del documento ya disponible, sin usar fechas objetivo como etiquetas. La sensibilidad estricta conserva su disponibilidad real de recuperación.

Para apilamiento, si se confirma su inclusión, estimar y guardar la matriz completa de pesos por experto y por hora usando exclusivamente la parada previa a 2022. `informacion.json` conserva medias por hora, insuficientes para restaurar la matriz completa. Conservar NNLS y los expertos originales; no elegir expertos, pesos ni beta por prueba. EWA, si se incluye, actualiza pérdidas con retraso de dos días, sin anticipar observaciones.

## Salidas y evaluación estadística

Guardar en `results/prueba_e3_gpu_<fecha_hora>/`, sin sobrescribir desarrollo ni el benchmark publicado:

- Registro fechado, alcance confirmado, versiones, GPU, semillas, hashes de fuentes e insumos, divisiones cronológicas y tiempos.
- Checkpoints completos con arquitectura, pesos, optimizador, mejor época, escalas, cotas de corrección y mapa de fuentes. Copiar o referenciar mediante hashes los vectores y sus índices.
- Predicciones por semilla con origen, horizonte, observado y predicho en kWh. Exportar MASE, MAE y RMSE globales, por año y por horizonte; informar media y dispersión entre semillas. La métrica principal es MASE, con q168 del contrato y margen relativo del 3 %.
- Comparaciones pareadas entre C+R, corrector numérico y variantes textuales; metadatos y desplazamientos separan contenido de disponibilidad. Conservar 9.999 remuestreos por bloques de 14 días dentro de cada año y Holm por la familia registrada. Los IC temporales y la variabilidad entre semillas son fuentes de incertidumbre diferentes.
- Para un ensamble nuevo, registrar sus contrastes antes de evaluar y ajustar su familia; no reutilizar un p de desarrollo como evidencia de prueba. No presentar un IC puntual como equivalencia simultánea de todas las parejas.
- Informe con resultados de desarrollo y prueba claramente separados. No comparar directamente MASE de exploración 2018–2019 con MASE del corrector 2019–2021 o prueba 2022–2025.

Comprobar integridad, cobertura, restauración y disponibilidad temporal. Compartir primero los resultados con el usuario; no actualizar ni publicar JBook o dashboard sin una nueva solicitud.

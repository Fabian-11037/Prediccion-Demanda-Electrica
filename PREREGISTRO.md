# Preregistro de los modelos base (versión 1)

Fecha de redacción: 2026-11-07, **antes** de leer el conjunto de prueba y antes de ejecutar la búsqueda de hiperparámetros con los datos reales. Antes de este documento solo se ejecutaron pruebas unitarias y pruebas de humo (muestras pequeñas, pocas épocas) para verificar que el flujo corre; sus cifras no se usaron para decidir nada. Este archivo se versiona en git antes de ejecutar las etapas `buscar` y `final`; el historial de commits es la evidencia de la fecha.

## 1. Problema
- Objetivo: demanda real horaria del SIN (`demanda_real_kwh`), pronóstico directo multi-salida de H = 24 horas, historia L = 168 horas, un origen por día a las 00:00 (hora de Colombia).
- Datos: `data/processed/panel_horario.parquet` (2000-01-01 a 2025-12-31). Entradas as-of con los rezagos de publicación de `src/demanda/config.py` (supuestos conservadores, no certificados por la fuente).
- Covariables de los modelos base: demanda rezagada y calendario conocido a futuro (festivos, hora, día de la semana, Semana Santa, temporada navideña). Sin texto, sin clima, sin precio, sin hidrología, sin ISE.

## 2. Partición y validación
- **Prueba:** 2022-01-01 a 2025-12-31, congelada. Se lee una sola vez por modelo, en la etapa `final`. Declaración: los valores de 2022-2025 ya se habían visto en exploraciones descriptivas de este material (medias por año), antes de este preregistro; la prueba no es prospectiva.
- **Selección de hiperparámetros:** validación cruzada de ventana expansiva (forward chaining) con 4 pliegues anuales de validación (2018, 2019, 2020, 2021). En cada pliegue se entrena con todos los orígenes cuyo objetivo termina antes del 1 de enero del año validado (purga = 24 h); el último 10 % cronológico de esos orígenes (con purga) se reserva solo para parada temprana. Criterio: MASE medio entre pliegues.
- 2020 (pandemia) entra como pliegue. Se declara como limitación: el MASE medio queda influido por un año atípico (se reportan también los MASE por pliegue).
- **Ajuste final:** todos los orígenes cuyo objetivo termina antes de 2022-01-01, con el último 10 % para parada temprana; evaluación en la prueba.
- Normalización: demanda dividida por su media en la ventana; calendario estandarizado con estadísticas de 2000-2017.

## 3. Modelos y búsqueda (mismo esfuerzo para los tres)
Modelos: (1) ARX lineal con penalización L2 o L1, (2) MLP, (3) LSTM, todos en PyTorch (AdamW, pérdida MSE, recorte de gradiente 1.0, tasa de aprendizaje 1e-3). Búsqueda aleatoria con el mismo número de configuraciones por modelo (ver `results/base/entorno_buscar.json`), 1 semilla (1), semilla de muestreo 1. Espacios:
- ARX lineal: penalización {ridge, lasso}; lambda log-uniforme (ridge 1e-6..1e-1; lasso 1e-7..1e-2); rezagos {24, 48, 96, 168}; batch {32, 64, 128, 256}.
- MLP: capas {1, 2, 3}; unidades {64, 128, 256, 512}; dropout U(0, 0.5); weight decay log-uniforme 1e-6..1e-2; batch {32, 64, 128, 256}.
- LSTM: capas {1, 2, 3}; unidades {32, 64, 128, 256}; dropout U(0, 0.5); weight decay log-uniforme 1e-6..1e-2; batch {32, 64, 128, 256}.
Presupuesto: 12 configuraciones por modelo, igual para los tres. Épocas máximas 60 y paciencia 8 para los tres.

**Orden de ejecución (declarado):** el LSTM es demasiado lento en CPU (más de 80 s por época con datos completos) y se ejecutará después, en GPU, con las mismas opciones. Se ejecutan primero ARX lineal y MLP. La configuración del LSTM se elegirá solo con la validación cruzada, sin mirar la prueba de los otros modelos; no se modificará el espacio de búsqueda, el presupuesto ni las opciones del LSTM después de ver resultados de prueba. La prueba se lee una vez por modelo, con su configuración ya fijada.

## 4. Métricas
- Primaria: MASE medio sobre h = 1..24 con escala = MAE del ingenuo de 168 h en 2000-2017. Secundarias: RMSE, MAE, sMAPE, MAPE, MASE en h = 1, 12, 24.
- Referencias (no cuentan entre los tres modelos base): ingenuo de 24 h e ingenuo de 168 h con la misma disponibilidad de datos que los modelos.
- Advertencia: la escala del MASE está en kWh de 2000-2017 y la demanda crece con el tiempo; el MASE de la prueba no es comparable con el de la validación. Se comparan modelos solo dentro de una misma partición.

## 5. Semillas e inferencia
- Piloto (5 semillas, 100-104) y estimación del número de ejecuciones por potencia (α = 0.05, potencia 0.80, diferencia relativa de interés δ = 3 %, **valor provisional**: el autor no lo ha fijado). **Desviación declarada:** si la corrida final se ejecuta antes del piloto, se usan 5 semillas (1000-1004), el mínimo exigido, y se declara que el tamaño de muestra no está sustentado por el piloto.
- Resultados finales: media, desviación estándar e IC 95 % (t) del MASE de prueba por modelo. Las pruebas estadísticas entre modelos (Holm, Diebold-Mariano, TOST) no están incluidas en este preregistro y se preregistrarán antes de aplicarlas.
- «No significativo» no se interpreta como equivalencia.

## 6. Determinismo y entorno
Semillas fijadas (random, numpy, torch). Determinismo estricto de PyTorch por defecto. El dispositivo, versiones y commit de cada corrida quedan en `results/base/entorno_*.json`. Los resultados de CPU y GPU no coinciden bit a bit.

## 7. Lo que NO está decidido
Si el MLP reemplaza definitivamente a LightGBM como segundo modelo base; la justificación de los tres modelos con la revisión de literatura; el tratamiento de valores anómalos de demanda (2007-04-26); el bloque de prueba prospectivo (2026).

---

# Versión 2 (2026-11-07, redactada antes de cualquier nueva ejecución)

Esta versión **no modifica** la versión 1 ni sus resultados (`results/base/`, 60 épocas máximas), que se conservan y se reportan como «v1».

## Cambios decididos
1. **Épocas máximas: 200** (antes 60) para los tres modelos, manteniendo la parada temprana sobre la pérdida de validación (paciencia 8, sustituida por el hiperparámetro `paciencia` en la ampliación de más abajo). Motivo: en la v1 dos semillas del ARX lineal terminaron en el tope de 60 épocas con la mejor época en el límite (posible falta de convergencia). El motivo se observó en las curvas de entrenamiento, pero la decisión se toma después de haber leído la prueba de la v1 para ARX lineal y MLP; se declara como posible contaminación leve.
2. Las corridas de la v2 escribirán en un directorio distinto (`--salida results/base_v2`) para no sobrescribir la v1.
3. Si se amplía el espacio de hiperparámetros (ver `CONTEXTO_PROYECTO.md`, «Opciones de hiperparametrización sugeridas»), el cambio se aplicará a los tres modelos con el mismo presupuesto y se anotará aquí **antes** de ejecutar.
4. La prueba se volverá a leer en la v2 para ARX lineal, MLP y LSTM. Los resultados de la v2 no reemplazan a los de la v1: se reportan ambos y se declara que la v2 se diseñó después de ver la prueba de la v1.

## Ampliación de la v2 (2026-11-07, antes de ejecutar la v2; sigue a la lectura de la prueba de la v1 para ARX lineal y MLP)
Decisiones del autor y su implementación en el código (commit que acompaña a este texto). Se aplican **a los tres modelos con el mismo presupuesto**:
1. **Hiperparámetros de optimización añadidos al espacio** (ARX lineal, MLP y LSTM; se suman a los de la v1):
   - tasa de aprendizaje `lr`: log-uniforme 1e-4..1e-2 (en la v1 fija en 1e-3);
   - planificador `scheduler`: {ninguno, coseno (T_max = épocas máximas), reducción en meseta (factor 0.5, paciencia = max(2, paciencia/3))};
   - pérdida `perdida`: {MSE, MAE, Huber}; delta de Huber = 0.05 con normalización «nivel» y 0.5 con «global» (unidades de la pérdida normalizada; regla fijada antes de ver resultados);
   - paciencia de la parada temprana `paciencia`: {8, 15, 25}.
   Tamaño del espacio: ARX lineal 8 hiperparámetros, MLP 9, LSTM 9. El optimizador (AdamW), el recorte de gradiente (1.0), la activación y la arquitectura no se buscan.
2. **Parada temprana:** mejora mínima `min_delta` = 1e-6 sobre la pérdida de validación (en la v1: 1e-8); épocas máximas 200.
3. **Presupuesto:** **20 configuraciones por modelo** (v1: 12) y **3 semillas por configuración** en la búsqueda (semillas de búsqueda 1, 2, 3; se promedia el MASE de validación cruzada; v1: 1 semilla). Mismo presupuesto para los tres modelos. Criterio de selección: MASE medio entre pliegues y semillas.
4. **δ = 3 %** (diferencia relativa de interés) **confirmado por el autor** para el margen de equivalencia (TOST/ROPE) y la estimación de potencia (α = 0.05, potencia 0.80).
5. Resultados de la v2 en `results/base_v2` (no se sobrescribe `results/base`). Las pruebas se ejecutan en la GPU del autor; en este entorno **no** se entrenó nada con estos cambios, solo pruebas unitarias y un humo de 2 épocas con 40 orígenes (sus cifras no se usaron).
6. Como la v2 se diseñó después de ver la prueba de la v1 (ARX lineal y MLP), la comparación v1–v2 no es una validación independiente.

## Pendiente de preregistrar antes de ejecutar
- Pruebas estadísticas entre modelos (ver `CONTEXTO_PROYECTO.md`, «Preregistro de las pruebas estadísticas»), usando δ = 3 %.
- Piloto de 5 semillas y número de ejecuciones finales por potencia (con δ = 3 %).

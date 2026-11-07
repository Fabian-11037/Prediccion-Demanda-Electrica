# Contexto del proyecto (para cualquier agente o persona que continúe el trabajo)

Idioma de trabajo: **español**. Autor: Fabian José Manrique Morelos (Maestría en Estadística Aplicada, Universidad del Norte).

## 0. Regla de redacción (obligatoria)
Este proyecto es **independiente de cualquier trabajo anterior del autor**. En los artículos, cuadernos, informes, README, dashboards, presentaciones y código que se desarrollen aquí **no se debe mencionar, citar, comparar ni insinuar ningún proyecto previo** (otro objetivo de pronóstico, otro conjunto de datos, otro repositorio, resultados o decisiones tomadas en él). Tampoco se copian textos, tablas, figuras, cifras ni diccionarios de datos de ese trabajo. Todo se redacta como si este fuera el único proyecto: el problema es el pronóstico de la **demanda real de energía eléctrica**, los datos son los de este repositorio y las decisiones son las que quedan documentadas aquí. Si algún material heredado menciona otro objetivo (p. ej. en metadatos de datos), se reescribe antes de usarlo. `tests/test_higiene_redaccion.py` revisa que el repositorio no contenga referencias a nombres del trabajo anterior; no lo desactives.

## 1. Objetivo
Proyecto final del curso de Deep Learning. Meta global: un **modelo original y novedoso que combine modelos de series de tiempo con NLP** (variables temporales + texto) para mejorar el pronóstico de la **demanda real de energía eléctrica del Sistema Interconectado Nacional (SIN) de Colombia**. Hay tres entregables encadenados; el conjunto de datos, la partición, las métricas y los márgenes de equivalencia que se fijen en el primero se heredan en los siguientes.

| Entregable | Qué pide (resumen) |
|---|---|
| **E1** Estado del arte, EDA y 3 baselines | EDA exhaustivo, extracción de 3 modelos base de la literatura, piloto, ≥5 semillas, benchmark, baseline e inferencia, Jupyter Book. Los baselines no tienen que ser redes neuronales. |
| **E2** Benchmark de 10 modelos | Top-10 (≥4 familias) + baselines, replicar un paper guía e identificar su limitación, preregistro, auditoría de fuga, búsqueda de 1000 arquitecturas en dos fases, ablación (≥3 componentes), robustez (≥3 niveles), potencia, Friedman/Holm/MCS/DM. |
| **E3** Modelo original | Mecanismo, pérdida o entrenamiento nuevos; implementación propia; superioridad significativa frente al Top-10 y baselines fuertes en el conjunto principal y en ≥2 de 3 conjuntos adicionales de naturaleza distinta (elegidos antes de ver resultados); demostración formal de una propiedad; dashboard público; informe; sustentación. |

Las pautas oficiales están en el proyecto de claude.ai «DL Project» (`Entregable{1,2,3}DeepLearningProject.pdf`).

## 2. Problema definido
- **Objetivo (confirmado):** demanda real (`demanda_real_kwh`, kWh por intervalo horario del SIN).
- **Propuesta por confirmar:** frecuencia horaria, ámbito nacional (SIN), horizonte H = 24 horas, historia L = 168 horas (una semana). Origen `t0` en hora entera; el pronóstico cubre los intervalos que empiezan en `t0, …, t0+23h`. Origen diario a las 00:00 por defecto (`paso_h=24`).
- **Métrica primaria sugerida:** MASE con escala del ingenuo estacional (24 h o 168 h) calculada solo con entrenamiento; secundarias RMSE, MAE y sMAPE por horizonte y tipo de día. Hay tendencia creciente de la demanda (demanda horaria media de ≈4.6 GWh en 2000 a ≈9.5 GWh en 2025), por lo que la escala absoluta de entrenamiento infla el error en periodos tardíos: comparar modelos solo dentro de una misma partición o usar una escala relativa al nivel.
- **Tipo de problema:** series de tiempo (usa rezagos del objetivo).
- **Modelos base (E1), decididos:** (1) regresión lineal ARX con penalización L2 (ridge) o L1 (lasso) sobre rezagos de demanda, resúmenes de covariables y calendario del horizonte, tipo LEAR; (2) MLP; (3) LSTM. **Todos se entrenan en PyTorch** (GPU si está disponible). Inicialmente se había elegido LightGBM como segundo modelo, pero no se entrena con PyTorch; queda como opción en CPU (`--modelos lightgbm`) y **debe confirmarse con el autor si el MLP lo reemplaza definitivamente**. Los modelos base deben justificarse con la revisión de literatura (el ingenuo estacional y la persistencia se reportan como referencia aparte). Nota: el L1 del ARX se optimiza por descenso de gradiente con penalización (no coordenadas cíclicas), por lo que no produce ceros exactos; declararlo.
- **Pronóstico directo multi-salida** (una salida por hora del horizonte). Normalización por defecto: demanda dividida por su media en la ventana (`--normalizacion nivel`), que absorbe la tendencia de largo plazo; el resto de entradas se estandariza con estadísticas de entrenamiento.

## 3. Datos (carpeta `data/`)
Inventario y detalle en `data/raw/README.md` y `data/external/README.md`.

| Bloque | Cobertura | Estado |
|---|---|---|
| Demanda real, demanda comercial, precio de bolsa (XM) | horario 2000-01-01 → 2025-12-31 23:00, 227 928 filas, sin huecos ni NaN | incluido (`data/raw/numeric/hourly`) |
| Clima observado: temperatura, humedad, precipitación en 8 ciudades (ERA5 vía Open-Meteo) | igual | incluido; **reanálisis, no pronóstico** |
| Habitantes por ciudad (censo 2018 del DANE ajustado por omisión censal) | estático | incluido en `data/raw/numeric/ciudades.csv`; cifras de **fuente secundaria** (Portafolio, 17-oct-2019, citando al DANE); sustituir por el archivo oficial |
| Hidrología diaria (llenado, reserva útil, capacidad útil, aportes) | 2000-01-01 → 2025-12-31 | incluido |
| Titulares de prensa (7 medios) y boletines IDEAM | 568 295 titulares, 2004-2025 | incluido (`data/raw/noticias/titulares.parquet`) |
| Festivos de Colombia | 2000-2035 | generado (`data/external/festivos_colombia.csv`) |
| ISE (DANE), actividad económica | mensual | **pendiente de descarga** (script de importación listo, sin probar con el archivo real) |
| Pronósticos de temperatura con emisión previa | GFS desde 2021-03; resto desde 2024-01 | **pendiente de descarga** (script listo, sin probar) |

Los datos de 2026 (hasta la fecha) **no están incluidos**; ver §6.

### Tabla consolidada
`python scripts/construir_panel.py` genera en `data/processed/`: `panel_horario.parquet` (índice = inicio del intervalo, hora de Colombia), `feature_registry.json` (rol, clase de disponibilidad, rezago y fuente de cada columna), `calidad_datos.json` y `cobertura_titulares_{estricto,declarado}.csv`.

Prefijos de columnas:
- `y_demanda_real_kwh`: objetivo; solo se lee como etiqueta del horizonte.
- `x_*`: entradas **as-of** (solo pasado, con rezago de publicación ya aplicado): demanda real rezagada, precio de bolsa, clima observado por ciudad, clima ponderado por habitantes (`x_obs_*_ponderada`, promedio de las 8 ciudades con pesos del censo 2018; pesos estáticos, declarar que el censo es posterior a parte del periodo), hidrología.
- `cal_*`: calendario conocido a futuro (hora, día de la semana, festivos, víspera, puentes, Semana Santa, temporada navideña, codificaciones cíclicas).
- `aux_*`: solo auditoría; no usar como entrada.

## 4. Control de fuga de información (reglas que NO se deben romper)
1. **Clases de disponibilidad:** `known_future` (calendario, festivos), `past_only` (demanda, precio, clima observado, hidrología, ISE), `oracle_only` (clima ERA5 de la hora objetivo: prohibido como entrada), `excluded_input` (`demanda_comercial_kwh`: se deriva de la demanda real más pérdidas y se liquida después; redundante y con riesgo de fuga).
2. **Rezagos de disponibilidad** (supuestos conservadores en `src/demanda/config.py`, a declarar en el preregistro): demanda 1 h, precio de bolsa 24 h, clima observado 1 h, hidrología 24 h tras terminar el día, ISE fin de mes + 60 días, pronósticos meteorológicos emisión + margen 6 h + latencia 6 h. El panel ya trae estos rezagos aplicados: una ventana que termina en la fila `p-1` es usable en el origen `t0 = índice[p]`.
3. **Ventanas por origen** (`src/demanda/ventanas.py`): las columnas `x_*` nunca se leen en filas `>= p`; solo `cal_*` y pronósticos con emisión previa se leen en el horizonte; `y_*` solo como etiqueta. Prueba que lo verifica corrompiendo el futuro.
4. **Partición cronológica con purga** (`src/demanda/particion.py`): cada intervalo-objetivo queda completo dentro de una partición (purga = H). **El conjunto de prueba está congelado**: `origenes(..., 'test')` exige `allow_test=True` y solo debe invocarse en la corrida final.
5. **Escalas, imputación, vocabulario y climatología se ajustan solo con entrenamiento** (`AjustadorEscala`, `ClimatologiaTemperatura`; hay pruebas que alteran validación/prueba y verifican que no cambia nada).
6. **Texto:** solo titulares. Dos escenarios aplicados igual en todas las particiones: *estricto* (versión archivada verificada; disponibilidad = captura verificada; cubre desde 2021-04-25) y *declarado* (fecha declarada; sin hora → día siguiente 00:00 de Colombia; no certifica que la versión del titular sea la original). Nunca se usan como característica `url`, fechas declaradas, `availability_*`, `provenance_source`. Registros en cuarentena y sin fecha quedan fuera. Con el escenario *declarado* no se puede afirmar «libre de fuga»: declararlo como limitación.
7. **Clima:** el clima de `data/raw` es reanálisis. Como covariable del futuro solo valen pronósticos con emisión conocida (cobertura corta) o la climatología de entrenamiento. No interpolar pronósticos hacia atrás con ERA5.
8. Los preregistros se escriben **antes** de buscar, hacer piloto o ejecutar, y se versionan.
9. Mismo esfuerzo de búsqueda para todos los modelos. «No significativo» no es equivalencia (usar TOST/ROPE). Reportar p exactos, ajustados y sin ajustar, IC y tamaño de efecto.
10. Antes de afirmar algo sobre literatura, verificarlo en el texto completo. No inventar referencias ni DOI. Las cifras de los textos deben salir de celdas ejecutadas.

## 5. Partición (PROPUESTA, no preregistrada)
Train 2000-2017, validación 2018-2021, prueba 2022-2025 (`PARTICION` en `config.py`). Observaciones:
- 2020-2021 (pandemia) cae en validación: la validación no es representativa del régimen posterior.
- Los valores 2022-2025 ya se han mirado en exploraciones descriptivas de este material. Si se usan como prueba final, hay que declararlo. **Alternativa recomendada:** bloque prospectivo desde 2026-01-01 (descargar 2026 de XM y Open-Meteo antes de preregistrar) y dejar 2022-2025 como validación o prueba secundaria.
- La disponibilidad de texto es muy desigual entre particiones (ver §7). Para experimentos con texto conviene fijar a priori el periodo y las fuentes.

## 6. Pendientes y decisiones abiertas

### Datos que faltan
| Dato | Por qué falta | Qué hacer |
|---|---|---|
| ISE (DANE), actividad económica | El entorno de desarrollo no alcanza el DANE | Descargar el anexo del ISE y correr `scripts/adquisicion/importar_ise.py` (**sin probar con el archivo real**; revisar el orden de columnas) |
| Pronósticos de temperatura con emisión previa | El entorno no alcanza Open-Meteo | Correr `scripts/adquisicion/descargar_pronosticos_temperatura.py` (**sin probar**); comprobar que `previous_dayN` entrega corridas distintas y revisar huecos. Cobertura: GFS desde 2021-03, resto desde 2024-01 |
| Datos de 2026 (demanda, precio, hidrología, clima, titulares) | El conjunto termina el 2025-12-31 | Necesarios si se adopta el bloque de prueba prospectivo (§5) |
| Habitantes oficiales (DANE) y proyecciones anuales | Solo se obtuvo el censo 2018 ajustado por una fuente secundaria | Descargar las proyecciones municipales del DANE y decidir si la población es estática o anual |
| Pronóstico oficial de demanda de XM | No se verificó que exista un archivo histórico con fecha de emisión | Verificar antes de usarlo como covariable o comparador (hoy está fuera) |
| Horas reales de publicación (demanda, precio, hidrología) | La fuente no las certifica | Los rezagos de `config.py` son supuestos; documentarlos o medirlos |
| `PREREGISTRO.md` | Aún no se ha escrito | **Obligatorio antes de la etapa `final`**: el script lo exige |

### Decisiones abiertas
1. Confirmar frecuencia (horaria), horizonte (24 h) y ámbito (SIN).
2. Confirmar que el MLP reemplaza a LightGBM como segundo modelo base (§2) y justificar los tres con la literatura.
3. Elegir el bloque de prueba (¿prospectivo 2026?) y preregistrar, incluyendo el delta relativo de interés (el script usa 3 % por defecto solo como valor de partida).
4. Definir el tratamiento de valores anómalos de demanda (p. ej. 2007-04-26 10:00-12:00 cae a 6-39 % de lo esperable; ver `data/processed/calidad_datos.json`). Los datos **no se corrigieron**.
5. Decidir si los titulares (metadato con derechos del editor) se mantienen en un repositorio público; solo se incluye el titular y la URL únicamente de registros con versión archivada verificada.

### Preregistro de las pruebas estadísticas (pendiente, obligatorio antes de aplicarlas)
Hay que escribirlo, fecharlo y versionarlo antes de calcular cualquier prueba entre modelos. Debe fijar como mínimo:
- **Comparaciones:** los 3 contrastes pareados entre modelos base (ARX–MLP, ARX–LSTM, MLP–LSTM) y cada modelo frente al ingenuo de 168 h; familia de hipótesis para la corrección.
- **Unidad de independencia y apareamiento:** origen de pronóstico (cada día a las 00:00) para las pruebas sobre errores; semillas apareadas por índice para las pruebas sobre el MASE por semilla; declarar la dependencia serial entre orígenes consecutivos (bootstrap por bloques o errores estándar robustos).
- **Pruebas:** Diebold-Mariano con corrección de Harvey-Leybourne-Newbold por horizonte (h = 1, 12, 24 y media); prueba sobre semillas (t pareada o Wilcoxon, justificar); corrección de Holm; equivalencia con TOST o ROPE usando el margen δ (diferencia relativa de interés) que fije el autor — «no significativo» no es equivalencia.
- **Reporte:** p exactos, ajustados y sin ajustar, IC (bootstrap por bloques), tamaño de efecto (p. ej. delta de Cliff o diferencia relativa de MASE) y potencia α = 0.05, 0.80.
- **Qué datos:** solo la prueba, una vez, con las predicciones guardadas en `results/*/preds/test.npz`; sin repetir la prueba para ajustar.
- **Para E2/E3:** Friedman con post hoc sobre semillas, Model Confidence Set y las exigencias de la pauta (≥3 pruebas por contraste, |Cliff δ| ≥ 0.147 en E3); incluirlas cuando corresponda.

### Opciones de hiperparametrización sugeridas (no implementadas)
Cualquier opción nueva se aplica a los tres modelos con el mismo presupuesto y se anota en el preregistro antes de ejecutar. Por prioridad:
1. **Tasa de aprendizaje y planificador** (log-uniforme 1e-4..1e-2; coseno, reducción en meseta o calentamiento): hoy `lr = 1e-3` es fijo y es el parámetro que más afecta la convergencia del ARX y del LSTM.
2. **Parada temprana:** paciencia {8, 15, 25} y `min_delta`; con 200 épocas conviene evaluar paciencia mayor.
3. **Optimizador:** AdamW / Adam / SGD con momento (y `betas`); recorte de gradiente {0.5, 1, 5}.
4. **Función de pérdida:** MSE / MAE / Huber (con `delta`); con demanda, Huber o MAE suelen ser más robustos a valores anómalos.
5. **Arquitectura MLP:** activación {ReLU, GELU, SiLU}, normalización (LayerNorm/BatchNorm), conexiones residuales, ancho decreciente.
6. **Arquitectura LSTM:** bidireccional en la historia, agregación por atención sobre todos los pasos (en vez del último estado), dropout recurrente, tamaño de la rama de calendario.
7. **ARX:** elastic net (`l1_ratio`), rezagos selectivos (24, 48, 72, 168, 336) en vez de ventana contigua, interacciones calendario × hora.
8. **Datos:** longitud de ventana `L` {72, 168, 336}, conjunto de covariables (`serie`, `calendario`, `completo`), modo de normalización (`nivel`, `global`), paso entre orígenes de entrenamiento (24 h o 6 h para aumentar muestras).
9. **Búsqueda:** subir `--n-config` a 20-30 o usar reducción sucesiva (successive halving) con presupuesto igual por modelo; promediar 2-3 semillas por configuración en los candidatos finales para reducir el ruido de selección (hoy la selección usa 1 semilla y la varianza entre semillas, sd ≈ 0.045-0.050, es comparable a las diferencias entre modelos).

### Trabajo por hacer
- E1: EDA, extracción de 3 modelos base de la literatura, piloto, ≥5 semillas, baseline e inferencia, Jupyter Book; completar la revisión de literatura (cadenas booleanas con fecha, bases de datos, exclusiones, tabla de síntesis con DOI).
- E2/E3: Top-10, paper guía, auditoría de fuga del texto, búsqueda de 1000 arquitecturas, modelo original (ideas candidatas: compuerta de utilidad del texto guiada por incertidumbre con propiedad de «no empeorar»; pérdida de texto placebo; núcleo de respuesta al impulso de noticias; pronóstico conformal condicionado por texto — confirmar novedad con búsqueda dirigida).

### Estado del código de entrenamiento y resultados
- **Ejecutado (2026-11-07, CPU):** búsqueda con validación cruzada forward chaining (pliegues anuales 2018-2021, ventana expansiva, purga de 24 h, parada temprana con el último 10 % del entrenamiento) y etapa final sobre la prueba para **ARX lineal y MLP** (12 configuraciones cada uno, 5 semillas finales 1000-1004). No hubo piloto previo: desviación declarada en `PREREGISTRO.md`. Todo en `results/base/`.
- **El LSTM no se ha ejecutado** (más de 80 s por época en CPU): debe correrse en GPU con las mismas opciones, sin cambiar su espacio ni su presupuesto (`buscar --modelos lstm` y luego `final --modelos lstm --allow-test`). **La prueba ya fue leída para ARX lineal y MLP**: no ajustar nada del LSTM en función de esos resultados.
- La validación cruzada **no** usa el paquete `timeseries-cv` (`tsxv`): su forward chaining es univariado, produce una ventana de validación por pliegue y no admite purga ni covariables; se implementó el mismo esquema en `src/demanda/particion.py`.

#### Configuraciones elegidas (MASE medio en los 4 pliegues)
| Modelo | Configuración | MASE CV | Por pliegue 2018 / 2019 / 2020 / 2021 |
|---|---|---|---|
| ARX lineal | lasso, lambda 4.45e-6, rezagos 168, batch 256 | 0.724 | 0.624 / 0.601 / 0.904 / 0.767 |
| MLP | 2 capas, 128 unidades, dropout 0.32, weight decay 2.6e-3, batch 32 | 0.684 | ver `results/base/busqueda.csv` |
| Ingenuo 168 h (referencia) | — | 1.142 | 1.079 / 1.113 / 1.157 / 1.218 |
| Ingenuo 24 h (referencia) | — | 1.836 | 1.874 / 1.902 / 1.737 / 1.830 |

#### Métricas en la prueba 2022-2025 (5 semillas; media, desviación estándar, IC 95 % t)
| Modelo | MASE | IC 95 % | RMSE (kWh) | MAE (kWh) | sMAPE % | MASE h=1 | h=12 | h=24 |
|---|---|---|---|---|---|---|---|---|
| ARX lineal (lasso) | 0.878 (sd 0.045) | 0.822-0.933 | 248 656 | 189 230 | 2.07 | 0.566 | 1.161 | 0.820 |
| MLP | 0.867 (sd 0.050) | 0.805-0.930 | 239 011 | 186 952 | 2.05 | 0.779 | 1.115 | 1.051 |
| Ingenuo 168 h | 1.355 | — | 444 632 | 292 142 | 3.21 | 1.080 | 1.597 | 1.148 |
| Ingenuo 24 h | 2.077 | — | 659 593 | 447 779 | 4.98 | 1.024 | 2.841 | 1.665 |

Lectura permitida con la evidencia actual: ambos modelos superan con claridad a los ingenuos; ARX lineal y MLP **no se distinguen** con 5 semillas (IC 95 % solapados; no se ha aplicado ninguna prueba de hipótesis ni de equivalencia). A h=1 y h=24 el ARX es mejor en MASE medio; a h=12 y en RMSE el MLP es algo mejor. El MASE de prueba no es comparable con el de validación (escala en kWh de 2000-2017 y demanda creciente: el ingenuo de 168 h pasa de 1.14 en CV a 1.35 en la prueba).

#### Limitaciones de estas corridas
- Sin piloto previo: 5 semillas son el mínimo, no un tamaño sustentado por potencia.
- Dos semillas del ARX (1001 y 1002) llegaron al tope de 60 épocas con la mejor época en el límite: el ARX puede estar sin converger. **Decisión del autor (2026-11-07): mantener la parada temprana y subir las épocas máximas a 200** para los tres modelos (ya es el valor por defecto del script; está anotado como versión 2 en `PREREGISTRO.md`). La v2 debe ejecutarse en `--salida results/base_v2` para conservar los resultados v1 (60 épocas), y se reportan ambos.
- La varianza entre semillas del ARX y del MLP es alta (sd ≈ 0.045-0.050) frente a la diferencia entre ellos.
- 2020 (pandemia) es un pliegue de selección; el ARX lasso tiene su peor pliegue en 2020.
- Los valores de 2022-2025 ya se habían visto en exploraciones descriptivas antes del preregistro.
- Solo calendario y demanda como covariables; clima, precio, hidrología, ISE y texto no se han probado.
- Sin pruebas estadísticas entre modelos (Holm, Diebold-Mariano, TOST): pendientes y deben preregistrarse.
- **La ruta de GPU no se ha probado** (el entorno de desarrollo no tiene GPU). Con CUDA, el determinismo estricto exige `CUBLAS_WORKSPACE_CONFIG` (el código lo define); los resultados de CPU y GPU no coinciden bit a bit: declarar el dispositivo en cada corrida (`entorno_*.json` lo registra).
- Tiempos medidos en CPU de 2 núcleos: búsqueda de 12 configuraciones con 4 pliegues ≈ 14 min para ARX lineal + MLP; ajuste final ≈ 4 s (ARX) y ≈ 17 s (MLP) por semilla; LSTM > 80 s por época.

## 7. Limitaciones de datos conocidas (declararlas en los artículos)
- **Cobertura de texto desigual:** 2008-2019 es casi solo El Colombiano (sitemap, fecha declarada sin hora); desde 2020 hay otras fuentes con volúmenes muy distintos por año (Semana: ≈17.6 mil en 2022 y ≈0.1 mil en 2025); los boletines IDEAM suben en 2024-2025. El cambio de composición de fuentes entre entrenamiento y prueba es un riesgo de sesgo de distribución, no de fuga.
- El escenario estricto de texto (62 572 titulares) empieza en 2021-04-25.
- Las horas de publicación históricas de demanda, precio e hidrología no están certificadas; los rezagos son supuestos. XM puede revisar cifras históricas.
- El ISE se revisa; una serie descargada hoy tiene sesgo de *vintage*.
- ERA5 es reanálisis y no representa lo que un operador habría tenido como pronóstico.

## 8. Estructura del repositorio
```
CONTEXTO_PROYECTO.md    este archivo (leer primero)
data/raw/               numeric/{hourly,hydrology,ciudades.csv}, noticias/titulares.parquet, README.md
data/external/          festivos_colombia.csv, actividad_economica/, pronosticos_temperatura/, README.md
data/processed/         panel_horario.parquet, feature_registry.json, calidad_datos.json, cobertura_titulares_*.csv
src/demanda/            config, calendario, consolidar, particion, ventanas, noticias, actividad_economica,
                        pronosticos, climatologia, metricas, modelos_base
scripts/                construir_panel.py, generar_festivos.py, entrenar_modelos_base.py,
                        adquisicion/{descargar_pronosticos_temperatura,importar_ise}.py
results/base/           (se crea al entrenar) busqueda.csv, mejor_config.json, piloto*.csv/json, final_runs.csv,
                        resumen_final.csv, hist/, preds/, entorno_*.json
tests/                  calendario, fuga del panel y las ventanas, particiones, externos, adquisición, modelos base, higiene de redacción
```

## 9. Cómo ejecutar
```bash
pip install -r requirements.txt
python scripts/generar_festivos.py        # festivos
python scripts/construir_panel.py         # panel as-of + registro + cobertura
pytest -q tests                           # pruebas de fuga, calendario y pipeline de entrenamiento
```
Ejemplo de ventanas sin fuga:
```python
import json, pandas as pd
from demanda.particion import origenes, AjustadorEscala
from demanda.ventanas import construir_ventanas, columnas_por_clase
panel = pd.read_parquet("data/processed/panel_horario.parquet")
reg = json.load(open("data/processed/feature_registry.json"))
x, k = columnas_por_clase(reg, "past_only"), columnas_por_clase(reg, "known_future")
esc = AjustadorEscala().ajustar(panel, x + ["y_demanda_real_kwh"])
w = construir_ventanas(panel, origenes(panel.index, "train"), x, k, escala=esc)
```

### Entrenamiento de los modelos base (`scripts/entrenar_modelos_base.py`)
Etapas, en orden: `listar` (espacios y opciones) → `buscar` → `piloto` → `final`.
```bash
python scripts/entrenar_modelos_base.py listar
python scripts/entrenar_modelos_base.py buscar --n-config 12 --covariables calendario      # validación; 1 semilla; mismo presupuesto para los 3
python scripts/entrenar_modelos_base.py piloto                                              # 5 semillas (100-104) sobre validación; estima n de semillas
python scripts/entrenar_modelos_base.py final --allow-test --prereg PREREGISTRO.md          # PRUEBA: una sola vez, tras preregistrar
```
Opciones principales (todas con `--help`):
- **Hiperparámetros buscados** (búsqueda aleatoria; `--n-config` igual para todos): ARX lineal: penalización (ridge/lasso), lambda (rango según penalización), rezagos de demanda {24, 48, 96, 168}, batch size. MLP y LSTM: capas {1,2,3}, unidades, dropout [0, 0.5], weight decay (log), batch size.
- **Fijar o ampliar:** `--fijar modelo.param=valor` (repetible) saca un parámetro de la búsqueda; `--espacio ruta.json` reemplaza espacios.
- **Entrenamiento (no se buscan):** `--max-epocas` (por defecto 200 desde la v2; la v1 usó 60), `--paciencia` (parada temprana sobre la pérdida de validación), `--lr`, `--perdida {mse,mae,huber}`, `--clip`, `--max-arboles` (LightGBM).
- **Datos:** `--covariables {serie,calendario,completo}`, `--clima-por-ciudad`, `--ventana`, `--horizonte`, `--paso-h`, `--hora-origen`, `--normalizacion {nivel,global}`, `--escala-mase-lag {24,168}`.
- **Dispositivo:** `--dispositivo {auto,cpu,cuda,cuda:N}` (auto usa GPU si existe), `--determinismo {estricto,rapido}`, `--amp` (bfloat16 en GPU), `--max-gb-gpu`, `--hilos`.
- **Semillas:** búsqueda 1; piloto 100-104; final 1000+ (n = máximo entre 5 y lo que estime el piloto con `--delta`, `--alfa`, `--potencia`). La búsqueda es reanudable (`busqueda.csv`).
- **Referencias** (ingenuo 24 h y 168 h con la misma disponibilidad de datos que los modelos) se reportan aparte en `referencias_{val,test}.json`.
- **GPU:** instalar PyTorch con CUDA (en la web oficial de PyTorch, elegir la versión de CUDA que corresponda al controlador) y comprobar `python -c "import torch; print(torch.cuda.is_available())"`.

## 10. Git y accesos
Repositorio: `github.com/Fabian-11037/prediccion-demanda-electrica` (rama `main`). Los commits de Claude llevan el trailer `Co-Authored-By`. El código y los textos se producen con asistencia de IA y deben ser revisados por el autor.

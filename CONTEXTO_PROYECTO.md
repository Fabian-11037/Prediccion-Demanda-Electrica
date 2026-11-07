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

## 3. Datos (carpeta `data/`)
Inventario y detalle en `data/raw/README.md` y `data/external/README.md`.

| Bloque | Cobertura | Estado |
|---|---|---|
| Demanda real, demanda comercial, precio de bolsa (XM) | horario 2000-01-01 → 2025-12-31 23:00, 227 928 filas, sin huecos ni NaN | incluido (`data/raw/numeric/hourly`) |
| Clima observado: temperatura, humedad, precipitación en 8 ciudades (ERA5 vía Open-Meteo) | igual | incluido; **reanálisis, no pronóstico** |
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
- `x_*`: entradas **as-of** (solo pasado, con rezago de publicación ya aplicado): demanda real rezagada, precio de bolsa, clima observado, hidrología.
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
1. Confirmar frecuencia (horaria), horizonte (24 h) y ámbito (SIN).
2. Elegir el bloque de prueba (¿prospectivo 2026?) y preregistrar.
3. Descargar e importar el ISE (DANE) y los pronósticos de temperatura (scripts en `scripts/adquisicion/`, **sin probar** porque el entorno de desarrollo no alcanza esas fuentes). Verificar que `previous_dayN` entrega corridas distintas y revisar huecos.
4. Descargar 2026 (XM, Open-Meteo, titulares) si se adopta el bloque prospectivo.
5. Definir el tratamiento de valores anómalos de demanda (p. ej. 2007-04-26 10:00-12:00 cae a 6-39 % de lo esperable; ver `data/processed/calidad_datos.json`). Los datos **no se corrigieron**.
6. Decidir si los titulares (metadato con derechos del editor) se mantienen en un repositorio público; ahora solo se incluye el titular (sin cuerpo) y la URL únicamente de registros con versión archivada verificada.
7. E1: EDA, extracción de 3 modelos base de la literatura, piloto, ≥5 semillas, baseline e inferencia, Jupyter Book. Completar la revisión de literatura (cadenas booleanas con fecha, bases de datos, exclusiones, tabla de síntesis con DOI).
8. E2/E3: Top-10, paper guía, auditoría de fuga del texto, búsqueda de 1000 arquitecturas, modelo original (ideas candidatas: compuerta de utilidad del texto guiada por incertidumbre con propiedad de «no empeorar»; pérdida de texto placebo; núcleo de respuesta al impulso de noticias; pronóstico conformal condicionado por texto — confirmar novedad con búsqueda dirigida).

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
                        pronosticos, climatologia
scripts/                construir_panel.py, generar_festivos.py, adquisicion/{descargar_pronosticos_temperatura,importar_ise}.py
tests/                  calendario, fuga del panel y las ventanas, particiones, externos, adquisición, higiene de redacción
```

## 9. Cómo ejecutar
```bash
pip install -r requirements.txt
python scripts/generar_festivos.py        # festivos
python scripts/construir_panel.py         # panel as-of + registro + cobertura
pytest -q tests                           # 22+ pruebas de fuga/calendario
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

## 10. Git y accesos
Repositorio: `github.com/Fabian-11037/prediccion-demanda-electrica` (rama `main`). Los commits de Claude llevan el trailer `Co-Authored-By`. El código y los textos se producen con asistencia de IA y deben ser revisados por el autor.

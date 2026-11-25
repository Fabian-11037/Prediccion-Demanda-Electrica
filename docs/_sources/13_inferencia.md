(inferencia-pesos-congelados)=
(inferencia-en-vivo)=
# 13. Inferencia en vivo

El panel calcula el pronóstico de las próximas 24 horas mediante los pesos congelados de los dieciséis procedimientos del estudio: las diez arquitecturas del Top 10, MLP, LSTM, CNN1D, ARX, el control numérico de TGForecaster y M4<sub>N</sub>. Los comparadores utilizan la semilla 1000 y M4<sub>N</sub>, la 101. La ejecución utiliza [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/) en la CPU del navegador; las historias cargadas se procesan en el dispositivo.

## Panel de pronóstico

Se seleccionan el modelo y uno de los cinco orígenes preparados y se pulsa **Generar pronóstico**. El control de demanda modifica las últimas 24 horas de la historia; un CSV propio sustituye la ventana de 168 horas. El panel muestra la trayectoria horaria, la energía diaria, el pico y el tiempo de cálculo, y permite descargar las predicciones. Los pesos se cargan al ejecutar el modelo seleccionado; los archivos de mayor tamaño requieren más tiempo en la primera carga.

```{raw} html
<iframe src="_static/inferencia_m4/index.html" title="Inferencia nativa del catálogo en el JBook" style="width:100%;height:1450px;border:1px solid #d9e3e6;border-radius:6px" loading="lazy"></iframe>
```

## Entradas y disponibilidad temporal

Cada origen conserva su calendario, historia, documentos disponibles y estado residual de M4<sub>N</sub>. El CSV contiene `fecha` y las treinta variables históricas de la plantilla, con 168 horas consecutivas anteriores al origen y alineadas por sus rezagos. Cada modelo consume únicamente las variables y máscaras de su contrato: los cuatro baselines de demanda/calendario no incorporan las covariables ampliadas.

La historia numérica puede modificarse manteniendo los documentos del origen seleccionado. TaTS, fusión tardía y TGForecaster con texto recalculan la relevancia documental a partir de esa historia y de los vectores congelados. TaTS conserva sus siete ventanas diarias; las otras dos arquitecturas utilizan la ventana documental del origen. El catálogo compacta los índices documentales sin alterar los vectores, los pesos de relevancia ni la selección temporal. El control numérico de TGForecaster conserva sus descripciones de canales.

M4<sub>N</sub> recalcula PatchTST, el experto E, el experto periódico, la combinación M1 y la memoria residual. Su banco utiliza episodios de dos a 365 días de edad, observables antes del origen. La carga de una historia alternativa conserva ese banco y el calendario del origen elegido. Los cinco estados preparados no sustituyen una adquisición continua de observaciones para fechas nuevas.

## Interpretación del pronóstico

La curva verde muestra la trayectoria calculada y la gris, las últimas 24 horas de la historia de entrada. La energía diaria suma los 24 intervalos y el pico identifica la mayor demanda horaria prevista. En M4<sub>N</sub>, la curva ocre corresponde a M1; su separación respecto del pronóstico revela la corrección residual. La compuerta indica cuánto del residuo recuperado se incorpora y los episodios muestran su distribución de atención.

Los cambios de la trayectoria ante una perturbación describen sensibilidad local, manteniendo las otras entradas y el estado del origen. Cada consulta utiliza una inicialización e incorpora intervalos calibrados para ese mismo checkpoint, cuando existe una reserva anterior al origen. El tiempo medido corresponde al dispositivo del lector y no sustituye las mediciones homogéneas del benchmark.

<!-- intervalos-vivo-inicio -->
(intervalos-vivo)=
## Intervalos del checkpoint en vivo

El panel permite elegir niveles nominales del 80 %, 90 % y 95 %, por hora o para la trayectoria completa de 24 horas. La banda se centra en la predicción calculada por la semilla 101 de M4 o la 1000 de cada comparador. Se calibra con los errores de esa misma realización; no se reutilizan los intervalos del promedio de semillas del benchmark.

Solo se admiten errores de días s≤t−2. La ventana conserva hasta 365 días recientes, con un mínimo de 60 observaciones, y utiliza el estadístico de orden ⌈(n+1)(1−α)⌉. El intervalo horario emplea errores absolutos por horizonte. La banda conjunta utiliza el máximo diario del error normalizado por una escala horaria congelada en la reserva inicial de 2022.

$$
C_{t,h}^{1-\alpha}=[\widehat y_{t,h}-q_{t,h}^{1-\alpha},\;\widehat y_{t,h}+q_{t,h}^{1-\alpha}].
$$ (eq-13-inferencia-001)

La calibración comienza en 2022 y las bandas se emiten desde 2023. Para 2022-01-01 se conserva el pronóstico puntual sin banda, porque todavía no existe esa reserva. Las historias intervenidas conservan la calibración histórica: no se atribuye cobertura condicional garantizada a una historia nueva. La dependencia temporal limita las garantías de intercambiabilidad.

En los 1.096 orígenes de 2023–2025, el checkpoint de M4 obtiene cobertura horaria 88,80 % y conjunta 88,59 % al nivel nominal del 90 %. Estas coberturas empíricas caracterizan esa realización y periodo.

La descarga CSV incorpora los límites, el nivel y el alcance. El servicio Python utiliza las mismas ventanas y cuantiles que el navegador. [Cobertura de los 16 checkpoints en 2023–2025](_static/resultados/intervalos_vivo/cobertura_checkpoint.csv) · [Protocolo](_static/resultados/intervalos_vivo/protocolo.json).

```{literalinclude} _static/inferencia_m4/calibration.js
:language: javascript
```

```{literalinclude} _static/codigo_e3/inferencia/intervalos_vivo.py
:language: python
```

```{literalinclude} _static/codigo_e3/mejoras_456/preparar_intervalos_vivo.py
:language: python
```
<!-- intervalos-vivo-fin -->

## Ejecución fiel de TimesNet

TimesNet utiliza lote uno y vuelve a seleccionar los periodos para cada historia y cada bloque. La ejecución conserva la transformada discreta de Fourier y la ponderación por amplitudes:

$$
X_{k,c}=\sum_{t=0}^{T-1}x_{t,c}e^{-2\pi i kt/T},\qquad s_k=\frac{1}{C}\sum_{c=1}^{C}|X_{k,c}|,\qquad p_k=\lfloor T/k\rfloor.
$$ (eq-13-inferencia-002)

Las frecuencias de mayor amplitud determinan los periodos; cada rama reorganiza la serie, aplica las convoluciones congeladas y recupera su longitud. La combinación utiliza pesos softmax y suma la conexión residual antes de la normalización del bloque. La selección se recalcula en JavaScript y las operaciones aprendidas se ejecutan en ONNX, conservando la arquitectura.

## Código de preparación y ejecución

La preparación reconstruye el nivel semanal, los indicadores de ausencia, el escalado histórico y de calendario y las entradas propias de cada procedimiento. Los grafos y sus partes se verifican mediante SHA-256. N-BEATS conserva sus tensores FP32; el navegador ensambla las partes del archivo antes de ejecutarlo.

### Contexto numérico y memoria de M4

```{literalinclude} _static/inferencia_m4/engine.js
:language: javascript
:linenos:
```

### Catálogo, escaladores y periodos dinámicos

```{literalinclude} _static/inferencia_m4/catalogue_engine.js
:language: javascript
:linenos:
```

### Ejecución en segundo plano

```{literalinclude} _static/inferencia_m4/worker.js
:language: javascript
:linenos:
```

### Interfaz, gráfico y descarga

```{literalinclude} _static/inferencia_m4/widget.js
:language: javascript
:linenos:
```

### Exportación de los pesos conservados

```{literalinclude} _static/codigo_e3/inferencia/exportar_catalogo_nativo.py
:language: python
:linenos:
```

La {ref}`reproducibilidad y procedencia <reproducibilidad-inferencia>` documenta las comprobaciones de exportación y el paquete portable de M4. El servicio Python para el catálogo completo se describe a continuación.

(inferencia-catalogo-python)=
## Servicio Python para el catálogo de modelos

El dashboard integra Contexto, EDA, resultados históricos e inferencia interactiva. La inferencia calcula una trayectoria de 24 horas a partir de las 168 horas anteriores, calendario y entradas documentales cuando corresponden al modelo. El escalado y los pesos permanecen fijos. Los comparadores utilizan la semilla 1000 y M4ₙ, la 101; el benchmark resume por separado sus realizaciones.

El servicio acepta una ventana CSV con las 30 variables históricas, alineadas por disponibilidad. Sin archivo, la historia local permite emitir hasta 2026-01-01. El origen se valida junto con la continuidad horaria, la forma de las entradas y las máscaras. M4ₙ recalcula los expertos y su combinación y restringe la memoria a episodios entre dos y 365 días de edad. Una prueba que altera observaciones posteriores al corte conserva la predicción.

La salida muestra trayectoria, energía diaria, pico, tiempo de respuesta y procedencia del checkpoint. La sensibilidad a un aumento del 1 % de la demanda reciente caracteriza una respuesta local del modelo, manteniendo las demás entradas y la memoria. El aviso de extrapolación utiliza los rangos del desarrollo 2000–2021. La vista de M4ₙ muestra también su compuerta, la base M1 y los pesos de los episodios recuperados.

La prueba de restauración compara un origen conservado por procedimiento, con tolerancia relativa máxima 0,001. Los dieciséis procedimientos satisfacen esa tolerancia; TimesNet coincide numéricamente en el origen comprobado. Se verifica además la generación de 24 valores finitos para 2026-01-01, la respuesta a cambios de entrada y el rechazo de CSV incompatibles. Esta comprobación puntual no sustituye una prueba de portabilidad completa.

```console
python dashboard/app.py
```

El panel se abre en `http://127.0.0.1:8815/inferencia/`. Se seleccionan modelo y origen, se carga opcionalmente el CSV de historia y se pulsa **Generar pronóstico**. La plantilla descargable especifica las columnas de entrada. La trayectoria de 24 horas y su procedencia pueden exportarse en CSV. Las dependencias se fijan en `dashboard/requirements-live.lock.txt`; el servicio requiere los artefactos de los modelos seleccionados y `DEMANDA_ARTEFACTOS` permite ubicar el consolidado.

### Consulta desde Python

Con el servicio iniciado, el siguiente código solicita una trayectoria de M4ₙ. El origen del ejemplo utiliza la última historia incluida. Para una fecha posterior se proporciona un CSV con las 168 horas anteriores y las variables alineadas por sus rezagos; los documentos, cuando intervienen, también deben estar disponibles antes del origen.

```python
from pathlib import Path
import requests
import pandas as pd
import matplotlib.pyplot as plt

entrada = {
    "model": "m4_n",
    "origin": "2026-01-01",
    "demand_pct": 0,
    "explain": True,
}
# Para una historia propia, indicar el archivo preparado:
# entrada["csv"] = Path("historia_168h.csv").read_text(encoding="utf-8-sig")
respuesta = requests.post(
    "http://127.0.0.1:8815/api/inferencia", json=entrada, timeout=180
)
respuesta.raise_for_status()
resultado = respuesta.json()
pronostico = pd.DataFrame({
    "fecha": pd.to_datetime(resultado["dates"]),
    "demanda_GWh": resultado["prediction_gwh"],
})
assert len(pronostico) == 24 and not resultado["training"]
ax = pronostico.plot(x="fecha", y="demanda_GWh", legend=False)
ax.set(xlabel="Fecha y hora de Colombia", ylabel="Demanda (GWh por intervalo)")
plt.show()
pronostico.to_csv("pronostico_m4.csv", index=False)
```

La trayectoria representa la respuesta del checkpoint a la historia suministrada. Su perfil horario permite examinar el nivel y la localización del pico; la sensibilidad activada cuantifica el cambio ante una perturbación de la demanda de entrada. Se utiliza una sola inicialización, con intervalos calibrados a partir de sus propios errores históricos. La fecha seleccionada no descarga nuevas observaciones.

### Ejecución portable de M4ₙ

La {ref}`inferencia portable <inferencia-portable>` permite ejecutar los cinco módulos ONNX en CPU con NumPy y ONNX Runtime. El paquete incluye ejemplos, escaladores y memoria preparada; la entrada propia debe cumplir ese contrato. No necesita iniciar el servicio Flask ni instalar PyTorch.

### Código del servicio y preparación de entradas

El código siguiente implementa la interfaz, la API y la preparación e inferencia de los pesos.

```{literalinclude} _static/codigo_e3/inferencia/app.py
:language: python
:linenos:
```

```{literalinclude} _static/codigo_e3/inferencia/runtime_inferencia.py
:language: python
:linenos:
```

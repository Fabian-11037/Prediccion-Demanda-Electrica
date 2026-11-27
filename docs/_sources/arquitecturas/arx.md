(e2-arx)=
# 10.2. ARX lineal regularizado

La {ref}`notación y las operaciones compartidas <fundamentos-arquitecturas>` definen los tensores, el nivel de demanda, la atención, las activaciones y la optimización.

## Arquitectura y fundamentos

ARX relaciona la demanda futura con sus propios rezagos y con variables externas conocidas: aquí, el calendario. Cada una de las 24 horas tiene una combinación lineal distinta de los mismos atributos. Una subida de un rezago produce un cambio proporcional determinado por su coeficiente; no hay capas ocultas ni interacciones aprendidas entre atributos. Es una referencia de complejidad baja para valorar si las redes justifican su coste.

La entrada contiene los últimos 48 valores de demanda divididos por el nivel de la ventana y las 23 variables de calendario de cada una de las 24 horas futuras: $48+24\times23=600$ atributos. Aunque la ventana que calcula el nivel tiene 168 horas, la regresión utiliza directamente 48 rezagos. Cada atributo se estandariza con la media y desviación del tramo de ajuste.

$$
z_j=\frac{u_j-\mu_j}{s_j},\qquad \widehat{\mathbf r}=Wz+b,\qquad \widehat{\mathbf y}=\ell_t\widehat{\mathbf r}.
$$ (eq-arquitecturas-arx-001)

Aquí $W\in\mathbb R^{24\times600}$ y $b\in\mathbb R^{24}$; hay 14.424 parámetros entrenables. La configuración seleccionada utiliza Lasso, con $\lambda=1,3734\times10^{-7}$. Se minimiza el MAE normalizado más $\lambda\sum_{h,j}|W_{hj}|$, lo que favorece coeficientes pequeños o nulos. La penalización actúa sobre los pesos, no sobre los sesgos. Los valores de escalado se conservan junto al checkpoint; no se vuelven a estimar en prueba.

### Esquema global

[![Arquitectura global de arx](../_static/arquitecturas_presentacion/arx_global.png)](../_static/arquitecturas_presentacion/arx_global.png)

La historia se divide por su nivel y se seleccionan 48 rezagos. La concatenación con el calendario produce 600 atributos; una matriz genera las 24 salidas.

### Detalle de las operaciones

[![Detalle de la arquitectura de arx](../_static/arquitecturas_presentacion/arx_detalle.png)](../_static/arquitecturas_presentacion/arx_detalle.png)

Cada hora utiliza una fila distinta de la matriz de pesos. El sesgo completa la combinación lineal; la pérdida incorpora penalización lasso y el pronóstico se restituye a kWh con el nivel de la ventana.

El {ref}`código de exportación de los esquemas <exportacion-esquemas>` forma parte del preprocesamiento compartido.

### Código de la arquitectura

```{literalinclude} ../_static/codigo_e2/arx_red.py
:language: python
:linenos:
```

<!-- diseno-entrenamiento:inicio -->

## Diseño de entrenamiento y prueba

(diseno-entrenamiento-arx)=

### Selección en validación

Se evaluaron **20 configuraciones** y 3 llegaron a la comparación de selección. La **configuración arx_lineal_02** obtuvo el menor MASE de validación (0,6838) sobre 2018–2021. La configuración fijada para prueba es **arx_lineal_02**, con 5 semillas.

#### Configuración conservada
| Parámetro | Valor |
|---|---|
| Penalización | lasso |
| Rezagos directos | 48 |
| Lote de ajuste | 64 |
| Tasa de aprendizaje | 0,000658261 |
| Plan de tasa | plateau |
| Pérdida | mae |
| Intensidad de penalización | 1,3734e-07 |

#### Etapas del entrenamiento y prueba

| Etapa | Especificación |
|---|---|
| Filtro inicial | 20 configuraciones; semilla 42; validación 2019; máximo 30 épocas |
| Confirmación y criterio | 3 finalistas; semillas 43 y 44; años 2018–2021; máximo 60 épocas. Menor MASE medio de validación; cada año tiene el mismo peso. |
| Ajuste final | 2000–2021, excluida la reserva interna de parada |
| Parada interna final | Último 10 % de orígenes anteriores a 2022; purga de 24 horas |
| Prueba | 2022-01-01 a 2025-12-31; 1.461 orígenes diarios |
| Semillas finales | 1000, 1001, 1002, 1003, 1004 |
| Historia y horizonte | 168 y 24 horas, respectivamente |
| Pérdida y optimizador | MAE normalizado más penalización L1; AdamW |
| Épocas y parada | Máximo 60; mínimo 10; paciencia efectiva 6; restauración del menor valor interno |
| Tasa de aprendizaje | ReduceLROnPlateau: factor 0,5; paciencia 2 |
| Lotes | Ajuste: 64; evaluación de prueba: hasta 4096 orígenes por lote |
| Épocas ejecutadas | 27–41; épocas de los checkpoints: 21, 35, 29, 25, 24 |


#### Configuraciones evaluadas en desarrollo
El filtro y la selección corresponden a periodos y cantidades de repeticiones distintos. El guion identifica las configuraciones que no avanzaron a confirmación; no representa un resultado nulo.
| Configuración | Hiperparámetros | MASE del filtro | MASE de validación | Estado |
|---|---|---:|---:|---|
| **arx_lineal_02** | Penalización: lasso; Rezagos directos: 48; Lote de ajuste: 64; Tasa de aprendizaje: 0,000658261; Plan de tasa: plateau; Pérdida: mae; Intensidad de penalización: 1,3734e-07 | 0,6418 | 0,6838 | **Seleccionada para prueba** |
| arx_lineal_03 | Penalización: lasso; Rezagos directos: 168; Lote de ajuste: 256; Tasa de aprendizaje: 0,000456524; Plan de tasa: cosine; Pérdida: mae; Intensidad de penalización: 3,28076e-06 | 0,6349 | 0,6931 | Finalista |
| arx_lineal_01 | Penalización: ridge; Rezagos directos: 96; Lote de ajuste: 256; Tasa de aprendizaje: 0,000194231; Plan de tasa: plateau; Pérdida: huber; Intensidad de penalización: 3,62374e-05 | 0,6392 | 0,7263 | Finalista |
| arx_lineal_11 | Penalización: ridge; Rezagos directos: 96; Lote de ajuste: 256; Tasa de aprendizaje: 0,000241675; Plan de tasa: plateau; Pérdida: mae; Intensidad de penalización: 9,0494e-06 | 0,6582 | — | Filtro inicial |
| arx_lineal_18 | Penalización: ridge; Rezagos directos: 24; Lote de ajuste: 256; Tasa de aprendizaje: 0,000213312; Plan de tasa: plateau; Pérdida: mse; Intensidad de penalización: 3,83434e-05 | 0,6631 | — | Filtro inicial |
| arx_lineal_09 | Penalización: ridge; Rezagos directos: 96; Lote de ajuste: 256; Tasa de aprendizaje: 0,0015342; Plan de tasa: plateau; Pérdida: huber; Intensidad de penalización: 0,0158273 | 0,6839 | — | Filtro inicial |
| arx_lineal_04 | Penalización: ridge; Rezagos directos: 48; Lote de ajuste: 256; Tasa de aprendizaje: 0,000640068; Plan de tasa: none; Pérdida: huber; Intensidad de penalización: 2,04912e-05 | 0,7074 | — | Filtro inicial |
| arx_lineal_19 | Penalización: ridge; Rezagos directos: 96; Lote de ajuste: 128; Tasa de aprendizaje: 0,000620172; Plan de tasa: none; Pérdida: mae; Intensidad de penalización: 2,0534e-05 | 0,7686 | — | Filtro inicial |
| arx_lineal_06 | Penalización: ridge; Rezagos directos: 96; Lote de ajuste: 64; Tasa de aprendizaje: 0,000357917; Plan de tasa: cosine; Pérdida: huber; Intensidad de penalización: 0,0707338 | 0,8002 | — | Filtro inicial |
| arx_lineal_12 | Penalización: ridge; Rezagos directos: 24; Lote de ajuste: 32; Tasa de aprendizaje: 0,00527919; Plan de tasa: plateau; Pérdida: huber; Intensidad de penalización: 0,000228849 | 0,8714 | — | Filtro inicial |
| arx_lineal_20 | Penalización: lasso; Rezagos directos: 48; Lote de ajuste: 128; Tasa de aprendizaje: 0,00184637; Plan de tasa: none; Pérdida: huber; Intensidad de penalización: 0,000423126 | 0,9950 | — | Filtro inicial |
| arx_lineal_05 | Penalización: ridge; Rezagos directos: 168; Lote de ajuste: 32; Tasa de aprendizaje: 0,000934075; Plan de tasa: none; Pérdida: mse; Intensidad de penalización: 0,0643111 | 1,0964 | — | Filtro inicial |
| arx_lineal_10 | Penalización: lasso; Rezagos directos: 96; Lote de ajuste: 128; Tasa de aprendizaje: 0,00320672; Plan de tasa: cosine; Pérdida: mse; Intensidad de penalización: 0,00125353 | 1,1091 | — | Filtro inicial |
| arx_lineal_13 | Penalización: lasso; Rezagos directos: 48; Lote de ajuste: 256; Tasa de aprendizaje: 0,00195633; Plan de tasa: none; Pérdida: mse; Intensidad de penalización: 0,00150607 | 1,1771 | — | Filtro inicial |
| arx_lineal_08 | Penalización: lasso; Rezagos directos: 168; Lote de ajuste: 64; Tasa de aprendizaje: 0,00114072; Plan de tasa: none; Pérdida: huber; Intensidad de penalización: 2,04997e-07 | 1,2424 | — | Filtro inicial |
| arx_lineal_16 | Penalización: lasso; Rezagos directos: 24; Lote de ajuste: 64; Tasa de aprendizaje: 0,00688933; Plan de tasa: plateau; Pérdida: huber; Intensidad de penalización: 0,00267671 | 1,6745 | — | Filtro inicial |
| arx_lineal_07 | Penalización: ridge; Rezagos directos: 96; Lote de ajuste: 64; Tasa de aprendizaje: 0,00176595; Plan de tasa: none; Pérdida: mae; Intensidad de penalización: 0,00116149 | 1,9139 | — | Filtro inicial |
| arx_lineal_14 | Penalización: lasso; Rezagos directos: 48; Lote de ajuste: 64; Tasa de aprendizaje: 0,0018996; Plan de tasa: none; Pérdida: huber; Intensidad de penalización: 0,00658195 | 2,0878 | — | Filtro inicial |
| arx_lineal_17 | Penalización: ridge; Rezagos directos: 96; Lote de ajuste: 64; Tasa de aprendizaje: 0,00344377; Plan de tasa: none; Pérdida: huber; Intensidad de penalización: 0,0143342 | 2,6418 | — | Filtro inicial |
| arx_lineal_15 | Penalización: lasso; Rezagos directos: 24; Lote de ajuste: 32; Tasa de aprendizaje: 0,00615789; Plan de tasa: cosine; Pérdida: huber; Intensidad de penalización: 8,86156e-05 | 4,8562 | — | Filtro inicial |

[Resultados de desarrollo por año y semilla](../_static/resultados/diseno_entrenamiento/arx_lineal.csv) · [Configuraciones completas](../_static/resultados/diseno_entrenamiento/arx_lineal_configuraciones.csv).

<!-- diseno-entrenamiento:fin -->

## Pipeline de entrenamiento y prueba

<!-- entrenamiento-portable:inicio -->
### Comandos de entrenamiento y evaluación

Validación temporal de la configuración seleccionada:

```bash
python scripts/entrenar_arquitectura.py --modelo arx_lineal --anio 2021 --semilla 1000 --dispositivo auto --salida salidas/arx_lineal_validacion_1000
```

Comprobación funcional breve:

```bash
python scripts/entrenar_arquitectura.py --modelo arx_lineal --smoke --dispositivo cpu --salida salidas/arx_lineal_funcional
```

Evaluación final, con ajuste y parada anteriores a 2022:

```bash
python scripts/entrenar_arquitectura.py --modelo arx_lineal --fase prueba --allow-test --semilla 1000 --dispositivo auto --salida salidas/arx_lineal_prueba_1000
```

El código completo de preparación, optimización, guardado y restauración está en el {ref}`pipeline ejecutable compartido <entrenamiento-portable>`. Cada salida debe ser una carpeta nueva.
<!-- entrenamiento-portable:fin -->

La configuración utilizada por esta familia es:

```json
{
  "id": "arx_lineal_02",
  "ventana": 168,
  "params": {
    "penalizacion": "lasso",
    "lags": 48,
    "batch_size": 64,
    "lr": 0.0006582612236208865,
    "scheduler": "plateau",
    "perdida": "mae",
    "paciencia": 15,
    "lambda": 1.3733953296838922e-07
  }
}
```

El procedimiento separa ajuste, parada interna y evaluación externa, estima las transformaciones únicamente en ajuste y restaura el mínimo de parada. La función admite validación o prueba con autorización explícita y conserva las salidas en una carpeta nueva. Los bloques siguientes documentan el código; construir el libro no ejecuta entrenamientos.

### Escalado por pliegue y optimización

```{literalinclude} ../_static/codigo_e2/base/src/demanda/reentrenamiento_v3.py
:language: python
:linenos:
```


El siguiente bloque vincula las implementaciones mostradas y permite seleccionar CPU o GPU sin funciones exclusivas del sistema operativo. La inicialización del dispositivo se conserva en resolver_dispositivo.

```python
from pathlib import Path
import json,sys
import numpy as np
import pandas as pd

def ejecutar_configuracion(raiz, familia, semilla=1000, dispositivo='auto',
                          fase='validacion', anio=2021, permitir_prueba=False,
                          salida=None):
    raiz=Path(raiz).resolve()
    procedencia=json.loads((raiz/'results/jbook/modelos_actualizados.json').read_text())
    fuentes=raiz/'results/repositorio'/procedencia['commit']
    sys.path.insert(0,str(fuentes/'src'))
    from demanda import modelos_base as M, particion as P
    from demanda import reentrenamiento_v3 as motor
    from demanda.variantes_v4 import Secuencial as Recurrente
    from demanda.metricas import evaluar
    import torch
    registro=json.loads((fuentes/'results/consolidado/modelos.json').read_text())
    config=registro[familia]['configuracion']
    panel=pd.read_parquet(raiz/'data/processed/panel_horario.parquet')
    variables=json.loads((fuentes/'data/processed/feature_registry.json').read_text())
    cfg=M.ConfigDatos(ventana=config['ventana'])
    if fase=='prueba':
        if not permitir_prueba:raise PermissionError('La prueba requiere acceso explícito')
        tr,es=P.pretest(panel.index,ventana=cfg.ventana)
        ev=P.origenes(panel.index,'test',ventana=cfg.ventana,allow_test=True)
    else:
        fold=P.folds_forward_chaining(panel.index,[anio],ventana=cfg.ventana)[0]
        tr,es,ev=fold['tr'],fold['es'],fold['val']
    datos=motor.DatosFold(panel,variables,cfg,tr,secuencial=familia in ('lstm','cnn1d'))
    train,stop,evaluacion=[datos.conjunto_pos(p) for p in [tr,es,ev]]
    assert train.origen.max()+pd.Timedelta(hours=23)<stop.origen.min()
    assert stop.origen.max()+pd.Timedelta(hours=23)<evaluacion.origen.min()
    assert pd.Timestamp(datos.auditoria['fin_ajuste'])<stop.origen.min()
    fabrica=motor.Secuencial
    try:
        if familia=='lstm':motor.Secuencial=Recurrente
        modelo=motor.Entrenador(familia)
        info=modelo.ajustar(train,stop,datos,config['params'],semilla,
                           M.Opciones(max_epocas=60,dispositivo=dispositivo,hilos=2))
    finally:
        motor.Secuencial=fabrica
    pred=datos.invertir(modelo.predecir(evaluacion),evaluacion)
    metricas=evaluar(evaluacion.y,pred,datos.escala_mase)
    if salida is not None:
        destino=Path(salida)/familia/str(semilla)/str(anio if fase=='validacion' else fase)
        destino.mkdir(parents=True,exist_ok=False)
        escala=datos.escaladores()
        torch.save({'state_dict':{k:v.detach().cpu() for k,v in modelo.red.state_dict().items()},
                    'modelo':familia,'params':config['params'],'escaladores':escala},
                    destino/'modelo.pt')
        np.savez_compressed(destino/'predicciones.npz',y=evaluacion.y,yhat=pred,
                            origen=evaluacion.origen.as_unit('ns').asi8)
        for nombre,objeto in [('escaladores',escala),('historia',info['historia']),
                              ('metricas',metricas)]:
            (destino/(nombre+'.json')).write_text(json.dumps(objeto,ensure_ascii=False,
                                                indent=2,default=str),encoding='utf-8')
    return modelo,datos,evaluacion,pred,info,metricas

# Selección: comparar el MASE medio de confirmación, sin leer prueba.
# Piloto: repetir cada configuración seleccionada con semillas 100–104
# en los cuatro años externos. Final: semillas 1000–1004 con acceso explícito.
# El argumento salida conserva pesos, escaladores, curvas, métricas y predicciones.

```



Para esta página, `familia` toma el valor `arx_lineal`. Las definiciones completas de datos y métricas se muestran en el {ref}`preprocesamiento compartido <fundamentos-arquitecturas>`; los cuerpos de las redes y del bucle de ajuste están visibles en esta página.

## Evaluación

Los resultados de esta familia se integran en la {ref}`evaluación estadística conjunta <evaluacion-e2>`, con métricas por semilla, horizonte y origen. La comparación distingue las entradas demanda/calendario de las entradas numéricas ampliadas y el aporte de texto oficial.

## Fuentes

La formulación y las dimensiones se verifican en las clases del código mostrado.. La descripción corresponde a la implementación del experimento; las adaptaciones se identifican en el texto.

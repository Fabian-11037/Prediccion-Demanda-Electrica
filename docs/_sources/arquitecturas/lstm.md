(e2-lstm)=
# 10.4. LSTM: memoria recurrente con compuertas

La {ref}`notación y las operaciones compartidas <fundamentos-arquitecturas>` definen los tensores, el nivel de demanda, la atención, las activaciones y la optimización.

## Arquitectura y fundamentos

Una LSTM recorre la historia hora por hora y conserva un estado de memoria. Sus compuertas deciden qué información retener, incorporar y exponer. Esta organización permite representar dependencias de varias horas sin asignar un coeficiente independiente a cada posición, aunque no garantiza utilizar toda la ventana.

La implementación seleccionada recibe 168 pasos con 24 variables por paso: demanda normalizada y 23 variables de calendario pasado. Tiene una capa recurrente de 48 unidades. El estado oculto final se concatena con los 552 atributos de calendario futuro y una capa lineal de 24 salidas aprende desviaciones respecto de uno. Hay 28.632 parámetros.

$$
\begin{aligned}
f_s&=\sigma(W_f x_s+U_fh_{s-1}+b_f),&i_s&=\sigma(W_i x_s+U_ih_{s-1}+b_i),\\
\tilde c_s&=\tanh(W_c x_s+U_ch_{s-1}+b_c),&c_s&=f_s\odot c_{s-1}+i_s\odot\tilde c_s,\\
o_s&=\sigma(W_o x_s+U_oh_{s-1}+b_o),&h_s&=o_s\odot\tanh(c_s).
\end{aligned}
$$ (eq-arquitecturas-lstm-001)

La memoria $c_s$ y la salida $h_s$ son distintas; $\odot$ representa producto componente a componente. El decodificador usado es directo: $\widehat{\mathbf y}=\ell_t[\mathbf1+W_o'[D(h_{168});\operatorname{vec}(F)]+b_o']$. No realimenta una predicción para obtener la siguiente. El dropout de 0,1 se aplica al estado antes de la salida; no hay dropout entre capas recurrentes porque solo existe una. El entrenamiento utiliza Huber, AdamW y parada interna.

### Esquema global

[![Arquitectura global de lstm](../_static/arquitecturas_presentacion/lstm_global.png)](../_static/arquitecturas_presentacion/lstm_global.png)

Una misma celda recurrente procesa las 168 horas. El estado oculto final de 48 componentes se concatena con el calendario del horizonte para generar las 24 salidas.

### Detalle de las operaciones

[![Detalle de la arquitectura de lstm](../_static/arquitecturas_presentacion/lstm_detalle.png)](../_static/arquitecturas_presentacion/lstm_detalle.png)

Las compuertas de olvido, entrada y salida regulan la memoria y el estado oculto. La demanda se expresa respecto del nivel y de la dispersión de ajuste; la lectura utiliza el estado final y el calendario conocido.

El {ref}`código de exportación de los esquemas <exportacion-esquemas>` forma parte del preprocesamiento compartido.

### Código de la arquitectura

```{literalinclude} ../_static/codigo_e2/lstm_red.py
:language: python
:linenos:
```

<!-- diseno-entrenamiento:inicio -->

## Diseño de entrenamiento y prueba

(diseno-entrenamiento-lstm)=

### Selección en validación

Se evaluaron **24 configuraciones** y 6 llegaron a la comparación de selección. La **configuración lstm_v4_01** obtuvo el menor MASE de validación (0,7273) sobre 2018–2021. La configuración fijada para prueba es **lstm_v4_01**, con 5 semillas.

#### Configuración conservada
| Parámetro | Valor |
|---|---|
| Dropout | 0,1 |
| Decaimiento de pesos | 0,0001 |
| Lote de ajuste | 256 |
| Plan de tasa | plateau |
| Pérdida | huber |
| Capas | 1 |
| Unidades | 48 |
| Tasa de aprendizaje | 0,001 |

#### Etapas del entrenamiento y prueba

| Etapa | Especificación |
|---|---|
| Filtro inicial | 24 configuraciones; semilla 42; validación 2019; máximo 30 épocas |
| Confirmación y criterio | 6 finalistas; semillas 43 y 44; años 2018–2021; máximo 60 épocas. Menor MASE medio de validación; cada año tiene el mismo peso. |
| Ajuste final | 2000–2021, excluida la reserva interna de parada |
| Parada interna final | Último 10 % de orígenes anteriores a 2022; purga de 24 horas |
| Prueba | 2022-01-01 a 2025-12-31; 1.461 orígenes diarios |
| Semillas finales | 1000, 1001, 1002, 1003, 1004 |
| Historia y horizonte | 168 y 24 horas, respectivamente |
| Pérdida y optimizador | Huber normalizada; transición 0,05; AdamW |
| Épocas y parada | Máximo 60; mínimo 10; paciencia efectiva 6; restauración del menor valor interno |
| Tasa de aprendizaje | ReduceLROnPlateau: factor 0,5; paciencia 2 |
| Lotes | Ajuste: 256; evaluación de prueba: hasta 4096 orígenes por lote |
| Épocas ejecutadas | 36–48; épocas de los checkpoints: 30, 42, 36, 40, 37 |


#### Configuraciones evaluadas en desarrollo
El filtro y la selección corresponden a periodos y cantidades de repeticiones distintos. El guion identifica las configuraciones que no avanzaron a confirmación; no representa un resultado nulo.
| Configuración | Hiperparámetros | MASE del filtro | MASE de validación | Estado |
|---|---|---:|---:|---|
| **lstm_v4_01** | Dropout: 0,1; Decaimiento de pesos: 0,0001; Lote de ajuste: 256; Plan de tasa: plateau; Pérdida: huber; Capas: 1; Unidades: 48; Tasa de aprendizaje: 0,001 | 0,6938 | 0,7273 | **Seleccionada para prueba** |
| lstm_v4_02 | Dropout: 0,1; Decaimiento de pesos: 0,0001; Lote de ajuste: 256; Plan de tasa: plateau; Pérdida: huber; Capas: 1; Unidades: 64; Tasa de aprendizaje: 0,001 | 0,6573 | 0,7416 | Finalista |
| lstm_16 | Capas: 1; Dropout: 0,1; Decaimiento de pesos: 0,0001; Lote de ajuste: 256; Tasa de aprendizaje: 0,001; Plan de tasa: plateau; Pérdida: huber; Unidades: 32 | 0,7666 | 0,7945 | Finalista |
| lstm_18 | Capas: 1; Dropout: 0,2; Decaimiento de pesos: 0,001; Lote de ajuste: 256; Tasa de aprendizaje: 0,001; Plan de tasa: plateau; Pérdida: huber; Unidades: 32 | 0,7903 | 0,8278 | Finalista |
| lstm_20 | Capas: 1; Dropout: 0,3; Decaimiento de pesos: 0,001; Lote de ajuste: 256; Tasa de aprendizaje: 0,001; Plan de tasa: plateau; Pérdida: huber; Unidades: 32 | 0,8136 | 0,8645 | Finalista |
| lstm_v4_04 | Dropout: 0,1; Decaimiento de pesos: 0,0001; Lote de ajuste: 256; Plan de tasa: plateau; Pérdida: huber; Capas: 2; Unidades: 32; Tasa de aprendizaje: 0,001 | 1,6600 | 1,6072 | Finalista |
| lstm_06 | Capas: 1; Dropout: 0,1; Decaimiento de pesos: 0,0001; Lote de ajuste: 256; Tasa de aprendizaje: 0,001; Plan de tasa: plateau; Pérdida: huber; Unidades: 16 | 0,8268 | — | Filtro inicial |
| lstm_11 | Capas: 1; Dropout: 0,1; Decaimiento de pesos: 0,0001; Lote de ajuste: 256; Tasa de aprendizaje: 0,001; Plan de tasa: plateau; Pérdida: huber; Unidades: 32 | 0,8362 | — | Filtro inicial |
| lstm_13 | Capas: 1; Dropout: 0,2; Decaimiento de pesos: 0,001; Lote de ajuste: 256; Tasa de aprendizaje: 0,001; Plan de tasa: plateau; Pérdida: huber; Unidades: 32 | 0,8733 | — | Filtro inicial |
| lstm_15 | Capas: 1; Dropout: 0,3; Decaimiento de pesos: 0,001; Lote de ajuste: 256; Tasa de aprendizaje: 0,001; Plan de tasa: plateau; Pérdida: huber; Unidades: 32 | 0,8957 | — | Filtro inicial |
| lstm_01 | Capas: 1; Dropout: 0,1; Decaimiento de pesos: 0,0001; Lote de ajuste: 256; Tasa de aprendizaje: 0,001; Plan de tasa: plateau; Pérdida: huber; Unidades: 16 | 0,9629 | — | Filtro inicial |
| lstm_12 | Capas: 1; Dropout: 0,1; Decaimiento de pesos: 0,0001; Lote de ajuste: 256; Tasa de aprendizaje: 0,0003; Plan de tasa: plateau; Pérdida: huber; Unidades: 32 | 0,9909 | — | Filtro inicial |
| lstm_03 | Capas: 1; Dropout: 0,2; Decaimiento de pesos: 0,001; Lote de ajuste: 256; Tasa de aprendizaje: 0,001; Plan de tasa: plateau; Pérdida: huber; Unidades: 16 | 1,0132 | — | Filtro inicial |
| lstm_14 | Capas: 1; Dropout: 0,2; Decaimiento de pesos: 0,001; Lote de ajuste: 256; Tasa de aprendizaje: 0,0003; Plan de tasa: plateau; Pérdida: huber; Unidades: 32 | 1,0444 | — | Filtro inicial |
| lstm_05 | Capas: 1; Dropout: 0,3; Decaimiento de pesos: 0,001; Lote de ajuste: 256; Tasa de aprendizaje: 0,001; Plan de tasa: plateau; Pérdida: huber; Unidades: 16 | 1,0728 | — | Filtro inicial |
| lstm_08 | Capas: 1; Dropout: 0,2; Decaimiento de pesos: 0,001; Lote de ajuste: 256; Tasa de aprendizaje: 0,001; Plan de tasa: plateau; Pérdida: huber; Unidades: 16 | 1,6273 | — | Filtro inicial |
| lstm_10 | Capas: 1; Dropout: 0,3; Decaimiento de pesos: 0,001; Lote de ajuste: 256; Tasa de aprendizaje: 0,001; Plan de tasa: plateau; Pérdida: huber; Unidades: 16 | 1,6340 | — | Filtro inicial |
| lstm_17 | Capas: 1; Dropout: 0,1; Decaimiento de pesos: 0,0001; Lote de ajuste: 256; Tasa de aprendizaje: 0,0003; Plan de tasa: plateau; Pérdida: huber; Unidades: 32 | 1,6369 | — | Filtro inicial |
| lstm_19 | Capas: 1; Dropout: 0,2; Decaimiento de pesos: 0,001; Lote de ajuste: 256; Tasa de aprendizaje: 0,0003; Plan de tasa: plateau; Pérdida: huber; Unidades: 32 | 1,6408 | — | Filtro inicial |
| lstm_02 | Capas: 1; Dropout: 0,1; Decaimiento de pesos: 0,0001; Lote de ajuste: 256; Tasa de aprendizaje: 0,0003; Plan de tasa: plateau; Pérdida: huber; Unidades: 16 | 1,6454 | — | Filtro inicial |
| lstm_04 | Capas: 1; Dropout: 0,2; Decaimiento de pesos: 0,001; Lote de ajuste: 256; Tasa de aprendizaje: 0,0003; Plan de tasa: plateau; Pérdida: huber; Unidades: 16 | 1,6552 | — | Filtro inicial |
| lstm_07 | Capas: 1; Dropout: 0,1; Decaimiento de pesos: 0,0001; Lote de ajuste: 256; Tasa de aprendizaje: 0,0003; Plan de tasa: plateau; Pérdida: huber; Unidades: 16 | 1,6573 | — | Filtro inicial |
| lstm_v4_03 | Dropout: 0,1; Decaimiento de pesos: 0,0001; Lote de ajuste: 256; Plan de tasa: plateau; Pérdida: huber; Capas: 2; Unidades: 16; Tasa de aprendizaje: 0,001 | 1,6649 | — | Filtro inicial |
| lstm_09 | Capas: 1; Dropout: 0,2; Decaimiento de pesos: 0,001; Lote de ajuste: 256; Tasa de aprendizaje: 0,0003; Plan de tasa: plateau; Pérdida: huber; Unidades: 16 | 1,6650 | — | Filtro inicial |

[Resultados de desarrollo por año y semilla](../_static/resultados/diseno_entrenamiento/lstm.csv) · [Configuraciones completas](../_static/resultados/diseno_entrenamiento/lstm_configuraciones.csv).

<!-- diseno-entrenamiento:fin -->

## Pipeline de entrenamiento y prueba

<!-- entrenamiento-portable:inicio -->
### Comandos de entrenamiento y evaluación

Validación temporal de la configuración seleccionada:

```bash
python scripts/entrenar_arquitectura.py --modelo lstm --anio 2021 --semilla 1000 --dispositivo auto --salida salidas/lstm_validacion_1000
```

Comprobación funcional breve:

```bash
python scripts/entrenar_arquitectura.py --modelo lstm --smoke --dispositivo cpu --salida salidas/lstm_funcional
```

Evaluación final, con ajuste y parada anteriores a 2022:

```bash
python scripts/entrenar_arquitectura.py --modelo lstm --fase prueba --allow-test --semilla 1000 --dispositivo auto --salida salidas/lstm_prueba_1000
```

El código completo de preparación, optimización, guardado y restauración está en el {ref}`pipeline ejecutable compartido <entrenamiento-portable>`. Cada salida debe ser una carpeta nueva.
<!-- entrenamiento-portable:fin -->

La configuración utilizada por esta familia es:

```json
{
  "id": "lstm_v4_01",
  "ventana": 168,
  "params": {
    "dropout": 0.1,
    "weight_decay": 0.0001,
    "batch_size": 256,
    "scheduler": "plateau",
    "perdida": "huber",
    "paciencia": 6,
    "capas": 1,
    "unidades": 48,
    "lr": 0.001
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



Para esta página, `familia` toma el valor `lstm`. Las definiciones completas de datos y métricas se muestran en el {ref}`preprocesamiento compartido <fundamentos-arquitecturas>`; los cuerpos de las redes y del bucle de ajuste están visibles en esta página.

## Evaluación

Los resultados de esta familia se integran en la {ref}`evaluación estadística conjunta <evaluacion-e2>`, con métricas por semilla, horizonte y origen. La comparación distingue las entradas demanda/calendario de las entradas numéricas ampliadas y el aporte de texto oficial.

## Fuentes

La formulación y las dimensiones se verifican en las clases del código mostrado.. La descripción corresponde a la implementación del experimento; las adaptaciones se identifican en el texto.

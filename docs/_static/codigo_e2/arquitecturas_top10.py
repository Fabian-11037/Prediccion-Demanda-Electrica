"""Fichas y esquemas reproducibles de N-BEATS y TimesNet."""
from pathlib import Path
import json,ast
ROOT=Path(__file__).resolve().parents[1]

def main():
 from esquemas_top10_detallados import generar
 generar(ROOT)
 source=(ROOT/'book/_static/codigo_e2/top10_adicionales/src/entrenamiento/redes_e2.py').read_text(encoding='utf8');tree=ast.parse(source)
 for f,names in [('nbeats',['_Base','_BloqueNBEATS','NBEATS']),('timesnet',['_Base','TimesNetE2'])]:
  parts=['import torch\nfrom torch import nn\n']
  if f=='timesnet':parts+=['import sys\nfrom pathlib import Path\nsys.path.insert(0, str(Path(__file__).resolve().parent / "top10_adicionales/src/vendor/timesnet"))\nimport TimesNet as _TN\nfrom layers.Embed import TokenEmbedding, PositionalEmbedding\n']
  parts +=[ast.get_source_segment(source,n) for n in tree.body if isinstance(n,ast.ClassDef) and n.name in names]
  (ROOT/f'book/_static/codigo_e2/{f}_red.py').write_text('\n\n'.join(parts),encoding='utf8')
 pipeline='''"""Inferencia causal de checkpoints: un origen por llamada."""
from pathlib import Path
import json,numpy as np,torch
from entrenamiento.datos import DatosNumericos
from entrenamiento.protocolo_e2 import _escalas
from entrenamiento.redes_e2 import construir

def predecir_checkpoint(raiz_datos,carpeta_checkpoint,familia,origenes,dispositivo="cpu"):
    carpeta=Path(carpeta_checkpoint)
    meta=json.loads((carpeta/"resultado.json").read_text(encoding="utf8"))
    escalas=json.loads((carpeta/"escaladores.json").read_text(encoding="utf8"))
    datos=DatosNumericos(Path(raiz_datos))
    posiciones=datos.indice.get_indexer(origenes)
    assert (posiciones>=168).all()
    entradas=datos.construir(posiciones,_escalas(escalas,datos))
    red=construir(familia,meta["configuracion"],33,23,168,24,None,None).to(dispositivo)
    red.load_state_dict(torch.load(carpeta/"modelo.pt",map_location=dispositivo,weights_only=True),strict=True)
    red.eval();red.requires_grad_(False)
    resultados=[]
    with torch.inference_mode():
        for i in range(len(origenes)):
            args=[torch.from_numpy(a[i:i+1]).to(dispositivo) for a in
                  [entradas.x,entradas.cal_p,entradas.cal_f,entradas.dia]]
            pred=red(*args).cpu().numpy().astype(float)[0]*entradas.nivel[i]
            resultados.append(pred)
    pred=np.stack(resultados)
    error=pred-entradas.y
    metricas={"MASE":float(np.abs(error).mean()/escalas["q168_kwh"]),
              "MAE":float(np.abs(error).mean()),"RMSE":float(np.sqrt((error**2).mean()))}
    return pred,metricas
'''
 (ROOT/'book/_static/codigo_e2/top10_adicionales/pipeline_prueba.py').write_text(pipeline,encoding='utf8')
 for f in ['nbeats','timesnet']:
  prefix=f'''(e2-{f})=\n# 10.{14 if f=='nbeats' else 15}. {'N-BEATS: reconstrucción y pronóstico residual' if f=='nbeats' else 'TimesNet: variación temporal bidimensional'}\n\n'''
  if f=='nbeats':
   desc=r'''## Arquitectura y fundamentos

N-BEATS divide el pronóstico en contribuciones sucesivas. Cada bloque recibe el residuo de la demanda, reconstruye una parte de la historia (*backcast*) y produce un pronóstico parcial. La reconstrucción se resta de la entrada del bloque siguiente; los pronósticos se suman. Así, los bloques pueden representar estructuras temporales complementarias sin una descomposición estacional impuesta.

La configuración seleccionada usa 30 bloques independientes, cada uno con cuatro capas densas de 512 unidades, ReLU y dropout 0,1. El residuo contiene las 168 horas de demanda en orden inverso. Cada bloque incorpora además las últimas 24 horas de las 29 covariables y tres máscaras de ausencia, y las 23 variables de calendario para las 24 horas futuras: la entrada tiene dimensión $168+24\times32+24\times23=1488$.

$$u_b=[r_b,\operatorname{vec}(X_{exo,24}),\operatorname{vec}(C_f)],\quad
h_b=\operatorname{MLP}_b(u_b),\quad [\widehat b_b,\widehat f_b]=W_bh_b+a_b,$$
$$r_{b+1}=r_b-\widehat b_b,\qquad
\widehat{\mathbf y}=\ell_t\left(\mathbf1+\sum_{b=1}^{30}\widehat f_b\right).$$

La base genérica convierte directamente los coeficientes aprendidos en reconstrucción y pronóstico; no corresponde a las bases interpretables de tendencia y estacionalidad de otra variante de N-BEATS. Las covariables se mantienen en todos los bloques y solo se actualiza el residuo de demanda. El modelo contiene 49.464.960 parámetros. Su coste denso crece aproximadamente como $O(B[d_{in}w+3w^2+w(L+H)])$, con $B=30$ bloques y ancho $w=512$.

La arquitectura sigue el mecanismo de doble residuo de [Oreshkin et al., N-BEATS, ICLR 2020](https://arxiv.org/abs/1905.10437), ampliado con las covariables numéricas y el calendario del proyecto.
'''
  else:
   desc=r'''## Arquitectura y fundamentos

TimesNet representa una serie en dos dimensiones: posición dentro de un ciclo y posición entre ciclos. Primero identifica periodos mediante la transformada de Fourier. Para cada periodo reorganiza la historia como una matriz y aplica convoluciones bidimensionales, que pueden aprender variaciones dentro del ciclo y entre ciclos. Después devuelve cada representación al eje temporal y combina sus resultados según la amplitud de las frecuencias seleccionadas.

La entrada contiene 168 horas y 33 canales: 30 variables históricas y tres máscaras. Se normaliza cada canal con la media y desviación de su propia ventana; esta operación usa exclusivamente historia disponible. Una convolución de valores, una representación posicional y una proyección del calendario histórico producen vectores de dimensión 16. Una capa lineal extiende el eje temporal de 168 a 192 posiciones.

$$Z=(X-\mu_t)/\sqrt{\sigma_t^2+10^{-5}},\qquad
E=\operatorname{Conv1D}(Z)+P+W_cC_p.$$

Cada uno de los dos TimesBlock selecciona cinco frecuencias dominantes. Para $T=192$, la frecuencia $f_i$ determina $p_i=\lfloor T/f_i\rfloor$. El relleno permite reorganizar los vectores como una matriz de ciclos por periodo; un módulo Inception con seis tamaños de núcleo extrae patrones bidimensionales.

$$U_i=\operatorname{reshape}_{p_i}(E),\quad
V_i=\operatorname{restore}(\operatorname{Inception2D}(U_i)),\quad
E'=\operatorname{LayerNorm}\left(E+\sum_{i=1}^{5}w_iV_i\right),\quad
w=\operatorname{softmax}(A).$$

La proyección final recupera 33 canales y revierte la normalización por ventana. El canal de demanda de las últimas 24 posiciones se combina con el calendario futuro mediante una lectura lineal compartida, y el nivel de demanda $\ell_t$ restituye las unidades: $\widehat y_{t+h}=\ell_t[1+W_o(d_{t+h},C_{t+h})]$. Hay 328.266 parámetros entrenables. El coste combina FFT $O(Td\log T)$ y convoluciones sobre cinco representaciones periódicas por bloque.

La selección de frecuencias del código oficial promedia amplitudes sobre los elementos del lote. Por ello la prueba recibe un único origen por llamada: cada pronóstico depende de su propia historia. La validación y las métricas conservadas de parada corresponden al protocolo de ajuste registrado; la inferencia de prueba y de perturbaciones usa lote uno.

La implementación reutiliza TimesBlock e Inception de [Wu et al., TimesNet, ICLR 2023](https://arxiv.org/abs/2210.02186), con calendario histórico y lectura futura adaptados al panel multivariado.
'''
  figs=f'''\n## Esquema global\n\n[![Arquitectura global de {f}](../_static/arquitecturas_presentacion/{f}_global.png)](../_static/arquitecturas_presentacion/{f}_global.png)\n\n'''+('Los bloques reducen sucesivamente el residuo de la demanda y suman sus contribuciones a las siguientes 24 horas.' if f=='nbeats' else 'La transformación periódica permite combinar dependencias dentro y entre ciclos; la lectura final incorpora el calendario conocido.')+f'''\n\n## Detalle de las operaciones\n\n[![Operaciones de {f}](../_static/arquitecturas_presentacion/{f}_detalle.png)](../_static/arquitecturas_presentacion/{f}_detalle.png)\n\n'''+('La reconstrucción de 168 valores y el pronóstico de 24 valores son dos salidas del mismo bloque; solo la reconstrucción modifica la entrada del siguiente.' if f=='nbeats' else 'Los pesos por frecuencia agregan cinco representaciones, sin utilizar observaciones posteriores al origen de pronóstico.')
  code=f'''\n\n## Código de la arquitectura\n\n```{{literalinclude}} ../_static/codigo_e2/{f}_red.py\n:language: python\n:linenos:\n```\n\n## Pipeline de entrenamiento y prueba\n\nSe comparan 20 configuraciones en 2018; cinco finalistas se evalúan en 2019–2021 y se selecciona la menor media de los cuatro años. El ajuste final utiliza cinco semillas (1000–1004), máximo de 200 épocas, mínimo de 10 y paciencia de 6. El checkpoint se restaura en el mínimo de parada interna. La reserva contiene 803 orígenes de 2019-10-21 a 2021-12-31; la prueba comprende 1.461 orígenes de 2022–2025. Las imputaciones y escalas se ajustan con el tramo de entrenamiento. Precio, hidrología y clima se usan como historia con sus rezagos de disponibilidad.\n\n### Preparación de datos\n\n```{{literalinclude}} ../_static/codigo_e2/top10_adicionales/src/entrenamiento/datos.py\n:language: python\n:linenos:\n```\n\n### Selección y ajuste con parada temprana\n\n```{{literalinclude}} ../_static/codigo_e2/top10_adicionales/src/entrenamiento/protocolo_e2.py\n:language: python\n:pyobject: modelo_estandar\n:linenos:\n```\n\n```{{literalinclude}} ../_static/codigo_e2/top10_adicionales/src/entrenamiento/protocolo.py\n:language: python\n:pyobject: corrida\n:linenos:\n```\n\n### Prueba por origen con pesos congelados\n\nEl siguiente código restaura la configuración y los escaladores del checkpoint y obtiene las métricas sin modificar los pesos. TimesNet se evalúa con lote uno.\n\n```{{literalinclude}} ../_static/codigo_e2/top10_adicionales/pipeline_prueba.py\n:language: python\n:linenos:\n```\n\nLos resultados, curvas de cinco semillas, métricas por horizonte y comparaciones están en la {{ref}}`evaluación integrada <evaluacion-e2>`.\n'''
  training="""### Optimización y restauración del checkpoint

```{literalinclude} ../_static/codigo_e2/top10_adicionales/src/entrenamiento/motor.py
:language: python
:pyobject: entrenar
:linenos:
```

"""
  code=code.replace('### Prueba por origen con pesos congelados',training+'### Prueba por origen con pesos congelados')
  vendor="""
### Bloque periódico y convolución de múltiples escalas

```{literalinclude} ../_static/codigo_e2/top10_adicionales/src/vendor/timesnet/TimesNet.py
:language: python
:pyobject: FFT_for_Period
:linenos:
```

```{literalinclude} ../_static/codigo_e2/top10_adicionales/src/vendor/timesnet/TimesNet.py
:language: python
:pyobject: TimesBlock
:linenos:
```

```{literalinclude} ../_static/codigo_e2/top10_adicionales/src/vendor/timesnet/layers/Conv_Blocks.py
:language: python
:pyobject: Inception_Block_V1
:linenos:
```
"""
  if f=='timesnet':code=code.replace('## Pipeline de entrenamiento y prueba',vendor+'\n## Pipeline de entrenamiento y prueba')
  figs+='\n\nEl {ref}`código de generación de estos esquemas <esquemas-top10-detallados>` conserva sus dimensiones y conexiones.\n'
  (ROOT/f'book/arquitecturas/{f}.md').write_text(prefix+desc+figs+code,encoding='utf8')
 path=ROOT/'results/arquitecturas_e2/indice_documentacion.json';idx=json.loads(path.read_text(encoding='utf8'));idx=[m for m in idx if m['key'] not in ['nbeats','timesnet']]
 idx += [dict(key=f,title=label,code=f'{f}_red.py',names=['NBEATS' if f=='nbeats' else 'TimesNetE2'],package='top10_adicionales') for f,label in [('nbeats','N-BEATS'),('timesnet','TimesNet')]]
 path.write_text(json.dumps(idx,ensure_ascii=False,indent=2),encoding='utf8')
 print('Dos fichas, cuatro esquemas y código de preparación, ajuste y prueba incorporados.')
if __name__=='__main__':main()

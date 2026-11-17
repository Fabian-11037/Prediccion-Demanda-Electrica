(fundamentos-arquitecturas)=
# 10.10.1. Notación y operaciones compartidas

## Qué recibe y qué produce una red

En el origen diario $t$, la historia $X_t$ tiene $L=168$ filas horarias y una columna por variable. El calendario futuro $F_t$ tiene $H=24$ filas. Un lote de $B$ orígenes se representa con tensores $X\in\mathbb R^{B\times L\times C}$ y $F\in\mathbb R^{B\times H\times23}$. La salida tiene forma $B\times24$: cada fila contiene las 24 predicciones emitidas simultáneamente para un día. La letra $B$ denota aquí el tamaño del lote, no el número de remuestreos estadísticos.

El nivel $\ell_t=L^{-1}\sum_{s=t-L}^{t-1}x^{dem}_s$ procede exclusivamente de la demanda histórica habilitada. La etiqueta normalizada es $r_{t,h}=y_{t,h}/\ell_t$ y la inversión devuelve $\widehat y_{t,h}=\ell_t\widehat r_{t,h}$. Las redes que predicen una corrección utilizan $\widehat r=1+g_\theta(X,F)$; el MLP y ARX producen directamente la razón normalizada. La etiqueta y el nivel cumplen funciones diferentes: el nivel puede calcularse en el origen; la etiqueta se conoce después.

Las covariables se estandarizan con estadísticas del tramo de ajuste. Una observación ausente se sustituye por una mediana de ese mismo tramo y conserva su indicador de ausencia. Esas estadísticas se mantienen al transformar parada interna, validación y prueba. RevIN, cuando se utiliza dentro de PatchTST, añade una normalización reversible por ventana y canal; no reemplaza las reglas temporales del escalado externo.

## Proyecciones, activaciones y compuertas

Una capa lineal calcula $z=Wx+b$: combina atributos y cambia su dimensión. ReLU aplica $\max(0,z)$ y permite que una composición de capas represente relaciones no lineales. La sigmoide $\sigma(z)=1/(1+e^{-z})$ produce pesos entre cero y uno; $\tanh(z)$ produce valores entre menos uno y uno. Una compuerta multiplica una representación por un peso aprendido: puede atenuarla de forma distinta para cada observación.

El dropout utiliza durante entrenamiento una máscara $m_i\sim\operatorname{Bernoulli}(1-p)$ y calcula $D_p(z_i)=m_i z_i/(1-p)$. En evaluación actúa como la identidad. La normalización de capa, LayerNorm, centra y escala los atributos de cada representación; BatchNorm utiliza estadísticas por canal del lote y conserva estadísticas para evaluación. Las implementaciones de las fichas especifican cuál se utiliza y dónde.

## Qué significan consulta, clave y valor

La atención compara una consulta con las claves de los elementos disponibles y mezcla sus valores. La consulta expresa qué información busca una representación; la clave permite medir afinidad; el valor contiene la información que se transmite. Son proyecciones aprendidas:

$$Q=ZW_Q,\qquad K=MW_K,\qquad V=MW_V,\qquad
A=\operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}+\mathcal M\right),\qquad O=AV.$$

El softmax se aplica por consulta, $\alpha_j=\exp(s_j)/\sum_k\exp(s_k)$, por lo que los pesos de los elementos permitidos suman uno. La máscara $\mathcal M$ vale cero donde se permite consultar y menos infinito donde se prohíbe. En atención propia, $Z=M$; en atención cruzada, las consultas y la memoria proceden de representaciones distintas.

La atención multicabeza repite el cálculo con proyecciones distintas. Habitualmente concatena las salidas y aplica otra proyección: $\operatorname{MultiHead}=\operatorname{Concat}(O_1,\ldots,O_a)W_O$. TFT utiliza la variante interpretable descrita en su ficha, con valores compartidos y promedio entre cabezas. Un bloque Transformer añade conexiones residuales, normalización y una red por posición, $\operatorname{FFN}(z)=W_2\phi(W_1z+b_1)+b_2$. La posición y el tipo de token determinan qué relación modela: entre parches en PatchTST, entre variables en iTransformer y entre consultas del horizonte y memoria en TGForecaster.

Los pesos de atención son componentes del cálculo, no estimaciones de efectos causales. Tampoco identifican por sí solos la importancia global de una variable.

## Pérdida, optimización y parada

Para un lote, el MAE normalizado es $\mathcal L=(BH)^{-1}\sum_{b,h}|r_{b,h}-\widehat r_{b,h}|$. Algunas familias base utilizan Huber con transición $\eta=0,05$:

$$\ell_\eta(e)=\begin{cases}\frac12e^2,& |e|\le\eta,\\
\eta\left(|e|-\frac12\eta\right),& |e|>\eta.
\end{cases}$$

AdamW actualiza los pesos utilizando promedios de gradientes y de sus cuadrados, con decaimiento de pesos separado. El recorte limita la norma global del gradiente a uno. La reducción de tasa en meseta y la paciencia de parada son reglas distintas: la primera modifica la tasa de aprendizaje y la segunda termina el ajuste. Las fichas conservan la paciencia efectiva del motor correspondiente. El estado restaurado es el mínimo absoluto observado en la reserva de parada interna, aunque el contador de paciencia exija una mejora mínima para reiniciarse.

El MASE usado para comparar se calcula después de invertir a kWh y divide el MAE por la escala fija q168. No se confunden pérdida de optimización normalizada y métrica de evaluación. La dispersión entre cinco semillas caracteriza inicialización; los intervalos por bloques caracterizan variación temporal condicional a esas semillas.

## Cómo interpretar el coste

Una proyección densa requiere aproximadamente $O(d_{in}d_{out})$ operaciones por observación. Una LSTM de ancho $d$ sobre $L$ horas requiere $O(Ld(C+d))$; una convolución requiere $O(LkC_{in}C_{out})$. La atención propia sobre $T$ tokens incluye un término $O(T^2d)$ y la atención cruzada, $O(T_qT_md)$. Estos términos explican por qué cambia el coste al usar horas, parches o variables como tokens. No sustituyen tiempos medidos ni incluyen toda la memoria del ajuste, redes internas, preprocesamiento o codificación de texto.

Las fichas contienen el código de las redes y los pipelines completos. Los programas de entrenamiento se presentan como fuentes reproducibles; la construcción documental ejecuta el análisis de artefactos conservados. La compatibilidad del adaptador portable se mantiene separada de los orquestadores originales dependientes de un sistema operativo.

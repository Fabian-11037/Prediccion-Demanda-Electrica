"""Documenta las implementaciones identificadas, sin importar ni ejecutar sus redes."""
from pathlib import Path
import ast,json,shutil,textwrap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,Circle,FancyArrowPatch

R=Path(__file__).resolve().parents[1]
O=R/'results/arquitecturas_e2';FU=O/'fuentes'
BOOK=R/'book/arquitecturas';BOOK.mkdir(exist_ok=True)
STATIC=R/'book/_static/arquitecturas';STATIC.mkdir(exist_ok=True)
COD=R/'book/_static/codigo_e2';COD.mkdir(exist_ok=True)
shutil.copy2(R/'scripts/pipeline_ampliado_portable.py',COD/'pipeline_ampliado_portable.py')
shutil.copytree(FU,COD,dirs_exist_ok=True)
OLD=R/'results/repositorio/d1ae9a4e09dd1dead4af1329f16aa3e8d0f2c3d3'
shutil.copytree(OLD/'src',COD/'base/src',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
reg=json.loads((OLD/'results/consolidado/modelos.json').read_text())
models=[]
def add(key,title,blocks,mathcards,text,code,names,package=None):
    models.append(dict(key=key,title=title,blocks=blocks,cards=mathcards,text=text,code=code,names=names,package=package))

add('arx','ARX lineal regularizado',
    ['48 rezagos de demanda\n+ calendario de 24 h','Estandarización\n600 atributos','Mapa lineal\n600 → 24','Reescala por nivel\n24 pronósticos'],
    [r'$u=[r_{t-48:t};\mathrm{vec}(F)]$',r'$z_j=(u_j-\mu_j)/s_j$',r'$\hat r=Wz+b$',r'$J=\mathrm{MAE}+\lambda\|W\|_1$'],
r'''ARX relaciona la demanda futura con sus propios rezagos y con variables externas conocidas: aquí, el calendario. Cada una de las 24 horas tiene una combinación lineal distinta de los mismos atributos. Una subida de un rezago produce un cambio proporcional determinado por su coeficiente; no hay capas ocultas ni interacciones aprendidas entre atributos. Es una referencia de complejidad baja para valorar si las redes justifican su coste.

La entrada contiene los últimos 48 valores de demanda divididos por el nivel de la ventana y las 23 variables de calendario de cada una de las 24 horas futuras: $48+24\times23=600$ atributos. Aunque la ventana que calcula el nivel tiene 168 horas, la regresión utiliza directamente 48 rezagos. Cada atributo se estandariza con la media y desviación del tramo de ajuste.

$$z_j=\frac{u_j-\mu_j}{s_j},\qquad \widehat{\mathbf r}=Wz+b,\qquad \widehat{\mathbf y}=\ell_t\widehat{\mathbf r}.$$

Aquí $W\in\mathbb R^{24\times600}$ y $b\in\mathbb R^{24}$; hay 14.424 parámetros entrenables. La configuración seleccionada utiliza Lasso, con $\lambda=1,3734\times10^{-7}$. Se minimiza el MAE normalizado más $\lambda\sum_{h,j}|W_{hj}|$, lo que favorece coeficientes pequeños o nulos. La penalización actúa sobre los pesos, no sobre los sesgos. Los valores de escalado se conservan junto al checkpoint; no se vuelven a estimar en prueba.''',
    'base/src/demanda/modelos_base.py',['Tabular','ARXLineal'])

add('mlp','MLP: perceptrón multicapa',
    ['168 rezagos\n+ 552 datos de calendario','Atributos estandarizados\n720 entradas','Densas 128 → 128\nReLU + dropout','Capa lineal de 24\n× nivel de ventana'],
    [r'$u\in\mathbb{R}^{720}$',r'$h_1=\mathrm{ReLU}(W_1u+b_1)$',r'$h_2=\mathrm{ReLU}(W_2h_1+b_2)$',r'$\hat y=\ell_t(W_3h_2+b_3)$'],
r'''El MLP trata la historia y el calendario como un vector. A diferencia de una regresión, sus capas pueden combinar no linealmente una hora pasada con un festivo o una estación del año. La posición temporal se conserva por el orden fijo de los rezagos; la red no tiene memoria recurrente ni un filtro compartido entre horas.

Los 168 valores de demanda normalizados por el nivel y los $24\times23=552$ valores de calendario futuro forman un vector de 720 atributos. Después de estandarizarlo con entrenamiento, se aplican dos capas de 128 unidades, ReLU y dropout. La salida lineal produce simultáneamente 24 razones de demanda respecto del nivel de la ventana.

$$h^{(j)}=D_{p}\!\left[\max(0,W_jh^{(j-1)}+b_j)\right],\quad j=1,2,
\qquad \widehat{\mathbf y}=\ell_t(W_3h^{(2)}+b_3).$$

$D_p$ apaga aleatoriamente unidades durante entrenamiento y se desactiva en evaluación. La tasa seleccionada es aproximadamente 0,00355. Las matrices tienen dimensiones $128\times720$, $128\times128$ y $24\times128$, para 111.896 parámetros. La red aprende qué combinaciones de rezagos son útiles, pero su aplanamiento exige parámetros separados para patrones desplazados en el tiempo.

La pérdida de ajuste es Huber normalizada con transición 0,05: es cuadrática cerca de cero y lineal para errores grandes. AdamW añade decaimiento de pesos de aproximadamente 0,000383; la tasa inicial es 0,00469. El mínimo de la pérdida de parada interna determina los pesos restaurados; el MAE/MASE de prueba se calcula después en kWh.''',
    'base/src/demanda/modelos_base.py',['Tabular','MLP'])

add('lstm','LSTM: memoria recurrente con compuertas',
    ['168 h × 24 variables\nDemanda y calendario pasado','LSTM de una capa\n48 unidades','Estado final + calendario\n48 + 552 atributos','Densa 600 → 24\n1 + corrección'],
    [r'$f_t,i_t,o_t=\sigma(W[x_t,h_{t-1}]+b)$',r'$c_t=f_t c_{t-1}+i_t\tilde c_t$',r'$h_t=o_t\tanh(c_t)$',r'$\hat y=\ell_t[1+W(h_L,F)+b]$'],
r'''Una LSTM recorre la historia hora por hora y conserva un estado de memoria. Sus compuertas deciden qué información retener, incorporar y exponer. Esta organización permite representar dependencias de varias horas sin asignar un coeficiente independiente a cada posición, aunque no garantiza utilizar toda la ventana.

La implementación seleccionada recibe 168 pasos con 24 variables por paso: demanda normalizada y 23 variables de calendario pasado. Tiene una capa recurrente de 48 unidades. El estado oculto final se concatena con los 552 atributos de calendario futuro y una capa lineal de 24 salidas aprende desviaciones respecto de uno. Hay 28.632 parámetros.

$$\begin{aligned}
f_s&=\sigma(W_f x_s+U_fh_{s-1}+b_f),&i_s&=\sigma(W_i x_s+U_ih_{s-1}+b_i),\\
\tilde c_s&=\tanh(W_c x_s+U_ch_{s-1}+b_c),&c_s&=f_s\odot c_{s-1}+i_s\odot\tilde c_s,\\
o_s&=\sigma(W_o x_s+U_oh_{s-1}+b_o),&h_s&=o_s\odot\tanh(c_s).
\end{aligned}$$

La memoria $c_s$ y la salida $h_s$ son distintas; $\odot$ representa producto componente a componente. El decodificador usado es directo: $\widehat{\mathbf y}=\ell_t[\mathbf1+W_o'[D(h_{168});\operatorname{vec}(F)]+b_o']$. No realimenta una predicción para obtener la siguiente. El dropout de 0,1 se aplica al estado antes de la salida; no hay dropout entre capas recurrentes porque solo existe una. El entrenamiento utiliza Huber, AdamW y parada interna.''',
    'base/src/demanda/variantes_v4.py',['Secuencial'])

add('cnn1d','CNN1D: filtros locales sobre la historia',
    ['168 h × 24 variables\nsolo historia disponible','Conv1D: 16 filtros\nkernel 5 + ReLU','Dropout + promedio de 2\n16 × 84 → vector','Calendario + densa 24\n1 + corrección'],
    [r'$z_{s,c}=\mathrm{ReLU}(\sum_{i,j}K_{cij}x_{s+i,j}+b_c)$',r'$\bar z_{u,c}=(z_{2u,c}+z_{2u+1,c})/2$',r'$v=[\mathrm{vec}(\bar z);\mathrm{vec}(F)]$',r'$\hat y=\ell_t[1+Wv+b]$'],
r'''La CNN1D busca patrones cortos, como una rampa o un cambio de pendiente, mediante filtros compartidos en todas las posiciones. Compartir pesos permite detectar una forma similar en distintas horas. La agregación posterior combina esas respuestas con el calendario de las horas que se pronostican.

La red seleccionada tiene una sola convolución de 16 filtros y núcleo de cinco horas sobre 24 canales de entrada. Se añade relleno simétrico de dos posiciones para conservar la longitud de 168. Después de ReLU y dropout por canal de 0,1, un promedio de dos horas reduce la longitud a 84. Los $16\times84=1.344$ atributos se concatenan con los 552 del calendario futuro; una capa $1.896\rightarrow24$ produce las correcciones. El total es de 47.464 parámetros.

$$z_{s,c}=\operatorname{ReLU}\!\left(b_c+\sum_{j=1}^{24}\sum_{i=-2}^{2}K_{cij}x_{s+i,j}\right),\quad
\widehat{\mathbf y}=\ell_t\left[\mathbf1+W[\operatorname{vec}(\operatorname{AvgPool}_2z);\operatorname{vec}(F)]+b\right].$$

El filtro no es causal dentro de la ventana, porque utiliza posiciones vecinas a izquierda y derecha. Esto no implica consultar el horizonte: todas las posiciones de la ventana son anteriores al origen. Es diferente de la TCN, que sí impone causalidad en cada convolución. La capa final reúne toda la historia filtrada, mientras que el campo local de cada filtro es de cinco horas. Se ajusta con Huber y AdamW; los escaladores y los pesos restaurados se mantienen fijos al evaluar.''',
    'base/src/demanda/reentrenamiento_v3.py',['Secuencial'])

add('patchtst','PatchTST numérico',
    ['168 h × 83 canales\nRevIN por canal','Parches 16 / paso 8\n21 tokens por canal','Transformer: 3 capas\nd=16, 4 cabezas','Cabeza por canal\n+ lectura afín exógena'],
    [r'$\tilde x=(x-\mu_t)/s_t$',r'$E=WP_{16,8}(\tilde x)+E_{pos}$',r'$A=\mathrm{softmax}(QK^T/\sqrt{d_k})V$',r'$\hat r_h=w^T[p_h;F_h]+b$'],
r'''PatchTST divide cada serie en pequeños segmentos superpuestos y trata cada segmento como un token. La atención conecta segmentos distantes sin recorrerlos secuencialmente. El codificador comparte parámetros entre canales, de modo que la demanda, el precio y cada covariable se procesan con la misma transformación temporal.

La configuración numérica usa 30 variables históricas, 23 de calendario pasado y 30 máscaras de ausencia: 83 canales. Cada historia de 168 horas se normaliza reversiblemente con RevIN, se rellena con ocho copias del último valor y se divide en parches de 16 con paso 8: $\lfloor(168+8-16)/8\rfloor+1=21$ tokens. Se proyectan a dimensión 16 y se añaden posiciones aprendidas. Tres bloques de atención, cuatro cabezas y una red interna de ancho 128 producen la representación.

$$P_c=\operatorname{patch}_{16,8}(\tilde x_c),\quad E_c=P_cW_P+E_{pos},\qquad
\operatorname{Att}(Q,K,V)=\operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V.$$

RevIN resta media y divide por dispersión de cada ventana y canal; su inversión devuelve la salida a la escala del canal. Estas estadísticas de entrada no incluyen etiquetas futuras. La cabeza aplanada genera 24 valores por canal. Una lectura afín explícita reúne esos valores y el calendario futuro para pronosticar únicamente demanda; así las covariables sí tienen un camino hacia el objetivo. Sin esa lectura, el procesamiento independiente no transferiría sus salidas a demanda. Hay 24.982 parámetros en la configuración seleccionada P20.

Esta es una adaptación del código oficial, no una identidad con el experimento del artículo. La normalización externa por nivel y las máscaras son parte del contrato numérico; RevIN opera además dentro del núcleo. El motor utiliza MAE normalizado, AdamW, hasta 200 épocas y paciencia efectiva de diez desde un mínimo de diez épocas.''',
    'numerico_repositorio/src/demanda/patchtst_numerico.py',['Preparador','PatchTSTNumerico','entrenar','predecir'],package='numerico_repositorio')

add('nhits','N-HiTS: interpolación jerárquica residual',
    ['Demanda normalizada\n+ exógenas y calendario','Tres bloques multiescala\npooling 2, 2, 1','MLP 512 + backcast\nnodos 6, 12, 24','Suma de contribuciones\n+ último nivel observado'],
    [r'$r_0=\mathrm{reverse}(y/\ell_t)$',r'$\theta_b=\mathrm{MLP}_b(\mathrm{pool}(r_b),X,F)$',r'$r_{b+1}=r_b-\hat b_b$',r'$\hat r=r_{ultimo}+\sum_b I_b\theta_b^{f}$'],
r'''N-HiTS construye el pronóstico por contribuciones de distinta resolución. Un bloque explica una parte de la historia y propone una parte de la salida; el siguiente trabaja con el residuo que queda. La interpolación permite que un bloque describa el horizonte con pocos nodos, mientras otro añade variaciones más finas.

La implementación tiene tres bloques oficiales de N-HiTS con MLP de ancho 512, agrupamientos temporales 2, 2 y 1 y factores de reducción de salida 4, 2 y 1. Para 24 horas se predicen respectivamente 6, 12 y 24 nodos. Una matriz fija realiza interpolación lineal equivalente a `interpolate(..., align_corners=False)`. Cada bloque usa la historia residual, 59 exógenas históricas —29 variables distintas de demanda y 30 máscaras— y las 23 variables de calendario conocidas.

$$\theta_b=f_b(\operatorname{pool}_b(r_b),X_{exo},F),\quad
\widehat b_b=\theta_b^{back},\quad \widehat f_b=I_b\theta_b^{for},\qquad
r_{b+1}=r_b-\widehat b_b.$$

El pronóstico normalizado empieza en el último valor de demanda de la ventana y suma $\sum_b\widehat f_b$. El *backcast* reconstruye historia; no es una etiqueta futura ni una segunda serie objetivo. El modelo devuelve $\widehat{\mathbf y}=\ell_t\widehat{\mathbf r}$. Su configuración seleccionada N14 contiene 17.489.442 parámetros. Este tamaño se explica por las entradas multivariadas aplanadas de los bloques y no implica, por sí mismo, mayor capacidad predictiva efectiva.

El modelo reutiliza el preparador y el optimizador numérico de PatchTST, pero la red es diferente y no incorpora su lectura afín. Se conserva el procesamiento multiescala, el esquema residual y la interpolación. Las fuentes, licencia y revisión de NeuralForecast acompañan al código.''',
    'numerico_repositorio/src/demanda/nhits_numerico.py',['BaseLinealDeterminista','NHITSNumerico'],package='numerico_repositorio')

add('tft','TFT: selección de variables y atención interpretable',
    ['56 variables pasadas\n23 futuras conocidas','Selección GRN + softmax\nLSTM encoder / decoder','Compuertas + atención\ncausal, 4 cabezas','GRN por posición\n24 salidas escalares'],
    [r'$v_s=\mathrm{softmax}(\mathrm{GRN}([e_{s,j}]))$',r'$z_s=\sum_j v_{s,j}\mathrm{GRN}_j(e_{s,j})$',r'$A=\mathrm{softmax}(QK^T/\sqrt{d_k}+M)$',r'$\hat y_h=\ell_t[1+w^T\psi_h+b]$'],
r'''TFT combina selección de variables, memoria recurrente y atención. Primero decide qué variables ponderar en cada hora; después resume la secuencia y permite que cada hora del horizonte consulte los pasos relevantes de la historia. Las compuertas residuales regulan cuánto de cada transformación se incorpora.

La configuración seleccionada tiene ancho 160 y cuatro cabezas. Recibe 56 variables pasadas: 30 históricas, tres indicadores de ausencia por bloque y 23 de calendario. El futuro contiene únicamente 23 variables de calendario. Cada variable continua tiene un embedding lineal y una GRN propia. Una segunda GRN produce pesos softmax $v_{s,j}$; su suma es uno y la representación seleccionada es $z_s=\sum_jv_{s,j}\operatorname{GRN}_j(e_{s,j})$.

Una GRN utiliza una transformación no lineal, una compuerta y un atajo:

$$\operatorname{GRN}(a)=\operatorname{LayerNorm}\!\left(Sa+\operatorname{GLU}(W_1\operatorname{ELU}(W_2a+b_2)+b_1)\right),\quad
\operatorname{GLU}(u)=\sigma(W_gu+b_g)\odot(W_vu+b_v).$$

El codificador LSTM procesa las 168 horas y su estado inicializa al decodificador que recibe el calendario futuro. El enriquecimiento GRN precede a la atención interpretable: las cabezas comparten la proyección de valores y se promedian. La máscara impide que la consulta de la hora $h$ utilice representaciones del decodificador posteriores a ella. Las GRN por posición y las compuertas finales producen una salida escalar por hora. El total es de 11.715.596 parámetros.

Se omiten codificadores estáticos porque se estudia una sola serie. Se entrena un pronóstico puntual con MAE normalizado, no todos los cuantiles del artículo. Los pesos de selección describen el mecanismo del modelo; no son efectos causales de las variables. Esta documentación y los resultados usan la configuración TFT seleccionada del paquete, identificada por sus fuentes y checkpoints.''',
    'Modelo3_TFT/src/entrenamiento/redes.py',['_GLU','_CompuertaSuma','_GRN','_SeleccionVariables','_AtencionInterpretable','TFT'],package='Modelo3_TFT')

add('tcn','TCN: convoluciones causales dilatadas',
    ['168 h × 56 canales\nPasado numérico y calendario','Tres bloques residuales\n64, 64, 64 filtros','Dos conv / bloque\ndilataciones 1, 4, 16','Último estado + calendario\nMLP → 24 horas'],
    [r'$z_s=\sum_{i=0}^{k-1}W_i x_{s-di}$',r'$h_{b+1}=\mathrm{ReLU}(S h_b+F_b(h_b))$',r'$R=1+2(k-1)(1+4+16)=169$',r'$\hat y=\ell_t[1+\mathrm{MLP}(h_L,F)]$'],
r'''Una TCN aumenta su alcance temporal separando las posiciones que consulta cada filtro. Con dilatación cuatro, por ejemplo, un núcleo de cinco pesos lee posiciones separadas cuatro horas. El apilamiento permite integrar semanas de historia sin usar recurrencia. La causalidad se impone recortando el relleno derecho de cada convolución.

La red seleccionada usa tres bloques de 64 filtros, dilataciones 1, 4 y 16 y núcleo cinco. Cada bloque contiene dos convoluciones, normalización de pesos, ReLU y una conexión residual; una proyección $1\times1$ adapta el atajo si cambia el número de canales. El dropout de 0,2 está activo al final del bloque central, de acuerdo con el patrón (0,1,0).

$$\operatorname{Conv}_{d}(x)_s=\sum_{i=0}^{k-1}W_i x_{s-di}+b,\qquad
h_{b+1}=\operatorname{ReLU}(S_bh_b+F_b(h_b)).$$

Con dos convoluciones por bloque, el campo receptivo es $R=1+2(k-1)\sum_bd_b=169$ horas, suficiente para las 168 de entrada. El resultado depende de contar ambas convoluciones: una sola por bloque no cubriría la ventana. La representación de la última hora se concatena con una proyección del calendario futuro y una cabeza densa produce las 24 correcciones normalizadas. Hay 169.944 parámetros.

La implementación utiliza únicamente datos anteriores al origen en las convoluciones. El calendario futuro entra por una rama separada porque sí se conoce de antemano. Los 20 candidatos efectivamente evaluados no equivalen a completar el espacio de 1.000 arquitecturas exigido por la pauta cuando no existe una configuración publicada adoptada.''',
    'Modelo4_TCN/src/entrenamiento/redes.py',['campo_receptivo_tcn','_BloqueTCN','TCN'],package='Modelo4_TCN')

add('itransformer','iTransformer: atención entre variables',
    ['33 canales históricos\n+ 23 de calendario','Una ficha por variable\nproyección 168 / 192 → 128','Atención entre 56 fichas\n2 capas, 8 cabezas','Ficha de demanda\nproyección a 24 horas'],
    [r'$e_j=W_j x_{1:L,j}+p_j$',r'$e_c=W_c[F_{pas};F_{fut}]+p_c$',r'$Z=\mathrm{Encoder}([e_j;e_c])$',r'$\hat y=\ell_t[1+W_o Z_0+b_o]$'],
r'''iTransformer invierte el significado habitual de los tokens: cada ficha representa una variable completa, no una hora. Así, la atención relaciona la historia de demanda con la del precio, el clima y la hidrología. La estructura temporal de cada variable se comprime mediante una proyección lineal antes de la atención.

Las 30 variables históricas y tres indicadores producen 33 tokens de longitud 168. Cada variable de calendario usa historia y futuro conocidos, 192 valores, y genera uno de los 23 tokens adicionales. Las proyecciones llevan ambos tipos a dimensión 128 y se suman embeddings de identidad de variable. Las mismas matrices se comparten dentro de cada tipo; $p_j$ distingue los canales.

$$e_j=W_Hx_{1:168,j}+p_j,\quad
e_c=W_C[F_{pas,c};F_{fut,c}]+p_c,\quad
Z=\operatorname{TransformerEncoder}([e_1,\ldots,e_{56}]).$$

El codificador tiene dos capas, ocho cabezas, red interna de ancho 256, GELU y normalización posterior con LayerNorm final. La atención compara consultas y claves de variables: $A=\operatorname{softmax}(QK^\top/\sqrt{d_k})$, seguido de $AV$. Solo el token de demanda alimenta la cabeza de 24 salidas: $\widehat{\mathbf y}=\ell_t[\mathbf1+W_OZ_0+b_O]$. La configuración 1 contiene 321.816 parámetros.

El calendario futuro puede relacionarse con las variables históricas dentro de la atención, pero no se crean tokens de clima o precio realizados durante el horizonte. La identidad por variable y la integración del calendario constituyen adaptaciones explícitas del experimento.''',
    'Modelo5_iTransformer/src/entrenamiento/redes.py',['NucleoITransformer','NumericoITransformer'],package='Modelo5_iTransformer')

add('tats','TaTS: texto como variables temporales auxiliares',
    ['Historia numérica\n+ calendario permitido','Texto diario: atención\n768 → 64 → 32 → 12','Canales texto + presencia\nPatchTST: d=128, 3 capas','Cabeza de demanda\n+ contexto multicanal'],
    [r'$\alpha_{dj}=\mathrm{softmax}_j(q^TKe_{dj}/\sqrt{64})$',r'$z_d=\mathrm{MLP}(\sum_j\alpha_{dj}Ve_{dj})$',r'$X^+=[X;\mathrm{repeat}_{24}(z_d,m_d)]$',r'$\hat r=1+W_{lin}Z_0+\mathrm{MLP}_{ctx}(Z,F)$'],
r'''TaTS convierte el texto en canales adicionales de una serie. En lugar de modificar el pronóstico al final, permite que el núcleo temporal procese conjuntamente la información numérica y una representación textual diaria. Los embeddings de documentos son constantes; la relevancia que los pondera se aprende con la pérdida de pronóstico.

Para cada uno de los siete cierres diarios de la historia se reúnen documentos oficiales disponibles en las 72 horas anteriores, hasta 64. La consulta $q_t$ se obtiene de los promedios numéricos de las últimas 24 y 168 horas. Cada embedding E5 $e_j\in\mathbb R^{768}$ recibe un peso:

$$\alpha_{d,j}=\operatorname{softmax}_j\!\left(\frac{q_t^TK e_{d,j}}{\sqrt{64}}\right),\quad
z_d=\operatorname{MLP}_{64\rightarrow64\rightarrow32\rightarrow12}\!\left(\sum_j\alpha_{d,j}Ve_{d,j}\right).$$

La representación diaria y su indicador de presencia se repiten en las 24 horas del bloque y se concatenan a las entradas históricas. Sin documentos, ambos quedan en cero. Se conservan únicamente documentos habilitados al cierre correspondiente; la ventana textual no lee artículos futuros.

La variante oficial seleccionada utiliza PatchTST de dimensión 128, tres capas y 16 cabezas, con 21 parches por canal. Una cabeza lineal opera sobre los parches de demanda y otra MLP de contexto combina resúmenes de todos los canales y calendario. Esta segunda vía permite que el texto y las exógenas influyan en el objetivo aun con un codificador independiente por canal. El total es de 1.428.380 parámetros.

La configuración numérica seleccionada usa un núcleo más pequeño: dimensión 64, dos capas y cuatro cabezas. Por tanto, el contraste de resultados mide el procedimiento completo de selección con y sin texto, no un efecto puro de contenido con arquitectura idéntica. La agregación aprendida y la cabeza de contexto son adaptaciones al mecanismo de TaTS.''',
    'Modelo6_TaTS/src/modelo6/redes.py',['AgregadorTexto','NucleoITransformer','_CapaPatch','NucleoPatchTST','Modelo6'],package='Modelo6_TaTS')

add('tgforecaster','TGForecaster: consultas del horizonte guiadas por texto',
    ['Historia: 33 canales\n21 parches por canal','Memoria: parches\n+ descripción + calendario','5 consultas de horizonte\ntexto + calendario + posición','Atención cruzada\nparches 8 / paso 4 → 24 h'],
    [r'$M=\mathrm{Encoder}(P(X))+W_dE_{canal}$',r'$q_k=W_dE_{dem}+W_fF_k+p_k+W_tz$',r'$H=\mathrm{softmax}(QK_M^T/\sqrt{d_k})V_M$',r'$\hat r_h=1+\sum_{k:h\in k}\omega_{kh}o_{kh}$'],
r'''TGForecaster utiliza descripciones textuales de los canales para orientar la relación entre información documental y señales numéricas. La atención cruzada distingue qué se pregunta —tokens del horizonte— de dónde se busca la respuesta —la memoria de la historia—. La variante N conserva las descripciones estáticas de canales y no incorpora documentos que cambien por origen; N+O añade el contexto documental oficial disponible.

Cada uno de los 33 canales históricos se divide en 21 parches de 16 horas con paso ocho. Sus embeddings se codifican y se suman a una proyección de la descripción E5 del canal; siete tokens adicionales representan el calendario diario pasado. La memoria reúne así $33\times21+7=700$ tokens. No se aplica RevIN dentro de este codificador; la normalización externa del contrato permanece.

El horizonte de 24 horas se divide en cinco parches de ocho con paso cuatro. La consulta de cada parche combina descripción de demanda, calendario futuro conocido, posición y resumen de documentos oficiales disponibles en el origen:

$$q_k=W_DE_{dem}+W_FF_k+p_k+m_tW_Tz_t,\quad
H=\operatorname{softmax}\!\left(\frac{QK_M^\top}{\sqrt{d_k}}\right)V_M.$$

El texto participa como parte de la consulta; las claves y valores proceden de la memoria numérica. Cada token decodifica ocho correcciones. Para las horas cubiertas por varios parches se aprende una combinación convexa: $\omega_{k,h}=\operatorname{softmax}_{k:h\in k}(a_{k,h})$, de modo que $\sum_k\omega_{k,h}=1$. La salida es $\ell_t[1+\sum_k\omega_{k,h}o_{k,h}]$.

El paquete adapta el artículo a una sola serie objetivo y a documentos disponibles al emitir el pronóstico; no incorpora texto realizado durante el horizonte. Las descripciones y los embeddings son extractores congelados; la selección documental y la fusión son entrenables. La variante N selecciona dimensión256, tres capas temporales, dos de mezcla y ocho cabezas; N+O selecciona dimensión64, dos capas temporales, una de mezcla y cuatro cabezas. Sus configuraciones distintas hacen que el contraste estime el procedimiento completo de selección, además de la incorporación de documentos.''',
    'Modelo7_TGForecaster/src/entrenamiento/redes.py',['Relevancia','_parches','_CapaPatch','_CapaMezcla','TGForecaster'],package='Modelo7_TGForecaster')

add('fusion','Fusión tardía de pronósticos numéricos y textuales',
    ['Historia y calendario\nPatchTST d=64, 2 capas','Pronóstico numérico\nb(X,F) de 24 horas','Texto: relevancia + MLP\n64 → 128 → 24','Suma de salidas\nb + alfa × presencia × texto'],
    [r'$b=f_{num}(X,F)$',r'$z=\sum_j\alpha_jVe_j$',r'$u=\mathrm{MLP}_{64\rightarrow128\rightarrow24}(z)$',r'$\hat y=\ell_t[b+\alpha m u]$'],
r'''La fusión tardía mantiene dos caminos hasta la salida: uno produce un pronóstico con la serie y el calendario; el otro propone un aporte basado en documentos. La combinación se realiza después de ambos cálculos. Esto contrasta con TaTS, que introduce el texto dentro de la secuencia que procesa el núcleo.

El núcleo seleccionado es PatchTST de dimensión 64, dos capas y cuatro cabezas, con cabeza lineal de demanda y MLP de contexto. Los documentos oficiales disponibles en las 72 horas anteriores al origen se ponderan por relevancia aprendida. El resumen de dimensión 64 pasa por una capa de 128, GELU, dropout y una salida de 24. Un escalar $\alpha$ aprendido combina las ramas:

$$\widehat{\mathbf r}=b(X,F)+\alpha\,m_t\,g(z_t),\qquad
\widehat{\mathbf y}=\ell_t\widehat{\mathbf r}.$$

La última capa textual se inicializa en cero y $\alpha$ en 0,5; el modelo empieza con el pronóstico numérico. Sin documentos, $m_t=0$ y la contribución textual es exactamente cero para los mismos pesos del núcleo. Ambas ramas se entrenan conjuntamente: no se congela la base. Por ello, esa identidad no garantiza igualdad con un checkpoint numérico entrenado por separado.

La suma no es una mezcla convexa: $\alpha$ no está restringido a [0,1] y la rama textual puede tener cualquier signo. Tampoco es la corrección residual acotada del diseño de investigación: aquí no hay radio $\rho$ ni garantía de deterioro acotado. La configuración contiene 584.649 parámetros. Su control sin texto coincide en las cinco predicciones con el control numérico de TaTS y se identifica como una única referencia en el benchmark, conservando la trazabilidad de ambos paquetes.''',
    'FusionTardia/src/entrenamiento/redes.py',['Relevancia','NucleoITransformer','_CapaPatch','NucleoPatchTST','FusionTardia'],package='FusionTardia')

def diagrama(m):
    fig,ax=plt.subplots(figsize=(16,6.1));ax.set(xlim=(0,16),ylim=(0,6.1));ax.axis('off')
    ax.text(8,5.8,m['title'],ha='center',va='center',fontsize=17,fontweight='bold',color='#253340')
    cols=['#e5f1fb','#e8f3df','#e8f3df','#eee5f6'];edges=['#7da9ce','#7fa35e','#7fa35e','#a58bc3']
    def box(x,y,w,h,txt,c,edge,fs=11):
        p=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.10',fc=c,ec=edge,lw=1.5);ax.add_patch(p)
        ax.text(x+w/2,y+h/2,txt,ha='center',va='center',fontsize=fs,color='#253340',linespacing=1.5)
    for i,txt in enumerate(m['blocks']):
        x=1.25+i*3.45;box(x,3.55,2.75,1.13,txt,cols[i],edges[i])
        if i<3:ax.add_patch(FancyArrowPatch((x+2.85,4.12),(x+3.35,4.12),arrowstyle='-|>',mutation_scale=16,color='#405363',lw=1.6))
    for x,label,col in [(.48,r'$X$','#9ed8d0'),(15.55,r'$\hat y$','#d8c7e9')]:
        ax.add_patch(Circle((x,4.12),.38,facecolor=col,edgecolor='#687985',lw=1.4));ax.text(x,4.12,label,ha='center',va='center',fontsize=17)
    ax.add_patch(FancyArrowPatch((.88,4.12),(1.15,4.12),arrowstyle='-|>',mutation_scale=14,color='#405363'))
    ax.add_patch(FancyArrowPatch((14.45,4.12),(15.1,4.12),arrowstyle='-|>',mutation_scale=14,color='#405363'))
    textual=m['key'] in {'tats','tgforecaster','fusion'}
    txt='Documentos disponibles → E5 congelado (768) → relevancia aprendida + presencia' if textual else 'Calendario de las 24 horas del horizonte: conocido en el origen'
    box(1.3,2.25,12.9,.68,txt,'#fff5d6','#d3ae45',11)
    if textual:ax.add_patch(FancyArrowPatch((8.0,3.03),(8.0,3.47),arrowstyle='-|>',mutation_scale=16,color='#a37b19',lw=1.5))
    for i,formula in enumerate(m['cards']):
        x=.25+i*4.0;box(x,.3,3.55,1.32,'',cols[i],edges[i])
        ax.text(x+.13,1.35,f'{i+1}. '+['Entrada','Representación','Composición','Pronóstico'][i],fontsize=10.5,fontweight='bold',color='#253340')
        ax.text(x+1.77,.89,formula,fontsize=10 if len(formula)<85 else 8.5,ha='center',va='center')
    fig.savefig(STATIC/(m['key']+'.svg'),bbox_inches='tight');fig.savefig(STATIC/(m['key']+'.png'),dpi=140,bbox_inches='tight');plt.close(fig)

def extraer_red(m):
    p=COD/m['code'];text=p.read_text(encoding='utf-8-sig');tree=ast.parse(text);lines=text.splitlines()
    pieces=[]
    for n in tree.body:
        if isinstance(n,(ast.Import,ast.ImportFrom,ast.Assign,ast.AnnAssign)):
            if isinstance(n,(ast.Import,ast.ImportFrom)) or (hasattr(n,'lineno') and n.end_lineno-n.lineno<5):
                pieces.append('\n'.join(lines[n.lineno-1:n.end_lineno]))
    for name in m['names']:
        found=next((n for n in ast.walk(tree) if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name==name),None)
        assert found is not None,(m['key'],name)
        start=min([found.lineno]+[d.lineno for d in found.decorator_list])
        if not any('from torch import nn' in s for s in pieces):pieces.append('import torch\nfrom torch import nn')
        pieces.append(textwrap.dedent('\n'.join(lines[start-1:found.end_lineno])))
    out=COD/(m['key']+'_red.py');out.write_text('\n\n'.join(pieces)+'\n',encoding='utf-8')
    ast.parse(out.read_text(encoding='utf-8'))

def codigo_esquema(m):
    fuente=Path(__file__).read_text(encoding='utf-8')
    node=next(n for n in ast.parse(fuente).body if isinstance(n,ast.FunctionDef) and n.name=='diagrama')
    cuerpo='\n'.join(fuente.splitlines()[node.lineno-1:node.end_lineno])
    pre='from pathlib import Path\nimport matplotlib.pyplot as plt\nfrom matplotlib.patches import FancyBboxPatch,Circle,FancyArrowPatch\nSTATIC=Path(__file__).resolve().parent.parent/"arquitecturas"\n\n'
    cfg={k:m[k] for k in ['key','title','blocks','cards']}
    (COD/('esquema_'+m['key']+'.py')).write_text(pre+cuerpo+'\n\nif __name__=="__main__":\n    diagrama('+repr(cfg)+')\n',encoding='utf-8')

def literal(rel,titulo):
    return f'### {titulo}\n\n```{{literalinclude}} ../_static/codigo_e2/{rel}\n:language: python\n:linenos:\n```\n\n'

oldpipe=(R/'results/reorganizacion_capitulo10/fuentes/10_pipeline.md').read_text(encoding='utf-8').split('## Pipeline integrado\n',1)[1].split('## Reconstrucción del benchmark',1)[0]
for n,m in enumerate(models,2):
    diagrama(m);extraer_red(m);codigo_esquema(m)
    t=f"(e2-{m['key']})=\n# 10.{n}. {m['title']}\n\n"
    t+='La {ref}`notación y las operaciones compartidas <fundamentos-arquitecturas>` definen los tensores, el nivel de demanda, la atención, las activaciones y la optimización.\n\n'
    t+='## Arquitectura y fundamentos\n\n'+m['text']+'\n\n'
    t+=f"![Flujo de {m['title']}](../_static/arquitecturas/{m['key']}.svg)\n\n"
    t+='El esquema distingue el procesamiento de la historia, la composición de representaciones y la salida multipaso. Las expresiones inferiores resumen las operaciones desarrolladas en las ecuaciones y en la implementación. El texto, cuando interviene, entra únicamente tras la selección por disponibilidad.\n\n'
    t+=literal('esquema_'+m['key']+'.py','Código del esquema')
    t+=literal(m['key']+'_red.py','Código de la arquitectura')
    t+='## Pipeline de entrenamiento y prueba\n\n'
    if m['package'] is None:
        t+='La configuración utilizada por esta familia es:\n\n```json\n'+json.dumps(reg[{'arx':'arx_lineal','mlp':'mlp','lstm':'lstm','cnn1d':'cnn1d'}[m['key']]]['configuracion'],ensure_ascii=False,indent=2)+'\n```\n\n'
        t+='El procedimiento separa ajuste, parada interna y evaluación externa, estima las transformaciones únicamente en ajuste y restaura el mínimo de parada. La función admite validación o prueba con autorización explícita y conserva las salidas en una carpeta nueva. Los bloques siguientes documentan el código; construir el libro no ejecuta entrenamientos.\n\n'
        t+=literal('base/src/demanda/reentrenamiento_v3.py','Escalado por pliegue y optimización')
        t+=oldpipe+'\n\n'
        t+=f"Para esta página, `familia` toma el valor `{'arx_lineal' if m['key']=='arx' else m['key']}`. Las definiciones completas de datos y métricas se muestran en el {{ref}}`preprocesamiento compartido <fundamentos-arquitecturas>`; los cuerpos de las redes y del bucle de ajuste están visibles en esta página.\n"
    elif m['package']=='numerico_repositorio':
        t+='La preparación conserva 30 variables históricas, 23 de calendario y 30 máscaras. Se usan las mismas ventanas para ajuste, parada y prueba, con imputaciones y escalas estimadas solo en ajuste. El motor recibe el dispositivo explícitamente, guarda el mínimo de parada interna y convierte la salida normalizada a kWh. El código de la arquitectura incluye el motor para PatchTST; N-HiTS comparte ese motor sin compartir su red.\n\n'
        t+=literal('numerico_repositorio/src/demanda/patchtst_numerico.py','Preparación, ajuste y predicción numérica')
        if m['key']=='nhits':
            t+=literal('numerico_repositorio/src/demanda/vendor/nhits/core.py','Bloque N-HiTS e interpolación')
        else:
            t+=literal('numerico_repositorio/src/demanda/vendor/patchtst/PatchTST_backbone.py','Núcleo PatchTST utilizado')
        script='entrenar_patchtst_numerico.py' if m['key']=='patchtst' else 'entrenar_nhits_numerico.py'
        t+=literal('numerico_repositorio/scripts/'+script,'Selección, piloto, ajuste final y prueba')
        t+='El orquestador preservado utiliza bloqueo `fcntl` y exige GPU en su entrada principal. El siguiente adaptador conserva red, configuración, preparación, parada y predicción, y permite escoger CPU o GPU sin ese controlador de Linux. Su inclusión documental no ejecuta un nuevo ajuste.\n\n'
        t+=literal('pipeline_ampliado_portable.py','Pipeline explícito compatible con Windows y Linux')
    else:
        pa=m['package'];mod='modelo6' if pa=='Modelo6_TaTS' else 'entrenamiento'
        t+='El pipeline fija el plan y las huellas de entrada, genera ventanas cronológicas, ajusta las transformaciones por pliegue y evalúa 20 candidatos en 2018. Confirma cinco finalistas en 2019–2021, selecciona por MASE medio y reajusta con cinco semillas finales usando la reserva interna anterior a 2022. La lectura de la prueba solo ocurre después de cerrar la selección. El libro muestra las implementaciones y no llama al entrenamiento.\n\n'
        t+=literal(pa+'/src/'+mod+'/datos.py','Datos, referencias y escaladores por pliegue')
        t+=literal(pa+'/src/'+mod+'/motor.py','Bucle de ajuste y predicción')
        t+=literal(pa+'/src/'+mod+'/protocolo.py','Búsqueda, confirmación y evaluación final')
        if m['key'] in {'tats','tgforecaster','fusion'}:t+=literal(pa+'/src/'+mod+'/texto.py','Disponibilidad, embeddings y preparación del texto')
        t+=literal(pa+'/ejecutar.py','Entrada Python y registro de la ejecución')
        t+='La configuración exacta del paquete se conserva a continuación. Identifica variantes, candidatos, semillas, parada y presupuesto; se mantiene separada de las métricas calculadas a partir de artefactos.\n\n'
        cfg='modelo6.json' if mod=='modelo6' else 'modelo.json'
        t+=f'```{{literalinclude}} ../_static/codigo_e2/{pa}/config/{cfg}\n:language: json\n```\n\n'
    t+='\n## Evaluación\n\n'
    if m['key']=='tgforecaster':t+='Las variantes N y N+O conservan cinco semillas finales cada una, con métricas y predicciones para 2022–2025. Su contraste documental y la inferencia conjunta se presentan en la {ref}`evaluación integrada <evaluacion-e2>`. Las descripciones estáticas están presentes en ambas variantes; la comparación aísla el procedimiento con y sin documentos oficiales variables en el tiempo, con hiperparámetros seleccionados por separado.\n'
    else:t+='Los resultados de esta familia se integran en la {ref}`evaluación estadística conjunta <evaluacion-e2>`, con métricas por semilla, horizonte y origen. La comparación distingue las entradas demanda/calendario de las entradas numéricas ampliadas y el aporte de texto oficial.\n'
    t+='\n## Fuentes\n\n'
    refs={'patchtst':'[Nie et al., PatchTST](https://arxiv.org/abs/2211.14730)',
          'nhits':'[Challu et al., N-HiTS](https://ojs.aaai.org/index.php/AAAI/article/view/25854)',
          'tft':'[Lim et al., TFT](https://arxiv.org/abs/1912.09363)',
          'tcn':'[Bai et al., convoluciones causales dilatadas](https://arxiv.org/abs/1803.01271)',
          'itransformer':'[Liu et al., iTransformer](https://arxiv.org/abs/2310.06625)',
          'tats':'[Li et al., TaTS, versión 3](https://arxiv.org/html/2502.08942v3)',
          'tgforecaster':'[Xu et al., TGForecaster, versión 2](https://arxiv.org/html/2405.13522v2)'}
    t+=refs.get(m['key'],'La formulación y las dimensiones se verifican en las clases del código mostrado.')+'. La descripción corresponde a la implementación del experimento; las adaptaciones se identifican en el texto.\n'
    (BOOK/(m['key']+'.md')).write_text(t,encoding='utf-8')

(O/'indice_documentacion.json').write_text(json.dumps([{k:v for k,v in m.items() if k not in ['text','cards','blocks']} for m in models],ensure_ascii=False,indent=2),encoding='utf-8')
shutil.copy2(__file__,O/'crear_arquitecturas_e2.py')
print('Páginas y esquemas generados:',len(models))

from reorganizar_capitulo10 import fundamentos,escribir_toc
fundamentos(R);escribir_toc(R)
from esquemas_presentacion import instalar as instalar_esquemas_presentacion
instalar_esquemas_presentacion(R)

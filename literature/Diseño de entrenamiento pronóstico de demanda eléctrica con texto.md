# Diseño de entrenamiento: pronóstico de demanda eléctrica con texto

Nov 9, 2026 · @Fabian

## Objetivo y alcance

Este diseño fija cómo se entrenan y comparan 7 arquitecturas para pronosticar la demanda real horaria del SIN: 5 solo numéricas y 2 que combinan un modelo numérico con texto. Se suman a los 3 modelos base del Entregable 1 (MLP, LSTM, CNN1D) para completar el Top 10 del Entregable 2.

- **Tarea:** salida directa de 24 pasos a partir de 168 h de historia, con origen diario a las 00:00 de Colombia.
- **Pregunta que responde:** si el texto oficial del sector mejora el pronóstico frente al mismo modelo sin texto, bajo igual presupuesto de ajuste.
- **Qué fija:** datos, tratamiento del texto, mecanismos de fusión, espacios de búsqueda, controles de fuga y evaluación.
- **Qué queda fuera:** la implementación del modelo original del Entregable 3 (aquí solo se proponen ideas) y la tabla de puntajes de la revisión sistemática, que debe justificar esta selección.

Todo lo que aquí se declara debe copiarse a `PREREGISTRO.md` antes de entrenar.

## Datos y partición

Se usan dos tablas ya versionadas (`corpus_oficial_20261109`): el panel numérico horario y el corpus de texto. La partición es la del preregistro vigente y no cambia.

| Elemento | Definición |
| --- | --- |
| Panel numérico | `panel_horario.parquet`: demanda real, calendario y festivos; clima e hidrología como covariables opcionales |
| Corpus de texto | `corpus_texto.parquet`: 7.588 documentos oficiales con cuerpo (XM, MinEnergía, IDEAM, CREG, UPME) y 561.072 titulares de prensa |
| Selección de hiperparámetros | Ventana expansiva con 4 pliegues anuales de validación: 2018, 2019, 2020 y 2021 |
| Ajuste final | Orígenes cuyo objetivo termina antes de 2022-01-01; último 10 % para parada temprana |
| Prueba | 2022-01-01 a 2025-12-31, congelada, una lectura por modelo |
| Prueba prospectiva (recomendada) | Desde 2026-01-01, aún sin descargar |

**Escenarios de disponibilidad del texto**

- **Declarado (principal):** usa `disponible_declarado`. Un documento con solo día entra a las 00:00 de Bogotá del día siguiente.
- **Estricto (sensibilidad):** usa `disponible_estricto`. Solo 10 documentos oficiales con cuerpo caen en 2008-2019, así que no sirve para entrenar y se reporta como análisis de sensibilidad.

**Limitaciones que condicionan el diseño**

- El texto oficial con cuerpo cubre el 37 % de los días de 2008-2019 y el 73 % de 2022-2025.
- La composición cambia: IDEAM pasa de cerca del 12 % de los documentos en 2008-2019 a cerca del 43 % en 2022-2025, y CREG de cerca del 12 % al 2 %.
- La fecha declarada no es verificable para la mayoría de los documentos, y su texto es la versión recuperada en 2026.

## Modelos

Se entrenan 7 arquitecturas en este orden de prioridad: primero las numéricas baratas y fuertes, y al final las de texto, que dependen de un buen modelo numérico de base. Las referencias están al final del documento.

| # | Modelo | Familia | Datos | Búsqueda de 1000 arquitecturas (§14) | Idea tomada de |
| --- | --- | --- | --- | --- | --- |
| 1 | PatchTST | Transformer por parches | Numérico | No, arquitectura publicada | Nie et al. (2023) |
| 2 | N-HiTS | MLP jerárquico con interpolación | Numérico | No | Challu et al. (2023) |
| 3 | TFT | Recurrente con atención y covariables | Numérico y calendario | No | Lim et al. (2021) |
| 4 | TCN | Convolucional dilatado | Numérico | Sí | Bai et al. (2018) |
| 5 | iTransformer | Transformer por variable | Numérico multivariado | No | Liu et al. (2024) |
| 6 | TaTS sobre el mejor entre PatchTST e iTransformer | Texto como variables auxiliares | Numérico y texto | No | Li et al. (2025) |
| 7 | TGForecaster | Atención cruzada entre texto y serie | Numérico y texto | No | Xu et al. (2024) |

- **Modelos base heredados:** MLP, LSTM y CNN1D del Entregable 1, más el ARX y los ingenuos diario y semanal.
- **Modelo base de TaTS:** se elige entre PatchTST e iTransformer por MASE de validación, antes de añadir el texto.
- **Preentrenamiento:** ningún modelo numérico usa pesos preentrenados. Los modelos 6 y 7 usan un codificador de texto preentrenado y congelado, que se declara según §11.1.

## Procesamiento del texto

El texto se convierte una sola vez en vectores fijos con un codificador congelado, y los modelos solo ven esos vectores. Así el tratamiento es idéntico para los modelos 6 y 7, cabe en una GPU de 12 GB y queda auditado con hash.

1. **Selección.** Se excluyen los registros en cuarentena y los que no tienen cuerpo. Las fuentes del experimento principal se fijan a priori: XM, MinEnergía, CREG, UPME e IDEAM con cuerpo. La prensa solo entra en una ablación, filtrada por una lista de términos declarada antes de mirar la prueba.
2. **Campos de entrada.** Solo `titulo_verificado` (o `titulo` si falta) y `texto`. Nunca entran `url`, identificadores, fechas, campos de disponibilidad ni `alertas`.
3. **Limpieza.** Normalización de espacios y eliminación de encabezados y pies repetidos. No se corrige el OCR a mano.
4. **Enmascarado.** Las fechas absolutas, los años y los números de resolución o circular se sustituyen por marcadores (`[FECHA]`, `[NUM]`). Evita que el modelo aprenda el calendario desde el texto.
5. **Fragmentación.** Cada documento se divide en fragmentos de hasta 512 tokens con solape de 64, y se conservan como máximo los primeros 8. El límite iguala el peso de un comunicado de 340 palabras y un informe de 5.400.
6. **Codificación.** Un codificador de frases multilingüe, congelado, produce un vector por fragmento. El vector del documento es el promedio de sus fragmentos, normalizado. Propuesta: multilingual-E5 base (Wang et al., 2024), con promedio de tokens como en Sentence-BERT (Reimers y Gurevych, 2019).
7. **Alineación por día.** Para cada día de la ventana de historia se toman los documentos cuya disponibilidad cae en ese día y es anterior al origen. Se usa `demanda.corpus.documentos_en_origen`.
8. **Agregación.** Un vector diario por grupo de fuentes: energía (XM, MinEnergía, CREG, UPME) y clima (IDEAM). Se calcula como promedio simple o ponderado por recencia; la opción es un hiperparámetro.
9. **Ausencia de texto.** Un día sin documentos recibe el vector de la frase fija «sin información disponible», como en TaTS (Li et al., 2025), más una bandera de ausencia y el conteo de documentos.
10. **Registro.** Se guardan los vectores con el hash del texto, la versión del codificador y la fecha de cálculo.

**Decisiones fijas, fuera del presupuesto de búsqueda:** codificador, enmascarado, tamaño de fragmento, grupos de fuentes y escenario de disponibilidad. **Dentro del presupuesto:** ventana de texto (24, 72 o 168 h), tipo de agregación y dimensión de proyección.

Los valores de fragmentación (512, 64, 8) y el codificador son propuestas de este diseño y no vienen de los artículos. TaTS usa GPT-2 en sus experimentos principales y TGForecaster usa modelos de frases en inglés; aquí se cambian por uno multilingüe porque el corpus está en español, y esa desviación debe declararse.

## Combinación de texto e información numérica

Se prueban dos mecanismos de fusión distintos sobre los mismos vectores de texto: uno temprano, que añade el texto como canales de entrada (modelo 6), y uno intermedio, que lo cruza con la serie mediante atención (modelo 7).

### Modelo 6: texto como variables auxiliares (TaTS)

La idea viene de Li et al. (2025): los textos emparejados con la serie se tratan como variables auxiliares y se concatenan a los canales numéricos, sin modificar el modelo base.

1. El vector diario de cada grupo de fuentes se repite en las 24 horas de su día, de modo que cada una de las 168 horas de historia tiene su vector de texto.
2. Un MLP de tres capas reduce cada vector a una dimensión pequeña (8, 16 o 32) y se entrena junto con el modelo base.
3. Los canales reducidos se concatenan a los numéricos y entran al modelo base sin cambios.
4. De la salida solo se conservan los canales numéricos; aquí, las 24 horas de demanda.

```latex
U = [\,X \;;\; Z\,] \in \mathbb{R}^{168 \times (N + d)}, \qquad Z_t = \mathrm{MLP}(e_t), \qquad \hat{y} = f_\theta(U)_{[1:N]}
```

X son los N canales numéricos, e\_t el vector de texto de la hora t y d la dimensión reducida.

### Modelo 7: atención cruzada guiada por texto (TGForecaster)

La idea viene de Xu et al. (2024): un codificador por parches tipo PatchTST procesa la serie y dos etapas de atención cruzada incorporan el texto.

1. **Relevancia de cada documento.** La descripción del canal («demanda real horaria del SIN de Colombia») actúa como consulta, y los vectores de los documentos de la ventana son las claves y los valores. El resultado es un resumen del texto ponderado por relevancia.
2. **Mezcla de modalidades.** El resumen de texto actúa como consulta sobre los vectores temporales del codificador de la serie, que son las claves y los valores.
3. **Salida.** Un decodificador por tokens proyecta cada token a su tramo del horizonte de 24 h.

En este modelo los documentos entran uno por uno, hasta un máximo M por ventana, y no como promedio diario.

### Desviaciones respecto a los artículos

- **Sin texto futuro.** En TGForecaster el texto se alinea con el horizonte de pronóstico. Aquí solo se usa texto disponible antes del origen, y las consultas del horizonte son la descripción del canal más una codificación de posición aprendida para las 24 horas.
- **Un solo canal objetivo.** La primera etapa de TGForecaster reparte noticias entre varios canales; aquí pondera documentos para un único canal.
- **Texto diario sobre serie horaria.** TaTS asume un texto por paso de tiempo; aquí el vector diario se repite en sus 24 horas.

### Controles obligatorios

| Control | Qué se entrena | Qué descarta |
| --- | --- | --- |
| Sin texto | El mismo modelo base, con igual presupuesto | Que la mejora venga de más ajuste y no del texto |
| Texto placebo | Los vectores reales, permutados en el tiempo dentro de cada partición | Que la mejora venga de tener más parámetros o canales |
| Solo ausencia y conteo | Bandera de ausencia y número de documentos, sin vectores | Que la señal sea el volumen de publicaciones y no el contenido |
| Por grupo de fuentes | Solo energía, solo clima, ambos | Que el efecto dependa del cambio de composición hacia IDEAM |
| Prensa filtrada (ablación) | Oficial más titulares filtrados, en canal separado | Que la prensa aporte algo sobre el texto oficial |

## Protocolo de entrenamiento e hiperparámetros

Los 7 modelos reciben el mismo presupuesto que los modelos base en la versión 2 del preregistro: 20 configuraciones por modelo y 3 semillas por configuración, seleccionadas por MASE medio de validación cruzada.

**Común a todos los modelos**

- **Entorno:** PyTorch en la GPU del autor; se registra dispositivo, versiones y commit por corrida.
- **Optimización:** AdamW y recorte de gradiente 1,0. La pérdida (MSE, MAE o Huber) y el scheduler son hiperparámetros, como en la versión 2 del preregistro.
- **Parada temprana:** sobre la pérdida de validación, con un máximo de 200 épocas y mejora mínima de 1e-6.
- **Búsqueda:** aleatoria o bayesiana, solo con validación. Se guardan todas las configuraciones evaluadas.
- **Espacio común:** el mismo de los modelos base: tasa de aprendizaje de 1e-4 a 1e-2 y weight decay de 1e-6 a 1e-2 (ambos en escala logarítmica), lote de 32, 64, 128 o 256, dropout de 0,0 a 0,5, scheduler, pérdida y paciencia de 8, 15 o 25.
- **Ajuste final:** igual para todos, con las semillas de prueba y el número de ejecuciones que resulte del cálculo de potencia.

**Espacio propio de cada modelo**

| Modelo | Hiperparámetros a explorar |
| --- | --- |
| PatchTST | Parche 8, 12, 16 o 24; paso 4, 8 o igual al parche; dimensión 64, 128 o 256; cabezas 4, 8 o 16; capas 2 a 4 |
| N-HiTS | Pilas 2 a 4; agrupación (2,2,1), (4,2,1) o (8,4,1); ancho 128, 256 o 512; bloques por pila 1 a 3 |
| TFT | Tamaño oculto 32, 64 o 128; cabezas 1, 2 o 4; capas LSTM 1 o 2; conjunto de covariables |
| TCN | Núcleo 5 o 7; dilataciones con campo receptivo de al menos 168 h, como 1-4-16 con núcleo 5 o 7; normalización por lote o por pesos |
| iTransformer | Dimensión 64, 128 o 256; cabezas 4 u 8; capas 2 a 4; variables de entrada |
| TaTS | Los del modelo base; proyección del texto 8, 16 o 32; ventana de texto 24, 72 o 168 h; agregación simple o por recencia |
| TGForecaster | Dimensión 64, 128 o 256; cabezas 4 u 8; capas 1 a 3; ventana de texto 24, 72 o 168 h; máximo M de documentos por ventana |

**Búsqueda de arquitectura (§14), solo para TCN**

- TCN no tiene una arquitectura fijada en su artículo, así que recorre el espacio obligatorio: 3 bloques, filtros por bloque en {16, 32, 64, 128, 256} y dropout de 0,2 activable por bloque, es decir 125 × 8 = 1000 arquitecturas.
- Antes de la búsqueda se fijan los demás hiperparámetros con la arquitectura de referencia (64, 64, 64) y dropout (0, 1, 0).
- La fase 1 evalúa las 1000 sin poda; la fase 2 y la selección del ganador siguen el §14, con el criterio declarado a priori.
- Los modelos 1, 2, 3, 5, 6 y 7 usan la arquitectura de su artículo y no entran a esta búsqueda.

**Orden de ejecución**

1. N-HiTS y PatchTST, que fijan la referencia numérica.
2. iTransformer y TFT.
3. TCN, con su búsqueda de 1000 arquitecturas.
4. Cálculo y congelación de los vectores de texto.
5. TaTS y TGForecaster, cada uno con sus controles.

Las configuraciones publicadas de cada arquitectura y los intervalos de esta tabla deben confirmarse contra cada artículo al llenar la hoja de extracción.

## Definición matemática de cada opción

Todos los modelos son funciones que llevan una ventana de 168 horas y el calendario futuro a un vector de 24 pronósticos, en la misma escala normalizada del repositorio. Las fórmulas de los modelos publicados están escritas de memoria a partir de sus artículos y deben contrastarse con ellos antes de programar.

### Notación común

```latex
X \in \mathbb{R}^{L \times C},\; L = 168 \qquad F \in \mathbb{R}^{H \times C_f},\; H = 24 \qquad y,\ \hat{y} \in \mathbb{R}^{H} \qquad \hat{y} = f_\theta(X, F)
```

X es la historia (demanda y covariables pasadas), F el calendario conocido a futuro, y B el tamaño del lote. La atención que usan varios modelos es:

```latex
\mathrm{Attn}(Q, K, V) = \mathrm{softmax}\!\left(\frac{Q K^{\top}}{\sqrt{d}}\right) V
```

### Modelo 1: PatchTST

Cada canal se procesa por separado. La serie se corta en N parches de longitud P con paso S, cada parche se proyecta a dimensión D y pasa por un codificador Transformer; una capa lineal sobre los tokens aplanados da el pronóstico.

```latex
N = \left\lfloor \frac{L - P}{S} \right\rfloor + 2, \qquad z_i^{(0)} = W_p\, x_{[iS,\, iS+P)} + e_i^{\mathrm{pos}}, \qquad \hat{y} = W_o\, \mathrm{vec}\big(Z^{(n)}\big) + b_o
```

El +2 supone que se rellena el final de la serie con S copias del último valor. Con P = 16 y S = 8 resultan N = 21 parches. Cada capa aplica atención con varias cabezas y una red feed-forward, con conexiones residuales y normalización.

### Modelo 2: N-HiTS

Una pila de bloques procesa la entrada a varias escalas. El bloque l submuestrea su entrada con un máximo por ventanas de tamaño k\_l, un MLP produce coeficientes de pronóstico y de reconstrucción, y el pronóstico se obtiene interpolando pocos coeficientes al horizonte completo.

```latex
x_l^{\downarrow} = \mathrm{MaxPool}_{k_l}(x_l), \qquad (\theta_l^{f}, \theta_l^{b}) = \mathrm{MLP}_l\big(x_l^{\downarrow}\big), \qquad \hat{y}_l = \mathrm{Interp}\big(\theta_l^{f}\big) \in \mathbb{R}^{H}
```

```latex
x_{l+1} = x_l - \mathrm{Interp}\big(\theta_l^{b}\big), \qquad \hat{y} = \sum_{l} \hat{y}_l
```

El número de coeficientes de pronóstico del bloque l es una fracción r\_l de H, de modo que los bloques con mayor k\_l modelan las frecuencias bajas.

### Modelo 3: TFT

Su pieza básica es la red residual con compuerta (GRN), que se usa para seleccionar variables y para enriquecer la representación. Un LSTM codifica la historia y decodifica sobre el calendario futuro, y una atención interpretable conecta ambos tramos.

```latex
\mathrm{GRN}(a, c) = \mathrm{LayerNorm}\big(a + \mathrm{GLU}(\eta_1)\big), \quad \eta_1 = W_1 \eta_2 + b_1, \quad \eta_2 = \mathrm{ELU}(W_2 a + W_3 c + b_2)
```

```latex
\mathrm{GLU}(\gamma) = \sigma(W_4 \gamma + b_4) \odot (W_5 \gamma + b_5), \qquad \tilde{\xi}_t = \sum_{j} v_t^{(j)}\, \mathrm{GRN}_j\big(\xi_t^{(j)}\big), \quad v_t = \mathrm{softmax}\big(\mathrm{GRN}_v(\Xi_t)\big)
```

Los pesos v\_t son la selección de variables en cada hora. La salida original es por cuantiles; aquí se usa la mediana o una salida puntual, con la pérdida común.

### Modelo 4: TCN

Tres bloques residuales, cada uno con dos convoluciones causales dilatadas de núcleo k y dilatación d\_j. Los filtros por bloque (w\_1, w\_2, w\_3) y el patrón de dropout salen de la búsqueda de 1000 arquitecturas.

```latex
(x *_{d} f)(s) = \sum_{i=0}^{k-1} f(i)\, x_{s - d\, i}, \qquad o = \mathrm{ReLU}\big(x + \mathcal{F}(x)\big), \qquad R = 1 + 2\,(k - 1) \sum_{j=1}^{3} d_j
```

R es el campo receptivo y debe cubrir las 168 horas. Con dilataciones (1, 4, 16) se cumple para k = 5 (R = 169) y k = 7 (R = 253), pero no para k = 3 (R = 85). Con (1, 2, 4) no se cumple para ningún núcleo de la lista. El pronóstico sale de una capa lineal sobre el último paso temporal, concatenado con el calendario futuro.

### Modelo 5: iTransformer

Invierte el papel de los ejes: cada variable completa es un token. La atención opera entre variables y la red feed-forward sobre cada una.

```latex
h_c^{(0)} = W_e\, X_{:, c} + b_e \in \mathbb{R}^{D}, \quad c = 1, \dots, C \qquad H^{(n)} = \mathrm{Encoder}\big(H^{(0)}\big) \qquad \hat{y} = W_o\, h_{\mathrm{dem}}^{(n)} + b_o
```

Solo se conserva la proyección del token de demanda. Las covariables de calendario futuro pueden entrar como tokens adicionales.

### Modelo 6: TaTS

Definido en la sección de combinación. Con G grupos de fuentes y dimensión reducida d, el modelo base recibe C + G·d canales:

```latex
Z_t = \big[\mathrm{MLP}_1(e_t^{(1)});\ \dots;\ \mathrm{MLP}_G(e_t^{(G)})\big] \in \mathbb{R}^{G d}, \qquad U = [X;\ Z] \in \mathbb{R}^{L \times (C + G d)}, \qquad \hat{y} = f_\theta(U, F)
```

### Modelo 7: TGForecaster

Con E la matriz de vectores de los M documentos de la ventana, q la descripción del canal, p\_h la posición aprendida de la hora h y T los tokens temporales del codificador por parches:

```latex
s = \mathrm{Attn}\big(q,\ E,\ E\big) \in \mathbb{R}^{D}, \qquad u_h = \mathrm{Attn}\big(s + p_h,\ T,\ T\big), \qquad \hat{y}_h = w^{\top} u_h + b, \quad h = 1, \dots, H
```

Los documentos de relleno se excluyen de la primera atención con una máscara. Si la ventana no tiene documentos, s se sustituye por un vector aprendido de ausencia.

### Candidatos a modelo original

Las ideas A, B, C, E y F se describen en la sección de ideas; aquí queda su definición completa. Todas las cotas están en unidades normalizadas y se convierten a kWh con la escala de la ventana.

**Idea A: fusión residual acotada**

Sea f\_θ el modelo numérico, z el resumen de texto, c = enc(X) un vector de contexto y m la bandera de presencia de texto.

```latex
\hat{y}_{\mathrm{num}} = f_\theta(X, F), \qquad r = h_\phi\big([z;\, c]\big) \in \mathbb{R}^{H}, \qquad g = m \cdot \sigma\!\big(w_g^{\top} c + b_g\big), \qquad \hat{y} = \hat{y}_{\mathrm{num}} + g\, \rho\, \tanh(r)
```

Proposición 1. Para todo X, F y z, y toda hora h:

```latex
\big|\hat{y}_h - \hat{y}_{\mathrm{num},h}\big| = g\,\rho\,|\tanh(r_h)| \le \rho \quad\Longrightarrow\quad \Big|\tfrac{1}{H}\textstyle\sum_h |y_h - \hat{y}_h| - \tfrac{1}{H}\sum_h |y_h - \hat{y}_{\mathrm{num},h}|\Big| \le \rho
```

La primera desigualdad sale de g ∈ \[0, 1\] y |tanh| < 1; la segunda, de la desigualdad triangular inversa aplicada hora a hora. Si m = 0, entonces g = 0 y la salida coincide con la numérica. Supuesto: f\_θ es el mismo en ambos términos, lo que exige congelarlo al entrenar la rama de texto o comparar contra el f\_θ final.

- **Entrenamiento en dos etapas:** primero f\_θ con la pérdida común (es el control sin texto); después φ, w\_g y b\_g con f\_θ congelado.
- **Inicialización:** b\_g negativo (por ejemplo −3) para que la corrección arranque cerca de cero.
- **Hiperparámetros propios:** ρ en {0,05; 0,1; 0,2; 0,4}, ancho de h\_φ y fuente de z (promedio diario o idea B).

**Idea B: núcleo de respuesta al impulso**

Para los documentos i con vector e\_i y antigüedad Δ\_i = t\_0 − t\_i > 0 en horas:

```latex
z = \sum_{i=1}^{M} \mu_i\, \kappa(\Delta_i)\, W e_i, \qquad \kappa(\Delta) = \sum_{k=1}^{K} a_k\, e^{-\Delta / \tau_k}, \qquad a_k = \mathrm{softplus}(\alpha_k), \quad \tau_k = \mathrm{softplus}(\beta_k) + \tau_{\min}
```

μ\_i es la máscara (1 para documentos reales con Δ\_i > 0, 0 para relleno o documentos no disponibles).

Proposición 2. z no cambia al permutar los documentos, no depende de ningún documento con μ\_i = 0, y está acotado:

```latex
\|z\| \le \Big(\sum_k a_k\Big)\, \|W\|_2 \sum_{i} \mu_i \|e_i\| \le M \Big(\sum_k a_k\Big) \|W\|_2
```

La última desigualdad usa que los vectores están normalizados a norma 1. K = 3 con τ iniciales de 12, 48 y 168 horas es un punto de partida razonable.

**Idea C: pérdida con texto placebo**

Sea z̃ el texto de otro origen de la misma partición, elegido por permutación dentro del lote, y ℓ la pérdida común.

```latex
\mathcal{L} = \ell\big(\hat{y}(X, z),\, y\big) + \lambda\, \big\|\hat{y}(X, \tilde{z}) - \hat{y}_{\mathrm{num}}(X)\big\|^2
```

Proposición 3. Si z es independiente de (X, y) y ℓ es el error cuadrático, el minimizador poblacional es E\[y | X\], que no depende del texto. Bajo independencia, (X, z) y (X, z̃) tienen la misma distribución, así que ambos términos se minimizan a la vez con una función que ignora z. El supuesto es fuerte y debe declararse: en los datos reales el texto no es independiente de la historia.

**Idea E: fusión residual acotada de covariables exógenas**

Igual que la idea A, con X dividido en demanda y calendario (X\_dem) y covariables exógenas (X\_exo), y sin bandera de ausencia:

```latex
\hat{y} = f_\theta(X_{\mathrm{dem}}, F) + \sigma\!\big(w_g^{\top} c + b_g\big)\, \rho\, \tanh\!\big(h_\phi([\mathrm{enc}_{\mathrm{exo}}(X_{\mathrm{exo}});\, c])\big)
```

La proposición 1 se aplica sin cambios, con f\_θ(X\_dem, F) en el papel del modelo numérico.

**Idea F: mezcla convexa de perfiles diarios**

Trabaja con la demanda sin centrar, x ∈ R^L positiva.

```latex
\ell(x) = \sum_{j=1}^{L} \omega_j x_j, \quad \omega = \mathrm{softmax}(\nu) \qquad \pi = \mathrm{softmax}\big(\mathrm{MLP}([x / \ell(x);\ F])\big) \in \Delta^{K-1} \qquad P_k = H \cdot \mathrm{softmax}(u_k) \in \mathbb{R}^{H}_{>0}
```

```latex
\hat{y} = \ell(x) \sum_{k=1}^{K} \pi_k P_k
```

Proposición 4. Para toda constante a > 0, ŷ(a x, F) = a ŷ(x, F). El nivel es lineal en x, así que ℓ(a x) = a ℓ(x); el cociente x / ℓ(x) no cambia, luego π tampoco. Además cada perfil promedia 1, así que la media diaria del pronóstico es exactamente ℓ(x). La pérdida debe calcularse sobre el error relativo al nivel para que el entrenamiento no dependa de la escala.

## Especificación para el script de entrenamiento en PyTorch

Los modelos nuevos se añaden al entrenador que ya existe (`scripts/entrenar_modelos_base.py` y `src/demanda/modelos_base.py`), sin escribir otro bucle de entrenamiento. Así heredan las etapas, la búsqueda, la parada temprana, el determinismo y el formato de resultados de los modelos base.

### Contrato que debe cumplir cada red

| Elemento | Definición en el repositorio |
| --- | --- |
| Entrada `xp` | Tensor (B, 168, Fx + Kc): entradas pasadas con la demanda en la columna 0, concatenadas con el calendario pasado |
| Entrada `cf` | Tensor (B, 24, Kc): calendario futuro |
| Salida | Tensor (B, 24) en la escala de `Datos.objetivo_norm` (normalización `nivel` por defecto) |
| `forward(xp, cf)` | Pronóstico. Los modelos con texto reciben un tercer argumento `txt` |
| `preparar(xp, cf)` | Estadísticos calculados solo con entrenamiento; puede no hacer nada |
| `penalizacion()` | Término de regularización que se suma a la pérdida; 0,0 si no aplica |
| Registro | Nombre en `CLASES` y `MODELOS_PYTORCH`, espacio en `ESPACIOS` y rama en `construir_red` |

### Qué hace ya el entrenador común

1. Fija las semillas y el dispositivo, y construye la red con `construir_red`.
2. Normaliza el objetivo y llama a `preparar` con datos de entrenamiento.
3. Entrena con AdamW y recorte de gradiente 1,0. La pérdida (MSE, MAE o Huber), el scheduler (ninguno, coseno o meseta), la tasa de aprendizaje y la paciencia son hiperparámetros.
4. Evalúa la pérdida de validación en cada época, guarda el mejor estado y se detiene cuando no mejora en al menos 1e-6 durante `paciencia` épocas, con un máximo de 200.
5. Restaura el mejor estado, invierte la normalización y calcula el MASE con la escala del ingenuo de 168 h.

Las etapas `buscar`, `piloto` y `final` y los archivos de salida (`busqueda.csv`, `mejor_config.json`, `final_runs.csv`, `preds`, `hist`, `entorno_*.json`) no cambian. Se recomienda una carpeta de salida nueva, por ejemplo `results/top10`.

### Cambios necesarios

- [ ] Crear `src/demanda/modelos_top10.py` con las 7 redes y registrarlas.
- [ ] Añadir a `ESPACIOS` el espacio propio de cada modelo, junto con el bloque común `_OPT`, `weight_decay`, `batch_size` y `dropout`.
- [ ] Escribir `scripts/construir_vectores_texto.py`, que aplica el procesamiento del texto y guarda los vectores.
- [ ] Extender `Conjunto` con un campo opcional `txt` y hacer que `ModeloTorch._tensores` lo pase a las redes que declaren `usa_texto = True`.
- [ ] Añadir la etapa de búsqueda de 1000 arquitecturas para TCN, con los demás hiperparámetros fijos.
- [ ] Añadir los controles de texto como opción `--control` con valores `sin_texto`, `placebo` y `solo_conteo`.
- [ ] Actualizar `PREREGISTRO.md` antes de la primera corrida.

### Tensores de texto

Los vectores se calculan una vez y se guardan por día y por documento; las ventanas se arman al indexar, no se duplican en disco.

| Tensor | Forma | Uso |
| --- | --- | --- |
| `e_dia` | (B, 7, G, d\_e) | Vector diario por grupo de fuentes, para TaTS. Se repite 24 veces dentro de la red |
| `m_dia` | (B, 7, G) | 1 si ese día y grupo tiene documentos |
| `n_dia` | (B, 7, G) | Número de documentos, en logaritmo |
| `e_doc` | (B, M, d\_e) | Vectores de hasta M documentos de la ventana, para TGForecaster y la idea B |
| `delta` | (B, M) | Horas entre la disponibilidad del documento y el origen; siempre positivas |
| `mu` | (B, M) | 1 para documentos reales, 0 para relleno |

G = 2 grupos (energía y clima) y d\_e es la dimensión del codificador. Guardar los vectores en precisión media reduce el tamaño a la mitad sin efecto apreciable.

### Esqueletos de los módulos propios

Núcleo de respuesta al impulso (idea B):

```python
import torch
from torch import nn
from torch.nn import functional as Fn


class NucleoImpulso(nn.Module):
    """z = sum_i mu_i * kappa(delta_i) * W e_i, con kappa = sum_k a_k exp(-delta / tau_k)."""

    def __init__(self, d_e, d_z, tau_ini=(12.0, 48.0, 168.0), tau_min=1.0):
        super().__init__()
        self.W = nn.Linear(d_e, d_z, bias=False)
        self.alpha = nn.Parameter(torch.zeros(len(tau_ini)))
        t = torch.tensor(tau_ini) - tau_min
        self.beta = nn.Parameter(t + torch.log(-torch.expm1(-t)))  # inversa estable de softplus
        self.tau_min = tau_min

    def forward(self, e, delta, mu):  # e (B, M, d_e); delta (B, M) en horas; mu (B, M) con 0 o 1
        a = Fn.softplus(self.alpha)  # (K,), pesos no negativos
        tau = Fn.softplus(self.beta) + self.tau_min  # (K,), escalas de decaimiento
        mu = mu * (delta > 0)  # un documento no disponible antes del origen nunca entra
        kappa = (a * torch.exp(-delta.clamp(min=0).unsqueeze(-1) / tau)).sum(-1)  # (B, M)
        return ((mu * kappa).unsqueeze(-1) * self.W(e)).sum(1)  # (B, d_z)
```

Fusión residual acotada (ideas A y E):

```python
class FusionAcotada(nn.Module):
    """y = y_num + g * rho * tanh(h([z; c])), con g en [0, 1]; |y - y_num| <= rho."""

    def __init__(self, base, f_in, d_z, H=24, rho=0.1, d_c=64, ancho=64, sesgo_g=-3.0):
        super().__init__()
        self.base = base  # modelo numerico f_theta ya entrenado
        for p in self.base.parameters():
            p.requires_grad_(False)  # se congela: la cota es frente a este mismo modelo
        self.rho = rho
        self.enc = nn.Sequential(nn.Flatten(1), nn.Linear(24 * f_in, d_c), nn.GELU())
        self.h = nn.Sequential(nn.Linear(d_z + d_c, ancho), nn.GELU(), nn.Linear(ancho, H))
        self.g = nn.Linear(d_c, 1)
        nn.init.zeros_(self.g.weight)
        nn.init.constant_(self.g.bias, sesgo_g)  # la correccion arranca cerca de cero

    def forward(self, xp, cf, z, m):  # z (B, d_z); m (B,) con 0 o 1
        with torch.no_grad():
            y_num = self.base(xp, cf)
        c = self.enc(xp[:, -24:])  # contexto: ultimas 24 horas
        g = m.unsqueeze(-1) * torch.sigmoid(self.g(c))  # (B, 1)
        return y_num + g * self.rho * torch.tanh(self.h(torch.cat([z, c], -1)))
```

Para la idea E, `z` es la salida de un codificador de las covariables exógenas y `m` es un vector de unos. Estos esqueletos son un punto de partida: el Entregable 3 exige que el núcleo se escriba desde la formulación y se pueda explicar y modificar en vivo, y que el uso de herramientas de IA se declare.

### Detalles que evitan errores

- **Determinismo en GPU.** Con el modo estricto algunas operaciones fallan o no son deterministas. En N-HiTS conviene escribir la interpolación como el producto por una matriz fija en vez de usar `interpolate`.
- **Base congelada.** Al entrenar la fusión, el modelo base debe estar en modo evaluación para que su dropout y su normalización no cambien la referencia.
- **Unidades de ρ.** Con la normalización `nivel`, ρ se expresa como fracción del nivel de la ventana. Hay que confirmarlo leyendo `Datos.objetivo_norm`.
- **Canales independientes.** En PatchTST el calendario futuro no entra por los parches; se concatena antes de la capa de salida.
- **Memoria.** TGForecaster con M documentos y vectores grandes es el modelo más pesado. Proyectar los vectores a la dimensión del modelo antes de la atención lo mantiene dentro de 12 GB.

### Pruebas unitarias mínimas

| Prueba | Qué comprueba |
| --- | --- |
| Formas | Cada red devuelve (B, 24) y rechaza entradas con forma incorrecta |
| Sobreajuste | Cada red lleva la pérdida cerca de cero en un lote sintético pequeño |
| Gradientes | Todo parámetro entrenable recibe gradiente finito |
| Reproducibilidad | Misma semilla, mismas salidas |
| Campo receptivo | En TCN, R es al menos 168 para toda configuración del espacio |
| Proposición 1 | Con texto aleatorio y adversario, la desviación frente al modelo base nunca supera ρ; con m = 0 es exactamente 0 |
| Proposición 2 | Permutar documentos o cambiar los de máscara 0 no altera la salida |
| Proposición 4 | Multiplicar la historia por una constante multiplica el pronóstico por la misma constante |
| Sin fuga | En todo lote, la disponibilidad de cada documento usado es anterior al origen |
| Placebo | La permutación del texto no mezcla particiones |

## Control de fuga de información

Ningún dato posterior al origen del pronóstico puede entrar al modelo, ni por los valores, ni por el texto, ni por el ajuste. Las reglas se aplican igual en todas las particiones.

| Etapa | Regla |
| --- | --- |
| Disponibilidad del texto | Solo documentos con disponibilidad estrictamente anterior al origen. La regla actual admite la igualdad (`disponibilidad <= origen`); se propone cambiarla a `<` |
| Fecha con solo día | El documento entra a las 00:00 de Bogotá del día siguiente; con solo año, al año siguiente |
| Contenido del texto | Fechas, años y números de documento enmascarados. Sin `url`, identificadores ni campos de disponibilidad |
| Versión del texto | El texto descargado en 2026 pudo editarse después de su fecha declarada. Se declara como limitación y se contrasta con el escenario estricto |
| Codificador de texto | Congelado y sin ajuste sobre el corpus. Pudo ver textos de 2022-2025 en su preentrenamiento; por eso se recomienda la prueba prospectiva desde 2026 |
| Normalización numérica | Estadísticos calculados solo con el entrenamiento de cada pliegue |
| Covariables | Solo calendario y festivos como conocidas a futuro. El clima es reanálisis y no pronóstico, así que no entra como dato futuro |
| Selección | Hiperparámetros, arquitectura, lista de términos de prensa y grupos de fuentes se fijan con validación. La prueba se lee una vez por modelo |
| Placebo | La permutación temporal del texto se hace dentro de cada partición, nunca entre particiones |

**Auditoría antes de entrenar**

- [ ] Revisar a mano una muestra por fuente y año, y registrar el porcentaje de fechas erróneas.
- [ ] Revisar una muestra de los cerca de 1.400 documentos oficiales que mencionan un año posterior al de su disponibilidad.
- [ ] Comprobar que ningún vector de texto usado en un origen tiene disponibilidad igual o posterior a ese origen.
- [ ] Copiar estas reglas a `PREREGISTRO.md` con fecha anterior al primer entrenamiento con texto.

## Evaluación

La métrica primaria sigue siendo el MASE medio sobre las 24 horas, con la escala del ingenuo de 168 h en 2000-2017, y la diferencia relativa de interés es del 3 %. Los modelos se comparan solo dentro de una misma partición.

- **Métricas secundarias:** RMSE, MAE, sMAPE, MAPE y MASE en las horas 1, 12 y 24.
- **Costo:** parámetros, tiempo de entrenamiento y de ajuste, latencia de inferencia y memoria pico por modelo.
- **Reporte:** media, desviación estándar e intervalo del 95 % por modelo, con los resultados por semilla y por pliegue.

**Comparaciones declaradas**

| Comparación | Prueba | Propósito |
| --- | --- | --- |
| Los 10 modelos entre sí | Friedman con Nemenyi, y pares con corrección de Holm | Ordenar el Top 10 con control del error por familia en 0,05 |
| Modelo con texto frente a su control sin texto | Diebold-Mariano con corrección HLN sobre los errores por origen, y prueba pareada por semilla | Decidir si el texto mejora el pronóstico |
| Modelo con texto frente a texto placebo | Las mismas | Descartar que la mejora venga de la capacidad añadida |
| Equivalencia | TOST con margen del 3 % | Afirmar que el texto no empeora, cuando no mejora |

**Desgloses obligatorios para los modelos con texto**

- Días con texto frente a días sin texto.
- Por grupo de fuentes (energía, clima).
- Escenario declarado frente a estricto.
- Por año de la prueba, porque la composición de fuentes cambia en 2024 y 2025.

Las pruebas estadísticas entre modelos aún no están en el preregistro y deben añadirse antes de aplicarlas. Para TCN, el resultado del ganador de la búsqueda de arquitectura se corrige por el sesgo de selección, como pide el §14.

## Validación estadística exigida por los entregables

La validación estadística se decide antes de entrenar, porque fija cuántas semillas correr, qué debe guardar cada corrida y qué comparaciones se podrán afirmar. Si el script no guarda los errores por semilla, por origen y por hora, varias pruebas obligatorias no se pueden calcular después.

### Qué exige cada entregable

| Aspecto | Entregable 1 | Entregable 2 | Entregable 3 |
| --- | --- | --- | --- |
| Ejecuciones por modelo | Piloto de 5 semillas y tamaño por potencia | Mínimo 5; el número final sale del análisis de potencia | Mínimo 10, o más si la potencia lo exige |
| Potencia | 0,80 para detectar el margen del 3 % | Al menos 0,80, con nivel ajustado por comparaciones múltiples | Al menos 0,90 por contraste, y potencia conjunta reportada |
| Pruebas por contraste | t pareada, Wilcoxon y bootstrap | Al menos tres de la Tabla C: una paramétrica o de remuestreo, una no paramétrica o bayesiana y una global | Al menos tres, y deben rechazar todas; decide el mayor p ajustado |
| Comparaciones múltiples | Holm | Nivel 0,05 con control por familia (Holm) o de falsos descubrimientos por horizontes | Holm sobre toda la familia: Top 10 más baselines, cerca de 12 contrastes |
| Ajuste y sobreajuste | Normalidad de residuos | Tablas A y B completas, con margen de sobreajuste declarado | Las mismas tablas aplicadas al modelo original |
| Tamaño del efecto | d de Cohen pareada | d de Cohen o delta de Cliff, con intervalo | Delta de Cliff de al menos 0,147 o d de al menos 0,2, e intervalo que excluya el cero |
| Equivalencia | TOST o ROPE con el margen | TOST o ROPE; no significativo no es equivalente | TOST o ROPE para afirmar que no es peor |
| Evidencia adicional | No aplica | Recomendada una réplica externa | Tres conjuntos de datos adicionales, elegidos antes de ver resultados |
| Registro previo | Métrica, margen, unidad y partición | Plantilla de preregistro completa (§9.1) | `estandar.md`, `protocolo.md` con tag de Git y bitácora de diseño |

### Qué debe guardar el script para que las pruebas sean posibles

| Dato por corrida | Para qué se usa |
| --- | --- |
| Pronóstico y valor real por origen y por hora, en validación y en prueba | Diebold-Mariano, bootstrap por bloques, Mincer-Zarnowitz, error por horizonte |
| Métrica primaria en entrenamiento y en validación, por pliegue | Brecha de sobreajuste con margen (Tabla B) |
| Métrica primaria en prueba por semilla | Pruebas pareadas entre semillas, Friedman, tamaño del efecto |
| Semilla, configuración, commit, dispositivo y tiempos | Pareado estricto, reproducibilidad y frontera de costo |
| Residuos de validación por hora | Intervalos de predicción y cobertura empírica |
| Toda corrida descartada, con su motivo | Reglas de exclusión declaradas a priori |

- **Pareado estricto.** Todos los modelos usan la misma lista de semillas, los mismos orígenes y el mismo preprocesamiento. Sin eso, las pruebas pareadas no son válidas.
- **Prueba sellada.** La prueba se lee una vez por modelo, con hiperparámetros y punto de parada fijados en validación.
- **Sin promedios solos.** Los entregables no aceptan comparaciones basadas únicamente en medias; cada afirmación necesita su prueba, su p exacto, su p ajustado y su intervalo.

### Batería de pruebas para la ruta de series de tiempo

El proyecto sigue la ruta temporal del Entregable 2, que pide tres grupos de pruebas: adecuación del ajuste, ausencia de sobreajuste y diferencias entre modelos.

**Tabla A: el modelo está bien ajustado**

| Prueba | Qué contrasta | Cómo aplicarla aquí |
| --- | --- | --- |
| Diebold-Mariano frente al ingenuo | Que el modelo supera al ingenuo semanal | Sobre la pérdida por origen; mejora mayor que el margen y p ajustado menor que 0,05 |
| Mincer-Zarnowitz | Pronóstico insesgado y eficiente: intercepto 0 y pendiente 1 | Regresión del valor real sobre el pronóstico, con errores estándar HAC |
| Ljung-Box, ARCH-LM y BDS | Residuos sin autocorrelación, sin efectos ARCH e independientes | Sobre los residuos de la primera hora del horizonte |
| Jarque-Bera | Normalidad de los residuos | Si se rechaza, usar intervalos por bootstrap y decirlo |
| PIT y cobertura | Intervalos con la cobertura nominal | Solo si se reportan intervalos; cobertura por hora |

**Tabla B: el modelo no está sobreajustado**

| Prueba | Qué contrasta | Cómo aplicarla aquí |
| --- | --- | --- |
| Brecha entre entrenamiento y validación con margen | Que la brecha es menor que un margen fijado a priori | Wilcoxon unilateral sobre la brecha por corrida; con 5 corridas el p mínimo es 0,031, se recomiendan 6 o más |
| Razón de error fuera y dentro de muestra | Degradación al salir de la muestra | Por hora del horizonte |
| Validación frente a prueba | Que el desempeño en prueba es coherente con el de validación | Intervalo por bootstrap de la diferencia. La escala del MASE cambia entre particiones, así que conviene compararlo como razón frente al ingenuo |
| Estabilidad de Giacomini-Rossi | Que el error no tiene quiebres en el tiempo | Sobre los orígenes de la prueba; importante por 2020 y por el cambio de fuentes de texto |
| Sensibilidad a la semilla | Que la varianza entre semillas es tolerable | Coeficiente de variación y Kruskal-Wallis entre semillas |

**Tabla C: las diferencias entre modelos son reales**

Cada contraste confirmatorio necesita al menos tres pruebas, una de cada tipo.

| Tipo exigido | Prueba propuesta | Unidad de análisis |
| --- | --- | --- |
| Paramétrica o de remuestreo pareado | t pareada sobre semillas, si pasa Shapiro-Wilk; si no, bootstrap pareado | Semilla |
| No paramétrica o bayesiana | Wilcoxon de rangos con signo, o prueba bayesiana con ROPE de ±3 % | Semilla |
| Global o con control múltiple | Friedman con Nemenyi o con pares de Holm; Model Confidence Set al 0,10 | Semilla u origen |
| Propia de series de tiempo | Diebold-Mariano con corrección HLN y varianza HAC, por hora | Origen de pronóstico |
| Modelos anidados | Clark-West o Giacomini-White | Origen de pronóstico |

- **Modelos anidados.** El modelo con texto contiene a su control sin texto, y la fusión acotada con compuerta en cero es el modelo numérico. Para esos pares Diebold-Mariano no es la prueba adecuada; corresponde Clark-West.
- **Regla de decisión.** Un contraste se declara significativo solo si rechazan las tres pruebas y la diferencia supera el margen del 3 %. Se reporta siempre la prueba que menos rechaza.
- **Análisis de sensibilidad.** La conclusión principal se repite con una prueba alternativa, con una métrica secundaria preregistrada y con otro subconjunto de semillas. Si la decisión cambia, se reporta como no robusta.
- **Por hora del horizonte.** Son 24 pruebas por contraste: se controlan con Benjamini-Hochberg y se declaran como familia aparte.

### Potencia y número de semillas

El número de semillas es la decisión que más condiciona el entrenamiento: con las 5 del piloto solo se detectan diferencias grandes, y el Entregable 3 no acepta menos de 10.

```latex
R \approx \frac{\left(z_{1-\alpha'/2} + z_{1-\beta}\right)^2 \hat{\sigma}_d^{\,2}}{\delta^2}, \qquad \alpha' = \frac{0{,}05}{m}
```

R es el número de semillas, m el número de contrastes de la familia, δ el margen del 3 % y σ\_d la desviación de las diferencias pareadas entre dos modelos, estimada en el piloto.

| Margen frente a la desviación de las diferencias | Potencia 0,80, 1 contraste | Potencia 0,80, 12 contrastes | Potencia 0,90, 12 contrastes |
| --- | --- | --- | --- |
| Igual (δ = σ\_d) | 8 | 14 | 18 |
| Tres cuartos (δ = 0,75 σ\_d) | 14 | 25 | 31 |
| La mitad (δ = 0,5 σ\_d) | 32 | 55 | 69 |

- **Punto de partida.** El libro del Entregable 1 ya declaró que la potencia era insuficiente: estimó entre 45 y 60 semillas frente a las 5 corridas.
- **El piloto va primero.** Antes de la corrida final se corre un piloto de 5 semillas sobre dos o tres modelos representativos, se estima σ\_d para cada par y se calcula R. El número se fija en el preregistro.
- **Potencia conjunta.** Con 12 contrastes y potencia de 0,90 en cada uno, la probabilidad de rechazar los 12 a la vez es cercana a 0,28; con 0,95 sube a 0,54. Hay que reportarla.
- **Pruebas de rangos.** Wilcoxon no puede rechazar al 0,05 con 5 pares (p mínimo 0,0625). Con 12 contrastes y Holm necesita al menos 9 pares. Con 10 semillas conviene apoyarse en bootstrap pareado o t pareada.
- **Si el cómputo no alcanza.** Se aumentan las semillas antes de congelar el protocolo. Si aun así no se llega, se declara como limitación y se reporta la diferencia mínima detectable. No se puede reducir la familia de contrastes ni relajar el estándar después de ver resultados.
- **Dos fuentes de variabilidad.** Variar solo la semilla permite concluir sobre este conjunto de prueba. Para concluir sobre el modelo en general hay que variar también la partición, y por eso el Entregable 3 pide tres conjuntos adicionales.

**Cómo reducir el número de semillas necesario sin hacer trampa**

- Usar las mismas semillas, el mismo orden de lotes y los mismos orígenes en todos los modelos: el pareado reduce σ\_d.
- Promediar la métrica sobre los 1.461 orígenes diarios de la prueba antes de comparar semillas.
- Estimar el costo total antes de empezar: semillas por modelos por conjuntos de datos. Con 12 modelos entrenables (Top 10, ARX y el original), 4 conjuntos y 18 semillas son 864 entrenamientos finales.

### Unidad de análisis, familias e intervalos

Cada prueba debe declarar su unidad de análisis, y las observaciones correlacionadas se agregan a esa unidad antes de probar.

| Decisión | Regla para este proyecto |
| --- | --- |
| Unidad en pruebas entre semillas | La semilla: un MASE de prueba por corrida |
| Unidad en pruebas sobre errores | El origen de pronóstico. Las 24 horas de un mismo día no son independientes: se promedian por origen o se prueban hora por hora |
| Bootstrap | Por bloques de orígenes consecutivos, con al menos 2.000 remuestreos. La longitud del bloque se justifica con la autocorrelación del diferencial de pérdida; con estacionalidad semanal, no menos de 7 días |
| Intervalos | Siempre sobre la diferencia pareada. Dos intervalos que se solapan no prueban que no haya diferencia |
| Tamaño del efecto | d de Cohen pareada o delta de Cliff, con intervalo, e interpretado frente al margen del 3 % |
| Probabilidad de superioridad | Fracción de orígenes en que un modelo supera a otro; fácil de comunicar |
| Varianza | Descomponerla entre modelo, origen, semilla y residuo con un modelo de efectos mixtos. Si la varianza por semilla es comparable a la diferencia entre modelos, la conclusión no es estable |

**Familias de contrastes, declaradas a priori y controladas por separado**

| Familia | Contrastes | Control |
| --- | --- | --- |
| Principal del Entregable 2 | Cada modelo del Top 10 frente al mejor baseline fuerte | Holm |
| Texto frente a control | Modelos 6 y 7 frente a su versión sin texto y frente al placebo | Holm |
| Principal del Entregable 3 | Modelo original frente a cada modelo del Top 10 y cada baseline | Holm; deben rechazarse todos |
| Por hora del horizonte | 24 pruebas por contraste | Benjamini-Hochberg |
| Ablación | Cada variante frente al modelo completo | Holm, familia aparte |
| Conjuntos adicionales | Los tres conjuntos del Entregable 3 | Friedman con Iman-Davenport y pares con Holm |

No se mezclan familias para justificar un nivel más permisivo, y ninguna puede recortarse después de ver resultados.

### Selección de arquitectura y sesgo del ganador

Elegir la mejor de 1000 arquitecturas con una métrica ruidosa infla su resultado: según la guía, con ruido de 0,01 el máximo de 1000 candidatos queda cerca de 0,03 por encima de su valor real.

- La arquitectura de TCN se elige solo con validación, por mayor media de la métrica primaria; en empate técnico gana la de menos parámetros.
- Entre los finalistas se identifica el conjunto indistinguible del mejor con TOST, ROPE o Model Confidence Set, y se reporta su tamaño.
- El desempeño final se estima solo en la prueba sellada. No se hacen pruebas pareadas entre las 1000 arquitecturas.
- Lo mismo aplica a la búsqueda de hiperparámetros y al diseño del modelo original: cada variante probada se anota en la bitácora y su número se declara.

### Requisitos propios del modelo original

- **Desarrollo y confirmación separados.** El diseño se itera en una partición de desarrollo que no se reutiliza para la comparación confirmatoria.
- **Benchmarks congelados.** Los resultados del Top 10 se congelan con hash antes de comparar. No se reajustan para favorecer al modelo propio.
- **Mismo presupuesto.** Igual número de configuraciones, épocas máximas y criterio de parada. Si el modelo propio cuesta más por configuración, se iguala el tiempo de cómputo o se reportan ambos análisis.
- **Capacidad controlada.** Si el modelo propio tiene más parámetros que el mejor benchmark, se incluye una variante del benchmark de tamaño comparable.
- **Consistencia.** Debe ser mejor en al menos dos de los tres conjuntos adicionales y no ser significativamente peor en ninguno.
- **Adecuación propia.** También debe pasar las Tablas A y B: superar al ingenuo, brecha dentro del margen y residuos sin estructura.
- **Intervalos con cobertura verificada.** Se piden intervalos de predicción con su cobertura empírica frente a la nominal, por hora.

### Lista de verificación antes de la corrida final

- [ ] Fijar el margen de sobreajuste, que hoy no está en el preregistro.
- [ ] Listar en el preregistro cada familia de contrastes, sus pruebas y su corrección.
- [ ] Correr el piloto, estimar la desviación de las diferencias y fijar el número de semillas.
- [ ] Declarar la longitud de bloque del bootstrap y el nivel del Model Confidence Set.
- [ ] Declarar las reglas de exclusión de corridas (divergencia, valores no numéricos).
- [ ] Comprobar que el script guarda pronósticos por origen y por hora, y la métrica de entrenamiento.
- [ ] Registrar el hash del conjunto de prueba y de las particiones.
- [ ] Etiquetar como exploratorio todo análisis que no esté en el preregistro.

En la redacción no se admite «tendencia a la significancia»: cada resultado se declara según la regla fijada, y la falta de significancia no se presenta como equivalencia.

## Qué cuenta como modelo original en el Entregable 3

El Entregable 3 considera original un modelo solo si cumple a la vez tres condiciones, y además exige un estándar teórico único. La calificación máxima depende de ellas: 3,0 si falla R1 o R2, 4,0 si falla R3 y 5,0 si se cumplen las tres.

| Condición | Qué exige |
| --- | --- |
| R1. Novedad para el problema | El modelo no se ha usado en la literatura para pronosticar demanda eléctrica, y su diferencia frente al trabajo cercano es de mecanismo, de función de pérdida o de esquema de entrenamiento. La novedad solo de aplicación no basta |
| R2. Implementación propia | El componente central se programa desde su formulación matemática, sin importar el modelo de una librería ni copiar código ajeno, y debe poder explicarse línea por línea |
| R3. Superioridad verificada | Menor error que cada modelo del Top 10 y cada baseline fuerte, con al menos tres pruebas por contraste, corrección de Holm sobre toda la familia y diferencia mayor que el margen del 3 % |

**Estándar teórico obligatorio (§2 del Entregable 3)**

- Al menos una propiedad demostrada formalmente, con supuestos explícitos y verificación numérica: cota de estabilidad o de Lipschitz, invarianza, convergencia, expresividad, cota de generalización o propiedad de la pérdida.
- Evaluación en al menos tres conjuntos de datos adicionales de naturaleza distinta, elegidos antes de ver resultados.
- Potencia de al menos 0,90 por contraste y un mínimo de 10 semillas.
- Ablación de al menos tres componentes propios, con control de capacidad.

**Transferencia de mecanismo.** Se permite tomar un mecanismo de otra aplicación si la fuente se cita, se justifica por qué resuelve la limitación y la adaptación cambia el algoritmo (estructura, regla de propagación, pérdida o entrenamiento), no solo los datos o los hiperparámetros.

**Lo que no cuenta como original**

- Importar una arquitectura y cambiarle hiperparámetros, activación, profundidad, ancho o dropout.
- Ajustar un modelo preentrenado con otra cabeza u otro prompt.
- Promediar o apilar modelos del benchmark sin una regla nueva de combinación.
- Aplicar tal cual un mecanismo existente a demanda eléctrica.

La novedad no se declara, se demuestra: tres bases de datos, cinco cadenas de búsqueda, matriz de novedad, búsqueda por citación y verificación por un tercero (§6.3). Por eso las ideas de la sección siguiente son hipótesis de novedad hasta que esa búsqueda las confirme.

## Ideas de arquitectura para el modelo original

La propuesta recomendada es una fusión residual acotada: el texto solo puede corregir el pronóstico numérico dentro de un radio fijo, y esa cota se demuestra. Si el texto no resulta influyente, el mismo mecanismo se aplica a covariables numéricas (idea E). Responde a la limitación medida en este corpus: el texto falta en la mayoría de los días de entrenamiento y cambia de composición en la prueba.

**Limitación que atacan las cuatro ideas.** TaTS y TGForecaster no garantizan nada cuando el texto falta, es ruidoso o cambia de fuente. El propio artículo de TaTS reporta que con texto muy ruidoso hay que descartar parte de los textos para recuperar el desempeño del modelo numérico. Aquí el texto oficial cubre el 37 % de los días de 2008-2019, e IDEAM pasa de cerca del 12 % al 43 % de los documentos en la prueba. Esta limitación debe cuantificarse con un piloto sobre los modelos 6 y 7 antes de fijar el diseño.

### Idea A (recomendada): fusión residual acotada con compuerta de ausencia

El pronóstico es el del modelo numérico más una corrección de texto acotada y modulada por una compuerta.

```latex
\hat{y} = \hat{y}_{\mathrm{num}} + g \cdot \rho \, \tanh\!\big(h_\phi(z, c)\big), \qquad g = m \cdot \sigma\!\big(a_\psi(c)\big) \in [0, 1]
```

z es el resumen de texto, c el contexto numérico, m la bandera de presencia de texto (0 si no hay) y ρ el radio máximo de corrección.

- **Propiedad demostrable:** para cualquier texto, cada hora del pronóstico difiere del numérico en a lo sumo ρ. Por la desigualdad triangular, el error absoluto medio no puede empeorar en más de ρ frente al modelo numérico. Sin texto (m = 0), la salida es idéntica a la numérica.
- **Verificación numérica:** texto adversario y texto aleatorio, comprobando que la desviación máxima nunca supera ρ.
- **Componentes para la ablación:** la compuerta, la cota y la bandera de ausencia.

### Idea B: núcleo de respuesta al impulso de los documentos

Cada documento aporta su vector multiplicado por un núcleo aprendido que decae con el tiempo transcurrido desde su disponibilidad, en vez de promediar por ventana fija.

```latex
z(t_0) = \sum_{i:\, t_i < t_0} \kappa(t_0 - t_i)\, e_i, \qquad \kappa(\Delta) = \sum_{k=1}^{K} a_k \, e^{-\Delta/\tau_k}, \quad a_k \ge 0
```

- **Propiedad demostrable:** invarianza al orden de los documentos e invarianza a todo documento disponible después del origen, es decir, ausencia de fuga por construcción. También una cota de estabilidad: la norma del resumen no supera la suma de los pesos por la norma máxima de los vectores.
- **Qué resuelve:** sustituye la elección de ventana (24, 72 o 168 h) por un decaimiento aprendido, y trata bien la llegada irregular de documentos.

### Idea C: pérdida con texto placebo

Esquema de entrenamiento que añade un término de penalización: con el texto permutado en el tiempo, el pronóstico debe coincidir con el del modelo numérico.

- **Propiedad demostrable:** propiedad de la pérdida. Si el texto es independiente de la historia y del objetivo, el minimizador poblacional no usa el texto.
- **Categoría de novedad:** función de pérdida o esquema de entrenamiento, que también sustenta R1.

### Idea D: intervalos de predicción condicionados por texto

El ancho del intervalo conforme se escala con una medida de incertidumbre derivada del texto.

- **Límite:** no reduce el error puntual, así que por sí sola no sirve para R3. Su garantía de cobertura es heredada y supone intercambiabilidad, que una serie temporal viola. Se deja como complemento.

### Por qué sería original frente a trabajos similares

| Trabajo cercano | Qué hace | Qué añade la propuesta | Tipo de diferencia |
| --- | --- | --- | --- |
| TaTS, Li et al. (2025) | Concatena el texto como canales de entrada | Cota demostrada sobre el efecto del texto y salida idéntica sin texto | Mecanismo |
| TGForecaster, Xu et al. (2024) | Atención cruzada entre texto y serie, con texto alineado al horizonte | Solo texto pasado, con decaimiento aprendido y garantía de no fuga | Mecanismo |
| [Time-LLM](https://arxiv.org/abs/2310.01728), Jin et al. (2024) | Reprograma la serie hacia un modelo de lenguaje con un prompt estático | Texto dinámico y fechado, sin depender de un modelo de lenguaje grande | Mecanismo |
| Gated Multimodal Units, Arevalo et al. (2017) | Compuerta que mezcla modalidades en clasificación | Compuerta residual con radio acotado y bandera de ausencia, para regresión de 24 pasos | Mecanismo transferido y adaptado |
| [ReZero](https://arxiv.org/abs/2003.04887), Bachlechner et al. (2020) | Rama residual que arranca en cero | La rama es otra modalidad y su magnitud queda acotada durante todo el entrenamiento | Mecanismo transferido y adaptado |
| [Neural Hawkes Process](https://www.arxiv.org/abs/1612.09328v1), Mei y Eisner (2017) | Núcleos de decaimiento para eventos puntuales | El núcleo pondera vectores de documentos para pronosticar una serie continua | Mecanismo transferido y adaptado |

Gated Multimodal Units se cita de memoria y falta localizar su identificador. Los modelos de demanda eléctrica con texto que aparezcan en la revisión sistemática deben añadirse a esta tabla, porque son los que deciden R1.

### Opción solo numérica, si el texto no resulta influyente

Si el texto no aporta, el mismo mecanismo de la idea A se conserva cambiando la rama de texto por una rama de covariables numéricas. Así no se pierde la demostración ni el código ya escrito.

**Regla de cambio, a declarar antes de congelar el protocolo.** Se pasa a la opción numérica si, en el piloto de validación, los modelos 6 y 7 no mejoran el MASE de su control sin texto en al menos un 3 %, o si no superan al texto placebo. La decisión se toma con validación, nunca con la prueba, y se anota en la bitácora de diseño.

**Idea E (recomendada como respaldo): fusión residual acotada de covariables exógenas**

```latex
\hat{y} = \hat{y}_{\mathrm{dem}} + g \cdot \rho \, \tanh\!\big(h_\phi(x_{\mathrm{exo}}, c)\big), \qquad g = \sigma\!\big(a_\psi(c)\big) \in [0, 1]
```

El primer término usa solo la historia de demanda y el calendario. La rama exógena recibe clima, hidrología y actividad económica.

- **Limitación que ataca:** las covariables exógenas son poco fiables en este proyecto. El clima es reanálisis y no pronóstico, las horas de publicación de la hidrología no están certificadas y 2020 es un año atípico. Un modelo que las mezcla sin límite puede degradarse cuando cambian.
- **Propiedad demostrable:** la misma cota de la idea A. Cada hora del pronóstico difiere del modelo de solo demanda en a lo sumo ρ, cualquiera que sea el valor de las covariables.
- **Verificación numérica:** covariables perturbadas, faltantes o desplazadas, comprobando que la desviación nunca supera ρ.
- **Componentes para la ablación:** la compuerta, la cota y la separación en dos ramas.

**Idea F: mezcla convexa de perfiles diarios con nivel separado**

El pronóstico de las 24 horas es un nivel de demanda multiplicado por una combinación convexa de K perfiles diarios aprendidos, con pesos que dependen del calendario y de la historia.

```latex
\hat{y} = \ell(x) \cdot \sum_{k=1}^{K} \pi_k(x)\, P_k, \qquad \pi_k \ge 0, \quad \sum_k \pi_k = 1
```

- **Limitación que ataca:** la demanda crece con el tiempo, y el preregistro ya advierte que la escala de 2000-2017 no es comparable con la de la prueba. Además los festivos cambian la forma del día, no solo su nivel.
- **Propiedad demostrable:** equivarianza de escala. Si el nivel se calcula con una función homogénea de grado 1 y los pesos con entradas normalizadas, multiplicar la historia por una constante multiplica el pronóstico por esa constante. También que la forma diaria queda dentro de la envolvente convexa de los perfiles.
- **Ventaja adicional:** los perfiles y sus pesos son interpretables y se pueden mostrar en el dashboard.

**Por qué serían originales frente a trabajos numéricos cercanos**

| Trabajo cercano | Qué hace | Qué añade la propuesta | Tipo de diferencia |
| --- | --- | --- | --- |
| TFT, Lim et al. (2021) | Selecciona y mezcla covariables con compuertas, sin límite sobre su efecto | Efecto de las covariables acotado y demostrado (idea E) | Mecanismo |
| NBEATSx, Olivares et al. (2023) | Añade covariables exógenas a bloques de N-BEATS | Rama exógena residual con radio fijo (idea E) | Mecanismo |
| N-BEATS, Oreshkin et al. (2020) | Expansión en bases de tendencia y estacionalidad | Pesos sobre un símplex y nivel separado, con equivarianza exacta (idea F) | Mecanismo |
| RevIN, Kim et al. (2022) | Normaliza cada ventana y revierte la escala a la salida | Equivarianza garantizada por construcción y forma diaria acotada (idea F) | Mecanismo, por confirmar |

NBEATSx, N-BEATS y RevIN se citan de memoria y sus referencias deben verificarse. La idea F es la de novedad más dudosa: los métodos de día similar y de perfiles de carga son antiguos en demanda eléctrica, y la búsqueda del §6.3 puede mostrar que ya existe algo equivalente.

**Qué cambia en el resto del diseño si se toma esta opción**

- Los modelos 6 y 7 siguen en el Top 10 y sus resultados se reportan, aunque el texto no aporte.
- Los tres conjuntos adicionales pueden ser series públicas de demanda o carga con covariables, sin texto, que son más fáciles de conseguir.
- La auditoría de fuga del texto se sustituye por la de covariables: solo valores disponibles antes del origen, y clima futuro solo si viene de un pronóstico con fecha de emisión.

### Recomendación y riesgos

- **Combinación sugerida:** A como núcleo, con el resumen de texto de B. Da tres componentes aislables (compuerta, cota, núcleo temporal) y una propiedad sencilla de demostrar. C queda como variante de entrenamiento.
- **Hipótesis falsable:** la fusión acotada reduce el MASE en los días con texto en al menos un 3 % y no lo cambia en los días sin texto. Predicción secundaria: la ventaja desaparece con texto placebo.
- **Riesgo para R3:** la cota asegura no empeorar, pero la superioridad exige que el texto aporte señal. Con unos 12 contrastes y potencia de 0,90 por contraste, harán falta bastantes más de 10 semillas. Si el piloto muestra un efecto pequeño, el resultado probable es el tope de 4,0, y conviene definir desde ya el hallazgo alternativo.
- **Pregunta abierta:** qué tres conjuntos públicos de series con texto emparejado se usarán como evaluación adicional. Deben elegirse antes de ver resultados.
- **Pregunta abierta:** si el modelo base de la idea A será el mejor modelo numérico del Top 10 o uno propio, porque R2 exige que el núcleo sea código propio.

## Referencias

Cada idea del diseño se apoya en uno de estos 9 trabajos. Los dos de fusión con texto se leyeron en su texto completo; de los otros 7 se comprobó título e identificador, y falta confirmar el DOI de la versión publicada.

| Referencia | Idea que se toma | Enlace |
| --- | --- | --- |
| Li, Z., Lin, X., Liu, Z., Zou, J., Wu, Z., Zheng, L., Fu, D., Zhu, Y., Hamann, H., Tong, H. y He, J. (2025). *Language in the Flow of Time: Time-Series-Paired Texts Weaved into a Unified Temporal Narrative* | Texto como variables auxiliares concatenadas a la serie; reducción con MLP; frase fija para texto ausente (modelo 6) | [arXiv:2502.08942](https://arxiv.org/html/2502.08942v3) |
| Xu, Z., Bian, Y., Zhong, J., Wen, X. y Xu, Q. (2024). *Beyond Trend and Periodicity: Guiding Time Series Forecasting with Textual Cues* | Atención cruzada en dos etapas entre texto y serie sobre un codificador por parches; texto precodificado (modelo 7) | [arXiv:2405.13522](https://arxiv.org/html/2405.13522v2) |
| Nie, Y., Nguyen, N. H., Sinthong, P. y Kalagnanam, J. (2023). *A Time Series is Worth 64 Words: Long-term Forecasting with Transformers*. ICLR | Parches de la serie y canales independientes (modelo 1 y codificador del modelo 7) | [arXiv:2211.14730](https://arxiv.org/abs/2211.14730v1) |
| Challu, C., Olivares, K. G., Oreshkin, B. N., Garza, F., Mergenthaler-Canseco, M. y Dubrawski, A. (2023). *NHITS: Neural Hierarchical Interpolation for Time Series Forecasting*. AAAI | Muestreo a varias escalas e interpolación jerárquica (modelo 2) | [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/25854) |
| Lim, B., Arık, S. Ö., Loeff, N. y Pfister, T. (2021). *Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting*. International Journal of Forecasting | Covariables conocidas a futuro y selección de variables (modelo 3) | [arXiv:1912.09363](https://arxiv.org/pdf/1912.09363) |
| Bai, S., Kolter, J. Z. y Koltun, V. (2018). *An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling* | Convoluciones causales dilatadas con bloques residuales (modelo 4) | [arXiv:1803.01271](https://arxiv.org/pdf/1803.01271) |
| Liu, Y., Hu, T., Zhang, H., Wu, H., Wang, S., Ma, L. y Long, M. (2024). *iTransformer: Inverted Transformers Are Effective for Time Series Forecasting*. ICLR | Cada variable como un token; atención entre variables (modelo 5) | [arXiv:2310.06625](https://arxiv.org/abs/2310.06625) |
| Wang, L., Yang, N., Huang, X., Yang, L., Majumder, R. y Wei, F. (2024). *Multilingual E5 Text Embeddings: A Technical Report* | Codificador de texto multilingüe propuesto | [arXiv:2402.05672](https://arxiv.org/pdf/2402.05672) |
| Reimers, N. y Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*. EMNLP | Vector de frase por promedio de tokens | [arXiv:1908.10084](https://arxiv.org/abs/1908.10084) |

Las reglas de equidad, la búsqueda de 1000 arquitecturas y las pruebas estadísticas vienen de la guía del Entregable 2 (§11 y §14) y del `PREREGISTRO.md` del repositorio.

TaTS y TGForecaster se evaluaron en sus artículos con conjuntos de datos propios. Al citarlos en el libro, la justificación debe apoyarse en el mecanismo y no en esos conjuntos.

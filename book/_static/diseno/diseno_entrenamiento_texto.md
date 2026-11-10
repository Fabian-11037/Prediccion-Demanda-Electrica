# Diseño de entrenamiento: pronóstico de demanda eléctrica con texto

**Fabian José Manrique Morelos y Gloria Bastidas Rodriguez**
Versión de diseño: 10 de noviembre de 2026.

## 1. Pregunta y comparación principal

¿La incorporación de documentos oficiales y, posteriormente, de prensa mejora el pronóstico horario de demanda eléctrica del SIN a 24 horas, con respecto a modelos que reciben la misma información numérica?

El problema es de **series de tiempo multivariadas con pronóstico multipaso directo**. Cada origen diario, a las 00:00 de Bogotá, utiliza 168 horas de historia y genera los 24 intervalos siguientes. Se comparan tres variantes sobre los mismos orígenes y etiquetas:

1. **N:** demanda, calendario, clima, hidrología y precio.
2. **N+O:** las mismas variables numéricas y documentos oficiales.
3. **N+O+P:** las mismas variables numéricas, documentos oficiales y prensa.

El contraste N+O frente a N estima el aporte oficial; N+O+P frente a N+O estima el aporte adicional de prensa. Cada arquitectura textual tendrá su control sin texto con el mismo núcleo y un presupuesto comparable. Los ingenuos diario y semanal y ARX se conservan como referencias adicionales. Las métricas actuales de demanda y calendario no representan el rendimiento del conjunto ampliado.

## 2. Contrato numérico común

El contrato ejecutable se conserva en `config/diseno_entrenamiento_texto.json`; `src/demanda/diseno_numerico.py` prepara sus columnas sin modificar el panel congelado. Todas las familias reciben **30 variables históricas y 23 variables de calendario**. Los indicadores de ausencia se añaden después, bajo una política común.

| Bloque | Variables | Cantidad | Uso temporal |
|---|---|---:|---|
| Demanda | Demanda real horaria del SIN, kWh | 1 | Historia; rezago heredado de 1 h |
| Clima urbano | Temperatura a 2 m, humedad relativa a 2 m y precipitación de la hora precedente, para Bogotá, Medellín, Cali, Barranquilla, Cartagena, Bucaramanga, Pereira y Cúcuta | 24 | Solo historia; rezago de diseño de 168 h |
| Hidrología | Llenado, reserva útil, capacidad útil y aportes diarios del SIN | 4 | Último día habilitado, 24 h después de terminar el día |
| Precio | Precio nacional de bolsa, COP/kWh | 1 | Solo historia; rezago de 24 h |
| Calendario | Hora, semana, mes, ciclo anual, festivos, vísperas, días posteriores y temporadas | 23 | Historia y horizonte; conocidas en el origen |

La lista exacta de nombres está en el JSON y se verifica contra el registro de variables. No se seleccionan subconjuntos diferentes mediante la búsqueda de cada modelo. Los promedios climáticos ponderados no se añaden a la especificación principal porque duplican las observaciones urbanas y dependen de pesos poblacionales revisados. ISE y pronósticos meteorológicos emitidos se reservan para análisis separados con periodos y versiones compatibles.

### Disponibilidad meteorológica

El panel conserva ERA5 retrospectivo. El nuevo preparador añade 167 h al desplazamiento climático de 1 h ya aplicado: el rezago efectivo es 168 h. Esta decisión permite estudiar clima pasado con una política uniforme; **no acredita que la versión recuperada estuviera publicada históricamente**. ECMWF distingue ERA5T, inicial y aproximadamente cinco días retrasado, de ERA5 final, publicado dos a tres meses después. ERA5T está disponible desde finales de 2019. El experimento 2000–2025 con el reanálisis recuperado se interpreta como retrospectivo, no como simulación operativa certificada.

Se fija una sensibilidad de 90 días de rezago climático, manteniendo el mismo resto del contrato. Tampoco sustituye una auditoría de versiones históricas. Una aplicación operativa requiere observaciones o pronósticos cuya emisión y publicación se registren. No se entrega al modelo el clima, la hidrología o el precio realizados durante el horizonte.

Fuente: [ECMWF, disponibilidad de ERA5T y ERA5](https://www.ecmwf.int/en/about/media-centre/news/2026/era5t-reanalysis-data).

### Ausencias y transformaciones

El preparador conserva los valores ausentes y produce una máscara por entrada. Dentro de cada pliegue, primero se separan ajuste, parada interna y validación externa; después se calculan medianas y escalas exclusivamente en ajuste. No se interpolan valores usando fechas posteriores. Una columna sin observaciones en ajuste se representa por cero con máscara activa y se registra como no estimable. Los días sin texto no se eliminan. La demanda necesaria para formar entradas, nivel y etiquetas debe estar completa; la elegibilidad se determina una sola vez para todas las variantes.

La demanda se normaliza por el nivel de la ventana y por estadísticas de ajuste según el contrato del motor validado; las covariables se estandarizan con sus propias escalas, nunca con la escala de demanda. El nivel se conserva para invertir la predicción a kWh. Cualquier recorte o transformación se ajusta sin consultar parada externa ni prueba.

## 3. Datos, versiones y partición

El manifiesto `data/processed/dataset_ampliado_20261110.json` identifica archivos y SHA-256. La etiqueta Git `datos_adicionales_20261110` apunta a `08459773b00fbd8fec8765f09225e07aff9bdbba`. El corpus integrado contiene 575.473 registros, incluidos 7.907 versiones documentales oficiales; 729 titulares fueron enriquecidos. Son totales de inventario, no conteos de documentos elegibles en cada origen.

Se utiliza el **escenario declarado como dataset textual oficial**. El escenario estricto es una sensibilidad de disponibilidad, cuya limitada cobertura debe acompañar los resultados. Las versiones históricas no verificadas mantienen su advertencia. Fecha sin hora: disponibilidad a las 00:00 del día siguiente en Bogotá. Una captura o recuperación posterior no demuestra publicación anterior de esa versión. Los documentos en cuarentena o sin fecha utilizable no entran al modelo.

La referencia temporal es 2000–2017; los cuatro pliegues externos validan 2018, 2019, 2020 y 2021 con historia expansiva. El último 10 % cronológico de cada historia se reserva para parada interna, con purga de 24 h. Las escalas y las imputaciones se ajustan fuera de esa reserva. El ajuste final utiliza historia anterior a 2022 con la misma regla interna. El bloque 2022–2025 se conserva para evaluación retrospectiva: ya fue examinado. La unidad de evaluación es el origen diario; MASE es la métrica, no la unidad de independencia.

Los datos 2026 descargados también son retrospectivos. Una evaluación prospectiva empieza después de congelar protocolo y modelo y registrar, en tiempo real, entradas y pronósticos. No se modifica ni se retrofecha el preregistro existente. Antes de ejecutar la comparación ampliada se registrarán configuración final, hipótesis, familias de contrastes, presupuesto, semillas, hashes y fecha efectiva.

## 4. Representación del texto

Se conservan canales independientes: **oficial** (XM, IDEAM, CREG, UPME y MinEnergía) y **prensa** (El Colombiano, El Tiempo, El Espectador, La República, Portafolio, Semana y Dinero). La relevancia se aprende mediante la pérdida de pronóstico. No se filtra por términos, categorías ni tópicos manuales; las fuentes y el alcance de extracción se conservan para auditoría y análisis por segmento.

La ventana principal es de 72 h, con sensibilidades de 24 y 168 h fijadas en desarrollo. Cada canal admite hasta 64 documentos por origen, seleccionados por disponibilidad más reciente y desempate por identificador; cada documento admite hasta ocho fragmentos, en orden de lectura. El presupuesto de prensa no desplaza al oficial. Se registran documentos y tokens descartados, porcentaje de truncación y alcance (titular, fragmento o cuerpo). Estos límites controlan memoria; no garantizan igual cantidad de información entre documentos.

Se adopta `intfloat/multilingual-e5-base`, revisión `d128750597153bb5987e10b1c3493a34e5a4502a`, como extractor congelado de 768 dimensiones. Se fijan el mismo commit para modelo/tokenizer, prefijo `query: ` para esta tarea distinta de recuperación, promedio de tokens con máscara y normalización L2. El máximo de 512 tokens incluye prefijo y tokens especiales; el contenido se fragmenta con el espacio restante y solapamiento de 64 tokens. No se descargan ni ajustan pesos mediante esta actualización documental.

El caché se identifica por SHA-256 del texto, revisión del encoder/tokenizer, prefijo, pooling y política de fragmentación; no solo por URL. Se conservan embeddings por fragmento/documento. Dentro de cada canal:

$$z_{c,t}=\sum_{j\in D_{c,t}}\alpha_{c,t,j}e_{c,j},\qquad
\alpha_{c,t,j}=\operatorname{softmax}_j s_{\theta_c}(e_{c,j},q_t).$$

La consulta $q_t$ depende únicamente de la historia numérica habilitada. La ponderación se aprende en ajuste; después puede producir una secuencia diaria para TaTS o tokens para atención cruzada. Sin documentos, $z_{c,t}=0$ y el indicador de ausencia del canal está activo. El promedio diario fijo no reemplaza esta selección aprendida.

Las URL y fechas de recuperación no se concatenan al contenido. Se conservan magnitudes, horizontes y fechas anunciadas que puedan expresar anticipación legítima; mencionar un año futuro no demuestra fuga. La muestra de fechas sospechosas se audita usando desarrollo. Cualquier enmascarado se fija antes de evaluar y no se presenta como solución a revisiones editoriales o contaminación del preentrenamiento. El corte completo del corpus del encoder no está certificado para este problema.

Fuente: [ficha oficial de multilingual-E5-base](https://huggingface.co/intfloat/multilingual-e5-base).

## 5. Familias del benchmark

El Top 10 combina MLP, LSTM y CNN1D con siete familias adicionales. ARX e ingenuos son referencias externas al conteo. Todas las nuevas comparaciones usan el contrato numérico ampliado; los checkpoints existentes permanecen conservados.

| Modelo | Mecanismo y adaptación de interés | Comparación |
|---|---|---|
| MLP | Mapeo denso de ventana y calendario a 24 salidas | Control numérico |
| LSTM | Estado recurrente de la historia multivariada | Control numérico |
| CNN1D | Filtros locales sobre la secuencia | Control numérico |
| PatchTST | Tokens de parches temporales, atención y cabeza multipaso | Numérico y núcleo candidato para fusión |
| N-HiTS | Bloques residuales, agrupación e interpolación multiescala | Numérico |
| TFT | Selección de variables, recurrencia y atención; separa conocidas a futuro | Numérico |
| TCN | Bloques causales dilatados, dos convoluciones por bloque | Numérico |
| iTransformer | Tokens por variable y atención entre variables | Numérico y núcleo candidato para fusión |
| TaTS adaptado | Proyección de representaciones textuales como variables auxiliares | N, N+O y N+O+P |
| TGForecaster adaptado | Relevancia de texto y fusión por atención cruzada | N, N+O y N+O+P |

El núcleo de TaTS se elige entre PatchTST e iTransformer por menor MASE medio de validación, sin prueba; su coste de selección se contabiliza. TGForecaster se adapta a documentos exclusivamente habilitados antes del origen; no dispone de texto futuro alineado al horizonte. La agregación aprendida y los dos canales son adaptaciones del experimento. Estas descripciones no equivalen a reproducir literalmente los resultados de los artículos.

Para TCN, con dos convoluciones causales de núcleo $k$ en cada bloque y dilataciones $(1,4,16)$:

$$R=1+2(k-1)(1+4+16).$$

Así, $R=169$ para $k=5$ y $R=253$ para $k=7$, suficientes para 168 h. Una sola convolución por bloque daría 85 o 127; cada implementación debe probar el campo receptivo real y la ausencia de acceso futuro.

## 6. Entrenamiento, selección y presupuesto

El entrenador futuro debe reproducir el contrato de escalado por pliegue, parada interna y restauración de pesos de las fuentes v3/v4 conservadas en `results/repositorio/`, commit de resultados `d1ae9a4e09dd1dead4af1329f16aa3e8d0f2c3d3`. El módulo activo antiguo no se considera una reproducción automática de esos checkpoints. La ampliación se implementará en un motor versionado separado y no sobrescribirá pesos ni resultados existentes.

Se fija un presupuesto nuevo para los hiperparámetros no arquitectónicos: 20 configuraciones por familia, con tres semillas comunes y los cuatro pliegues externos; máximo 200 épocas, pérdida MAE normalizada, AdamW, recorte de gradiente 1 y parada interna. La tasa, lote y regularización se ajustan con ese presupuesto. En cada evaluación se restaura el mínimo absoluto de parada interna. La búsqueda arquitectónica posterior mantiene fijos esos parámetros y su criterio de parada; no vuelve a escogerlos por candidato. Las ejecuciones fallidas y sus costes se registran y se repiten solo bajo una regla técnica común.

La pauta E2 §14 exime de la búsqueda de arquitecturas únicamente cuando el artículo define una configuración concreta que se adopta y documenta. Nombrar una familia publicada no basta. Antes de ejecutar, la ficha de cada modelo debe enlazar la configuración y su traducción a los datos; si no queda definida, se aplica el espacio obligatorio. TCN se declara sin diseño predefinido. Lo mismo se aplica al cabezal propio cuando no tenga configuración publicada; no se exime por utilizar un encoder preentrenado.

El espacio obligatorio contiene tres bloques con anchos $(w_1,w_2,w_3)\in\{16,32,64,128,256\}^3$ y patrones de dropout $(b_1,b_2,b_3)\in\{0,1\}^3$, tasa 0,2: **1.000 arquitecturas por modelo sujeto a §14**. En TCN cada ancho corresponde a un bloque con dos convoluciones y el dropout se aplica al final del bloque. Se enumeran todas las combinaciones sin poda previa.

- **Fase 1:** 1.000 candidatos, semilla 42, máximo 50 épocas (25 % del presupuesto completo), mismo criterio de parada.
- **Fase 2:** diez mejores por menor MASE externo y el control $(64,64,64),(0,1,0)$ si no está incluido; máximo 200 épocas, semillas 100–104. La regla de desempate favorece menor complejidad y después identificador.
- **Piloto de precisión:** al menos cinco semillas, conservando diferencias pareadas por origen y semilla; dimensionar las ejecuciones definitivas según el margen del 3 %, la varianza observada y la dependencia temporal. Cinco semillas son un mínimo, no acreditan por sí mismas potencia 0,80.
- **Evaluación definitiva:** semillas 1000 en adelante, en número fijado por el piloto antes de ejecutar. Si el presupuesto impide alcanzar la potencia objetivo, se declara la limitación y no se afirma equivalencia a partir de ausencia de significación.

Los costes de ajuste del núcleo, fusión, arquitectura, controles y codificación se contabilizan por separado. Las tres variantes reciben el mismo número de oportunidades de selección aplicables y el mismo protocolo. Se informa además tiempo, memoria y número de parámetros; igualdad de oportunidades no implica igual tiempo de cómputo. Este presupuesto es una especificación nueva, no una descripción de la búsqueda ya realizada.

La selección minimiza el **MASE medio entre semillas y pliegues**, con ponderaciones fijadas antes de la comparación. Se usan pliegues anuales con igual peso. No se escoge arquitectura por prueba, no se repite la búsqueda después de observarla y no se convierte una sensibilidad favorable en el resultado principal.

## 7. Evaluación y controles

La escala se conserva fija, $q_{168}=215617,90361589016$ kWh, calculada en entrenamiento de referencia. Para cada origen:

$$\operatorname{MASE}_t=\frac{1}{24q_{168}}\sum_{h=0}^{23}|y_{t+h}-\hat y_{t+h}|.$$

Menor MASE es mejor. Se reportan MAE, RMSE, errores por horizonte, festivos, precio y condiciones hidrológicas, junto con cobertura de texto y ausencia numérica. El margen práctico es **3 % relativo**: para una referencia $B$, $\Delta_{rel}=100(MASE_A-MASE_B)/MASE_B$. El mismo denominador de referencia rige dimensionamiento, contraste de mejora y equivalencia de esa comparación; no se cambian márgenes entre etapas. La referencia primaria del aporte oficial es N y la del aporte de prensa es N+O. Las comparaciones globales con MLP se identifican por separado.

Los errores horarios de un día no son 24 réplicas independientes. La inferencia primaria utiliza diferencias pareadas por origen, promediadas sobre semillas para el estimando principal, y bootstrap temporal por bloques de 14 días, 10.000 remuestreos, con sensibilidad de 7 y 28 días. La incertidumbre de inicialización se informa aparte; los IC temporales condicionan las semillas ejecutadas. Se aplica Holm a la familia preregistrada de contrastes principales. La potencia objetivo es 0,80 para el benchmark E2 y 0,90 para las afirmaciones de superioridad del modelo original E3; no se supone acreditada antes del piloto. Se dimensiona con diferencias en la misma escala y remuestreo que respete dependencia; no se usa una varianza horaria como si fueran días independientes.

DM con corrección HAC sobre pérdida absoluta es un contraste secundario cuando sus supuestos sean defendibles. El anidamiento de una compuerta no habilita automáticamente Clark–West: su ajuste de errores cuadrados no es un contraste de MAE/MASE. TOST sobre el margen relativo se acompaña de un IC compatible con dependencia. Friedman y comparaciones por semillas/pliegues se presentan con sus unidades y limitaciones; no sustituyen la inferencia temporal. No se escoge la prueba primaria según Shapiro–Wilk con cinco semillas. Tampoco se calcula potencia conjunta como $0,9^{12}$ sin independencia.

Controles registrados: contenido real frente a placebo; indicadores de ausencia y conteo sin contenido; ablación de cada canal; ablación por fuente y balance de volumen; degradación de cobertura al 100 %, 50 % y 0 %. El placebo conserva máscara y volumen mediante ruido independiente o documentos donantes disponibles antes de cada origen, procedentes del tramo de ajuste; una permutación dentro de prueba no asegura causalidad temporal. Los sorteos no usan resultados ni textos posteriores al origen.

## 8. Contribución propia: fusión residual acotada

La alternativa principal de investigación utiliza una base numérica congelada $b(X_t,F_t)$ y una corrección con relevancia textual aprendida:

$$\hat y_t=b(X_t,F_t)+m_t\,\rho\,\sigma(g(q_t,z_{O,t},z_{P,t}))\odot\tanh h(q_t,z_{O,t},z_{P,t}).$$

Aquí $m_t$ vale cero cuando faltan ambos canales; cada canal mantiene también su máscara propia. En escala normalizada, $|\hat y_{t,h}-b_{t,h}|\le\rho$. Por la desigualdad triangular inversa, $|MAE(\hat y)-MAE(b)|\le\rho$: es una **cota de deterioro posible**, no una garantía de mejora. En kWh la cota depende del nivel $\ell_t$, y en MASE queda acotada por $\rho\,\overline{\ell}/q_{168}$. El radio $\rho$ no es el margen del 3 %. Sin texto la predicción coincide exactamente con la base.

La base mantiene `requires_grad=False` y modo `eval` incluso cuando se activa `train` en la fusión. `no_grad` no desactiva dropout. La implementación debe sobrescribir `train` para restaurar `base.eval()` y probar identidad sin texto, cota, ausencia de gradientes y causalidad. Las afirmaciones de novedad requieren contrastar de cinco a diez métodos cercanos; añadir una compuerta por sí solo no demuestra originalidad.

### Pérdida placebo y otras extensiones

Si la penalización cuadrática obliga a aproximarse a una base congelada $b(X)$ y el texto es independiente de $(X,Y)$, el óptimo no restringido es

$$f^*(X)=\frac{E[Y\mid X]+\lambda b(X)}{1+\lambda}.$$

Solo coincide con la media condicional si la base ya es óptima o $\lambda=0$. El radio y la capacidad de la red restringen adicionalmente el óptimo. Por tanto, la penalización es una hipótesis de regularización a evaluar, no una prueba de recuperación de la media condicional.

Un núcleo temporal positivo puede ponderar documentos mediante mezclas de decaimiento exponencial, siempre sobre documentos elegibles. La suma se normaliza para evitar confundir volumen con contenido y su escala se fija con ajuste. La corrección exógena acotada es una ablación, no un modo de excluir clima, hidrología o precio del control numérico común.

La mezcla convexa de perfiles diarios es una alternativa: $\hat y=\ell(x)a(x/\ell(x),X_{exo},F)\sum_k\pi_k P_k$, con $a>0$ y perfiles de media uno. Un factor $a$ dependiente del perfil normalizado, covariables y calendario permite cambiar el nivel del día siguiente. Si $a$ y $\pi$ no dependen de la escala absoluta de demanda, se conserva $\hat y(cx,X_{exo},F)=c\hat y(x,X_{exo},F)$ para $c>0$ manteniendo exógenas fijas. Sin $a$, la media pronosticada queda forzada a $\ell(x)$ y puede impedir cambios asociados a festivos. Su utilidad y novedad requieren evaluación propia.

## 9. Entregables de implementación

Antes del siguiente entrenamiento se deben completar las fichas de configuración publicada, los adaptadores de covariables, el motor común, las pruebas de causalidad y el tamaño definitivo de semillas. La configuración JSON marca este estado como diseño para implementación: no es un registro de entrenamientos ejecutados. El cálculo de embeddings, la búsqueda y la prueba requieren una ejecución explícita posterior.

El pipeline guardará, por pliegue, la lista de entradas, latencias, máscaras, filas de ajuste, escaladores, configuración, semillas, curva de pérdida, mejor época y predicciones con origen/horizonte. Las tres variantes deben producir exactamente el mismo índice de evaluación. Se guardan SHA-256 de datos, código, caché y checkpoints. CPU o GPU se seleccionan explícitamente; los resultados de precisión o latencia indican dispositivo, versiones y tamaño de lote. Las salidas se escriben en una carpeta nueva, sin reemplazar el benchmark conservado.

## Referencias de las arquitecturas

- [PatchTST, Nie et al., ICLR 2023](https://arxiv.org/abs/2211.14730).
- [N-HiTS, Challu et al., AAAI 2023](https://ojs.aaai.org/index.php/AAAI/article/view/25854).
- [TFT, Lim et al., 2021](https://arxiv.org/abs/1912.09363).
- [TCN, Bai et al., 2018](https://arxiv.org/abs/1803.01271).
- [iTransformer, Liu et al., ICLR 2024](https://arxiv.org/abs/2310.06625).
- [TaTS, Li et al., versión 3 del 7 de marzo de 2026; primera prepublicación de 2025](https://arxiv.org/html/2502.08942v3).
- [TGForecaster, Xu et al., versión 2](https://arxiv.org/html/2405.13522v2).
- [Multilingual E5, Wang et al., 2024](https://arxiv.org/abs/2402.05672).

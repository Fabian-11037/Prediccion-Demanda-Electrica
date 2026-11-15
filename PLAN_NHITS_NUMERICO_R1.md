# N-HiTS numérico: arquitectura fija y selección escalonada

Fecha de lanzamiento prevista: 14 de noviembre de 2026, America/Bogota. Ejecución exploratoria autorizada de entrenamiento y prueba. Modelo quinto del Top 10; las corridas y checkpoints previos se conservan.

## Fuente e integración

Artículo: Challu et al., N-HiTS, AAAI 2023, https://arxiv.org/abs/2201.12886. Código oficial Nixtla/neuralforecast, revisión `94b6ce131838be38a1fb5157349f02bbc8c89ef6`, archivo `neuralforecast/models/nhits.py`, licencia Apache 2.0 conservada en `src/demanda/vendor/nhits`. Se conserva el archivo original completo y se extraen literalmente sus bloques para integrarlos con el entrenador común del proyecto.

Configuración fija del código de referencia: tres escalas, un bloque por escala, `mlp_units=[[512,512]]*3`, MaxPool con tamaños `[2,2,1]`, factores de reducción de nodos `[4,2,1]`, ReLU e interpolación lineal. La revisión fijada construye una proyección de entrada a512 y tres transformaciones512→512 por bloque; se adopta esa construcción exacta. Historia168, horizonte24, nodos6/12/24. Se conservan residuales de reconstrucción y suma de contribuciones de pronóstico, con nivel inicial igual al último valor disponible. No hay cabezal adicional. No se afirma réplica exacta del artículo ni exención automática del requisito E2 de1000arquitecturas: esta corrida adopta explícitamente la ruta exploratoria fija propuesta.

La interpolación lineal de PyTorch no tiene backward determinista en la GPU actual. Se utiliza una matriz fija generada en CPU por la interpolación oficial con `align_corners=False`, seguida por multiplicación matricial. Se verifican equivalencia de salidas y gradientes para6/12/24nodos y actualización real en GPU con determinismo estricto.

## Datos y separación temporal

Contrato `config/diseno_entrenamiento_texto.json`:30variables numéricas históricas,23de calendario,30máscaras,clima con latencia efectiva168h. Se reutiliza exactamente `Preparador` de PatchTST: medianas y escalas calculadas solo sobre las filas históricas usadas por optimización de cada fold. La demanda se reconstruye como demanda/nivel en la rama autoregresiva; las otras29variables y30máscaras forman59entradas exógenas históricas; el calendario23entra en pasado y horizonte. No se duplica la demanda como exógena ni se permiten clima/precio/hidrología realizados a futuro. Los orígenes son diarios, a00:00deColombia, con etiquetas de24h completas.

Pliegues expansivos2018–2021, reserva interna final10% y purga24h. Imputación y escala no leen parada interna, validación externa ni prueba. La prueba2022–2025es retrospectiva por exposición previa del benchmark. Se accede a ella solo tras guardar selección y semillas definitivas; no se usa para ajustar candidatos, épocas o presupuesto.

## Búsqueda y entrenamiento

Antes del primer ajuste se congelan20configuraciones y las huellas del código y datos. N01:lr0.001,lote32,dropout0.1,weight_decay0.0001. Las otras19usan RNG20261114:lrlog-uniforme[1e-4,3e-3],lote{16,32,64},dropoutuniforme[0.05,0.30],weight_decaylog-uniforme[1e-6,1e-2]. La geometría permanece fija. No se modifican opciones a partir de resultados ya observados de otros modelos.

Entrenamiento PyTorch GPU/ROCm, precisión completa, determinismo estricto, MAE normalizada por nivel, AdamW, clip1, máximo200épocas, ReduceLROnPlateau(factor0.5,paciencia2). Parada después de mínimo10épocas y10sin mejora significativa `max(1e-6,0.002*referencia)`. Se restaura siempre el mínimo absoluto de pérdida interna.

Filtro2018:20configuraciones×semillas43–45=60ajustes. Retener cinco por menor MASE; confirmación2019–2021:5×3semillas×3años=45. La selección definitiva usa solo2019–2021con igual peso anual; empates por ID. Piloto:semillas100–104×4años=20. Total previo a prueba125. Es una reducción exploratoria respecto a las260evaluaciones previas a prueba del esquema completo de la guía; no se presenta como búsqueda uniforme de20configuraciones en cuatrofolds.

## Dimensionamiento y prueba autorizada

Métrica primaria MASE, escala fija215617.90361589016kWh. Piloto pareado contra ingenuo semanal, bloques temporales circulares14d dentro de cada año, pesos anuales iguales,10000réplicas,α0.05,margen relativo3%,potencia objetivo0.80. Se reutiliza el dimensionamiento exploratorio versionado; cinco semillas piloto no certifican potencia futura.

Antes de leer prueba se fija R según el piloto. Si no es dimensionable, se ejecutan cinco semillas exploratorias1000–1004, conforme a la autorización de entrenamiento y prueba, sin atribuir potencia acreditada. Si se dimensiona, se utilizan Rsemillas1000enadelante. En ambos casos se conserva la limitación retrospectiva y no se modifica la arquitectura tras ver prueba. Se reajusta con historia anterior a2022y parada interna separada; se evalúan ingenuos diario/semanal en los mismos orígenes. Se reportan MAE,RMSE,MASE,sMAPE,MAPEy errores por horizonte, medias y dispersión entre semillas.

## Presupuesto, artefactos y progreso

Límite24horas desde el lanzamiento, persistido en JSON y no renovado al reiniciar. Los ajustes completos se reutilizan; uno incompleto se repite con la misma semilla. Al agotarse el plazo se guardan checkpoint e historial parciales y se informa presupuesto agotado; no se usa el parcial para selección.

Salidas `results/nhits_numerico_r1`:plan,hardware,configuraciones,ranking,selección,piloto,dimensionamiento,plan de prueba,métricas por semilla y resumen. Por ajuste:identidad,escaladores,pesos,historial completo,predicciones y etiquetas por origenUTCy semilla,coste temporal,parámetros y pico de memoriaGPU. Barra local de actualización automática en puerto39973. Se mantiene la instrucción de apagar al finalizar:el monitor espera el cierre, comprueba artefactos y sincroniza los archivos antes del apagado. Sin publicación automática en GitHub.

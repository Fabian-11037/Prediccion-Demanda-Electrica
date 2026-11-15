# TFT compacto: entrenamiento y prueba en cola

14 de noviembre de2026, America/Bogota. Se autoriza preparar el modelo6 y ejecutarlo después del cierre de N-HiTS, incluyendo su prueba. El apagado antes programado al final de N-HiTS se difiere hasta el final de TFT. No se ejecutan ambas corridas simultáneamente.

## Arquitectura y procedencia

Familia Temporal Fusion Transformer, Lim et al., https://arxiv.org/abs/1912.09363. Implementación de referencia Nixtla/neuralforecast, revisión94b6ce131838be38a1fb5157349f02bbc8c89ef6, archivo`neuralforecast/models/tft.py`, licenciaApache2.0. El código deNixtla no se atribuye como implementación original deGoogle. Se conservan archivo completo, componentes y método de propagación sin cambios de lógica en`src/demanda/vendor/tft`.

Adaptación compacta para limitar cómputo:dimensión interna16,cuatro cabezas,una capa LSTM para pasado y una para futuro,selección de variables mediante VSN,redes residuales con compuertas ELU,atención temporal causal y lectura lineal24×1. Sin variables estáticas,se usan contextos e inicializaciones cero de referencia. No hay cabezal adicional. La dimensión16difiere del valor128predeterminado deNixtla; no se afirma réplica literal del artículo,configuración publicada específica ni exención automática de las1000arquitecturas deE2. Es una corrida exploratoria de arquitectura fija conforme a las restricciones de tiempo vigentes.

## Datos y entrenamiento

Mismo contrato `config/diseno_entrenamiento_texto.json` y mismo `Preparador` que N-HiTS/PatchTST:historia168h,horizonte24h,30numéricas históricas,23de calendario y30máscaras. Rama objetivo:demanda histórica/nivel;59exógenas históricas(restantes29numéricas y30máscaras);calendario conocido pasado/futuro. No hay texto ni exógenas realizadas del horizonte. El clima conserva168hde latencia efectiva. Medianas,medias,desviaciones y escala se ajustan solo con optimización de cadafold. La inicialización y pérdida no utilizan etiquetas futuras como entradas.

Entrenador PyTorchGPU/ROCm,determinismo estricto,precisión completa,AdamW,MAEnormalizada,clip1,máximo200épocas,plateau(factor0.5,paciencia2),mínimo10épocas yparada tras10sin mejora significativa `max(1e-6,0.002*referencia)`. Se restaura el mínimo absoluto interno. Antes de entrenar se comprueba forward/backward en GPU con lote64; esa comprobación sintética ocurre solo después de N-HiTS. Si falla,no se cambia silenciosamente arquitectura,determinismo ni dispositivo;se registra el error.

Veinte configuraciones congeladas antes de encolar:T01lr0.001,lote32,dropout0.1,weight_decay0.0001. Las otras19usan RNG2026111406:lrlog-uniforme[1e-4,3e-3],lote{16,32,64},dropoutuniforme[0.05,0.30],weight_decaylog-uniforme[1e-6,1e-2]. El mismo dropout se aplica a VSN/GRN y atención. No se selecciona capacidad mediante los resultados de prueba de otros modelos.

## Selección y prueba

Filtro2018:20configuraciones×semillas43–45=60ajustes. Cinco finalistas×3semillas×2019–2021=45. Selección definitiva por menor MASE medio solo en2019–2021,igual peso anual,desempate porID. Piloto:5semillas100–104×4años=20. Total125antes de prueba. Es una adaptación exploratoria del presupuesto completo260+Rde la guía. Reserva interna cronológica10% ypurga24h en todos los ajustes.

Dimensionamiento exploratorio frente al ingenuo semanal:δrel3%,α0.05,potenciaobjetivo0.80,10000remuestreos en bloques14d dentro de año,igual peso anual. Si el piloto dimensiona R,se fijan semillas1000en adelante;si no,se usan cinco exploratorias1000–1004,como política registrada antes de ver prueba. No se acredita potencia futura ni se convierte ausencia de significación en equivalencia.

La selección y semillas se guardan antes del primer acceso a prueba2022–2025. Reajuste anterior a2022con parada interna separada;prueba retrospectiva por exposición previa del benchmark. MASEcon escala fija215617.90361589016kWh,MAE,RMSE,sMAPE,MAPE y errores por horizonte;referencias diaria y semanal en los mismos orígenes. Se conserva el resultado de todas las semillas,no se elige la mejor mediante prueba.

## Cola, presupuesto y artefactos

`scripts/encadenar_nhits_tft.py` espera el cierre del proceso N-HiTS y un estado terminal válido;comprueba sus artefactos ysincroniza archivos antes de lanzar TFT. Si N-HiTS falla o desaparece sin cierre válido,la cola se detiene yconserva el diagnóstico. La cola tiene bloqueo exclusivo yno requiere mantener abierto este chat.

Resultados en`results/tft_numerico_r1`,barra puerto39974. Mientras espera,se muestra estadoEncola. El límite persistente24hse inicia al comenzar TFT,excluyendo espera;no se renueva al reanudar. Al agotarse se guardan parciales yno se declaran completos los pendientes. Se conservan pesos,escaladores,historiales,predicciones por origen ysemilla,métricas,costes,memoria yhuellas de código/datos. El monitor de apagado se activa para TFT después de confirmar el inicio;espera su cierre,comprueba resultados ysincroniza antes de apagar. No hay publicación automática.

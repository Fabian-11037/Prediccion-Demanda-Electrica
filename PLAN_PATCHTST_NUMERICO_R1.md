# PatchTST numérico · ruta 1 · 10 de noviembre de 2026

## Autorización y alcance

Se autoriza iniciar PatchTST por la ruta de arquitectura fija con búsqueda de hiperparámetros. Se preservan todos los resultados anteriores y los baselines permanecen en espera. El contrato numérico es `config/diseno_entrenamiento_texto.json`:30variables históricas y23de calendario,168h de historia,24h de horizonte,clima con rezago efectivo168h,sin ISE,pronósticos emitidos ni ponderaciones poblacionales como entradas comunes. Se agregan30máscaras de ausencia:83canales históricos en total. El escenario es numérico N, sin encoder textual.

## Ficha de arquitectura y adaptación

Fuente primaria: Nie,Nguyen,Sinthong yKalagnanam,ICLR2023, https://arxiv.org/abs/2211.14730 . Implementación oficialApache2.0, repositorio `yuqinie98/PatchTST`, revisión `204c21efe0b39603ad6e2ca640ef5896646ab1a9`. Configuración concreta: `PatchTST_supervised/scripts/PatchTST/etth1.sh`:3capas,4cabezas,d_model16,d_ff128,parche16,paso8,dropout0.3. Se adoptan geometría del núcleo,BatchNorm,atención residual,GELU,posición aprendida,RevINsin affine,headflatten compartido ypadding por repetición al final. Las dimensiones se adaptan de336horas ylos horizontesETT a168/24del SIN.

El núcleo original es independiente por canal. Seleccionar únicamente su salida de demanda ignoraría la información exógena del origen. Se conserva el núcleo oficial para los83canales y se añade una lectura afín compartida por horizonte sobre sus83salidas y23variables de calendario futuro;produce una corrección alrededor de1en escala de nivel. No tiene capas ocultas ni anchos sometidos a búsqueda. Demanda histórica: cociente respecto al nivel,centrado ydesviación estimada con entrenamiento. La lectura se inicializa cerca de una predicción de demanda,con pesos pequeños no nulos para todas las entradas;no se selecciona usando prueba.

Es **PatchTST adaptado**,no una réplica exacta deETTh1 ni prueba automática de la exenciónE2§14para un cabezal propio. La ruta fija se ejecuta por solicitud expresa;su uso para acreditar el requisito académico debe revisarse por separado. No se afirma que la configuración completa adaptada haya sido publicada por los autores. El código oficial yla licencia se conservan en `src/demanda/vendor/patchtst`,con huellas yregistro de cambios.

## Preparación y búsqueda

El panel congelado conserva suSHA-256. Se determinan orígenes elegibles solo por demanda necesaria y etiquetas completas, sin descartar días por ausencia exógena. Cada pliegue separa ajuste,último10%para parada yvalidación anual,purgando cruces de etiquetas24h. Medianas,medias,desviaciones yescala de demanda se ajustan solo sobre la unión exacta de filas históricas usadas por las ventanas de entrenamiento. Se conservan máscaras;una columna no estimable se representa mediante cero yse registra. Las covariables usan sus propias escalas;calendario futuro conocido nunca contiene clima,precio ohidrología realizados en el horizonte.

20configuraciones fijas antes del primer ajuste,semillas43/44/45 ycuatro pliegues2018–2021:240ajustes. P01mantiene lr0.0001,dropout0.3yusa lote32/weight_decay0.0001;el lote difiere del128del script original por las83variables yla GPU disponible. P02–P20se muestrean con semilla20261110:lrlog-uniforme[1e-4,3e-3],lote{16,32,64},dropoutuniforme[0.05,0.3],weight_decaylog-uniforme[1e-6,1e-2]. Geometría ylectura afín permanecen fijas. El planJSONy las huellas se escriben antes de entrenar.

EntrenamientoGPUROCm,precisión completa,determinismo estricto,AdamW,MAEnormalizada,clip1,máximo200épocas,plateau(factor0.5,paciencia2). Parada temprana:mínimo10épocas,10sin mejora significativa max(1e-6,0.002×pérdida_referencia);se restaura siempre el mínimo absoluto. Se registran curvas ycostes. Los fallos técnicos se conservan en el log;no se cambia silenciosamente lote ni candidatos. Reanudar recupera ajustes completos yrepite desde cero los incompletos con las mismas semillas yfuentes. Selección:menorMASEmedio de las12combinaciones semilla/pliegue,igualpeso anual;empates por identificador.

## Piloto, dimensionamiento y prueba

Configuración elegida:semillas100–104en los cuatro pliegues:20ajustes adicionales. Se conservan errores diarios apareados contra el ingenuo semanal sobre los mismos orígenes. Contraste primario exploratorio:PatchTSTadaptado frente al ingenuo semanal;referencia paraδ=3%es elMASEdel ingenuo semanal de validación. α0.05bilateral,potenciaobjetivo0.80. No se comparan modelos numéricos con distinta información como si fueran controles ampliados equivalentes.

Dimensionamiento antes de prueba:10000remuestreos circulares de bloques14d dentro de cada año,con peso anual uniforme ylos mismos índices temporales para semillas. Se remuestrean semillas pareadas mediante conteos multinomiales para cadaR;se estima potencia bajo una diferencia mínima±3%usando distribución bootstrap centrada yregión nula bilateral. Se utiliza la menor potencia de ambos signos. Primero se evalúa el límite temporal con semillas promediadas:si no alcanza0.8,más inicializaciones no reparan la falta de precisión temporal yno se ejecuta automáticamente prueba. En caso contrario se busca el menorRque alcance0.8en la rejilla5–100,de10en10hasta500yde50en50hasta2000. Si no se logra,se registra `Requiere decisión`,sin inventar potencia ni imponer cinco semillas finales por defecto. La estimación depende de solo cinco semillas piloto yno certifica potencia futura;no equivale a una garantía. La inferencia entre semillas yla temporal se distinguen.

Tras cerrar selección yR,se reentrena usando historia anterior a2022con reserva interna separada;semillas1000en adelante. Se evalúa una vez por semilla en el bloque2022–2025,con pesos fijos yventanas actualizadas por origen. Es una reevaluación retrospectiva por exposición previa,no una prueba confirmatoria independiente. El denominadorMASEes215617.90361589016kWh. Se reportanMASE,MAE,RMSE,sMAPEyerrores por horizonte;no se inicia búsqueda nueva según esos resultados.

## Artefactos y progreso

Carpeta nueva `results/patchtst_numerico_r1`:planconhuellas,hardware,estado ybarraHTML. Por ajuste:configuración/semilla/pliegue,escaladores/medianas/máscaras,noestimables,curvas,pesos,predicciones con origenUTCns,etiquetas ymétricas. Se conserva el código de reconstrucción. La evaluación final depende de un dimensionamiento válido;se distingueCompletadodeRequiere decisión. No hay apagado automático. No se inicia publicación delJBook ni cálculo de texto.


## Límite temporal

Se conserva el límite de24horas solicitado para esta fase exploratoria. El plazo se inicia al lanzar la ejecución, se persiste y no se renueva al reiniciar. Al agotarse se guardan los ajustes completos y el estado parcial del ajuste activo, que no se utiliza para seleccionar. No se recortan candidatos ni semillas silenciosamente para aparentar protocolo completo. Reanudar después del plazo exige una instrucción del autor. El tiempo medido en una prueba técnica GPU no acredita que todas las etapas quepan en ese plazo.

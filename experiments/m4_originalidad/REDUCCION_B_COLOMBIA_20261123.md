# Reducción operativa B: Colombia y ablaciones

Fecha: 23 de noviembre de 2026. Decisión del usuario: opción B, con presupuesto
orientativo de 4–8 horas desde el cambio. Esta decisión ocurre tras inspección
de la prueba colombiana; no constituye un nuevo preregistro confirmatorio.

Se sustituye la cola extensa por una evaluación retrospectiva de Colombia con
diez semillas 101–110. Se preservan todos los ajustes completos, checkpoints,
registros y diseños anteriores. UCI, GEFCom, OPSD y los nuevos ajustes del
Top10 quedan diferidos. El ajuste TFT interrumpido conserva su checkpoint.

## Procedimientos y ajuste

Se conservan M1, M4_N, MLP_CE, EWMA_MLP_CE, memoria fija, sin compuerta,
recuperación uniforme y por similitud, NNLS, EWMA, ridge y kNN residuales,
y las adaptaciones ya especificadas de RATL y FreshCast. Estas últimas no
se presentan como réplicas oficiales.

Se añaden las ablaciones de residuos en muestra y OOF temporal de configuración
fija, frente al OOF por selección anidada. Solo se sustituye el banco histórico
del tramo de ajuste; la base de consulta conserva el procedimiento original.
Se incluyen M4_O, M4_OP, metadatos, cinco desplazamientos textuales de
49/91/154/217/301 días, disponibilidad estricta y ausencia de entrada numérica.

La configuración 6 permanece fija. No se buscan arquitecturas ni se eligen
variantes mediante prueba. Las ablaciones nuevas solo reciben el ajuste final
por semilla: optimización anterior a 2021, reserva interna de 2021 con purga,
y prueba 2022–2025. Su validación externa ampliada queda diferida; los resultados
OOF existentes se preservan sin presentarlos como una validación nueva.
AdamW, regularización y parada interna mantienen los protocolos registrados.
Diez semillas es un tope operativo; no se amplía automáticamente por potencia.

La inferencia original de TimesNet y N-BEATS se obtiene de sus cinco pesos
1000–1004, sin optimizadores ni actualización de pesos, con verificación contra
resultados canónicos y tolerancias ya registradas. Los otros ocho procedimientos
históricos conservan sus pérdidas publicadas. No se fabrican predicciones ni
réplicas a partir de agregados. El Top10 histórico conserva cinco semillas.

## Inferencia, publicación y apagado

MASE y escala 215617,90361589016 kWh permanecen fijos. La unidad temporal es el
origen diario; se promedian pérdidas entre semillas. Se conservan 49.999 réplicas
de bootstrap estacionario, longitud esperada de bloque 14, DM–HLN con HAC 13,
GW bilateral con instrumento observable de dos días antes y Holm con cota
familiar 1.536. No se reduce la multiplicidad por eliminar tareas. No se afirma
potencia 0,90, originalidad, confirmación prospectiva ni cierre completo del E3.

Un solo proceso computacional usa la GPU a la vez. La barra existente informa
bloques operativos, no porcentaje de tiempo. Se publican predicciones por
fecha/semilla, historiales, pesos seleccionados, escaladores, métricas y
manifiestos; los optimizadores parciales quedan locales. El apagado se solicita
solo tras verificar ambas publicaciones en GitHub. Cualquier error mantiene
el equipo encendido y registra el fallo.

Ejecutor: `scripts/m4_cierre_reducido_b.py`. Artefactos:
`results/m4_cierre_gpu_20261123/reducido_B/`. El registro JSON y la etiqueta Git
se guardan antes de iniciar la nueva cola. Las comprobaciones de implementación
y las pruebas del repositorio se realizan una vez antes de registrar el código;
se mantienen las comprobaciones mínimas de integridad y publicación final.

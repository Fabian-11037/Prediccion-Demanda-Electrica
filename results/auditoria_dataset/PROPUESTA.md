# Propuesta de búsqueda rápida

20 configuraciones por familia (40 en total), detalladas en `configuraciones_rapidas.csv` y `plan_rapido.json`. No se inició entrenamiento. La propuesta requiere implementar las correcciones y adaptaciones; no debe pasarse sin cambios al runner original.

Cada una de cinco estructuras se combina con historias de72/168horas y tasas0.0003/0.001: cuatro configuraciones por estructura. LSTM:1×16,1×32,1×64,2×16,2×32. CNN:1×16 k3,1×16 k5,1×32 k3,1×32 k5,2×16 k3. EnCNN,se propone pooling local de factor2seguido de flatten,capa de32unidades ysalida24; conserva36/84posiciones en vez de promediar días completos. No es el pooling del modelo anterior.

Comunes:AdamW,lote256,Dropout0.05,weight_decay1e-5,Huber(delta0.05),clip1,plateau(factor0.5,paciencia2),máximo60épocas,mínimo10,parada tras6épocas sin mejora superior a max(1e-6,0.002×mejor_pérdida). La mejora se mide sobre pérdida deES;selección por MASE anual. Registrar también MAE/sMAPE,horas de entrenamiento y sensibilidad a permutar demanda.

Filtro:40ajustes con semilla42,pliegue2019,máximo30épocas. Confirmación:3finalistas por familia,cuatro pliegues,dos semillas nuevas43/44,máximo60épocas:48ajustes. Total88. El filtro de un año puede descartar candidatos útiles en otros regímenes;por eso no se reporta como equivalente a la búsqueda exhaustiva anterior. El tiempo total requiere medir candidatos piloto;no se promete una duración sin medición.

Precondición:escaladores por fold sinES,demanda estandarizada con entrenamiento ycomprobaciones de sensibilidad. Concatenar el calendario futuro a la representación de historia antes de una cabeza pequeña,sin rama densa separada de calendario. La arquitectura y las intervenciones se documentan antes de entrenar. El test2022–2025ya fue observado;esta fase es exploratoria yno se repite prueba como confirmación.

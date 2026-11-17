# Reevaluación exploratoria v3

Autorizada el 8 de noviembre de 2026 para corregir el preprocesamiento y repetir entrenamiento y prueba. El periodo 2022–2025 ya fue observado: esta reevaluación no constituye confirmación independiente. Se conservan sin modificación los resultados v2.

Se repiten ARX y MLP con sus veinte configuraciones originalmente muestreadas. LSTM y CNN1D usan veinte variantes pequeñas por familia: una capa, LSTM16/32unidades yCNN8/16filtros,kernels3/5;historia72/168horas;lr0.0003/0.001;Dropout0.1/0.2/0.3;weight_decay0.0001/0.001. Las veinte combinaciones exactas se congelan en `results/base_v3/configuraciones.json` antes del primer ajuste.

Escaladores por pliegue ajustados sólo a filas históricas de entrenamiento, sinES ni validación. Demanda normalizada por nivel y, en secuenciales, centrada y estandarizada con entrenamiento. Etiquetas y su inversión permanecen y/nivel y multiplicación por nivel. MASE conserva el denominador fijo de referencia2000–2017 para comparabilidad;es un componente de métrica,no una entrada de optimización. Se declara por separado de la corrección del escalado de características.

LSTM usa último estado,Dropout ysalida lineal. CNN usaConv1d,ReLU,Dropout1d,pooling localfactor2,flatten,Dropout ysalida lineal. Calendario futuro concatenado antes de salida,sin rama densa separada. Salida residual alrededor de1en escala de nivel. No se imponen precios ni restricciones ajenas al objetivo de demanda.

Presupuesto acotado: filtro20configuraciones por familia,pliegue2019,semilla42,máximo30épocas (80ajustes). Tres finalistas por familia se comparan en cuatro pliegues2018–2021 ysemillas43/44,máximo60épocas (96ajustes). Selección por MASE medio. Piloto de la configuración elegida:semillas100–104,cuatro pliegues (80ajustes). Prueba:5semillas1000–1004por familia (20ajustes);no se dimensiona a195semillas yno se afirma potencia para detectar3%. Total276ajustes. Las20configuraciones originales ARX/MLP conservan sus hiperparámetros;el presupuesto yla parada cambian para esta fase.

Entrenador común:AdamW,clip1,precisión completa,determinismo estricto,mínimo10épocas,patiencia6sobreES,mejora significativa max(1e-6,0.002×pérdida_referencia). Se restaura el estado de menor pérdida absoluta, aunque la paciencia sólo se reinicia ante mejora significativa. Validación por MASE;lossHuber(delta0.05)para secuenciales. Se conservan las pérdidas yplanificadores originales en ARX/MLP;secuenciales usanplateau(factor0.5,paciencia2).

Se guardan por ajuste pesos,escaladores,curvas,predicciones con origen temporal ysemilla,métricas,huellas yestado de progreso. Se entrega resumen de validación,piloto,prueba,referencias ycomparación descriptiva conv2. No se presupone mejora. No se ejecuta apagado automático en esta fase antes de entregar el resumen.

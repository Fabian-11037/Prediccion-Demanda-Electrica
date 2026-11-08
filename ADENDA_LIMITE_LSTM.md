# Reducción de la búsqueda LSTM por costo computacional

Por instrucción del autor se completará la quinta configuración LSTM, activa al tomar la decisión, con sus tres semillas y cuatro pliegues. Se omiten las configuraciones sexta a vigésima y se continúa con las veinte configuraciones CNN1D. Se conservan todos los resultados obtenidos y los hiperparámetros muestreados; no se modifican pérdidas, datos, particiones ni criterios de parada.

La búsqueda queda en 780 ajustes: 240 por cada familia ARX, MLP y CNN1D, y 60 de LSTM. Los presupuestos de selección son desiguales; esta limitación debe declararse en las comparaciones. La decisión se toma durante la búsqueda, con resultados parciales de validación disponibles, por costo computacional. No constituye el presupuesto originalmente preregistrado.

Piloto y evaluación final continúan para las cuatro familias con las opciones vigentes. El apagado automático permanece condicionado a la finalización completa y al guardado de resultados.

# Selección escalonada PatchTST autorizada durante búsqueda

El10de noviembre de2026el usuario autoriza la alternativa recomendada para reducir el tiempo. La decisión se toma durante la búsqueda,con34ajustes completos de2018yuna ejecución activaP12/semilla44;los resultados parciales de validación ya estaban disponibles. No es el presupuesto original yno se retrofecha.

1. Completar20configuraciones×3semillas(43,44,45)en2018:60ajustes.
2. Retener cinco por menorMASEmedio en2018;empates por identificador. Las otras quince no se evalúan en los años posteriores.
3. Confirmar las cinco en2019–2021con las mismas tres semillas:45ajustes. La configuración definitiva se elige únicamente por media de estos tres años ysemillas,con peso anual uniforme;2018sirve para el filtro yno vuelve a sumarse a la selección definitiva.
4. Mantener el piloto de la configuración elegida:5semillas×4años,20ajustes. Reportar que2018participó en el filtro.
5. DimensionarRyreevaluar prueba con el procedimiento existente,sin cambios. Total125+R.

No se modifican arquitectura,configuraciones,semillas,datos,preprocesamiento,pérdida,planificador ni parada. Se conservan cachés yartefactos completos. El controlador termina el ajuste activo antes de reiniciar con el nuevo plan;puede quedar un inicio descartado del siguiente ajuste por el intervalo de observación. No se cuenta como completado yse recupera con la misma semilla. El límite24horas conserva exactamente el plazo inicial;no se renueva.

El plan ylas fuentes previas quedan archivados en `results/patchtst_numerico_r1/fuentes_antes_escalonado/`. La migración explícita valida que solo cambie elrunner yse añadan la política ysu documentación;los hashes de datos,arquitectura,preparador yentrenador deben permanecer iguales. La barra cambia a125ajustes anteriores aprueba. Es selección exploratoria adaptativa,con menor exhaustividad;no se presenta como la búsqueda uniforme de20configuraciones en cuatropliegues ni comocumplimiento automático delpresupuesto académico original.

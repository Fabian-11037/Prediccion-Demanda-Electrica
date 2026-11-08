# Diccionario de datos

Cada fila del panel representa un intervalo horario del SIN. Las variables se documentan por su significado físico u operativo, unidad y disponibilidad temporal. Las categorías del calendario describen la codificación implementada.

Las columnas `x_*` incorporan los rezagos del panel y se consultan únicamente en filas anteriores al origen. Una fila con índice `t` puede contener información disponible hasta `t + 1 h`; el índice no es la fecha de publicación. Los rezagos de disponibilidad son supuestos del protocolo. Las columnas `cal_*` pueden consultarse en el horizonte; `aux_*` se reservan para auditoría.

## Panel horario

### Índice y objetivo

| Variable | Significado | Unidad |
| --- | --- | --- |
| y_demanda_real_kwh | Energía eléctrica demandada por el SIN durante el intervalo horario objetivo. | kWh/intervalo horario |
| inicio | Inicio del intervalo horario al que corresponde la fila; índice temporal del panel. | fecha-hora America/Bogota |

### Demanda y precio históricos

| Variable | Significado | Unidad |
| --- | --- | --- |
| x_demanda_real_kwh | Demanda real horaria del SIN desplazada una fila horaria respecto de la serie original. | kWh/intervalo horario |
| x_precio_bolsa_cop_kwh | Precio horario de la energía en la bolsa de XM, desplazado 24 filas horarias respecto de la serie original. | COP/kWh |

### Meteorología

| Variable | Significado | Unidad |
| --- | --- | --- |
| x_obs_temperature_2m_c_bogota | Temperatura del aire a 2 m sobre la superficie para el punto de malla asociado a Bogotá. Reanálisis observado con desplazamiento de una fila horaria. | °C |
| x_obs_temperature_2m_c_medellin | Temperatura del aire a 2 m sobre la superficie para el punto de malla asociado a Medellín. Reanálisis observado con desplazamiento de una fila horaria. | °C |
| x_obs_temperature_2m_c_cali | Temperatura del aire a 2 m sobre la superficie para el punto de malla asociado a Cali. Reanálisis observado con desplazamiento de una fila horaria. | °C |
| x_obs_temperature_2m_c_barranquilla | Temperatura del aire a 2 m sobre la superficie para el punto de malla asociado a Barranquilla. Reanálisis observado con desplazamiento de una fila horaria. | °C |
| x_obs_temperature_2m_c_cartagena | Temperatura del aire a 2 m sobre la superficie para el punto de malla asociado a Cartagena. Reanálisis observado con desplazamiento de una fila horaria. | °C |
| x_obs_temperature_2m_c_bucaramanga | Temperatura del aire a 2 m sobre la superficie para el punto de malla asociado a Bucaramanga. Reanálisis observado con desplazamiento de una fila horaria. | °C |
| x_obs_temperature_2m_c_pereira | Temperatura del aire a 2 m sobre la superficie para el punto de malla asociado a Pereira. Reanálisis observado con desplazamiento de una fila horaria. | °C |
| x_obs_temperature_2m_c_cucuta | Temperatura del aire a 2 m sobre la superficie para el punto de malla asociado a Cúcuta. Reanálisis observado con desplazamiento de una fila horaria. | °C |
| x_obs_relative_humidity_2m_pct_bogota | Humedad relativa del aire a 2 m sobre la superficie para el punto de malla asociado a Bogotá. Reanálisis observado con desplazamiento de una fila horaria. | % |
| x_obs_relative_humidity_2m_pct_medellin | Humedad relativa del aire a 2 m sobre la superficie para el punto de malla asociado a Medellín. Reanálisis observado con desplazamiento de una fila horaria. | % |
| x_obs_relative_humidity_2m_pct_cali | Humedad relativa del aire a 2 m sobre la superficie para el punto de malla asociado a Cali. Reanálisis observado con desplazamiento de una fila horaria. | % |
| x_obs_relative_humidity_2m_pct_barranquilla | Humedad relativa del aire a 2 m sobre la superficie para el punto de malla asociado a Barranquilla. Reanálisis observado con desplazamiento de una fila horaria. | % |
| x_obs_relative_humidity_2m_pct_cartagena | Humedad relativa del aire a 2 m sobre la superficie para el punto de malla asociado a Cartagena. Reanálisis observado con desplazamiento de una fila horaria. | % |
| x_obs_relative_humidity_2m_pct_bucaramanga | Humedad relativa del aire a 2 m sobre la superficie para el punto de malla asociado a Bucaramanga. Reanálisis observado con desplazamiento de una fila horaria. | % |
| x_obs_relative_humidity_2m_pct_pereira | Humedad relativa del aire a 2 m sobre la superficie para el punto de malla asociado a Pereira. Reanálisis observado con desplazamiento de una fila horaria. | % |
| x_obs_relative_humidity_2m_pct_cucuta | Humedad relativa del aire a 2 m sobre la superficie para el punto de malla asociado a Cúcuta. Reanálisis observado con desplazamiento de una fila horaria. | % |
| x_obs_precipitation_preceding_hour_mm_bogota | Precipitación acumulada durante la hora precedente a la marca meteorológica para el punto de malla asociado a Bogotá. Reanálisis observado con desplazamiento de una fila horaria. | mm/hora precedente |
| x_obs_precipitation_preceding_hour_mm_medellin | Precipitación acumulada durante la hora precedente a la marca meteorológica para el punto de malla asociado a Medellín. Reanálisis observado con desplazamiento de una fila horaria. | mm/hora precedente |
| x_obs_precipitation_preceding_hour_mm_cali | Precipitación acumulada durante la hora precedente a la marca meteorológica para el punto de malla asociado a Cali. Reanálisis observado con desplazamiento de una fila horaria. | mm/hora precedente |
| x_obs_precipitation_preceding_hour_mm_barranquilla | Precipitación acumulada durante la hora precedente a la marca meteorológica para el punto de malla asociado a Barranquilla. Reanálisis observado con desplazamiento de una fila horaria. | mm/hora precedente |
| x_obs_precipitation_preceding_hour_mm_cartagena | Precipitación acumulada durante la hora precedente a la marca meteorológica para el punto de malla asociado a Cartagena. Reanálisis observado con desplazamiento de una fila horaria. | mm/hora precedente |
| x_obs_precipitation_preceding_hour_mm_bucaramanga | Precipitación acumulada durante la hora precedente a la marca meteorológica para el punto de malla asociado a Bucaramanga. Reanálisis observado con desplazamiento de una fila horaria. | mm/hora precedente |
| x_obs_precipitation_preceding_hour_mm_pereira | Precipitación acumulada durante la hora precedente a la marca meteorológica para el punto de malla asociado a Pereira. Reanálisis observado con desplazamiento de una fila horaria. | mm/hora precedente |
| x_obs_precipitation_preceding_hour_mm_cucuta | Precipitación acumulada durante la hora precedente a la marca meteorológica para el punto de malla asociado a Cúcuta. Reanálisis observado con desplazamiento de una fila horaria. | mm/hora precedente |
| x_obs_temperature_2m_c_ponderada | Temperatura del aire a 2 m sobre la superficie; promedio de las ocho ciudades ponderado por población estática del censo 2018. Reanálisis observado con desplazamiento de una fila horaria. | °C |
| x_obs_relative_humidity_2m_pct_ponderada | Humedad relativa del aire a 2 m sobre la superficie; promedio de las ocho ciudades ponderado por población estática del censo 2018. Reanálisis observado con desplazamiento de una fila horaria. | % |
| x_obs_precipitation_preceding_hour_mm_ponderada | Precipitación acumulada durante la hora precedente a la marca meteorológica; promedio de las ocho ciudades ponderado por población estática del censo 2018. Reanálisis observado con desplazamiento de una fila horaria. | mm/hora precedente |

### Hidrología y auditoría

| Variable | Significado | Unidad |
| --- | --- | --- |
| x_hid_llenado_valor_original | Indicador de llenado conservado en su escala original, correspondiente a la razón entre reserva útil y capacidad útil. No se multiplica por 100 ni se recortan valores superiores a 1. | razón adimensional |
| x_hid_reserva_util_kwh | Energía equivalente almacenada en la reserva útil de los embalses del SIN para la fecha hidráulica disponible. | kWh |
| x_hid_capacidad_util_kwh | Capacidad útil de almacenamiento de los embalses del SIN, expresada como energía equivalente. | kWh |
| x_hid_aportes_kwh_dia | Aportes hídricos diarios al SIN expresados como energía equivalente de los caudales recibidos. | kWh/día |
| aux_hid_fecha_dato | Fecha del dato hidrológico diario seleccionado por la unión temporal; permite auditar su antigüedad. | fecha |

### Calendario

| Variable | Significado | Unidad |
| --- | --- | --- |
| cal_dow | Día de la semana: 0 lunes, 1 martes, 2 miércoles, 3 jueves, 4 viernes, 5 sábado y 6 domingo. | categoría 0–6 |
| cal_mes | Mes del año: 1 enero hasta 12 diciembre. | categoría 1–12 |
| cal_dia_anio | Posición del día dentro del año; el 29 de febrero se conserva en años bisiestos. | día 1–366 |
| cal_es_festivo | Indica si la fecha corresponde a un festivo del calendario colombiano implementado. | 0/1 |
| cal_es_domingo | Indica si la fecha es domingo. | 0/1 |
| cal_es_sabado | Indica si la fecha es sábado, sea o no festivo. | 0/1 |
| cal_es_no_laboral | Indica domingo o festivo. Un sábado no festivo toma valor 0 en esta definición operativa. | 0/1 |
| cal_tipo_dia | Tipo de día: 0 lunes a viernes no festivo; 1 sábado no festivo; 2 domingo o festivo. | categoría 0–2 |
| cal_dias_al_siguiente_festivo | Días hasta el festivo estrictamente posterior a la fecha actual; distancias mayores de 30 se truncan a 30. | días |
| cal_dias_desde_festivo_previo | Días transcurridos desde el festivo estrictamente anterior a la fecha actual; distancias mayores de 30 se truncan a 30. | días |
| cal_es_vispera_festivo | Indica que el día siguiente es festivo. | 0/1 |
| cal_es_dia_post_festivo | Indica que el día anterior es festivo. | 0/1 |
| cal_es_fin_semana_largo | Marca viernes, sábado, domingo y lunes del bloque cuyo viernes o lunes es festivo. | 0/1 |
| cal_es_semana_santa | Indica pertenencia al intervalo entre Domingo de Ramos y Domingo de Pascua, ambos incluidos. | 0/1 |
| cal_es_temporada_navidena | Indica pertenencia al periodo del 16 de diciembre al 6 de enero, ambos incluidos. | 0/1 |
| cal_es_fin_de_anio | Indica pertenencia al periodo del 24 de diciembre al 1 de enero, ambos incluidos. | 0/1 |
| cal_hora | Hora local de inicio del intervalo horario en America/Bogota. | hora 0–23 |
| cal_hora_sin | Codificación cíclica de la hora local: seno(2π × hora / 24). | adimensional [−1, 1] |
| cal_hora_cos | Codificación cíclica de la hora local: coseno(2π × hora / 24). | adimensional [−1, 1] |
| cal_dow_sin | Codificación cíclica del día de la semana: seno(2π × dow / 7). | adimensional [−1, 1] |
| cal_dow_cos | Codificación cíclica del día de la semana: coseno(2π × dow / 7). | adimensional [−1, 1] |
| cal_doy_sin | Codificación cíclica del día del año: seno(2π × dia_anio / 366). | adimensional [−1, 1] |
| cal_doy_cos | Codificación cíclica del día del año: coseno(2π × dia_anio / 366). | adimensional [−1, 1] |

## Disponibilidad y fuentes

| Familia | Fuente | Regla aplicada |
| --- | --- | --- |
| Demanda real | XM | Objetivo horario; x_demanda desplazada 1 h. |
| Precio de bolsa | XM | Precio desplazado 24 h; sólo historia anterior al origen. |
| Meteorología | ERA5 vía Open-Meteo | Reanálisis observado desplazado 1 h; no es pronóstico meteorológico. |
| Hidrología | XM diario | Dato diario disponible tras el cierre + 24 h; unión temporal hacia atrás. |
| Calendario | Calendario colombiano implementado | Conocido para el horizonte. |

La demanda comercial se excluye como entrada por su relación con la demanda real y su liquidación posterior. El precio contemporáneo al objetivo y el clima observado del horizonte no son entradas conocidas al origen.

## Corpus de titulares

El contenido analítico es `titular`. Identificadores, fuentes, fechas, URLs y evidencias permiten seleccionar registros y auditar cobertura; no son atributos textuales del pronóstico.

| Campo | Significado |
| --- | --- |
| id | Identificador del registro documental para trazabilidad y deduplicación. |
| source | Medio de comunicación o entidad que publica el titular. |
| source_kind | Tipo de fuente registrado en el corpus, por ejemplo prensa o entidad institucional. |
| title_kind | Tipo de pieza o encabezado registrado durante la recopilación. |
| titular | Encabezado de la noticia o informe; contenido textual empleado en la exploración léxica. |
| url | Dirección de la pieza original, utilizada para trazabilidad. |
| publication_date_claimed | Fecha de publicación declarada por la fuente; no acredita por sí sola la disponibilidad de esa versión. |
| published_at_claimed | Marca temporal de publicación declarada, con la precisión conservada por el archivo. |
| date_precision | Precisión temporal de la fecha declarada, por ejemplo fecha diaria o marca con hora. |
| availability_status | Estado de verificación de la disponibilidad histórica de la versión del titular. |
| available_at_verified | Instante desde el que la versión archivada tiene disponibilidad verificable; determina el escenario estricto. |
| capture_at_verified | Instante de captura archivística verificada que respalda la versión documental. |
| availability_proof | Tipo de evidencia de disponibilidad; el escenario estricto exige wayback_payload_digest_and_capture. |
| available_at_assumed | Disponibilidad asignada en el escenario declarado: se conserva la hora declarada cuando existe; con fecha sin hora se usa el día siguiente a las 00:00 de Colombia. |
| assumed_availability_basis | Regla o evidencia utilizada para asignar available_at_assumed. |
| quarantined | Indicador de cuarentena; los registros con valor verdadero se excluyen de los escenarios de análisis. |
| quarantine_reason | Motivo documentado de la cuarentena del registro. |
| provenance_source | Archivo o proceso de recuperación del que procede el registro. |

## Inventario de ciudades

| Campo | Significado | Unidad |
| --- | --- | --- |
| ciudad | Identificador de la ciudad utilizado en los nombres de las columnas meteorológicas. | texto |
| lat_solicitada | Latitud geográfica solicitada para consultar la serie de la ciudad. | grados decimales |
| lon_solicitada | Longitud geográfica solicitada para consultar la serie de la ciudad. | grados decimales |
| lat_malla | Latitud del punto de malla efectivamente utilizado por el proveedor meteorológico. | grados decimales |
| lon_malla | Longitud del punto de malla efectivamente utilizado por el proveedor meteorológico. | grados decimales |
| elevacion_m | Elevación reportada para el punto de consulta meteorológica. | m |
| habitantes_censo_2018_ajustado | Población del censo 2018 ajustada por omisión censal; determina los pesos urbanos estáticos. La cifra procede de una fuente secundaria que cita al DANE. | habitantes |

El diccionario estructurado se conserva en `data/processed/diccionario_datos.csv` y `data/processed/diccionario_datos.json`. Su cobertura se verifica contra los esquemas de los archivos, sin consultar valores del conjunto de prueba.

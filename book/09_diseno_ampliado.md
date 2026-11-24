(diseno-ampliado)=
# 9.1. Datos, modelos y protocolo temporal

El estudio evalúa pronósticos diarios de 24 horas con 168 horas de historia. El benchmark proporciona las referencias numéricas y documentales; M4<sub>N</sub> estudia la recuperación de errores históricos de una combinación congelada de expertos.

## Contrato de entradas

| Bloque | Variables | Cantidad | Disponibilidad |
|---|---|---:|---|
| Demanda | Demanda real del SIN, kWh por intervalo | 1 | Historia; rezago de una hora |
| Clima urbano | Temperatura, humedad relativa y precipitación en Bogotá, Medellín, Cali, Barranquilla, Cartagena, Bucaramanga, Pereira y Cúcuta | 24 | Historia; rezago efectivo de 168 horas |
| Hidrología | Llenado, reserva útil, capacidad útil y aportes del SIN | 4 | Habilitación 24 horas después del cierre diario |
| Precio | Precio nacional de bolsa, COP/kWh | 1 | Historia; rezago de 24 horas |
| Calendario | Hora, día de semana, mes, ciclo anual, festivos y temporadas | 23 | Conocido en el origen y en el horizonte |

Las arquitecturas ampliadas utilizan las 30 variables históricas y el calendario. MLP, LSTM, CNN1D y ARX constituyen el grupo de demanda y calendario, con su especificación propia. TaTS, fusión tardía y TGForecaster incorporan documentos oficiales; sus controles numéricos permiten examinar el comportamiento sin contenido documental. Los registros de cada procedimiento identifican entradas, máscaras y presupuesto efectivo.

El panel climático procede de reanálisis ERA5 recuperado retrospectivamente. El desplazamiento de 168 horas restringe la información temporal, pero no prueba la disponibilidad histórica de la versión descargada. El horizonte solo utiliza calendario conocido; clima, hidrología y precio realizados permanecen fuera de las entradas futuras.

## Ausencias y transformaciones

La imputación y el escalado se ajustan exclusivamente con el tramo de optimización. Las máscaras distinguen una observación ausente de un valor numérico real. Los periodos sin texto mantienen sus orígenes de evaluación; su ausencia se representa explícitamente. El escenario declarado determina la disponibilidad del corpus oficial. Las fechas de recuperación y las URL se conservan para auditoría, fuera del contenido utilizado por los modelos.

## Partición y selección

El desarrollo de los comparadores utiliza validación temporal hacia adelante en 2018–2021; N-HiTS dispone de 2019–2021. Cada ajuste separa optimización, parada interna y evaluación externa. La tabla de selección registra las configuraciones, semillas, máximos de épocas y paciencia efectivamente utilizados. Se restaura el checkpoint de menor pérdida interna.

La base de M4<sub>N</sub>, denominada M1, combina PatchTST, el experto E y el experto periódico. Sus predicciones históricas se generan mediante selección por prefijos y OOF temporal anidado. La memoria se ajusta con 2013–2020, utiliza 2021 para parada y conserva la configuración 6: 16 vecinos, edad máxima 365 días, proyecciones de dimensión 32 y compuerta de ancho 64. La evaluación contiene diez semillas, 101–110. La memoria admite únicamente residuos de episodios d−2 o anteriores y mantiene sus parámetros congelados durante prueba.

## Estimandos y comparaciones

La prueba común comprende los 1.461 orígenes diarios de 2022–2025. MASE es la métrica primaria, con q₁₆₈=215.617,90361589016 kWh, calculado en 2000–2017; MAE y RMSE se presentan en GWh por intervalo. Cada métrica se calcula por realización antes de promediar semillas. La banda entre semillas expresa variación algorítmica.

La comparación temporal utiliza el origen diario, después de promediar pérdidas individuales entre semillas. El benchmark conserva su bootstrap circular de 10.000 réplicas y sus familias originales. M4<sub>N</sub> utiliza los contrastes guardados: bootstrap estacionario con 49.999 réplicas y bloque esperado de 14 días, DM–HLN con HAC 13 y GW bilateral con instrumento retrasado dos días. Se presentan los márgenes de 0 y 3 % y las cotas de Holm de la familia registrada de 1.536 hipótesis. GW contrasta capacidad predictiva condicional y no establece por sí solo la dirección de la ganancia.

Los comparadores históricos tienen cinco semillas y M4<sub>N</sub>, diez. La comparación no empareja identificadores de semilla entre grupos. Los presupuestos, las entradas y la actualización de memoria delimitan la interpretación: la mejora corresponde al procedimiento completo. El periodo de prueba había sido inspeccionado durante el desarrollo, por lo que los contrastes tienen alcance retrospectivo y requieren confirmación independiente.

## Fuentes de especificación

[Selección del benchmark](_static/resultados/evaluacion_e2/diseno_seleccion.csv) · [Metadatos de los comparadores](_static/resultados/evaluacion_e2/metadatos_procedimientos.csv) · [Protocolo del benchmark](_static/resultados/evaluacion_e2/protocolo.json) · [Resultados y procedencia de M4](_static/resultados/evaluacion_m4/manifiesto.json).

La {ref}`ficha de M4ₙ <m4-n>` desarrolla la formulación, el código y los cuatro esquemas. La {ref}`evaluación integrada <resultados-m4>` presenta métricas, curvas y contrastes.

# Auditoría de entrenamiento y validación

Se verificó el panel hasta el 31 de diciembre de 2021; no se evaluó prueba. La auditoría comprobó cada ventana de demanda y etiqueta de los cuatro pliegues, la continuidad horaria del índice y las formas, finitud y reversibilidad de la normalización. El calendario se contrastó con la fuente en tres orígenes de cada subconjunto.

## Resultado

- Índice horario, único, ordenado y con zona America/Bogota.
- Demanda histórica exactamente igual al objetivo desplazado una fila.
- Entrada original: (N,168,24): una demanda más 23 canales de calendario. Calendario futuro: (N,24,23); objetivo: (N,24).
- Para un origen t0, las filas históricas son t0−168h a t0−1h, pero su demanda ya tiene un rezago de disponibilidad de una hora: corresponde a observaciones t0−169h a t0−2h. No es un desfase accidental; implementa el supuesto de publicación.
- Etiquetas exactamente t0 a t0+23h. Sin cruces de etiquetas entre entrenamiento, parada temprana y validación anual.
- Sin NaN ni infinitos en ventanas aceptadas. Se descarta un origen inicial en cada pliegue por el primer dato rezagado no disponible.
- Permutar características históricas futuras no altera la entrada del origen auditado.

## Salvedad metodológica detectada

`Datos.__init__` ajusta medias y desviaciones sobre todo 2000–2017 antes de separar la parada temprana. Incluye 15.768 filas horarias del tramo ES del pliegue 2018 y 7.872 del de 2019. No usa las etiquetas de validación anual ni prueba para escalar, pero incumple la separación estricta entre ajuste del escalador y parada temprana. Los pliegues 2020 y 2021 no presentan ese solapamiento con la escala fija.

Antes de la nueva búsqueda se debe ajustar cada escalador exclusivamente con el subconjunto efectivo de entrenamiento de su pliegue y reutilizarlo sin reajuste en ES y validación. La escala MASE fija del periodo base es una decisión separada del protocolo de métricas; debe mantenerse declarada o redefinirse explícitamente sin mezclar escalas entre comparaciones. La salvedad no demuestra que explique el bajo rendimiento exclusivo de LSTM/CNN: afecta la preparación compartida.

Conclusión: las ventanas y etiquetas están correctamente estructuradas; el pipeline de escalado no se aprueba sin corregir la salvedad. Las configuraciones propuestas son un plan condicionado, no una repetición ya ejecutada ni resultados confirmatorios.

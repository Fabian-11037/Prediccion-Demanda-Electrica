# Corrector semántico por hora sobre E + C+R (desarrollo 2019-2021)

Orígenes evaluados: 1096; con anuncio oficial alineado: 878.

| Contraste | MASE A | MASE B | Reducción | IC 95 % | p Holm |
|---|---|---|---|---|---|
| global: num_horario frente a C+R | 0.5419 | 0.5467 | 0.89 % | [0.52; 1.27] | 0.0010 |
| global: oficial_reciente frente a C+R | 0.5417 | 0.5467 | 0.92 % | [0.51; 1.32] | 0.0010 |
| global: oficial frente a C+R | 0.5440 | 0.5467 | 0.50 % | [0.19; 0.83] | 0.0090 |
| global: oficial_prensa frente a C+R | 0.5427 | 0.5467 | 0.73 % | [0.45; 1.03] | 0.0010 |
| global: metadatos frente a C+R | 0.5424 | 0.5467 | 0.78 % | [0.49; 1.09] | 0.0010 |
| global: oficial frente a num_horario | 0.5440 | 0.5419 | -0.39 % | [-0.52; -0.26] | 1.0000 |
| global: oficial_prensa frente a num_horario | 0.5427 | 0.5419 | -0.16 % | [-0.32; 0.00] | 1.0000 |
| global: oficial_prensa frente a metadatos | 0.5427 | 0.5424 | -0.05 % | [-0.11; 0.01] | 1.0000 |
| global: oficial frente a oficial_reciente | 0.5440 | 0.5417 | -0.42 % | [-0.63; -0.20] | 1.0000 |
| global: oficial_prensa frente a desalineado | 0.5427 | 0.5422 | -0.10 % | [-0.16; -0.03] | 1.0000 |
| con_anuncio_oficial_alineado: num_horario frente a C+R | 0.5626 | 0.5684 | 1.03 % | [0.63; 1.45] | 0.0010 |
| con_anuncio_oficial_alineado: oficial_reciente frente a C+R | 0.5623 | 0.5684 | 1.09 % | [0.64; 1.53] | 0.0010 |
| con_anuncio_oficial_alineado: oficial frente a C+R | 0.5644 | 0.5684 | 0.71 % | [0.37; 1.07] | 0.0010 |
| con_anuncio_oficial_alineado: oficial_prensa frente a C+R | 0.5634 | 0.5684 | 0.89 % | [0.59; 1.20] | 0.0010 |
| con_anuncio_oficial_alineado: metadatos frente a C+R | 0.5633 | 0.5684 | 0.91 % | [0.60; 1.23] | 0.0010 |
| con_anuncio_oficial_alineado: oficial frente a num_horario | 0.5644 | 0.5626 | -0.32 % | [-0.46; -0.18] | 1.0000 |
| con_anuncio_oficial_alineado: oficial_prensa frente a num_horario | 0.5634 | 0.5626 | -0.15 % | [-0.33; 0.03] | 1.0000 |
| con_anuncio_oficial_alineado: oficial_prensa frente a metadatos | 0.5634 | 0.5633 | -0.02 % | [-0.09; 0.05] | 1.0000 |
| con_anuncio_oficial_alineado: oficial frente a oficial_reciente | 0.5644 | 0.5623 | -0.38 % | [-0.62; -0.12] | 1.0000 |
| con_anuncio_oficial_alineado: oficial_prensa frente a desalineado | 0.5634 | 0.5629 | -0.08 % | [-0.16; -0.01] | 1.0000 |
| sensibilidad_estricta: oficial_prensa_estricto frente a C+R | 0.5419 | 0.5467 | 0.87 % | [0.50; 1.25] | nan |
| sensibilidad_estricta: oficial_prensa_estricto frente a num_horario | 0.5419 | 0.5419 | -0.01 % | [-0.20; 0.15] | nan |
| sensibilidad_estricta: oficial_prensa_estricto frente a oficial_prensa | 0.5419 | 0.5427 | 0.14 % | [-0.07; 0.36] | nan |

Veredicto preregistrado:

- oficial: aporte del texto demostrado = no; reducción relevante (≥ 3 %) frente a C+R = no; se descarta un aporte del texto de 3 % = sí (cota superior -0.2627113550901413).
- oficial_prensa: aporte del texto demostrado = no; reducción relevante (≥ 3 %) frente a C+R = no; se descarta un aporte del texto de 3 % = sí (cota superior 0.007452079118229414).

Regla: Aporte del texto demostrado: oficial (u oficial_prensa) mejora a num_horario y, en el caso de oficial_prensa, también a metadatos y al texto desalineado (media de 5 desplazamientos), con p de Holm < 0,05 en la familia global de 10 contrastes. Reducción relevante: reducción frente a C+R >= 3 % con el límite inferior del IC > 0. Equivalencia: si el límite superior del IC de los contrastes de aporte del texto queda por debajo de 3 %, se descarta un aporte del texto de 3 %. Mismos contrastes en los días con un anuncio oficial alineado (definidos solo por el texto), en una familia aparte.

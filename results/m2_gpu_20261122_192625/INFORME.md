# M2 · energía diaria y distribución horaria

Prueba retrospectiva 2022–2025, 1.461 días, 24 horas y cinco semillas. Selección exclusiva en desarrollo2018–2021. Checkpoints y escaladores finales congelados antes de prueba. No se actualizan parámetros en prueba.

| Procedimiento | MASE | DE semillas | MAE kWh | RMSE kWh |
|---|---:|---:|---:|---:|
| NNLS | 0.620904 | 0.008338 | 133877.92 | 176527.32 |
| M1 | 0.634923 | 0.010050 | 136900.69 | 180108.41 |
| solo_distribucion | 0.657655 | 0.032779 | 141802.23 | 190693.14 |
| patchtst | 0.683149 | 0.029009 | 147299.20 | 196124.79 |
| directa_24 | 5.236521 | 0.312028 | 1129087.73 | 1212514.83 |
| M2 | 5.326852 | 0.466176 | 1148564.75 | 1232190.01 |
| solo_energia | 5.331279 | 0.466459 | 1149519.13 | 1226907.29 |

Se conservan diagnósticos separados de energía y perfil. La conservación de energía y la positividad son propiedades estructurales, no garantías de precisión. Las ablaciones mantienen la configuración seleccionada del modelo completo.

El codificador PatchTST se restaura congelado por año/semilla. Su estado de contexto y calendario se conserva en el caché OOF; cada año procede de un codificador estimado solo en años anteriores. Las bases latentes pueden variar entre modelos anuales; es una limitación de esta implementación OOF. Las escalas de las cabezas se estiman solo en ajuste. Los errores usados como entrada llevan retraso de dos días.

Bootstrap pareado por origen:9.999 réplicas, bloques14 y sensibilidad7/28, dentro de cada año. Las semillas se promedian en la pérdida por origen. IC95 puntuales. La familia prevista es de28 hipótesis: se reúnen las cuatro previas de M1 y las cuatro nuevas de M2; las20 restantes están pendientes. Se informa cota conservadora de Holm con20 valores uno adicionales, no un ajuste definitivo de todos los modelos. M1 y NNLS son checkpoints existentes; no se reentrenan ni se incluyen como entradas de M2. No se necesita OOF anidado de M1 para este modelo.

Estos años ya se conocían durante el diseño: evaluación retrospectiva informada. No se afirma originalidad ni superioridad sin contraste. Resultados locales; sin publicación automática.

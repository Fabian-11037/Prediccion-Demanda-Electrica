# Evaluación reducida B: Colombia

Diez semillas 101–110, configuración 6 fija y pesos restaurados en su mejor época interna. Se reutilizan los ajustes completos compatibles. Las nuevas ablaciones se ajustan una sola vez por semilla: optimización hasta 2020, parada interna en 2021 y prueba 2022–2025. No se realiza una nueva búsqueda ni una nueva validación externa de esas ablaciones.

La base conserva la construcción OOF por prefijos. La memoria operativa solo incorpora residuos observables con retraso de dos días y antigüedad máxima de 365 días. Las retropredicciones en muestra solo sustituyen los residuos históricos de la ablación correspondiente.

TimesNet y N-BEATS se evalúan desde cinco pesos históricos sin entrenamiento; los demás comparadores históricos conservan sus pérdidas publicadas y sus cinco semillas. Las realizaciones no se emparejan artificialmente por identificadores de semilla. La unidad inferencial es el origen diario, promediando las pérdidas entre semillas.

| Procedimiento | MASE | Semillas |
|---|---:|---:|
| M4_N_similitud | 0.562758 | 10 |
| kNN_residual | 0.563885 | 10 |
| M4_desalineado_91 | 0.572967 | 10 |
| M4_O | 0.574523 | 10 |
| M4_desalineado_301 | 0.574560 | 10 |
| M4_metadatos | 0.575204 | 10 |
| M4_N_sin_compuerta | 0.575592 | 10 |
| M4_desalineado_promedio | 0.575736 | 10 |
| M4_OP_estricto | 0.576049 | 10 |
| residuos_OOF_no_anidado | 0.576087 | 10 |
| M4_desalineado_154 | 0.576310 | 10 |
| M4_desalineado_217 | 0.576728 | 10 |
| M4_N | 0.577037 | 10 |
| M4_desalineado_49 | 0.578116 | 10 |
| M4_OP | 0.580221 | 10 |
| EWMA_residual | 0.580301 | 10 |
| residuos_en_muestra | 0.596771 | 10 |
| EWMA_MLP_CE | 0.597724 | 10 |
| FreshCast_adaptado | 0.601650 | 10 |
| MLP_CE | 0.602165 | 10 |
| M4_N_uniforme | 0.602589 | 10 |
| M4_sin_num | 0.603304 | 10 |
| NNLS | 0.617653 | 10 |
| M4_N_fija | 0.626358 | 10 |
| RATL_adaptado | 0.631580 | 10 |
| M1 | 0.631580 | 10 |
| historico_TimesNet | 0.642928 | 5 |
| Ridge_residual | 0.664289 | 10 |
| historico_adj_modelo6_20261112_1035__N+O | 0.668433 | 5 |
| historico_adj_modelo6_20261112_1035__N | 0.672502 | 5 |
| historico_adj_fusion_tardia_20261113_1728__N+O | 0.703797 | 5 |
| historico_adj_modelo5_20261114_1327__N | 0.741814 | 5 |
| historico_adj_modelo4_20261114_1224__N | 0.926633 | 5 |
| historico_NBEATS | 0.953343 | 5 |
| historico_repo_nhits_numerico_r1__N | 1.023629 | 5 |
| historico_adj_modelo3_20261115_0419__N | 1.051026 | 5 |
| historico_adj_modelo7_20261114_0430__N+O | 1.122419 | 5 |
| ingenuo_168h | 1.354906 | 1 |
| ingenuo_24h | 2.076725 | 1 |

Los contrastes finales utilizan 49.999 réplicas, bloques esperados de 14 días, DM–HLN con HAC 13, GW bilateral con instrumento observable de rezago dos y Holm con cota familiar de 1.536 hipótesis. Las no ejecutadas no se presentan como resultados; se conserva la cota familiar previamente registrada. Un rechazo GW no establece dirección.

Colombia conserva carácter retrospectivo por exposición previa a prueba. No se certifica potencia, originalidad ni cumplimiento completo del E3.

## Alcance pendiente

- UCI, GEFCom y OPSD: diferidos por presupuesto temporal
- Nuevos ajustes y validación ampliada del Top10: diferidos
- Validación externa forward chaining de las nuevas ablaciones residuales y textuales: diferida
- Ampliación de semillas según potencia: diferida; no se acredita potencia 0,90
- Confirmación prospectiva y originalidad metodológica: pendientes
- Windows nativo y ONNX GPU: no ejecutados; se conserva el rechazo histórico en empates

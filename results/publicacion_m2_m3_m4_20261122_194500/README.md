# Resultados GPU de M2, M3 y M4

Entrenamiento y prueba retrospectiva2022–2025:1.461 orígenes diarios,24 horas y cinco semillas101–105. Configuraciones y transformaciones congeladas antes de prueba.

| Modelo | Configuración | MASE de prueba | Informe |
|---|---:|---:|---|
| M2 | 7 | 5.326852 | [Informe](../m2_gpu_20261122_192625/INFORME.md) |
| M3_OP | 4 | 0.637692 | [Informe](../m3_gpu_20261122_193240/INFORME.md) |
| M4_OP | 6 | 0.578898 | [Informe](../m4_gpu_20261122_194500/INFORME.md) |

M3 y M4 utilizan M1 fuera de muestra mediante selección anidada por prefijo. Sus fuentes y registros documentan arquitectura, disponibilidad, controles, ablaciones y presupuesto. M4 incluye el ajuste Holm conjunto de las28 hipótesis; consultar sus contrastes completos antes de afirmar superioridad.

El benchmark ya era conocido al diseñar los modelos: evaluación retrospectiva informada, no un holdout nunca consultado. No se descartan resultados desfavorables. La publicación incluye los checkpoints intermedios conservados y los finales; no repite entrenamientos ni modifica métricas.

ALCANCE.json enumera las exclusiones locales y MANIFIESTO_PUBLICADO.json identifica los archivos incluidos. Los manifiestos originales se preservan. No se modifica ni despliega el JBook o el dashboard.

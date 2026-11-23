# M4 · memoria causal numérica y textual

Prueba retrospectiva2022–2025:1.461 orígenes diarios,24 horas, cinco semillas. M1 y E5 congelados. Selección de ocho configuraciones exclusivamente con M4_OP en desarrollo2018–2021; controles y ablaciones heredan la seleccionada.

| Procedimiento | MASE | DE semillas | MAE kWh | RMSE kWh |
|---|---:|---:|---:|---:|
| M4_desalineado_91 | 0.568063 | 0.004846 | 122484.64 | 163712.86 |
| M4_O | 0.571659 | 0.007552 | 123259.91 | 164339.57 |
| M4_desalineado_154 | 0.572078 | 0.013304 | 123350.25 | 165098.73 |
| M4_sin_compuerta | 0.572099 | 0.005381 | 123354.84 | 164976.70 |
| M4_N | 0.573144 | 0.007952 | 123580.16 | 165149.39 |
| M4_desalineado_301 | 0.573294 | 0.004479 | 123612.43 | 165212.81 |
| M4_metadatos | 0.573357 | 0.011832 | 123626.07 | 165196.74 |
| M4_OP_estricto | 0.574689 | 0.003360 | 123913.22 | 165420.72 |
| M4_desalineado_217 | 0.575717 | 0.010848 | 124134.83 | 165744.43 |
| M4_OP | 0.578898 | 0.006909 | 124820.87 | 166587.12 |
| M4_desalineado_49 | 0.579852 | 0.008552 | 125026.46 | 166696.18 |
| M4_sin_num | 0.604458 | 0.009736 | 130331.88 | 172688.93 |
| NNLS | 0.620904 | 0.008338 | 133877.92 | 176527.32 |
| M4_memoria_fija | 0.630091 | 0.007885 | 135858.83 | 179245.37 |
| M1 | 0.634923 | 0.010050 | 136900.69 | 180108.41 |

Se reutiliza el OOF anidado de M1 preparado por M3; no se utiliza el ganador global para filas históricas. Proyección32, relevancia aprendida por fragmento y canal, atención entre episodios y compuerta según contexto, similitud, dispersión, cantidad y antigüedad del respaldo.

La memoria es creciente y solo admite episodios d−2 o anteriores, dentro de la edad máxima. La máscara temporal se aplica antes de top-k y antes de consultar residuos. Sin vecinos, identidad exacta con M1. Los residuos corresponden a M1 fuera de muestra. Durante prueba, pesos, escalas y mapas de fuentes permanecen congelados; solo se incorporan episodios ya observables. Se guardan vecinos, fechas, atenciones, compuertas y log de disponibilidad. Las proyecciones se vectorizan por fecha sin estadísticas entre orígenes; las máscaras evitan influencia de filas futuras.

Se incluyen control numérico, oficial, metadatos, cinco desplazamientos, memoria fija en2021, sin compuerta, sin contexto numérico y sensibilidad estricta. La variante numérica coincide con la ablación sin texto. El contrato textual específico de §5 del diseño Markdown y los vectores E5 son los mismos que M3; se registra su diferencia con el bloque genérico del JSON.

Se completa la familia de28 hipótesis de los cuatro modelos. Holm conjunto por cada tamaño de bloque, con14 días principal y7/28 de sensibilidad;9999 réplicas pareadas dentro del año. Las semillas se promedian en pérdidas antes del bootstrap. IC95 puntuales. Los cinco desplazamientos se agregan por pérdidas. No se deduce equivalencia de ausencia de significancia.

Diseño informado por un benchmark ya conocido; evaluación retrospectiva. No se garantiza positividad de la corrección aditiva ni mejora respecto de la base. Los resultados se guardan localmente, sin publicación o despliegue automático.

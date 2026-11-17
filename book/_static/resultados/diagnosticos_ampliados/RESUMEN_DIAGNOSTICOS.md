# Diagnósticos de los modelos ampliados · 16 de noviembre de 2026

Se completaron los diez procedimientos ampliados y sus 50 checkpoints, manteniendo pesos, escaladores, configuraciones y benchmark. Las 50 reproducciones CPU abarcan 1.461 días y 24 horas por origen. La diferencia máxima por pronóstico frente a los archivos conservados es 778.483 kWh; el mayor cambio relativo del MAE es 0.005961 %.

## Importancia agrupada

192 fechas comunes (días 1, 8, 15 y 22 de cada mes de 2022–2025), cinco semillas, nueve grupos y tres desplazamientos por año: 1.350 mediciones. Se preserva la trayectoria horaria interna, con donantes a al menos 28 días. Se comparan contra el basal CPU del mismo checkpoint. El p10–p90 entre desplazamientos es dispersión descriptiva, no un IC temporal. La hora del día no cambia entre orígenes diarios y no queda identificada por este mecanismo.

El calendario es el grupo de mayor impacto en siete procedimientos y el perfil de demanda en tres. El máximo observado es 221.538 % (PatchTST (repositorio) · N, calendario); entre precio, hidrología y clima, el mayor incremento es 1.477 % (N-HiTS · N, precio). Los efectos son condicionales a cada modelo y no son aditivos ni causales. La perturbación del nivel incluye la restitución a kWh. Las cuatro familias base conservan su análisis de 1.461 fechas y diez desplazamientos; no se presentan las dos muestras como un único experimento homogéneo.

## Sensibilidad al texto

Cinco semillas y 1.461 días por procedimiento. El control histórico remuestrea con reemplazo contenido de la misma fuente cuya disponibilidad declarada es anterior a 2022; conserva las listas, el volumen y la presencia originales. No se clasifica ni filtra por tema. El control cero conserva presencia y sesgos. Las descripciones estáticas se estudian por separado. Los IC95 % son puntuales, con bootstrap circular pareado de 14 días y 10.000 remuestras después de promediar semillas y repeticiones; no acreditan potencia entre inicializaciones.

- TaTS adaptado · N+O: retirar documentos +17.0654 % de MAE; contenido cero +15.0910 %; contenido histórico de la misma fuente +0.0152 %, IC95 % [+0.0032; +0.0281], divergencia media 476.8 kWh por intervalo.
- Fusión tardía · N+O: retirar documentos +11.6111 % de MAE; contenido cero +3.1296 %; contenido histórico de la misma fuente +1.7576 %, IC95 % [+0.9813; +2.5451], divergencia media 35923.1 kWh por intervalo.
- TGForecaster · N+O: retirar documentos +30.3923 % de MAE; contenido cero +28.4251 %; contenido histórico de la misma fuente -1.6524 %, IC95 % [-2.4510; -0.8369], divergencia media 57748.3 kWh por intervalo.

- TGForecaster · N: descripciones a cero +167.2748 % de MAE; permutadas +4.1954 %.
- TGForecaster · N+O: descripciones a cero +69.9672 % de MAE; permutadas +9.4964 %.

TaTS apenas responde al reemplazo por contenido histórico de la misma fuente, aunque sí al paso a cero: depender del canal no equivale a explotar la información particular del día. El control histórico reduce descriptivamente el error en TGForecaster con documentos. La dependencia de una rama puede coexistir con un peor resultado que el control N entrenado por separado. Estos diagnósticos no sustituyen los contrastes N/N+O del benchmark ni seleccionan arquitecturas o fuentes con la prueba.

## Latencia y reproducción

Se midieron las 70 realizaciones de los 14 procedimientos en la misma CPU, con un hilo, batch uno y los mismos tres orígenes. Cada corrida tiene 20 calentamientos y 100 mediciones: 210 corridas y 21.000 llamadas temporizadas. Se informa la mediana de 15 medianas por procedimiento y la mediana de sus p95; no es un p95 combinado. ARX presenta 0.02395 ms y TFT · N 54.32780 ms, extremos de las medianas observadas. Sólo se mide forward con entradas y vectores en memoria; no carga, preparación ni encoder E5. Los contadores documentales incluyen los siete cierres diarios de TaTS cuando corresponde.

Resultados en los CSV/NPZ de esta carpeta; procedencia y SHA-256 en fuentes_diagnosticos.json; condiciones en protocolo.json y entorno.json. Código en scripts/inferencia_ampliada.py y scripts/diagnosticos_ampliados.py; presentación en scripts/diagnosticos_ampliados_jbook.py y scripts/actualizar_diagnosticos_jbook.py. El JBook presenta tres figuras nuevas, sus interpretaciones, el código completo del diagnóstico y la tabla de latencia de los 14 procedimientos.

Verificación: 86 pruebas correctas y cinco omitidas; 13 cuadernos, 97 celdas analíticas, 53 gráficas, 25 tablas calculadas, 24 tablas Markdown y 12 esquemas. Construcción sin errores ni advertencias, cero enlaces rotos y revisión visual de las tres figuras nuevas. Panel, benchmark base y checkpoints verificados por SHA-256. HTML local actualizado en docs; sin entrenamiento ni publicación.

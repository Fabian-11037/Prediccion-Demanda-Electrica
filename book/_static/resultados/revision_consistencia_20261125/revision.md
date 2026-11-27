# Revisión integral del Entregable 3

Fecha: 25 de noviembre de 2026. Alcance: fuentes del JBook, tablas, gráficas, esquemas, dashboard y artefactos de evaluación.

## Inventario y correspondencia

- 15 cuadernos; 82 figuras analíticas con interpretación contigua; 32 esquemas.
- 114 tablas: 35 calculadas y 79 nativas.
- 16 procedimientos aprendidos y 85 realizaciones; dos referencias ingenuas completan los 18 procedimientos integrados. El benchmark de referencia conserva 15 procedimientos y 75 realizaciones.
- La incertidumbre común contiene 864 filas de cobertura y 17 comparadores de WIS. Calibración inicial: 2022; evaluación: 2023–2025.
- 24 CSV del libro coinciden exactamente con los resultados de origen comprobados en esta revisión. Las métricas puntuales, intervalos y contrastes se verifican también mediante las pruebas del dashboard.
- La revisión de prosa no encuentra variantes propias descartadas. Los modelos de terceros citados como antecedentes permanecen en la revisión bibliográfica.

## Correcciones documentales

Se precisan los conjuntos de 15 y 18 procedimientos, se retiran dos tablas de métricas que repetían la tabla maestra y la repetición de incertidumbre en las conclusiones, y se actualizan los enlaces de reproducibilidad. Se eliminan notas de autorización y planificación de la exposición científica, así como cinco avisos informativos del backend gráfico. Se conservan las limitaciones metodológicas, las interpretaciones y el código analítico. El inventario se genera desde las fuentes, sin fijar conteos antiguos en las comprobaciones.

## Estado frente a la pauta

| Requisito y referencia | Evidencia actual | Remanente |
|---|---|---|
| Correcciones de E2, §4 | Tabla de observaciones, acciones, evidencia e impacto en capítulo 12; fórmulas numeradas, bibliografía y 45 parejas internas | Las observaciones sobre clasificador adversarial y equidad no se convierten en experimentos realizados por documentarlas |
| Selección, §5.1 | Tabla maestra de 18 procedimientos, con métricas, IC, significancia y coste CPU común | Los presupuestos de búsqueda siguen siendo distintos; el orden de prueba no sustituye la selección por validación |
| Novedad,R1 y§6.3 | Seis antecedentes cercanos, fuentes primarias, consultas públicas y diferencias candidatas | Búsqueda nativa en Scopus/WoS/IEEE, seguimiento completo, revisión independiente, relación con la limitación del paper guía y defensa de adaptación algorítmica sustantiva |
| Implementación,R2 y§6.5 | Memoria propia en PyTorch; ecuaciones, pseudocódigo y pipelines expuestos | Trazabilidad de autoría y vinculación de pruebas con integración continua |
| Propiedad,§§2y7.3 | Cotas condicionales, diez comprobaciones,128 consultas y gradientes verificados | La cota no garantiza menor error ni continuidad global de top-k; esa limitación permanece en el enunciado |
| Cronología,§8.1 | Fechas, pesos y huellas conservados; exposición retrospectiva documentada | No convertir el estudio existente en confirmación a priori; preregistrar la confirmación independiente |
| Equidad, §8.2 | Presupuestos y entradas identificados; control residual con 60.929 parámetros frente a 60.993 de M4 ya evaluado | Comparadores con cinco semillas frente a diez; presupuestos, covariables y políticas sin equiparar. La capacidad igualada no está pendiente de ejecución |
| Potencia,§8.7.2 | Dimensionamiento del benchmark y precisiones de parada | Potencia≥0,90 por contraste y conjunta para M4; dimensionar con desarrollo y ampliar semillas cuando corresponda |
| Inferencia, R3 y §§8.5–8.7 | Familias temporales, MCS; efectos estandarizados y sensibilidad MASE/RMSE y primeras/últimas tres semillas con Holm408 | Confirmación en reservas independientes; no volver a listar estos complementos como no ejecutados |
| Generalización,§8.3 | Conjunto colombiano y segmentación por año/horizonte | Tres conjuntos adicionales de naturaleza distinta, elegidos antes de evaluarlos; toda la familia y mejora en al menos dos |
| Ablación, §6.5 | Seis controles numéricos con diez realizaciones, capacidad igualada y 36 contrastes Holm; integrados en JBook y dashboard | La compuerta no acredita beneficio y la recuperación por distancia tiene menor MASE; discutir la contribución sin afirmar necesidad de todo el diseño. Generalización externa diferida |
| Robustez y escalabilidad,§8.8 | Cuatro perturbaciones, tres severidades, 96 orígenes y cinco realizaciones completas por procedimiento; curvas e inferencia de degradación | Ampliar a diez realizaciones completas; sensibilidad de hiperparámetros, curvas por tamaño y escalabilidad para toda la familia |
| Adecuación de M4, §8.8 | Diagnósticos y segmentos; checkpoints restaurados evaluados en ajuste y parada, MASE 0,4193 y 0,4786, brecha 14,15 % con IC95 % amplio | El cálculo está completado; no acredita brecha pequeña ni separa sobreajuste y cambio temporal |
| Incertidumbre, §8.8 | Evaluación común y bandas de los dieciséis checkpoints en vivo con cortes causales; 576 casos de paridad Python/JS | Confirmación en periodos nuevos y cobertura condicional de entradas intervenidas; dependencia temporal |
| Interpretabilidad y coste, §8.8 | FLOPs de todas las redes, pico global del proceso, Pareto y dos escenarios energéticos; latencia CPU común conservada | Medición GPU no disponible en el entorno CPU; energía real no medida; coste de búsqueda heterogéneo |
| Exportación, §5.2 | Catálogo de 16 procedimientos en navegador, 96 casos y 2.304 valores horarios; paridad relativa dentro de 1e-4 | La verificación cubre estados preparados y una perturbación; faltan adquisición operativa y validación de estados nuevos |
| Dashboard,§9.4 | Contexto,EDA,Modelos; filtros, resultados e inferenciaCPU; explicación local y aviso de extrapolación | Interfaz principalHTML/JS/Plotly y panelDash de inferencia. Migración íntegra aDash si se exige literalmente; tabla de líderesDash ordenable/filtrable con todos los campos; incertidumbre en inferencia en vivo |
| Despliegue,§10 | Versión local comprobada y Dockerfile del paqueteM4 | ContenedorLinux, servicio de producción y prueba pública; la edición auditada no se publica |
| Reproducibilidad y entrega,§§11–12 | Fuentes con salidas, código y hashes; paqueteM4 autónomo sobre entradas preparadas | Reproducción independiente del proyecto completo; DOI/release, declaración de autoría y uso de herramientas firmada, ficha del modelo y declaración de datos; completar la estructura formal solicitada |

## Prioridades metodológicas

1. Anexar la verificación independiente específica de novedad; la revisión bibliográfica dirigida ya existe y la auditoría documental general está cerrada.
2. Completar la reproducción independiente, los metadatos de entrega y la trazabilidad de autoría y revisión.
3. Para una confirmación posterior, equiparar presupuestos y semillas, dimensionar potencia y evaluar tres conjuntos adicionales. Las ablaciones y el control de capacidad ya están ejecutados e integrados.
4. Ampliar escalabilidad y estabilidad de explicaciones; completar medición GPU cuando exista hardware utilizable. Los intervalos en inferencia operativa ya están implementados. La robustez controlada, los diagnósticos de M4, la sensibilidad temporal y la incertidumbre común ya están incorporados.

Evidencia: `results/auditoria_consistencia_20261125/`. Las hojas de contacto conservan las 82 figuras y los 32 esquemas revisados. El alcance de las verificaciones automáticas se registra en `verificacion.json`; la revisión no equivale a una reproducción independiente completa ni a una confirmación prospectiva.

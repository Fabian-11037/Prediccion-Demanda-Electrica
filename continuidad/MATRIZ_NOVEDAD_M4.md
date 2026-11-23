# Matriz inicial de antecedentes de M4

Fecha de consulta: 23-11-2026. Revisión preliminar; no constituye una búsqueda sistemática completada. El cierre exige entre cinco y diez textos completos y trazabilidad de sus métodos. La atribución de una diferencia a M4 sigue pendiente de contrastación.

| Trabajo | Evidencia consultada | Mecanismo próximo | Qué contrastar con M4 |
|---|---|---|---|
| [RATL: Learning from Retrieved Residuals for Robust Multivariate Time-Series Forecasting](https://arxiv.org/html/2609.03937v1) | Texto completo HTML v1 | Memoria de residuos específica de base congelada, recuperación causal y router aprendido | Construcción de residuos en muestra/OOF/anidada; banco fijo frente a creciente; forma de abstención y supervisión del router |
| [Retrieval Is Not Enough: Refreshing Memory for Frozen Time-Series Forecasters (FreshCast)](https://arxiv.org/html/2610.07834v3) | Texto completo HTML v3 | Memoria actualizada con observaciones reveladas, regresión por kernel y peso calibrado | Valores recuperados frente a residuos; calibración cerrada frente a compuerta; OOF anidado y acceso temporal equiparado |
| [Retrieval Augmented Time Series Forecasting (RAFT)](https://proceedings.mlr.press/v267/han25d.html) | Resumen y ficha de ICML 2025; lectura completa pendiente | Recuperación de contextos históricos y sus continuaciones observadas | Corrección residual frente a aumento de entradas; causalidad, presupuesto y base equiparados |
| [Multi-Modal Time Series Prediction via Mixture of Modulated Experts (MoME)](https://arxiv.org/abs/2601.21547) | Resumen; lectura completa pendiente | Condicionamiento multimodal y modulación de expertos | Qué parte de la relevancia aprendida por texto o del enrutamiento ya existe; no confundir con otros artículos llamados MoME |
| [Mixture-of-Linear-Experts for Long-term Time Series Forecasting (MoLE)](https://proceedings.mlr.press/v238/ni24a.html) | Resumen; lectura completa pendiente | Enrutamiento adaptativo de expertos lineales | Diferencia entre combinar expertos paramétricos y recuperar episodios; control simple del router |

Por cada texto completo registrar autores, año/versión, DOI o identificador, dominio/datasets, ecuaciones, construcción del banco, instante de disponibilidad, función objetivo, complejidad, ablations y limitaciones. Guardar páginas/secciones que sustenten cada celda. Incorporar los trabajos adicionales hallados por búsquedas reproducibles; no afirmar ausencia de antecedentes solo por no encontrarlos en esta lista.

La comparación con RATL y FreshCast es prioritaria: la recuperación residual y la actualización de memoria ya tienen antecedentes explícitos. La hipótesis a auditar es la contribución de la construcción OOF temporal anidada y su integración en la política de recuperación disponible, no una supuesta invención de la memoria o de la compuerta.

# Ampliación exploratoria de capacidad v4

Seis variantes CNN1D y cuatro LSTM conforme a `PLAN_VARIANTES_V4.md`. Preprocesamiento corregido v3. Selección exclusivamente con validación; prueba2022–2025ya observada yreevaluada por autorización del autor. Los resultados no representan una confirmación independiente.

El runner guarda configuración y huellas en `plan.json`, progreso en `estado.json` y `progreso.html`, y por ajuste pesos, escaladores, historiales por época y predicciones con origen temporal. Tras completar los108ajustes genera `resumen_final.csv` y `RESUMEN.md` con comparaciones frente av3 yla referencia semanal. Los baselines MLP y ARX de v3 se conservan sin reentrenar. Para ejecutar o reanudar usar el entorno mlenv con acceso GPU y el mismo plan congelado.

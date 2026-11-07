# Instrucciones para agentes

Lee `CONTEXTO_PROYECTO.md` antes de hacer cualquier cosa; es la fuente de verdad (objetivo, reglas anti-fuga, decisiones abiertas).

Reglas clave:
1. Idioma de trabajo: español.
2. **Nunca mencionar, citar ni comparar con ningún proyecto previo del autor** (otro objetivo, otro conjunto de datos u otro repositorio) en artículos, cuadernos, informes, README, dashboards ni código. No copiar material de ese trabajo.
3. No usar el conjunto de prueba (`allow_test=True`) salvo en la corrida final preregistrada.
4. Escalas, imputaciones y vocabularios se ajustan solo con entrenamiento; las entradas `x_*` nunca se leen en o después del origen.
5. El clima de `data/raw` es reanálisis: no es una entrada válida para el futuro.
6. No inventar referencias ni cifras; las cifras de los textos salen de celdas ejecutadas.
7. Antes de commit: `pytest -q tests`.
8. La prueba ya fue leída para ARX lineal y MLP (`results/base/`). No cambiar espacios, presupuesto ni opciones de entrenamiento después de ver esos resultados; el LSTM se corre en GPU con las mismas opciones.
9. La validación cruzada es forward chaining propio (`src/demanda/particion.py`); no se usa el paquete `timeseries-cv`.

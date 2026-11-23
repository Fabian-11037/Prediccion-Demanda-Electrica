# Antecedentes y alcance de la implementación adicional

23 de noviembre de2026. Se consultan los textos completos accesibles y sus secciones de método; esta revisión dirigida no constituye una revisión sistemática exhaustiva. La matriz no acredita prioridad mundial ni revisión independiente.

| Texto primario | Mecanismo pertinente | Diferencia concreta evaluable |
|---|---|---|
| [ResMem](https://arxiv.org/html/2302.01576), Yang et al. | Añade un regresor vecino a residuos de una red | La evaluación temporal exige disponibilidad y procedencia OOF; memorizar residuos no es una contribución nueva |
| [RAFT](https://arxiv.org/html/2505.04163), Han et al. | Recupera continuaciones y las fusiona con historia a varias escalas | M4 recupera errores de una base específica, no directamente continuaciones |
| [kNN-MTS](https://arxiv.org/html/2505.11625), Zhang et al. | Memoria de representaciones y objetivos; combinación según distancias | M4 conserva trayectorias residuales y añade selección supervisada |
| [RATL](https://arxiv.org/html/2609.03937), He et al. | Recuperación residual y router por bloques sobre base congelada | Contrastar la procedencia temporal anidada y la memoria creciente; la recuperación residual aprendida ya tiene antecedente directo |
| [FreshCast](https://arxiv.org/html/2610.07834v3), He et al. | Memoria actualizada, lecturas por valor/forma y combinación calibrada | La actualización observable tampoco demuestra novedad por sí sola |

Conclusión de la revisión dirigida: la evidencia consultada no basta para afirmar originalidad metodológica demostrada de M4_N. Se presenta como adaptación evaluada con construcción temporal OOF anidada. Una propiedad local y ganancias empíricas no prueban prioridad ni superioridad universal. La trazabilidad OOF es una diferencia del protocolo evaluado; no se afirma que sea desconocida en toda la literatura.

Las nuevas comparaciones cercanas son **adaptaciones a M1**, no réplicas oficiales. Ambas utilizan orígenes diarios, retraso de dos días, antigüedad máxima365d y el mismo objetivo agregado. Se fijan antes de su ajuste. No se sustituye el nombre del procedimiento por el de un artículo si no coincide su receta completa.

RATL_adaptado: claves de doce estadísticos,64 candidatos, bloques8, candidato cero, atención de conjuntos de dimensión64 y cuatro cabezas, imitación del oráculo con coeficiente0,4 y temperatura0,1, escala residual aprendida y gamma en0:0,1:1 elegida por MSE interno. El banco permanece fijo en el ajuste. La capa de atención, codificador resumido y demás decisiones no especificadas en el suplemento disponible son elecciones explícitas de esta adaptación; no se atribuyen a una configuración oficial.

FreshCast_adaptado: historia96 para similitud de forma, referencias estacionales y hasta200 análogos, once descriptores y kernel lineal+MLP64–64 con temperatura aprendida; combinación estimada en forma cerrada y recortada a[0,1] en la reserva interna. Se entrena por dataset, con banco diario limitado365d, en lugar de entrenar un kernel compartido sobre los conjuntos del artículo. Se conserva la misma información observable de M4. La activación GELU y el presupuesto común de esta ejecución son decisiones de adaptación.

Ambas guardan arquitectura, presupuesto, escalas y pesos; no equivalen al grupo de capacidad igualada. Las cifras comparativas se generan únicamente tras ejecutar. La falta de mejora de M4 frente a una adaptación no se ocultará. Un resultado favorable frente a estas implementaciones no prueba superioridad frente a todo el código oficial ni elimina el efecto de decisiones de implementación.

El benchmark numérico adicional de diez familias es un experimento nuevo con semillas101–110, parámetros fijos y pérdida común MAE/q. No reemplaza el Top10 canónico con variantes textuales ni sus cifras históricas. Las predicciones canónicas no disponibles se registran como faltantes y nunca se reconstruyen a partir de promedios o rankings.

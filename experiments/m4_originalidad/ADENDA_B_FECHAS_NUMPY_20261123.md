# Compatibilidad de fechas en la cola B

El 23 de noviembre de 2026, la ejecución se detuvo antes de entrenar la primera
pila de residuos en muestra. pandas rechazó un escalar `numpy.str_` en
`Timestamp`. Se normalizan los dos límites mediante `str` antes de construir
sus marcas temporales. No cambian fechas, purga, datos, pérdidas, semillas,
hiperparámetros ni criterios de selección.

La fuente original permanece intacta. La cola reducida utiliza
`m4_cierre_procedencia_b.py`, copia de compatibilidad con esas dos conversiones.
El registro y el ejecutor anteriores se archivan en
`reducido_B/incidencias/20261123_fechas_numpy/`; el registro operativo pasa a
versión 2 antes de reanudar. Se conserva la inferencia histórica completada y
todos los ajustes terminados. La prueba de regresión verifica fechas NumPy y
Python, los límites efectivos y la purga sin entrenar modelos.

Se mantienen publicación verificada y apagado final. Un error interrumpe
la cola y conserva encendido el equipo.

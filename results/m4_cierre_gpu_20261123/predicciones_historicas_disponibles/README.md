# Pronósticos históricos disponibles

Se extraen 35.064 pronósticos horarios por modelo para 2022–2025, con origen diario a las 00:00 de America/Bogota. El horizonte 1 corresponde al intervalo 00:00–01:00. Los CSV conservan el promedio y los percentiles 10/90 de las cinco semillas publicados en el dashboard, convertidos de GWh a kWh, con precisión de 0,01 kWh.

No contienen las cinco realizaciones individuales: estas no se recuperan de los agregados. Los percentiles describen dispersión entre semillas, no cobertura predictiva. Los contrastes del benchmark utilizan las pérdidas individuales promediadas por origen, disponibles en los JSON originales; no el error del pronóstico promedio.

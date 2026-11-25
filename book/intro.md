# Pronóstico de la demanda eléctrica del SIN colombiano

**Fabian José Manrique Morelos y Gloria Bastidas Rodriguez**  
Maestría en Estadística Aplicada · Universidad del Norte

Este estudio evalúa el pronóstico horario de la demanda real del Sistema Interconectado Nacional de Colombia. Utiliza 168 horas de historia para anticipar las siguientes 24 horas, con demanda, calendario, clima, hidrología, precio y documentos oficiales según la arquitectura. El análisis exploratorio caracteriza los cambios de régimen, la dependencia temporal y la cobertura documental.

La propuesta M4<sub>N</sub> recupera errores de episodios históricos mediante una memoria numérica aprendida sobre una combinación congelada de expertos. En 1.461 orígenes diarios de 2022–2025 y diez semillas obtiene MASE 0,5770, MAE 0,1244 GWh y RMSE 0,1660 GWh por intervalo. Reduce el error un 10,25 % frente a TimesNet, el comparador de menor MASE del Top 10. Los contrastes temporales ajustados respaldan mejoras superiores al 3 % frente a sus diez arquitecturas, dentro del alcance retrospectivo del estudio.

Las contribuciones comprenden la caracterización reproducible del conjunto de datos, un benchmark de quince procedimientos con 75 realizaciones y dos referencias estacionales, y la evaluación de una memoria residual con diez realizaciones adicionales. El libro desarrolla datos y EDA, literatura, diseño temporal, arquitecturas, resultados y conclusiones. Cada figura integra su interpretación y el código analítico acompaña los resultados.

[Dashboard](https://demanda-electrica-dashboard.onrender.com/index.html) · {ref}`Fundamento de M4 <m4-n>` · {ref}`Evaluación integrada <resultados-m4>`.

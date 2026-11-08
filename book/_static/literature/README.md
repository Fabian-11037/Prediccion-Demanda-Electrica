# Evidencia bibliográfica

Los archivos registran evidencia dirigida sobre demanda eléctrica, precio y métodos de series temporales. Los estratos tienen propósitos y denominadores distintos; no sumar sus tamaños como estudios únicos. estudios_destacados.csv conserva 23 fichas complementarias; literature/estudios_jbook.csv aporta cuatro estudios de demanda. Los 20 textos completos del registro de comparadores se solapan con las fichas destacadas.

extracciones_modelos.csv distingue comparador, comparador_benchmark, propuesto, componente y ablacion. El capítulo cuenta una vez por estudio y familia, sin descomponer híbridos. Los estratos principal, sin preprints y ampliado contienen 16, 14 y 19 estudios respectivamente. Los abstracts no se usan para contar comparadores empíricos de texto completo.

manifiesto.json fija SHA-256 y tamaño de las fuentes estructuradas consolidadas. Las extracciones incluyen hashes y localizadores de los documentos revisados. Los artículos no se redistribuyen. El inventario de bases no está reconciliado como un flujo PRISMA único; la revisión es de un revisor y posterior al entrenamiento.

La generación está en scripts/literatura_integral_jbook.py y el análisis visible en book/08_literatura.ipynb. La reconstrucción del capítulo no entrena modelos ni consulta predicciones.

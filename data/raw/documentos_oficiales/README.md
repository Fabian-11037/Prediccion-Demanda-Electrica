# Corpus de documentos oficiales

La tabla consolidada está en data/processed/corpus_texto.parquet. El inventario, originales HTTP, miembros PDF de ZIP, textos, OCR e índices CDX se conservan en esta carpeta. Consultar results/documentos_oficiales/INFORME_COBERTURA.md para cifras, cobertura y limitaciones.

Secuencia reproducible, desde la raíz del proyecto:

```bash
python scripts/adquisicion/descargar_documentos_oficiales.py descubrir
python scripts/adquisicion/descargar_documentos_oficiales.py ampliar_xm
python scripts/adquisicion/descargar_documentos_oficiales.py descargar
python scripts/adquisicion/descargar_documentos_oficiales.py reintentar
python scripts/adquisicion/descargar_documentos_oficiales.py originales_creg
python scripts/adquisicion/descargar_documentos_oficiales.py expandir_paquetes
python scripts/verificar_disponibilidad_oficial.py --descargar-indices
python scripts/integrar_documentos_oficiales.py --solo-extraer
python scripts/adquisicion/ocr_documentos_oficiales.py --preparar
python scripts/adquisicion/ocr_documentos_oficiales.py --max-paginas 4
python scripts/integrar_documentos_oficiales.py --solo-integrar
python scripts/finalizar_documentos_oficiales.py
```

El OCR requiere Windows, idioma es-MX y pdftoppm; se ejecuta en un entorno separado con requirements-ocr-windows.txt. El resto de la adquisición y los resultados OCR guardados se pueden consumir sin ese motor. Cambiar max-paginas permite ampliar OCR de anexos; las páginas parciales no se presentan como documentos íntegramente transcritos.

Uso por origen:

```python
from demanda.corpus import cargar_corpus, documentos_en_origen
corpus = cargar_corpus('estricto')
documentos = documentos_en_origen(corpus, '2023-08-15 00:00', ventana_horas=72)
```

El origen sin zona se interpreta en America/Bogota. Elegir declarado o estricto explícitamente y aplicar el mismo escenario en todas las particiones. No utilizar fecha_declarada para saltarse disponible_desde. URLs, hashes, pruebas y metadatos no se concatenan automáticamente al texto. No ejecutar entrenamiento ni publicar durante la reproducción de esta integración.

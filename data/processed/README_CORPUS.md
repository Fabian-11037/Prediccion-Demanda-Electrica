# Corpus de texto procesado

Versión activa: `corpus_oficial_ampliado_20261120`. El archivo `corpus_texto.parquet` contiene 625.577 registros de 13 fuentes, incluidos 50.104 documentos oficiales incorporados. El escenario principal es declarado; estricto se conserva para sensibilidad. El diccionario describe las columnas y el manifiesto conserva procedencia, reglas temporales y SHA-256.

## Obtener el archivo completo

El corpus se almacena mediante Git LFS por su tamaño (159.086.185 bytes). Instalar Git LFS y ejecutar desde el repositorio:

```console
git lfs install
git lfs pull --include="data/processed/corpus_texto.parquet"
```

Descargar el repositorio como ZIP de GitHub puede entregar un puntero LFS. Comprobar que el archivo recuperado tiene el tamaño y el hash indicados en el manifiesto.

```python
from pathlib import Path
import hashlib
import pandas as pd

ruta = Path("data/processed/corpus_texto.parquet")
with ruta.open("rb") as archivo:
    assert hashlib.file_digest(archivo, "sha256").hexdigest() == "e82f1b0fccc09ffb4deb9bd8c2dfebb7bedcc3cabc2c0fd355b2ae1d2413898c"
corpus = pd.read_parquet(ruta)
assert len(corpus) == 625577 and corpus["id"].is_unique
```

Usar `demanda.corpus.cargar_corpus` para excluir cuarentenas y seleccionar la disponibilidad del escenario. Una fecha sin hora entra desde las 00:00 del día siguiente en Colombia. Los metadatos de auditoría no se concatenan al contenido textual.

La cobertura mensual y anual por fuente y escenario, los registros en cuarentena y la validación temporal están en `results/ampliacion_documental_20261120/`. Los archivos crudos, originales recuperados y scripts enumerados en el manifiesto documentan la adquisición; esta actualización publica el corpus procesado y no redistribuye todos los originales.

Los resultados vigentes conservan su corpus congelado de 14.162 documentos. El corpus activo ampliado requiere nuevos vectores versionados para un entrenamiento futuro; los vectores anteriores no corresponden a sus filas.

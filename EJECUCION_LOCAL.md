# Demanda eléctrica del SIN: ejecución local

Autores: Fabian José Manrique Morelos y Gloria Bastidas Rodriguez.

Este paquete contiene el JBook actual y el dashboard con sus datos exportados. El JBook se reconstruye desde sus cuadernos con salidas guardadas; no ejecuta entrenamiento al construirse. El dashboard permite explorar las predicciones históricas y las métricas de los 15 procedimientos.

## Dashboard

Solo requiere Python 3.10 o superior, sin instalaciones adicionales. Desde la carpeta descomprimida:

```console
python ejecutar.py dashboard --puerto 8068
```

Abrir http://127.0.0.1:8068/index.html. No abrir index.html directamente desde el explorador: los datos se cargan mediante HTTP.

## JBook

Instalar Jupyter Book 1.x en un entorno virtual. En Windows:

```console
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements-jbook.txt
.venv\Scripts\python ejecutar.py jbook --puerto 8000
```

En Linux o macOS:

```console
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-jbook.txt
.venv/bin/python ejecutar.py jbook --puerto 8000
```

Abrir http://127.0.0.1:8000/intro.html. La construcción conserva la opción `execute_notebooks: off` del proyecto. También se puede construir sin iniciar el servidor:

```console
python ejecutar.py jbook --solo-construir
```

Los comandos anteriores se ejecutan con el Python del entorno virtual. La instalación inicial requiere Internet; el libro puede utilizar servicios externos para representar fórmulas. Para detener el servidor, usar Ctrl+C.

## Contenido

- `book/`: capítulos, cuadernos, esquemas, código mostrado y recursos utilizados por el libro.
- `dashboard/site/`: aplicación estática, Plotly y resultados exportados, incluidas las fichas descargables.
- `ejecutar.py`: construcción y servidor local para ambos sitios.
- `requirements-jbook.txt`: dependencia necesaria para reconstruir el libro.
- `MANIFIESTO_SHA256.json`: inventario e integridad de los archivos.

El paquete permite reconstruir y visualizar la entrega. El recálculo de todos los análisis y el entrenamiento de redes requieren los datos, entornos y artefactos del repositorio de investigación; no forman parte de este paquete mínimo. Los pipelines conservados en el libro documentan esos procedimientos.

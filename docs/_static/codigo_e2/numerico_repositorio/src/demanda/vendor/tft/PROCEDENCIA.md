# Implementación de referencia TFT

Fuente:Nixtla/neuralforecast, revisión94b6ce131838be38a1fb5157349f02bbc8c89ef6, archivo`neuralforecast/models/tft.py`. LicenciaApache2.0 conservada.

`tft_original.py` conserva el archivo original completo. `core.py` extrae literalmente funciones y componentes independientes de BaseModel; el métodoTFT.forwardse dedenta y renombra`tft_forward`sin cambios de lógica. La arquitectura compacta y el entrenador propio se declaran como adaptaciones exploratorias; no es réplica exacta del artículo ni implementación original deGoogle.

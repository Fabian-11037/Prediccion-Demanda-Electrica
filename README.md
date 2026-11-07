# Predicción de demanda eléctrica (SIN, Colombia) con series de tiempo y texto

Datos consolidados y libres de fuga de información para entrenar modelos que combinen series de tiempo y NLP en el pronóstico de la **demanda real horaria** del Sistema Interconectado Nacional.

Lee primero [`CONTEXTO_PROYECTO.md`](CONTEXTO_PROYECTO.md): define el problema, las reglas anti-fuga, las decisiones abiertas y las limitaciones de los datos.

## Contenido
- `data/raw`: serie horaria 2000-2025 (demanda, precio de bolsa, clima en 8 ciudades), hidrología diaria y 568 295 titulares con metadatos de disponibilidad.
- `data/external`: festivos de Colombia (generados), y scripts/plantillas para ISE (DANE) y pronósticos de temperatura (pendientes de descarga).
- `data/processed`: panel horario *as-of* (`panel_horario.parquet`), registro de variables con su clase de disponibilidad, informe de calidad y cobertura de titulares.
- `src/demanda`: calendario, consolidación, particiones, ventanas por origen, titulares por escenario, ISE, pronósticos y climatología (todo ajustado solo con entrenamiento).
- `tests`: pruebas de fuga (ventanas, rezagos, escalas, particiones, pronósticos, ISE, titulares) y de calendario.

## Uso rápido
```bash
pip install -r requirements.txt
python scripts/construir_panel.py
pytest -q tests
```

## Diseño anti-fuga en una línea
Cada columna `x_*` ya trae su rezago de publicación y solo se lee antes del origen; el futuro solo expone calendario (`cal_*`) y pronósticos con emisión previa; las escalas se ajustan con entrenamiento; la partición purga el horizonte y la prueba está congelada.

## Estado
Panel y pruebas listos. Pendientes: descargar ISE y pronósticos de temperatura (scripts sin probar con las fuentes reales), decidir el bloque de prueba (se recomienda uno prospectivo desde 2026) y preregistrar.

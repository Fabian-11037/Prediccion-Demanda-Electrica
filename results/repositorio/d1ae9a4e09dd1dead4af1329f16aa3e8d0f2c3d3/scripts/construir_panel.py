"""Genera data/processed/{panel_horario.parquet, feature_registry.json, calidad_datos.json}."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from demanda.consolidar import construir_panel, guardar  # noqa: E402

if __name__ == "__main__":
    panel, reg = construir_panel()
    guardar(panel, reg)
    print(panel.shape, panel.index.min(), panel.index.max())
    # cobertura de titulares por fuente y año (ambos escenarios)
    from demanda import config as C  # noqa: E402
    from demanda import noticias as N  # noqa: E402

    for esc in N.ESCENARIOS:
        N.cobertura(N.cargar_titulares(esc)).to_csv(C.PROCESADO / f"cobertura_titulares_{esc}.csv")

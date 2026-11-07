"""Genera data/external/festivos_colombia.csv (2000-2035) con la implementación propia."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from demanda import config as C  # noqa: E402
from demanda.calendario import festivos_colombia  # noqa: E402

if __name__ == "__main__":
    f = festivos_colombia(range(2000, 2036))
    f["fecha"] = f["fecha"].dt.strftime("%Y-%m-%d")
    f.to_csv(C.EXTERNO / "festivos_colombia.csv", index=False)
    print(len(f), "festivos escritos")

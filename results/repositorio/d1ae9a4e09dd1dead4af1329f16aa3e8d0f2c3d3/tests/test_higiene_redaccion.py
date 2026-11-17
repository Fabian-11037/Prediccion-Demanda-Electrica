"""El repositorio no debe contener referencias a trabajos anteriores del autor."""
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
PROHIBIDAS = ["Time-Series-Project", "DeepLearningProject", "Time-MMD", "proyecto previo", "proyecto anterior",
              "predicción de precio", "prediccion de precio"]
EXENTOS = {"CONTEXTO_PROYECTO.md", "CLAUDE.md", "AGENTS.md", "test_higiene_redaccion.py"}
EXT = {".md", ".py", ".json", ".txt", ".csv", ".ipynb", ".yml", ".yaml", ".toml"}


def test_sin_referencias_a_trabajos_previos():
    malos = []
    for p in RAIZ.rglob("*"):
        if not p.is_file() or ".git" in p.parts or p.name in EXENTOS or p.suffix not in EXT:
            continue
        if p.stat().st_size > 5_000_000:
            continue
        t = p.read_text(encoding="utf-8", errors="ignore").lower()
        malos += [(str(p.relative_to(RAIZ)), s) for s in PROHIBIDAS if s.lower() in t]
    assert not malos, malos

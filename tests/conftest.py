import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture(scope="session")
def panel_registro():
    from demanda.consolidar import construir_panel
    return construir_panel()


@pytest.fixture(scope="session")
def panel(panel_registro):
    return panel_registro[0]


@pytest.fixture(scope="session")
def registro(panel_registro):
    return panel_registro[1]

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "adquisicion"))
import descargar_pronosticos_temperatura as D  # noqa: E402
import importar_ise as I  # noqa: E402


def test_parser_pronosticos_un_modelo_y_multimodelo():
    js = {"hourly": {"time": ["2024-03-01T00:00", "2024-03-01T01:00"],
                     "temperature_2m_previous_day1": [13.1, 12.9], "temperature_2m_previous_day2": [13.0, None]}}
    d = D.parsear_respuesta(js, "bogota")
    assert set(d.lead_dias) == {1, 2} and len(d) == 3
    js2 = {"hourly": {"time": ["2024-03-01T00:00"], "temperature_2m_previous_day1_gfs_seamless": [13.1]}}
    assert D.parsear_respuesta(js2, "bogota").modelo.iloc[0] == "gfs_seamless"
    with pytest.raises(ValueError):
        D.parsear_respuesta({"hourly": {"time": [], "temperature_2m": []}}, "bogota")


def test_parser_ise_sintetico():
    meses = ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio", "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"]
    filas = [["Serie", None, None, None]] + [[f"{m} 2023", 100.0 + i, 101.0 + i, 102.0 + i] for i, m in enumerate(meses)]
    d = I.extraer(pd.DataFrame(filas))
    assert len(d) == 12 and str(d.mes.iloc[0]) == "2023-01" and d.ise_original.iloc[-1] == 111.0

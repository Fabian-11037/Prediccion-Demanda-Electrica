"""La confirmación reducida no reutiliza el año de filtro para elegir el modelo."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from entrenar_tft_reducido import elegir_final


def test_seleccion_final_usa_solo_confirmacion_y_semilla_comun():
    a, b = dict(id='A'), dict(id='B')
    filas = [dict(configuracion=c, fold=2018, semilla=s, MASE=v)
             for c, v in [(a, .01), (b, 10)] for s in [43, 44, 45]]
    filas += [dict(configuracion=c, fold=y, semilla=43, MASE=v)
              for c, v in [(a, .9), (b, .7)] for y in [2019, 2020, 2021]]
    score, elegido = elegir_final(filas, [a, b])
    assert elegido == b
    assert abs(score - .7) < 1e-12

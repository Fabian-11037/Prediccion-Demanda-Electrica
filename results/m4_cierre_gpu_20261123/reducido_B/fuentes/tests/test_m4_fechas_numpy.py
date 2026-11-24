"""Regresión de las fechas escalares en el límite de parada, sin GPU ni ajuste."""
import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


@pytest.mark.parametrize('dtype', ['U10', object])
def test_residuos_en_muestra_acepta_fechas_numpy_y_conserva_purga(tmp_path, dtype):
    fuente = Path(__file__).resolve().parents[1] / 'scripts/m4_cierre_procedencia_b.py'
    funcion = next(n for n in ast.parse(fuente.read_text()).body
                   if isinstance(n, ast.FunctionDef) and n.name == 'insample')
    indice = pd.date_range('2020-01-01', periods=24 * 24, freq='h', tz='America/Bogota')
    datos = SimpleNamespace(indice=indice, pos=np.arange(24 * 8, len(indice), 24))
    dias = pd.date_range('2020-01-10', periods=11).strftime('%Y-%m-%d').to_numpy().astype(dtype)
    meta = {'dia': dias}
    capturado = {}

    class EntrenamientoNoEjecutado(Exception):
        pass

    def interceptar(datos, nombre, cfg, s, tr, st, ev, destino, progreso):
        capturado.update(tr=tr, st=st)
        raise EntrenamientoNoEjecutado

    entorno = dict(np=np, pd=pd, REPO=tmp_path,
        leer=lambda p: {'expertos_configuraciones_seleccionadas': {
            n: {'configuracion': {}} for n in ['patchtst', 'E', 'periodica']}},
        E=SimpleNamespace(entrenar_base=interceptar))
    exec(compile(ast.Module(body=[funcion], type_ignores=[]), str(fuente), 'exec'), entorno)
    with pytest.raises(EntrenamientoNoEjecutado):
        entorno['insample'](datos, meta, np.arange(7), np.arange(7, 11),
                            101, tmp_path, lambda **kw: None)
    assert indice[capturado['tr'][-1]].strftime('%Y-%m-%d') == '2020-01-15'
    assert list(indice[capturado['st']].strftime('%Y-%m-%d')) == [
        '2020-01-17', '2020-01-18', '2020-01-19', '2020-01-20']
    assert indice[capturado['tr'][-1]] + pd.Timedelta(hours=23) < indice[capturado['st'][0]]

"""La cola no inicia TFT mientras N-HiTS corre ni después de un fallo."""
import importlib.util
import json
from pathlib import Path
import pytest


def preparar(tmp_path, monkeypatch, estado):
    ruta = Path(__file__).resolve().parents[1] / 'scripts/encadenar_nhits_tft.py'
    spec = importlib.util.spec_from_file_location('cola_test', ruta)
    cola = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cola)
    previo = tmp_path / 'nhits'
    out = tmp_path / 'tft'
    previo.mkdir()
    out.mkdir()
    (previo / 'estado.json').write_text(json.dumps(dict(pid=123, estado=estado)))
    (out / 'plan.json').write_text('{}')
    monkeypatch.setattr(cola, 'PREVIO', previo)
    monkeypatch.setattr(cola, 'OUT', out)
    llamadas = []
    monkeypatch.setattr(cola.subprocess, 'Popen', lambda *a, **k: llamadas.append(a))
    return cola, llamadas


def test_no_lanza_tft_mientras_nhits_esta_activo(tmp_path, monkeypatch):
    cola, llamadas = preparar(tmp_path, monkeypatch, 'Entrenando')
    monkeypatch.setattr(cola, 'activo', lambda *args: True)
    def terminar_espera(*args):
        raise InterruptedError('fin de prueba')
    monkeypatch.setattr(cola.time, 'sleep', terminar_espera)
    with pytest.raises(InterruptedError):
        cola.main()
    assert not llamadas
    assert json.loads((cola.OUT / 'cola.json').read_text())['estado'] == 'esperando_nhits'


def test_error_nhits_no_dispara_otro_entrenamiento(tmp_path, monkeypatch):
    cola, llamadas = preparar(tmp_path, monkeypatch, 'Error')
    monkeypatch.setattr(cola, 'activo', lambda *args: False)
    with pytest.raises(RuntimeError, match='N-HiTS terminó con error'):
        cola.main()
    assert not llamadas

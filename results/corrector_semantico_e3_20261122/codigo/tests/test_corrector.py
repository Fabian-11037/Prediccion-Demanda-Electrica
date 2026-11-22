"""Pruebas del corrector semántico: fechas anunciadas, candidatos sin fuga, cota y máscaras del modelo, rasgos numéricos
retrasados y utilidades estadísticas."""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
from corrector import evaluacion as EV       # noqa: E402
from corrector import fechas as F            # noqa: E402
from corrector import modelo as MO           # noqa: E402
from corrector import texto as TX            # noqa: E402


# ------------------------------------------------------------------ fechas
@pytest.mark.parametrize("texto,pub,esperado", [
    ("El aislamiento regirá a partir del 25 de marzo hasta el 13 de abril de 2020.", (2020, 3, 14), [((2020, 3, 25), (2020, 4, 13))]),
    ("Racionamiento: a partir del 2 de abril, durante dos semanas.", (2020, 3, 14), [((2020, 4, 2), (2020, 4, 15))]),
    ("Restricciones del 20 al 24 de marzo en Bogotá", (2020, 3, 14), [((2020, 3, 20), (2020, 3, 24))]),
    ("Hasta el próximo 24 de abril se espera una pleamar", (2015, 4, 20), [((2015, 4, 20), (2015, 4, 24))]),
    ("El simulacro se realizará el 19 de marzo", (2020, 3, 14), [((2020, 3, 19), (2020, 3, 19))]),
    ("A partir del 5 de enero rige la nueva tarifa", (2019, 12, 28), [((2020, 1, 5), (2020, 1, 11))]),
    ("En la reunión del 3 de febrero de 2020 se discutió el tema", (2020, 3, 14), []),
    ("Desde el 3 de febrero de 2019 se aplica la medida", (2020, 3, 14), []),            # pasado lejano: narración, no anuncio
])
def test_intervalos(texto, pub, esperado):
    r = [(a, b) for a, b, _ in F.intervalos(texto, dt.date(*pub))]
    assert r == [(dt.date(*a), dt.date(*b)) for a, b in esperado]


def test_plano_conserva_posiciones():
    s = "Á partir del 5 de marzo, Ñandú"
    assert len(F.plano(s)) == len(s)


# ------------------------------------------------------------------ candidatos
def _corpus():
    disp = pd.to_datetime(["2020-03-10 05:00", "2020-03-12 05:00", "2020-03-13 15:00", "2020-03-14 05:00", "2020-03-16 05:00"], utc=True)
    c = pd.DataFrame(dict(id=list("abcde"), fuente=["IDEAM", "El Tiempo", "CREG", "El Tiempo", "IDEAM"],
                          canal=["oficial", "prensa", "oficial", "prensa", "oficial"],
                          titulo=["Boletín", "Titular 1", "Resolución", "Titular 2", "Boletín 2"],
                          texto=["La medida rige a partir del 20 de marzo hasta el 22 de marzo. " + "x" * 3000, "", "Texto " * 400, "", "Nada"],
                          disp=disp))
    return c


def test_candidatos_sin_fuga_y_alineados_en_su_intervalo():
    c = _corpus()
    fr = TX.fragmentos(c)
    it = TX.intervalos_corpus(c, fr, lambda *a, **k: None)
    assert len(it) == 1 and it.fragmento.iat[0] == 0
    dias = pd.date_range("2020-03-11", "2020-03-25")
    k = TX.candidatos(c, fr, it, dias, sorted(c.fuente.unique()), 4, 6, 6, 72, 120)
    origen = pd.DatetimeIndex(dias).tz_localize("America/Bogota").tz_convert("UTC").as_unit("ns").asi8
    disp_frag = pd.DatetimeIndex(c.disp).as_unit("ns").asi8[fr.doc.to_numpy()]
    for i in range(len(dias)):
        usados = k["cand"][i][k["cand"][i] >= 0]
        assert (disp_frag[usados] <= origen[i]).all()            # ningún documento posterior al origen
        al = k["cand"][i, :4]
        if (al >= 0).any():
            assert pd.Timestamp("2020-03-20") <= dias[i] <= pd.Timestamp("2020-03-22")
    assert (k["cand"][dias.get_loc(pd.Timestamp("2020-03-21")), :4] >= 0).sum() == 1
    assert (k["cand"][dias.get_loc(pd.Timestamp("2020-03-19")), :4] >= 0).sum() == 0
    assert k["cand"].shape[1] == 16


def test_reciente_respeta_ventana_de_72_horas():
    c = _corpus()
    fr = TX.fragmentos(c)
    dias = pd.date_range("2020-03-11", "2020-03-25")
    k = TX.candidatos(c, fr, TX.intervalos_corpus(c, fr, lambda *a, **kk: None), dias, sorted(c.fuente.unique()), 4, 6, 6, 72, 120)
    i = dias.get_loc(pd.Timestamp("2020-03-18"))                    # 72 h antes: 15-mar 00:00; solo el boletín del 16 cuenta
    of = k["cand"][i, 4:10]
    assert set(fr.doc.to_numpy()[of[of >= 0]]) == {4}


# ------------------------------------------------------------------ modelo
def _modelo(cota=0.05):
    torch.manual_seed(0)
    return MO.Corrector(n_num=10, n_feat=7, n_fuentes=3, d=16, cabezas=4, dropout=0.0, cota=np.full(24, cota, dtype="float32"))


def _entrada(B=5, M=8, semilla=0):
    g = torch.Generator().manual_seed(semilla)
    return (torch.randn(B, 10, generator=g), torch.randn(B, M, 768, generator=g), torch.randn(B, M, 7, generator=g),
            torch.randint(0, 4, (B, M), generator=g), torch.rand(B, M, generator=g) > 0.3)


def test_corrector_arranca_en_cero_y_respeta_la_cota():
    m = _modelo().eval()
    x = _entrada()
    assert torch.allclose(m(*x), torch.zeros(5, 24))
    for p in m.parameters():
        torch.nn.init.normal_(p, std=3.0)
    out = m(*x)
    assert out.shape == (5, 24) and float(out.abs().max()) <= 0.05 + 1e-6


def test_sin_candidatos_el_texto_no_influye():
    m = _modelo().eval()
    for p in m.salida[-1].parameters():
        torch.nn.init.normal_(p)
    num, emb, feat, fte, _ = _entrada()
    nada = torch.zeros_like(_entrada()[4])
    a = m(num, emb, feat, fte, nada)
    b = m(num, emb * 0 + 7.0, feat * 0 - 3.0, fte, nada)
    assert torch.allclose(a, b, atol=1e-6)


def test_mascaras_de_variantes():
    res = dict(alineados=(0, 2), oficial=(2, 5), prensa=(5, 8))
    cand = np.arange(8)[None].repeat(3, 0)
    feat = np.zeros((3, 8, 7), dtype="float32")
    feat[:, :5, 0] = 1                                              # alineados y oficial: canal oficial
    feat[:, 1, 0] = 0                                               # un alineado de prensa
    feat[..., 5:] = 1
    m, _ = MO.mascara_variante("num_horario", res, cand, feat)
    assert not m.any()
    m, _ = MO.mascara_variante("oficial_reciente", res, cand, feat)
    assert m[0].tolist() == [0, 0, 1, 1, 1, 0, 0, 0]
    m, _ = MO.mascara_variante("oficial", res, cand, feat)
    assert m[0].tolist() == [1, 0, 1, 1, 1, 0, 0, 0]
    m, f = MO.mascara_variante("metadatos", res, cand, feat)
    assert m[0].tolist() == [0, 0, 1, 1, 1, 1, 1, 1] and (f[..., 5:] == 0).all()
    m, _ = MO.mascara_variante("oficial_prensa", res, cand, feat)
    assert m.all()


def test_rasgos_numericos_usan_solo_residuos_de_dos_dias_o_mas():
    dias = np.array(pd.date_range("2019-01-01", periods=30).strftime("%Y-%m-%d"))
    rng = np.random.default_rng(0)
    y = 1000 + rng.normal(0, 10, (30, 24)).astype("float32")
    cr = dict(dia=dias, anio=np.full(30, 2019), y=y, yhat_cr=np.full((30, 24), 1000, dtype="float32"))
    C = pd.DataFrame(np.zeros((40, 3)), index=pd.date_range("2018-12-25", periods=40))
    cand = dict(dias=np.array(pd.date_range("2018-01-01", "2019-12-31").strftime("%Y-%m-%d")))
    a = MO.Datos(cr, C, cand, 100.0)
    y2 = y.copy()
    y2[19] += 500                                                   # cambia el día 20 (índice 19) y el 21 (índice 20)
    y2[20] += 500
    b = MO.Datos(dict(cr, y=y2), C, cand, 100.0)
    assert np.array_equal(a.num[20], b.num[20])                     # el día 21 no ve su propio residuo ni el del día anterior
    assert not np.array_equal(a.num[21], b.num[21])                 # el día 22 ya ve el del día 20 (d - 2)


# ------------------------------------------------------------------ estadística
def test_holm_y_bootstrap_dentro_de_cada_anio():
    assert EV.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    an = np.repeat([2019, 2020], [50, 40])
    idx = EV.indices_bootstrap(an, 20, 7, 0)
    assert (an[idx] == an[None, :]).all()


def test_contraste_signo():
    rng = np.random.default_rng(1)
    b = 0.5 + 0.05 * rng.random(200)
    a = b * 0.9
    idx = EV.indices_bootstrap(np.full(200, 2019), 499, 14, 0)
    r = EV.contraste(a, b, idx)
    assert r["reduccion_pct"] == pytest.approx(10.0) and r["ic95_inf"] > 0 and r["p_unilateral"] < 0.01

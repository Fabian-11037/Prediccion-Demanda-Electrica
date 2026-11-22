"""Pruebas unitarias del modelo original (§6.5 del Entregable 3). Datos sintéticos: no leen el panel ni la prueba.

(i) sobreajuste de un lote diminuto; (ii) reproducibilidad con la misma semilla; (iii) formas de salida y entradas
inválidas; (iv) gradiente finito y no nulo en todos los parámetros; (v) verificación numérica de las propiedades
demostradas (Proposiciones 1-3 y las del núcleo de impulso, más gradcheck); (vi) guardar, cargar y exportar a ONNX.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from original import mecanismos as OM                                # noqa: E402

L, H, NCAL, NX, NM = 168, 24, 6, 8, 2         # 6 históricas + 2 indicadores de ausencia
IDEAS = ["A", "AB", "AC", "E", "F"]


def cfg(idea: str) -> dict:
    if idea == "F":
        return dict(perfiles=8, ancho=32, capas_mlp=2, dropout=0.0, rho_nivel=0.2)
    p = dict(nucleo="patchtst", d_model=16, capas=1, cabezas=4, dropout=0.0, rho=0.05, ancho_correccion=16)
    if idea == "AB":
        p.update(resumen="impulso", dias_impulso=7)
    if idea == "AC":
        p.update(lambda_placebo=0.5)
    return p


def texto(n_dias=400, n_docs=60, semilla=0) -> dict:
    g = torch.Generator().manual_seed(semilla)
    vec = torch.nn.functional.normalize(torch.randn(n_docs, 768, generator=g), dim=1)
    lista = torch.randint(-1, n_docs, (n_dias, 8), generator=g)
    lista_dia = torch.randint(-1, n_docs, (n_dias, 4), generator=g)
    return dict(vectores=vec, lista=lista, lista_dia=lista_dia)


def lote(n=16, semilla=0):
    g = torch.Generator().manual_seed(semilla)
    x = torch.randn(n, L, NX, generator=g)
    x[:, :, -NM:] = (torch.rand(n, L, NM, generator=g) < 0.2).float()
    return x, torch.randn(n, L, NCAL, generator=g), torch.randn(n, H, NCAL, generator=g), torch.randint(30, 390, (n,), generator=g)


def modelo(idea: str, semilla: int = 0, con_texto: bool = True, **extra):
    torch.manual_seed(semilla)
    tx = texto() if (idea in OM.TEXTO and con_texto) else None
    return OM.construir(idea, dict(cfg(idea), **extra), NX, NCAL, L, H, tx, n_mascaras=NM)


def objetivo(x, cal_f):
    """Etiqueta sintética con solución conocida: forma diaria que depende del calendario y del último día de demanda."""
    return 1.0 + 0.05 * torch.sin(torch.arange(H) / 24 * 6.283)[None] + 0.02 * x[:, -24:, 0].mean(1, keepdim=True) + 0.01 * cal_f[:, :, 0]


# ------------------------------------------------------------------ (i) sobreajuste
@pytest.mark.parametrize("idea", IDEAS)
def test_sobreajusta_lote_pequeno(idea):
    x, cp, cf, dia = lote(16)
    y = objetivo(x, cf)
    m = modelo(idea)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    inicial = (m(x, cp, cf, dia) - y).abs().mean().item()
    for _ in range(500):
        opt.zero_grad()
        perdida = (m(x, cp, cf, dia) - y).abs().mean()
        perdida.backward()
        opt.step()
    assert perdida.item() < 0.03 and perdida.item() < 0.2 * inicial


# ------------------------------------------------------------------ (ii) reproducibilidad
@pytest.mark.parametrize("idea", IDEAS)
def test_reproducible_con_misma_semilla(idea):
    x, cp, cf, dia = lote(4)
    a, b = modelo(idea, 7).eval(), modelo(idea, 7).eval()
    torch.testing.assert_close(a(x, cp, cf, dia), b(x, cp, cf, dia))


# ------------------------------------------------------------------ (iii) formas y entradas inválidas
@pytest.mark.parametrize("idea", IDEAS)
def test_formas_y_entrada_invalida(idea):
    x, cp, cf, dia = lote(5)
    m = modelo(idea).eval()
    assert m(x, cp, cf, dia).shape == (5, H)
    with pytest.raises(RuntimeError):
        m(torch.randn(5, L, NX + 3), cp, cf, dia)          # número de canales incorrecto
    with pytest.raises((RuntimeError, ValueError)):
        m(x[:, :100], cp[:, :100], cf, dia)                 # historia más corta que la del modelo


def test_parametros_invalidos():
    with pytest.raises(ValueError):
        OM.CorreccionAcotada(4, H, 8, rho=0.0)
    with pytest.raises(ValueError):
        OM.construir("Z", {}, NX, NCAL, L, H)


# ------------------------------------------------------------------ (iv) flujo de gradientes
@pytest.mark.parametrize("idea", IDEAS)
def test_todos_los_parametros_reciben_gradiente(idea):
    x, cp, cf, dia = lote(16)
    y = objetivo(x, cf)
    m = modelo(idea)
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    for _ in range(3):        # la última capa de v empieza en cero: tras unos pasos todos los parámetros participan
        opt.zero_grad()
        (m(x, cp, cf, dia) - y).abs().mean().backward()
        opt.step()
    opt.zero_grad()
    perdida = (m(x, cp, cf, dia) - y).abs().mean()
    if hasattr(m, "perdida_adicional"):
        perdida = perdida + m.perdida_adicional(dict(x=x, cal_p=cp, cal_f=cf, dia=dia), torch.arange(16))
    perdida.backward()
    for nombre, p in m.named_parameters():
        assert p.grad is not None, nombre
        assert torch.isfinite(p.grad).all(), nombre
        assert p.grad.abs().sum() > 0, nombre


# ------------------------------------------------------------------ (v) propiedades demostradas
@pytest.mark.parametrize("idea", ["A", "AB", "AC"])
def test_prop1_cota_frente_a_texto_adversario(idea):
    """|o − o_base| ≤ ρ para cualquier texto, incluso maximizando la desviación por ascenso de gradiente."""
    x, cp, cf, dia = lote(8)
    m = modelo(idea)
    with torch.no_grad():                                   # corrección no trivial
        for q in m.correccion.parameters():
            q.add_(torch.randn_like(q))
    m.eval()
    v = m.texto.vectores.clone().requires_grad_(True)
    opt = torch.optim.Adam([v], lr=0.5)
    rho = m.cota()
    for _ in range(60):
        m.texto.vectores = v
        dev = (m(x, cp, cf, dia) - m.base(x, cp, cf, None)).abs().max()
        opt.zero_grad()
        (-dev).backward()
        opt.step()
        assert dev.item() <= rho + 1e-6
    assert dev.item() > 0.5 * rho                            # el adversario sí se acerca a la cota: no es una cota vacía


@pytest.mark.parametrize("idea", ["A", "AB", "AC"])
def test_prop1_sin_texto_identica_a_la_base(idea):
    x, cp, cf, dia = lote(8)
    m = modelo(idea).eval()
    m.texto.lista = torch.full_like(m.texto.lista, -1) if hasattr(m.texto, "lista") else None
    if hasattr(m.texto, "lista_dia"):
        m.texto.lista_dia = torch.full_like(m.texto.lista_dia, -1)
    with torch.no_grad():
        for q in m.correccion.parameters():
            q.add_(torch.randn_like(q))
    torch.testing.assert_close(m(x, cp, cf, dia), m.base(x, cp, cf, None), rtol=0, atol=0)


def test_prop1_cota_frente_a_covariables_arbitrarias_E():
    """Idea E: cualquier valor de las covariables (escaladas, desplazadas, faltantes) mueve la salida a lo sumo ρ."""
    x, cp, cf, dia = lote(8)
    m = modelo("E")
    with torch.no_grad():
        for q in m.correccion.parameters():
            q.add_(3 * torch.randn_like(q))
    m.eval()
    base = m.base(x[:, :, :1], cp, cf, None)
    for f in (lambda z: z * 1e6, lambda z: z + 50.0, lambda z: torch.zeros_like(z), lambda z: torch.randn_like(z) * 100):
        u = x.clone()
        u[:, :, 1:NX - NM] = f(u[:, :, 1:NX - NM])
        assert (m(u, cp, cf, dia) - base).abs().max().item() <= m.cota() + 1e-6
    # la base no depende de las covariables
    u = x.clone(); u[:, :, 1:] = torch.randn_like(u[:, :, 1:])
    torch.testing.assert_close(m.base(u[:, :, :1], cp, cf, None), base)


def test_prop1_cota_del_mae():
    """MAE(ŷ) ≤ MAE(ŷ_base) + ρ · media(nivel) (desigualdad triangular), con niveles en kWh."""
    x, cp, cf, dia = lote(16)
    m = modelo("E")
    with torch.no_grad():
        for q in m.correccion.parameters():
            q.add_(torch.randn_like(q))
    m.eval()
    nivel = torch.rand(16, 1) * 1e5 + 1e5
    y = torch.rand(16, H) * 2e5 + 1e5
    mae = ((m(x, cp, cf, dia) * nivel) - y).abs().mean()
    mae_b = ((m.base(x[:, :, :1], cp, cf, None) * nivel) - y).abs().mean()
    assert mae <= mae_b + m.cota() * nivel.mean() + 1e-3


def test_ablacion_sin_cota_no_esta_acotada():
    c = OM.CorreccionAcotada(4, H, 8, rho=0.05, cota=False)
    with torch.no_grad():
        for q in c.parameters():
            q.add_(torch.randn_like(q))
    assert c(torch.randn(64, 4) * 1e3).abs().max().item() > 0.05


def test_gradcheck_correccion_acotada():
    c = OM.CorreccionAcotada(4, 3, 5, rho=0.1).double()
    with torch.no_grad():
        for q in c.parameters():
            q.add_(torch.randn_like(q))
    u = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(c, (u,), eps=1e-6, atol=1e-5)


def test_lipschitz_empirica_por_debajo_de_la_cota():
    """|Δ(u) − Δ(u')| ≤ ρ (Lip(v) + Lip(g)/4) ‖u − u'‖ con Lip de cada MLP acotada por el producto de normas espectrales."""
    torch.manual_seed(1)
    c = OM.CorreccionAcotada(6, H, 16, rho=0.1)
    with torch.no_grad():
        for q in c.parameters():
            q.add_(torch.randn_like(q) * 0.5)
    lip = lambda mlp: float(np.prod([torch.linalg.matrix_norm(l.weight, 2).item() for l in mlp.red if isinstance(l, torch.nn.Linear)]))   # noqa: E731, E741
    K = c.rho * (lip(c.v) + 0.25 * lip(c.g))       # GELU tiene constante de Lipschitz ≤ 1,13: se usa 1,13 por capa
    K *= 1.13
    u = torch.randn(500, 6)
    up = u + 1e-2 * torch.randn(500, 6)
    with torch.no_grad():
        cociente = ((c(u) - c(up)).norm(dim=1) / (u - up).norm(dim=1)).max().item()
    assert cociente <= K


def test_impulso_invarianza_al_orden_y_sin_fuga():
    """Idea B: permutar los documentos de cada día no cambia z; cambiar los días posteriores al origen tampoco; ‖z‖ ≤ max ‖V e‖."""
    torch.manual_seed(0)
    tx = texto()
    nb = OM.NucleoImpulso(tx["vectores"], tx["lista_dia"].clone(), NX, dias=7)
    with torch.no_grad():
        nb.alfa.add_(1.0)
    x, cp, cf, dia = lote(6)
    z0, hay, _ = nb(x, dia)
    perm = torch.randperm(nb.lista_dia.shape[1])
    nb.lista_dia = nb.lista_dia[:, perm]
    z1 = nb(x, dia)[0]
    torch.testing.assert_close(z0, z1, rtol=1e-5, atol=1e-6)
    permutada = nb.lista_dia.clone()
    for i, d in enumerate(dia.tolist()):
        futuro = permutada.clone()
        futuro[d + 1:] = torch.randint(0, tx["vectores"].shape[0], futuro[d + 1:].shape)
        nb.lista_dia = futuro
        torch.testing.assert_close(nb(x[i:i + 1], dia[i:i + 1])[0], z1[i:i + 1], rtol=1e-5, atol=1e-6)
    cota = nb.valor(tx["vectores"]).norm(dim=1).max()
    assert (z1.norm(dim=1) <= cota + 1e-6).all()


def test_prop2_equivarianza_de_escala_F():
    """Idea F con el preprocesamiento del proyecto: multiplicar la demanda por c > 0 multiplica el pronóstico en kWh por c."""
    m = modelo("F").eval()
    g = torch.Generator().manual_seed(3)
    d = 1e5 * (1 + 0.1 * torch.rand(4, L, generator=g))             # demanda en kWh
    sigma = 0.07
    _, cp, cf, dia = lote(4)

    def pronostico(dem):
        nivel = dem.mean(1, keepdim=True)
        x = torch.zeros(4, L, NX)
        x[:, :, 0] = (dem / nivel - 1) / sigma
        return m(x, cp, cf, dia) * nivel
    for c in (0.5, 3.0, 17.0):
        torch.testing.assert_close(pronostico(c * d), c * pronostico(d), rtol=1e-5, atol=1e-3)


def test_prop2_envolvente_y_positividad_F():
    m = modelo("F")
    with torch.no_grad():
        for q in m.parameters():
            q.add_(torch.randn_like(q))
    m.eval()
    lo, hi = m.envolvente()
    for escala in (1.0, 100.0, 1e4):
        x, cp, cf, dia = lote(32, semilla=int(escala))
        o = m(x * escala, cp * escala, cf * escala, dia)
        assert (o >= lo[None] - 1e-6).all() and (o <= hi[None] + 1e-6).all() and (o > 0).all()
    P = m.perfiles()
    torch.testing.assert_close(P.mean(1), torch.ones(P.shape[0]))


def test_prop3_placebo_reduce_el_uso_de_texto_independiente():
    """Idea C: con texto independiente del objetivo, la penalización placebo deja una corrección de texto menor."""
    x, cp, cf, dia = lote(32)
    y = objetivo(x, cf)
    efectos = {}
    for lam in (0.0, 1.0):
        m = modelo("AC", 0, lambda_placebo=lam)
        opt = torch.optim.Adam(m.parameters(), lr=3e-3)
        t = dict(x=x, cal_p=cp, cal_f=cf, dia=dia)
        for _ in range(150):
            opt.zero_grad()
            ((m(x, cp, cf, dia) - y).abs().mean() + m.perdida_adicional(t, torch.arange(32))).backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            efectos[lam] = m.aporte(x, cf, dia).abs().mean().item()
    assert efectos[1.0] < efectos[0.0]


def test_placebo_solo_usa_dias_anteriores():
    dia = torch.arange(400, 900)
    d = OM._desplazamiento_placebo(dia, torch.Generator().manual_seed(0))
    assert (d >= 28).all() and (d < 365).all()
    assert (OM._desplazamiento_placebo(dia, fijo=True) >= 28).all()


# ------------------------------------------------------------------ (vi) guardar, cargar y exportar
@pytest.mark.parametrize("idea", IDEAS)
def test_guardar_y_cargar_conserva_salidas(idea):
    x, cp, cf, dia = lote(4)
    m1 = modelo(idea, 2).eval()
    m2 = modelo(idea, 99).eval()
    buf = io.BytesIO()
    torch.save(m1.state_dict(), buf)
    buf.seek(0)
    m2.load_state_dict(torch.load(buf, weights_only=True))
    torch.testing.assert_close(m1(x, cp, cf, dia), m2(x, cp, cf, dia))


@pytest.mark.parametrize("idea", IDEAS)
def test_exportacion_onnx_ida_y_vuelta(idea, tmp_path):
    ort = pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    x, cp, cf, dia = lote(6)
    m = modelo(idea, 4)
    with torch.no_grad():
        for q in m.parameters():
            q.add_(0.05 * torch.randn_like(q))
    m.eval()
    ruta = tmp_path / f"{idea}.onnx"

    class Env(torch.nn.Module):
        def __init__(self, r):
            super().__init__()
            self.r = r

        def forward(self, a, b, c, d):
            return self.r(a, b, c, d)
    torch.onnx.export(Env(m).eval(), (x[:2], cp[:2], cf[:2], dia[:2]), str(ruta), opset_version=17, input_names=["x", "cal_p", "cal_f", "dia"], output_names=["o"],
                      dynamic_axes={k: {0: "n"} for k in ("x", "cal_p", "cal_f", "dia", "o")}, dynamo=False)
    s = ort.InferenceSession(str(ruta), providers=["CPUExecutionProvider"])
    todas = dict(x=x.numpy(), cal_p=cp.numpy(), cal_f=cf.numpy(), dia=dia.numpy())
    o = s.run(None, {i.name: todas[i.name] for i in s.get_inputs()})[0]     # el exportador omite entradas que la red no usa
    with torch.no_grad():
        ref = m(x, cp, cf, dia).numpy()
    assert np.abs(o - ref).max() < 1e-4


def test_parches_equivale_a_unfold():
    """La extracción de parches sin ``unfold`` (exportable a ONNX) da exactamente lo mismo que la original."""
    from entrenamiento import redes as R
    s = torch.randn(7, L)
    ref = torch.cat([s, s[:, -1:].expand(-1, R.PASO)], dim=1).unfold(1, R.PARCHE, R.PASO)
    torch.testing.assert_close(R._parches(s), ref, rtol=0, atol=0)

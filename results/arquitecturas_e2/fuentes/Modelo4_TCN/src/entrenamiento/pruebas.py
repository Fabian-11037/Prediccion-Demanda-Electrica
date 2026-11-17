"""Comprobaciones previas al entrenamiento. Si alguna falla, no se entrena."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from demanda import config as C
from demanda.corpus import documentos_en_origen

from .datos import DatosNumericos
from . import redes as R


def numericas(datos: DatosNumericos) -> dict:
    """Las ventanas, las etiquetas y las escalas no dependen de filas en el origen o posteriores."""
    out = {}
    f = datos.pliegues()[0]
    esc = datos.ajustar_escalas(f["tr"])
    pos = np.concatenate([f["tr"][[0, len(f["tr"]) // 2, -1]], f["val"][[0, -1]]])
    antes = datos.construir(pos, esc)
    X0, K0 = datos.X, datos.K
    try:
        ok_x = True
        for i, p in enumerate(pos):
            datos.X = X0.copy(); datos.X[p:] = 1e9          # se altera todo lo que ocurre en el origen o después
            despues = datos.construir(pos[i:i + 1], esc)
            ok_x &= bool(np.array_equal(antes.x[i], despues.x[0]) and np.array_equal(antes.cal_p[i], despues.cal_p[0]))
        out["ventanas_sin_filas_desde_el_origen"] = ok_x
        corte = int(f["tr"].max())
        datos.X = X0.copy(); datos.X[corte:] = 1e9
        datos.K = K0.copy(); datos.K[corte:] = 1e9
        esc2 = datos.ajustar_escalas(f["tr"])
        out["escalas_solo_con_filas_de_ajuste"] = bool(all(np.allclose(esc[k], esc2[k], rtol=1e-8, atol=1e-10) for k in ("mediana", "media", "std", "media_cal", "std_cal")))  # tolerancia de redondeo
    finally:
        datos.X, datos.K = X0, K0
    out["etiqueta_es_el_horizonte"] = bool(np.array_equal(antes.y, datos.Y[pos[:, None] + np.arange(datos.H)]))
    ini_prueba = pd.Timestamp(C.PARTICION["test"][0], tz=C.TZ)
    tr, es = datos.ajuste_final()
    fin_obj = datos.indice[np.concatenate([tr, es])] + pd.Timedelta(hours=datos.H - 1)
    out["ajuste_final_termina_antes_de_la_prueba"] = bool(fin_obj.max() < ini_prueba)
    out["parada_posterior_al_ajuste_con_purga"] = bool(datos.indice[tr].max() + pd.Timedelta(hours=datos.H - 1) < datos.indice[es].min())
    out["pliegues_ordenados_en_el_tiempo"] = bool(all(datos.indice[g["tr"]].max() < datos.indice[g["es"]].min() <= datos.indice[g["es"]].max() < datos.indice[g["val"]].min()
                                                       for g in datos.pliegues()))
    out["clima_en_las_entradas"] = bool(sum(c.startswith("x_obs_") for c in datos.hist) == 24 and datos.contrato["numericas"]["rezago_clima_h"] == 168)
    return out


def texto(datos: DatosNumericos, canal: dict, cfg_texto: dict) -> dict:
    """Las listas por día coinciden con la regla del repositorio y ningún documento es posterior a su día."""
    out = {}
    corpus, lista = canal["corpus"], canal["lista"]
    n_dias = lista.shape[0]
    dias = pd.DatetimeIndex([datos.dia_base + pd.Timedelta(days=i) for i in range(n_dias)])
    disp = corpus["disponible_desde"].to_numpy()
    ultimo = np.where(lista >= 0, lista, 0).max(axis=1)
    con = (lista >= 0).any(axis=1)
    out["ningun_documento_posterior_a_su_dia"] = bool((pd.DatetimeIndex(disp[ultimo[con]], tz="UTC") <= dias[con].tz_convert("UTC")).all())
    g = np.random.default_rng(7)
    muestra = np.concatenate([g.choice(np.flatnonzero(con), size=min(40, int(con.sum())), replace=False), g.choice(n_dias, size=10, replace=False)])
    iguales = True
    for i in muestra:
        ref = documentos_en_origen(corpus, dias[i], cfg_texto["ventana_h"], cfg_texto["max_documentos_canal"]).index.to_numpy()
        iguales &= bool(np.array_equal(ref, lista[i][lista[i] >= 0]))
    out["listas_iguales_a_la_regla_del_repositorio"] = iguales
    return out


def _activar(red) -> None:
    """Da valores no nulos a los parámetros que empiezan en cero, para que las pruebas vean todas las rutas."""
    with torch.no_grad():
        for v in red.parameters():
            if float(v.abs().sum()) == 0.0:
                v.normal_(std=0.05)


def redes(datos: DatosNumericos, nombre: str, configs: list[dict], canal: dict | None, descripciones: np.ndarray | None) -> dict:
    """Formas, cobertura del horizonte y causalidad del canal de texto, con una versión pequeña de la red."""
    out = {}
    torch.manual_seed(0)
    p0 = configs[0]
    n_x, n_cal, B = len(datos.hist) + len(datos.nombres_mascara), len(datos.cal), 6
    x, cp, cf = torch.randn(B, datos.L, n_x), torch.randn(B, datos.L, n_cal), torch.randn(B, datos.H, n_cal)
    chico = dict(p0, dropout=0.0)
    chico.update({k: v for k, v in (("d_model", 16), ("capas", 1), ("cabezas", 4), ("capas_mezcla", 1), ("d_fusion", 16), ("anchos", [8, 8, 8])) if k in p0})
    if nombre == "tcn":
        out["tcn_campo_receptivo_cubre_la_historia_en_todas_las_configuraciones"] = all(R.campo_receptivo_tcn(p["kernel"]) >= datos.L for p in configs)
        red = R.construir(nombre, chico, n_x, n_cal, datos.L, datos.H).eval()
        xg = x.clone().requires_grad_(True)
        red(xg, cp, cf, None).sum().backward()
        out["tcn_la_salida_depende_de_la_primera_y_la_ultima_hora"] = bool(xg.grad[:, 0].abs().sum() > 0) and bool(xg.grad[:, -1].abs().sum() > 0)
        s = torch.cat([x, cp], dim=2).transpose(1, 2)
        with torch.no_grad():
            antes = red.bloques[0](s)
            s2 = s.clone(); s2[:, :, 100:] += 5.0                      # se alteran las horas 100 en adelante
            despues = red.bloques[0](s2)
        out["tcn_convoluciones_causales"] = bool(torch.equal(antes[:, :, :100], despues[:, :, :100])) and not bool(torch.equal(antes[:, :, 100:], despues[:, :, 100:]))
    desc = torch.from_numpy(descripciones) if descripciones is not None else None
    nucleos = ["itransformer", "patchtst"] if "nucleo" in p0 else [None]
    variantes = [(dict(chico, nucleo=n) if n else chico, n or nombre) for n in nucleos]
    if nombre in ("fusion_tardia", "tgforecaster"):
        out["historia_en_21_parches"] = R._parches(torch.zeros(2, datos.L)).shape[1] == 21
    cero = torch.zeros(B, dtype=torch.long)
    for pp, et in variantes:
        red = R.construir(nombre, pp, n_x, n_cal, datos.L, datos.H, None, desc).eval()
        with torch.no_grad():
            y = red(x, cp, cf, cero)
        out[f"{et}_N_salida_24_finita"] = tuple(y.shape) == (B, datos.H) and bool(torch.isfinite(y).all())
        if nombre == "tgforecaster":
            w = red.pesos_horizonte()
            out["horizonte_cubierto_y_pesos_suman_1"] = bool(red.cubre.any(0).all()) and bool(torch.allclose(w.sum(0), torch.ones(datos.H)))
            out["una_descripcion_por_canal"] = desc is not None and desc.shape[0] == n_x
    if canal is None:
        return out
    vec, lista = torch.from_numpy(canal["vectores"]).clone(), torch.from_numpy(canal["lista"])
    con = (lista >= 0).any(1)
    hueco = int(torch.nonzero(~con)[60])                                       # un día sin documentos en sus 72 h
    llenos = torch.nonzero(con).flatten()
    dia = torch.cat([torch.tensor([hueco]), llenos[torch.linspace(len(llenos) // 4, len(llenos) - 1, B - 1).long()]])
    for pp, et in variantes:
        red = R.construir(nombre, pp, n_x, n_cal, datos.L, datos.H, dict(vectores=vec, lista=lista), desc)
        _activar(red)
        red.eval()
        with torch.no_grad():
            z, hay, _ = red.texto_del_origen(x, dia)
            y0 = red(x, cp, cf, dia)
            out[f"{et}_salida_24_finita"] = tuple(y0.shape) == (B, datos.H) and bool(torch.isfinite(y0).all())
            out[f"{et}_texto_cero_sin_documentos"] = bool((z[0] == 0).all()) and float(hay[0]) == 0.0 and bool(z[1:].abs().sum(1).min() > 0)
            if nombre == "fusion_tardia":
                out[f"{et}_sin_documentos_es_el_nucleo_numerico"] = bool(torch.equal(y0[0], red.nucleo(x, cp, cf, None)[0]))
            # Se alteran todos los documentos que no son elegibles para estos orígenes: la salida no debe cambiar.
            usados = torch.unique(lista[dia])
            usados = usados[usados >= 0]
            otro = vec.clone(); marca = torch.ones(len(vec), dtype=torch.bool); marca[usados] = False
            otro[marca] = torch.randn(int(marca.sum()), vec.shape[1])
            red.relevancia.vectores = otro
            out[f"{et}_no_usa_documentos_no_elegibles"] = bool(torch.equal(y0, red(x, cp, cf, dia)))
            red.relevancia.vectores = torch.randn_like(vec)
            y2 = red(x, cp, cf, dia)
            out[f"{et}_el_texto_llega_a_la_salida"] = bool(torch.equal(y0[0], y2[0])) and bool(((y0[1:] - y2[1:]).abs().amax(1) > 0).all())
    return out

"""Comprobaciones previas al entrenamiento. Si alguna falla, no se entrena."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from demanda import config as C
from demanda.corpus import documentos_en_origen

from .datos import DatosNumericos
from .redes import Modelo6, NucleoPatchTST


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


def redes(datos: DatosNumericos, canal: dict | None) -> dict:
    out = {}
    torch.manual_seed(0)
    n_x, n_cal, B = len(datos.hist) + len(datos.nombres_mascara), len(datos.cal), 6
    x, cp, cf = torch.randn(B, datos.L, n_x), torch.randn(B, datos.L, n_cal), torch.randn(B, datos.H, n_cal)
    base = dict(d_model=16, capas=1, cabezas=4, dropout=0.0, d_texto=6)
    out["patchtst_21_parches"] = NucleoPatchTST(n_x, n_cal, 0, datos.L, datos.H, dict(base, nucleo="patchtst")).n_parches == 21
    if canal is None:
        return out
    vec, lista = torch.from_numpy(canal["vectores"]).clone(), torch.from_numpy(canal["lista"])
    con = (lista >= 0).any(1)
    hueco = next(int(d) for d in range(60, lista.shape[0]) if not con[d - 6:d + 1].any())       # semana sin documentos
    lleno = int(torch.nonzero(con)[len(torch.nonzero(con)) // 2])
    dia = torch.tensor([hueco, lleno, lleno + 1, lleno + 2, lleno + 30, lleno + 365]).clamp(max=lista.shape[0] - 400)
    dia[0] = hueco
    for nucleo in ("itransformer", "patchtst"):
        red = Modelo6(dict(base, nucleo=nucleo), n_x, n_cal, datos.L, datos.H, dict(vectores=vec, lista=lista)).eval()
        with torch.no_grad():
            canales, _ = red.agregador(x, dia)
            y0 = red(x, cp, cf, dia)
            out[f"{nucleo}_salida_24"] = tuple(y0.shape) == (B, datos.H)
            out[f"{nucleo}_texto_cero_sin_documentos"] = bool((canales[0] == 0).all()) and bool(canales[1].abs().sum() > 0)
            # Se alteran todos los documentos que no son elegibles para estos orígenes: la salida no debe cambiar.
            usados = torch.unique(lista[(dia[:, None] + torch.arange(-6, 1)[None, :]).clamp(min=0)])
            usados = usados[usados >= 0]
            otro = vec.clone(); marca = torch.ones(len(vec), dtype=torch.bool); marca[usados] = False
            otro[marca] = torch.randn(int(marca.sum()), vec.shape[1])
            red.agregador.vectores = otro
            out[f"{nucleo}_no_usa_documentos_no_elegibles"] = bool(torch.equal(y0, red(x, cp, cf, dia)))
            red.agregador.vectores = torch.randn_like(vec)
            y2 = red(x, cp, cf, dia)
            out[f"{nucleo}_el_texto_llega_a_la_salida"] = bool(torch.equal(y0[0], y2[0])) and not bool(torch.equal(y0[1:], y2[1:]))
    return out

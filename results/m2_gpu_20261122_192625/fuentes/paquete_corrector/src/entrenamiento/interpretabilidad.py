"""Interpretabilidad del original frente al mejor rival (§6.5 del Entregable 3), implementada desde su formulación.

Gradientes integrados (Sundararajan et al., 2017) de la salida media de las 24 h respecto de las entradas normalizadas,
con referencia x' = 0 (demanda igual al nivel de la ventana; covariables y calendario en su media de ajuste; indicadores
de ausencia en cero) y 32 pasos de la regla del punto medio:
        IG_i(x) = (x_i − x'_i) · (1/S) Σ_s ∂f(x' + (s − ½)/S · (x − x')) / ∂x_i
La atribución se agrega por grupos (demanda de cada uno de los 7 días previos, cada bloque de covariables, indicadores,
calendario pasado y del horizonte) como suma de valores absolutos y se normaliza a participaciones. El texto entra por
índices discretos, así que su importancia se mide por oclusión (salida con el texto frente a sin texto).
Fidelidad: borrar (llevar a la referencia) los k grupos más importantes frente a k grupos al azar (20 repeticiones),
diferencia de MASE. Estabilidad: correlación de Spearman entre semillas de las participaciones por grupo.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from scipy import stats

from . import motor
from . import protocolo_e2 as P2
from .protocolo import leer_json

PREFIJOS = {"precio": "x_precio_", "hidrologia": "x_hid_", "clima": "x_obs_"}


def grupos(datos) -> dict[str, tuple[str, list[int]]]:
    """nombre -> (tensor, índices de canal o de paso)."""
    g = {f"demanda_dia_{-d}": ("x0", list(range(datos.L - 24 * d, datos.L - 24 * (d - 1)))) for d in range(datos.L // 24, 0, -1)}
    usados = {0}
    for n, pre in PREFIJOS.items():
        js = [j for j, c in enumerate(datos.hist) if c.startswith(pre)]
        if js:
            g[n] = ("x", js)
            usados |= set(js)
    otras = [j for j in range(len(datos.hist)) if j not in usados]
    if otras:
        g["otras_historicas"] = ("x", otras)
    if datos.nombres_mascara:
        g["indicadores_ausencia"] = ("x", list(range(len(datos.hist), len(datos.hist) + len(datos.nombres_mascara))))
    g["calendario_pasado"] = ("cal_p", None)
    g["calendario_horizonte"] = ("cal_f", None)
    return g


def _ig(red, t: dict, pasos: int) -> dict[str, torch.Tensor]:
    with torch.backends.cudnn.flags(enabled=False):          # las LSTM de cuDNN no admiten retropropagación en modo evaluación
        return _ig_pasos(red, t, pasos)


def _ig_pasos(red, t: dict, pasos: int) -> dict[str, torch.Tensor]:
    acum = {k: torch.zeros_like(t[k]) for k in ("x", "cal_p", "cal_f")}
    for s in range(pasos):
        a = (s + 0.5) / pasos
        e = {k: (a * t[k]).detach().requires_grad_(True) for k in acum}
        red(e["x"], e["cal_p"], e["cal_f"], t["dia"]).mean(1).sum().backward()
        for k in acum:
            acum[k] += e[k].grad
    return {k: t[k] * acum[k] / pasos for k in acum}


def _por_grupo(attr: dict, g: dict) -> dict[str, float]:
    out = {}
    for n, (tensor, ix) in g.items():
        if tensor == "x0":
            out[n] = float(attr["x"][:, ix, 0].abs().sum())
        elif tensor == "x":
            out[n] = float(attr["x"][:, :, ix].abs().sum())
        else:
            out[n] = float(attr[tensor].abs().sum())
    tot = sum(out.values()) or 1.0
    return {k: v / tot for k, v in out.items()}


def _borrar(t: dict, g: dict, nombres: list[str]) -> dict:
    u = {k: v.clone() for k, v in t.items()}
    for n in nombres:
        tensor, ix = g[n]
        if tensor == "x0":
            u["x"][:, ix, 0] = 0.0
        elif tensor == "x":
            u["x"][:, :, ix] = 0.0
        else:
            u[tensor][:] = 0.0
    return u


def _mase(red, t, conj, datos) -> float:
    o = motor.predecir(red, t)
    return float(np.abs(o * conj.nivel[:, None] - conj.y).mean() / datos.q168)


def ejecutar(e3) -> dict:
    pi = e3.plan["interpretabilidad"]
    datos = e3.datos
    pos = datos.prueba()[::pi["paso_origenes"]]
    if e3.rapido:
        pos = pos[:16]
    rival = e3.mejor_rival() or "patchtst"
    g = grupos(datos)
    imp, fid, ocl = [], [], []
    gen = np.random.default_rng(20261118)
    for m in [m for m in e3.miembros() if m["clave"] in ("original", rival) and m["tipo"] == "red"]:
        carpetas = sorted((e3.salida / "familia" / m["clave"]).glob("semilla_*"), key=lambda p: int(p.name.split("_")[1]))[: (2 if e3.rapido else pi["semillas"])]
        for car in carpetas:
            esc = P2._escalas(leer_json(car / "escaladores.json"), datos)
            conj = datos.construir(pos, esc)
            red, r = e3.cargar_red(m, car, conj)
            t = motor.a_tensores(conj, e3.dispositivo)
            red.eval()
            partes = []
            for i in range(0, len(pos), 32):
                ti = {k: v[i:i + 32] for k, v in t.items()}
                partes.append({k: v.detach().cpu() for k, v in _ig(red, ti, 4 if e3.rapido else pi["pasos_ig"]).items()})
            attr = {k: torch.cat([p[k] for p in partes]) for k in ("x", "cal_p", "cal_f")}
            sh = _por_grupo(attr, g)
            imp += [dict(clave=m["clave"], semilla=r["semilla"], grupo=k, participacion=v) for k, v in sh.items()]
            base = _mase(red, t, conj, datos)
            orden = sorted(sh, key=sh.get, reverse=True)
            for k in pi["k_borrado"]:
                top = _mase(red, _borrar(t, g, orden[:k]), conj, datos) - base
                azar = [_mase(red, _borrar(t, g, list(gen.choice(list(g), k, replace=False))), conj, datos) - base for _ in range(3 if e3.rapido else pi["borrados_aleatorios"])]
                fid.append(dict(clave=m["clave"], semilla=r["semilla"], k=k, aumento_MASE_top=top, aumento_MASE_azar_medio=float(np.mean(azar)),
                                fraccion_azar_mayor_o_igual=float(np.mean(np.array(azar) >= top))))
            if m["variante"] == "N+O":
                red0, _ = e3.cargar_red(m, car, conj, "texto_ausente")
                o1, o0 = motor.predecir(red, t), motor.predecir(red0, t)
                ocl.append(dict(clave=m["clave"], semilla=r["semilla"], efecto_medio_abs_texto=float(np.abs(o1 - o0).mean()),
                                MASE_con_texto=base, MASE_sin_texto=_mase(red0, t, conj, datos)))
    ti, tf = pd.DataFrame(imp), pd.DataFrame(fid)
    sal = e3.salida / "interpretabilidad"
    sal.mkdir(parents=True, exist_ok=True)
    ti.to_csv(sal / "importancia_por_grupo.csv", index=False)
    tf.to_csv(sal / "fidelidad_borrado.csv", index=False)
    pd.DataFrame(ocl).to_csv(sal / "oclusion_texto.csv", index=False)
    est = []
    for k, d in ti.groupby("clave"):
        w = d.pivot(index="grupo", columns="semilla", values="participacion")
        rho = [stats.spearmanr(w[a], w[b]).statistic for i, a in enumerate(w.columns) for b in w.columns[i + 1:]]
        est.append(dict(clave=k, spearman_medio_entre_semillas=float(np.nanmean(rho)) if rho else None, pares=len(rho)))
    pd.DataFrame(est).to_csv(sal / "estabilidad_entre_semillas.csv", index=False)
    return dict(rival=rival, grupos=list(g), estabilidad=est)

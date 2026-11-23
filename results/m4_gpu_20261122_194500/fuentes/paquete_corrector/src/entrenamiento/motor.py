"""Entrenamiento de una red con las reglas del motor v3 del proyecto.

AdamW, pérdida MAE sobre demanda / nivel, ReduceLROnPlateau(factor 0.5, paciencia 2), recorte
de gradiente, parada cuando la época es >= 10 y hay 6 épocas sin mejora relativa de 0.2 %,
y restauración del mejor estado absoluto en la parada interna. Extensiones: si la red define ``perdida_adicional(t, ix)``,
ese término se suma a la pérdida de cada lote (Entregable 3); si define ``perdida_principal(salida, t, ix)``, reemplaza al MAE
del lote (exploración: DRO temporal). La parada sigue usando solo el MAE en todos los casos.

Receta opcional de promedio de pesos (exploración de E), leída de la configuración ``p``:
  rec_promedio = "ema"  media exponencial de todos los tensores de estado tras cada paso, con decaimiento
                        min(d, (1 + k) / (10 + k)) en el paso k (arranque gradual); d = 1 − 1 / (rec_epocas_ema · pasos por
                        época), es decir, una memoria de unas rec_epocas_ema épocas sea cual sea el tamaño de lote
  rec_promedio = "swa"  media aritmética de los estados al final de cada época desde la época rec_inicio; las
                        estadísticas de normalización por lotes se recalculan con una pasada sobre el ajuste
La parada interna, el planificador y el estado restaurado usan el modelo promediado: el resultado es un único modelo
con los mismos parámetros que la base (exportable igual que ella).
"""
from __future__ import annotations

import random
import time

import numpy as np
import torch
from torch import nn


def fijar_semilla(semilla: int) -> None:
    random.seed(semilla)
    np.random.seed(semilla)
    torch.manual_seed(semilla)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(semilla)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def a_tensores(conj, dev) -> dict:
    t = dict(x=torch.from_numpy(conj.x).to(dev), cal_p=torch.from_numpy(conj.cal_p).to(dev), cal_f=torch.from_numpy(conj.cal_f).to(dev),
             dia=torch.from_numpy(np.array(conj.dia, dtype="int64")).to(dev), y=torch.from_numpy(conj.y_norm).to(dev),
             nivel=torch.from_numpy(np.asarray(conj.nivel, dtype="float32")).to(dev))
    for k in ("sig", "grupo"):                  # solo los usan los mecanismos de la exploración (coherencia y DRO temporal)
        if getattr(conj, k, None) is not None:
            t[k] = torch.from_numpy(np.asarray(getattr(conj, k), dtype="int64")).to(dev)
    return t


def _salida(red, t, ix):
    return red(t["x"][ix], t["cal_p"][ix], t["cal_f"][ix], t["dia"][ix])


@torch.no_grad()
def predecir(red, t: dict, lote: int = 256) -> np.ndarray:
    """Predicción normalizada (demanda / nivel) en modo evaluación."""
    red.eval()
    n = len(t["y"])
    partes = [_salida(red, t, slice(i, i + lote)).float().cpu().numpy() for i in range(0, n, lote)]
    return np.concatenate(partes, axis=0)


@torch.no_grad()
def _perdida(red, t: dict, lote: int = 256) -> float:
    red.eval()
    n, tot = len(t["y"]), 0.0
    for i in range(0, n, lote):
        s = slice(i, i + lote)
        tot += float((_salida(red, t, s) - t["y"][s]).abs().mean()) * len(t["y"][s])
    return tot / n


def _flotantes(red) -> dict:
    return {k: v.detach().clone() for k, v in red.state_dict().items()}


@torch.no_grad()
def _recalibrar_bn(red, tr: dict, lote: int) -> None:
    capas = [m for m in red.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    if not capas:
        return
    momentos = {}
    for m in capas:
        m.reset_running_stats()
        momentos[m], m.momentum = m.momentum, None              # media acumulada exacta
    red.train()
    n = len(tr["y"])
    for i in range(0, n, lote):
        ix = torch.arange(i, min(n, i + lote), device=tr["x"].device)
        if len(ix) > 1:
            _salida(red, tr, ix)
    for m in capas:
        m.momentum = momentos[m]


def entrenar(red, tr: dict, es: dict, p: dict, semilla: int, opciones: dict, observador=None) -> dict:
    """Ajusta ``red`` con ``tr`` y detiene con ``es``. Devuelve la historia y deja en ``red`` el mejor estado."""
    dev = tr["x"].device
    perdida = nn.L1Loss()
    opt = torch.optim.AdamW(red.parameters(), lr=float(p["lr"]), weight_decay=float(p["weight_decay"]))
    plan = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
    mejor = referencia = float("inf")
    estado, sin_mejora, historia = None, 0, []
    gen = torch.Generator().manual_seed(int(semilla))
    lote, n = int(p["lote"]), len(tr["y"])
    tipo_prom = p.get("rec_promedio")
    if tipo_prom not in (None, "ema", "swa"):
        raise ValueError(f"Promedio de pesos desconocido: {tipo_prom}")
    prom, n_prom, pasos = None, 0, 0
    t0 = time.time()
    for ep in range(1, int(opciones["max_epocas"]) + 1):
        red.train()
        tot = 0.0
        for ix in torch.randperm(n, generator=gen).split(lote):
            if len(ix) < 2:      # la normalización por lotes necesita más de una muestra
                continue
            ix = ix.to(dev)
            opt.zero_grad(set_to_none=True)
            salida = _salida(red, tr, ix)
            if hasattr(red, "perdida_principal"):        # exploración: DRO temporal por grupos de años (la parada sigue con el MAE)
                loss = red.perdida_principal(salida, tr, ix)
            else:
                loss = perdida(salida, tr["y"][ix])
            if hasattr(red, "perdida_adicional"):        # esquema de entrenamiento propio (idea C: texto placebo)
                loss = loss + red.perdida_adicional(tr, ix)
            if not torch.isfinite(loss):
                raise FloatingPointError("Pérdida no finita")
            loss.backward()
            nn.utils.clip_grad_norm_(red.parameters(), float(opciones["recorte_gradiente"]))
            opt.step()
            pasos += 1
            if tipo_prom == "ema":
                with torch.no_grad():
                    if prom is None:
                        prom = _flotantes(red)
                    else:
                        d = min(1.0 - 1.0 / (float(p.get("rec_epocas_ema", 2.0)) * max(1, n // lote)), (1 + pasos) / (10 + pasos))
                        for k, v in red.state_dict().items():
                            if v.dtype.is_floating_point:
                                prom[k].mul_(d).add_(v.detach(), alpha=1 - d)
                            else:
                                prom[k].copy_(v)
            tot += float(loss.detach()) * len(ix)
        if tipo_prom == "swa" and ep >= int(p.get("rec_inicio", 5)):
            with torch.no_grad():
                if prom is None:
                    prom, n_prom = _flotantes(red), 1
                else:
                    n_prom += 1
                    for k, v in red.state_dict().items():
                        if v.dtype.is_floating_point:
                            prom[k].add_((v.detach() - prom[k]) / n_prom)
                        else:
                            prom[k].copy_(v)
        en_curso = None
        if prom is not None:                                   # se evalúa (y se guarda) el modelo promediado
            en_curso = _flotantes(red)
            red.load_state_dict(prom)
            if tipo_prom == "swa":
                _recalibrar_bn(red, tr, int(opciones.get("lote_evaluacion", 256)))
                prom = _flotantes(red)
        vl = _perdida(red, es, int(opciones.get("lote_evaluacion", 256)))
        historia.append(dict(epoca=ep, perdida_ajuste=tot / n, perdida_parada=vl, lr=opt.param_groups[0]["lr"], segundos=round(time.time() - t0, 1)))
        if vl < mejor:
            mejor = vl
            estado = {k: v.detach().clone() for k, v in red.state_dict().items()}
        if en_curso is not None:
            red.load_state_dict(en_curso)                      # el entrenamiento sigue con los pesos sin promediar
        if not np.isfinite(referencia) or vl < referencia - max(1e-6, 0.002 * abs(referencia)):
            referencia, sin_mejora = vl, 0
        else:
            sin_mejora += 1
        plan.step(vl)
        if observador is not None:
            observador(historia[-1])
        if ep >= int(opciones["min_epocas"]) and sin_mejora >= int(opciones["paciencia"]):
            break
    red.load_state_dict(estado)
    return dict(optimizador=opt, epocas=len(historia), mejor_epoca=int(np.argmin([h["perdida_parada"] for h in historia]) + 1),
                mejor_perdida_parada=float(mejor), historia=historia, segundos=round(time.time() - t0, 1))

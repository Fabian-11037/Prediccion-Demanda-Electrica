"""Optimización y serialización conservadas de M4_N."""
from pathlib import Path
import copy, json, time, random
import numpy as np
import pandas as pd
import torch
from torch import nn
from entrenamiento import motor

def cpu(obj):
    if torch.is_tensor(obj): return obj.detach().cpu().clone()
    if isinstance(obj, dict): return {k: cpu(v) for k, v in obj.items()}
    if isinstance(obj, list): return [cpu(v) for v in obj]
    if isinstance(obj, tuple): return tuple(cpu(v) for v in obj)
    return copy.deepcopy(obj)

def checkpoint(p, valor):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_name(p.name + '.tmp'); torch.save(cpu(valor), t); t.replace(p)

def cargar(p):
    return torch.load(p, map_location='cpu', weights_only=False)

def guardar(p, valor):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_name(p.name + '.tmp')
    t.write_text(json.dumps(valor, ensure_ascii=False, indent=1, default=str), encoding='utf-8'); t.replace(p)

def ajustar(red, n, perdida, evaluar, p, opciones, semilla, carpeta, progreso, etiqueta):
    """Pérdida/evaluador solo reciben índices de ajuste o parada. Checkpoint completo por época."""
    assert next(red.parameters()).is_cuda, 'No se permite ajuste neuronal en CPU'
    carpeta = Path(carpeta); carpeta.mkdir(parents=True, exist_ok=True)
    opt = torch.optim.AdamW(red.parameters(), lr=p['lr'], weight_decay=p['weight_decay'])
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=.5, patience=2) if opciones.get('scheduler') else None
    gen = torch.Generator().manual_seed(semilla)
    historia, inicio, mejor, referencia, sin = [], 1, float('inf'), float('inf'), 0
    mejor_estado = None
    parcial = carpeta / 'reanudar.pt'
    if parcial.exists():
        r = cargar(parcial); red.load_state_dict(r['ultimo_modelo']); opt.load_state_dict(r['optimizador'])
        if sch: sch.load_state_dict(r['scheduler'])
        historia, inicio, mejor, referencia, sin = r['historia'], r['epoca']+1, r['mejor'], r['referencia'], r['sin_mejora']
        mejor_estado = r['mejor_modelo']; gen.set_state(r['generador'])
        torch.set_rng_state(r['rng_torch']); torch.cuda.set_rng_state_all(r['rng_cuda']); np.random.set_state(r['rng_numpy']); random.setstate(r['rng_python'])
        if sin>=opciones['paciencia'] and r['epoca']>=opciones.get('min_epocas',1):inicio=opciones['max_epocas']+1
    t0 = time.time()
    for ep in range(inicio, opciones['max_epocas']+1):
        red.train(); total = 0.; vistos = 0
        for ix in torch.randperm(n, generator=gen).split(p['lote']):
            if len(ix)<2: continue
            opt.zero_grad(set_to_none=True)
            loss = perdida(ix.to('cuda'))
            if not torch.isfinite(loss): raise FloatingPointError(etiqueta + ': pérdida no finita')
            loss.backward(); nn.utils.clip_grad_norm_(red.parameters(), 1.); opt.step()
            total += float(loss.detach())*len(ix); vistos += len(ix)
        red.eval()
        with torch.no_grad(): vl = float(evaluar())
        if not np.isfinite(vl): raise FloatingPointError('Parada no finita')
        h = dict(epoca=ep, perdida_ajuste=total/max(vistos,1), perdida_parada=vl, lr=opt.param_groups[0]['lr'], segundos=time.time()-t0)
        historia.append(h)
        if vl < mejor:
            mejor = vl; mejor_estado = cpu(red.state_dict())
        delta = max(1e-6, .002*abs(referencia)) if opciones.get('relativa') and np.isfinite(referencia) else opciones.get('min_delta', 1e-6)
        if vl < referencia-delta: referencia, sin = vl, 0
        else: sin += 1
        if sch: sch.step(vl)
        checkpoint(parcial, dict(ultimo_modelo=red.state_dict(), mejor_modelo=mejor_estado, optimizador=opt.state_dict(),
                  scheduler=sch.state_dict() if sch else None, historia=historia, epoca=ep, mejor=mejor, referencia=referencia,
                  sin_mejora=sin, generador=gen.get_state(), rng_torch=torch.get_rng_state(), rng_cuda=torch.cuda.get_rng_state_all(),
                  rng_numpy=np.random.get_state(), rng_python=random.getstate(), configuracion=p, opciones=opciones, semilla=semilla))
        pd.DataFrame(historia).to_csv(carpeta/'historia.csv', index=False)
        progreso(etiqueta=etiqueta, epoca=ep, max_epocas=opciones['max_epocas'], perdida_parada=vl)
        if ep >= opciones.get('min_epocas',1) and sin >= opciones['paciencia']: break
    if mejor_estado is None: raise RuntimeError('No hay estado óptimo')
    red.load_state_dict(mejor_estado); red.eval()
    return dict(epocas=len(historia), mejor_epoca=int(np.argmin([x['perdida_parada'] for x in historia])+1), mejor_perdida=mejor,
                segundos=time.time()-t0, historia=historia)
"""Motor y utilidades de la corrida final del corrector y desarrollo de M1."""
from __future__ import annotations
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

REPO = Path(__file__).resolve().parents[1]
PAQUETE = REPO.parent / 'Corrector_Semantico_E3'
os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '10.3.0')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('HF_HOME', str(PAQUETE / '.cache_huggingface'))
sys.path[:0] = [str(PAQUETE / '.deps'), str(REPO.parent / '.demanda-python-deps'), str(PAQUETE / 'src')]
import numpy as np
import pandas as pd
import torch
from torch import nn
from entrenamiento import motor, redes_e3 as redes
from entrenamiento.datos import DatosNumericos, escalas_a_json
from corrector import residuos as RS

def leer(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))

def guardar(p, valor):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_name(p.name + '.tmp')
    t.write_text(json.dumps(valor, ensure_ascii=False, indent=1, default=str), encoding='utf-8'); t.replace(p)

def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def verificar_json_portable(p,esperado):
    """Admite únicamente la diferencia LF/CRLF, conservando todos los demás bytes."""
    b=Path(p).read_bytes();json.loads(b)
    variantes={'original':b,'LF':b.replace(b'\r\n',b'\n'),'CRLF':b.replace(b'\r\n',b'\n').replace(b'\n',b'\r\n')}
    coincide=[k for k,v in variantes.items() if hashlib.sha256(v).hexdigest()==esperado]
    if not coincide:raise ValueError('El JSON difiere del diseño más allá de sus finales de línea: '+str(p))
    return dict(sha256_local=hashlib.sha256(b).hexdigest(),sha256_diseno=esperado,equivalencia=coincide)

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

def metricas(y, pred, q):
    e = np.asarray(y, dtype=np.float64) - np.asarray(pred, dtype=np.float64)
    return dict(MASE=float(np.abs(e).mean()/q), MAE_kwh=float(np.abs(e).mean()), RMSE_kwh=float(np.sqrt((e*e).mean())))

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

REDES = {'E':'original_E', 'patchtst':'tats', 'periodica':'x_periodica'}

def construir_experto(datos, nombre, p):
    return redes.construir(REDES[nombre], p, len(datos.hist)+len(datos.nombres_mascara), len(datos.cal), datos.L, datos.H,
                          n_mascaras=len(datos.nombres_mascara))

def experto(datos, nombre, p, semilla, anio, carpeta, progreso, esperado):
    """Ajuste histórico o final SIN acceso a etiquetas de prueba durante la optimización."""
    carpeta = Path(carpeta)
    if (carpeta/'TERMINADO.json').exists(): return leer(carpeta/'TERMINADO.json')
    motor.fijar_semilla(semilla)
    if anio == 'final':
        tr, es = datos.ajuste_final(); ev = None
    else:
        pos = RS.pliegue_anual(datos, anio); tr, es, ev = pos['tr'], pos['es'], pos['ev']
    assert datos.indice[tr].max()+pd.Timedelta(hours=23) < datos.indice[es].min()
    assert datos.indice[es].max().year <= (2021 if anio=='final' else anio-1)
    esc = datos.ajustar_escalas(tr)
    conj = {k:datos.construir(v,esc) for k,v in [('tr',tr),('es',es)]}
    tens = {k:motor.a_tensores(v,'cuda') for k,v in conj.items()}
    red = construir_experto(datos,nombre,p).cuda()
    assert red.n_parametros()==esperado,(nombre,red.n_parametros(),esperado)
    def salida(k,ix):
        z=tens[k];return red(z['x'][ix],z['cal_p'][ix],z['cal_f'][ix],z['dia'][ix])
    def loss(ix): return (salida('tr',ix)-tens['tr']['y'][ix]).abs().mean()
    def val():
        return sum(float((salida('es',slice(i,i+256))-tens['es']['y'][i:i+256]).abs().mean())*len(tens['es']['y'][i:i+256]) for i in range(0,len(es),256))/len(es)
    opciones=dict(max_epocas=200,min_epocas=10,paciencia=6,relativa=True,scheduler=True)
    info=ajustar(red,len(tr),loss,val,p,opciones,semilla,carpeta,progreso,f'{nombre} · {anio} · s{semilla}')
    met=dict(nombre=nombre,configuracion=p,semilla=semilla,red=REDES[nombre],n_parametros=red.n_parametros(),
             ajuste=[str(conj['tr'].origen[0]),str(conj['tr'].origen[-1])],parada=[str(conj['es'].origen[0]),str(conj['es'].origen[-1])],
             opciones=opciones,dispositivo='cuda',**{k:v for k,v in info.items() if k!='historia'})
    checkpoint(carpeta/'modelo.pt',dict(modelo=red.state_dict(),escalas=esc,metadatos=met))
    guardar(carpeta/'escaladores.json',escalas_a_json(esc,datos))
    # Restauración independiente: salida idéntica antes de cualquier evaluación externa.
    z=tens['es']
    with torch.no_grad():antes=salida('es',slice(0,2)).detach()
    restaurada=construir_experto(datos,nombre,p).cuda();restaurada.load_state_dict(cargar(carpeta/'modelo.pt')['modelo']);restaurada.eval()
    with torch.no_grad(): despues=restaurada(z['x'][:2],z['cal_p'][:2],z['cal_f'][:2],z['dia'][:2])
    assert torch.equal(antes,despues),'Restauración no idéntica'
    del restaurada,tens,conj
    if ev is not None:
        c=datos.construir(ev,esc);t=motor.a_tensores(c,'cuda');pred=motor.predecir(red,t,256)*c.nivel[:,None]
        np.savez_compressed(carpeta/'predicciones.npz',dia=c.origen.strftime('%Y-%m-%d').to_numpy().astype('U10'),y=c.y.astype('float32'),pred=pred.astype('float32'))
        met['metricas']=metricas(c.y,pred,datos.q168)
        del c,t
    guardar(carpeta/'TERMINADO.json',met)
    del red;torch.cuda.empty_cache()
    return met

def lags_diarios(dias, errores):
    """Rasgos causales: errores d−2 y media d−8..d−2; nunca el objetivo actual."""
    dias=np.asarray(dias,dtype='datetime64[D]'); z=np.zeros(errores.shape[1:],dtype='float32'); mapa={d:i for i,d in enumerate(dias)}; un=np.timedelta64(1,'D')
    l2=np.stack([errores[mapa[d-2*un]] if d-2*un in mapa else z for d in dias])
    m7=np.stack([np.mean([errores[mapa[d-k*un]] for k in range(2,9) if d-k*un in mapa] or [z],axis=0) for d in dias])
    return np.concatenate([l2.reshape(len(dias),-1),m7.reshape(len(dias),-1)],axis=1)

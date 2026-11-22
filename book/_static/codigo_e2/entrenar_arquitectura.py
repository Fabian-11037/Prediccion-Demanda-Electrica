"""Entrenamiento portable de las configuraciones conservadas en el benchmark.

Uso: python scripts/entrenar_arquitectura.py --modelo tcn --salida ruta_nueva
--smoke comprueba optimización, guardado y restauración con datos de desarrollo.
La prueba 2022–2025 exige --fase prueba --allow-test. No se llama desde el build.
Cada modelo se ejecuta en un proceso nuevo para aislar las fuentes congeladas.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = Path(__file__).resolve().parents[1]
Q168 = 215617.90361589016


def huella(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def leer(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def jsonable(obj):
    if hasattr(obj, 'tolist'):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    return obj


def registro(raiz=ROOT):
    return leer(Path(raiz) / 'config/entrenamiento_congelado.json')


def comprobar_solicitud(modelo, fase, anio, permitir_prueba, raiz):
    # Guardas antes de importar Torch, cargar paneles o crear carpetas.
    if modelo not in registro(raiz):
        raise ValueError('Modelo no registrado: ' + modelo)
    if fase not in ('validacion', 'prueba'):
        raise ValueError('Fase desconocida')
    if fase == 'prueba' and not permitir_prueba:
        raise PermissionError('La prueba requiere --allow-test y autorización de evaluación final')
    if fase == 'validacion' and anio not in (2018, 2019, 2020, 2021):
        raise ValueError('Validación limitada a los años 2018–2021')


def fuentes(raiz, ficha):
    if any(k in sys.modules for k in ('demanda', 'entrenamiento', 'modelo6')):
        raise RuntimeError('Ejecute cada arquitectura en un proceso nuevo')
    origen = Path(raiz) / ficha['fuentes'] / 'src'
    if not origen.is_dir():raise FileNotFoundError('Faltan las fuentes de la arquitectura: '+str(origen))
    sys.path.insert(0, str(origen))


def texto_congelado(raiz, indice):
    import numpy as np
    import pandas as pd
    import torch
    ruta = Path(raiz) / 'data/processed/entrenamiento_texto'
    for name, meta in leer(ruta / 'manifiesto.json').items():
        if huella(ruta / name) != meta['sha256']:
            raise RuntimeError('Vector textual alterado: ' + name)
    corpuspath=Path(raiz) / 'results/evaluacion_e2/entradas/texto/corpus_oficial.parquet'
    if huella(corpuspath)!=leer(ruta/'procedencia.json')['corpus_sha256']:
        raise RuntimeError('El corpus no corresponde al de los vectores')
    v = torch.from_numpy(np.load(ruta / 'vectores_documento.npy', allow_pickle=False).astype(np.float32))
    d = torch.from_numpy(np.load(ruta / 'descripciones.npy', allow_pickle=False).astype(np.float32))
    corpus = pd.read_parquet(Path(raiz) / 'results/evaluacion_e2/entradas/texto/corpus_oficial.parquet')
    fechas = pd.DatetimeIndex(pd.to_datetime(corpus.disponible_desde, utc=True)).as_unit('ns').asi8
    if len(v) != len(corpus) or np.any(np.diff(fechas) < 0):
        raise ValueError('El orden del corpus no corresponde al de los vectores')
    dias = pd.date_range(indice[0].normalize(), indice[-1].normalize(), freq='D').tz_convert('UTC').as_unit('ns').asi8
    hi = np.searchsorted(fechas, dias, side='right')
    lo = np.maximum(np.searchsorted(fechas, dias - 72 * 3600 * 10**9, side='left'), hi - 64)
    lista = np.full((len(dias), 64), -1, dtype=np.int64)
    for i, (a, b) in enumerate(zip(lo, hi)):
        lista[i, :b-a] = np.arange(a, b)
    return dict(vectores=v, lista=torch.from_numpy(lista)), d


def crear_red(modelo, ficha, forma, texto=None, descripciones=None):
    import numpy as np
    p = ficha['params']
    if ficha['grupo'] == 'base':
        from demanda import modelos_base as M, reentrenamiento_v3 as V3, variantes_v4 as V4
        tr = SimpleNamespace(x=np.zeros((2, forma['L'], forma['nx']), dtype='float32'),
                             cal_p=np.zeros((2, forma['L'], forma['nc']), dtype='float32'),
                             cal_f=np.zeros((2, forma['H'], forma['nc']), dtype='float32'))
        if modelo == 'lstm':
            return V4.Secuencial(modelo, p, tr)
        if modelo == 'cnn1d':
            return V3.Secuencial(modelo, p, tr)
        return M.construir_red(modelo, p, tr)
    if ficha['grupo'] == 'nhits':
        from demanda.nhits_numerico import NHITSNumerico
        return NHITSNumerico(p, forma['sigma_demanda'])
    if modelo in ('patchtst', 'tats'):
        from modelo6.redes import Modelo6
        return Modelo6(p, forma['nx'], forma['nc'], forma['L'], forma['H'], texto)
    mod = importlib.import_module('entrenamiento.redes_e2' if modelo in ('nbeats', 'timesnet') else 'entrenamiento.redes')
    return mod.construir('tgforecaster' if modelo == 'tgforecaster_n' else modelo, p,
                         forma['nx'], forma['nc'], forma['L'], forma['H'], texto, descripciones)


def preparar(raiz, modelo, ficha, fase, anio, smoke):
    import numpy as np
    import pandas as pd
    from demanda import particion as P
    raiz = Path(raiz)
    contrato = leer(raiz / 'config/diseno_entrenamiento_texto.json')
    panelpath = raiz / 'data/processed/panel_horario.parquet'
    if huella(panelpath) != contrato['panel_sha256']:
        raise RuntimeError('El panel no coincide con el contrato')
    panel = pd.read_parquet(panelpath)
    reg = leer(raiz / 'data/processed/feature_registry.json')
    if smoke:
        # Tramos disjuntos anteriores a 2022; no consulta del conjunto de prueba.
        dias = [pd.date_range(a, periods=n, freq='D', tz='America/Bogota') for a,n in
                [('2020-08-01', 8), ('2020-09-01', 2), ('2020-10-01', 2)]]
        pos = [panel.index.get_indexer(d) for d in dias]
    elif fase == 'prueba':
        tr, es = P.pretest(panel.index, 24, 168, 24, 0, .10)
        pos = [tr, es, P.origenes(panel.index, 'test', 24, 168, 24, 0, allow_test=True)]
    else:
        f = P.folds_forward_chaining(panel.index, [anio], 24, 168, 24, 0, .10)[0]
        pos = [f[k] for k in ('tr', 'es', 'val')]
    if any(len(p) < 2 or (p < 168).any() for p in pos):
        raise ValueError('Particiones vacías o ventanas insuficientes')
    for a, b in zip(pos, pos[1:]):
        if panel.index[a.max()] + pd.Timedelta(hours=23) >= panel.index[b.min()]:
            raise ValueError('Las etiquetas de las particiones se solapan')
    if ficha['grupo'] == 'base':
        from demanda import modelos_base as M, reentrenamiento_v3 as V
        cfg = M.ConfigDatos(covariables='calendario', normalizacion='nivel', ventana=168)
        datos = V.DatosFold(panel, reg, cfg, pos[0], secuencial=modelo in ('lstm','cnn1d'))
        conjuntos = [datos.conjunto_pos(p) for p in pos]
        escalas = datos.escaladores()
        forma = dict(nx=conjuntos[0].x.shape[2], nc=23, L=168, H=24)
    elif ficha['grupo'] == 'nhits':
        from demanda.patchtst_numerico import Preparador
        datos = Preparador(panel, reg, contrato)
        pos = [datos.elegibles(p) for p in pos]
        datos.ajustar(pos[0])
        conjuntos = [datos.transformar(p) for p in pos]
        escalas = datos.stats
        forma = dict(nx=83, nc=23, L=168, H=24, sigma_demanda=datos.stats['sigma_demanda'])
    else:
        ns = 'modelo6' if modelo in ('patchtst','tats') else 'entrenamiento'
        dm = importlib.import_module(ns + '.datos')
        datos = dm.DatosNumericos(raiz)
        pos = [datos.elegibles(p) for p in pos]
        esc = datos.ajustar_escalas(pos[0])
        conjuntos = [datos.construir(p, esc) for p in pos]
        escalas = dm.escalas_a_json(esc, datos)
        forma = dict(nx=conjuntos[0].x.shape[2], nc=23, L=168, H=24)
    for c in conjuntos:
        if len(c.y) < 2 or not np.isfinite(c.x).all():
            raise ValueError('Conjunto sin datos o entradas no finitas')
    return datos, conjuntos, escalas, forma, panel.index


def tensores(c, grupo, datos):
    import numpy as np
    import torch
    if grupo == 'base':
        x = np.concatenate([c.x, c.cal_p], axis=2)
        return dict(x=torch.from_numpy(x), cf=torch.from_numpy(c.cal_f),
                    y=torch.from_numpy(datos.objetivo_norm(c)))
    if grupo == 'nhits':
        return dict(x=torch.from_numpy(c.x), cf=torch.from_numpy(c.calendario),
                    y=torch.from_numpy((c.y/c.nivel[:,None]).astype('float32')))
    return dict(x=torch.from_numpy(c.x), cp=torch.from_numpy(c.cal_p), cf=torch.from_numpy(c.cal_f),
                dia=torch.from_numpy(np.array(c.dia, dtype=np.int64, copy=True)), y=torch.from_numpy(c.y_norm))


def adelante(red, t, ix, dev):
    a = {k:v[ix].to(dev) for k,v in t.items() if k != 'y'}
    if 'dia' in a:
        return red(a['x'], a['cp'], a['cf'], a['dia'])
    return red(a['x'], a['cf'])


def predecir(red, t, dev, lote):
    import numpy as np
    import torch
    red.eval()
    with torch.no_grad():
        partes = [adelante(red,t,slice(i,i+lote),dev).float().cpu().numpy() for i in range(0,len(t['y']),lote)]
    pred = np.concatenate(partes)
    if pred.shape != tuple(t['y'].shape) or not np.isfinite(pred).all():
        raise FloatingPointError('Predicciones no finitas o de forma incorrecta')
    return pred


def optimizar(red, tr, es, ficha, modelo, semilla, dev, epocas):
    import numpy as np
    import torch
    from torch import nn
    p=ficha['params']; base=ficha['grupo']=='base'; nhits=ficha['grupo']=='nhits'
    perdida={'mae':nn.L1Loss(), 'mse':nn.MSELoss(), 'huber':nn.HuberLoss(delta=.05)}[p.get('perdida','mae')]
    opt=torch.optim.AdamW(red.parameters(), lr=p['lr'], weight_decay=p.get('weight_decay',0.))
    scheduler=p.get('scheduler','plateau')
    plan=(torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epocas) if scheduler=='cosine' else
          torch.optim.lr_scheduler.ReduceLROnPlateau(opt,factor=.5,patience=2) if scheduler=='plateau' else None)
    lote=int(p.get('lote',p.get('batch_size',64)))
    # TimesNet: periodos FFT definidos por el lote; evaluación estable por origen.
    lote_ev=1 if modelo=='timesnet' else (lote if nhits else (512 if base else 256))
    gen=torch.Generator().manual_seed(semilla)
    mejor=referencia=float('inf'); estado=None; stale=0; historia=[]; pasos=0; gradientes=[]; modulos_grad={}
    paciencia=10 if nhits else 6
    for ep in range(1,epocas+1):
        red.train(); total=0.; n=0
        partes=list(torch.randperm(len(tr['y']),generator=gen).split(lote))
        # Incorporar el último elemento al lote anterior, sin omitir etiquetas.
        if len(partes)>1 and len(partes[-1])==1:
            partes[-2]=torch.cat(partes[-2:]);partes.pop()
        for ix in partes:
            opt.zero_grad(set_to_none=True)
            loss=perdida(adelante(red,tr,ix,dev),tr['y'][ix].to(dev))
            penalizacion=red.penalizacion() if hasattr(red,'penalizacion') else 0.
            objetivo=loss+penalizacion
            if not torch.isfinite(objetivo):raise FloatingPointError('Pérdida no finita')
            objetivo.backward()
            norma=nn.utils.clip_grad_norm_(red.parameters(),1.,error_if_nonfinite=True)
            for nombre, par in red.named_parameters():
                if par.grad is not None:
                    clave=nombre.split('.')[0]
                    modulos_grad[clave]=max(modulos_grad.get(clave,0.),float(par.grad.norm()))
            gradientes.append(float(norma));opt.step();pasos+=1
            total+=float(loss.detach())*len(ix);n+=len(ix)
        red.eval();vl=0.
        with torch.no_grad():
            for i in range(0,len(es['y']),lote_ev):
                ix=slice(i,i+lote_ev)
                vl+=float(perdida(adelante(red,es,ix,dev),es['y'][ix].to(dev)))*len(es['y'][ix])
        vl/=len(es['y'])
        if not np.isfinite(vl):raise FloatingPointError('Parada no finita')
        historia.append(dict(epoca=ep,perdida_ajuste=total/n,perdida_parada=vl,lr=opt.param_groups[0]['lr'],muestras_ajuste=n))
        if vl<mejor:
            mejor=vl;estado={k:v.detach().cpu().clone() for k,v in red.state_dict().items()}
        if not np.isfinite(referencia) or vl<referencia-max(1e-6,.002*abs(referencia)):
            referencia=vl;stale=0
        else:stale+=1
        if plan is not None:
            plan.step(vl) if scheduler=='plateau' else plan.step()
        if ep>=10 and stale>=paciencia:break
    red.load_state_dict(estado,strict=True)
    if not pasos or not any(g>0 for g in gradientes):raise RuntimeError('No se produjo una actualización con gradiente')
    return historia,dict(pasos=pasos,norma_gradiente_max=max(gradientes),lote_evaluacion=lote_ev,gradientes_por_modulo=modulos_grad)


def cargar_modelo(raiz, ruta, dispositivo='cpu'):
    """Restauración independiente: fábrica original, buffers, configuración y escalas."""
    import numpy as np
    import torch
    ck=torch.load(ruta,map_location='cpu',weights_only=True)
    ficha=ck['ficha'];modelo=ck['modelo']
    if 'demanda' not in sys.modules:fuentes(raiz,ficha)
    elif not Path(sys.modules['demanda'].__file__).resolve().is_relative_to((Path(raiz)/ficha['fuentes']/'src').resolve()):
        raise RuntimeError('Las fuentes cargadas pertenecen a otra arquitectura; use un proceso nuevo')
    texto=descripciones=None
    if ficha['grupo']=='ampliado' and (ficha.get('variante')=='N+O' or modelo.startswith('tgforecaster')):
        # Estos buffers no son persistentes en las clases originales.
        texto=ck['entradas_texto']['texto']
        descripciones=ck['entradas_texto']['descripciones']
    red=crear_red(modelo,ficha,ck['forma'],texto,descripciones)
    if ficha['grupo']=='base' and hasattr(red,'preparar'):
        f=ck['forma'];red.preparar(torch.zeros(2,f['L'],f['nx']+f['nc']),torch.zeros(2,f['H'],f['nc']))
    red.load_state_dict(ck['state_dict'],strict=True);red.to(dispositivo);red.eval()
    return red,ck



def inferir(raiz, checkpoint, origenes, dispositivo='cpu', permitir_prueba=False):
    """Predice con las escalas guardadas, sin ajustar transformaciones ni parámetros."""
    import numpy as np
    import pandas as pd
    fechas=pd.DatetimeIndex(origenes)
    if fechas.tz is None:raise ValueError('Los orígenes necesitan zona horaria')
    fechas=fechas.tz_convert('America/Bogota')
    if fechas.empty or fechas.has_duplicates:raise ValueError('Orígenes vacíos o repetidos')
    if (fechas>=pd.Timestamp('2022-01-01',tz='America/Bogota')).any() and not permitir_prueba:
        raise PermissionError('La predicción de prueba requiere autorización explícita')
    red,ck=cargar_modelo(raiz,checkpoint,dispositivo)
    raiz=Path(raiz);ficha=ck['ficha'];sc=ck['escaladores'];grupo=ficha['grupo']
    contrato=leer(raiz/'config/diseno_entrenamiento_texto.json');path=raiz/'data/processed/panel_horario.parquet'
    if huella(path)!=contrato['panel_sha256']:raise RuntimeError('Panel alterado')
    panel=pd.read_parquet(path);reg=leer(raiz/'data/processed/feature_registry.json')
    pos=panel.index.get_indexer(fechas)
    if (pos<168).any() or (pos+23>=len(panel)).any():raise ValueError('Orígenes fuera del panel')
    if grupo=='base':
        from demanda import modelos_base as M
        # Restaurar la estructura sin volver a estimar ninguna transformación.
        datos=M.Datos.__new__(M.Datos)
        datos.panel=panel;datos.registro=reg;datos.cfg=M.ConfigDatos(**sc['datos'])
        datos.cols_x=sc['columnas_x'];datos.cols_cal=sc['columnas_cal']
        datos.escala=SimpleNamespace(media_=pd.Series(sc['media']),std_=pd.Series(sc['std']))
        datos.escala_mase=sc['escala_mase'];datos._cache={}
        conj=datos.conjunto_pos(pos)
        if sc['secuencial']:conj.x[:,:,0]=(conj.x[:,:,0]-1)/sc['sigma_demanda']
    elif grupo=='nhits':
        from demanda.patchtst_numerico import Preparador
        datos=Preparador(panel,reg,contrato);datos.stats=sc;conj=datos.transformar(pos)
    else:
        ns='modelo6' if ck['modelo'] in ('patchtst','tats') else 'entrenamiento'
        dm=importlib.import_module(ns+'.datos');datos=dm.DatosNumericos(raiz)
        esc={k:np.array([sc[k][c] for c in datos.hist]) for k in ('mediana','media','std')}
        esc.update(media_cal=np.array([sc['media_calendario'][c] for c in datos.cal]),
                   std_cal=np.array([sc['std_calendario'][c] for c in datos.cal]),sigma_demanda=sc['sigma_demanda'])
        conj=datos.construir(pos,esc)
    norm=predecir(red,tensores(conj,grupo,datos),dispositivo,ck['lote_evaluacion'])
    return datos.invertir(norm,conj) if grupo=='base' else norm*conj.nivel[:,None]


def ejecutar(modelo,salida,raiz=ROOT,semilla=1000,fase='validacion',anio=2021,
             dispositivo='auto',permitir_prueba=False,smoke=False,max_epocas=None):
    comprobar_solicitud(modelo,fase,anio,permitir_prueba,raiz)
    salida=Path(salida).resolve()
    if salida.exists():raise FileExistsError('Use una carpeta nueva para preservar los artefactos')
    if smoke and fase!='validacion':raise ValueError('La comprobación funcional solo usa desarrollo')
    import numpy as np
    import torch
    dev=('cuda' if torch.cuda.is_available() else 'cpu') if dispositivo=='auto' else dispositivo
    if dev.startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('GPU no disponible')
    if dev!='cpu' and not dev.startswith('cuda'):raise ValueError('Dispositivo inválido')
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    random.seed(semilla);np.random.seed(semilla);torch.manual_seed(semilla)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(semilla)
    ficha=registro(raiz)[modelo];fuentes(raiz,ficha)
    datos,conjuntos,escalas,forma,indice=preparar(raiz,modelo,ficha,fase,anio,smoke)
    tx=des=None
    if ficha['grupo']=='ampliado' and (ficha['variante']=='N+O' or modelo.startswith('tgforecaster')):
        tx,des=texto_congelado(raiz,indice)
        if ficha['variante']=='N':tx=None
    red=crear_red(modelo,ficha,forma,tx,des).to(dev)
    npar=sum(p.numel() for p in red.parameters() if p.requires_grad)
    if npar!=ficha['parametros']:raise AssertionError(f'Arquitectura diferente: {npar} != {ficha["parametros"]}')
    ts=[tensores(c,ficha['grupo'],datos) for c in conjuntos]
    if ficha['grupo']=='base':red.preparar(ts[0]['x'].to(dev),ts[0]['cf'].to(dev))
    inicial=hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in red.parameters())).hexdigest()
    epocas=2 if smoke else (ficha['max_epocas'] if max_epocas is None else max_epocas)
    if epocas<1:raise ValueError('El número de épocas debe ser positivo')
    inicio=time.monotonic();hist,diag=optimizar(red,ts[0],ts[1],ficha,modelo,semilla,dev,epocas)
    actual=hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in red.parameters())).hexdigest()
    if inicial==actual:raise RuntimeError('Los parámetros no cambiaron')
    prednorm=predecir(red,ts[2],dev,diag['lote_evaluacion'])
    pred=datos.invertir(prednorm,conjuntos[2]) if ficha['grupo']=='base' else prednorm*conjuntos[2].nivel[:,None]
    error=pred-conjuntos[2].y
    met=dict(MAE=float(np.abs(error).mean()),RMSE=float(np.sqrt((error**2).mean())),MASE=float(np.abs(error).mean()/Q168))
    salida.mkdir(parents=True,exist_ok=False)
    ck=dict(modelo=modelo,ficha=ficha,forma=forma,escaladores=jsonable(escalas),semilla=semilla,
            state_dict={k:v.detach().cpu() for k,v in red.state_dict().items()},lote_evaluacion=diag['lote_evaluacion'],
            entradas_texto=dict(texto=tx,descripciones=des))
    torch.save(ck,salida/'modelo.pt')
    restaurada,_=cargar_modelo(raiz,salida/'modelo.pt',dev)
    rec=predecir(restaurada,ts[2],dev,diag['lote_evaluacion'])
    np.testing.assert_allclose(rec,prednorm,rtol=0,atol=0)
    np.savez_compressed(salida/'predicciones.npz',y=conjuntos[2].y,yhat=pred,origen=conjuntos[2].origen.as_unit('ns').asi8)
    informe=dict(modelo=modelo,parametros=npar,dispositivo=dev,smoke=smoke,fase=fase,semilla=semilla,
                 muestras=[len(c.y) for c in conjuntos],periodos=[dict(inicio=str(c.origen[0]),fin=str(c.origen[-1])) for c in conjuntos],
                 epocas=len(hist),historia=hist,metricas=met,parametros_actualizados=True,restauracion_exacta=True,
                 segundos=time.monotonic()-inicio,checkpoint_sha256=huella(salida/'modelo.pt'),**diag)
    (salida/'resultado.json').write_text(json.dumps(jsonable(informe),ensure_ascii=False,indent=2),encoding='utf-8')
    (salida/'escaladores.json').write_text(json.dumps(jsonable(escalas),ensure_ascii=False,indent=2),encoding='utf-8')
    return informe


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--modelo',required=True,choices=list(registro()))
    p.add_argument('--salida',required=True,type=Path)
    p.add_argument('--semilla',default=1000,type=int)
    p.add_argument('--fase',choices=['validacion','prueba'],default='validacion')
    p.add_argument('--anio',default=2021,type=int)
    p.add_argument('--dispositivo',default='auto',choices=['auto','cpu','cuda'])
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--allow-test',dest='permitir_prueba',action='store_true')
    p.add_argument('--max-epocas',type=int)
    a=p.parse_args();res=ejecutar(**vars(a))
    print(json.dumps({k:res[k] for k in ('modelo','parametros','epocas','pasos','restauracion_exacta','segundos')},ensure_ascii=False))


if __name__=='__main__':main()

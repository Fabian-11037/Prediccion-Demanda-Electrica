"""Restauración de redes y entradas congeladas; únicamente inferencia CPU."""
from pathlib import Path
from types import ModuleType
import hashlib, importlib.util, json, sys
import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[1]
FU=ROOT/'results/arquitecturas_e2/fuentes'
CON=ROOT.parent/'Consolidado-Entrenamientos-20261115'
Q=215617.90361589016
PAQUETES={'adj_modelo3':'Modelo3_TFT','adj_modelo4':'Modelo4_TCN',
          'adj_modelo5':'Modelo5_iTransformer','adj_modelo6':'Modelo6_TaTS',
          'adj_modelo7':'Modelo7_TGForecaster','adj_fusion_tardia':'FusionTardia'}

def sha(path):
    with open(path,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def modulo(nombre,path):
    if nombre in sys.modules:return sys.modules[nombre]
    spec=importlib.util.spec_from_file_location(nombre,path)
    m=importlib.util.module_from_spec(spec);sys.modules[nombre]=m;spec.loader.exec_module(m)
    return m

def registro():
    d=json.loads((CON/'consolidado/indice_modelos.json').read_text(encoding='utf-8'))['modelos']
    return sorted([m for m in d if m['principal']],key=lambda m:(m['experimento'],m['variante'],m['semilla']))

class Entradas:
    def __init__(self):
        self.contrato=json.loads((ROOT/'config/diseno_entrenamiento_texto.json').read_text(encoding='utf-8'))
        panelpath=ROOT/'data/processed/panel_horario.parquet'
        assert sha(panelpath)==self.contrato['panel_sha256']
        p=pd.read_parquet(panelpath);self.indice=p.index
        self.hist=self.contrato['numericas']['historicas'];self.cal=self.contrato['numericas']['calendario']
        reg=json.loads((ROOT/'data/processed/feature_registry.json').read_text(encoding='utf-8'))
        raw=p[self.hist].copy()
        for c in self.hist:
            if c.startswith('x_obs_'):raw[c]=raw[c].shift(self.contrato['numericas']['rezago_clima_h']-reg[c]['rezago_h'])
        self.raw=raw.to_numpy(dtype=np.float64);self.calraw=p[self.cal].to_numpy(dtype=np.float64)
        self.fechas=pd.date_range('2022-01-01','2025-12-31',freq='D',tz='America/Bogota')
        self.pos=p.index.get_indexer(self.fechas);assert (self.pos>=168).all()
        self.il=self.pos[:,None]+np.arange(-168,0);self.ih=self.pos[:,None]+np.arange(24)
        assert (p.index.asi8[self.il[:,-1]]<self.fechas.as_unit('ns').asi8).all()
        self.y=p.y_demanda_real_kwh.to_numpy()[self.ih]
        self.dia=((self.fechas-p.index[0].normalize())/pd.Timedelta(days=1)).astype('int64').to_numpy()
        masks=[];self.mnames=[];nan=np.isnan(self.raw)
        for name,prefix in [('precio','x_precio_'),('hidrologia','x_hid_'),('clima','x_obs_')]:
            js=[j for j,c in enumerate(self.hist) if c.startswith(prefix)]
            if all(np.array_equal(nan[:,j],nan[:,js[0]]) for j in js):
                masks.append(nan[:,js[0]]);self.mnames.append('ausente_'+name)
            else:
                for j in js:masks.append(nan[:,j]);self.mnames.append('ausente_'+self.hist[j])
        self.masks=np.stack(masks,1).astype(np.float32)
        # Los vectores congelados se leen, nunca se recodifican.
        tx=CON/'fuentes/adjuntos/modelo7_20261114_0430/resultados_modelo7/texto'
        assert sha(tx/'vectores_documento.npy')=='efdfbe0ef0d4a27febba6ceedd1e0b1a5b35e5baadadc891539582c823bcdfcd'
        self.vectores=torch.from_numpy(np.load(tx/'vectores_documento.npy',allow_pickle=False).astype(np.float32))
        self.descripciones=torch.from_numpy(np.load(tx/'descripciones/vectores_documento.npy',allow_pickle=False).astype(np.float32))
        assert sha(tx/'descripciones/vectores_documento.npy')=='611f602f7dcd0eb0c8e39784d95b72fd7a9c10807610338fbadc27014dcae346'
        corpus=pd.read_parquet(ROOT/'results/evaluacion_e2/entradas/texto/corpus_oficial.parquet')
        self.corpus=corpus
        tm=modulo('texto_diagnostico',FU/'Modelo7_TGForecaster/src/entrenamiento/texto.py')
        lista,total=tm.listas_por_dia(pd.to_datetime(corpus.disponible_desde,utc=True),p.index[0].normalize(),int(self.dia.max())+1,72,64)
        self.lista=torch.from_numpy(lista);self.total=total
        old=pd.read_csv(tx/'cobertura_por_dia.csv')
        assert np.array_equal((lista>=0).sum(1),old.documentos_usados.to_numpy()[:len(lista)])
        self.cache={}

    def construir(self,sc,repo):
        key=hashlib.sha256(json.dumps(sc,sort_keys=True).encode()).hexdigest()
        if key in self.cache:return self.cache[key]
        assert pd.Timestamp(sc['fin_ajuste'])<self.fechas[0]
        cols=sc['columnas'] if repo else sc['columnas_historicas']
        assert cols==self.hist and sc['columnas_calendario']==self.cal
        def arr(k):return np.asarray(sc[k]) if repo else np.array([sc[k][c] for c in self.hist])
        z=self.raw[self.il];missing=~np.isfinite(z);filled=np.where(missing,arr('mediana'),z)
        nivel=filled[:,:,0].mean(1);scaled=(filled-arr('media'))/arr('std')
        scaled[:,:,0]=(filled[:,:,0]/nivel[:,None]-1)/sc['sigma_demanda']
        cm=np.asarray(sc['cal_media']) if repo else np.array([sc['media_calendario'][c] for c in self.cal])
        cs=np.asarray(sc['cal_std']) if repo else np.array([sc['std_calendario'][c] for c in self.cal])
        cp=((self.calraw[self.il]-cm)/cs).astype(np.float32)
        cf=((self.calraw[self.ih]-cm)/cs).astype(np.float32)
        if repo:x=np.concatenate([scaled,cp,missing.astype(float)],2).astype(np.float32)
        else:
            assert sc['mascaras']==self.mnames
            x=np.concatenate([scaled,self.masks[self.il]],2).astype(np.float32)
        assert np.isfinite(x).all() and np.isfinite(cp).all() and np.isfinite(cf).all()
        data=dict(x=x,cp=cp,cf=cf,nivel=nivel,dia=self.dia)
        self.cache[key]=data
        return data

    def placebo_historico(self,repeticion):
        """Contenido de la misma fuente disponible antes de 2022; conserva listas y volumen."""
        rng=np.random.default_rng(11037+repeticion)
        dates=pd.to_datetime(self.corpus.disponible_desde,utc=True)
        limit=pd.Timestamp('2022-01-01',tz='America/Bogota').tz_convert('UTC')
        donor=np.arange(len(dates))
        # No es una permutación biyectiva: se muestrea contenido previo con reemplazo.
        for fuente in sorted(self.corpus.fuente.unique()):
            ids=np.flatnonzero(self.corpus.fuente.eq(fuente).to_numpy())
            pool=ids[(dates.iloc[ids]<limit).to_numpy()]
            assert len(pool)>1
            donor[ids]=rng.choice(pool,len(ids),replace=True)
        assert (dates.iloc[donor]<limit).all()
        return self.vectores[torch.from_numpy(donor)],donor

def cargar(m,entradas):
    repo=m['experimento'].startswith('repo_');a=m['artefactos'];wp=CON/a['pesos']
    assert sha(wp)==m['sha256_checkpoint']
    ck=torch.load(wp,map_location='cpu',weights_only=True)
    sc=json.loads((CON/a['escaladores']).read_text(encoding='utf-8'))
    res=json.loads((CON/a['resultado']).read_text(encoding='utf-8'));p=res['configuracion']
    data=entradas.construir(sc,repo)
    if repo:
        name='numerico_diagnostico'
        if name not in sys.modules:
            pkg=ModuleType(name);pkg.__path__=[str(FU/'numerico_repositorio/src/demanda')];sys.modules[name]=pkg
        pm=modulo(name+'.patchtst_numerico',FU/'numerico_repositorio/src/demanda/patchtst_numerico.py')
        if 'nhits' in m['experimento']:
            nm=modulo(name+'.nhits_numerico',FU/'numerico_repositorio/src/demanda/nhits_numerico.py')
            red=nm.NHITSNumerico(p,sc['sigma_demanda'])
        else:red=pm.PatchTSTNumerico(p,sc['sigma_demanda'])
        state=ck['state_dict']
    else:
        prefix=next(k for k in PAQUETES if m['experimento'].startswith(k));pack=PAQUETES[prefix]
        sub='modelo6' if prefix=='adj_modelo6' else 'entrenamiento'
        mod=modulo('red_diagnostico_'+pack,FU/pack/'src'/sub/'redes.py')
        texto=dict(vectores=entradas.vectores,lista=entradas.lista) if m['variante']=='N+O' else None
        if prefix=='adj_modelo6':red=mod.Modelo6(p,data['x'].shape[2],23,168,24,texto)
        else:
            tipo={'adj_modelo3':'tft','adj_modelo4':'tcn','adj_modelo5':'itransformer','adj_modelo7':'tgforecaster','adj_fusion_tardia':'fusion_tardia'}[prefix]
            red=mod.construir(tipo,p,data['x'].shape[2],23,168,24,texto,entradas.descripciones if tipo=='tgforecaster' else None)
        state=ck
    red.load_state_dict(state,strict=True);red.eval()
    for par in red.parameters():par.requires_grad_(False)
    return red,data,repo

def forward(red,data,ix,repo):
    x=torch.from_numpy(np.ascontiguousarray(data['x'][ix]));cf=torch.from_numpy(np.ascontiguousarray(data['cf'][ix]))
    if repo:return red(x,cf)
    cp=torch.from_numpy(np.ascontiguousarray(data['cp'][ix]));dia=torch.from_numpy(np.ascontiguousarray(data['dia'][ix]))
    return red(x,cp,cf,dia)

def predecir(red,data,ix,repo,batch=8):
    out=[]
    with torch.inference_mode():
        for a in range(0,len(ix),batch):
            q=ix[a:a+batch];out.append(forward(red,data,q,repo).numpy().astype(np.float64)*data['nivel'][q,None])
    y=np.concatenate(out);assert y.shape==(len(ix),24) and np.isfinite(y).all()
    return y

def guardadas(m):
    # Solo datos numéricos; los orígenes NPY de tipo objeto no se deserializan.
    with np.load(CON/m['artefactos']['predicciones'],allow_pickle=False) as z:
        return z['yhat'] if 'yhat' in z.files else z['yhat_kwh']

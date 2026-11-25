"""Inferencia CPU desde pesos conservados, sin optimización ni lectura del objetivo futuro."""
from pathlib import Path
from types import SimpleNamespace
from collections import OrderedDict
from zipfile import ZipFile
import ast,copy,hashlib,io,json,sys,time,threading
import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[1]
PKG=ROOT/'results/corrector_final_y_m1_gpu_20261122_150345/fuentes/paquete_corrector'
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src'),str(PKG/'src')]
sys.path.append(str(ROOT/'.bookenv/Lib/site-packages'))
import inferencia_ampliada as IA
from demanda.calendario import calendario_horario
from entrenamiento import redes_e3,redes_e2
Q=215617.90361589016
STUDY=json.loads((ROOT/'dashboard/site/data/study.json').read_text(encoding='utf-8'))
MODELS={r['id']:r['name'] for r in STUDY['models']}|{'m4_n':'M4ₙ'}
LOCK=threading.RLock()

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def digest(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def frozen(red):
    red.eval();red.requires_grad_(False)
    return red

def saved_class(file,name):
    tree=ast.parse(Path(file).read_text(encoding='utf-8-sig'))
    node=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name==name)
    scope={'torch':torch,'nn':torch.nn,'np':np}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(file),'exec'),scope)
    return scope[name]

class Inputs:
    def __init__(self):
        self.contract=read(ROOT/'config/diseno_entrenamiento_texto.json')
        p=ROOT/'data/processed/panel_horario.parquet'
        if digest(p)!=self.contract['panel_sha256']:raise ValueError('El panel no coincide con el contrato.')
        panel=pd.read_parquet(p)
        self.hist=self.contract['numericas']['historicas'];self.cal=self.contract['numericas']['calendario']
        self.index=panel.index;self.original_end=panel.index[-1]
        self.raw=panel[self.hist].copy()
        reg=read(ROOT/'data/processed/feature_registry.json')
        for c in self.hist:
            if c.startswith('x_obs_'):self.raw[c]=self.raw[c].shift(self.contract['numericas']['rezago_clima_h']-reg[c]['rezago_h'])
        self.calendar=panel[self.cal].copy()
        self.origin=None;self.override=None;self.demanda_pct=0.;self._text_ready=False
        self.mnames=['ausente_precio','ausente_hidrologia','ausente_clima']
        self.bounds=self.raw.loc[:'2021-12-31'].agg(['min','max'])
        self.fechas=pd.DatetimeIndex([])

    def configure(self,origin,demand_pct=0.,csv=None):
        if not isinstance(origin,str) or len(origin)!=10:raise ValueError('Fecha con formato AAAA-MM-DD.')
        day=pd.Timestamp(origin,tz='America/Bogota')
        if day.strftime('%Y-%m-%d')!=origin or (day<pd.Timestamp('2022-01-01',tz='America/Bogota') or day>pd.Timestamp('2035-12-31',tz='America/Bogota')):
            raise ValueError('El origen debe ser posterior al cierre de ajuste de 2021.')
        if not np.isfinite(demand_pct) or abs(demand_pct)>20:raise ValueError('El cambio de demanda debe estar entre −20 % y 20 %.')
        times=pd.date_range(day-pd.Timedelta(hours=168),periods=168,freq='h')
        if csv:
            if len(csv.encode())>500000:raise ValueError('El archivo de entradas supera 500 kB.')
            frame=pd.read_csv(io.StringIO(csv))
            if set(frame.columns)!={'fecha',*self.hist}:raise ValueError('El CSV debe contener fecha y las 30 variables del contrato.')
            ix=pd.to_datetime(frame.pop('fecha'),utc=True).dt.tz_convert('America/Bogota')
            if len(frame)!=168 or not pd.DatetimeIndex(ix).equals(times):raise ValueError('Se requieren las 168 horas consecutivas anteriores al origen.')
            window=frame[self.hist].apply(pd.to_numeric,errors='raise').to_numpy(dtype=float,copy=True)
        else:
            if times[0]<self.index[0] or times[-1]>self.original_end:raise ValueError('Carga un CSV de historia para un origen sin datos disponibles.')
            window=self.raw.reindex(times).to_numpy(dtype=float,copy=True)
        if not np.isfinite(window[:,0]).all() or (window[:,0]<=0).any():raise ValueError('La demanda histórica debe ser finita y positiva.')
        if np.isinf(window).any():raise ValueError('Las covariables no admiten valores infinitos.')
        window[:,0]*=1+demand_pct/100
        self.origin=day;self.window=window;self.demanda_pct=demand_pct;self.fechas=pd.DatetimeIndex([day])
        self.cp=calendario_horario(times).add_prefix('cal_')[self.cal].to_numpy(dtype=float,copy=True)
        self.cf=calendario_horario(pd.date_range(day,periods=24,freq='h')).add_prefix('cal_')[self.cal].to_numpy(dtype=float,copy=True)
        self.day=int((day-self.index[0].normalize())/pd.Timedelta(days=1))
        outside=(window<self.bounds.loc['min'].to_numpy())|(window>self.bounds.loc['max'].to_numpy())
        self.outside=[self.hist[j] for j in np.flatnonzero(outside.any(0))]
        self.cache={}
        if self._text_ready:self.prepare_lists()

    def construir(self,sc,repo=False):
        if 'columnas_historicas' in sc:
            def arr(k):return np.array([sc[k][c] for c in self.hist])
            cm=np.array([sc['media_calendario'][c] for c in self.cal]);cs=np.array([sc['std_calendario'][c] for c in self.cal])
        else:
            def arr(k):return np.asarray(sc[k])
            cm=np.asarray(sc.get('cal_media',sc.get('media_cal')));cs=np.asarray(sc.get('cal_std',sc.get('std_cal')))
        z=self.window[None];missing=~np.isfinite(z);filled=np.where(missing,arr('mediana'),z)
        level=filled[:,:,0].mean(1);scaled=(filled-arr('media'))/arr('std')
        scaled[:,:,0]=(filled[:,:,0]/level[:,None]-1)/sc['sigma_demanda']
        cp=((self.cp[None]-cm)/cs).astype('float32');cf=((self.cf[None]-cm)/cs).astype('float32')
        if repo:x=np.concatenate([scaled,cp,missing.astype(float)],2)
        else:
            masks=[]
            for prefix in ['x_precio_','x_hid_','x_obs_']:
                js=[j for j,c in enumerate(self.hist) if c.startswith(prefix)]
                # El contrato conservado comparte una máscara por bloque.
                if any(not np.array_equal(missing[:,:,j],missing[:,:,js[0]]) for j in js):
                    raise ValueError('Las máscaras del CSV deben respetar el patrón por bloque del checkpoint.')
                masks.append(missing[:,:,js[0]])
            x=np.concatenate([scaled,np.stack(masks,-1)],2)
        if not np.isfinite(x).all():raise ValueError('No fue posible preparar entradas finitas.')
        return dict(x=x.astype('float32'),cp=cp,cf=cf,nivel=level,dia=np.array([self.day],dtype='int64'))

    def prepare_text(self):
        if self._text_ready:return
        tx=IA.CON/'fuentes/adjuntos/modelo7_20261114_0430/resultados_modelo7/texto'
        self.vectores=torch.from_numpy(np.load(tx/'vectores_documento.npy',allow_pickle=False).astype('float32'))
        self.descripciones=torch.from_numpy(np.load(tx/'descripciones/vectores_documento.npy',allow_pickle=False).astype('float32'))
        corpus=pd.read_parquet(ROOT/'results/evaluacion_e2/entradas/texto/corpus_oficial.parquet')
        self.publication=pd.to_datetime(corpus.disponible_desde,utc=True)
        self.tm=IA.modulo('texto_diagnostico',IA.FU/'Modelo7_TGForecaster/src/entrenamiento/texto.py')
        self._text_ready=True;self.prepare_lists()

    def prepare_lists(self):
        lista,_=self.tm.listas_por_dia(self.publication,self.index[0].normalize(),self.day+1,72,64)
        self.lista=torch.from_numpy(lista)

class Runtime:
    def __init__(self):
        torch.set_num_threads(1)
        self.inputs=Inputs();self.loaded=OrderedDict();self.bank=None
        self.Memoria=saved_class(ROOT/'scripts/documentacion_m4/pipeline_m4_n.py','Memoria')
        self.Router=saved_class(ROOT/'book/_static/codigo_e3/m4/e3_modelo1.py','Router')
        self.registry=IA.registro()
        self.snapshot=ROOT/'results/repositorio'/read(ROOT/'results/jbook/modelos_actualizados.json')['commit']
        self.baselines=read(self.snapshot/'results/consolidado/modelos.json')

    def put(self,key,value):
        self.loaded[key]=value
        while len(self.loaded)>3:self.loaded.popitem(last=False)
        return value

    def baseline(self,key):
        if key in self.loaded:return self.loaded[key]
        import demanda
        old=str(self.snapshot/'src/demanda')
        if old not in demanda.__path__:demanda.__path__.append(old)
        M=IA.modulo('demanda.live_modelos_base',self.snapshot/'src/demanda/modelos_base.py')
        C=IA.modulo('demanda.live_reentrenamiento',self.snapshot/'src/demanda/reentrenamiento_v3.py')
        L=IA.modulo('demanda.live_variantes',self.snapshot/'src/demanda/variantes_v4.py')
        record=self.baselines[key];a=next(r for r in record['artefactos'] if r['semilla']==1000)
        esc=read(self.snapshot/a['escaladores']);data=self.baseline_inputs(esc)
        shape=SimpleNamespace(x=data['x'][:,:,:1],cal_p=data['x'][:,:,1:],cal_f=data['cf'])
        p=record['configuracion']['params']
        red=L.Secuencial(key,p,shape) if key=='lstm' else C.Secuencial(key,p,shape) if key=='cnn1d' else M.construir_red(key,p,shape)
        file=self.snapshot/a['pesos'];red.load_state_dict(torch.load(file,map_location='cpu',weights_only=True)['state_dict'])
        return self.put(key,(frozen(red),esc,digest(file)))

    def baseline_inputs(self,sc):
        I=self.inputs;level=I.window[:,0].mean();x=(I.window[:,0]/level).astype('float32')
        if sc['secuencial']:x=(x-1)/sc['sigma_demanda']
        cm=np.array([sc['media'][c] for c in I.cal]);cs=np.array([sc['std'][c] for c in I.cal])
        cp=((I.cp-cm)/cs).astype('float32');cf=((I.cf-cm)/cs).astype('float32')
        return dict(x=np.concatenate([x[:,None],cp],1)[None],cf=cf[None],nivel=np.array([level]))

    def additional(self,key):
        if key in self.loaded:return self.loaded[key]
        family='timesnet' if 'timesnet' in key else 'nbeats'
        base=ROOT/'results/analisis_nbeats_timesnet_20261120/entrada_resultados/resultados_e2_nbeats_timesnet'
        rel=f'resultados_e2_nbeats_timesnet/nuevo_{family}/final/N/semilla_1000/modelo.pt'
        folder=base/f'nuevo_{family}/final/N/semilla_1000'
        zipfile=ROOT/'results/integracion_top10_20261120/pesos/ENVIAR_pesos_NBEATS_TimesNet_20261119_2125.zip'
        with ZipFile(zipfile) as z:raw=z.read(rel)
        red=redes_e2.construir(family,read(folder/'resultado.json')['configuracion'],33,23,168,24,None,None)
        red.load_state_dict(torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True))
        return self.put(key,(frozen(red),read(folder/'escaladores.json'),hashlib.sha256(raw).hexdigest()))

    def expanded(self,key):
        I=self.inputs;I.prepare_text()
        # El agregador depende del origen solicitado; no reutilizar su lista diaria.
        record=next(m for m in self.registry if m['experimento']+'__'+m['variante']==key and m['semilla']==1000)
        red,data,repo=IA.cargar(record,I)
        return red,data,repo,record['sha256_checkpoint']

    def prepare_memory(self):
        if self.bank is not None:return
        old=ROOT/'results/m3_gpu_20261122_193240'
        parts=[]
        for year in range(2013,2022):
            with np.load(old/f'base_oof/anio_{year}_semilla_101.npz',allow_pickle=False) as z:
                parts.append({k:z[k] for k in ['dia','y','pred_base','num','expertos']})
        with np.load(old/'base_final/semilla_101.npz',allow_pickle=False) as z:
            parts.append({k:z[k] for k in ['dia','y','pred_base','num','expertos']})
        self.bank={k:np.concatenate([p[k] for p in parts]) for k in parts[0]}
        self.bank_dates=np.asarray(self.bank['dia'],dtype='datetime64[D]')
        self.byday={str(d):i for i,d in enumerate(self.bank_dates)}
        root=ROOT/'results/corrector_final_y_m1_gpu_20261122_150345/M1'
        self.experts=[];self.hashes={}
        for name,kind in [('patchtst','tats'),('E','original_E'),('periodica','x_periodica')]:
            p=root/f'expertos_finales/{name}/semilla_101/modelo.pt';ck=torch.load(p,map_location='cpu',weights_only=False)
            red=redes_e3.construir(kind,ck['metadatos']['configuracion'],33,23,168,24,n_mascaras=3)
            red.load_state_dict(ck['modelo']);self.experts.append((frozen(red),ck['escalas']))
            self.hashes[name]=digest(p)
        p=root/'final/semilla_101/modelo.pt';self.router_ck=torch.load(p,map_location='cpu',weights_only=False)
        self.router=frozen(self.Router(self.router_ck['n_entrada'],self.router_ck['configuracion']))
        self.router.load_state_dict(self.router_ck['modelo']);self.hashes['M1']=digest(p)
        p=ROOT/'results/m4_cierre_gpu_20261123/reducido_B/colombia/final/M4_N/semilla_101/modelo.pt'
        expected=read(ROOT/'results/m4_cierre_gpu_20261123/reducido_B/CONGELACION_PREPRUEBA.json')['pesos'][p.relative_to(ROOT).as_posix()]
        if digest(p)!=expected:raise ValueError('El checkpoint de M4 no coincide con su registro congelado.')
        self.memory_ck=torch.load(p,map_location='cpu',weights_only=False);c=self.memory_ck
        self.memory=frozen(self.Memoria(c['n_num'],c['n_feat'],c['n_fuentes'],c['configuracion'],'M4_N'))
        self.memory.load_state_dict(c['modelo']);self.hashes['M4_N']=digest(p)

    def memory_prediction(self):
        self.prepare_memory();I=self.inputs;day=I.origin.strftime('%Y-%m-%d')
        predictions=[]
        for red,sc in self.experts:
            d=I.construir(sc)
            args=[torch.from_numpy(np.ascontiguousarray(d[k])) for k in ['x','cp','cf','dia']]
            predictions.append(red(*args).numpy().astype('float32')*d['nivel'][:,None])
        pred=np.stack(predictions,-1);level=I.window[:,0].mean()
        valid=np.isfinite(I.window);clean=np.where(valid,I.window,0.)
        summary=np.concatenate([clean[-1],clean.sum(0)/np.maximum(valid.sum(0),1),
            clean[-24:].sum(0)/np.maximum(valid[-24:].sum(0),1),1-valid.mean(0)])
        summary[[0,30,60]]/=level
        rows=[];zero=np.zeros((24,3),dtype='float32')
        now=np.datetime64(day,'D')
        for k in range(2,9):
            j=self.byday.get(str(now-np.timedelta64(k,'D')))
            rows.append((self.bank['y'][j,:,None]-self.bank['expertos'][j])/Q if j is not None else None)
        lag=np.concatenate([(rows[0] if rows[0] is not None else zero).ravel(),np.mean([r for r in rows if r is not None] or [zero],0).ravel()])
        x=np.concatenate([summary,I.cf.ravel(),(pred[0]/level).ravel(),pred[0].std(-1)/level,lag]).astype('float32')[None]
        c=self.router_ck
        base=self.router(torch.from_numpy((x-c['media'])/c['escala']),torch.from_numpy(pred))[0].numpy()[0]
        # Solo se materializan valores del banco que ya son observables.
        mask=self.bank_dates<=now-np.timedelta64(2,'D');idx=np.flatnonzero(mask)
        # Un año basta para recuperar; la consulta no contiene su etiqueta.
        idx=idx[self.bank_dates[idx]>=now-np.timedelta64(365,'D')]
        num=np.concatenate([self.bank['num'][idx],x]);res=(self.bank['y'][idx]-self.bank['pred_base'][idx])/Q
        days=np.r_[self.bank_dates[idx].astype('int64'),now.astype('int64')]
        c=self.memory_ck
        bank=SimpleNamespace(num=torch.from_numpy((num-c['media'])/c['escala']),
            dias=torch.from_numpy(days),residuos=torch.from_numpy(np.concatenate([res,np.zeros((1,24),dtype=res.dtype)])),
            base=torch.from_numpy(np.concatenate([self.bank['pred_base'][idx],base[None]])),limite=None)
        output,g,neighbors,weights,ok=self.memory(bank,torch.tensor([len(idx)]))
        ids=neighbors[0][ok[0]].numpy();alfa=weights[0][ok[0]].numpy()
        detail=dict(gate=float(g[0]),base_gwh=(base/1e6).tolist(),
            neighbors=[dict(fecha=str(self.bank_dates[idx[j]]),peso=float(a)) for j,a in zip(ids,alfa)])
        return output.numpy()[0].astype(float),self.hashes,detail

    def forward(self,key):
        I=self.inputs
        if key=='m4_n':return self.memory_prediction()
        if key in ['mlp','lstm','cnn1d','arx_lineal']:
            red,sc,h=self.baseline(key);d=self.baseline_inputs(sc)
            p=red(torch.from_numpy(d['x']),torch.from_numpy(d['cf'])).numpy()[0].astype(float)*d['nivel'][0]
        elif key in ['nuevo_nbeats__N','nuevo_timesnet__N']:
            red,sc,h=self.additional(key);d=I.construir(sc)
            p=red(*(torch.from_numpy(np.ascontiguousarray(d[k])) for k in ['x','cp','cf','dia'])).numpy()[0].astype(float)*d['nivel'][0]
        else:
            red,d,repo,h=self.expanded(key)
            p=IA.forward(red,d,np.array([0]),repo).numpy()[0].astype(float)*d['nivel'][0]
        return p,{'modelo':h},{}

    def predict(self,key,origin,demand_pct=0.,csv=None,explain=False,interval_level=.9,interval_scope="horario"):
        if key not in MODELS:raise ValueError('Modelo fuera del catálogo.')
        with LOCK,torch.inference_mode():
            t=time.perf_counter();self.inputs.configure(origin,float(demand_pct),csv)
            p,hashes,detail=self.forward(key)
            if p.shape!=(24,) or not np.isfinite(p).all():raise ValueError('El modelo no produjo 24 valores finitos.')
            delta=None
            if explain:
                old=self.inputs.window.copy();self.inputs.window[-24:,0]*=1.01
                try:other,_,_=self.forward(key);delta=float((other-p).mean()/1e6)
                finally:self.inputs.window=old
            from intervalos_vivo import interval
            band=interval(key,origin,p,interval_level,interval_scope)
            return dict(interval=band,model=key,name=MODELS[key],origin=origin,seed=101 if key=='m4_n' else 1000,
                dates=pd.date_range(self.inputs.origin,periods=24,freq='h').astype(str).tolist(),
                prediction_gwh=(p/1e6).tolist(),history_dates=pd.date_range(self.inputs.origin-pd.Timedelta(hours=168),periods=168,freq='h').astype(str).tolist(),
                history_gwh=(self.inputs.window[:,0]/1e6).tolist(),latency_ms=1000*(time.perf_counter()-t),
                outside_domain=([c for c in self.inputs.outside if c==self.inputs.hist[0]] if key in ['mlp','lstm','cnn1d','arx_lineal'] else self.inputs.outside),checkpoint_sha256=hashes,detail=detail,local_delta_gwh=delta,
                last_input=str(self.inputs.origin-pd.Timedelta(hours=1)),training=False,computed=True)

_runtime=None
def get_runtime():
    global _runtime
    with LOCK:
        if _runtime is None:_runtime=Runtime()
        return _runtime

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--model',default='m4_n');parser.add_argument('--origin',default='2023-07-01')
    args=parser.parse_args();r=get_runtime().predict(args.model,args.origin)
    print(json.dumps(r,ensure_ascii=False))

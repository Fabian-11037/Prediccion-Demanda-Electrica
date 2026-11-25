"""Perturbaciones de entradas con pesos congelados. No contiene optimizadores."""
from __future__ import annotations
import argparse, copy, hashlib, io, json, platform, sys, time
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'dashboard'),str(ROOT/'scripts')]
import numpy as np
import pandas as pd
import torch
from runtime_inferencia import Runtime, Inputs, IA, read, frozen, digest, redes_e2, redes_e3

OUT=ROOT/'results/robustez_congelada_20261124'
Q=215617.90361589016
DATES=[f'{y}-{m:02d}-{d:02d}' for y in range(2022,2026) for m in range(1,13) for d in [1,15]]
SEEDS=list(range(1000,1005))
KINDS={'ruido':[.10,.25,.50],'interrupcion':[6,24,48],'atipicos':[1.,3.,5.],'perdida_texto':[.25,.50,.75]}
CASES=[dict(id='original',tipo='original',nivel=0,severidad=0.,repeticion=0)]
for kind,levels in KINDS.items():
    for level,value in enumerate(levels,1):
        for rep in range(1 if kind=='interrupcion' else 3):
            CASES.append(dict(id=f'{kind}_{level}_{rep}',tipo=kind,nivel=level,severidad=value,repeticion=rep))


def write(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


class RobustInputs(Inputs):
    """Listas globales causales: cada red accede solo al índice del origen."""
    def prepare_lists(self):
        if getattr(self,'full_lists',False):return
        self.day=int((pd.Timestamp('2025-12-31',tz='America/Bogota')-self.index[0].normalize())/pd.Timedelta(days=1))
        super().prepare_lists();self.full_lists=True


class Evaluator(Runtime):
    def __init__(self):
        super().__init__()
        self.inputs=RobustInputs();self.files={};self.selected_seed=None;self.red=None;self.aggregator=None
        self.inputs.configure(DATES[0]);self.inputs.prepare_text();self.inputs.configure(DATES[0])
        self.scale=self.inputs.raw.loc['2008-01-01':'2017-12-31'].std().to_numpy(dtype=float)
        assert np.isfinite(self.scale).all() and (self.scale>0).all()
        self.target=pd.read_parquet(ROOT/'data/processed/panel_horario.parquet',columns=['y_demanda_real_kwh'])

    def protect(self,p):
        p=Path(p);h=digest(p)
        if str(p)in self.files:assert self.files[str(p)]==h
        self.files[str(p)]=h
        return h

    def load_model(self,key,seed):
        self.selected_seed=seed;I=self.inputs
        if key=='m4_n':
            self.load_memory(seed)
            self.red=None;self.scaler=None;self.repo=False
        elif key in ['mlp','lstm','cnn1d','arx_lineal']:
            import demanda
            old=str(self.snapshot/'src/demanda')
            if old not in demanda.__path__:demanda.__path__.append(old)
            M=IA.modulo('demanda.robust_modelos_base',self.snapshot/'src/demanda/modelos_base.py')
            C=IA.modulo('demanda.robust_reentrenamiento',self.snapshot/'src/demanda/reentrenamiento_v3.py')
            L=IA.modulo('demanda.robust_variantes',self.snapshot/'src/demanda/variantes_v4.py')
            record=self.baselines[key];a=next(r for r in record['artefactos']if r['semilla']==seed)
            self.scaler=read(self.snapshot/a['escaladores']);self.protect(self.snapshot/a['escaladores'])
            d=self.baseline_inputs(self.scaler);shape=SimpleNamespace(x=d['x'][:,:,:1],cal_p=d['x'][:,:,1:],cal_f=d['cf'])
            p=record['configuracion']['params']
            self.red=L.Secuencial(key,p,shape)if key=='lstm'else C.Secuencial(key,p,shape)if key=='cnn1d'else M.construir_red(key,p,shape)
            file=self.snapshot/a['pesos'];self.protect(file)
            self.red.load_state_dict(torch.load(file,map_location='cpu',weights_only=True)['state_dict'])
            self.expected_file=self.snapshot/a['predicciones'];self.repo=False
        elif key in ['nuevo_nbeats__N','nuevo_timesnet__N']:
            family='timesnet'if'timesnet'in key else'nbeats'
            base=ROOT/'results/analisis_nbeats_timesnet_20261120/entrada_resultados/resultados_e2_nbeats_timesnet'
            folder=base/f'nuevo_{family}/final/N/semilla_{seed}'
            self.scaler=read(folder/'escaladores.json');self.protect(folder/'escaladores.json')
            zipfile=ROOT/'results/integracion_top10_20261120/pesos/ENVIAR_pesos_NBEATS_TimesNet_20261119_2125.zip'
            if str(zipfile)not in self.files:self.protect(zipfile)
            with ZipFile(zipfile)as z:raw=z.read(f'resultados_e2_nbeats_timesnet/nuevo_{family}/final/N/semilla_{seed}/modelo.pt')
            self.red=redes_e2.construir(family,read(folder/'resultado.json')['configuracion'],33,23,168,24,None,None)
            self.red.load_state_dict(torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True));self.repo=False
            self.expected_file=None
        else:
            m=next(m for m in self.registry if m['experimento']+'__'+m['variante']==key and m['semilla']==seed)
            self.red,_,self.repo=IA.cargar(m,I)
            self.scaler=read(IA.CON/m['artefactos']['escaladores'])
            self.protect(IA.CON/m['artefactos']['pesos']);self.protect(IA.CON/m['artefactos']['escaladores'])
            self.expected_record=m;self.expected_file=None
        if self.red is not None:
            frozen(self.red);assert not any(p.requires_grad for p in self.red.parameters())
            self.aggregator=getattr(self.red,'agregador',None)or getattr(self.red,'relevancia',None)
        else:self.aggregator=None
        self.key=key

    def load_memory(self,seed):
        assert seed in range(101,106),'Cinco pipelines completos disponibles; no sustituir expertos entre semillas.'
        old=ROOT/'results/m3_gpu_20261122_193240';parts=[]
        for year in range(2013,2022):
            p=old/f'base_oof/anio_{year}_semilla_{seed}.npz'
            with np.load(p,allow_pickle=False)as z:parts.append({k:z[k]for k in ['dia','y','pred_base','num','expertos']})
        with np.load(old/f'base_final/semilla_{seed}.npz',allow_pickle=False)as z:
            parts.append({k:z[k]for k in ['dia','y','pred_base','num','expertos']})
        self.bank={k:np.concatenate([p[k]for p in parts])for k in parts[0]}
        self.bank_dates=np.asarray(self.bank['dia'],dtype='datetime64[D]');self.byday={str(d):i for i,d in enumerate(self.bank_dates)}
        base=ROOT/'results/corrector_final_y_m1_gpu_20261122_150345/M1';self.experts=[];self.hashes={}
        for name,kind in [('patchtst','tats'),('E','original_E'),('periodica','x_periodica')]:
            p=base/f'expertos_finales/{name}/semilla_{seed}/modelo.pt';self.protect(p)
            ck=torch.load(p,map_location='cpu',weights_only=False)
            r=redes_e3.construir(kind,ck['metadatos']['configuracion'],33,23,168,24,n_mascaras=3);r.load_state_dict(ck['modelo'])
            self.experts.append((frozen(r),ck['escalas']))
        p=base/f'final/semilla_{seed}/modelo.pt';self.protect(p)
        self.router_ck=torch.load(p,map_location='cpu',weights_only=False);ck=self.router_ck
        self.router=frozen(self.Router(ck['n_entrada'],ck['configuracion']));self.router.load_state_dict(ck['modelo'])
        p=ROOT/f'results/m4_cierre_gpu_20261123/reducido_B/colombia/final/M4_N/semilla_{seed}/modelo.pt';self.protect(p)
        self.memory_ck=torch.load(p,map_location='cpu',weights_only=False);ck=self.memory_ck
        self.memory=frozen(self.Memoria(ck['n_num'],ck['n_feat'],ck['n_fuentes'],ck['configuracion'],'M4_N'));self.memory.load_state_dict(ck['modelo'])

    def predict_loaded(self):
        I=self.inputs
        if self.key=='m4_n':return self.memory_prediction()[0]
        if self.key in ['mlp','lstm','cnn1d','arx_lineal']:
            d=self.baseline_inputs(self.scaler)
            return self.red(torch.from_numpy(d['x']),torch.from_numpy(d['cf'])).numpy()[0].astype(float)*d['nivel'][0]
        d=I.construir(self.scaler,self.repo)
        return IA.forward(self.red,d,np.array([0]),self.repo).numpy()[0].astype(float)*d['nivel'][0]

    def expected(self,date):
        j=int((pd.Timestamp(date)-pd.Timestamp('2022-01-01')).days)
        if self.key=='m4_n':
            p=ROOT/f'results/m4_cierre_gpu_20261123/reducido_B/colombia/prueba/M4_N_semilla_{self.selected_seed}.npz'
            with np.load(p,allow_pickle=False)as z:return z['pred'][j]
        if self.key in ['nuevo_nbeats__N','nuevo_timesnet__N']:return None
        if self.key in ['mlp','lstm','cnn1d','arx_lineal']:
            with np.load(self.expected_file,allow_pickle=False)as z:
                a=z['pred']if'pred'in z.files else z['yhat']if'yhat'in z.files else z['yhat_kwh']
                return a[j]
        return IA.guardadas(self.expected_record)[j]


def perturb(window,date,case,scale,hist):
    w=window.copy();kind=case['tipo'];value=case['severidad'];rep=case['repeticion']
    rng=np.random.default_rng(np.random.SeedSequence([20261124,int(date.replace('-','')),rep]))
    if kind=='ruido':
        noise=rng.normal(size=w.shape);w+=value*scale*noise
    elif kind=='atipicos':
        rows=rng.permutation(168)[:9];sign=rng.choice([-1.,1.],size=(9,len(scale)))
        w[rows]+=value*scale*sign
    elif kind=='interrupcion':
        h=int(value);w[-h:,0]=w[-h-1,0];w[-h:,1:]=np.nan
    if kind in ['ruido','atipicos']:
        for j,name in enumerate(hist):
            if 'temperature' in name:continue
            w[:,j]=np.maximum(w[:,j],1. if j==0 else 0.)
            if 'humidity'in name:w[:,j]=np.minimum(w[:,j],100.)
    assert np.isfinite(w[:,0]).all()and(w[:,0]>0).all()
    return w


def evaluate(keys=None,smoke=False,seed_position=None):
    OUT.mkdir(parents=True,exist_ok=True)
    r=Evaluator();I=r.inputs;torch.set_num_threads(1)
    study=read(ROOT/'dashboard/site/data/study.json')
    names={m['id']:m['name']for m in study['models']}|{'m4_n':'M4ₙ','naive_diario':'Ingenuo diario','naive_semanal':'Ingenuo semanal'}
    selected=keys or list(names)
    dates=DATES[:1]if smoke else DATES
    protocol=dict(fecha='2026-11-24',origenes=DATES,tipos=KINDS,casos=CASES,
        semillas_benchmark=SEEDS,semillas_M4=list(range(101,106)),replicas_estocasticas=3,
        escala_ruido='DE por variable del bloque 2008–2017, calculada solo para definir severidad; no modifica escaladores de modelos',
        atipicos='9 de 168 horas con impulso ± severidad × DE; mismas filas y signos entre modelos, semillas y severidades',
        interrupcion='Últimas 6/24/48 horas: demanda con último valor anterior; covariables NaN y tratamiento congelado del checkpoint',
        perdida_texto='Eliminación Bernoulli anidada por documento con probabilidad 0,25/0,50/0,75; conserva fechas y contenido de documentos restantes',
        calendario='Sin perturbaciones: conocido al origen',memoria='Banco histórico y errores ya observables conservados; se perturba la historia de la consulta y se recalculan expertos, M1 y recuperación',
        marco='Diagnóstico de entradas con pesos fijos; no modifica el benchmark global de prueba',
        semilla_M4_alcance='Se evalúan 101–105, con los tres expertos y M1 disponibles para cada semilla; no se mezclan con las otras cinco realizaciones del benchmark de M4',
        inferencia='CPU un hilo; lote uno en todos los modelos, en particular TimesNet',q168=Q,
        bootstrap=dict(reps=9999,bloque_meses=3,unidad='Mes calendario: dos orígenes; promedio previo de réplicas',semillas='Remuestreo de cinco realizaciones dentro de cada modelo'),
        comparaciones=dict(degradacion='18 procedimientos × 12 condiciones; Holm sobre 216 hipótesis de deterioro',M4='17 comparadores × 12 condiciones; Holm sobre 204 hipótesis de mejora relativa superior al 3 %'),
        entrenamiento=False,publicacion=False,seleccion_por_resultados=False,
        entorno=dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,sistema=platform.platform()))
    pp=OUT/'protocolo.json'
    if not pp.exists():write(pp,protocol)
    else:assert read(pp)==protocol,'El protocolo registrado no se modifica durante la evaluación.'
    scale_file=OUT/'escala_perturbacion.csv'
    if not scale_file.exists():pd.DataFrame({'variable':I.hist,'DE_2008_2017':r.scale}).to_csv(scale_file,index=False)
    for key in selected:
        seeds=[0]if key.startswith('naive_')else list(range(101,106))if key=='m4_n'else SEEDS
        if seed_position is not None:
            seeds=seeds if len(seeds)==1 and seed_position==0 else [] if len(seeds)==1 else [seeds[seed_position]]
        for seed in seeds[:1]if smoke else seeds:
            path=OUT/'predicciones'/f'{key}__semilla_{seed}.npz';path.parent.mkdir(exist_ok=True)
            if path.exists()and not smoke:print('Conservado',key,seed,flush=True);continue
            start=time.perf_counter();I.configure(dates[0])
            if not key.startswith('naive_'):r.load_model(key,seed)
            predictions=[];parity=[];drops=[]
            with torch.inference_mode():
                for number,date in enumerate(dates):
                    I.configure(date);original=I.window.copy();day=I.day
                    if not key.startswith('naive_')and r.aggregator is not None:
                        saved=r.aggregator.lista[day-6:day+1].clone()
                    else:saved=None
                    out=[];base=None
                    for case in CASES:
                        I.window=perturb(original,date,case,r.scale,I.hist)
                        if saved is not None:r.aggregator.lista[day-6:day+1]=saved
                        if case['tipo']=='perdida_texto':
                            if saved is None:
                                assert base is not None;pred=base.copy();out.append(pred);continue
                            rng=np.random.default_rng(np.random.SeedSequence([20261125,int(date.replace('-','')),case['repeticion']]))
                            uniforms=rng.random(len(I.vectores));q=saved.clone();valid=q>=0
                            reject=np.zeros(q.shape,dtype=bool);reject[valid.numpy()]=uniforms[q[valid].numpy()]<case['severidad']
                            q[torch.from_numpy(reject)]=-1;r.aggregator.lista[day-6:day+1]=q
                            drops.append(dict(origen=date,caso=case['id'],antes=int(valid[-1].sum()),despues=int((q[-1]>=0).sum())))
                        if key.startswith('naive_'):
                            pred=I.window[-24:,0].copy()if key=='naive_diario'else I.window[:24,0].copy()
                        else:pred=r.predict_loaded()
                        assert pred.shape==(24,)and np.isfinite(pred).all(),(key,seed,date,case)
                        if case['tipo']=='original':
                            base=pred.copy()
                            expected=None if key.startswith('naive_')else r.expected(date)
                            if expected is not None:
                                error=float(np.max(np.abs(pred-expected)));relative=error/max(float(np.abs(expected).max()),1.)
                                parity.append(dict(origen=date,error_max_kWh=error,error_relativo=relative))
                                assert relative<1e-4,(key,seed,date,'Paridad',error,relative)
                        out.append(pred)
                    I.window=original
                    if saved is not None:r.aggregator.lista[day-6:day+1]=saved
                    predictions.append(np.stack(out))
                    if number%24==0:print('Avance',key,seed,date,f'{time.perf_counter()-start:.1f}s',flush=True)
            targets=np.stack([r.target.reindex(pd.date_range(pd.Timestamp(d,tz='America/Bogota'),periods=24,freq='h')).iloc[:,0].to_numpy()for d in dates])
            dest=OUT/'smoke'/f'{key}__semilla_{seed}.npz'if smoke else path;dest.parent.mkdir(exist_ok=True)
            np.savez_compressed(dest,pred=np.stack(predictions),y=targets,dias=np.asarray(dates,dtype='U10'),casos=np.asarray([c['id']for c in CASES],dtype='U32'))
            write(dest.with_suffix('.json'),dict(modelo=key,nombre=names[key],semilla=seed,paridad=parity,perdida_documentos=drops,segundos=time.perf_counter()-start,
                pesos_sha256=r.files,entrenamiento=False))
            print('Terminado',key,seed,f'{time.perf_counter()-start:.1f}s',flush=True)
            if r.red is not None:r.red=None
    for file,h in r.files.items():assert digest(file)==h,(file,'El artefacto cambió durante la evaluación')
    print('Pesos y escaladores intactos.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--modelos',nargs='*');parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--posicion-semilla',type=int,choices=range(5))
    args=parser.parse_args();evaluate(args.modelos,args.smoke,args.posicion_semilla)

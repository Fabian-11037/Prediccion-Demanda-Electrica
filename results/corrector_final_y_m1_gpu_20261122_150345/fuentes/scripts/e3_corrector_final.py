"""Adaptador final del corrector: entrenamiento pre-2022, congelación, predicción causal y contraste."""
from __future__ import annotations
import gc
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from e3_gpu_comun import *
from corrector import modelo as MO, texto as TX, evaluacion as EV

DESARROLLO=PAQUETE/'resultados_corrector_e3'

def cargar_texto_historico(col):
    car=DESARROLLO/('texto' if col=='disponible_declarado' else 'texto_'+col)
    r=leer(car/'resumen.json');z=np.load(car/'candidatos.npz')
    cand={k:z[k] for k in ['dias','cand','feat','fuente']};cand.update(reservas=r['reservas'],n_fuentes=len(r['fuentes']))
    emb=torch.from_numpy(np.load(DESARROLLO/'texto_vectores'/f'vectores_{col}.npy'))
    return cand,emb.cuda(),r

def entradas_corrector(datos,cand,emb,variante,mu,sd,indices):
    f=datos.fila[np.asarray(indices)]
    ids=np.maximum(cand['cand'][f],0).astype('int64')
    masc,feat=MO.mascara_variante(variante,cand['reservas'],cand['cand'][f],cand['feat'][f].astype('float32'))
    e=torch.zeros((len(f),ids.shape[1],768),device='cuda') if variante=='metadatos' else emb[torch.as_tensor(ids,device=emb.device)].float().cuda()
    return (torch.as_tensor((datos.num[indices]-mu)/sd,device='cuda'),e,torch.as_tensor(feat,device='cuda'),
            torch.as_tensor(cand['fuente'][f].astype('int64'),device='cuda'),torch.as_tensor(masc,device='cuda'))

def entrenar_corrector(cr,C,cand,emb,variante,p,s,car,progreso,fuentes,desplazamiento=0):
    if (car/'TERMINADO.json').exists():return
    motor.fijar_semilla(s)
    datos=MO.Datos(cr,C,cand,215617.90361589016,desplazamiento)
    a=datos.tramo(range(2012,2021));es=datos.tramo([2021]);assert len(a) and len(es)
    assert datos.dias[a].max()<np.datetime64('2021-01-01') and datos.dias[es].max()<np.datetime64('2022-01-01')
    mu,sd=datos.num[a].mean(0),datos.num[a].std(0)+1e-6
    cota=np.percentile(np.abs(datos.r[a]),99,axis=0).astype('float32')
    red=MO.Corrector(datos.num.shape[1],cand['feat'].shape[2],cand['n_fuentes'],p['d'],p['cabezas'],p['dropout'],cota).cuda()
    def perdida(ix):
        ix=a[ix.cpu().numpy()]; delta=red(*entradas_corrector(datos,cand,emb,variante,mu,sd,ix))
        y=torch.as_tensor(datos.y[ix],device='cuda'); ycr=torch.as_tensor(datos.ycr[ix],device='cuda')
        return (y-ycr*(1+delta)).abs().mean()/datos.q168
    def evaluar():
        tot=0.
        for k in range(0,len(es),256):
            ix=es[k:k+256]; delta=red(*entradas_corrector(datos,cand,emb,variante,mu,sd,ix))
            err=(torch.as_tensor(datos.y[ix],device='cuda')-torch.as_tensor(datos.ycr[ix],device='cuda')*(1+delta)).abs().mean()/datos.q168
            tot+=float(err)*len(ix)
        return tot/len(es)
    info=ajustar(red,len(a),perdida,evaluar,p,dict(max_epocas=60,min_epocas=1,paciencia=8,min_delta=1e-6),s,car,progreso,
                 f'Corrector {variante} d{desplazamiento} s{s}')
    met=dict(variante=variante,desplazamiento=desplazamiento,semilla=s,configuracion=p,fuentes=fuentes,
             ajuste=[str(datos.dias[a[0]]),str(datos.dias[a[-1]])],parada=[str(datos.dias[es[0]]),str(datos.dias[es[-1]])],
             n_num=datos.num.shape[1],n_feat=cand['feat'].shape[2],n_fuentes=cand['n_fuentes'],**{k:v for k,v in info.items() if k!='historia'})
    ck=dict(modelo=red.state_dict(),mu=mu,sd=sd,cota=cota,metadatos=met,fuentes=fuentes)
    checkpoint(car/'modelo.pt',ck)
    # Comprobar restauración antes de preparar la prueba.
    z=entradas_corrector(datos,cand,emb,variante,mu,sd,es[:2])
    with torch.no_grad():ref=red(*z)
    otro=MO.Corrector(met['n_num'],met['n_feat'],met['n_fuentes'],p['d'],p['cabezas'],p['dropout'],cota).cuda()
    otro.load_state_dict(cargar(car/'modelo.pt')['modelo']);otro.eval()
    with torch.no_grad():assert torch.equal(ref,otro(*z))
    guardar(car/'TERMINADO.json',met);del red,otro,z;torch.cuda.empty_cache()

def expandir_texto(col,car,datos,progreso,log):
    """Mapas congelados de desarrollo; añade vectores sin recodificar los históricos."""
    if (car/'LISTO.json').exists():return
    car.mkdir(parents=True,exist_ok=True)
    historico=DESARROLLO/('texto' if col=='disponible_declarado' else 'texto_'+col)
    res=leer(historico/'resumen.json');fuentes=res['fuentes']
    plan=leer(PAQUETE/'config/plan_corrector.json')['texto']
    c=TX.cargar_corpus(REPO/'data/processed/corpus_texto.parquet',col,plan['desde'],'2026-01-01',None,21000)
    frag=TX.fragmentos(c,1500,200,16);inter=TX.intervalos_corpus(c,frag,log,1500,200)
    dias=pd.date_range(plan['primer_dia_candidatos'],'2025-12-31')
    k=TX.candidatos(c,frag,inter,dias,fuentes,32,48,48,72,120)
    usados=np.unique(k['cand'][k['cand']>=0]);mapa=np.full(len(frag),-1,dtype='int64');mapa[usados]=np.arange(len(usados))
    compacto=np.where(k['cand']>=0,mapa[np.maximum(k['cand'],0)],-1).astype('int32')
    seleccion=frag.assign(doc_id=c.id.to_numpy()[frag.doc],fuente=c.fuente.to_numpy()[frag.doc]).iloc[usados].reset_index(drop=True)
    viejo=pd.read_parquet(historico/'fragmentos_usados.parquet');campos=['doc_id','orden','c0','c1','fuente']
    claves_viejas={tuple(v):i for i,v in enumerate(viejo[campos].itertuples(index=False,name=None))}
    asignacion=np.array([claves_viejas.get(tuple(v),-1) for v in seleccion[campos].itertuples(index=False,name=None)])
    faltan=np.flatnonzero(asignacion<0)
    cfg=datos.contrato['texto']
    if len(faltan):
        textos=[TX.contenido(c,int(frag.doc.iat[usados[i]]))[int(frag.c0.iat[usados[i]]):int(frag.c1.iat[usados[i]])] for i in faltan]
        class Barra:
            def __init__(self): self.n=0
            def update(self,n):self.n+=n;progreso(etiqueta=f'E5 {col}: {self.n}/{len(textos)} fragmentos nuevos',fragmentos=self.n,total_fragmentos=len(textos))
        nuevos=TX.codificar(textos,cfg,car/'vectores_nuevos.npy','cuda',log,barra=Barra())
    anterior=np.load(DESARROLLO/'texto_vectores'/f'vectores_{col}.npy',mmap_mode='r')
    vect=np.lib.format.open_memmap(car/'vectores.npy',mode='w+',dtype='float16',shape=(len(usados),768))
    for a in range(0,len(usados),4096):
        ii=np.arange(a,min(a+4096,len(usados)));si=asignacion[ii]>=0
        vect[ii[si]]=anterior[asignacion[ii[si]]]
    if len(faltan):vect[faltan]=nuevos
    vect.flush();del vect
    # Invariancia exacta de candidatos anteriores a 2022 pese a extender el corpus.
    z=np.load(historico/'candidatos.npz');n=len(z['dias'])
    inversion={tuple(v):i for i,v in enumerate(seleccion[campos].itertuples(index=False,name=None))}
    convertir=np.array([inversion[tuple(v)] for v in viejo[campos].itertuples(index=False,name=None)])
    esperado=np.where(z['cand']>=0,convertir[np.maximum(z['cand'],0)],-1)
    assert np.array_equal(esperado,compacto[:n])
    assert np.array_equal(z['feat'],k['feat'][:n].astype('float16')) and np.array_equal(z['fuente'],k['fuente'][:n])
    # Auditar disponibilidad real de cada fragmento candidato frente al origen.
    disp=pd.DatetimeIndex(c.disp).as_unit('ns').asi8[frag.doc.to_numpy()[usados]]
    org=dias.tz_localize('America/Bogota').tz_convert('UTC').as_unit('ns').asi8
    assert ((compacto<0)|(disp[np.maximum(compacto,0)]<=org[:,None])).all()
    np.savez_compressed(car/'candidatos.npz',dias=dias.strftime('%Y-%m-%d').to_numpy().astype('U10'),cand=compacto,feat=k['feat'].astype('float16'),fuente=k['fuente'])
    seleccion.to_parquet(car/'fragmentos.parquet',index=False)
    guardar(car/'LISTO.json',dict(fuentes=fuentes,n_fuentes=len(fuentes),reservas=k['reservas'],nuevos=int(len(faltan)),reutilizados=int((asignacion>=0).sum()),
              candidatos_pre2022_identicos=True,disponibilidad_verificada=True,fuente_desconocida=0))
    del c,frag,inter,k;gc.collect();torch.cuda.empty_cache()

def prueba_base_cr(datos,s,basecar,car,C):
    if (car/'base_cr.npz').exists():return
    car.mkdir(parents=True,exist_ok=True)
    ck=cargar(basecar/'modelo.pt');p=ck['metadatos']['configuracion']
    red=construir_experto(datos,'E',p).cuda();red.load_state_dict(ck['modelo']);red.eval()
    pos=datos.prueba();assert len(pos)==1461
    conjunto=datos.construir(pos,ck['escalas']);t=motor.a_tensores(conjunto,'cuda')
    pred=motor.predecir(red,t,256)*conjunto.nivel[:,None]
    dias=conjunto.origen.strftime('%Y-%m-%d').to_numpy().astype('U10');y=conjunto.y.astype('float32')
    h=RS.leer_residuos(DESARROLLO,list(range(2011,2022)),s)
    histy,histp=np.stack(h.y),np.stack(h.yhat)
    alfa=leer(DESARROLLO/'cr/resumen.json')[str(s)]['alfas']['2021']
    combinado=pd.concat([h,pd.DataFrame(dict(dia=pd.to_datetime(dias),anio=conjunto.origen.year,y=list(y),yhat=list(pred)))],ignore_index=True)
    yy,pp=np.stack(combinado.y),np.stack(combinado.yhat);rr=((yy-pp)/pp).mean(1)
    x=RS._rasgos_cr(combinado,C,rr);m=combinado.anio.to_numpy()<2022
    sc=StandardScaler().fit(x[m]);ridge=Ridge(alpha=alfa).fit(sc.transform(x[m]),rr[m]);rhat=ridge.predict(sc.transform(x[~m]))
    ycr=pred*(1+rhat[:,None])
    checkpoint(car/'ridge.pt',dict(media=sc.mean_,escala=sc.scale_,coef=ridge.coef_,intercepto=ridge.intercept_,alpha=alfa,congelado_antes_prueba=True))
    # C+R histórico se conserva exactamente, no se recalcula con el modelo final.
    z=np.load(DESARROLLO/'cr'/f'semilla_{s}.npz')
    np.savez_compressed(car/'base_cr.npz',dia=np.concatenate([z['dia'],dias]),anio=np.concatenate([z['anio'],conjunto.origen.year]),
       y=np.concatenate([z['y'],y]),yhat_e=np.concatenate([z['yhat_e'],pred.astype('float32')]),yhat_cr=np.concatenate([z['yhat_cr'],ycr.astype('float32')]))
    del red,t,conjunto;torch.cuda.empty_cache()

def predecir_corrector(ck,cr,C,cand,emb,car):
    if (car/'predicciones.npz').exists():return
    r=cargar(ck);met=r['metadatos'];p=met['configuracion']
    datos=MO.Datos(cr,C,cand,215617.90361589016,met['desplazamiento']);ix=datos.tramo(range(2022,2026));assert len(ix)==1461
    red=MO.Corrector(met['n_num'],met['n_feat'],met['n_fuentes'],p['d'],p['cabezas'],p['dropout'],r['cota']).cuda()
    red.load_state_dict(r['modelo']);red.eval();delta=[]
    with torch.no_grad():
        for k in range(0,len(ix),256):delta.append(red(*entradas_corrector(datos,cand,emb,met['variante'],r['mu'],r['sd'],ix[k:k+256])).cpu().numpy())
    delta=np.concatenate(delta);pred=datos.ycr[ix]*(1+delta)
    np.savez_compressed(car/'predicciones.npz',dia=datos.dias[ix].astype('U10'),y=datos.y[ix],pred=pred,delta=delta)
    guardar(car/'metricas_prueba.json',metricas(datos.y[ix],pred,datos.q168));del red;torch.cuda.empty_cache()

def ejecutar_final(salida,datos,cfg,progreso,completar,log):
    salida=Path(salida);plan=leer(PAQUETE/'config/plan_corrector.json');C=RS.calendario_diario(datos)
    pE=cfg['corrector']['base']['config'];p=cfg['corrector']['seleccion'];semillas=cfg['corrector']['semillas']
    for s in semillas:
        progreso(fase='corrector_base_final',estado='Ejecutando',etiqueta=f'E final s{s}')
        experto(datos,'E',pE,s,'final',salida/'base'/f'semilla_{s}',progreso,242048);completar()
    tareas=[(v,v,0,'disponible_declarado') for v in cfg['corrector']['variantes']]
    tareas += [(f'desalineado_{d}','oficial_prensa',d,'disponible_declarado') for d in cfg['corrector']['desplazamientos_placebo']]
    tareas += [('estricto','oficial_prensa',0,'disponible_estricto')]
    for col in ['disponible_declarado','disponible_estricto']:
        cand,emb,res=cargar_texto_historico(col)
        for nombre,v,d,cc in tareas:
            if cc!=col:continue
            for s in semillas:
                progreso(fase='corrector_ajuste',etiqueta=f'{nombre} s{s}')
                z=np.load(DESARROLLO/'cr'/f'semilla_{s}.npz');cr={k:z[k] for k in z.files}
                entrenar_corrector(cr,C,cand,emb,v,p,s,salida/'corrector'/nombre/f'semilla_{s}',progreso,res['fuentes'],d);completar()
        del emb,cand;torch.cuda.empty_cache()
    pesos=[x for x in salida.rglob('modelo.pt')]
    assert len(pesos)==36
    guardar(salida/'CONGELACION_ANTES_PRUEBA.json',dict(fecha=pd.Timestamp.now(tz='America/Bogota').isoformat(),pesos={str(x.relative_to(salida)):sha(x) for x in pesos},
          parametros_fijos=True,seleccion_informada_por_benchmark_historico=True,politica='Solo actualización causal de errores d-2; sin cambios de pesos'))
    for col in ['disponible_declarado','disponible_estricto']:
        progreso(fase='corrector_texto_prueba',etiqueta=col,epoca=None)
        expandir_texto(col,salida/'texto'/col,datos,progreso,log)
    for s in semillas:prueba_base_cr(datos,s,salida/'base'/f'semilla_{s}',salida/'predicciones_base'/f'semilla_{s}',C)
    for col in ['disponible_declarado','disponible_estricto']:
        tx=salida/'texto'/col;z=np.load(tx/'candidatos.npz');res=leer(tx/'LISTO.json')
        cand={k:z[k] for k in z.files};cand.update(reservas=res['reservas'],n_fuentes=res['n_fuentes'])
        emb=torch.from_numpy(np.load(tx/'vectores.npy')).cuda()
        for nombre,v,d,cc in tareas:
            if cc!=col:continue
            for s in semillas:
                progreso(fase='corrector_prediccion',etiqueta=f'{nombre} s{s}',epoca=None)
                cr=dict(np.load(salida/'predicciones_base'/f'semilla_{s}'/'base_cr.npz'))
                car=salida/'corrector'/nombre/f'semilla_{s}';predecir_corrector(car/'modelo.pt',cr,C,cand,emb,car)
        del emb,cand;torch.cuda.empty_cache()
    progreso(fase='corrector_estadistica',etiqueta='Bootstrap por bloques y Holm',epoca=None)
    errores={k:[] for k in ['E','C+R']+[t[0] for t in tareas]};filas=[]
    for s in semillas:
        cr=dict(np.load(salida/'predicciones_base'/f'semilla_{s}'/'base_cr.npz'));m=cr['anio']>=2022;yy=cr['y'][m]
        for v,k in [('E','yhat_e'),('C+R','yhat_cr')]:
            pr=cr[k][m];errores[v].append(np.abs(yy-pr)/datos.q168);filas.append(dict(variante=v,semilla=s,**metricas(yy,pr,datos.q168)))
        for nombre,_,_,_ in tareas:
            z=np.load(salida/'corrector'/nombre/f'semilla_{s}'/'predicciones.npz');assert np.array_equal(z['dia'],cr['dia'][m])
            errores[nombre].append(np.abs(z['y']-z['pred'])/datos.q168);filas.append(dict(variante=nombre,semilla=s,**metricas(z['y'],z['pred'],datos.q168)))
    eh={k:np.mean(v,axis=0) for k,v in errores.items()};eh['desalineado']=np.mean([eh[f'desalineado_{d}'] for d in cfg['corrector']['desplazamientos_placebo']],axis=0)
    err={k:v.mean(1) for k,v in eh.items()};an=cr['anio'][m]
    zz=np.load(salida/'texto/disponible_declarado/candidatos.npz');pos={d:i for i,d in enumerate(zz['dias'])}
    ii=np.array([pos[d] for d in cr['dia'][m]])
    alineado=((zz['cand'][ii,:32]>=0)&(zz['feat'][ii,:32,0]>.5)).any(1)
    subconjuntos={'global':np.ones(len(an),dtype=bool)}
    if alineado.sum()>=20:subconjuntos['con_anuncio_oficial_alineado']=alineado
    contr=EV.evaluar(err,an,subconjuntos,9999,14)
    # Sensibilidad estricta separada de la familia primaria.
    idx=EV.indices_bootstrap(an,9999,14,20261121)
    extra=[dict(subconjunto='sensibilidad_estricta',A='estricto',B=b,**EV.contraste(err['estricto'],err[b],idx)) for b in ['C+R','num_horario','oficial_prensa']]
    out=salida/'evaluacion';out.mkdir(exist_ok=True)
    pd.concat([contr,pd.DataFrame(extra)]).to_csv(out/'contrastes.csv',index=False)
    pd.DataFrame(filas).to_csv(out/'metricas_por_semilla.csv',index=False)
    anual=[dict(variante=k,anio=int(a),MASE=float(v[an==a].mean())) for k,v in err.items() for a in np.unique(an)]
    pd.DataFrame(anual).to_csv(out/'metricas_por_anio.csv',index=False)
    pd.concat([EV.por_hora(eh,k,'C+R') for k in ['num_horario','oficial','oficial_prensa']]).to_csv(out/'por_hora.csv',index=False)
    guardar(out/'resultado.json',dict(MASE={k:float(v.mean()) for k,v in err.items()},veredicto=EV.veredicto(contr),origenes=len(an),periodo='2022-2025',semillas=semillas))
    (out/'RESUMEN.md').write_text('# Evaluación final del corrector · prueba 2022–2025\n\n'+pd.DataFrame(filas).groupby('variante').agg(MASE_media=('MASE','mean'),MASE_DE=('MASE','std')).to_csv()+'\nLos contrastes y sus intervalos están en contrastes.csv. La prueba fue consultada previamente para el benchmark; no se presenta como reserva nunca examinada.\n',encoding='utf-8')
    guardar(salida/'FINALIZADO.json',dict(fecha=pd.Timestamp.now(tz='America/Bogota').isoformat(),estado='finalizado',checkpoints=36,origenes=1461))
    log('Evaluación final del corrector terminada; continúa M1.')

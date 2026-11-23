"""M1: selección convexa de expertos con supervisión de complementariedad; desarrollo y ajuste preprueba."""
from __future__ import annotations
import itertools
import shutil
from e3_gpu_comun import *
from scipy.optimize import nnls, linprog

EXPERTOS=['patchtst','E','periodica']
PARES=[(0,1),(0,2),(1,2)]

def complementariedad(y,pred,q):
    r=(y[:,:,None]-pred)/q
    return np.stack([np.abs(r[:,:,i])+np.abs(r[:,:,j])-np.abs(r[:,:,i]+r[:,:,j]) for i,j in PARES],axis=-1).astype('float32')

class Router(nn.Module):
    def __init__(self,n,p):
        super().__init__();w=p['ancho']
        self.contexto=nn.Sequential(nn.Linear(n,w),nn.GELU(),nn.Dropout(p['dropout']),nn.Linear(w,w),nn.GELU())
        self.complemento=nn.Linear(w,24*3)
        self.logits=nn.Linear(w+24*3,24*3)
    def forward(self,x,pred):
        z=self.contexto(x);c=self.complemento(z).reshape(-1,24,3)
        w=self.logits(torch.cat([z,c.flatten(1)],1)).reshape(-1,24,3).softmax(-1)
        return (w*pred).sum(-1),c,w

def datos_meta(datos,car,semilla):
    partes=[]
    for anio in range(2011,2022):
        z=[np.load(car/'expertos'/k/f'anio_{anio}'/f'semilla_{semilla}'/'predicciones.npz') for k in EXPERTOS]
        assert all(np.array_equal(z[0]['dia'],x['dia']) and np.array_equal(z[0]['y'],x['y']) for x in z[1:])
        partes.append(dict(dia=z[0]['dia'],y=z[0]['y'],pred=np.stack([x['pred'] for x in z],-1)))
    dias=np.concatenate([x['dia'] for x in partes]);y=np.concatenate([x['y'] for x in partes]);pred=np.concatenate([x['pred'] for x in partes])
    assert len(np.unique(dias))==len(dias)
    fechas=pd.DatetimeIndex(dias).tz_localize('America/Bogota');pos=datos.indice.get_indexer(fechas);assert (pos>=0).all()
    # Rasgos brutos disponibles por origen. Solo el escalador de cada fold aprende parámetros.
    ii=pos[:,None]+np.arange(-datos.L,0);xx=datos.X[ii]
    # Ausencias se codifican con 0 y máscaras explícitas; no hay imputación estimada global.
    valid=np.isfinite(xx);limpio=np.where(valid,xx,0.)
    medias=limpio.sum(1)/np.maximum(valid.sum(1),1)
    ultimo=limpio[:,-1];m24=limpio[:,-24:].sum(1)/np.maximum(valid[:,-24:].sum(1),1)
    ausente=1-valid.mean(1)
    nivel=xx[:,:,0].mean(1);assert (nivel>0).all()
    cf=datos.K[pos[:,None]+np.arange(24)]
    resumen=np.concatenate([ultimo,medias,m24,ausente],1)
    resumen[:,[0,len(datos.hist),2*len(datos.hist)]]/=nivel[:,None]
    errores=(y[:,:,None]-pred)/datos.q168
    lag=lags_diarios(dias,errores)
    x=np.concatenate([resumen,cf.reshape(len(dias),-1),(pred/nivel[:,None,None]).reshape(len(dias),-1),
                      pred.std(-1)/nivel[:,None],lag],1).astype('float32')
    assert np.isfinite(x).all()
    return dict(dia=dias,anio=fechas.year.to_numpy(),x=x,y=y,pred=pred,c=complementariedad(y,pred,datos.q168),nivel=nivel)

def dividir(meta,anio):
    a=np.flatnonzero(meta['anio']<=anio-2);es=np.flatnonzero(meta['anio']==anio-1);ev=np.flatnonzero(meta['anio']==anio)
    # Purga explícita de un día entre bloques de objetivos; no elimina orígenes de evaluación.
    corte=np.datetime64(meta['dia'][es[0]],'D')-np.timedelta64(1,'D')
    a=a[np.asarray(meta['dia'][a],dtype='datetime64[D]')<corte]
    assert len(a) and len(es) and (len(ev) or anio==2022)
    return a,es,ev

def entrenar_router(meta,p,s,a,es,ev,car,progreso,q,aux=True):
    if (car/'TERMINADO.json').exists():return leer(car/'TERMINADO.json')
    motor.fijar_semilla(s);mu=meta['x'][a].mean(0);sd=meta['x'][a].std(0);sd=np.where(sd>1e-6,sd,1.)
    x=torch.as_tensor((meta['x']-mu)/sd,device='cuda');pr=torch.as_tensor(meta['pred'],device='cuda')
    y=torch.as_tensor(meta['y'],device='cuda');c=torch.as_tensor(meta['c'],device='cuda')
    red=Router(x.shape[1],p).cuda()
    def loss(ix):
        j=torch.as_tensor(a,device='cuda')[ix];sal,cc,w=red(x[j],pr[j]);principal=(sal-y[j]).abs().mean()/q
        auxiliar=nn.functional.smooth_l1_loss(cc,c[j]) if aux else sal.new_zeros(())
        return principal+(p['lambda_complementariedad'] if aux else 0)*auxiliar+.001*((w[:,1:]-w[:,:-1])**2).mean()
    def evaluar():
        total=0.
        for k in range(0,len(es),256):
            j=es[k:k+256];sal=red(x[j],pr[j])[0];total+=float((sal-y[j]).abs().mean()/q)*len(j)
        return total/len(es)
    info=ajustar(red,len(a),loss,evaluar,dict(lr=.001,weight_decay=.0001,lote=64),
                 dict(max_epocas=100,min_epocas=10,paciencia=10,min_delta=.0001),s,car,progreso,f'M1 cfg{p["id"]} s{s} · {meta["anio"][es[0]]+1}')
    ck=dict(modelo=red.state_dict(),media=mu,escala=sd,configuracion=p,n_entrada=x.shape[1],semilla=s,
            complemento_supervisado=aux,ajuste=[meta['dia'][a[0]],meta['dia'][a[-1]]],parada=[meta['dia'][es[0]],meta['dia'][es[-1]]])
    checkpoint(car/'modelo.pt',ck)
    otro=Router(x.shape[1],p).cuda();otro.load_state_dict(cargar(car/'modelo.pt')['modelo']);otro.eval()
    with torch.no_grad():
        assert torch.equal(red(x[es[:2]],pr[es[:2]])[0],otro(x[es[:2]],pr[es[:2]])[0])
        partes=[otro(x[ev[k:k+256]],pr[ev[k:k+256]])[0].cpu().numpy() for k in range(0,len(ev),256)] if len(ev) else []
    resultado=dict(configuracion=p,semilla=s,auxiliar=aux,n_parametros=sum(v.numel() for v in red.parameters()),
                   **{k:v for k,v in info.items() if k!='historia'},ajuste=ck['ajuste'],parada=ck['parada'])
    if partes:
        pred=np.concatenate(partes);resultado.update(metricas(meta['y'][ev],pred,q))
        np.savez_compressed(car/'predicciones.npz',dia=meta['dia'][ev],y=meta['y'][ev],pred=pred)
    guardar(car/'TERMINADO.json',resultado);del red,otro,x,pr,y,c;torch.cuda.empty_cache()
    return resultado

def controles(meta,a,es,ev,car,q):
    car.mkdir(parents=True,exist_ok=True);pesos=np.zeros((24,3));conv=np.zeros((24,3))
    for h in range(24):
        x=meta['pred'][es,h]/q;y=meta['y'][es,h]/q
        pesos[h]=nnls(x,y,maxiter=10000)[0]
        n=len(y);ident=np.eye(n)
        r=linprog(np.r_[np.zeros(3),np.ones(n)/n],A_ub=np.r_[np.c_[x,-ident],np.c_[-x,-ident]],
                  b_ub=np.r_[y,-y],A_eq=np.r_[np.ones(3),np.zeros(n)][None],b_eq=[1.],bounds=(0,None),method='highs')
        if not r.success:raise RuntimeError('Mezcla convexa no convergió: '+r.message)
        conv[h]=r.x[:3]
    guardar(car/'pesos.json',dict(NNLS=pesos.tolist(),convexa=conv.tolist(),EWA_eta=1,retraso_dias=2,ajuste_pesos='parada previa al año de evaluación'))
    if not len(ev):return
    # EWA por hora, temperatura prefijada eta=1; actualización solo con error observable d-2.
    orden=np.concatenate([es,ev]);acum=np.zeros((24,3));ew=[];mapa={np.datetime64(meta['dia'][j],'D'):j for j in orden}
    for j in orden:
        fecha=np.datetime64(meta['dia'][j],'D');pasado=mapa.get(fecha-np.timedelta64(2,'D'))
        if pasado is not None:acum+=np.abs(meta['y'][pasado,:,None]-meta['pred'][pasado])/q
        logits=-acum;w=np.exp(logits-logits.max(-1,keepdims=True));w/=w.sum(-1,keepdims=True);ew.append((w*meta['pred'][j]).sum(-1))
    pred_ewa=np.asarray(ew)[len(es):]
    guardar(car/'pesos.json',dict(NNLS=pesos.tolist(),convexa=conv.tolist(),EWA_eta=1,retraso_dias=2,ajuste_pesos='parada previa al año de evaluación'))
    filas=[]
    for nombre,pred in [('NNLS',(meta['pred'][ev]*pesos).sum(-1)),('convexa_fija',(meta['pred'][ev]*conv).sum(-1)),('media',meta['pred'][ev].mean(-1)),('EWA',pred_ewa)]:
        np.savez_compressed(car/f'{nombre}.npz',dia=meta['dia'][ev],y=meta['y'][ev],pred=pred)
        filas.append(dict(control=nombre,**metricas(meta['y'][ev],pred,q)))
    guardar(car/'metricas.json',filas)

def ejecutar_m1(salida,datos,diseno,config,corrector,progreso,completar,log):
    salida=Path(salida);semillas=diseno['validacion']['semillas'];especificaciones=diseno['expertos_configuraciones_seleccionadas']
    for anio in range(2011,2022):
        for nombre in EXPERTOS:
            pp=especificaciones[nombre]['configuracion'];esperado=especificaciones[nombre]['n_parametros']
            for s in semillas:
                car=salida/'expertos'/nombre/f'anio_{anio}'/f'semilla_{s}'
                progreso(fase='m1_expertos_oof',etiqueta=f'{nombre} {anio} s{s}')
                viejo=PAQUETE/'resultados_corrector_e3/residuos'/f'anio_{anio}'/f'semilla_{s}'
                if nombre=='E' and (viejo/'resultado.json').exists() and not (car/'TERMINADO.json').exists():
                    rr=leer(viejo/'resultado.json');assert rr['configuracion']==pp and rr['semilla']==s and rr['dispositivo']=='cuda'
                    assert rr['origen_evaluacion'][0].startswith(str(anio)) and rr['presupuesto']==dict(max_epocas=200,min_epocas=10,paciencia=6)
                    z=np.load(viejo/f'pred_val_{anio}.npz');car.mkdir(parents=True,exist_ok=True)
                    np.savez_compressed(car/'predicciones.npz',dia=z['origen'],y=z['y_kwh'],pred=z['yhat_kwh'])
                    guardar(car/'TERMINADO.json',dict(reutilizado=True,origen=str(viejo),sha256=sha(viejo/f'pred_val_{anio}.npz'),configuracion=pp,semilla=s,
                           contrato=sha(PAQUETE/'config/diseno_entrenamiento_texto.json'),panel=datos.hash_panel))
                else:experto(datos,nombre,pp,s,anio,car,progreso,esperado)
                completar()
    g=diseno['modelos']['M1']['rejilla'];cfgs=[dict(id=i+1,ancho=w,dropout=dp,lambda_complementariedad=lam) for i,(w,dp,lam) in enumerate(itertools.product(g['ancho'],g['dropout'],g['lambda_complementariedad']))]
    guardar(salida/'rejilla_registrada.json',cfgs)
    filas=[]
    for s in semillas:
        meta=datos_meta(datos,salida,s)
        for anio in diseno['validacion']['folds_externos']:
            a,es,ev=dividir(meta,anio)
            for p in cfgs:
                progreso(fase='m1_busqueda',etiqueta=f'M1 cfg{p["id"]} {anio} s{s}')
                r=entrenar_router(meta,p,s,a,es,ev,salida/'busqueda'/f'cfg_{p["id"]}'/f'anio_{anio}'/f'semilla_{s}',progreso,datos.q168)
                filas.append(dict(cfg=p['id'],anio=anio,semilla=s,MASE=r['MASE'],n_parametros=r['n_parametros']));completar()
            controles(meta,a,es,ev,salida/'controles'/f'anio_{anio}'/f'semilla_{s}',datos.q168)
    tabla=pd.DataFrame(filas);tabla.to_csv(salida/'seleccion_por_fold_semilla.csv',index=False)
    seleccion=tabla.groupby('cfg').agg(MASE=('MASE','mean'),n_parametros=('n_parametros','first')).reset_index().sort_values(['MASE','n_parametros','cfg'])
    seleccion.to_csv(salida/'seleccion_resumen.csv',index=False);elegida=next(p for p in cfgs if p['id']==int(seleccion.iloc[0]['cfg']))
    guardar(salida/'CONFIGURACION_ELEGIDA.json',dict(configuracion=elegida,criterio='Media de los cuatro folds y cinco semillas con pesos iguales',prueba_consultada_para_seleccion=False))
    for s in semillas:
        meta=datos_meta(datos,salida,s)
        for anio in diseno['validacion']['folds_externos']:
            a,es,ev=dividir(meta,anio);progreso(fase='m1_control_sin_auxiliar',etiqueta=f'Control M1 {anio} s{s}')
            entrenar_router(meta,elegida,s,a,es,ev,salida/'sin_complementariedad'/f'anio_{anio}'/f'semilla_{s}',progreso,datos.q168,False);completar()
        # Preparar los tres expertos finales necesarios para restaurar M1; no se predice prueba de M1.
        for nombre in EXPERTOS:
            car=salida/'expertos_finales'/nombre/f'semilla_{s}';progreso(fase='m1_expertos_finales',etiqueta=f'{nombre} final s{s}')
            base_corrector=Path(corrector)/'base'/f'semilla_{s}'
            if nombre=='E' and (base_corrector/'TERMINADO.json').exists() and not car.exists():shutil.copytree(base_corrector,car)
            else:experto(datos,nombre,especificaciones[nombre]['configuracion'],s,'final',car,progreso,especificaciones[nombre]['n_parametros'])
            completar()
        a,es,_=dividir(meta,2022);progreso(fase='m1_ajuste_final',etiqueta=f'M1 final preprueba s{s}')
        controles(meta,a,es,np.array([],dtype='int64'),salida/'controles_finales'/f'semilla_{s}',datos.q168)
        entrenar_router(meta,elegida,s,a,es,np.array([],dtype='int64'),salida/'final'/f'semilla_{s}',progreso,datos.q168);completar()
    (salida/'RESUMEN.md').write_text('# M1 · desarrollo y ajuste previo a prueba\n\n'+seleccion.to_csv(index=False)+'\nCinco semillas, folds 2018–2021, ocho candidatos. La selección es de desarrollo; no estima de forma independiente el sesgo de búsqueda. Checkpoints finales ajustados con OOF hasta 2020 y parada 2021. No se ejecutó la prueba 2022–2025 de M1. El OOF anidado de M1 para alimentar M3/M4 es una etapa futura, no sustituible por predicciones en muestra.\n',encoding='utf-8')
    guardar(salida/'FINALIZADO.json',dict(fecha=pd.Timestamp.now(tz='America/Bogota').isoformat(),configuracion=elegida,semillas=semillas,prueba_m1_ejecutada=False,
           oof_anidado_para_M3_M4='pendiente; no necesario para entrenar M1 sobre OOF de expertos'))
    log('Entrenamiento de M1 terminado y checkpoints guardados.')

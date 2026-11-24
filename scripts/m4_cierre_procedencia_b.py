# Variante de compatibilidad: normalización escalar de fechas NumPy.
# La fuente registrada original m4_cierre_procedencia.py permanece intacta.
"""Ablaciones de procedencia residual con bases externas idénticas y GPU obligatoria."""
from e3_gpu_comun import *
import m4_cierre_controles as C
import m4_cierre_datos as E
from e3_modelo1 import entrenar_router, complementariedad
from probar_m1_gpu import rasgos

MODOS=['residuos_en_muestra','residuos_OOF_no_anidado']

def metarouter(meta):
    return dict(dia=meta['dia'],anio=meta['anio'],x=meta['num'],y=meta['y'],pred=meta['expertos'],c=complementariedad(meta['y'],meta['expertos'],1.))

def desde_npz(p):
    with np.load(p) as z:return {k:z[k] for k in z.files}

def banco_residual(meta,pred_alternativa,a,q):
    """Solo sustituye residuos del ajuste. La actualización revelada sigue la base real."""
    assert pred_alternativa.shape==meta['y'].shape
    rr=((meta['y']-meta['pred_base'])/q).astype('float32')
    rr[a]=(meta['y'][a]-pred_alternativa[a])/q
    assert np.isfinite(rr).all()
    return rr

def inferir_experto(datos,nombre,ck,pos):
    c=datos.construir(pos,ck['escalas']);t=motor.a_tensores(c,'cuda')
    cfg=ck.get('configuracion',ck.get('metadatos',{}).get('configuracion'))
    red=construir_experto(datos,nombre,cfg).cuda().eval();red.load_state_dict(ck['modelo'])
    with torch.no_grad():pr=(motor.predecir(red,t,128)*c.nivel[:,None]).astype('float32')
    del red,t;torch.cuda.empty_cache();return pr

def insample(datos,meta,a,es,s,car,progreso):
    p=car/'residuos_fuente.npz'
    if p.exists():return desde_npz(p)['pred']
    # Reajustar la pila completa únicamente con el prefijo externo; sus
    # retropredicciones del ajuste son deliberadamente en muestra.
    indice=datos.indice.tz_localize(None);pos=indice.get_indexer(pd.DatetimeIndex(meta['dia']))
    assert (pos>=168).all()
    todas=datos.pos if hasattr(datos,'pos') else datos.elegibles(np.arange(24*8,len(datos.Y)-23,24))
    dias_todas=indice[todas];inicio=pd.Timestamp(str(meta['dia'][es[0]]));fin=pd.Timestamp(str(meta['dia'][es[-1]]))
    tr=todas[dias_todas<inicio-pd.Timedelta(days=1)]
    st=todas[(dias_todas>=inicio)&(dias_todas<=fin)]
    assert len(tr) and len(st) and indice[tr[-1]]+pd.Timedelta(hours=23)<indice[st[0]]
    esp=leer(REPO/'continuidad/DISENO_CUATRO_MODELOS_GPU.json')['expertos_configuraciones_seleccionadas'];pp=[]
    for nombre in ['patchtst','E','periodica']:
        d=car/'expertos'/nombre
        E.entrenar_base(datos,nombre,esp[nombre]['configuracion'],s,tr,st,np.array([],dtype='int64'),d,progreso)
        pp.append(inferir_experto(datos,nombre,cargar(d/'modelo.pt'),pos))
    pred=np.stack(pp,-1)
    # Rasgos reconstruidos con esta pila; escaladores del router solo aprenden a.
    anterior=datos.indice
    if datos.nombre!='colombia':datos.indice=indice.tz_localize('America/Bogota')
    try:x=rasgos(datos,meta['dia'],meta['y'],pred)
    finally:datos.indice=anterior
    mm=dict(dia=meta['dia'],anio=meta['anio'],x=x,y=meta['y'],pred=pred,c=complementariedad(meta['y'],pred,datos.q168))
    E.entrenar_router(mm,dict(id=8,ancho=64,dropout=.1,lambda_complementariedad=.1),s,a,es,np.array([],dtype='int64'),car/'router',progreso,datos.q168)
    z=E.inferir_meta(mm,np.arange(len(meta['dia'])),cargar(car/'router/modelo.pt'))
    car.mkdir(parents=True,exist_ok=True);np.savez_compressed(p,dia=meta['dia'],pred=z['pred_base'])
    guardar(car/'PROCEDENCIA.json',dict(tipo='Pila completa en muestra dentro del prefijo externo',ajuste_expertos=[str(indice[tr[0]]),str(indice[tr[-1]])],parada_expertos=[str(indice[st[0]]),str(indice[st[-1]])],ajuste_router=[meta['dia'][a[0]],meta['dia'][a[-1]]],advertencia='Retropredicciones solo para el banco histórico; no se califican como pronósticos operativos. La consulta conserva la base OOF original.'))
    return z['pred_base']

def oof_fija(meta,s,car,q,progreso,nombre,m1dir=None):
    p=car/'residuos_fuente.npz'
    if p.exists():return desde_npz(p)['pred']
    if nombre!='colombia':
        # No hubo selección de hiperparámetros en los otros conjuntos: no existe
        # una diferencia de anidamiento que pueda inventarse por repetir el ajuste.
        car.mkdir(parents=True,exist_ok=True);np.savez_compressed(p,dia=meta['dia'],pred=meta['pred_base'])
        guardar(car/'IDENTIDAD.json',dict(motivo='Configuración fijada antes de métricas, sin selección por prefijos; OOF y OOF anidado coinciden.',ajustes=0))
        return meta['pred_base']
    from e3_modelo1 import datos_meta, dividir
    from m4_cierre_colombia import HIST
    origen=HIST if s<=105 else m1dir
    assert origen is not None
    raw=datos_meta(DatosNumericos(PAQUETE),origen,s)
    pr=meta['pred_base'].copy();dias=pd.DatetimeIndex(meta['dia'])
    for anio in range(2013,2022):
        a,es,ev=dividir(raw,anio)
        entrenar_router(raw,dict(id=8,ancho=64,dropout=.1,lambda_complementariedad=.1),s,a,es,np.array([],dtype='int64'),car/f'prefijo_{anio}',progreso,q)
        rr=E.inferir_meta(raw,ev,cargar(car/f'prefijo_{anio}/modelo.pt'))
        ix=np.flatnonzero(dias.year==anio);assert np.array_equal(rr['dia'],meta['dia'][ix]);pr[ix]=rr['pred_base']
    car.mkdir(parents=True,exist_ok=True);np.savez_compressed(p,dia=meta['dia'],pred=pr)
    guardar(car/'PROCEDENCIA.json',dict(configuracion=8,tipo='OOF temporal sin selección anidada; configuración elegida con desarrollo completo histórico',alcance='Ablación retrospectiva de selección; no confirmatoria'))
    return pr

def ajustar(meta,residuos,a,es,ev,s,car,q,progreso):
    if (car/'TERMINADO.json').exists():return
    motor.fijar_semilla(s);mu=meta['num'][a].mean(0);sd=meta['num'][a].std(0);sd=np.where(sd>1e-6,sd,1.)
    b=C.preparar_banco(meta,mu,sd,q);b.residuos=torch.as_tensor(residuos,device='cuda')
    red=C.Memoria(meta['num'].shape[1],7,0,C.CFG,'M4_N').cuda();red.q_dataset=q;red.recuperar=C.recuperar_q.__get__(red,C.Memoria)
    y=torch.as_tensor(meta['y'],device='cuda');aa=torch.as_tensor(a,device='cuda')
    def salida(j):
        lim=int(j.max())+1;qv,kv=red.codificar(b,lim)
        return red.recuperar(qv,kv,b.dias[:lim],b.residuos[:lim],j,b.base[j])[0]
    def val():
        return sum(float((salida(torch.as_tensor(j,device='cuda'))-y[j]).abs().mean()/q)*len(j) for j in np.array_split(es,max(1,int(np.ceil(len(es)/128)))))/len(es)
    info=ajustar_motor(red,len(a),lambda j:(salida(aa[j])-y[aa[j]]).abs().mean()/q,val,dict(lr=.001,weight_decay=.0001,lote=64),dict(max_epocas=100,min_epocas=10,paciencia=10,min_delta=.0001),s,car,progreso,f'Procedencia residual s{s}')
    ck=dict(tipo='memoria',variante='M4_N',configuracion=C.CFG,n_num=meta['num'].shape[1],n_feat=7,n_fuentes=0,media=mu,escala=sd,limite_memoria=None,semilla=s,modelo=red.state_dict(),q=q)
    checkpoint(car/'modelo.pt',ck);np.savez_compressed(car/'residuos_banco.npz',dia=meta['dia'],residuos=residuos)
    guardar(car/'escaladores.json',dict(media=mu.tolist(),escala=sd.tolist(),ajuste=[meta['dia'][a[0]],meta['dia'][a[-1]]]))
    if len(ev):
        pred=predecir(ck,meta,residuos,ev);np.savez_compressed(car/'predicciones.npz',dia=meta['dia'][ev],y=meta['y'][ev],pred=pred)
    guardar(car/'TERMINADO.json',{k:v for k,v in info.items() if k!='historia'});del red,b,y;torch.cuda.empty_cache()

# Evitar que el nombre local del entrenador oculte el motor compartido.
from e3_gpu_comun import ajustar as ajustar_motor

def predecir(ck,meta,residuos,ix):
    b=C.preparar_banco(meta,ck['media'],ck['escala'],ck['q']);b.residuos=torch.as_tensor(residuos,device='cuda')
    red=C.Memoria(ck['n_num'],7,0,C.CFG,'M4_N').cuda().eval();red.load_state_dict(ck['modelo']);red.q_dataset=ck['q'];red.recuperar=C.recuperar_q.__get__(red,C.Memoria)
    pp=[]
    with torch.no_grad():
        qv,kv=red.codificar(b,len(meta['dia']))
        for jj in np.array_split(ix,max(1,int(np.ceil(len(ix)/128)))):
            j=torch.as_tensor(jj,device='cuda');pp.append(red.recuperar(qv,kv,b.dias,b.residuos,j,b.base[j])[0].cpu().numpy())
    del red,b;torch.cuda.empty_cache();return np.concatenate(pp)

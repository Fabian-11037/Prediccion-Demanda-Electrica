"""Datos adicionales y bases OOF trimestrales; sin texto ni covariables futuras."""
from e3_gpu_comun import *
from entrenamiento.datos import DatosNumericos, Conjunto
from e3_modelo1 import Router, entrenar_router, complementariedad
from probar_m1_gpu import rasgos
from scipy.optimize import nnls

class DatosAdicionales(DatosNumericos):
    def __init__(self,nombre):
        p=REPO/f'data/external/m4_cierre_20261123/{nombre}.parquet';t=pd.read_parquet(p)
        self.raiz=p.parent;self.hash_panel=sha(p);self.nombre=nombre
        self.indice=t.index.tz_localize('UTC');self.Y=t.y.to_numpy(dtype='float64')
        self.X=pd.DataFrame({'x':self.Y}).shift(1).to_numpy();self.hist=['x_demanda_real_kwh'];self.nombres_mascara=[]
        self.M=np.zeros((len(t),0),dtype='float32');self.L=168;self.H=24
        d=self.indice;h=d.hour.to_numpy();w=d.dayofweek.to_numpy();a=d.dayofyear.to_numpy();m=d.month.to_numpy()
        self.K=np.column_stack([np.sin(2*np.pi*h/24),np.cos(2*np.pi*h/24),np.sin(2*np.pi*w/7),np.cos(2*np.pi*w/7),np.sin(2*np.pi*a/365.25),np.cos(2*np.pi*a/365.25),(w>=5).astype(float),*[(w==i).astype(float) for i in range(7)]])
        self.cal=[f'cal_{i}' for i in range(self.K.shape[1])];self.dia_base=d[0].normalize()
        self._off_l=np.arange(-168,0);self._off_h=np.arange(24)
        # Escala fija de los primeros seis meses, anteriores al inicio OOF.
        inicial=self.Y[:24*181];self.q168=float(np.abs(inicial[168:]-inicial[:-168]).mean());assert self.q168>0
        self.pos=self.elegibles(np.arange(24*8,len(t)-23,24))
        self.test_anio={'uci':2014,'gefcom':2011,'opsd':2019}[nombre]
        self.inicio_oof={'uci':2013,'gefcom':2006,'opsd':2016}[nombre]

    def dias_pos(self,pos):return self.indice[pos].strftime('%Y-%m-%d').to_numpy().astype('U10')

def entrenar_base(datos,nombre,p,s,tr,es,ev,car,progreso):
    if (car/'TERMINADO.json').exists():return
    motor.fijar_semilla(s);esc=datos.ajustar_escalas(tr)
    conj={k:datos.construir(j,esc) for k,j in [('tr',tr),('es',es)]};t={k:motor.a_tensores(z,'cuda') for k,z in conj.items()}
    red=construir_experto(datos,nombre,p).cuda()
    def pred(k,j):z=t[k];return red(z['x'][j],z['cal_p'][j],z['cal_f'][j],z['dia'][j])
    def loss(j):return (pred('tr',j)-t['tr']['y'][j]).abs().mean()
    def val():return sum(float((pred('es',slice(i,i+128))-t['es']['y'][i:i+128]).abs().mean())*len(t['es']['y'][i:i+128]) for i in range(0,len(es),128))/len(es)
    info=ajustar(red,len(tr),loss,val,p,dict(max_epocas=200,min_epocas=10,paciencia=6,relativa=True,scheduler=True),s,car,progreso,f'{datos.nombre} {nombre} s{s}')
    checkpoint(car/'modelo.pt',dict(modelo=red.state_dict(),escalas=esc,configuracion=p,nombre=nombre,semilla=s))
    if len(ev):
        c=datos.construir(ev,esc);tt=motor.a_tensores(c,'cuda');pr=(motor.predecir(red,tt,128)*c.nivel[:,None]).astype('float32')
        np.savez_compressed(car/'predicciones.npz',dia=datos.dias_pos(ev),y=c.y.astype('float32'),pred=pr)
        del c,tt
    guardar(car/'escaladores.json',escalas_a_json(esc,datos));guardar(car/'TERMINADO.json',dict(configuracion=p,semilla=s,ajuste=[str(datos.indice[tr[0]]),str(datos.indice[tr[-1]])],parada=[str(datos.indice[es[0]]),str(datos.indice[es[-1]])],**{k:v for k,v in info.items() if k!='historia'}))
    del red,t,conj;torch.cuda.empty_cache()

def meta_desde_pred(datos,partes):
    dias=np.concatenate([p['dia'] for p in partes]);y=np.concatenate([p['y'] for p in partes]);pred=np.concatenate([p['pred'] for p in partes])
    # rasgos usa tz America/Bogota para localizar: el adaptador mantiene el mismo
    # reloj civil UTC como etiqueta de serie, sin aplicar festivos colombianos.
    anterior=datos.indice;datos.indice=anterior.tz_localize(None).tz_localize('America/Bogota')
    try:x=rasgos(datos,dias,y,pred)
    finally:datos.indice=anterior
    return dict(dia=dias,anio=pd.DatetimeIndex(dias).year.to_numpy(),x=x,y=y,pred=pred,c=complementariedad(y,pred,datos.q168))

def inferir_meta(meta,ix,ck):
    red=Router(ck['n_entrada'],ck['configuracion']).cuda().eval();red.load_state_dict(ck['modelo'])
    out=[];cc=[];ll=[]
    h=red.logits.register_forward_hook(lambda m,i,o:ll.append(o.detach().reshape(-1,24,3).cpu().numpy()))
    with torch.no_grad():
        for j in np.array_split(ix,max(1,int(np.ceil(len(ix)/128)))):
            x=torch.as_tensor((meta['x'][j]-ck['media'])/ck['escala'],device='cuda');p=torch.as_tensor(meta['pred'][j],device='cuda');pr,c,_=red(x,p);out.append(pr.cpu().numpy());cc.append(c.cpu().numpy())
    h.remove();del red;torch.cuda.empty_cache()
    return dict(dia=meta['dia'][ix],y=meta['y'][ix],pred_base=np.concatenate(out),num=meta['x'][ix],expertos=meta['pred'][ix],c_num=np.concatenate(cc),logits=np.concatenate(ll),c_real=meta['c'][ix])

def preparar_adicional(datos,s,car,progreso):
    destino=car/f'base_semilla_{s}.npz'
    if destino.exists():
        with np.load(destino) as z:return {k:z[k] for k in z.files}
    especificaciones=leer(REPO/'continuidad/DISENO_CUATRO_MODELOS_GPU.json')['expertos_configuraciones_seleccionadas']
    todo=[]
    periodos=list(pd.date_range(f'{datos.inicio_oof}-01-01',f'{datos.test_anio-1}-10-01',freq='QS',tz='UTC'))
    for inicio in periodos:
        final=pd.Timestamp(f'{datos.test_anio+1}-01-01',tz='UTC') if inicio.year==datos.test_anio else inicio+pd.DateOffset(months=3)
        corte=inicio-pd.DateOffset(months=2)
        tr=datos.pos[datos.indice[datos.pos]<corte-pd.Timedelta(days=1)]
        es=datos.pos[(datos.indice[datos.pos]>=corte)&(datos.indice[datos.pos]<inicio)]
        ev=datos.pos[(datos.indice[datos.pos]>=inicio)&(datos.indice[datos.pos]<final)]
        pp=[];y=None;dias=None
        for nombre in ['patchtst','E','periodica']:
            d=car/f'expertos/{inicio.strftime("%Y-%m")}/{nombre}/semilla_{s}'
            entrenar_base(datos,nombre,especificaciones[nombre]['configuracion'],s,tr,es,ev,d,progreso)
            with np.load(d/'predicciones.npz') as z:
                if y is None:y=z['y'];dias=z['dia']
                else:assert np.array_equal(y,z['y']) and np.array_equal(dias,z['dia'])
                pp.append(z['pred'])
        todo.append(dict(dia=dias,y=y,pred=np.stack(pp,-1)))
    meta=meta_desde_pred(datos,todo);salidas=[];cfg=dict(id=8,ancho=64,dropout=.1,lambda_complementariedad=.1)
    fechas=pd.DatetimeIndex(meta['dia']).tz_localize('UTC')
    for inicio in periodos:
        fin=pd.Timestamp(f'{datos.test_anio+1}-01-01',tz='UTC') if inicio.year==datos.test_anio else inicio+pd.DateOffset(months=3)
        ev=np.flatnonzero((fechas>=inicio)&(fechas<fin));corte=inicio-pd.DateOffset(months=3)
        a=np.flatnonzero(fechas<corte-pd.Timedelta(days=1));es=np.flatnonzero((fechas>=corte)&(fechas<inicio))
        if len(a)<30 or len(es)<30:
            # Prefijo inicial sin metaentrenamiento suficiente: media fija,
            # independiente de sus objetivos y común a todos los controles.
            r=dict(dia=meta['dia'][ev],y=meta['y'][ev],pred_base=meta['pred'][ev].mean(-1),num=meta['x'][ev],expertos=meta['pred'][ev])
        else:
            d=car/f'router/{inicio.strftime("%Y-%m")}/semilla_{s}'
            entrenar_router(meta,cfg,s,a,es,ev,d,progreso,datos.q168);r=inferir_meta(meta,ev,cargar(d/'modelo.pt'))
        salidas.append({k:r[k] for k in ['dia','y','pred_base','num','expertos']})
    out={k:np.concatenate([r[k] for r in salidas]) for k in salidas[0]};out['anio']=pd.DatetimeIndex(out['dia']).year.to_numpy()
    car.mkdir(exist_ok=True,parents=True);np.savez_compressed(destino,**out);return out

def final_adicional(datos,s,car,progreso):
    base=preparar_adicional(datos,s,car,progreso)
    # Reconstruir meta con pronósticos OOF para entrenar M1 final antes de prueba.
    meta=meta_desde_pred(datos,[dict(dia=base['dia'],y=base['y'],pred=base['expertos'])])
    inicio=pd.Timestamp(f'{datos.test_anio}-01-01',tz='UTC');corte=inicio-pd.DateOffset(months=3)
    fechas=pd.DatetimeIndex(meta['dia']).tz_localize('UTC')
    a=np.flatnonzero(fechas<corte-pd.Timedelta(days=1));es=np.flatnonzero((fechas>=corte)&(fechas<inicio))
    entrenar_router(meta,dict(id=8,ancho=64,dropout=.1,lambda_complementariedad=.1),s,a,es,np.array([],dtype='int64'),car/f'router_final/semilla_{s}',progreso,datos.q168)
    corte_base=inicio-pd.DateOffset(months=2)
    tr=datos.pos[datos.indice[datos.pos]<corte_base-pd.Timedelta(days=1)]
    st=datos.pos[(datos.indice[datos.pos]>=corte_base)&(datos.indice[datos.pos]<inicio)]
    esp=leer(REPO/'continuidad/DISENO_CUATRO_MODELOS_GPU.json')['expertos_configuraciones_seleccionadas']
    for nombre in ['patchtst','E','periodica']:
        entrenar_base(datos,nombre,esp[nombre]['configuracion'],s,tr,st,np.array([],dtype='int64'),car/f'expertos_finales/{nombre}/semilla_{s}',progreso)
    return base

def prueba_adicional(datos,s,car,base):
    pos=datos.pos[datos.indice[datos.pos].year>=datos.test_anio];pp=[]
    for nombre in ['patchtst','E','periodica']:
        ck=cargar(car/f'expertos_finales/{nombre}/semilla_{s}/modelo.pt');c=datos.construir(pos,ck['escalas']);tt=motor.a_tensores(c,'cuda')
        red=construir_experto(datos,nombre,ck['configuracion']).cuda().eval();red.load_state_dict(ck['modelo'])
        pp.append((motor.predecir(red,tt,128)*c.nivel[:,None]).astype('float32'));del red,tt
    parte=dict(dia=datos.dias_pos(pos),y=c.y.astype('float32'),pred=np.stack(pp,-1))
    meta=meta_desde_pred(datos,[dict(dia=base['dia'],y=base['y'],pred=base['expertos']),parte]);ix=np.arange(len(base['dia']),len(meta['dia']))
    pr=inferir_meta(meta,ix,cargar(car/f'router_final/semilla_{s}/modelo.pt'))
    out={k:np.concatenate([base[k],pr[k]]) for k in ['dia','y','pred_base','num','expertos']};out['anio']=pd.DatetimeIndex(out['dia']).year.to_numpy()
    return out

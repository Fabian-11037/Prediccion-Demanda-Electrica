"""Controles de capacidad y ablaciones numéricas; entrenamiento exclusivamente GPU."""
from types import SimpleNamespace
from e3_modelo4 import *
from m4_onnx_congelado import Encoder, Corrector

CFG=dict(id=6,k_vecinos=16,edad_maxima_dias=365,ancho_compuerta=64)
VARIANTES=['M4_N','MLP_CE','EWMA_MLP_CE','M4_N_fija','M4_N_sin_compuerta','M4_N_uniforme','M4_N_similitud']

class Densa(nn.Module):
    def __init__(self,n,h):
        super().__init__();self.red=nn.Sequential(nn.Linear(n,h),nn.GELU(),nn.Dropout(.1),nn.Linear(h,24))
        nn.init.zeros_(self.red[-1].weight);nn.init.zeros_(self.red[-1].bias)
    def forward(self,x):return self.red(x)

def ewma(meta,q):
    dias=np.asarray(meta['dia'],dtype='datetime64[D]').astype('int64')
    r=(meta['y']-meta['pred_base'])/q;out=np.zeros_like(r,dtype='float32');hay=np.zeros(len(dias),bool)
    for i,d in enumerate(dias):
        j=np.flatnonzero((d-dias>=2)&(d-dias<=365))
        if len(j):
            w=np.exp2(-(d-dias[j]-2)/30);out[i]=(w[:,None]*r[j]).sum(0)/w.sum();hay[i]=True
    return out,hay

def ancho(n,p):return min(range(1,4097),key=lambda h:(abs(h*(n+25)+24-p),h))

def preparar_banco(meta,mu,sd,q,limite=None):
    return SimpleNamespace(num=torch.as_tensor((meta['num']-mu)/sd,device='cuda'),
        base=torch.as_tensor(meta['pred_base'],device='cuda'),
        residuos=torch.as_tensor((meta['y']-meta['pred_base'])/q,device='cuda'),
        dias=torch.as_tensor(np.asarray(meta['dia'],dtype='datetime64[D]').astype('int64'),device='cuda'),limite=limite)

def ajustar_control(meta,s,a,es,ev,v,car,q,progreso,checkpoint_historico=None):
    car=Path(car)
    if (car/'TERMINADO.json').exists():return leer(car/'TERMINADO.json')
    motor.fijar_semilla(s);mu=meta['num'][a].mean(0);sd=meta['num'][a].std(0);sd=np.where(sd>1e-6,sd,1.)
    n=meta['num'].shape[1];objetivo=64*n+2625;resumen,hay=ewma(meta,q)
    limite=int(np.datetime64(meta['dia'][es[-1]],'D').astype('int64')) if v=='M4_N_fija' else None
    b=preparar_banco(meta,mu,sd,q,limite);y=torch.as_tensor(meta['y'],device='cuda')
    if v in ['MLP_CE','EWMA_MLP_CE']:
        xx=b.num if v=='MLP_CE' else torch.cat([b.num,torch.as_tensor(resumen,device='cuda')],1)
        h=ancho(xx.shape[1],objetivo);red=Densa(xx.shape[1],h).cuda();activos=sum(p.numel() for p in red.parameters())
        assert abs(activos-objetivo)/objetivo<=.05
        hh=torch.as_tensor(hay,device='cuda')
        def salida(j):return torch.where(hh[j,None],b.base[j]+q*red(xx[j]),b.base[j])
        tipo='densa';extra=dict(ancho=h,entrada=xx.shape[1])
    else:
        vv='M4_sin_compuerta' if v=='M4_N_sin_compuerta' else 'M4_N'
        red=Memoria(n,7,0,CFG,'M4_N').cuda()
        red.q_dataset=q
        red.recuperar=recuperar_q.__get__(red,Memoria)
        if v=='M4_N_sin_compuerta':
            # Mantener el codificador numérico, sin activar el camino textual.
            red.recuperar=recuperar_sin_compuerta.__get__(red,Memoria)
        def salida(j):
            lim=int(j.max().item())+1;qv,kv=red.codificar(b,lim)
            if v in ['M4_N_uniforme','M4_N_similitud']:
                return no_parametrica(b,j,q,v)
            return red.recuperar(qv,kv,b.dias[:lim],b.residuos[:lim],j,b.base[j],b.limite)[0]
        activos=0 if v in ['M4_N_uniforme','M4_N_similitud'] else (64*n+192 if v=='M4_N_sin_compuerta' else objetivo);tipo='memoria';extra={}
    aa=torch.as_tensor(a,device='cuda');ee=torch.as_tensor(es,device='cuda')
    def loss(ix):return (salida(aa[ix])-y[aa[ix]]).abs().mean()/q
    def val():return sum(float((salida(j)-y[j]).abs().mean()/q)*len(j) for j in ee.split(128))/len(es)
    if v in ['M4_N_uniforme','M4_N_similitud']:
        info=dict(epocas=0,mejor_epoca=0,mejor_perdida=val(),segundos=0,historia=[])
    elif checkpoint_historico is not None:
        viejo=cargar(checkpoint_historico);assert np.array_equal(mu,viejo['media']) and np.array_equal(sd,viejo['escala'])
        red=Memoria(n,viejo['n_feat'],viejo['n_fuentes'],CFG,'M4_N').cuda();red.load_state_dict(viejo['modelo']);red.eval()
        previo=leer(Path(checkpoint_historico).parent/'TERMINADO.json')
        info={k:previo[k] for k in ['epocas','mejor_epoca','mejor_perdida','segundos']};info['historia']=[]
        shutil=__import__('shutil');shutil.copy2(Path(checkpoint_historico).parent/'historia.csv',car_mkdir(car)/'historia.csv')
    else:
        info=ajustar(red,len(a),loss,val,dict(lr=.001,weight_decay=.0001,lote=64),dict(max_epocas=100,min_epocas=10,paciencia=10,min_delta=.0001),s,car,progreso,f'{v} s{s}')
    ck=dict(tipo=tipo,variante=v,configuracion=CFG,n_num=n,n_feat=7,n_fuentes=0,media=mu,escala=sd,limite_memoria=limite,semilla=s,modelo=red.state_dict(),q=q,**extra)
    # Al reutilizar se conservan dimensiones del contenedor histórico completo.
    if checkpoint_historico is not None:ck.update(n_feat=viejo['n_feat'],n_fuentes=viejo['n_fuentes'])
    checkpoint(car/'modelo.pt',ck);guardar(car/'escaladores.json',dict(media=mu.tolist(),escala=sd.tolist(),ajuste=[meta['dia'][a[0]],meta['dia'][a[-1]]]))
    r=dict(variante=v,semilla=s,parametros_activos=activos,parametros_almacenados=sum(p.numel() for p in red.parameters()),**{k:z for k,z in info.items() if k!='historia'})
    if len(ev):
        red.eval()
        with torch.no_grad():pred=np.concatenate([salida(torch.as_tensor(j,device='cuda')).cpu().numpy() for j in np.array_split(ev,max(1,int(np.ceil(len(ev)/128))))])
        np.savez_compressed(car/'predicciones.npz',dia=meta['dia'][ev],y=meta['y'][ev],pred=pred);r.update(metricas(meta['y'][ev],pred,q))
    guardar(car/'TERMINADO.json',r);del red,b,y;torch.cuda.empty_cache();return r

def car_mkdir(car):car.mkdir(parents=True,exist_ok=True);return car

def recuperar_sin_compuerta(self,q,k,dias,residuos,j,base,limite=None):
    # Solo cambia el valor de la compuerta, no el canal de codificación.
    anterior=self.variante;self.variante='M4_sin_compuerta'
    try:return recuperar_q(self,q,k,dias,residuos,j,base,limite)
    finally:self.variante=anterior

def recuperar_q(self,q,k,dias,residuos,j,base,limite=None):
    edad=dias[j,None]-dias[None,:];valido=(edad>=2)&(edad<=365)
    if limite is not None:valido &= dias[None,:]<=limite
    scores=(q[j]@k.T)/(32**.5);scores=scores.masked_fill(~valido,-1e9)
    valor,ids=scores.topk(min(16,len(k)),-1);ok=valido.gather(1,ids);hay=ok.any(-1)
    alfa=valor.softmax(-1)*ok;alfa=alfa/alfa.sum(-1,keepdim=True).clamp_min(1e-8)
    valores=torch.cat([residuos,residuos.new_zeros((1,24))],0)
    rr=valores[torch.where(ok,ids,torch.full_like(ids,len(residuos)))];media=(alfa[:,:,None]*rr).sum(1)
    dispersion=((rr-media[:,None]).square()*alfa[:,:,None]).sum(1).mean(-1).clamp_min(1e-12).sqrt()
    apoyo=ok.sum(-1).to(q.dtype)/16;maxsim=torch.where(hay,valor[:,0],torch.zeros_like(valor[:,0]))
    antig=(alfa*edad.gather(1,ids).to(q.dtype)).sum(-1)/365
    feat=torch.cat([q[j],maxsim[:,None],dispersion[:,None],apoyo[:,None],antig[:,None]],1)
    g=torch.sigmoid(self.compuerta(feat))[:,0]
    if self.variante=='M4_sin_compuerta':g=torch.ones_like(g)
    g=g*hay;pred=base+getattr(self,'q_dataset',215617.90361589016)*g[:,None]*media
    return torch.where(hay[:,None],pred,base),g,ids,alfa,ok

def no_parametrica(b,j,q,v):
    edad=b.dias[j,None]-b.dias[None,:];val=(edad>=2)&(edad<=365)
    # Uniforme utiliza los 16 días admisibles más recientes; similitud, contexto
    # estandarizado y distancia L2 con normalización por número de rasgos.
    score=-edad.float() if v=='M4_N_uniforme' else -(b.num[j,None]-b.num[None,:]).square().mean(-1)
    score=score.masked_fill(~val,-1e9);sc,ids=score.topk(min(16,len(b.num)),-1);ok=val.gather(1,ids)
    w=ok.float() if v=='M4_N_uniforme' else sc.softmax(-1)*ok
    w=w/w.sum(-1,keepdim=True).clamp_min(1e-8)
    rr=torch.where(ok[:,:,None],b.residuos[ids],torch.zeros_like(b.residuos[ids]))
    return b.base[j]+q*(w[:,:,None]*rr).sum(1)

def predecir(ck,meta,ix):
    q=ck['q'];resumen,hay=ewma(meta,q);b=preparar_banco(meta,ck['media'],ck['escala'],q,ck['limite_memoria'])
    v=ck['variante'];pred=[]
    if ck['tipo']=='densa':
        red=Densa(ck['entrada'],ck['ancho']).cuda().eval();red.load_state_dict(ck['modelo'])
        xx=b.num if v=='MLP_CE' else torch.cat([b.num,torch.as_tensor(resumen,device='cuda')],1);hh=torch.as_tensor(hay,device='cuda')
        with torch.no_grad():
            for jj in np.array_split(ix,max(1,int(np.ceil(len(ix)/128)))):
                j=torch.as_tensor(jj,device='cuda');pred.append(torch.where(hh[j,None],b.base[j]+q*red(xx[j]),b.base[j]).cpu().numpy())
    else:
        red=Memoria(ck['n_num'],ck['n_feat'],ck['n_fuentes'],CFG,'M4_N').cuda().eval();red.load_state_dict(ck['modelo'])
        red.q_dataset=q;red.recuperar=recuperar_q.__get__(red,Memoria)
        with torch.no_grad():
            qv,kv=red.codificar(b,len(b.num))
            for jj in np.array_split(ix,max(1,int(np.ceil(len(ix)/128)))):
                j=torch.as_tensor(jj,device='cuda')
                if v in ['M4_N_uniforme','M4_N_similitud']:p=no_parametrica(b,j,q,v)
                elif v=='M4_N_sin_compuerta':p=recuperar_sin_compuerta(red,qv,kv,b.dias,b.residuos,j,b.base[j],b.limite)[0]
                else:p=red.recuperar(qv,kv,b.dias,b.residuos,j,b.base[j],b.limite)[0]
                pred.append(p.cpu().numpy())
    pred=np.concatenate(pred);assert np.isfinite(pred).all();del red,b;torch.cuda.empty_cache();return pred

"""Adaptaciones univariantes explícitas de RATL y FreshCast a la misma M1.

No son réplicas oficiales: reloj diario, retraso2d, banco365d, claves de
estadísticos12 y entrenamiento por dataset. Detalles registrados en ADAPTACIONES.
"""
from e3_gpu_comun import *

NOMBRES=['RATL_adaptado','FreshCast_adaptado']

def ventanas(datos,dias):
    # Resolver por reloj civil, sin reinterpretar UTC como hora colombiana.
    indice=datos.indice.tz_localize(None);pos=indice.get_indexer(pd.DatetimeIndex(dias));assert (pos>=168).all()
    return datos.X[pos[:,None]+np.arange(-168,0),0].astype('float32')

def preparar(meta,w,a,q,nombre):
    mu=float(w[a].mean());sd=max(float(w[a].std()),1e-6);ww=(w-mu)/sd
    key=np.column_stack([ww[:,-1],ww.mean(1),ww.std(1),ww[:,-1]-ww[:,0],ww.reshape(len(w),8,21).mean(-1)]).astype('float32')
    fechas=np.asarray(meta['dia'],dtype='datetime64[D]').astype('int64');r=(meta['y']-meta['pred_base'])/q
    if nombre=='RATL_adaptado':
        tokens=[];rrs=[];oks=[];lim=fechas[a[-1]]
        for j,d in enumerate(fechas):
            edad=d-fechas;valid=(edad>=2)&(edad<=365)&(fechas<=lim);ix=np.flatnonzero(valid)
            score=-np.mean((key[ix]-key[j])**2,1);orden=np.argsort(-score,kind='stable')[:64];ix=ix[orden];score=score[orden]
            rr=np.zeros((65,24),dtype='float32');ok=np.zeros(65,bool);ok[0]=True
            rr[1:len(ix)+1]=r[ix];ok[1:len(ix)+1]=True
            alfa=np.exp(score-score.max()) if len(ix) else np.zeros(0);alfa=alfa/max(alfa.sum(),1e-8)
            directo=(alfa[:,None]*r[ix]).sum(0) if len(ix) else np.zeros(24)
            rb=rr.reshape(65,3,8).transpose(1,0,2)
            tok=np.concatenate([np.broadcast_to(key[j],(3,65,12)),rb,np.broadcast_to(directo.reshape(3,1,8),(3,65,8))],-1)
            tokens.append(tok);rrs.append(rb);oks.append(ok)
        salida=dict(x=np.asarray(tokens,dtype='float32'),rr=np.asarray(rrs),ok=np.asarray(oks),mu=mu,sd=sd)
    else:
        # Misma información observable que M4: episodios diarios d−2,365d.
        xx=ww[:,-96:];med=xx.mean(1);sigma=np.maximum(xx.std(1),.01);shape=(xx-med[:,None])/sigma[:,None]
        norm=np.maximum(np.linalg.norm(shape,axis=1),1e-8);shape=shape/norm[:,None]
        yy=(meta['y']-mu)/sd;features=[];valores=[];mascaras=[]
        inicial=ww[a].reshape(-1);ac=[]
        for lag in [24,168]:ac.append(float(np.corrcoef(inicial[lag:],inicial[:-lag])[0,1]))
        constantes=np.nan_to_num(np.asarray([*ac,np.diff(inicial).var()],dtype='float32'))
        for j,d in enumerate(fechas):
            edad=d-fechas;valid=(edad>=2)&(edad<=365);ix=np.flatnonzero(valid)
            sims=shape[ix]@shape[j];rec=ix[np.argsort(-sims,kind='stable')[:200]]
            seasonal=ix[(edad[ix]<=112)&(((edad[ix]<=8))|((edad[ix]%7==0)&(edad[ix]<=56)))];ids=np.r_[seasonal,rec][:216]
            tipo=np.r_[np.zeros(len(seasonal)),np.ones(len(rec))][:216]
            f=np.zeros((216,11),dtype='float32');val=np.zeros((216,24),dtype='float32');ok=np.zeros(216,bool);k=len(ids)
            if k:
                ages=edad[ids];dist=np.minimum(ages%7,7-ages%7)/3.5
                f[:k]=np.column_stack([1-tipo,tipo,shape[ids]@shape[j],ages/365,np.zeros(k),dist,np.abs(med[ids]-med[j])/sigma[j],np.log(sigma[ids]/sigma[j]),np.broadcast_to(constantes,(k,3))])
                rr=med[j]+sigma[j]/sigma[ids,None]*(yy[ids]-med[ids,None])
                val[:k]=np.where(tipo[:,None]>.5,rr,yy[ids]);ok[:k]=True
            features.append(f);valores.append(val);mascaras.append(ok)
        salida=dict(x=np.asarray(features),val=np.asarray(valores),ok=np.asarray(mascaras),mu=mu,sd=sd)
    return salida

class RATL(nn.Module):
    def __init__(self):
        super().__init__();self.entrada=nn.Linear(28,64);self.pos=nn.Embedding(3,64)
        self.set=nn.TransformerEncoderLayer(64,4,128,.1,batch_first=True,activation='gelu');self.score=nn.Linear(64,1);self.escala=nn.Parameter(torch.ones(()))
    def forward(self,x,rr,ok):
        perm=None
        if self.training:
            perm=torch.randperm(65,device=x.device);x=x[:,:,perm];rr=rr[:,:,perm];ok=ok[:,perm]
        n=len(x);z=self.entrada(x)+self.pos(torch.arange(3,device=x.device))[None,:,None,:]
        z=z.reshape(n*3,65,64);mask=~ok[:,None,:].expand(n,3,65).reshape(n*3,65)
        z=self.set(z,src_key_padding_mask=mask);logits=self.score(z).reshape(n,3,65).masked_fill(mask.reshape(n,3,65),-1e9)
        alpha=logits.softmax(-1);corr=self.escala*(alpha[:,:,:,None]*rr).sum(2).reshape(n,24)
        if perm is not None:alpha=alpha[:,:,torch.argsort(perm)]
        return corr,alpha

class FreshCast(nn.Module):
    def __init__(self):
        super().__init__();self.lineal=nn.Linear(11,1);self.mlp=nn.Sequential(nn.Linear(11,64),nn.GELU(),nn.Linear(64,64),nn.GELU(),nn.Linear(64,1));self.beta=nn.Parameter(torch.zeros(()))
    def forward(self,x,val,ok):
        score=(self.lineal(x)+self.mlp(x)).squeeze(-1)*nn.functional.softplus(self.beta)
        score=score.masked_fill(~ok,-1e9);w=score.softmax(-1)*ok;w=w/w.sum(-1,keepdim=True).clamp_min(1e-8)
        return (w[:,:,None]*val).sum(1),w

def ajustar_recuperacion(meta,w,a,es,s,nombre,car,q,progreso):
    if (car/'TERMINADO.json').exists():return
    motor.fijar_semilla(s);p=preparar(meta,w,a,q,nombre);tt={k:torch.as_tensor(v,device='cuda') for k,v in p.items() if isinstance(v,np.ndarray)}
    base=torch.as_tensor(meta['pred_base'],device='cuda');y=torch.as_tensor(meta['y'],device='cuda');aa=torch.as_tensor(a,device='cuda')
    red=(RATL() if nombre=='RATL_adaptado' else FreshCast()).cuda()
    def salida(j):
        if nombre=='RATL_adaptado':c,alpha=red(tt['x'][j],tt['rr'][j],tt['ok'][j]);return base[j]+q*c,alpha,c
        pred,alpha=red(tt['x'][j],tt['val'][j],tt['ok'][j]);pred=pred*p['sd']+p['mu'];hay=tt['ok'][j].any(-1);pred=torch.where(hay[:,None],pred,base[j]);return pred,alpha,(pred-base[j])/q
    def loss(ix):
        j=aa[ix];pred,alpha,c=salida(j);error=((pred-y[j])/q).square().mean()
        if nombre=='RATL_adaptado':
            real=((y[j]-base[j])/q).reshape(-1,3,1,8);err=(tt['rr'][j]-real).square().mean(-1)
            teacher=(-err/.1).masked_fill(~tt['ok'][j,None,:],-1e9).softmax(-1).detach();error=error+.4*(-teacher*alpha.clamp_min(1e-8).log()).sum(-1).mean()
        else:error=error.clamp(max=20.)
        return error
    def val():
        return sum(float(((salida(j)[0]-y[j])/q).square().mean())*len(j) for j in torch.as_tensor(es,device='cuda').split(64))/len(es)
    info=ajustar(red,len(a),loss,val,dict(lr=.001,weight_decay=.0001,lote=64),dict(max_epocas=100,min_epocas=10,paciencia=10,min_delta=.0001),s,car,progreso,f'{nombre} s{s}')
    with torch.no_grad():pp=np.concatenate([salida(j)[0].cpu().numpy() for j in torch.as_tensor(es,device='cuda').split(64)])
    d=pp-meta['pred_base'][es];e=meta['y'][es]-meta['pred_base'][es]
    if nombre=='RATL_adaptado':
        gamma=min(np.arange(11)/10,key=lambda g:(float(np.mean((e-g*d)**2)),g))
    else:gamma=float(np.clip(np.sum(e.astype('float64')*d)/max(np.sum(d.astype('float64')**2),1e-12),0,1))
    checkpoint(car/'modelo.pt',dict(nombre=nombre,modelo=red.state_dict(),semilla=s,a=a,escala_y=[p['mu'],p['sd']],gamma=float(gamma)))
    guardar(car/'TERMINADO.json',dict(nombre=nombre,semilla=s,gamma=float(gamma),parametros=sum(p.numel() for p in red.parameters()),**{k:v for k,v in info.items() if k!='historia'}))
    del red,tt,base,y,aa;torch.cuda.empty_cache()

def predecir_recuperacion(meta,w,ix,ck,q):
    nombre=ck['nombre'];p=preparar(meta,w,ck['a'],q,nombre);assert np.allclose(p['mu'],ck['escala_y'][0]) and np.allclose(p['sd'],ck['escala_y'][1])
    tt={k:torch.as_tensor(v,device='cuda') for k,v in p.items() if isinstance(v,np.ndarray)};red=(RATL() if nombre=='RATL_adaptado' else FreshCast()).cuda().eval();red.load_state_dict(ck['modelo']);sal=[]
    with torch.no_grad():
        for j in np.array_split(ix,max(1,int(np.ceil(len(ix)/64)))):
            if nombre=='RATL_adaptado':corr,_=red(tt['x'][j],tt['rr'][j],tt['ok'][j]);pr=meta['pred_base'][j]+q*ck['gamma']*corr.cpu().numpy()
            else:
                pr,_=red(tt['x'][j],tt['val'][j],tt['ok'][j]);pr=pr.cpu().numpy()*p['sd']+p['mu'];hay=p['ok'][j].any(-1);pr=np.where(hay[:,None],meta['pred_base'][j]+ck['gamma']*(pr-meta['pred_base'][j]),meta['pred_base'][j])
            sal.append(pr)
    out=np.concatenate(sal);assert np.isfinite(out).all();del red,tt;torch.cuda.empty_cache();return out

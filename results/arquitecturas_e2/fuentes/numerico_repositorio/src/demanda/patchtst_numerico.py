"""PatchTST oficial con contrato numérico ampliado y adaptador afín explícito."""
import time
from dataclasses import dataclass
import numpy as np
import pandas as pd
import torch
from torch import nn
from .diseno_numerico import preparar_numericas_comunes
from .vendor.patchtst.PatchTST_backbone import PatchTST_backbone

@dataclass
class Ventanas:
    x: np.ndarray
    calendario: np.ndarray
    y: np.ndarray
    nivel: np.ndarray
    origen: pd.DatetimeIndex

class Preparador:
    """Selecciona orígenes una sola vez; jamás descarta días por ausencia exógena."""
    def __init__(self,panel,registro,contrato):
        self.panel=panel;self.contrato=contrato
        self.original,self.mascara=preparar_numericas_comunes(panel,registro,contrato)
        self.cols=contrato['numericas']['historicas'];self.cal=contrato['numericas']['calendario']
        self.raw=self.original[self.cols].to_numpy(dtype=np.float64)
        self.calendar=self.original[self.cal].to_numpy(dtype=np.float64)
        self.y=panel['y_demanda_real_kwh'].to_numpy(dtype=np.float64)
        self.L=contrato['ventana_h'];self.H=contrato['horizonte_h']
    def elegibles(self,pos):
        pos=np.asarray(pos,dtype=int);offset=np.arange(-self.L,0)
        demand=self.raw[pos[:,None]+offset,0];target=self.y[pos[:,None]+np.arange(self.H)]
        ok=np.isfinite(demand).all(1)&np.isfinite(target).all(1)&(demand.mean(1)>0)
        return pos[ok]
    def ajustar(self,pos):
        pos=self.elegibles(pos)
        assert len(pos)>0
        # Unión exacta de filas utilizadas en entradas de entrenamiento.
        use=np.zeros(len(self.raw),dtype=bool)
        for p in pos:use[p-self.L:p]=True
        z=self.raw[use];med=np.zeros(z.shape[1]);noestimable=[]
        for j in range(z.shape[1]):
            valid=z[np.isfinite(z[:,j]),j]
            if len(valid):med[j]=np.median(valid)
            else:noestimable.append(self.cols[j])
        fill=np.where(np.isfinite(z),z,med);mu=fill.mean(0);sd=fill.std(0);sd[sd<1e-6]=1.
        cm=self.calendar[use].mean(0);cs=self.calendar[use].std(0);cs[cs<1e-6]=1.
        if not np.isfinite(cm).all() or not np.isfinite(cs).all():raise ValueError('Calendario incompleto')
        demand=self.raw[pos[:,None]+np.arange(-self.L,0),0];ratios=demand/demand.mean(1)[:,None]
        sigma=max(float(ratios.std()),1e-6)
        self.stats=dict(mediana=med.tolist(),media=mu.tolist(),std=sd.tolist(),cal_media=cm.tolist(),cal_std=cs.tolist(),sigma_demanda=sigma,
            columnas=self.cols,columnas_calendario=self.cal,no_estimables=noestimable,inicio_ajuste=str(self.panel.index[np.where(use)[0][0]]),fin_ajuste=str(self.panel.index[np.where(use)[0][-1]]),
            filas_ajuste=int(use.sum()),politica='Medianas y escalas sólo en unión de filas históricas de entrenamiento; máscaras sin escalar; demanda normalizada por nivel y sigma de entrenamiento')
        self.pos_train=pos
        return self
    def transformar(self,pos):
        pos=self.elegibles(pos);ix=pos[:,None]+np.arange(-self.L,0);s=self.stats
        raw=self.raw[ix];missing=~np.isfinite(raw);filled=np.where(missing,np.asarray(s['mediana']),raw)
        scaled=(filled-np.asarray(s['media']))/np.asarray(s['std'])
        nivel=raw[:,:,0].mean(1)
        scaled[:,:,0]=(raw[:,:,0]/nivel[:,None]-1)/s['sigma_demanda']
        calp=(self.calendar[ix]-np.asarray(s['cal_media']))/np.asarray(s['cal_std'])
        cf=(self.calendar[pos[:,None]+np.arange(self.H)]-np.asarray(s['cal_media']))/np.asarray(s['cal_std'])
        x=np.concatenate([scaled,calp,missing.astype(np.float64)],2).astype(np.float32)
        y=self.y[pos[:,None]+np.arange(self.H)]
        assert np.isfinite(x).all() and np.isfinite(cf).all() and np.isfinite(y).all()
        return Ventanas(x,cf.astype(np.float32),y,nivel,self.panel.index[pos])

class PatchTSTNumerico(nn.Module):
    """Núcleo ETTh1 oficial; adaptación afín pequeña para pronóstico de demanda."""
    def __init__(self,p,sigma_demanda,L=168,H=24,numericas=30,calendario=23):
        super().__init__();n=numericas*2+calendario
        self.core=PatchTST_backbone(c_in=n,context_window=L,target_window=H,patch_len=16,stride=8,n_layers=3,d_model=16,n_heads=4,d_ff=128,
            dropout=p['dropout'],attn_dropout=0.,head_dropout=0.,fc_dropout=p['dropout'],padding_patch='end',revin=True,affine=False,subtract_last=False,
            individual=False,norm='BatchNorm',res_attention=True,act='gelu',pre_norm=False,pe='zeros',learn_pe=True)
        self.readout=nn.Linear(n+calendario,1)
        nn.init.normal_(self.readout.weight,mean=0.,std=.01);nn.init.zeros_(self.readout.bias)
        with torch.no_grad():self.readout.weight[0,0]=sigma_demanda
    def forward(self,x,cal):
        features=self.core(x.transpose(1,2)).transpose(1,2)
        return 1.+self.readout(torch.cat([features,cal],2)).squeeze(-1)

def entrenar(model,tr,es,p,callback=None,max_epocas=200,device='cuda',deadline=None):
    opt=torch.optim.AdamW(model.parameters(),lr=p['lr'],weight_decay=p['weight_decay'])
    sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,factor=.5,patience=2)
    lossf=nn.L1Loss();model.to(device)
    # CPU arrays y lotes a GPU evitan replicar todo el conjunto en VRAM.
    xt=torch.from_numpy(tr.x);ct=torch.from_numpy(tr.calendario);yt=torch.from_numpy((tr.y/tr.nivel[:,None]).astype(np.float32))
    xv=torch.from_numpy(es.x);cv=torch.from_numpy(es.calendario);yv=torch.from_numpy((es.y/es.nivel[:,None]).astype(np.float32))
    gen=torch.Generator().manual_seed(p['semilla']);best=reference=float('inf');best_state=None;stale=0;hist=[];bs=p['batch_size']
    for ep in range(1,max_epocas+1):
        model.train();total=0.
        for ix in torch.randperm(len(xt),generator=gen).split(bs):
            if deadline is not None and time.time()>=deadline:raise TimeoutError('Presupuesto de24horas agotado')
            opt.zero_grad(set_to_none=True)
            loss=lossf(model(xt[ix].to(device),ct[ix].to(device)),yt[ix].to(device))
            if not torch.isfinite(loss):raise FloatingPointError('Pérdida no finita')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();total+=float(loss.detach())*len(ix)
        model.eval();value=0.
        with torch.no_grad():
            for i in range(0,len(xv),bs):value+=float(lossf(model(xv[i:i+bs].to(device),cv[i:i+bs].to(device)),yv[i:i+bs].to(device)))*len(xv[i:i+bs])
        value/=len(xv);hist.append(dict(epoca=ep,train_loss=total/len(xt),val_loss=value,lr=opt.param_groups[0]['lr']))
        if value<best:best=value;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        if not np.isfinite(reference) or value<reference-max(1e-6,.002*abs(reference)):reference=value;stale=0
        else:stale+=1
        sched.step(value)
        if callback:callback(hist[-1])
        if ep>=10 and stale>=10:break
    model.load_state_dict(best_state);return hist

@torch.no_grad()
def predecir(model,c,batch_size=32,device='cuda'):
    model.eval();out=[]
    for i in range(0,len(c.x),batch_size):
        q=model(torch.from_numpy(c.x[i:i+batch_size]).to(device),torch.from_numpy(c.calendario[i:i+batch_size]).to(device)).cpu().numpy()
        out.append(q*c.nivel[i:i+batch_size,None])
    return np.concatenate(out)

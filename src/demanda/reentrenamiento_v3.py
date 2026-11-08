"""Preprocesamiento por fold y redes pequeñas para la reevaluación exploratoria v3."""
from dataclasses import asdict
import numpy as np
import pandas as pd
import torch
from torch import nn
from . import modelos_base as M

class DatosFold(M.Datos):
    def __init__(self, panel, registro, cfg, posiciones_train, secuencial=False):
        super().__init__(panel, registro, cfg)
        pos=np.asarray(posiciones_train)
        inicio=panel.index[pos.min()-cfg.ventana]
        fin=panel.index[pos.max()-1]
        filas=panel.loc[(panel.index>=inicio)&(panel.index<=fin)]
        cols=self.cols_x+self.cols_cal
        self.escala.media_=filas[cols].mean()
        self.escala.std_=filas[cols].std().replace(0,1.)
        # Estadísticas del objetivo exclusivamente en etiquetas de entrenamiento.
        target='y_demanda_real_kwh'
        etiquetas=panel.loc[(panel.index>=panel.index[pos.min()])&(panel.index<=panel.index[pos.max()]+pd.Timedelta(hours=cfg.horizonte-1)),target]
        self.escala.media_[target]=etiquetas.mean();self.escala.std_[target]=etiquetas.std()
        self.secuencial=False
        train=super().conjunto_pos(pos)
        self.sigma_demanda=max(float(train.x[:,:,0].std()),1e-6)
        self.secuencial=secuencial
        self.auditoria=dict(inicio_ajuste=str(inicio),fin_ajuste=str(fin),filas=len(filas),sigma_demanda=self.sigma_demanda,secuencial=secuencial,
            escala_mase='Fija 2000–2017, como denominador de referencia de la ejecución original; no participa en optimización')
    def conjunto_pos(self,pos,clave=None):
        c=super().conjunto_pos(pos,clave)
        if self.secuencial:
            c.x[:,:,0]=(c.x[:,:,0]-1.)/self.sigma_demanda
        return c
    def escaladores(self):
        return dict(media=self.escala.media_.to_dict(),std=self.escala.std_.to_dict(),columnas_x=self.cols_x,columnas_cal=self.cols_cal,
            escala_mase=self.escala_mase,datos=asdict(self.cfg),**self.auditoria)

class Secuencial(nn.Module):
    def __init__(self,nombre,p,tr):
        super().__init__();self.nombre=nombre
        f=tr.x.shape[2]+tr.cal_p.shape[2];L=tr.x.shape[1];H,k=tr.cal_f.shape[1:]
        if nombre=='lstm':
            w=int(p['unidades']);self.encoder=nn.LSTM(f,w,num_layers=1,batch_first=True);d=w
        else:
            w=int(p['filtros']);kernel=int(p['kernel'])
            self.encoder=nn.Sequential(nn.Conv1d(f,w,kernel,padding=kernel//2),nn.ReLU(),nn.Dropout1d(p['dropout']),nn.AvgPool1d(2),nn.Flatten());d=w*(L//2)
        self.drop=nn.Dropout(p['dropout']);self.salida=nn.Linear(d+H*k,H)
        # Predicción inicial próxima al nivel; aprende desviaciones horarias.
        nn.init.zeros_(self.salida.bias)
    def forward(self,x,c):
        if self.nombre=='lstm':_,(h,_)=self.encoder(x);z=h[-1]
        else:z=self.encoder(x.transpose(1,2))
        return 1.+self.salida(torch.cat([self.drop(z),c.flatten(1)],1))
    def preparar(self,x,c):pass
    def penalizacion(self):return 0.

class Entrenador(M.ModeloTorch):
    def ajustar(self,tr,va,datos,p,semilla,op):
        M.fijar_semillas(semilla,op);self.dev=dev=M.resolver_dispositivo(op.dispositivo)
        self.red=red=(Secuencial(self.nombre,p,tr) if self.nombre in ('lstm','cnn1d') else M.construir_red(self.nombre,p,tr)).to(dev)
        xt,ct,yt=self._tensores(tr,datos.objetivo_norm(tr),dev,True)
        xv,cv,yv=self._tensores(va,datos.objetivo_norm(va),dev,True)
        red.preparar(xt,ct)
        lossf={'mse':nn.MSELoss(),'mae':nn.L1Loss(),'huber':nn.HuberLoss(delta=.05)}[p['perdida']]
        opt=torch.optim.AdamW(red.parameters(),lr=p['lr'],weight_decay=p.get('weight_decay',0.))
        sched=(torch.optim.lr_scheduler.ReduceLROnPlateau(opt,factor=.5,patience=2) if p['scheduler']=='plateau' else
            torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=op.max_epocas) if p['scheduler']=='cosine' else None)
        best=reference=float('inf');state=None;stale=0;hist=[];gen=torch.Generator().manual_seed(semilla)
        bs=int(p['batch_size'])
        for ep in range(1,op.max_epocas+1):
            red.train();tot=0.
            for ix in torch.randperm(len(xt),generator=gen).split(bs):
                ix=ix.to(dev);opt.zero_grad(set_to_none=True)
                loss=lossf(red(xt[ix],ct[ix]),yt[ix]);objective=loss+red.penalizacion()
                if not torch.isfinite(objective):raise FloatingPointError('Pérdida no finita')
                objective.backward();nn.utils.clip_grad_norm_(red.parameters(),op.clip);opt.step();tot+=float(loss.detach())*len(ix)
            red.eval()
            with torch.no_grad():
                vl=sum(float(lossf(red(xv[i:i+512],cv[i:i+512]),yv[i:i+512]))*len(xv[i:i+512]) for i in range(0,len(xv),512))/len(xv)
            hist.append(dict(epoca=ep,train_loss=tot/len(xt),val_loss=vl,lr=opt.param_groups[0]['lr']))
            if vl<best:
                best=vl;state={k:v.detach().clone() for k,v in red.state_dict().items()}
            if not np.isfinite(reference) or vl<reference-max(1e-6,.002*abs(reference)):
                reference=vl;stale=0
            else:stale+=1
            if sched is not None:sched.step(vl) if p['scheduler']=='plateau' else sched.step()
            callback=getattr(self,'observador',None)
            if callback:callback(hist[-1])
            if ep>=10 and stale>=6:break
        red.load_state_dict(state)
        return dict(n_parametros=sum(v.numel() for v in red.parameters()),epocas=len(hist),mejor_epoca=int(np.argmin([v['val_loss'] for v in hist])+1),dispositivo=str(dev),historia=hist)

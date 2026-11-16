"""Ampliación acotada de capacidad sobre el preprocesamiento corregido v3."""
import torch
from torch import nn

class Secuencial(nn.Module):
    def __init__(self,nombre,p,tr):
        super().__init__();self.nombre=nombre
        f=tr.x.shape[2]+tr.cal_p.shape[2];L=tr.x.shape[1];H,k=tr.cal_f.shape[1:]
        if nombre=='lstm':
            w=int(p['unidades']);n=int(p['capas'])
            self.encoder=nn.LSTM(f,w,num_layers=n,batch_first=True,dropout=p['dropout'] if n>1 else 0.);d=w
        else:
            w=int(p['filtros']);kernel=int(p['kernel']);blocks=[]
            for dilation in p['dilataciones']:
                blocks.extend([nn.Conv1d(f,w,kernel,dilation=dilation,padding=dilation*(kernel-1)//2),nn.ReLU(),nn.Dropout1d(p['dropout'])]);f=w
            blocks.extend([nn.AvgPool1d(2),nn.Flatten()]);self.encoder=nn.Sequential(*blocks);d=w*(L//2)
        self.drop=nn.Dropout(p['dropout']);self.salida=nn.Linear(d+H*k,H);nn.init.zeros_(self.salida.bias)
    def forward(self,x,c):
        if self.nombre=='lstm':_,(h,_)=self.encoder(x);z=h[-1]
        else:z=self.encoder(x.transpose(1,2))
        return 1.+self.salida(torch.cat([self.drop(z),c.flatten(1)],1))
    def preparar(self,x,c):pass
    def penalizacion(self):return 0.

def configuraciones():
    comunes=dict(dropout=.1,weight_decay=1e-4,batch_size=256,scheduler='plateau',perdida='huber',paciencia=6)
    lstm=[];cnn=[]
    for n,w in [(1,48),(1,64),(2,16),(2,32)]:
        lstm.append(dict(id=f'lstm_v4_{len(lstm)+1:02d}',ventana=168,params=dict(comunes,capas=n,unidades=w,lr=.001)))
    for w,k,dil in [(16,3,[1,1]),(16,5,[1,1]),(32,3,[1,1]),(32,5,[1,1]),(16,3,[1,2]),(16,5,[1,2])]:
        cnn.append(dict(id=f'cnn1d_v4_{len(cnn)+1:02d}',ventana=168,params=dict(comunes,capas=2,filtros=w,kernel=k,dilataciones=dil,lr=.0003)))
    return dict(cnn1d=cnn,lstm=lstm)

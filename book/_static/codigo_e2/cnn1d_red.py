from dataclasses import asdict

import numpy as np

import pandas as pd

import torch

from torch import nn

from . import modelos_base as M

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

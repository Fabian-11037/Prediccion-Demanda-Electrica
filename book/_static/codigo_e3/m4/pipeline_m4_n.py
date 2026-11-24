"""Pipeline numérico de M4_N: configuración 6, entrenamiento y prueba temporal.

Las matrices inactivas mantienen la disposición de los checkpoints conservados.
El perfil de dimensiones inactivas se toma del checkpoint, sin elegir un modelo.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from motor_m4_n import ajustar, checkpoint, cargar, guardar
from entrenamiento import motor

CONFIGURACION = dict(id=6, k_vecinos=16, edad_maxima_dias=365, ancho_compuerta=64)

class Banco:
    def __init__(self, meta, media, escala, q, device):
        self.num=torch.as_tensor((meta['num']-media)/escala,device=device)
        self.base=torch.as_tensor(meta['pred_base'],device=device)
        self.residuos=torch.as_tensor((meta['y']-meta['pred_base'])/q,device=device)
        self.dias=torch.as_tensor(np.asarray(meta['dia'],dtype='datetime64[D]').astype('int64'),device=device)
        self.limite=None

class Memoria(nn.Module):

    def __init__(self, n, n_feat, n_fuentes, cfg, variante='M4_N'):
        super().__init__()
        self.cfg = cfg
        self.variante = variante
        d = 32
        self.consulta_texto = nn.Linear(n, d)
        self.proy = nn.ModuleList([nn.Linear(768, d) for _ in range(2)])
        self.rasgo = nn.ModuleList([nn.Linear(n_feat, d) for _ in range(2)])
        self.fuente = nn.ModuleList([nn.Embedding(n_fuentes + 1, d) for _ in range(2)])
        self.norma = nn.ModuleList([nn.LayerNorm(d) for _ in range(2)])
        self.proy_q = nn.Sequential(nn.Linear(n + 2 * d, d), nn.LayerNorm(d))
        self.proy_k = nn.Sequential(nn.Linear(n + 2 * d, d), nn.LayerNorm(d))
        w = cfg['ancho_compuerta']
        self.compuerta = nn.Sequential(nn.Linear(d + 4, w), nn.GELU(), nn.Dropout(0.1), nn.Linear(w, 1))
        nn.init.zeros_(self.compuerta[-1].weight)
        nn.init.constant_(self.compuerta[-1].bias, -2.0)

    def codificar(self, banco, limite):
        """Relevancia aprendida por fragmento y canal; sin reducción entre fechas."""
        partes = []
        for i in range(0, limite, 64):
            ix = slice(i, min(i + 64, limite))
            num = banco.num[ix]
            texto = num.new_zeros((len(num), 64))
            partes.append(torch.cat([num, texto], 1))
        x = torch.cat(partes, 0)
        return (self.proy_q(x), self.proy_k(x))

    def recuperar(self, q, k, dias, residuos, j, base, limite=None):
        """La máscara temporal se aplica ANTES de top-k y de consultar valores."""
        edad = dias[j, None] - dias[None, :]
        valido = (edad >= 2) & (edad <= self.cfg['edad_maxima_dias'])
        if limite is not None:
            valido &= dias[None, :] <= limite
        scores = q[j] @ k.T / 32 ** 0.5
        scores = scores.masked_fill(~valido, -1000000000.0)
        kk = min(self.cfg['k_vecinos'], k.shape[0])
        valor, ids = scores.topk(kk, -1)
        ok = valido.gather(1, ids)
        hay = ok.any(-1)
        alfa = valor.softmax(-1) * ok
        alfa = alfa / alfa.sum(-1, keepdim=True).clamp_min(1e-08)
        valores = torch.cat([residuos, residuos.new_zeros((1, 24))], 0)
        seguros = torch.where(ok, ids, torch.full_like(ids, len(residuos)))
        rr = valores[seguros]
        media = (alfa[:, :, None] * rr).sum(1)
        dispersion = ((rr - media[:, None]).square() * alfa[:, :, None]).sum(1).mean(-1).clamp_min(1e-12).sqrt()
        apoyo = ok.sum(-1).to(q.dtype) / self.cfg['k_vecinos']
        maxsim = torch.where(hay, valor[:, 0], torch.zeros_like(valor[:, 0]))
        edades = edad.gather(1, ids).to(q.dtype)
        antig = (alfa * edades).sum(-1) / self.cfg['edad_maxima_dias']
        features = torch.cat([q[j], maxsim[:, None], dispersion[:, None], apoyo[:, None], antig[:, None]], 1)
        g = torch.sigmoid(self.compuerta(features))[:, 0]
        g = g * hay
        pred = base + 215617.90361589016 * g[:, None] * media
        pred = torch.where(hay[:, None], pred, base)
        assert (~ok | (edad.gather(1, ids) >= 2)).all()
        return (pred, g, ids, alfa, ok)

    def forward(self, banco, j, representaciones=None):
        limite = int(j.max().item()) + 1
        q, k = self.codificar(banco, limite) if representaciones is None else representaciones
        n = len(k)
        return self.recuperar(q, k, banco.dias[:n], banco.residuos[:n], j, banco.base[j], banco.limite)

def construir(ck, device='cuda'):
    return Memoria(ck['n_num'],ck['n_feat'],ck['n_fuentes'],ck['configuracion'],'M4_N').to(device)

def ajustar_m4_n(meta, semilla, ajuste, parada, carpeta, perfil, progreso, q=215617.90361589016):
    """Ajuste del corrector con base OOF preparada y configuración conservada.

    perfil contiene n_feat y n_fuentes de la estructura serializada original.
    ajuste y parada son índices cronológicos disjuntos anteriores a 2022.
    """
    carpeta=Path(carpeta)
    assert not (carpeta/'modelo.pt').exists(), 'Utilizar una carpeta de salida nueva'
    assert np.max(meta['dia'][ajuste]) < np.min(meta['dia'][parada])
    assert np.max(meta['dia'][parada]) < '2022-01-01'
    motor.fijar_semilla(semilla)
    media=meta['num'][ajuste].mean(0)
    escala=meta['num'][ajuste].std(0)
    escala=np.where(escala>1e-6,escala,1.)
    ck=dict(n_num=meta['num'].shape[1],n_feat=perfil['n_feat'],n_fuentes=perfil['n_fuentes'],
        configuracion=CONFIGURACION,semilla=semilla,media=media,escala=escala,variante='M4_N')
    banco=Banco(meta,media,escala,q,'cuda');red=construir(ck)
    y=torch.as_tensor(meta['y'],device='cuda')
    indices=torch.as_tensor(ajuste,device='cuda')
    def perdida(ix):
        j=indices[ix]
        return (red(banco,j)[0]-y[j]).abs().mean()/q
    def evaluar():
        representaciones=red.codificar(banco,int(parada[-1])+1)
        total=0.
        for i in range(0,len(parada),128):
            j=torch.as_tensor(parada[i:i+128],device='cuda')
            pred=red(banco,j,representaciones)[0]
            total+=float((pred-y[j]).abs().mean()/q)*len(j)
        return total/len(parada)
    info=ajustar(red,len(ajuste),perdida,evaluar,dict(lr=.001,weight_decay=.0001,lote=64),
        dict(max_epocas=100,min_epocas=10,paciencia=10,min_delta=.0001),semilla,carpeta,progreso,'M4_N')
    ck['modelo']=red.state_dict();checkpoint(carpeta/'modelo.pt',ck)
    guardar(carpeta/'TERMINADO.json',{k:v for k,v in info.items() if k!='historia'})
    return ck

def evaluar_m4_n(meta, indices, checkpoint_path, device='cuda', q=215617.90361589016):
    """Pesos y escalas congelados; elegibilidad d−2…d−365 por consulta."""
    ck=cargar(checkpoint_path);red=construir(ck,device)
    red.load_state_dict(ck['modelo']);red.eval();red.requires_grad_(False)
    banco=Banco(meta,ck['media'],ck['escala'],q,device)
    predicciones=[]
    with torch.no_grad():
        rep=red.codificar(banco,len(meta['dia']))
        for i in range(0,len(indices),128):
            j=torch.as_tensor(indices[i:i+128],device=device)
            pred,gate,vecinos,atencion,elegibles=red(banco,j,rep)
            edades=banco.dias[j,None]-banco.dias[vecinos]
            assert ((~elegibles)|((edades>=2)&(edades<=365))).all()
            assert torch.equal(pred[~elegibles.any(-1)],banco.base[j][~elegibles.any(-1)])
            predicciones.append(pred.cpu().numpy())
    pred=np.concatenate(predicciones);err=pred-meta['y'][indices]
    return pred,dict(MASE=float(np.abs(err).mean()/q),MAE_kwh=float(np.abs(err).mean()),
                     RMSE_kwh=float(np.sqrt(np.square(err).mean())))

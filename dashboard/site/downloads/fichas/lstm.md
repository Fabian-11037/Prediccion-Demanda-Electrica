# 10.4. LSTM: memoria recurrente con compuertas

La notación y las operaciones compartidas definen los tensores, el nivel de demanda, la atención, las activaciones y la optimización.

## Arquitectura y fundamentos

Una LSTM recorre la historia hora por hora y conserva un estado de memoria. Sus compuertas deciden qué información retener, incorporar y exponer. Esta organización permite representar dependencias de varias horas sin asignar un coeficiente independiente a cada posición, aunque no garantiza utilizar toda la ventana.

La implementación seleccionada recibe 168 pasos con 24 variables por paso: demanda normalizada y 23 variables de calendario pasado. Tiene una capa recurrente de 48 unidades. El estado oculto final se concatena con los 552 atributos de calendario futuro y una capa lineal de 24 salidas aprende desviaciones respecto de uno. Hay 28.632 parámetros.

$$\begin{aligned}
f_s&=\sigma(W_f x_s+U_fh_{s-1}+b_f),&i_s&=\sigma(W_i x_s+U_ih_{s-1}+b_i),\\
\tilde c_s&=\tanh(W_c x_s+U_ch_{s-1}+b_c),&c_s&=f_s\odot c_{s-1}+i_s\odot\tilde c_s,\\
o_s&=\sigma(W_o x_s+U_oh_{s-1}+b_o),&h_s&=o_s\odot\tanh(c_s).
\end{aligned}$$

La memoria $c_s$ y la salida $h_s$ son distintas; $\odot$ representa producto componente a componente. El decodificador usado es directo: $\widehat{\mathbf y}=\ell_t[\mathbf1+W_o'[D(h_{168});\operatorname{vec}(F)]+b_o']$. No realimenta una predicción para obtener la siguiente. El dropout de 0,1 se aplica al estado antes de la salida; no hay dropout entre capas recurrentes porque solo existe una. El entrenamiento utiliza Huber, AdamW y parada interna.

### Esquema global

[![Arquitectura global de lstm](https://demanda-electrica-dashboard.onrender.com/architectures/lstm_global.png)](https://demanda-electrica-dashboard.onrender.com/architectures/lstm_global.png)

Una misma celda recurrente procesa las 168 horas. El estado oculto final de 48 componentes se concatena con el calendario del horizonte para generar las 24 salidas.

### Detalle de las operaciones

[![Detalle de la arquitectura de lstm](https://demanda-electrica-dashboard.onrender.com/architectures/lstm_detalle.png)](https://demanda-electrica-dashboard.onrender.com/architectures/lstm_detalle.png)

Las compuertas de olvido, entrada y salida regulan la memoria y el estado oculto. La demanda se expresa respecto del nivel y de la dispersión de ajuste; la lectura utiliza el estado final y el calendario conocido.

El código de exportación de los esquemas forma parte del preprocesamiento compartido.

### Código de la arquitectura

```python
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
```

## Pipeline de entrenamiento y prueba

<!-- entrenamiento-portable:inicio -->
### Comandos de entrenamiento y evaluación

Validación temporal de la configuración seleccionada:

```bash
python scripts/entrenar_arquitectura.py --modelo lstm --anio 2021 --semilla 1000 --dispositivo auto --salida salidas/lstm_validacion_1000
```

Comprobación funcional breve:

```bash
python scripts/entrenar_arquitectura.py --modelo lstm --smoke --dispositivo cpu --salida salidas/lstm_funcional
```

Evaluación final autorizada, con ajuste y parada anteriores a 2022:

```bash
python scripts/entrenar_arquitectura.py --modelo lstm --fase prueba --allow-test --semilla 1000 --dispositivo auto --salida salidas/lstm_prueba_1000
```

El código completo de preparación, optimización, guardado y restauración está en el pipeline ejecutable compartido. Cada salida debe ser una carpeta nueva.
<!-- entrenamiento-portable:fin -->

La configuración utilizada por esta familia es:

```json
{
  "id": "lstm_v4_01",
  "ventana": 168,
  "params": {
    "dropout": 0.1,
    "weight_decay": 0.0001,
    "batch_size": 256,
    "scheduler": "plateau",
    "perdida": "huber",
    "paciencia": 6,
    "capas": 1,
    "unidades": 48,
    "lr": 0.001
  }
}
```

El procedimiento separa ajuste, parada interna y evaluación externa, estima las transformaciones únicamente en ajuste y restaura el mínimo de parada. La función admite validación o prueba con autorización explícita y conserva las salidas en una carpeta nueva. Los bloques siguientes documentan el código; construir el libro no ejecuta entrenamientos.

### Escalado por pliegue y optimización

```python
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
            descripcion_escala_mase='Fija 2000–2017, como denominador de referencia de la ejecución original; no participa en optimización')
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
```


El siguiente bloque vincula las implementaciones mostradas y permite seleccionar CPU o GPU sin funciones exclusivas del sistema operativo. La inicialización del dispositivo se conserva en resolver_dispositivo.

```python
from pathlib import Path
import json,sys
import numpy as np
import pandas as pd

def ejecutar_configuracion(raiz, familia, semilla=1000, dispositivo='auto',
                          fase='validacion', anio=2021, permitir_prueba=False,
                          salida=None):
    raiz=Path(raiz).resolve()
    procedencia=json.loads((raiz/'results/jbook/modelos_actualizados.json').read_text())
    fuentes=raiz/'results/repositorio'/procedencia['commit']
    sys.path.insert(0,str(fuentes/'src'))
    from demanda import modelos_base as M, particion as P
    from demanda import reentrenamiento_v3 as motor
    from demanda.variantes_v4 import Secuencial as Recurrente
    from demanda.metricas import evaluar
    import torch
    registro=json.loads((fuentes/'results/consolidado/modelos.json').read_text())
    config=registro[familia]['configuracion']
    panel=pd.read_parquet(raiz/'data/processed/panel_horario.parquet')
    variables=json.loads((fuentes/'data/processed/feature_registry.json').read_text())
    cfg=M.ConfigDatos(ventana=config['ventana'])
    if fase=='prueba':
        if not permitir_prueba:raise PermissionError('La prueba requiere acceso explícito')
        tr,es=P.pretest(panel.index,ventana=cfg.ventana)
        ev=P.origenes(panel.index,'test',ventana=cfg.ventana,allow_test=True)
    else:
        fold=P.folds_forward_chaining(panel.index,[anio],ventana=cfg.ventana)[0]
        tr,es,ev=fold['tr'],fold['es'],fold['val']
    datos=motor.DatosFold(panel,variables,cfg,tr,secuencial=familia in ('lstm','cnn1d'))
    train,stop,evaluacion=[datos.conjunto_pos(p) for p in [tr,es,ev]]
    assert train.origen.max()+pd.Timedelta(hours=23)<stop.origen.min()
    assert stop.origen.max()+pd.Timedelta(hours=23)<evaluacion.origen.min()
    assert pd.Timestamp(datos.auditoria['fin_ajuste'])<stop.origen.min()
    fabrica=motor.Secuencial
    try:
        if familia=='lstm':motor.Secuencial=Recurrente
        modelo=motor.Entrenador(familia)
        info=modelo.ajustar(train,stop,datos,config['params'],semilla,
                           M.Opciones(max_epocas=60,dispositivo=dispositivo,hilos=2))
    finally:
        motor.Secuencial=fabrica
    pred=datos.invertir(modelo.predecir(evaluacion),evaluacion)
    metricas=evaluar(evaluacion.y,pred,datos.escala_mase)
    if salida is not None:
        destino=Path(salida)/familia/str(semilla)/str(anio if fase=='validacion' else fase)
        destino.mkdir(parents=True,exist_ok=False)
        escala=datos.escaladores()
        torch.save({'state_dict':{k:v.detach().cpu() for k,v in modelo.red.state_dict().items()},
                    'modelo':familia,'params':config['params'],'escaladores':escala},
                    destino/'modelo.pt')
        np.savez_compressed(destino/'predicciones.npz',y=evaluacion.y,yhat=pred,
                            origen=evaluacion.origen.as_unit('ns').asi8)
        for nombre,objeto in [('escaladores',escala),('historia',info['historia']),
                              ('metricas',metricas)]:
            (destino/(nombre+'.json')).write_text(json.dumps(objeto,ensure_ascii=False,
                                                indent=2,default=str),encoding='utf-8')
    return modelo,datos,evaluacion,pred,info,metricas

# Selección: comparar el MASE medio de confirmación, sin leer prueba.
# Piloto: repetir cada configuración seleccionada con semillas 100–104
# en los cuatro años externos. Final: semillas 1000–1004 con acceso explícito.
# El argumento salida conserva pesos, escaladores, curvas, métricas y predicciones.

```



Para esta página, `familia` toma el valor `lstm`. Las definiciones completas de datos y métricas se muestran en el preprocesamiento compartido; los cuerpos de las redes y del bucle de ajuste están visibles en esta página.

## Evaluación

Los resultados de esta familia se integran en la evaluación estadística conjunta, con métricas por semilla, horizonte y origen. La comparación distingue las entradas demanda/calendario de las entradas numéricas ampliadas y el aporte de texto oficial.

## Fuentes

La formulación y las dimensiones se verifican en las clases del código mostrado.. La descripción corresponde a la implementación del experimento; las adaptaciones se identifican en el texto.

# 10.7. N-HiTS: interpolación jerárquica residual

La notación y las operaciones compartidas definen los tensores, el nivel de demanda, la atención, las activaciones y la optimización.

## Arquitectura y fundamentos

N-HiTS construye el pronóstico por contribuciones de distinta resolución. Un bloque explica una parte de la historia y propone una parte de la salida; el siguiente trabaja con el residuo que queda. La interpolación permite que un bloque describa el horizonte con pocos nodos, mientras otro añade variaciones más finas.

La implementación tiene tres bloques oficiales de N-HiTS con MLP de ancho 512, agrupamientos temporales 2, 2 y 1 y factores de reducción de salida 4, 2 y 1. Para 24 horas se predicen respectivamente 6, 12 y 24 nodos. Una matriz fija realiza interpolación lineal equivalente a `interpolate(..., align_corners=False)`. Cada bloque usa la historia residual, 59 exógenas históricas —29 variables distintas de demanda y 30 máscaras— y las 23 variables de calendario conocidas.

$$
\theta_b=f_b(\operatorname{pool}_b(r_b),X_{exo},F),\quad
\widehat b_b=\theta_b^{back},\quad \widehat f_b=I_b\theta_b^{for},\qquad
r_{b+1}=r_b-\widehat b_b.
$$ (eq-arquitecturas-nhits-001)

El pronóstico normalizado empieza en el último valor de demanda de la ventana y suma $\sum_b\widehat f_b$. El *backcast* reconstruye historia; no es una etiqueta futura ni una segunda serie objetivo. El modelo devuelve $\widehat{\mathbf y}=\ell_t\widehat{\mathbf r}$. Su configuración seleccionada N14 contiene 17.489.442 parámetros. Este tamaño se explica por las entradas multivariadas aplanadas de los bloques y no implica, por sí mismo, mayor capacidad predictiva efectiva.

El modelo reutiliza el preparador y el optimizador numérico de PatchTST, pero la red es diferente y no incorpora su lectura afín. Se conserva el procesamiento multiescala, el esquema residual y la interpolación. Las fuentes, licencia y revisión de NeuralForecast acompañan al código.

### Esquema global

[![Arquitectura global de nhits](../_static/arquitecturas_presentacion/nhits_global.png)](../_static/arquitecturas_presentacion/nhits_global.png)

Los tres bloques explican progresivamente la historia y suman contribuciones al pronóstico. Las salidas interpoladas de 6, 12 y 24 nudos se añaden al último nivel observado.

### Detalle de las operaciones

[![Detalle de la arquitectura de nhits](../_static/arquitecturas_presentacion/nhits_detalle.png)](../_static/arquitecturas_presentacion/nhits_detalle.png)

Cada bloque agrupa historia y exógenas, calcula coeficientes con un MLP y separa backcast y pronóstico. El backcast se resta de la historia; la interpolación fija lleva los nudos a las 24 horas del horizonte.

El código de exportación de los esquemas forma parte del preprocesamiento compartido.

### Código de la arquitectura

```python
import torch

from torch import nn

from .patchtst_numerico import Preparador, entrenar, predecir

from .vendor.nhits.core import NHITSBlock, _IdentityBasis

REVISION = '94b6ce131838be38a1fb5157349f02bbc8c89ef6'

class BaseLinealDeterminista(nn.Module):
    """La interpolación es lineal en sus nodos: se materializa su matriz en CPU."""
    def __init__(self, L, H, nodos):
        super().__init__()
        self.L = L
        matriz = torch.nn.functional.interpolate(torch.eye(nodos)[:, None, :],
                    size=H, mode='linear', align_corners=False)[:, 0, :]
        self.register_buffer('interpolacion', matriz)

    def forward(self, theta):
        return theta[:, :self.L], (theta[:, self.L:] @ self.interpolacion).unsqueeze(-1)

class NHITSNumerico(nn.Module):
    """Tres bloques oficiales, con exógenas históricas y calendario conocido."""
    def __init__(self, p, sigma_demanda, L=168, H=24):
        super().__init__()
        self.sigma_demanda = float(sigma_demanda)
        self.blocks = nn.ModuleList([
            NHITSBlock(input_size=L, h=H, n_theta=L + H // factor,
                mlp_units=[[512, 512]] * 3,
                basis=BaseLinealDeterminista(L, H, H // factor),
                futr_input_size=23, hist_input_size=59, stat_input_size=0,
                n_pool_kernel_size=pool, pooling_mode='MaxPool1d',
                dropout_prob=p['dropout'], activation='ReLU')
            for pool, factor in zip([2, 2, 1], [4, 2, 1])])

    def forward(self, x, cal):
        # Reconstrucción exacta de demanda/nivel, no de etiquetas futuras.
        demanda = 1 + x[:, :, 0] * self.sigma_demanda
        historicas = torch.cat([x[:, :, 1:30], x[:, :, 53:83]], dim=2)
        calendario = torch.cat([x[:, :, 30:53], cal], dim=1)
        residual = demanda.flip(dims=(-1,))
        pronostico = demanda[:, -1:, None]
        for bloque in self.blocks:
            backcast, contribucion = bloque(residual, calendario, historicas, None)
            residual = residual - backcast
            pronostico = pronostico + contribucion
        return pronostico.squeeze(-1)
```

<!-- diseno-entrenamiento:inicio -->

## Diseño de entrenamiento y prueba


### Selección en validación

Se evaluaron **20 configuraciones** y 5 llegaron a la comparación de selección. La **configuración N14** obtuvo el menor MASE de validación (0,9304) sobre 2019–2021. La configuración fijada para prueba es **N14**, con 5 semillas.

#### Configuración conservada
| Parámetro | Valor |
|---|---|
| Tasa de aprendizaje | 0,00010336 |
| Lote de ajuste | 64 |
| Dropout | 0,195608 |
| Decaimiento de pesos | 1,44162e-06 |

#### Etapas del entrenamiento y prueba

| Etapa | Especificación |
|---|---|
| Filtro inicial | 20 configuraciones; tres semillas; validación 2018 |
| Confirmación y criterio | Cinco finalistas; tres semillas; años 2019–2021; la puntuación de selección excluye el filtro de 2018. Menor MASE medio de validación; cada año tiene el mismo peso. |
| Ajuste final | 2000–2021, excluida la reserva interna de parada |
| Parada interna final | Último 10 % de orígenes anteriores a 2022; purga de 24 horas |
| Prueba | 2022-01-01 a 2025-12-31; 1.461 orígenes diarios |
| Semillas finales | 1000, 1001, 1002, 1003, 1004 |
| Historia y horizonte | 168 y 24 horas, respectivamente |
| Pérdida y optimizador | MAE de demanda dividida por el nivel histórico; AdamW |
| Épocas y parada | Máximo 200; mínimo 10; paciencia efectiva 10; restauración del menor valor interno |
| Tasa de aprendizaje | ReduceLROnPlateau: factor 0,5; paciencia 2 |
| Lotes | Ajuste: 64; evaluación de prueba: hasta 64 orígenes por lote |
| Épocas ejecutadas | 15–21; épocas de los checkpoints: 11, 6, 5, 5, 10 |


#### Configuraciones evaluadas en desarrollo
El filtro y la selección corresponden a periodos y cantidades de repeticiones distintos. El guion identifica las configuraciones que no avanzaron a confirmación; no representa un resultado nulo.
| Configuración | Hiperparámetros | MASE del filtro | MASE de validación | Estado |
|---|---|---:|---:|---|
| **N14** | Tasa de aprendizaje: 0,00010336; Lote de ajuste: 64; Dropout: 0,195608; Decaimiento de pesos: 1,44162e-06 | 0,6269 | 0,9304 | **Seleccionada para prueba** |
| N04 | Tasa de aprendizaje: 0,000103774; Lote de ajuste: 64; Dropout: 0,127354; Decaimiento de pesos: 3,94617e-06 | 0,6421 | 0,9332 | Finalista |
| N15 | Tasa de aprendizaje: 0,000263174; Lote de ajuste: 32; Dropout: 0,167637; Decaimiento de pesos: 0,000269924 | 0,6394 | 0,9541 | Finalista |
| N06 | Tasa de aprendizaje: 0,000100643; Lote de ajuste: 32; Dropout: 0,0786891; Decaimiento de pesos: 0,000922368 | 0,6548 | 0,9570 | Finalista |
| N13 | Tasa de aprendizaje: 0,000421486; Lote de ajuste: 32; Dropout: 0,116508; Decaimiento de pesos: 6,80881e-06 | 0,6605 | 0,9625 | Finalista |
| N11 | Tasa de aprendizaje: 0,000248237; Lote de ajuste: 64; Dropout: 0,0884637; Decaimiento de pesos: 5,74229e-06 | 0,6690 | — | Filtro inicial |
| N07 | Tasa de aprendizaje: 0,000197065; Lote de ajuste: 64; Dropout: 0,259409; Decaimiento de pesos: 4,70571e-05 | 0,6780 | — | Filtro inicial |
| N16 | Tasa de aprendizaje: 0,000964873; Lote de ajuste: 64; Dropout: 0,171782; Decaimiento de pesos: 2,72872e-06 | 0,7099 | — | Filtro inicial |
| N18 | Tasa de aprendizaje: 0,000459035; Lote de ajuste: 32; Dropout: 0,249701; Decaimiento de pesos: 7,78391e-06 | 0,7179 | — | Filtro inicial |
| N08 | Tasa de aprendizaje: 0,000400238; Lote de ajuste: 32; Dropout: 0,231424; Decaimiento de pesos: 8,06178e-05 | 0,7240 | — | Filtro inicial |
| N12 | Tasa de aprendizaje: 0,000179581; Lote de ajuste: 32; Dropout: 0,291245; Decaimiento de pesos: 8,83417e-06 | 0,7245 | — | Filtro inicial |
| N10 | Tasa de aprendizaje: 0,000758298; Lote de ajuste: 16; Dropout: 0,17722; Decaimiento de pesos: 1,28959e-06 | 0,7634 | — | Filtro inicial |
| N20 | Tasa de aprendizaje: 0,000878968; Lote de ajuste: 32; Dropout: 0,202249; Decaimiento de pesos: 2,63526e-06 | 0,7653 | — | Filtro inicial |
| N01 | Tasa de aprendizaje: 0,001; Lote de ajuste: 32; Dropout: 0,1; Decaimiento de pesos: 0,0001 | 0,7676 | — | Filtro inicial |
| N05 | Tasa de aprendizaje: 0,00196812; Lote de ajuste: 64; Dropout: 0,164915; Decaimiento de pesos: 2,14238e-05 | 0,8659 | — | Filtro inicial |
| N02 | Tasa de aprendizaje: 0,00252957; Lote de ajuste: 64; Dropout: 0,14839; Decaimiento de pesos: 6,08571e-06 | 0,8736 | — | Filtro inicial |
| N09 | Tasa de aprendizaje: 0,00230437; Lote de ajuste: 64; Dropout: 0,14619; Decaimiento de pesos: 0,000452441 | 0,8963 | — | Filtro inicial |
| N03 | Tasa de aprendizaje: 0,00122413; Lote de ajuste: 16; Dropout: 0,284497; Decaimiento de pesos: 0,00477104 | 1,0270 | — | Filtro inicial |
| N17 | Tasa de aprendizaje: 0,00226148; Lote de ajuste: 16; Dropout: 0,231026; Decaimiento de pesos: 1,16432e-06 | 1,4560 | — | Filtro inicial |
| N19 | Tasa de aprendizaje: 0,00230194; Lote de ajuste: 16; Dropout: 0,245892; Decaimiento de pesos: 4,57105e-06 | 1,7964 | — | Filtro inicial |

[Resultados de desarrollo por año y semilla](../_static/resultados/diseno_entrenamiento/repo_nhits_numerico_r1__N.csv) · [Configuraciones completas](../_static/resultados/diseno_entrenamiento/repo_nhits_numerico_r1__N_configuraciones.csv).

<!-- diseno-entrenamiento:fin -->

## Pipeline de entrenamiento y prueba

<!-- entrenamiento-portable:inicio -->
### Comandos de entrenamiento y evaluación

Validación temporal de la configuración seleccionada:

```bash
python scripts/entrenar_arquitectura.py --modelo nhits --anio 2021 --semilla 1000 --dispositivo auto --salida salidas/nhits_validacion_1000
```

Comprobación funcional breve:

```bash
python scripts/entrenar_arquitectura.py --modelo nhits --smoke --dispositivo cpu --salida salidas/nhits_funcional
```

Evaluación final, con ajuste y parada anteriores a 2022:

```bash
python scripts/entrenar_arquitectura.py --modelo nhits --fase prueba --allow-test --semilla 1000 --dispositivo auto --salida salidas/nhits_prueba_1000
```

El código completo de preparación, optimización, guardado y restauración está en el pipeline ejecutable compartido. Cada salida debe ser una carpeta nueva.
<!-- entrenamiento-portable:fin -->

La preparación conserva 30 variables históricas, 23 de calendario y 30 máscaras. Se usan las mismas ventanas para ajuste, parada y prueba, con imputaciones y escalas estimadas solo en ajuste. El motor recibe el dispositivo explícitamente, guarda el mínimo de parada interna y convierte la salida normalizada a kWh. El código de la arquitectura incluye el motor para PatchTST; N-HiTS comparte ese motor sin compartir su red.

### Preparación, ajuste y predicción numérica

```python
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
```

### Bloque N-HiTS e interpolación

```python
"""Bloques sin modificar de Nixtla/neuralforecast; véase PROCEDENCIA.md y LICENSE."""
from typing import Tuple
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class _IdentityBasis(nn.Module):
    def __init__(
        self,
        backcast_size: int,
        forecast_size: int,
        interpolation_mode: str,
        out_features: int = 1,
    ):
        super().__init__()
        assert (interpolation_mode in ["linear", "nearest"]) or (
            "cubic" in interpolation_mode
        )
        self.forecast_size = forecast_size
        self.backcast_size = backcast_size
        self.interpolation_mode = interpolation_mode
        self.out_features = out_features

    def forward(self, theta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        backcast = theta[:, : self.backcast_size]
        knots = theta[:, self.backcast_size :]

        # Interpolation is performed on default dim=-1 := H
        knots = knots.reshape(len(knots), self.out_features, -1)
        if self.interpolation_mode in ["nearest", "linear"]:
            # knots = knots[:,None,:]
            forecast = F.interpolate(
                knots, size=self.forecast_size, mode=self.interpolation_mode
            )
            # forecast = forecast[:,0,:]
        elif "cubic" in self.interpolation_mode:
            if self.out_features > 1:
                raise Exception(
                    "Cubic interpolation not available with multiple outputs."
                )
            batch_size = len(backcast)
            knots = knots[:, None, :, :]
            forecast = torch.zeros(
                (len(knots), self.forecast_size), device=knots.device
            )
            n_batches = int(np.ceil(len(knots) / batch_size))
            for i in range(n_batches):
                forecast_i = F.interpolate(
                    knots[i * batch_size : (i + 1) * batch_size],
                    size=self.forecast_size,
                    mode="bicubic",
                )
                forecast[i * batch_size : (i + 1) * batch_size] += forecast_i[
                    :, 0, 0, :
                ]  # [B,None,H,H] -> [B,H]
            forecast = forecast[:, None, :]  # [B,H] -> [B,None,H]

        # [B,Q,H] -> [B,H,Q]
        forecast = forecast.permute(0, 2, 1)
        return backcast, forecast

ACTIVATIONS = ["ReLU", "Softplus", "Tanh", "SELU", "LeakyReLU", "PReLU", "Sigmoid"]

POOLING = ["MaxPool1d", "AvgPool1d"]

class NHITSBlock(nn.Module):
    """
    NHITS block which takes a basis function as an argument.
    """

    def __init__(
        self,
        input_size: int,
        h: int,
        n_theta: int,
        mlp_units: list,
        basis: nn.Module,
        futr_input_size: int,
        hist_input_size: int,
        stat_input_size: int,
        n_pool_kernel_size: int,
        pooling_mode: str,
        dropout_prob: float,
        activation: str,
    ):
        super().__init__()

        pooled_hist_size = int(np.ceil(input_size / n_pool_kernel_size))
        pooled_futr_size = int(np.ceil((input_size + h) / n_pool_kernel_size))

        input_size = (
            pooled_hist_size
            + hist_input_size * pooled_hist_size
            + futr_input_size * pooled_futr_size
            + stat_input_size
        )

        self.dropout_prob = dropout_prob
        self.futr_input_size = futr_input_size
        self.hist_input_size = hist_input_size
        self.stat_input_size = stat_input_size

        assert activation in ACTIVATIONS, f"{activation} is not in {ACTIVATIONS}"
        assert pooling_mode in POOLING, f"{pooling_mode} is not in {POOLING}"

        activ = getattr(nn, activation)()

        self.pooling_layer = getattr(nn, pooling_mode)(
            kernel_size=n_pool_kernel_size, stride=n_pool_kernel_size, ceil_mode=True
        )

        # Block MLPs
        hidden_layers = [
            nn.Linear(in_features=input_size, out_features=mlp_units[0][0])
        ]
        for layer in mlp_units:
            hidden_layers.append(nn.Linear(in_features=layer[0], out_features=layer[1]))
            hidden_layers.append(activ)

            if self.dropout_prob > 0:
                # raise NotImplementedError('dropout')
                hidden_layers.append(nn.Dropout(p=self.dropout_prob))

        output_layer = [nn.Linear(in_features=mlp_units[-1][1], out_features=n_theta)]
        layers = hidden_layers + output_layer
        self.layers = nn.Sequential(*layers)
        self.basis = basis

    def forward(
        self,
        insample_y: torch.Tensor,
        futr_exog: torch.Tensor,
        hist_exog: torch.Tensor,
        stat_exog: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # Pooling
        # Pool1d needs 3D input, (B,C,L), adding C dimension
        insample_y = insample_y.unsqueeze(1)
        insample_y = self.pooling_layer(insample_y)
        insample_y = insample_y.squeeze(1)

        # Flatten MLP inputs [B, L+H, C] -> [B, (L+H)*C]
        # Contatenate [ Y_t, | X_{t-L},..., X_{t} | F_{t-L},..., F_{t+H} | S ]
        batch_size = len(insample_y)
        if self.hist_input_size > 0:
            hist_exog = hist_exog.permute(0, 2, 1)  # [B, L, C] -> [B, C, L]
            hist_exog = self.pooling_layer(hist_exog)
            hist_exog = hist_exog.permute(0, 2, 1)  # [B, C, L] -> [B, L, C]
            insample_y = torch.cat(
                (insample_y, hist_exog.reshape(batch_size, -1)), dim=1
            )

        if self.futr_input_size > 0:
            futr_exog = futr_exog.permute(0, 2, 1)  # [B, L, C] -> [B, C, L]
            futr_exog = self.pooling_layer(futr_exog)
            futr_exog = futr_exog.permute(0, 2, 1)  # [B, C, L] -> [B, L, C]
            insample_y = torch.cat(
                (insample_y, futr_exog.reshape(batch_size, -1)), dim=1
            )

        if self.stat_input_size > 0:
            insample_y = torch.cat(
                (insample_y, stat_exog.reshape(batch_size, -1)), dim=1
            )

        # Compute local projection weights and projection
        theta = self.layers(insample_y)
        backcast, forecast = self.basis(theta)
        return backcast, forecast
```

### Selección, piloto, ajuste final y prueba

```python
"""Ruta 1: NHITS adaptado, núcleo fijo y búsqueda de hiperparámetros."""
import fcntl,hashlib,html,json,os,sys,time,traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from demanda.nhits_numerico import Preparador,NHITSNumerico,entrenar,predecir,ARQUITECTURA
from demanda import particion as P,modelos_base as M
from demanda.metricas import evaluar
from demanda.potencia_pareada import dimensionar
from demanda.seleccion_escalonada import clasificar
OUT=ROOT/'results/nhits_numerico_r1';OUT.mkdir(parents=True,exist_ok=True)
S={};DONE=0;TOTAL=130;DEADLINE=None

def save(p,obj):
    temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str));temp.replace(p)
def state(**kw):
    S.update(kw);S.update(completados=DONE,total=TOTAL,actualizado=pd.Timestamp.now(tz='America/Bogota').isoformat(),pid=os.getpid())
    save(OUT/'estado.json',S)
    txt=' · '.join(f'{k}: {v}' for k,v in S.items() if k in ('estado','etapa','configuracion','semilla','fold','epoca'))
    body=f'''<!doctype html><html lang="es"><meta charset="utf-8"><meta http-equiv="refresh" content="8"><title>NHITS numérico</title><style>body{{font:18px system-ui;background:#14202c;color:#eef5fa;max-width:950px;margin:50px auto;padding:20px}}progress{{width:100%;height:32px}}p{{line-height:1.7}}</style><h1>NHITS · numéricas ampliadas · GPU</h1><progress value="{DONE}" max="{TOTAL}"></progress><p>{DONE}/{TOTAL} ajustes previstos · {100*DONE/max(TOTAL,1):.1f}%</p><p>{html.escape(txt)}</p><p>Filtro2018:20configuraciones × 3semillas = 60ajustes.<br>Confirmación2019–2021:5finalistas × 3semillas × 3pliegues = 45ajustes.<br>Piloto: 5 semillas × 4 pliegues = 20 ajustes.<br>Prueba: R semillas dimensionadas; cinco exploratorias si no es dimensionable.</p><p>30 variables históricas + 23 de calendario + 30 máscaras. Historia168h · horizonte24h · MAE normalizada · máximo200épocas.<br>Tres bloques oficiales de ancho512; pooling2/2/1, interpolación4/2/1; sin cabezal adicional. Evaluación retrospectiva2022–2025.</p><p>{html.escape(S.get('detalle',''))}</p><small>{S['actualizado']}</small></html>'''
    temp=OUT/'progreso.tmp';temp.write_text(body);temp.replace(OUT/'progreso.html')

def configuraciones():
    rng=np.random.default_rng(20261114);cc=[dict(id='N01',lr=.001,batch_size=32,dropout=.1,weight_decay=.0001)]
    for i in range(1,20):cc.append(dict(id=f'N{i+1:02d}',lr=float(np.exp(rng.uniform(np.log(1e-4),np.log(3e-3)))),batch_size=int(rng.choice([16,32,64])),dropout=float(rng.uniform(.05,.3)),weight_decay=float(np.exp(rng.uniform(np.log(1e-6),np.log(1e-2))))))
    return cc
CONFIG=configuraciones();DATA={};PREPARER=None;PANEL=None;CONTRACT=None

def pack(fold):
    if fold in DATA:return DATA[fold]
    if fold=='final':
        assert (OUT/'seleccion_cerrada.json').exists() and (OUT/'plan_prueba.json').exists(), 'La prueba requiere selección y semillas congeladas'
        tr,es=P.pretest(PANEL.index);ev=P.origenes(PANEL.index,'test',allow_test=True)
    else:
        f=P.folds_forward_chaining(PANEL.index,[fold])[0];tr,es,ev=f['tr'],f['es'],f['val']
    prep=Preparador(PANEL,REG,CONTRACT).ajustar(tr)
    a,b,c=prep.transformar(tr),prep.transformar(es),prep.transformar(ev)
    assert pd.Timestamp(prep.stats['fin_ajuste'])<b.origen.min()
    assert a.origen.max()+pd.Timedelta(hours=23)<b.origen.min()
    assert b.origen.max()+pd.Timedelta(hours=23)<c.origen.min()
    DATA.clear();DATA[fold]=(prep,a,b,c)
    return DATA[fold]

def references(c):
    pos=PANEL.index.get_indexer(c.origen);h=np.arange(24)
    y=PANEL.y_demanda_real_kwh.to_numpy();weekly=y[pos[:,None]+h-168]
    daily=y[pos[:,None]+h-np.where(h-24<=-2,24,48)]
    return dict(ingenuo_168h=weekly,ingenuo_24h=daily)

def fit(stage,cfg,seed,fold):
    global DONE
    identity=dict(etapa=stage,configuracion=cfg,semilla=seed,fold=fold,modelo='NHITS_adaptado')
    key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:20]
    folder=OUT/'ajustes'/stage/key;folder.mkdir(parents=True,exist_ok=True)
    state(etapa=stage,configuracion=cfg['id'],semilla=seed,fold=fold,epoca=0)
    if (folder/'resultado.json').exists():
        result=json.loads((folder/'resultado.json').read_text());DONE+=1;state();return result
    if DEADLINE is not None and time.time()>=DEADLINE:raise TimeoutError('Presupuesto de24horas agotado')
    prep,tr,es,ev=pack(fold);save(folder/'identidad.json',identity);save(folder/'escaladores.json',prep.stats)
    M.fijar_semillas(seed,M.Opciones(dispositivo='cuda',hilos=2))
    p=dict(cfg,semilla=seed);torch.cuda.reset_peak_memory_stats();model=NHITSNumerico(p,prep.stats['sigma_demanda']);history=[]
    def cb(item):
        history.append(item);save(folder/'historia_en_curso.json',history);state(epoca=item['epoca'],loss_train=item['train_loss'],loss_es=item['val_loss'])
    started=time.time()
    try:hist=entrenar(model,tr,es,p,cb,deadline=DEADLINE)
    except TimeoutError:
        torch.save(dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},configuracion=cfg,semilla=seed,escaladores=prep.stats,parcial=True),folder/'modelo_parcial.pt')
        save(folder/'historia_parcial.json',history)
        raise
    q=predecir(model,ev,cfg['batch_size']);assert np.isfinite(q).all()
    qscale=CONTRACT['metrica']['q168_kwh'];metrics=evaluar(ev.y,q,qscale)
    result=dict(**identity,segundos=time.time()-started,n_parametros=sum(v.numel() for v in model.parameters()),memoria_gpu_max_bytes=torch.cuda.max_memory_allocated(),epocas=len(hist),mejor_epoca=int(np.argmin([v['val_loss'] for v in hist])+1),**metrics)
    torch.save(dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},configuracion=cfg,semilla=seed,escaladores=prep.stats,arquitectura=ARQUITECTURA,modelo='NHITS_adaptado'),folder/'modelo.pt')
    np.savez_compressed(folder/'predicciones.npz',y=ev.y,yhat=q,origen=ev.origen.as_unit('ns').asi8,semilla=seed,zona_origen='UTC')
    save(folder/'historia.json',hist);save(folder/'resultado.json',result);DONE+=1;state()
    print(stage,cfg['id'],seed,fold,round(metrics['MASE'],4),round(result['segundos'],1),flush=True)
    return result

def table(name,rows):
    pd.DataFrame([{**r,'configuracion':json.dumps(r['configuracion'],ensure_ascii=False)} for r in rows]).to_csv(OUT/name,index=False)

def main():
    global PANEL,REG,CONTRACT,TOTAL,DEADLINE
    lock=(OUT/'gpu.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (OUT/'ejecucion_completa.json').exists():return
    assert torch.cuda.is_available(),'Se requiere GPU'
    CONTRACT=json.loads((ROOT/'config/diseno_entrenamiento_texto.json').read_text())
    paths=['PLAN_NHITS_NUMERICO_R1.md','src/demanda/seleccion_escalonada.py','config/diseno_entrenamiento_texto.json','src/demanda/nhits_numerico.py','src/demanda/patchtst_numerico.py','src/demanda/potencia_pareada.py','scripts/entrenar_nhits_numerico.py','src/demanda/diseno_numerico.py','src/demanda/particion.py','src/demanda/modelos_base.py','src/demanda/metricas.py','data/processed/panel_horario.parquet','data/processed/feature_registry.json']
    paths += [str(p.relative_to(ROOT)) for folder in ['nhits','patchtst'] for p in (ROOT/'src/demanda/vendor'/folder).glob('*') if p.is_file()]
    frozen=dict(politica_busqueda=dict(filtro_anios=[2018],finalistas=5,confirmacion_anios=[2019,2020,2021],total_preprueba=125),politica_prueba=dict(semilla_inicial=1000,semillas_si_no_dimensionable=5,exploratorio=True),sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},arquitectura=ARQUITECTURA,configuraciones=CONFIG)
    if (OUT/'plan.json').exists():
        assert json.loads((OUT/'plan.json').read_text())==frozen,'Fuentes o configuración modificadas; no se reanuda con cambios'
    save(OUT/'plan.json',frozen);save(OUT/'hardware.json',dict(torch=torch.__version__,hip=torch.version.hip,nombre=torch.cuda.get_device_name(),memoria=torch.cuda.get_device_properties(0).total_memory))
    PANEL=pd.read_parquet(ROOT/'data/processed/panel_horario.parquet');REG=json.loads((ROOT/'data/processed/feature_registry.json').read_text())
    assert frozen['sha256']['data/processed/panel_horario.parquet']==CONTRACT['panel_sha256']
    budget=OUT/'limite_tiempo.json'
    if not budget.exists():save(budget,dict(inicio=time.time(),deadline=time.time()+24*3600,horas=24))
    DEADLINE=json.loads(budget.read_text())['deadline']
    if time.time()>=DEADLINE:raise TimeoutError('Presupuesto de24horas agotado; se requiere autorización para renovar el plazo')
    state(estado='Entrenando');rows=[];scores=[]
    # Conservar íntegros los60ajustes del filtro2018.
    for cfg in CONFIG:
        for seed in (43,44,45):rows.append(fit('busqueda',cfg,seed,2018));table('busqueda.csv',rows)
    ranking=clasificar(rows,CONFIG,[2018],[43,44,45]);finalistas=[cfg for _,cfg in ranking[:5]]
    save(OUT/'seleccion_2018.json',dict(criterio='Media de las3semillas en2018;empates por ID',ranking=[dict(MASE=value,configuracion=cfg) for value,cfg in ranking],finalistas=finalistas))
    for year in (2019,2020,2021):
        for cfg in finalistas:
            for seed in (43,44,45):rows.append(fit('busqueda',cfg,seed,year));table('busqueda.csv',rows)
    score,best=clasificar(rows,finalistas,[2019,2020,2021],[43,44,45])[0]
    save(OUT/'mejor_config.json',dict(configuracion=best,MASE_cv=score,anios_confirmacion=[2019,2020,2021],filtro_2018=True))
    pilot=[];losses={s:[] for s in range(100,105)};refs=[];years=[]
    for year in (2018,2019,2020,2021):
        for seed in range(100,105):
            result=fit('piloto',best,seed,year);pilot.append(result);table('piloto.csv',pilot)
            key=hashlib.sha256(json.dumps(dict(etapa='piloto',configuracion=best,semilla=seed,fold=year,modelo='NHITS_adaptado'),sort_keys=True).encode()).hexdigest()[:20]
            with np.load(OUT/'ajustes/piloto'/key/'predicciones.npz') as z:losses[seed].append(np.abs(z['yhat']-z['y']).mean(1)/CONTRACT['metrica']['q168_kwh'])
        prep,tr,es,ev=pack(year);weekly=references(ev)['ingenuo_168h'];refs.append(np.abs(weekly-ev.y).mean(1)/CONTRACT['metrica']['q168_kwh']);years.extend([year]*len(ev.y))
    state(etapa='dimensionamiento',detalle='Bootstrap pareado por semilla y bloques temporales de14d;10000remuestreos')
    dim=dimensionar(np.stack([np.concatenate(losses[s]) for s in losses]),np.concatenate(refs),np.asarray(years));save(OUT/'dimensionamiento.json',dim)
    save(OUT/'seleccion_cerrada.json',dict(mejor=best,MASE_cv=score,dimensionamiento=dim,fecha=S['actualizado']))
    n=dim['n_final'] if dim['n_final'] is not None else 5
    TOTAL=125+n;final=[]
    prueba=dict(configuracion=best,semillas=list(range(1000,1000+n)),exploratorio=True,potencia_acreditada=False,dimensionamiento=dim,fecha=pd.Timestamp.now(tz='America/Bogota').isoformat())
    if (OUT/'plan_prueba.json').exists():
        antiguo=json.loads((OUT/'plan_prueba.json').read_text())
        assert antiguo['configuracion']==best and antiguo['semillas']==prueba['semillas']
    else:save(OUT/'plan_prueba.json',prueba)
    state(etapa='final',detalle='Prueba retrospectiva exploratoria; cinco semillas si el piloto no permite dimensionar. No se acredita potencia futura.')
    for seed in range(1000,1000+n):final.append(fit('final',best,seed,'final'));table('final_runs.csv',final)
    prep,tr,es,ev=pack('final');reference={name:evaluar(ev.y,q,CONTRACT['metrica']['q168_kwh']) for name,q in references(ev).items()};save(OUT/'referencias_test.json',reference)
    vals=np.array([r['MASE'] for r in final]);summary=dict(modelo='NHITS_adaptado',n_final=n,MASE_validacion=score,anios_confirmacion=[2019,2020,2021],MASE_prueba=float(vals.mean()),MASE_sd=float(vals.std(ddof=1)),MAE_prueba=float(np.mean([r['MAE'] for r in final])),dimensionamiento=dim,nota='Reevaluación retrospectiva exploratoria. El piloto no certifica potencia futura. Arquitectura oficial adaptada a covariables; sin exención E2 automática.')
    summary.update(exploratorio=True,potencia_acreditada=False,semillas=list(range(1000,1000+n)),mejora_ingenuo_semanal_pct=100*(1-vals.mean()/reference['ingenuo_168h']['MASE']))
    for metrica in ['RMSE','sMAPE','MAPE']:summary[metrica+'_prueba']=float(np.mean([r[metrica] for r in final]))
    save(OUT/'resumen_final.json',summary)
    (OUT/'RESUMEN.md').write_text('# NHITS numérico adaptado\n\n'+json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
    save(OUT/'ejecucion_completa.json',dict(ajustes=DONE,semillas=n,exploratorio=True,fecha=S['actualizado']));state(estado='Completado',detalle='Resultados y artefactos guardados.')
if __name__=='__main__':
    try:main()
    except TimeoutError as e:state(estado='Presupuesto temporal agotado',detalle=str(e))
    except BaseException as e:state(estado='Error',detalle=str(e));traceback.print_exc();raise
```

El orquestador preservado utiliza bloqueo `fcntl` y exige GPU en su entrada principal. El siguiente adaptador conserva red, configuración, preparación, parada y predicción, y permite escoger CPU o GPU sin ese controlador de Linux. Su inclusión documental no ejecuta un nuevo ajuste.

### Pipeline explícito compatible con Windows y Linux

```python
"""Compatibilidad del lanzador numérico con las arquitecturas seleccionadas."""
from pathlib import Path
import argparse
import importlib.util
import json


def ejecutar(raiz,familia,semilla=1000,fase='validacion',anio=2021,
             dispositivo='auto',salida=None,permitir_prueba=False):
    if familia not in ('patchtst','nhits'):raise ValueError('Familia desconocida')
    if fase=='prueba' and not permitir_prueba:raise PermissionError('La prueba requiere permiso explícito')
    if salida is None:raise ValueError('Indique una carpeta nueva de salida')
    raiz=Path(raiz).resolve()
    spec=importlib.util.spec_from_file_location('entrenamiento_portable',raiz/'scripts/entrenar_arquitectura.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    destino=Path(salida)/familia/f'semilla_{semilla}'/(str(anio) if fase=='validacion' else 'prueba')
    # PatchTST = Modelo6 N, configuración 18; N-HiTS = configuración N14.
    return mod.ejecutar(familia,destino,raiz,semilla,fase,anio,dispositivo,permitir_prueba)['metricas']


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--raiz',type=Path,default=Path.cwd())
    ap.add_argument('--familia',choices=['patchtst','nhits'],required=True)
    ap.add_argument('--semilla',type=int,default=1000)
    ap.add_argument('--fase',choices=['validacion','prueba'],default='validacion')
    ap.add_argument('--anio',type=int,default=2021)
    ap.add_argument('--dispositivo',choices=['auto','cpu','cuda'],default='auto')
    ap.add_argument('--salida',type=Path,required=True)
    ap.add_argument('--allow-test',action='store_true')
    a=ap.parse_args()
    print(json.dumps(ejecutar(a.raiz,a.familia,a.semilla,a.fase,a.anio,a.dispositivo,a.salida,a.allow_test),indent=2))


if __name__=='__main__':main()
```


## Evaluación

Los resultados de esta familia se integran en la evaluación estadística conjunta, con métricas por semilla, horizonte y origen. La comparación distingue las entradas demanda/calendario de las entradas numéricas ampliadas y el aporte de texto oficial.

## Fuentes

[Challu et al., N-HiTS](https://ojs.aaai.org/index.php/AAAI/article/view/25854). La descripción corresponde a la implementación del experimento; las adaptaciones se identifican en el texto.

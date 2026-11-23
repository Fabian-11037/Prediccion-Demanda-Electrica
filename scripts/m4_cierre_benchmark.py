"""Benchmark numérico adicional con configuraciones fijas y diez semillas.

No sustituye ni modifica el Top10 histórico con texto: los modelos sin corpus
se identifican como benchmark numérico adicional, no réplicas de TaTS/Fusión/TG.
"""
from e3_gpu_comun import *

MODELOS=['MLP','LSTM','CNN1D','TCN','iTransformer','PatchTST','NHITS','NBEATS','TimesNet','TFT']

class Basica(nn.Module):
    def __init__(self,n,c,nombre):
        super().__init__();self.nombre=nombre
        if nombre=='MLP':self.red=nn.Sequential(nn.Linear(168*n+24*c,128),nn.GELU(),nn.Dropout(.1),nn.Linear(128,128),nn.GELU(),nn.Dropout(.1),nn.Linear(128,24))
        elif nombre=='LSTM':self.red=nn.LSTM(n+c,48,batch_first=True);self.cabeza=nn.Sequential(nn.Dropout(.1),nn.Linear(48+24*c,24))
        else:self.red=nn.Sequential(nn.Conv1d(n+c,16,5,padding=2),nn.GELU(),nn.Dropout(.1),nn.AvgPool1d(2));self.cabeza=nn.Linear(84*16+24*c,24)
    def forward(self,x,cp,cf,dia):
        if self.nombre=='MLP':return 1+self.red(torch.cat([x.flatten(1),cf.flatten(1)],1))
        z=torch.cat([x,cp],2)
        if self.nombre=='LSTM':z=self.red(z)[0][:,-1]
        else:z=self.red(z.transpose(1,2)).flatten(1)
        return 1+self.cabeza(torch.cat([z,cf.flatten(1)],1))

def configuraciones():
    f=REPO/'config/m4_cierre_benchmark'
    d={}
    for nombre,paquete,i in [('TCN','Modelo4_TCN',7),('iTransformer','Modelo5_iTransformer',1),('TFT','Modelo3_TFT',1)]:d[nombre]=leer(f/f'{paquete}.json')['configuraciones'][i-1]
    e2=leer(PAQUETE/'config/plan_e2.json');d['TimesNet']=e2['nuevos']['timesnet']['configuraciones'][18];d['NBEATS']=e2['nuevos']['nbeats']['configuraciones'][15]
    d['PatchTST']=leer(REPO/'continuidad/DISENO_CUATRO_MODELOS_GPU.json')['expertos_configuraciones_seleccionadas']['patchtst']['configuracion']
    d['NHITS']=dict(dropout=.1956080336970198,lr=.00010335986527468987,weight_decay=1.4416188128158413e-6,lote=64)
    for n in ['MLP','LSTM','CNN1D']:d[n]=dict(dropout=.1,lr=.001,weight_decay=.0001,lote=64)
    return d

def construir(datos,nombre,p):
    n=len(datos.hist)+len(datos.nombres_mascara);c=len(datos.cal)
    if nombre in ['MLP','LSTM','CNN1D']:return Basica(n,c,nombre)
    nombres=dict(TCN='tcn',iTransformer='itransformer',TFT='tft',PatchTST='tats',NBEATS='nbeats',TimesNet='timesnet',NHITS='nhits')
    return redes.construir(nombres[nombre],p,n,c,168,24,n_mascaras=len(datos.nombres_mascara))

def ajustar_benchmark(datos,nombre,p,s,tr,es,car,progreso):
    if (car/'TERMINADO.json').exists():return
    motor.fijar_semilla(s);esc=datos.ajustar_escalas(tr)
    c={k:datos.construir(j,esc) for k,j in [('tr',tr),('es',es)]};tt={k:motor.a_tensores(z,'cuda') for k,z in c.items()}
    niveles={k:torch.as_tensor(z.nivel.astype('float32'),device='cuda') for k,z in c.items()}
    red=construir(datos,nombre,p).cuda();batch=1 if nombre=='TimesNet' else 64
    def loss(k,j):
        z=tt[k];pred=red(z['x'][j],z['cal_p'][j],z['cal_f'][j],z['dia'][j]);return ((pred-z['y'][j])*niveles[k][j,None]).abs().mean()/datos.q168
    def val():return sum(float(loss('es',slice(i,i+batch)))*len(tt['es']['y'][i:i+batch]) for i in range(0,len(es),batch))/len(es)
    optim=dict(lr=p.get('lr',.001),weight_decay=p.get('weight_decay',.0001),lote=p.get('lote',64))
    info=ajustar(red,len(tr),lambda j:loss('tr',j),val,optim,dict(max_epocas=100,min_epocas=10,paciencia=10,min_delta=.0001),s,car,progreso,f'{getattr(datos,"nombre","colombia")} {nombre} s{s}')
    checkpoint(car/'modelo.pt',dict(modelo=red.state_dict(),escalas=esc,configuracion=p,nombre=nombre,semilla=s))
    guardar(car/'escaladores.json',escalas_a_json(esc,datos));guardar(car/'TERMINADO.json',dict(semilla=s,nombre=nombre,parametros=sum(p.numel() for p in red.parameters()),**{k:v for k,v in info.items() if k!='historia'}))
    del red,tt,c,niveles;torch.cuda.empty_cache()

def prueba_benchmark(datos,nombre,s,pos,car):
    destino=car/'predicciones.npz'
    if destino.exists():
        with np.load(destino) as z:return z['pred']
    ck=cargar(car/'modelo.pt');c=datos.construir(pos,ck['escalas']);t=motor.a_tensores(c,'cuda')
    red=construir(datos,nombre,ck['configuracion']).cuda().eval();red.load_state_dict(ck['modelo'])
    pr=(motor.predecir(red,t,1 if nombre=='TimesNet' else 64)*c.nivel[:,None]).astype('float32')
    assert np.isfinite(pr).all();np.savez_compressed(destino,dia=c.origen.strftime('%Y-%m-%d').to_numpy().astype('U10'),y=c.y,pred=pr)
    del red,t,c;torch.cuda.empty_cache();return pr

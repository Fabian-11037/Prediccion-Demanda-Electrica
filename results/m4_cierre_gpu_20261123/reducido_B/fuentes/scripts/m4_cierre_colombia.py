"""Ampliación independiente de M1/M4_N a semillas nuevas, sin tocar históricos."""
from e3_gpu_comun import *
from e3_modelo1 import *
from e3_modelo3 import leer_base,inferir_base
from probar_m1_gpu import rasgos

HIST=REPO/'results/corrector_final_y_m1_gpu_20261122_150345/M1'
M3=REPO/'results/m3_gpu_20261122_193240'

def preparar(s,car,progreso):
    destino=car/'base_oof'/f'semilla_{s}.npz'
    if destino.exists():
        with np.load(destino) as z:return {k:z[k] for k in z.files}
    if s<=105:return leer_base(M3,s)
    datos=DatosNumericos(PAQUETE);d=leer(REPO/'continuidad/DISENO_CUATRO_MODELOS_GPU.json');m1=car/'M1'
    esp=d['expertos_configuraciones_seleccionadas']
    for anio in range(2011,2022):
        for nombre in EXPERTOS:
            experto(datos,nombre,esp[nombre]['configuracion'],s,anio,m1/'expertos'/nombre/f'anio_{anio}'/f'semilla_{s}',progreso,esp[nombre]['n_parametros'])
    meta=datos_meta(datos,m1,s);partes=[]
    for anio in range(2013,2022):
        p=leer(M3/f'prefijos/seleccion_{anio}.json')['configuracion'];a,es,ev=dividir(meta,anio)
        path=m1/'prefijos'/f'anio_{anio}'/f'semilla_{s}'
        entrenar_router(meta,p,s,a,es,ev,path,progreso,datos.q168)
        partes.append(inferir_base(meta,ev,path/'modelo.pt',car/'oof_trimestral'/f'anio_{anio}_semilla_{s}.npz'))
    base={k:np.concatenate([p[k] for p in partes]) for k in partes[0]};base['anio']=pd.DatetimeIndex(base['dia']).year.to_numpy()
    a,es,ev=dividir(meta,2022);cfg=leer(HIST/'CONFIGURACION_ELEGIDA.json')['configuracion']
    entrenar_router(meta,cfg,s,a,es,ev,m1/'final'/f'semilla_{s}',progreso,datos.q168)
    controles(meta,a,es,np.array([],dtype='int64'),m1/'controles_finales'/f'semilla_{s}',datos.q168)
    for nombre in EXPERTOS:
        experto(datos,nombre,esp[nombre]['configuracion'],s,'final',m1/'expertos_finales'/nombre/f'semilla_{s}',progreso,esp[nombre]['n_parametros'])
    destino.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(destino,**base);return base

def prueba(s,car,base):
    if s<=105:
        with np.load(M3/'base_final'/f'semilla_{s}.npz') as z:parte={k:z[k] for k in z.files}
    else:
        datos=DatosNumericos(PAQUETE);m1=car/'M1';pp=[];pos=datos.prueba();dias=datos.indice[pos].strftime('%Y-%m-%d').to_numpy().astype('U10')
        for nombre in EXPERTOS:
            ck=cargar(m1/'expertos_finales'/nombre/f'semilla_{s}'/'modelo.pt');c=datos.construir(pos,ck['escalas']);tt=motor.a_tensores(c,'cuda')
            red=construir_experto(datos,nombre,ck['metadatos']['configuracion']).cuda().eval();red.load_state_dict(ck['modelo'])
            pp.append((motor.predecir(red,tt,128)*c.nivel[:,None]).astype('float32'));del red,tt
        pred=np.stack(pp,-1);meta=datos_meta(datos,m1,s);n=len(meta['dia'])
        dd=np.concatenate([meta['dia'],dias]);y=np.concatenate([meta['y'],c.y.astype('float32')]);p=np.concatenate([meta['pred'],pred])
        xx=rasgos(datos,dd,y,p)
        completo=dict(dia=dd,y=y,pred=p,x=xx,c=complementariedad(y,p,datos.q168))
        parte=inferir_base(completo,np.arange(n,len(dd)),m1/'final'/f'semilla_{s}'/'modelo.pt',car/'base_final'/f'semilla_{s}.npz')
    claves=['dia','y','pred_base','num','expertos'];r={k:np.concatenate([base[k],parte[k]]) for k in claves}
    r['anio']=pd.DatetimeIndex(r['dia']).year.to_numpy();return r

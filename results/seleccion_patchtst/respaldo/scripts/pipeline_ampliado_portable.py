"""Ajuste/validación o ajuste/prueba explícitos de una configuración congelada.

La construcción del JBook sólo muestra este archivo: no ejecuta su main.
"""
from pathlib import Path
import argparse,json,random,sys
import numpy as np
import pandas as pd


def ejecutar(raiz,familia,semilla=1000,fase='validacion',anio=2021,
             dispositivo='auto',salida=None,permitir_prueba=False):
    if familia not in ['patchtst','nhits']:raise ValueError('Familia desconocida')
    if fase not in ['validacion','prueba']:raise ValueError('Fase desconocida')
    if fase=='prueba' and not permitir_prueba:raise PermissionError('La prueba requiere permiso explícito')
    raiz=Path(raiz).resolve()
    fuentes=raiz/'results/arquitecturas_e2/fuentes/numerico_repositorio'
    if 'demanda' in sys.modules:
        raise RuntimeError('Ejecute en un proceso nuevo para conservar la versión de fuentes')
    sys.path.insert(0,str(fuentes/'src'))
    import torch
    from demanda import particion as P
    from demanda.metricas import evaluar
    from demanda.patchtst_numerico import Preparador,PatchTSTNumerico,entrenar,predecir
    from demanda.nhits_numerico import NHITSNumerico
    dev=('cuda' if torch.cuda.is_available() else 'cpu') if dispositivo=='auto' else dispositivo
    if dev=='cuda' and not torch.cuda.is_available():raise RuntimeError('GPU no disponible')
    random.seed(semilla);np.random.seed(semilla);torch.manual_seed(semilla)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(semilla)
    seleccion=json.loads((raiz/'results/arquitecturas_e2/seleccion_numerica.json').read_text(encoding='utf-8'))
    p=dict(seleccion[familia]);p['semilla']=int(semilla)
    contrato=json.loads((fuentes/'config/diseno_entrenamiento_texto.json').read_text(encoding='utf-8'))
    panel=pd.read_parquet(raiz/contrato['panel'])
    registro=json.loads((raiz/'data/processed/feature_registry.json').read_text(encoding='utf-8'))
    datos=Preparador(panel,registro,contrato)
    if fase=='prueba':
        it,ies=P.pretest(panel.index,24,168,24,0,.10)
        iev=P.origenes(panel.index,'test',24,168,24,0,allow_test=True)
    else:
        fold=P.folds_forward_chaining(panel.index,[anio],24,168,24,0,.10)[0]
        it,ies,iev=fold['tr'],fold['es'],fold['val']
    it,ies,iev=[datos.elegibles(v) for v in [it,ies,iev]]
    assert panel.index[it.max()]+pd.Timedelta(hours=23)<panel.index[ies.min()]
    assert panel.index[ies.max()]+pd.Timedelta(hours=23)<panel.index[iev.min()]
    datos.ajustar(it)
    tr,es,ev=[datos.transformar(v) for v in [it,ies,iev]]
    clase={'patchtst':PatchTSTNumerico,'nhits':NHITSNumerico}[familia]
    red=clase(p,datos.stats['sigma_demanda'])
    historia=entrenar(red,tr,es,p,max_epocas=200,device=dev)
    pred=predecir(red,ev,batch_size=p['batch_size'],device=dev)
    met=evaluar(ev.y,pred,215617.90361589016)
    if salida is not None:
        destino=Path(salida)/familia/f'semilla_{semilla}'/(str(anio) if fase=='validacion' else 'prueba')
        destino.mkdir(parents=True,exist_ok=False)
        torch.save({'state_dict':{k:v.detach().cpu() for k,v in red.state_dict().items()},
                    'configuracion':p,'escaladores':datos.stats},destino/'modelo.pt')
        np.savez_compressed(destino/'predicciones.npz',y=ev.y,yhat=pred,origen=ev.origen.as_unit('ns').asi8)
        for nombre,obj in [('historia',historia),('escaladores',datos.stats),('metricas',met)]:
            (destino/(nombre+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
    return met


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

"""Integra N-BEATS y TimesNet conservando checkpoints y predicciones congeladas."""
from pathlib import Path
import ast,hashlib,json,shutil,re
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'results/evaluacion_e2';OUT=ROOT/'results/integracion_top10_20261120'
SOURCE=ROOT/'results/analisis_nbeats_timesnet_20261120/entrada_resultados/resultados_e2_nbeats_timesnet'
REEVAL=ROOT/'results/reevaluacion_timesnet_lote1_20261120'
BUNDLE=ROOT/'results/analisis_nbeats_timesnet_20261120/fuentes_paquete/NBEATS_TimesNet_E2'
FAMILIES={'nbeats':'N-BEATS','timesnet':'TimesNet'}
KEYS={f:'nuevo_'+f+'__N' for f in FAMILIES};Q=215617.90361589016

def fuente(root,family,seed=None):
 p=Path(root)/'results/evaluacion_e2/entradas/adicionales'/('nuevo_'+family)
 return p if seed is None else p/f'final/N/semilla_{seed}'

def agregar_metadatos(root,rows):
 for f in FAMILIES:
  p=fuente(root,f);sel=json.loads((p/'busqueda/configuracion_elegida.json').read_text(encoding='utf8'))['N']
  for seed in range(1000,1005):
   m=json.loads((fuente(root,f,seed)/'resultado.json').read_text(encoding='utf8'))
   rows.append(dict(clave=KEYS[f],semilla=seed,validacion=sel['MASE_medio_validacion'],anios_validacion='2018–2021',parametros=m['n_parametros'],segundos=m['segundos'],epocas=m['epocas'],mejor_epoca=m['mejor_epoca'],MAE_ajuste_normalizado=np.nan,MAE_parada_normalizado=np.nan,MASE_ajuste=m['MASE_ajuste'],MASE_parada=m['MASE_parada'],dispositivo=m['costes']['dispositivo']+' · '+m['costes'].get('gpu','')))
 return rows

def agregar_diseno(root,rows):
 for f in FAMILIES:
  rows.append(dict(clave=KEYS[f],candidatos_filtro=20,filtro='20 ajustes; 1 semilla; 2018',confirmacion='15 ajustes; 1 semilla; 2019–2021',piloto='Sin piloto independiente',final='5 semillas',max_epocas='200 / 200',paciencia=6))
 return rows

def agregar_parada(root,rows):
 for f in FAMILIES:
  for seed in range(1000,1005):
   p=fuente(root,f,seed)/'resultado.json';m=json.loads(p.read_text(encoding='utf8'));s=json.loads(p.with_name('escaladores.json').read_text(encoding='utf8'))
   rows.append(dict(clave=KEYS[f],semilla=seed,MASE_parada=m['MASE_parada'],n_parada=m['n_parada'],inicio_parada=m['origen_parada'][0],fin_parada=m['origen_parada'][1],q168_kwh=s['q168_kwh'],fuente_resultado=str(p.relative_to(root)).replace('\\','/'),sha256_resultado=hashlib.sha256(p.read_bytes()).hexdigest()))
 return rows

def importar():
 OUT.mkdir(exist_ok=True);backup=OUT/'respaldo';backup.mkdir(exist_ok=True)
 for p in [R/'entradas/consolidado/predicciones_finales.parquet',R/'entradas/consolidado/resumen_principal.csv',*R.glob('*.csv'),R/'protocolo.json',ROOT/'results/arquitecturas_e2/indice_documentacion.json']:
  d=backup/p.relative_to(ROOT);d.parent.mkdir(parents=True,exist_ok=True)
  if not d.exists():shutil.copy2(p,d)
 for f in FAMILIES:shutil.copytree(SOURCE/('nuevo_'+f),fuente(ROOT,f),dirs_exist_ok=True)
 frame=pd.read_parquet(R/'entradas/consolidado/predicciones_finales.parquet')
 frame=frame[~frame.experimento.isin(['nuevo_nbeats','nuevo_timesnet'])]
 old_count=len(frame);add=[];summaries=[];curves=[]
 for f,label in FAMILIES.items():
  sel=json.loads((fuente(ROOT,f)/'busqueda/configuracion_elegida.json').read_text(encoding='utf8'))['N'];ms=[];arrays=[]
  for seed in range(1000,1005):
   folder=fuente(ROOT,f,seed);m=json.loads((folder/'resultado.json').read_text(encoding='utf8'));ms.append(m)
   with np.load(REEVAL/f'predicciones_{seed}.npz',allow_pickle=False) as z:
    y=z['y_kwh'].copy();dates=pd.DatetimeIndex(z['origen']).tz_localize('America/Bogota')
    if f=='timesnet':p=z['yhat_lote1_kwh'].copy()
   if f=='nbeats':
    with np.load(folder/'pred_prueba.npz',allow_pickle=False) as z:p=z['yhat_kwh'].astype(float)
   assert y.shape==p.shape==(1461,24)
   # Publicar sólo la predicción canónica de cada modelo entre las entradas analíticas.
   np.savez_compressed(folder/'predicciones_canonicas.npz',origen=dates.strftime('%Y-%m-%d').to_numpy(dtype='U10'),y_kwh=y,yhat_kwh=p)
   add.append(pd.DataFrame(dict(experimento='nuevo_'+f,variante='N',semilla=seed,origen=np.repeat(dates,24),horizonte=np.tile(np.arange(1,25),1461),observado_kwh=y.ravel(),predicho_kwh=p.ravel())))
   h=pd.read_csv(folder/'historia.csv').rename(columns={'perdida_ajuste':'train_loss','perdida_parada':'val_loss'});h['clave']=KEYS[f];h['semilla']=seed;h['perdida']='MAE normalizado';curves.append(h[['clave','semilla','epoca','train_loss','val_loss','perdida']]);arrays.append(p)
  a=np.stack(arrays);err=a-y;metrics=np.abs(err).mean((1,2))/Q
  summaries.append(dict(experimento='nuevo_'+f,modelo=label,variante='N',principal=True,semillas=5,MASE_validacion=sel['MASE_medio_validacion'],anios_validacion='2018–2021',MASE_prueba=metrics.mean(),MASE_sd=metrics.std(ddof=1),MAE_kwh=np.abs(err).mean(),RMSE_kwh=np.sqrt((err**2).mean((1,2))).mean(),MAPE_pct=(100*np.abs(err)/np.abs(y)).mean(),sMAPE_pct=(200*np.abs(err)/(np.abs(a)+np.abs(y))).mean(),MASE_ajuste=np.mean([m['MASE_ajuste'] for m in ms]),MASE_parada=np.mean([m['MASE_parada'] for m in ms]),epocas_media=np.mean([m['epocas'] for m in ms]),epocas_min=min(m['epocas'] for m in ms),epocas_max=max(m['epocas'] for m in ms),mejor_epoca_media=np.mean([m['mejor_epoca'] for m in ms]),parametros=ms[0]['n_parametros'],segundos_finales=sum(m['segundos'] for m in ms),ajustes=40,configuracion=json.dumps(sel['configuracion']),indicadores_ausencia=3,codigo_fuente_incluido=True))
 pd.concat([frame,*add],ignore_index=True).to_parquet(R/'entradas/consolidado/predicciones_finales.parquet',index=False)
 s=pd.read_csv(R/'entradas/consolidado/resumen_principal.csv');s=s[~s.experimento.isin(['nuevo_nbeats','nuevo_timesnet'])]
 pd.concat([s,pd.DataFrame(summaries)],ignore_index=True).to_csv(R/'entradas/consolidado/resumen_principal.csv',index=False)
 h=pd.read_csv(R/'curvas_aprendizaje.csv');h=h[~h.clave.isin(KEYS.values())];pd.concat([h,*curves],ignore_index=True).to_csv(R/'curvas_aprendizaje.csv',index=False)
 # Las entradas guardadas permiten repetir el análisis sin los archivos de Descargas.
 code=ROOT/'book/_static/codigo_e2/top10_adicionales';shutil.copytree(BUNDLE/'src',code/'src',dirs_exist_ok=True)
 shutil.copytree(BUNDLE/'config',code/'config',dirs_exist_ok=True)
 for p in (code/'src').rglob('__pycache__'):shutil.rmtree(p)
 (OUT/'integracion.json').write_text(json.dumps(dict(fecha='2026-11-20',filas_anteriores_preservadas=old_count,filas_nuevas=sum(len(d) for d in add),procedimientos=15,realizaciones=75,referencias=2,top10=list(['PatchTST','N-HiTS','TFT','TCN','iTransformer','TaTS','Fusión tardía','TGForecaster','N-BEATS','TimesNet']),timesnet_lote_prueba=1,semillas=list(range(1000,1005)),entrenamiento=False,publicacion=False,fuente_predicciones_timesnet=str(REEVAL.relative_to(ROOT))),ensure_ascii=False,indent=2),encoding='utf8')
 print('Entradas canónicas incorporadas: 10 realizaciones; TimesNet lote uno.',flush=True)

def ajustar_celdas(root,cells):
 from narrativa_resultados import actualizar_celdas
 return actualizar_celdas(root,cells,'10_evaluacion')


def armonizar(root):
 from narrativa_resultados import aplicar
 reporte=aplicar(root)
 root=Path(root);dst=root/'book/_static/resultados/evaluacion_e2';dst.mkdir(exist_ok=True,parents=True)
 for path in (root/'results/evaluacion_e2').glob('*'):
  if path.is_file() and path.suffix in ['.csv','.json']:shutil.copy2(path,dst/path.name)
 return reporte


def integrar_diagnosticos():
 keys=set(pd.read_csv(R/'benchmark.csv').query('semillas==5').clave)
 d=ROOT/'results/diagnosticos_ampliados'
 for name in ['importancia_semillas.csv','importancia_resumen.csv','latencia_semillas.csv','latencia_resumen.csv']:
  target=OUT/'respaldo/results/diagnosticos_ampliados'/name;target.parent.mkdir(parents=True,exist_ok=True)
  if not target.exists():shutil.copy2(d/name,target)
 old=pd.read_csv(d/'importancia_semillas.csv');old=old[old.clave.isin(keys-set(KEYS.values()))]
 new=pd.concat([pd.read_csv(OUT/f'importancia_{f}_{seed}.csv') for f in FAMILIES for seed in range(1000,1005)])
 imp=pd.concat([old,new],ignore_index=True);assert imp.clave.nunique()==11 and len(imp)==11*5*3*9
 assert not imp.duplicated(['clave','semilla','repeticion','grupo']).any()
 imp.to_csv(d/'importancia_semillas.csv',index=False)
 reps=imp.groupby(['clave','grupo','repeticion']).agg(delta_MASE=('delta_MASE','mean'),base=('MAE_base','mean'),pert=('MAE_perturbado','mean')).reset_index()
 reps['cambio_pct']=100*(reps.pert/reps.base-1)
 reps.groupby(['clave','grupo']).agg(delta_MASE=('delta_MASE','mean'),cambio_pct=('cambio_pct','mean'),p10=('cambio_pct',lambda s:s.quantile(.1)),p90=('cambio_pct',lambda s:s.quantile(.9))).reset_index().to_csv(d/'importancia_resumen.csv',index=False)
 lat=pd.read_csv(d/'latencia_semillas.csv');lat=lat[lat.clave.isin(keys-set(KEYS.values()))]
 lat=pd.concat([lat,pd.read_csv(OUT/'latencia_nuevos.csv')],ignore_index=True)
 assert set(lat.clave)==keys and len(lat)==225 and not lat.duplicated(['clave','semilla','origen']).any()
 lat.to_csv(d/'latencia_semillas.csv',index=False)
 lat.groupby('clave').agg(mediana_ms=('mediana_ms','median'),p95_ms=('p95_ms','median'),minimo_ms=('mediana_ms','min'),maximo_ms=('mediana_ms','max'),mediciones=('mediana_ms','size')).reset_index().to_csv(d/'latencia_resumen.csv',index=False)
 shutil.copy2(OUT/'protocolo_diagnosticos.json',d/'protocolo_top10_adicionales.json')
 print('Importancia: 11 procedimientos ampliados; latencia: 15 procedimientos y 225 mediciones resumidas.')

if __name__=='__main__':
 import sys
 integrar_diagnosticos() if '--diagnosticos' in sys.argv else importar()

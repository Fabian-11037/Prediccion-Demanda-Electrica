"""Evalúa los checkpoints finales de M4 en ajuste y parada; no optimiza parámetros."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from completar_e3_sin_entrenar import load,Q,sha
OUT=ROOT/'results/complementos_m4_20261125'

def calculate():
 OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1)
 rows=[];daily=[];sources=[];val_predictions=[];train_losses=[];val_losses=[]
 for seed in range(101,111):
  red,ck,meta,b=load(seed)
  assert not red.training and not any(p.requires_grad for p in red.parameters())
  days=np.asarray(meta['dia'],dtype='datetime64[D]')
  assert days.max()<np.datetime64('2022-01-01')
  a=np.flatnonzero(days<np.datetime64('2020-12-31'));v=np.flatnonzero((days>=np.datetime64('2021-01-01'))&(days<np.datetime64('2022-01-01')))
  assert np.all(np.diff(days)==np.timedelta64(1,'D')) and len(v)==365
  assert str(days[a[0]])=='2013-01-01'and str(days[a[-1]])=='2020-12-30'
  assert np.allclose(meta['num'][a].mean(0),ck['media'],rtol=0,atol=1e-7)
  run=ROOT/'results/m4_cierre_gpu_20261123/reducido_B'
  cp=run/f'colombia/final/M4_N/semilla_{seed}/modelo.pt';before=sha(cp)
  info=json.loads((cp.parent/'TERMINADO.json').read_text(encoding='utf-8'))
  with torch.inference_mode():
   reps=red.codificar(b,len(days));part_losses={}
   for label,ix in [('ajuste',a),('parada',v)]:
    p=[]
    for start in range(0,len(ix),64):
     j=torch.tensor(ix[start:start+64]);pred,gate,ids,att,ok=red(b,j,reps)
     ages=b.dias[j,None]-b.dias[ids]
     assert ((~ok)|((ages>=2)&(ages<=365))).all()
     p.append(pred.numpy())
    pp=np.concatenate(p);loss=np.abs(pp.astype(float)-meta['y'][ix].astype(float)).mean(1)/Q
    part_losses[label]=loss
    daily.extend(dict(semilla=seed,particion=label,dia=str(days[i]),MASE=float(l))for i,l in zip(ix,loss))
    if label=='parada':val_predictions.append(pp)
   aa=part_losses['ajuste'];vv=part_losses['parada'];saved=info['mejor_perdida']
   assert abs(vv.mean()-saved)<1e-6,(seed,vv.mean(),saved)
   train_losses.append(aa);val_losses.append(vv)
  assert sha(cp)==before
  sources.append(dict(checkpoint=cp.relative_to(ROOT).as_posix(),sha256=before,base=(run/f'colombia/base_oof_semilla_{seed}.npz').relative_to(ROOT).as_posix(),base_sha256=sha(run/f'colombia/base_oof_semilla_{seed}.npz')))
  hist=pd.read_csv(cp.parent/'historia.csv');best=hist.loc[hist.epoca.eq(info['mejor_epoca'])].iloc[0]
  rows.append(dict(semilla=seed,mejor_epoca=info['mejor_epoca'],n_ajuste=len(a),n_parada=len(v),
   MASE_ajuste=float(aa.mean()),MASE_parada=float(vv.mean()),brecha_MASE=float(vv.mean()-aa.mean()),
   brecha_relativa_pct=float(100*(vv.mean()/aa.mean()-1)),perdida_ajuste_epoca_guardada=float(best.perdida_ajuste),
   perdida_parada_guardada=float(saved),diferencia_parada_reproduccion=float(vv.mean()-saved)))
  print('Evaluación congelada ajuste/parada:',seed,rows[-1]['brecha_relativa_pct'],flush=True)
 pd.DataFrame(rows).to_csv(OUT/'brecha_por_semilla.csv',index=False)
 pd.DataFrame(daily).to_csv(OUT/'brecha_perdidas_diarias.csv',index=False)
 np.savez_compressed(OUT/'predicciones_parada.npz',dia=meta['dia'][v],y=meta['y'][v],pred=np.stack(val_predictions),semillas=np.arange(101,111))
 from evaluar_complementos_m4 import counts
 from threadpoolctl import threadpool_limits
 with threadpool_limits(limits=4):
  ta=np.mean(train_losses,axis=0);tv=np.mean(val_losses,axis=0)
  ba=counts(len(ta),20261126)@ta;bv=counts(len(tv),20261127)@tv
  gap=100*(bv/ba-1);ci=np.quantile(gap,[.025,.975])
 summary=dict(MASE_ajuste=float(ta.mean()),MASE_parada=float(tv.mean()),brecha_MASE=float(tv.mean()-ta.mean()),
  brecha_relativa_pct=float(100*(tv.mean()/ta.mean()-1)),ic95_inf_pct=float(ci[0]),ic95_sup_pct=float(ci[1]),
  n_ajuste=len(ta),n_parada=len(tv),inicio_ajuste=str(days[a[0]]),fin_ajuste=str(days[a[-1]]),
  inicio_parada=str(days[v[0]]),fin_parada=str(days[v[-1]]),semillas=10,
  max_diferencia_parada=float(max(abs(r['diferencia_parada_reproduccion'])for r in rows)),
  modo='eval(), dropout desactivado, inference_mode, pesos/escalas congelados; mismo cálculo MAE/Q',
  IC='Bootstrap estacionario56,9999 réplicas; periodos remuestreados por separado, diez inicializaciones fijas. IC puntual de la razón de pérdidas medias.',
  interpretacion='La diferencia entre periodos combina ajuste del corrector y cambio temporal. La pérdida online de cada época promedia pesos cambiantes y dropout activo; no es la pérdida del checkpoint restaurado.',
  entrenamiento=False,publicacion=False,checkpoints_sin_cambios=True,fuentes=sources)
 (OUT/'brecha_resumen.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 print(json.dumps({k:v for k,v in summary.items()if k!='fuentes'},ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':calculate()

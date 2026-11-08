"""Auditoría del panel y de ventanas de entrenamiento/validación, sin prueba."""
import json,sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from demanda import modelos_base as M
from demanda import particion as P
r=ROOT/'results/auditoria_dataset';r.mkdir(exist_ok=True)
panel=pd.read_parquet(ROOT/'data/processed/panel_horario.parquet');panel=panel.loc[panel.index<pd.Timestamp('2022-01-01',tz='America/Bogota')].copy()
reg=json.loads((ROOT/'data/processed/feature_registry.json').read_text());d=M.Datos(panel,reg,M.ConfigDatos())
assert panel.index.is_unique and panel.index.is_monotonic_increasing
assert str(panel.index.tz)=='America/Bogota'
assert (panel.index[1:]-panel.index[:-1] == pd.Timedelta(hours=1)).all()
y=panel.y_demanda_real_kwh;x=panel.x_demanda_real_kwh
assert np.allclose(x.iloc[1:],y.iloc[:-1],rtol=0,atol=0,equal_nan=True)
assert all(reg[c]['clase']=='past_only' for c in d.cols_x)
assert all(reg[c]['clase']=='known_future' for c in d.cols_cal)
report={'filas_hasta_2021':len(panel),'canales_demanda':d.cols_x,'canales_calendario':d.cols_cal,'comprobaciones':{'indice_unico_ordenado_horario':True,'rezago_demanda_1h_exacto':True,'clases_disponibilidad':True},'folds':[]}
for f in P.folds_forward_chaining(panel.index):
 tr,es,va=f['tr'],f['es'],f['val']
 assert panel.index[tr[-1]]+pd.Timedelta(hours=23)<panel.index[es[0]]
 assert panel.index[es[-1]]+pd.Timedelta(hours=23)<panel.index[va[0]]
 item={'anio':f['anio'],'origenes_train':len(tr),'origenes_es':len(es),'origenes_val':len(va),'inicio_es':str(panel.index[es[0]])}
 for label,pos in [('tr',tr),('es',es),('val',va)]:
  c=d.conjunto_pos(pos)
  descartados=np.setdiff1d(pos,panel.index.get_indexer(c.origen))
  assert all(x.iloc[p-168:p].isna().any() for p in descartados)
  item['descartados_nan_'+label]=len(descartados)
  pos=panel.index.get_indexer(c.origen)
  assert c.x.shape==(len(pos),168,1) and c.cal_p.shape==(len(pos),168,len(d.cols_cal)) and c.cal_f.shape==(len(pos),24,len(d.cols_cal)) and c.y.shape==(len(pos),24)
  assert all(np.isfinite(a).all() for a in (c.x,c.cal_p,c.cal_f,c.y,c.nivel))
  assert (c.nivel>0).all() and (c.y>0).all()
  assert np.allclose(c.x[:,:,0].mean(1),1,atol=1e-6)
  assert np.allclose(d.invertir(d.objetivo_norm(c),c),c.y,rtol=1e-6)
  # Revisión exhaustiva de demanda y etiquetas para cada origen.
  for j,p in enumerate(pos):
   raw=x.iloc[p-168:p].to_numpy();assert np.allclose(c.x[j,:,0],raw/raw.mean(),rtol=1e-6)
   assert np.array_equal(c.y[j],y.iloc[p:p+24].to_numpy())
  # Verifica calendario en extremos y punto medio contra las filas fuente.
  for j in (0,len(pos)//2,len(pos)-1):
   p=pos[j];cols=d.cols_cal;mu=d.escala.media_[cols].to_numpy();sd=d.escala.std_[cols].to_numpy()
   assert np.allclose(c.cal_p[j],(panel[cols].iloc[p-168:p].to_numpy()-mu)/sd,rtol=1e-6,atol=1e-6)
   assert np.allclose(c.cal_f[j],(panel[cols].iloc[p:p+24].to_numpy()-mu)/sd,rtol=1e-6,atol=1e-6)
  if label=='tr':item['forma_entrada_secuencial']=[len(pos),168,1+len(d.cols_cal)]
 # El ajustador original usa 2000–2017, incluyendo parte de ES en folds tempranos.
 fin_escala=pd.Timestamp('2018-01-01',tz=panel.index.tz)
 overlap=((panel.index>=panel.index[es[0]])&(panel.index<fin_escala)&(panel.index<=panel.index[es[-1]]+pd.Timedelta(hours=23))).sum()
 item['filas_es_incluidas_en_escala_fija']=int(overlap)
 report['folds'].append(item)
# Prueba de no lectura del futuro de las características históricas.
pos=np.array([P.folds_forward_chaining(panel.index)[0]['val'][20]]);p=int(pos[0]);base=d.conjunto_pos(pos)
corrupt=panel.copy();corrupt.loc[corrupt.index>=panel.index[p],d.cols_x]=1e20
alt=M.Datos(corrupt,reg,M.ConfigDatos()).conjunto_pos(pos)
assert np.array_equal(base.x,alt.x) and np.array_equal(base.cal_f,alt.cal_f)
report['comprobaciones']['futuro_x_no_modifica_entrada']=True
report['conclusion']='Alineación, formas y valores correctos. Corrección metodológica pendiente: ajustar escaladores dentro de cada fold excluyendo ES; el ajuste fijo de 2000–2017 incluye ES de folds tempranos.'
(r/'auditoria.json').write_text(json.dumps(report,indent=2,ensure_ascii=False));print(json.dumps(report,indent=2,ensure_ascii=False))

"""Contrasta el MASE de la exportación en cuatro días fijados en el round-trip."""
from pathlib import Path
import json,sys,importlib.util
ROOT=Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT/'.bookenv/Lib/site-packages'))
import numpy as np
import pandas as pd
OUT=ROOT/'results/auditoria_e3_final_20261124';OUT.mkdir(exist_ok=True)
PACKAGE=ROOT/'results/cierre_e3_sin_entrenar_20261124/paquete_m4_cpu'
Q=215617.90361589016

def main():
 spec=importlib.util.spec_from_file_location('m4_onnx_auditado',PACKAGE/'ejecutar.py');runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
 protocol=json.loads((PACKAGE/'exportacion.json').read_text(encoding='utf-8'))
 panel=pd.read_parquet(ROOT/'data/processed/panel_horario.parquet',columns=['y_demanda_real_kwh'])
 rows=[];native=[];onnx=[];labels=[]
 for i,record in enumerate(protocol['resultados']):
  date=record['origen']
  if date>='2026-01-01':continue
  pred=np.asarray(runner.run(PACKAGE,PACKAGE/f'ejemplo_{i}.npz')['prediccion_kwh'])
  original=np.load(PACKAGE/f'esperado_{i}.npy',allow_pickle=False)
  index=pd.date_range(date,periods=24,freq='h',tz='America/Bogota');y=panel.loc[index,'y_demanda_real_kwh'].to_numpy()
  native.append(original);onnx.append(pred);labels.append(y)
  rows.append(dict(origen=date,MASE_nativo=float(np.abs(original-y).mean()/Q),MASE_ONNX=float(np.abs(pred-y).mean()/Q),max_diferencia_kwh=float(np.abs(pred-original).max())))
 n=np.stack(native);o=np.stack(onnx);y=np.stack(labels)
 summary=dict(semilla=101,n_dias=len(rows),n_intervalos=int(y.size),fechas=[r['origen'] for r in rows],
  MASE_nativo=float(np.abs(n-y).mean()/Q),MASE_ONNX=float(np.abs(o-y).mean()/Q),
  diferencia_MASE=float((np.abs(o-y).mean()-np.abs(n-y).mean())/Q),
  max_diferencia_kwh=float(np.abs(o-n).max()),entrenamiento=False,
  alcance='Panel de verificación de cuatro fechas prefijadas; no es el benchmark completo de1461días ni promedio de diez semillas')
 pd.DataFrame(rows).to_csv(OUT/'roundtrip_metricas.csv',index=False)
 (OUT/'roundtrip_metricas.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps(summary,ensure_ascii=False))

if __name__=='__main__':main()

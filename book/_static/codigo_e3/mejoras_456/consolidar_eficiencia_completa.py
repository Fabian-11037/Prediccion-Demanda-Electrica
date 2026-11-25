"""Consolida eficiencia y Pareto sin modificar las métricas originales."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
def main(root=ROOT):
 root=Path(root);out=root/'results/eficiencia_completa_20261125';rows=[];weights={}
 base=pd.read_csv(root/'results/tabla_maestra_20261125/tabla_maestra.csv')
 zero=set('alias cat clamp_min clone copy_ detach expand flip index index_select lift_fresh masked_fill max_pool2d_with_indices new_empty new_zeros permute repeat scatter select slice split split_with_sizes squeeze stack t topk transpose unbind unfold unsqueeze view where zeros_like _local_scalar_dense _unsafe_view'.split())
 for row in base.to_dict('records'):
  d=json.loads((out/'modelos'/(row['clave'].replace('+','_')+'.json')).read_text(encoding='utf-8'))
  unknown={n.split('.')[-1]for v in d['origenes']for n in v['sin_formula']};assert not unknown-zero,(row['clave'],unknown-zero)
  weights.update(d['fuentes_sha256']);flops=[v['FLOPs']for v in d['origenes']]
  rows.append(dict(**row,FLOPs_forward_medio=float(np.mean(flops)),FLOPs_forward_min=int(min(flops)),FLOPs_forward_max=int(max(flops)),pico_proceso_MiB=d['pico_proceso_bytes']/2**20,latencia_GPU_ms=None,estado_GPU='No disponible en el entorno CPU',energia_15W_J=.015*row['latencia_mediana_ms'],energia_57W_J=.057*row['latencia_mediana_ms']))
 t=pd.DataFrame(rows)
 for x,label in [('latencia_mediana_ms','latencia'),('FLOPs_forward_medio','FLOPs')]:
  a=t[[x,'MASE_prueba']].to_numpy();t['Pareto_'+label]=[not any(np.all(b<=v)and np.any(b<v)for b in a)for v in a]
 t.to_csv(out/'rendimiento_coste_completo.csv',index=False)
 for p,h in weights.items():
  with Path(p).open('rb')as f:assert hashlib.file_digest(f,'sha256').hexdigest()==h
 return t
if __name__=='__main__':print(main()[['modelo','FLOPs_forward_medio','Pareto_latencia']].to_string(index=False))

"""Diagnóstico exploratorio con entrenamiento y validación; nunca accede a prueba."""
import importlib.util,json,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
spec=importlib.util.spec_from_file_location('entrenador',ROOT/'scripts/entrenar_modelos_base.py');E=importlib.util.module_from_spec(spec);spec.loader.exec_module(E)
M=E.M
OUT=ROOT/'results/diagnostico_secuenciales';OUT.mkdir(exist_ok=True)
a=E.parser().parse_args(['buscar','--dispositivo','cuda'])
op=E.opciones(a);M.fijar_semillas(42,op)
datos=E.cargar_datos(a);fold=E.hacer_folds(a,datos)[0]
val=datos.conjunto_pos(fold['val']);train=datos.conjunto_pos(fold['tr'][np.linspace(0,len(fold['tr'])-1,16).astype(int)])
def tens(c):
 return (torch.tensor(np.concatenate([c.x,c.cal_p],2),device='cuda'),torch.tensor(c.cal_f,device='cuda'),torch.tensor(datos.objetivo_norm(c),device='cuda'))
x,c,y=tens(val);xt,ct,yt=tens(train)
rows=[]
for path in sorted((ROOT/'results/base_v2/ajustes/piloto').glob('*/identidad.json')):
 i=json.loads(path.read_text())
 if i['modelo'] not in ('mlp','lstm','cnn1d') or not i['inicio_evaluacion'].startswith('2018'):continue
 cp=torch.load(path.parent/'modelo.pt',map_location='cpu',weights_only=False)
 red=M.construir_red(i['modelo'],cp['params'],val).cuda();red.load_state_dict(cp['state_dict']);red.eval()
 rng=np.random.default_rng(42);perm=torch.tensor(rng.permutation(len(x)),device='cuda')
 row={'modelo':i['modelo'],'semilla':i['semilla']}
 for label in ('original','demanda_permutada','calendario_permutado'):
  xx=x.clone();cc=c.clone()
  if label=='demanda_permutada':xx[:,:,0]=x[perm,:,0]
  if label=='calendario_permutado':xx[:,:,1:]=x[perm,:,1:];cc=c[perm]
  with torch.no_grad():q=red(xx,cc).cpu().numpy()
  row[label]=float(np.abs(datos.invertir(q,val)-val.y).mean()/datos.escala_mase)
 rows.append(row)
import csv
with (OUT/'ablaciones_5_semillas.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print(json.dumps(rows),flush=True)

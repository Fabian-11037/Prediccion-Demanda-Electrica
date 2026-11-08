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
report={'dispositivo':torch.cuda.get_device_name(),'fold_validacion':2018,'semilla_checkpoint':100,'semilla_diagnostico':42,'n_tiny':16,'pasos_tiny':400,'resultados':{}}
for nombre in ('mlp','lstm','cnn1d'):
 paths=[]
 for p in (ROOT/'results/base_v2/ajustes/piloto').glob('*/identidad.json'):
  i=json.loads(p.read_text())
  if i['modelo']==nombre and i['semilla']==100 and i['inicio_evaluacion'].startswith('2018'):paths.append(p.parent)
 assert len(paths)==1
 folder=paths[0];cp=torch.load(folder/'modelo.pt',map_location='cpu',weights_only=False);p=cp['params']
 red=M.construir_red(nombre,p,val).cuda();red.load_state_dict(cp['state_dict']);red.eval()
 def pred(z,k):
  with torch.no_grad():return red(z,k).cpu().numpy()
 base=pred(x,c)
 with np.load(folder/'predicciones.npz') as z:expected=z['yhat']
 actual=datos.invertir(base,val)
 err=float(np.max(np.abs(actual-expected)))
 rng=np.random.default_rng(42)
 ab={}
 for label in ('demanda_constante','demanda_permutada','calendario_cero'):
  xx=x.clone();cc=c.clone()
  if label=='demanda_constante':xx[:,:,0]=1
  elif label=='demanda_permutada':xx[:,:,0]=xx[torch.tensor(rng.permutation(len(x)),device='cuda'),:,0]
  else:xx[:,:,1:]=0;cc.zero_()
  q=pred(xx,cc);ab[label]={'MASE':float(np.abs(datos.invertir(q,val)-val.y).mean()/datos.escala_mase),'cambio_rms_normalizado':float(np.sqrt(np.mean((q-base)**2)))}
 red.train();xx=x[:32].clone().requires_grad_(True);red(xx,c[:32]).square().mean().backward();grad=xx.grad.abs().mean((0,1)).detach().cpu().numpy()
 saved=json.loads((folder/'resultado.json').read_text())
 item={'checkpoint':str(folder.relative_to(ROOT)),'params':p,'reproduccion_error_max_kwh':err,'MASE_original':saved['MASE'],'MASE_reconstruido':float(np.abs(actual-val.y).mean()/datos.escala_mase),'ablaciones':ab,'gradiente_demanda':float(grad[0]),'gradiente_calendario_promedio':float(grad[1:].mean()),'tiny':{}}
 for modo in ('original','control_sin_dropout_lr_1e3'):
  M.fijar_semillas(42,op);pp=dict(p)
  if modo!='original':pp.update(dropout=0.,lr=.001,weight_decay=0.)
  net=M.construir_red(nombre,pp,train).cuda();net.preparar(xt,ct)
  optimizer=torch.optim.AdamW(net.parameters(),lr=pp['lr'],weight_decay=pp.get('weight_decay',0))
  lossf={'mse':torch.nn.MSELoss(),'mae':torch.nn.L1Loss(),'huber':torch.nn.HuberLoss(delta=.05)}[pp['perdida']]
  history=[];start=time.time()
  for step in range(401):
   if step%100==0:
    net.eval()
    with torch.no_grad():q=net(xt,ct);history.append({'paso':step,'MAE_normalizado':float((q-yt).abs().mean()),'loss':float(lossf(q,yt))})
   if step==400:break
   net.train();optimizer.zero_grad();loss=lossf(net(xt,ct),yt);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.);optimizer.step()
  item['tiny'][modo]={'historia':history,'segundos':time.time()-start}
 report['resultados'][nombre]=item
 (OUT/'diagnostico.json').write_text(json.dumps(report,indent=2))
 print(nombre,json.dumps(item),flush=True)

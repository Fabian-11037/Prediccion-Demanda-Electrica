"""Cuenta FLOPs de inferencia con operaciones explícitas; no ajusta modelos."""
from pathlib import Path
import argparse,json,sys,math,os,ctypes,platform
from collections import Counter
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'dashboard')]
import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode,flop_registry
from torch.utils._pytree import tree_flatten
from runtime_inferencia import Runtime,read,digest
from evaluar_robustez_congelada import Evaluator,RobustInputs
OUT=ROOT/'results/eficiencia_completa_20261125'
DATES=['2022-01-01','2023-07-01','2025-12-31']

def write(p,obj):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def prod(shape):return math.prod(shape)if isinstance(shape,(list,tuple,torch.Size))else 1

def elem(*args,out_shape=None,**kwargs):return prod(out_shape)
def factor(n):return lambda *args,out_shape=None,**kwargs:n*prod(out_shape)
def reduce_sum(shape,*args,out_shape=None,**kwargs):return max(0,prod(shape)-prod(out_shape))
def reduce_mean(shape,*args,out_shape=None,**kwargs):return prod(shape)
def soft(shape,*args,out_shape=None,**kwargs):
 dim=int(args[0])if args else int(kwargs.get('dim',-1));rows=prod(shape)//shape[dim]
 return 4*prod(shape)-rows

def ln(shape,normalized,*args,out_shape=None,**kwargs):return 7*prod(shape)+2*prod(shape)//prod(normalized)
def gn(shape,weight,bias,N,C,HxW,group,eps,*args,out_shape=None,**kwargs):return 7*prod(shape)+2*N*group

def fft(shape,dim,normalization,onesided,*args,out_shape=None,**kwargs):
 n=prod([shape[d]for d in dim]);return int(2.5*prod(shape)*math.log2(n))

def mha(q,k,v,D,heads,*args,out_shape=None,**kwargs):
 B,T,D=q;S=k[1]
 return 2*B*(T+2*S)*D*D+2*B*T*D*D+4*B*T*S*D+4*B*heads*T*S-B*heads*T+4*B*T*D+B*T*D

def rnn(x,w0,w1,w2,w3,hx,cx,reverse,batch_sizes,mode,H,layers,has_bias,bidir,batch_first,train,*args,out_shape=None,**kwargs):
 assert mode==2, 'Contar explícitamente solo LSTM'
 T,B,F=x
 return int(8*T*B*H*(F+H)+30*T*B*H)

def batchnorm(x,*args,out_shape=None,**kwargs):return 4*prod(x)+2*x[1]
def sdpa(q,k,v,*args,out_shape=None,**kwargs):
 batch=prod(q[:-2]);T=q[-2];S=k[-2];D=q[-1];V=v[-1]
 return 2*batch*T*S*(D+V)+4*batch*T*S-batch*T+batch*T*D

def weightnorm(v,g,*args,out_shape=None,**kwargs):return 4*prod(v)+prod(g)
def average_pool(x,kernel,*args,out_shape=None,**kwargs):return prod(out_shape)*prod(kernel)
def variance(x,*args,out_shape=None,**kwargs):return 4*prod(x)

class DetailedCounter(FlopCounterMode):
 def __init__(self):
  custom={}
  for n in ['add','add_','sub','sub_','mul','mul_','div','div_','true_divide','pow','square','sqrt','rsqrt','reciprocal','neg','abs','exp','log','log1p','tanh','sin','cos','silu','rsub','relu','leaky_relu']:
   if hasattr(torch.ops.aten,n):custom[getattr(torch.ops.aten,n)]=elem
  for n,f in [('sum',reduce_sum),('mean',reduce_mean),('_softmax',soft),('softmax',soft),('native_layer_norm',ln),('native_group_norm',gn),('gelu',factor(5)),('sigmoid',factor(4)),('_fft_r2c',fft),('elu',factor(3)),('_weight_norm_interface',weightnorm),('avg_pool2d',average_pool),('var',variance),('_native_multi_head_attention',mha),('mkldnn_rnn_layer',rnn),('_native_batch_norm_legit_no_training',batchnorm),('_scaled_dot_product_flash_attention_for_cpu',sdpa),('_scaled_dot_product_flash_attention',sdpa),('_scaled_dot_product_efficient_attention',sdpa)]:
   if hasattr(torch.ops.aten,n):custom[getattr(torch.ops.aten,n)]=f
  super().__init__(display=False,custom_mapping=custom);self.unhandled={};self.seen=Counter()
 def _count_flops(self,func_packet,out,args,kwargs):
  name=str(func_packet);self.seen[name]+=1
  vals=tree_flatten((args,kwargs))[0];outs=tree_flatten(out)[0]
  floating=[v for v in vals if torch.is_tensor(v)and(v.is_floating_point()or v.is_complex())]
  if not floating:return out
  if func_packet not in self.flop_registry:
   if floating:self.unhandled[name]=dict(llamadas=self.seen[name],entradas=[list(v.shape)for v in floating[:6]],salidas=[list(v.shape)for v in outs if torch.is_tensor(v)])
  return super()._count_flops(func_packet,out,args,kwargs)

def process_peak():
 if sys.platform=='win32':
  from ctypes import wintypes
  class C(ctypes.Structure):_fields_=[('cb',wintypes.DWORD),('pf',wintypes.DWORD)]+[(n,ctypes.c_size_t)for n in ['peak','rss','p1','p2','p3','p4','p5','p6']]
  c=C();c.cb=ctypes.sizeof(c);handle=ctypes.windll.kernel32.GetCurrentProcess;handle.restype=ctypes.c_void_p
  f=ctypes.windll.psapi.GetProcessMemoryInfo;f.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_ulong]
  assert f(handle(),ctypes.byref(c),c.cb);return int(c.peak)
 for line in Path('/proc/self/status').read_text().splitlines():
  if line.startswith('VmHWM:'):return int(line.split()[1])*1024
 raise RuntimeError('No se puede consultar pico del proceso')

def worker(key):
 torch.set_num_threads(1);torch.set_num_interop_threads(1)
 r=Evaluator.__new__(Evaluator);Runtime.__init__(r);r.inputs.__class__=RobustInputs
 r.files={};r.selected_seed=None;r.red=None;r.aggregator=None
 I=r.inputs;I.configure(DATES[0])
 if '+O'in key or 'modelo7'in key:I.prepare_text()
 if key.startswith('ingenuo_'):
  for date in DATES:I.configure(date)
  write(OUT/'modelos'/f'{key}.json',dict(clave=key,semilla=0,origenes=[dict(origen=d,FLOPs=0,operaciones={},sin_formula={})for d in DATES],pico_proceso_bytes=process_peak(),gpu_disponible=False,torch=torch.__version__,entrenamiento=False,fuentes_sha256={}))
  return
 seed=101 if key=='m4_n'else 1000;r.load_model(key,seed)
 rows=[]
 with torch.inference_mode():
  for date in DATES:
   I.configure(date);before=r.predict_loaded()
   fast=torch.backends.mha.get_fastpath_enabled();torch.backends.mha.set_fastpath_enabled(False)
   try:
    with DetailedCounter()as counter:after=r.predict_loaded()
   finally:torch.backends.mha.set_fastpath_enabled(fast)
   assert np.allclose(before,after,rtol=1e-5,atol=2),key
   rows.append(dict(origen=date,FLOPs=int(counter.get_total_flops()),operaciones={str(k):int(v)for k,v in counter.flop_counts['Global'].items()},sin_formula=counter.unhandled))
 for p,h in r.files.items():assert digest(Path(p))==h
 write(OUT/'modelos'/f'{key.replace("+","_")}.json',dict(clave=key,semilla=seed,origenes=rows,pico_proceso_bytes=process_peak(),gpu_disponible=torch.cuda.is_available(),torch=torch.__version__,entrenamiento=False,fuentes_sha256=r.files))
 print(key,[x['FLOPs']for x in rows],list(set(k for x in rows for k in x['sin_formula'])),flush=True)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--modelo',required=True);worker(a.parse_args().modelo)

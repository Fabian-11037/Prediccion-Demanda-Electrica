"""Auditoría GPU y exportación ONNX de cinco checkpoints M4_N, sin reentrenarlos."""
from __future__ import annotations
import argparse, copy, importlib.metadata, json, platform, sys, time, traceback
from pathlib import Path
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT.parent/'.onnx-m4-deps'))
from e3_modelo4 import Memoria, leer_base, cargar, guardar, sha, torch, np, pd
from m4_onnx_congelado import Encoder, Corrector, Controlador, Q
import onnx, onnxruntime as ort

def sesion(path):
    op=ort.SessionOptions();op.intra_op_num_threads=1;op.inter_op_num_threads=1
    return ort.InferenceSession(str(path),sess_options=op,providers=['CPUExecutionProvider'])

def misma(a,b):
    a,b=np.asarray(a),np.asarray(b)
    return bool(np.isfinite(a).all() and np.isfinite(b).all() and np.all(np.abs(a-b)<=1e-5+1e-5*np.abs(b)))

def original(red,args):
    consulta,k,r,fechas,validas,origen,base=args
    # Los lugares inválidos se representan con una fecha futura. Añadir consulta
    # propia inválida permite utilizar sin cambios la firma histórica recuperar.
    n=len(k);qq=torch.cat([torch.zeros_like(k),consulta]);kk=torch.cat([k,torch.zeros_like(consulta)])
    dd=torch.cat([torch.where(validas,fechas,origen[0]+10000),origen])
    rr=torch.cat([r,torch.zeros((1,24),device=r.device)])
    return red.recuperar(qq,kk,dd,rr,torch.tensor([n],device=r.device),base)

def auditar(red,ck,out):
    n=ck['n_num'];enc=Encoder(red).cuda();cor=Corrector(red).cuda().eval()
    activos=sum(p.numel() for p in enc.parameters())+sum(p.numel() for p in cor.parameters())
    assert activos==64*n+2625==60993
    red.train();torch.manual_seed(20261123);acumulados={k:torch.zeros_like(v) for k,v in red.named_parameters()}
    for lote in range(3):
        red.zero_grad(set_to_none=True);num=torch.randn(40,n,device='cuda');b=SimpleNamespace(num=num)
        q,k=red.codificar(b,40);dia=torch.arange(40,device='cuda')+17000
        residuos=torch.randn(40,24,device='cuda')*.5;base=torch.ones(16,24,device='cuda')*1e7
        pred=red.recuperar(q,k,dia,residuos,torch.arange(24,40,device='cuda'),base)[0]
        ((pred-base*.999).square().mean()/Q**2).backward()
        for nombre,p in red.named_parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(),nombre
                acumulados[nombre]+=p.grad.abs()
    detalle=[]
    for nombre,p in red.named_parameters():
        activo=nombre.startswith(('proy_q.','proy_k.','compuerta.'))
        g=acumulados[nombre]
        if nombre in ['proy_q.0.weight','proy_k.0.weight']:
            assert not g[:,n:].any();g=g[:,:n]
        if activo:assert (g>0).all(),f'Parámetro activo sin gradiente: {nombre}, {(g==0).sum().item()}'
        else:assert not g.any(),nombre
        detalle.append(dict(modulo=nombre,almacenados=p.numel(),activos=g.numel() if activo else 0,gradientes_no_nulos=int((g>0).sum())))
    guardar(out/'parametros_activos.json',dict(almacenados=sum(p.numel() for p in red.parameters()),activos=activos,modulos=detalle))
    red.eval()
    # Todas las dimensiones activas se comprueban en múltiples lotes sin alterar pesos.
    return activos

def entradas(caso,device='cpu'):
    gen=torch.Generator().manual_seed(20261123)
    q=torch.randn(1,32,generator=gen);k=torch.randn(365,32,generator=gen)
    r=torch.randn(365,24,generator=gen)*.3;fechas=20000-torch.arange(365)-2
    validas=torch.ones(365,dtype=torch.bool);base=torch.full((1,24),1e7);origen=torch.tensor([20000])
    if caso=='vacia':validas[:]=False
    if caso=='menos_k':validas[5:]=False
    if caso=='limites':fechas[:6]=torch.tensor([19999,19998,19635,19634,20000,20001])
    if caso=='empates':k[:]=k[0].clone()
    if caso=='padding':validas[20:]=False;k[20:]=1e8;r[20:]=float('nan')
    if caso=='futuro':fechas[100:]=20001;k[100:]=1e8;r[100:]=-1e9
    return tuple(x.to(device) for x in (q,k,r,fechas,validas,origen,base))

def ejecutar(out):
    out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
    assert torch.cuda.is_available(),'GPU obligatoria para la referencia PyTorch'
    m4=ROOT/'results/m4_gpu_20261122_194500';m3=ROOT/'results/m3_gpu_20261122_193240'
    fuentes=[Path(__file__),ROOT/'scripts/m4_onnx_congelado.py',ROOT/'scripts/e3_modelo4.py']
    pesos=[m4/f'final/M4_N/semilla_{s}/modelo.pt' for s in range(101,106)]
    registro=dict(fecha=pd.Timestamp.now(tz='America/Bogota').isoformat(),alcance='corrector ONNX con M1 externa; auditoría de pesos congelados',
        semillas=list(range(101,106)),opset=18,atol=1e-5,rtol=1e-5,diferencia_MASE_maxima=1e-5,
        casos=['vacia','menos_k','normal','limites','empates','padding','futuro'],recorrido=['desarrollo2021','prueba_retrospectiva2022-2025'],
        banco=365,lote=1,precision='FP32',versiones={k:importlib.metadata.version(k) for k in ['torch','onnx','onnxruntime','onnxscript','onnx_ir']},
        sistema=platform.platform(),gpu=torch.cuda.get_device_name(0),proveedores_disponibles=ort.get_available_providers(),
        Windows='no disponible en este PC; pendiente de ejecución real',onnx_GPU='no validado; referencia PyTorch ROCm y ONNX Runtime CPU Linux',
        hashes={str(p.relative_to(ROOT)):sha(p) for p in fuentes+pesos})
    guardar(out/'REGISTRO_PREVIO.json',registro)
    # Validar prefijos con resultados solo de años anteriores.
    pref=[]
    for a in range(2013,2022):
        d=json.loads((m3/f'prefijos/seleccion_{a}.json').read_text());u=d['ultimo_anio_seleccion']
        assert u is None if a==2013 else u<a
        pref.append(d)
    guardar(out/'auditoria_prefijos.json',dict(verificado=True,decisiones=pref))
    resultados=[];filas=[];fallos=[]
    for s,peso in zip(range(101,106),pesos):
        guardar(out/'estado.json',dict(estado='Ejecutando',semilla=s,fase='auditoria_exportacion'))
        car=out/f'semilla_{s}';car.mkdir(exist_ok=True)
        ck=cargar(peso);red=Memoria(ck['n_num'],ck['n_feat'],ck['n_fuentes'],ck['configuracion'],'M4_N').cuda()
        red.load_state_dict(ck['modelo']);red.eval();auditar(red,ck,car)
        enc=Encoder(red).cpu().eval();cor=Corrector(red).cpu().eval()
        for nombre,mod,args,nombres,salidas in [
            ('m4_n_encoder',enc,(torch.zeros(1,912),),['num'],['consulta','clave']),
            ('m4_n_corrector',cor,entradas('normal'),['consulta','claves','residuos','fechas','validas','origen','base'],['pred','compuerta','indices','atencion','elegibles'])]:
            torch.onnx.export(mod,args,str(car/f'{nombre}.onnx'),input_names=nombres,output_names=salidas,opset_version=18,dynamo=True,report=True,verify=True,artifacts_dir=str(car),external_data=False)
            onnx.checker.check_model(onnx.load(str(car/f'{nombre}.onnx')))
        es,cs=sesion(car/'m4_n_encoder.onnx'),sesion(car/'m4_n_corrector.onnx')
        for caso in registro['casos']:
            args=entradas(caso);ag=tuple(x.cuda() for x in args)
            with torch.no_grad():ref=original(red,ag);py=cor(*args)
            oo=cs.run(None,{n:x.numpy() for n,x in zip(['consulta','claves','residuos','fechas','validas','origen','base'],args)})
            # Empates pueden revelar distintas reglas legítimas de top-k; se rechaza
            # la equivalencia si modifican la salida, sin perturbar puntajes.
            ok=misma(oo[0]/Q,ref[0].cpu().numpy()/Q) and misma(oo[1],ref[1].cpu().numpy()) and misma(oo[3],ref[3].cpu().numpy())
            wrapper=misma(py[0].numpy()/Q,ref[0].cpu().numpy()/Q)
            if caso=='vacia':ok=ok and np.array_equal(oo[0],args[-1].numpy()) and not oo[1].any() and not oo[3].any()
            filas.append(dict(semilla=s,caso=caso,equivalente_ONNX_GPU=ok,wrapper_equivalente=wrapper,max_error_q=float(np.nanmax(np.abs(oo[0]-ref[0].cpu().numpy()))/Q)))
            if not ok or not wrapper:fallos.append(dict(semilla=s,caso=caso))
        # Recorre fechas completas con el banco cronológico y revela residuos d+2.
        hist=leer_base(m3,s)
        with np.load(m3/f'base_final/semilla_{s}.npz') as z:test={k:z[k] for k in z.files}
        meta={k:np.concatenate([hist[k],test[k]]) for k in test}
        num=((meta['num']-ck['media'])/ck['escala']).astype('float32');dias=np.asarray(meta['dia'],dtype='datetime64[D]').astype('int64')
        with torch.no_grad():
            banco=SimpleNamespace(num=torch.as_tensor(num,device='cuda'))
            qg,kg=red.codificar(banco,len(num))
            qcpu,kcpu=enc(torch.from_numpy(num))
        qn,kn=[],[]
        for x in num:
            a,b=es.run(None,{'num':x[None]});qn.append(a[0]);kn.append(b[0])
        qn,kn=np.asarray(qn),np.asarray(kn)
        assert misma(qn,qg.cpu().numpy()) and misma(kn,kg.cpu().numpy()),'Encoder fuera de tolerancia'
        controlador=Controlador();preds=[];referencias=[];ix=[];errores=[];tiempos=[]
        rg=torch.as_tensor((meta['y']-meta['pred_base'])/Q,device='cuda');dg=torch.as_tensor(dias,device='cuda')
        with torch.no_grad():
            for i,d in enumerate(dias):
                if str(meta['dia'][i])>='2021-01-01':
                    keys,res,fechas,validas=controlador.banco(int(d))
                    inp=dict(consulta=qn[i:i+1],claves=keys,residuos=res,fechas=fechas,validas=validas,origen=np.array([d],dtype='int64'),base=meta['pred_base'][i:i+1])
                    t=time.perf_counter();oo=cs.run(None,inp);tiempos.append(time.perf_counter()-t)
                    ref=red.recuperar(qg,kg,dg,rg,torch.tensor([i],device='cuda'),torch.as_tensor(inp['base'],device='cuda'))[0].cpu().numpy()
                    if not misma(oo[0]/Q,ref/Q):errores.append(dict(dia=str(meta['dia'][i]),max_error_q=float(np.max(np.abs(oo[0]-ref))/Q)))
                    preds.append(oo[0][0]);referencias.append(ref[0]);ix.append(i)
                controlador.insertar(int(d),kn[i],((meta['y'][i]-meta['pred_base'][i])/Q).astype('float32'),int(d)+2)
        preds=np.asarray(preds);refs=np.asarray(referencias);y=meta['y'][ix]
        mase=float(np.abs(y.astype('float64')-preds).mean()/Q);mref=float(np.abs(y.astype('float64')-refs).mean()/Q)
        assert np.isfinite(preds).all()
        if abs(mase-mref)>1e-5 or errores:fallos.append(dict(semilla=s,caso='recorrido',fallos=len(errores),diferencia_MASE=abs(mase-mref)))
        np.savez_compressed(car/'recorrido.npz',dia=meta['dia'][ix],y=y,ONNX=preds,PyTorch_GPU=refs)
        guardar(car/'recorrido.json',dict(origenes=len(ix),MASE_ONNX=mase,MASE_PyTorch_GPU=mref,diferencia_MASE=abs(mase-mref),fallos=errores,latencia_ONNX_corrector_ms_mediana=float(np.median(tiempos[20:])*1000),latencia_ms_p95=float(np.percentile(tiempos[20:],95)*1000),base_externa=True))
        resultados.append(dict(semilla=s,origenes=len(ix),MASE_ONNX=mase,MASE_PyTorch_GPU=mref))
        assert sha(peso)==registro['hashes'][str(peso.relative_to(ROOT))]
        del red,enc,cor,qg,kg,rg,dg;torch.cuda.empty_cache()
    pd.DataFrame(filas).to_csv(out/'casos_equivalencia.csv',index=False)
    guardar(out/'RESULTADO.json',dict(auditoria_GPU='aprobada',equivalencia_Linux='rechazada' if fallos else 'aprobada',fallos=fallos,recorridos=resultados,Windows='pendiente',ONNX_GPU='sin proveedor validado',paquete='corrector con M1 externa; no sistema completo',historicos_intactos=True))
    guardar(out/'estado.json',dict(estado='Finalizado',fallos_equivalencia=len(fallos)))
    guardar(out/'MANIFIESTO.json',[dict(archivo=str(p.relative_to(out)),sha256=sha(p),bytes=p.stat().st_size) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='MANIFIESTO.json'])

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);a=parser.parse_args()
    try:ejecutar(a.out)
    except Exception:
        a.out.mkdir(parents=True,exist_ok=True);guardar(a.out/'ERROR.json',dict(error=traceback.format_exc()));raise

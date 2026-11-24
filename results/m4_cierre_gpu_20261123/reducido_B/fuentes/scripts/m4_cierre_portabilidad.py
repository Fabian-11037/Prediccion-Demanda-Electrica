"""Audita diez semillas y exporta subgrafos con pesos congelados. No corrige resultados."""
from e3_gpu_comun import *
import argparse,platform,shutil,traceback
sys.path.insert(0,str(REPO.parent/'.onnx-m4-deps'))
import onnx,onnxruntime as ort
from m4_onnx_congelado import Encoder,Corrector,Q,Controlador
from pruebas_m4_congelado import entradas,original,misma,sesion
from m4_cierre_verificar_paquete import verificar
from e3_modelo1 import Router
import m4_cierre_colombia as COL
from m4_cierre_procedencia import desde_npz


def exportar(red,args,p,nombres,salidas):
    torch.onnx.export(red,args,str(p),input_names=nombres,output_names=salidas,opset_version=18,dynamo=True,verify=True,report=True,external_data=False,artifacts_dir=str(p.parent))
    onnx.checker.check_model(onnx.load(str(p)))


def caso_guardar(p,args,nombres,ref,salidas):
    d={'in_'+k:x.detach().cpu().numpy() for k,x in zip(nombres,args)}
    d.update({'ref_'+k:x.detach().cpu().numpy() for k,x in zip(salidas,ref) if k not in ['indices','elegibles']})
    np.savez_compressed(p,**d)


def ejecutar(out):
    car=out/'complemento/portabilidad';car.mkdir(parents=True,exist_ok=True);grafos=[];rechazos=[];empates=[];costes=[]
    torch.set_num_threads(4);assert torch.cuda.is_available()
    datos=DatosNumericos(PAQUETE);nombres=['consulta','claves','residuos','fechas','validas','origen','base'];salidas=['pred','compuerta','indices','atencion','elegibles']
    for s in range(101,111):
        guardar(out/'progreso/estado.json',dict(estado='Ejecutando',fase='Portabilidad y auditoría congelada',tareas=s-101,total=10,gpu=torch.cuda.get_device_name(0),etiqueta=f'Semilla{s}',actualizado=pd.Timestamp.now(tz='America/Bogota').isoformat()))
        d=car/f'semilla_{s}';d.mkdir(exist_ok=True)
        peso=out/f'colombia/final/M4_N/semilla_{s}/modelo.pt';h=sha(peso);ck=cargar(peso)
        from e3_modelo4 import Memoria
        red=Memoria(ck['n_num'],ck['n_feat'],ck['n_fuentes'],ck['configuracion'],'M4_N').cuda().eval()
        red.load_state_dict(ck['modelo']);enc=Encoder(red).cpu().eval();cor=Corrector(red).cpu().eval()
        try:
            exportar(enc,(torch.zeros(1,ck['n_num']),),d/'encoder.onnx',['num'],['consulta','clave'])
            exportar(cor,entradas('normal'),d/'corrector.onnx',nombres,salidas)
        except Exception:
            rechazos.append(dict(semilla=s,grafo='corrector',error=traceback.format_exc()));continue
        casos=[];enc_casos=[]
        for nombre in ['vacia','menos_k','normal','limites','empates','padding','futuro']:
            args=entradas(nombre);ag=tuple(x.cuda() for x in args)
            with torch.no_grad():
                ref=original(red,ag);pc=cor(*args);pg=cor.cuda()(*ag);cor.cpu()
            p=d/f'caso_{nombre}.npz';caso_guardar(p,args,nombres,ref,salidas);casos.append(str(p.relative_to(car)))
            empates.append(dict(semilla=s,caso=nombre,wrapper_CPU_vs_original_GPU=misma(pc[0].numpy()/Q,ref[0].cpu().numpy()/Q),wrapper_GPU_vs_original_GPU=misma(pg[0].cpu().numpy()/Q,ref[0].cpu().numpy()/Q),indices_original=ref[2].cpu().tolist(),indices_wrapper_CPU=pc[2].tolist(),indices_wrapper_GPU=pg[2].cpu().tolist()))
        # Recorrido real completo: se mantienen retriever histórico y pesos originales.
        if s<=105:
            hist=COL.leer_base(COL.M3,s);prueba=desde_npz(COL.M3/f'base_final/semilla_{s}.npz')
        else:
            hist=desde_npz(out/f'colombia/base_oof/semilla_{s}.npz');prueba=desde_npz(out/f'colombia/base_final/semilla_{s}.npz')
        meta={k:np.concatenate([hist[k],prueba[k]]) for k in ['dia','y','pred_base','num']};num=((meta['num']-ck['media'])/ck['escala']).astype('float32');dias=np.asarray(meta['dia'],dtype='datetime64[D]').astype('int64')
        from types import SimpleNamespace
        with torch.no_grad():qg,kg=red.codificar(SimpleNamespace(num=torch.as_tensor(num,device='cuda')),len(num))
        es,cs=sesion(d/'encoder.onnx'),sesion(d/'corrector.onnx');qs=[];ks=[]
        for x in num:
            a,b=es.run(None,{'num':x[None]});qs.append(a[0]);ks.append(b[0])
        qs,ks=np.asarray(qs),np.asarray(ks);control=Controlador();pp=[];rr=[];jj=[];lat=[];fallos=0
        rg=torch.as_tensor((meta['y']-meta['pred_base'])/Q,device='cuda');dg=torch.as_tensor(dias,device='cuda')
        with torch.no_grad():
            for i,dia in enumerate(dias):
                if meta['dia'][i]>='2021-01-01':
                    keys,res,fechas,validas=control.banco(int(dia));inp=dict(consulta=qs[i:i+1],claves=keys,residuos=res,fechas=fechas,validas=validas,origen=np.array([dia],dtype='int64'),base=meta['pred_base'][i:i+1]);t=time.perf_counter();oo=cs.run(None,inp);lat.append(time.perf_counter()-t)
                    ref=red.recuperar(qg,kg,dg,rg,torch.tensor([i],device='cuda'),torch.as_tensor(inp['base'],device='cuda'))[0].cpu().numpy()
                    fallos+=int(not misma(oo[0]/Q,ref/Q));pp.append(oo[0][0]);rr.append(ref[0]);jj.append(i)
                control.insertar(int(dia),ks[i],((meta['y'][i]-meta['pred_base'][i])/Q).astype('float32'),int(dia)+2)
        pp,rr=np.asarray(pp),np.asarray(rr);delta=abs(float(np.abs(pp.astype('float64')-meta['y'][jj]).mean()/Q)-float(np.abs(rr.astype('float64')-meta['y'][jj]).mean()/Q))
        guardar(d/'RECORRIDO.json',dict(origenes=len(jj),fallos=fallos,diferencia_MASE=delta,aprobado=fallos==0 and delta<=1e-5,base_externa=True));np.savez_compressed(d/'recorrido.npz',dia=meta['dia'][jj],ONNX=pp,PyTorch_GPU=rr)
        costes.append(dict(semilla=s,memoria_pesos_GPU_bytes=sum(p.numel()*p.element_size() for p in red.parameters()),memoria_banco_maximo_bytes=365*(32*4+24*4+8+1),cache_embeddings_historicos_bytes=int(ks.nbytes),latencia_corrector_CPU_ms_mediana=float(np.median(lat[20:])*1000),latencia_p95_ms=float(np.percentile(lat[20:],95)*1000)))
        for i in np.linspace(0,len(num)-1,24,dtype=int):
            arg=(torch.from_numpy(num[i:i+1]),)
            with torch.no_grad():ref=Encoder(red).cuda().eval()(arg[0].cuda())
            p=d/f'encoder_{i}.npz';caso_guardar(p,arg,['num'],ref,['consulta','clave']);enc_casos.append(str(p.relative_to(car)))
        for nombre,cas,outs,escala in [('encoder',enc_casos,['consulta','clave'],1.),('corrector',casos,salidas,Q)]:grafos.append(dict(archivo=str((d/f'{nombre}.onnx').relative_to(car)),sha256=sha(d/f'{nombre}.onnx'),casos=cas,salidas=outs,escala=escala))
        # Intentar subgrafos de toda la base, sin cambiar operadores ni congelar
        # decisiones dependientes del dato. Un fallo de exportación se conserva.
        m1=COL.HIST if s<=105 else out/'colombia/M1';pc=m1/f'final/semilla_{s}/modelo.pt'
        rc=cargar(pc);router=Router(rc['n_entrada'],rc['configuracion']).eval();router.load_state_dict(rc['modelo'])
        rawnum=(hist['num'][-24:]-rc['media'])/rc['escala'];expertos=hist['expertos'][-24:]
        mods=[('router',router,(torch.from_numpy(rawnum[:1]),torch.from_numpy(expertos[:1])),['num','expertos'],['pred','complementariedad','pesos'])]
        pos=datos.indice.tz_localize(None).get_indexer(pd.DatetimeIndex(hist['dia'][-24:]))
        for nombre in ['patchtst','E','periodica']:
            ep=cargar(m1/f'expertos_finales/{nombre}/semilla_{s}/modelo.pt');cfg=ep.get('configuracion',ep.get('metadatos',{}).get('configuracion'));net=construir_experto(datos,nombre,cfg).eval();net.load_state_dict(ep['modelo']);c=datos.construir(pos,ep['escalas']);tt=motor.a_tensores(c,'cpu');mods.append((nombre,net,tuple(tt[k][:1] for k in ['x','cal_p','cal_f','dia']),['x','cal_p','cal_f','dia'],['pred']))
        for nombre,mod,args,ins,outs in mods:
            try:
                exportar(mod,args,d/f'{nombre}.onnx',ins,outs)
                with torch.no_grad():ref=mod.cuda()(*[a.cuda() for a in args]);mod.cpu()
                if not isinstance(ref,tuple):ref=(ref,)
                p=d/f'base_{nombre}.npz';caso_guardar(p,args,ins,ref,outs);grafos.append(dict(archivo=str((d/f'{nombre}.onnx').relative_to(car)),sha256=sha(d/f'{nombre}.onnx'),casos=[str(p.relative_to(car))],salidas=outs,escala=1.))
            except Exception:rechazos.append(dict(semilla=s,grafo=nombre,error=traceback.format_exc()))
        assert sha(peso)==h;del red,enc,cor,qg,kg;torch.cuda.empty_cache()
    guardar(car/'PAQUETE.json',dict(grafos=grafos,opset=18,atol=1e-5,rtol=1e-5,base='Subgrafos individuales; preprocesamiento, memoria y controlador cronológico externos. No paquete completo validado.',no_cambio_reglas='Pesos, top-k y tolerancias originales; empates discrepantes se rechazan.'))
    shutil.copy2(REPO/'scripts/m4_cierre_verificar_paquete.py',car/'verificar_paquete.py');shutil.copy2(REPO/'scripts/m4_onnx_congelado.py',car/'controlador_referencia.py')
    cpu=verificar(car,'CPUExecutionProvider');gpu=[]
    for provider in ort.get_available_providers():
        if provider in ['CUDAExecutionProvider','ROCMExecutionProvider','MIGraphXExecutionProvider','DmlExecutionProvider']:
            try:gpu.append(verificar(car,provider))
            except Exception:rechazos.append(dict(proveedor=provider,error=traceback.format_exc()))
    pd.DataFrame(empates).to_csv(car/'auditoria_empates_dispositivos.csv',index=False);pd.DataFrame(costes).to_csv(car/'costes_inferencia.csv',index=False)
    guardar(car/'ESTADO_PORTABILIDAD.json',dict(Linux_CPU_casos=cpu['aprobado'],Windows='pendiente: requiere ejecución nativa del verificador; no se simula con Wine',ONNX_GPU='ejecutado' if gpu else 'pendiente: no hay proveedor GPU compatible ejecutable',rechazos=rechazos,empates_wrapper_equivalente=all(f['wrapper_CPU_vs_original_GPU'] for f in empates),base_completa_validada=False,certificacion=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);ejecutar(p.parse_args().out)

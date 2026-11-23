"""Cola persistente de evaluación adicional. Publica antes de habilitar apagado."""
from __future__ import annotations
import argparse, fcntl, shutil, subprocess, traceback
from e3_gpu_comun import *
import m4_cierre_colombia as COL
import m4_cierre_datos as EXT
import m4_cierre_controles as CONT
import m4_cierre_benchmark as BEN
import m4_cierre_recuperacion as REC
import m4_cierre_fuertes as FU
from e3_modelo1 import dividir
from m4_cierre_estadistica import comparar

SEMILLAS=list(range(101,111))

def particiones(meta,nombre,anio_prueba):
    if nombre=='colombia':
        return [(str(a),*dividir(meta,a)) for a in [2018,2019,2020,2021,2022]]
    fechas=pd.DatetimeIndex(meta['dia']);partes=[]
    for inicio in pd.date_range(f'{anio_prueba-1}-01-01',f'{anio_prueba}-01-01',freq='QS'):
        corte=inicio-pd.DateOffset(months=3);a=np.flatnonzero(fechas<corte-pd.Timedelta(days=1));es=np.flatnonzero((fechas>=corte)&(fechas<inicio));ev=np.flatnonzero((fechas>=inicio)&(fechas<inicio+pd.DateOffset(months=3)))
        if len(a)>=30 and len(es)>=30 and (len(ev) or inicio.year==anio_prueba):partes.append((inicio.strftime('%Y-%m'),a,es,ev))
    assert partes and not len(partes[-1][-1]);return partes

def guardar_pred(car,nombre,s,dias,y,pred):
    car.mkdir(parents=True,exist_ok=True);assert pred.shape==y.shape and np.isfinite(pred).all()
    np.savez_compressed(car/f'{nombre}_semilla_{s}.npz',dia=dias,y=y,pred=pred)
    pd.DataFrame(dict(fecha=np.repeat(dias,24),hora=np.tile(np.arange(24),len(dias)),semilla=s,observado=y.ravel(),predicho=pred.ravel())).to_csv(car/f'{nombre}_semilla_{s}.csv.gz',index=False,compression='gzip')

def traza(meta,ix,ck,car):
    banco=CONT.preparar_banco(meta,ck['media'],ck['escala'],ck['q']);red=CONT.Memoria(ck['n_num'],ck['n_feat'],ck['n_fuentes'],CONT.CFG,'M4_N').cuda().eval();red.load_state_dict(ck['modelo']);red.q_dataset=ck['q'];red.recuperar=CONT.recuperar_q.__get__(red,CONT.Memoria)
    filas=[];gate=[];alfa=[];validas=[]
    with torch.no_grad():
        q,k=red.codificar(banco,len(meta['dia']))
        for jj in np.array_split(ix,max(1,int(np.ceil(len(ix)/128)))):
            j=torch.as_tensor(jj,device='cuda');_,g,ids,a,ok=red.recuperar(q,k,banco.dias,banco.residuos,j,banco.base[j]);filas.append(ids.cpu().numpy());gate.append(g.cpu().numpy());alfa.append(a.cpu().numpy());validas.append(ok.cpu().numpy())
    ids=np.concatenate(filas);ok=np.concatenate(validas);dd=np.asarray(meta['dia'],dtype='datetime64[D]');edad=(dd[ix,None]-dd[ids]).astype('timedelta64[D]').astype(int)
    assert ((~ok)|((edad>=2)&(edad<=365))).all()
    np.savez_compressed(car,dia=meta['dia'][ix],ids=ids,fechas=meta['dia'][ids],elegibles=ok,compuerta=np.concatenate(gate),atencion=np.concatenate(alfa),edad=edad)
    del banco,red,q,k;torch.cuda.empty_cache()

def ejecutar(out,publicar,apagar):
    out.mkdir(parents=True,exist_ok=True);lock=(out/'cierre.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert (out/'FASE_CAPACIDAD_TERMINADA.json').exists(),'Primero terminar fase de capacidad101–105'
    assert not (out/'ERROR_CAPACIDAD.json').exists()
    torch.set_num_threads(4);assert torch.cuda.is_available();motor.fijar_semilla(101)
    fuentes=list((REPO/'scripts').glob('m4_cierre_*.py'))+[Path(__file__),REPO/'experiments/estandar.md',REPO/'scripts/publicar_cierre_m4.py']
    fuentes += [REPO/'scripts'/n for n in ['e3_gpu_comun.py','e3_modelo1.py','e3_modelo3.py','e3_modelo4.py','probar_m1_gpu.py']]
    huellas={str(p.relative_to(REPO)):sha(p) for p in fuentes};registro=out/'REGISTRO_COLA_COMPLETA.json'
    entradas=[REPO/f'data/external/m4_cierre_20261123/{n}.parquet' for n in ['uci','gefcom','opsd']]+[PAQUETE/'data/processed/panel_horario.parquet']
    entradas += list(COL.M3.glob('base_oof/*.npz'))+list(COL.M3.glob('base_final/*.npz'))+list(COL.M3.glob('prefijos/*.json'))
    entradas += list((REPO/'results/m4_gpu_20261122_194500/final/M4_N').glob('semilla_*/modelo.pt'))
    insumos={str(p):sha(p) for p in entradas}
    paquete_huellas={str(p):sha(p) for p in (PAQUETE/'src').rglob('*.py')}
    obj=dict(fecha=pd.Timestamp.now(tz='America/Bogota').isoformat(),fuentes=huellas,paquete_fuentes=paquete_huellas,entradas=insumos,semillas=SEMILLAS,datasets=['colombia','uci','gefcom','opsd'],publicar=publicar,apagar=apagar,posterior_a_prueba_colombia=True)
    if registro.exists():assert leer(registro)['fuentes']==huellas
    else:guardar(registro,obj)
    for p in fuentes:
        q=out/'fuentes'/p.relative_to(REPO);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
    shutil.copytree(PAQUETE/'src',out/'fuentes/paquete_corrector/src',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
    fase='Preparación';tareas=0;total=10
    def progreso(**kw):
        guardar(out/'progreso/estado.json',dict(estado='Ejecutando',fase=fase,tareas=tareas,total=total,gpu=torch.cuda.get_device_name(0),actualizado=pd.Timestamp.now(tz='America/Bogota').isoformat(),**kw))
    resultados={};cfgs=BEN.configuraciones();guardar(out/'CONFIGURACIONES_BENCHMARK.json',cfgs)
    for nombre in ['colombia','uci','gefcom','opsd']:
        car=out/nombre;car.mkdir(exist_ok=True);fase=f'{nombre}: OOF y ampliación independiente';tareas=0;total=10;progreso()
        datos=DatosNumericos(PAQUETE) if nombre=='colombia' else EXT.DatosAdicionales(nombre)
        anio=2022 if nombre=='colombia' else datos.test_anio;metas={}
        for s in SEMILLAS:
            progreso(etiqueta=f'Base OOF y expertos finales · semilla{s}')
            meta=COL.preparar(s,car,progreso) if nombre=='colombia' else EXT.final_adicional(datos,s,car,progreso)
            metas[s]=meta;tareas+=1;progreso()
        fase=f'{nombre}: controles y ajuste final';tareas=0;total=sum(len(particiones(metas[s],nombre,anio))*len(CONT.VARIANTES)+2+10 for s in SEMILLAS);progreso()
        for s in SEMILLAS:
            meta=metas[s];partes=particiones(meta,nombre,anio)
            for etiqueta,a,es,ev in partes:
                final=not len(ev);base=car/('final' if final else f'desarrollo/anio_{etiqueta}')
                # Colombia101–105 ya está registrado y ejecutado en fase anterior.
                for v in CONT.VARIANTES:
                    previo=None
                    if v=='M4_N' and nombre=='colombia' and s<=105:
                        viejo=REPO/'results/m4_gpu_20261122_194500';previo=viejo/('final/M4_N' if final else f'controles/M4_N/anio_{etiqueta}')/f'semilla_{s}'/'modelo.pt'
                    progreso(etiqueta=f'{v} · {etiqueta} · s{s}')
                    CONT.ajustar_control(meta,s,a,es,ev,v,base/v/f'semilla_{s}',datos.q168,progreso,previo);tareas+=1
            _,a,es,_=partes[-1];w=REC.ventanas(datos,meta['dia'])
            for v in REC.NOMBRES:
                REC.ajustar_recuperacion(meta,w,a,es,s,v,car/'final'/v/f'semilla_{s}',datos.q168,progreso);tareas+=1
            FU.ajustar(meta,a,es,car/f'fuertes/semilla_{s}',datos.q168)
            if nombre=='colombia':tr,st=datos.ajuste_final()
            else:
                inicio=pd.Timestamp(f'{anio}-01-01',tz='UTC');corte=inicio-pd.DateOffset(months=2)
                tr=datos.pos[datos.indice[datos.pos]<corte-pd.Timedelta(days=1)];st=datos.pos[(datos.indice[datos.pos]>=corte)&(datos.indice[datos.pos]<inicio)]
            for v in BEN.MODELOS:
                BEN.ajustar_benchmark(datos,v,cfgs[v],s,tr,st,car/'benchmark'/v/f'semilla_{s}',progreso);tareas+=1;progreso()
        pesos=list((car/'final').rglob('modelo.pt'))+list((car/'benchmark').rglob('modelo.pt'))
        assert len(pesos)==190,(nombre,len(pesos))
        congelados={str(p.relative_to(car)):sha(p) for p in pesos};guardar(car/'CONGELACION_PREPRUEBA.json',dict(fecha=pd.Timestamp.now(tz='America/Bogota').isoformat(),huellas=congelados,semillas=SEMILLAS))
        fase=f'{nombre}: prueba cronológica congelada';tareas=0;total=10;progreso();preds={};dest=car/'prueba';dest.mkdir(exist_ok=True)
        pos=datos.prueba() if nombre=='colombia' else datos.pos[datos.indice[datos.pos].year==anio]
        for s in SEMILLAS:
            progreso(etiqueta=f'Inferencia sin ajuste · s{s}')
            meta=COL.prueba(s,car,metas[s]) if nombre=='colombia' else EXT.prueba_adicional(datos,s,car,metas[s])
            ix=np.flatnonzero(meta['anio']>=anio);dias=meta['dia'][ix];y=datos.Y[pos[:,None]+np.arange(24)]
            assert np.array_equal(y.astype('float32'),meta['y'][ix]);assert np.array_equal(dias,datos.indice[pos].strftime('%Y-%m-%d').to_numpy().astype('U10'))
            pr=dict(M1=meta['pred_base'][ix])
            for v in CONT.VARIANTES:
                ck=cargar(car/'final'/v/f'semilla_{s}'/'modelo.pt');pr[v]=CONT.predecir(ck,meta,ix)
                if v=='M4_N':traza(meta,ix,ck,dest/f'memoria_semilla_{s}.npz')
            w=REC.ventanas(datos,meta['dia'])
            for v in REC.NOMBRES:pr[v]=REC.predecir_recuperacion(meta,w,ix,cargar(car/'final'/v/f'semilla_{s}'/'modelo.pt'),datos.q168)
            pr.update(FU.predecir(meta,ix,car/f'fuertes/semilla_{s}',datos.q168))
            for v in BEN.MODELOS:pr['benchmark_'+v]=BEN.prueba_benchmark(datos,v,s,pos,car/'benchmark'/v/f'semilla_{s}')
            for v,p in pr.items():
                guardar_pred(dest,v,s,dias,y,p);preds.setdefault(v,[]).append(p)
            tareas+=1;progreso()
        for v,p in datos.ingenuos(pos).items():
            preds[v]=[p];guardar_pred(dest,v,0,dias,y,p)
        assert all(sha(car/p)==h for p,h in congelados.items()),'Un checkpoint cambió durante prueba'
        resultados[nombre]=(dias,y,preds,datos.q168)
        guardar(car/'FINALIZADO.json',dict(fecha=pd.Timestamp.now(tz='America/Bogota').isoformat(),semillas=SEMILLAS,prueba=True,modelos=list(preds)))
        del datos,metas;torch.cuda.empty_cache()
    fase='Inferencia estadística e informe';tareas=0;total=1;progreso();estad=out/'estadistica';estad.mkdir(exist_ok=True)
    resumen=comparar(resultados,estad)
    lineas=['# Evaluación adicional de M4_N','','Diez semillas independientes101–110. Colombia es retrospectiva; UCI,GEFCom2014-L y AlemaniaOPSD son los conjuntos adicionales fijados antes de métricas. La búsqueda amplia permanece cerrada.','','| Dataset | Procedimiento | MASE | Semillas |','|---|---|---:|---:|']
    for r in resumen:lineas.append(f'| {r["dataset"]} | {r["modelo"]} | {r["MASE"]:.6f} | {r["semillas"]} |')
    lineas+=['','La evaluación no certifica originalidad ni potencia0,90. La revisión dirigida identifica antecedentes directos. La propiedad es local con banco/top-k fijos. ONNX conserva el rechazo observado en empates y las plataformas no ejecutadas quedan pendientes. El benchmark numérico adicional no sustituye al Top10 histórico con texto. Consultar protocolos, familias estadísticas, controles de capacidad y artefactos por semilla.','']
    (out/'INFORME.md').write_text('\n'.join(lineas),encoding='utf-8')
    assert all(sha(REPO/p)==h for p,h in huellas.items())
    assert all(sha(p)==h for p,h in insumos.items())
    assert all(sha(p)==h for p,h in paquete_huellas.items())
    guardar(out/'FINALIZADO.json',dict(fecha=pd.Timestamp.now(tz='America/Bogota').isoformat(),entrenamiento=True,prueba=True,datasets=list(resultados),semillas=SEMILLAS,certificacion_E3=False))
    guardar(out/'MANIFIESTO_FINAL.json',[dict(archivo=str(p.relative_to(out)),sha256=sha(p),bytes=p.stat().st_size) for p in sorted(out.rglob('*')) if p.is_file() and not any(x in p.parts for x in ['progreso','__pycache__']) and p.name not in ['MANIFIESTO_FINAL.json','cierre.lock'] and p.suffix not in ['.log','.lock']])
    if publicar:
        fase='Publicación en GitHub';progreso()
        subprocess.run([sys.executable,str(REPO/'scripts/publicar_cierre_m4.py'),'--out',str(out)],check=True)
        recibo=leer(out/'PUBLICADO.json');assert recibo['verificado']
    if apagar:
        assert publicar and leer(out/'PUBLICADO.json')['verificado']
        fase='Publicación verificada; apagado en dos minutos';tareas=1;progreso()
        r=subprocess.run(['shutdown','-h','+2'],capture_output=True,text=True)
        guardar(out/'APAGADO.json',dict(retorno=r.returncode,salida=r.stdout,error=r.stderr));assert r.returncode==0

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--publicar',action='store_true');ap.add_argument('--apagar',action='store_true');args=ap.parse_args()
    try:ejecutar(args.out.resolve(),args.publicar,args.apagar)
    except Exception:
        guardar(args.out/'ERROR_COLA_COMPLETA.json',dict(fecha=pd.Timestamp.now(tz='America/Bogota').isoformat(),error=traceback.format_exc(),apagado=False));raise

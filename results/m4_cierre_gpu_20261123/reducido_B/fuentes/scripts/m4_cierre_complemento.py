"""Cola complementaria registrada: desarrollo, dimensionamiento, ajustes y prueba."""
from e3_gpu_comun import *
import argparse,shutil,traceback,subprocess
import m4_cierre_colombia as COL
import m4_cierre_datos as EXT
import m4_cierre_controles as C
import m4_cierre_benchmark as BEN
import m4_cierre_recuperacion as REC
import m4_cierre_fuertes as FU
import m4_cierre_procedencia as PRO
import m4_cierre_diagnosticos as DIA
import e3_modelo4 as TEX
from ejecutar_cierre_m4 import particiones,guardar_pred,traza

TEXTOS=['M4_O','M4_OP','M4_metadatos']+[f'M4_desalineado_{d}' for d in [49,91,154,217,301]]+['M4_OP_estricto','M4_sin_num']
DATASETS=['colombia','uci','gefcom','opsd']

def cargar_meta(nombre,s,out,progreso):
    work=out/nombre if s<=110 else out/'complemento'/nombre/'ampliacion_base'
    datos=DatosNumericos(PAQUETE) if nombre=='colombia' else EXT.DatosAdicionales(nombre)
    datos.nombre=nombre
    m=COL.preparar(s,work,progreso) if nombre=='colombia' else EXT.final_adicional(datos,s,work,progreso)
    return datos,m,work

def posiciones(datos,dias):
    p=datos.indice.tz_localize(None).get_indexer(pd.DatetimeIndex(dias));assert (p>=datos.L).all();return p

def texto_pred(meta,textos,ck,ix,q):
    red=TEX.construir(ck);red.load_state_dict(ck['modelo']);red.eval()
    b=TEX.Banco(meta,textos,ck['variante'],ck['mapa_fuentes'],ck['media'],ck['escala'],ck['limite_memoria'],q)
    pp=[]
    with torch.no_grad():
        rep=red.codificar(b,len(meta['dia']))
        for jj in np.array_split(ix,max(1,int(np.ceil(len(ix)/128)))):pp.append(red(b,torch.as_tensor(jj,device='cuda'),rep)[0].cpu().numpy())
    del b,red,rep;torch.cuda.empty_cache();return np.concatenate(pp)

def entrenar_texto(meta,s,etiqueta,a,es,ev,destino,textos,progreso,q):
    for v in TEXTOS:
        d=destino/v/f'semilla_{s}'
        if (d/'TERMINADO.json').exists():continue
        previo=REPO/'results/m4_gpu_20261122_194500'/('final' if etiqueta=='final' else f'controles')
        previo=previo/v/(f'anio_{etiqueta}' if etiqueta!='final' else '')/f'semilla_{s}'
        if v=='M4_OP' and etiqueta!='final':previo=REPO/'results/m4_gpu_20261122_194500/busqueda/cfg_6'/f'anio_{etiqueta}'/f'semilla_{s}'
        if s<=105 and (previo/'modelo.pt').exists():
            ck=cargar(previo/'modelo.pt')
            mu=meta['num'][a].mean(0);sd=meta['num'][a].std(0);sd=np.where(sd>1e-6,sd,1.)
            assert ck['configuracion']==C.CFG and np.array_equal(mu,ck['media']) and np.array_equal(sd,ck['escala'])
            d.mkdir(parents=True,exist_ok=True)
            for f in previo.iterdir():
                if f.name in ['modelo.pt','historia.csv','escaladores.json','TERMINADO.json','predicciones.npz']:shutil.copy2(f,d/f.name)
        else:
            col='disponible_estricto' if v.endswith('_estricto') else 'disponible_declarado'
            TEX.ajustar_memoria(meta,textos[col],C.CFG,s,a,es,ev,v,d,q,progreso)

def ajustar_procedencias(datos,meta,s,partes,car,work,progreso):
    pr=PRO.oof_fija(meta,s,car/f'fuentes/semilla_{s}/oof_fija',datos.q168,progreso,datos.nombre,work/'M1')
    for etiqueta,a,es,ev in partes:
        dd=car/('final' if not len(ev) else f'desarrollo/{etiqueta}')
        for modo in PRO.MODOS:
            d=dd/modo/f'semilla_{s}'
            if (d/'TERMINADO.json').exists() or (d/'IDENTIDAD.json').exists():continue
            pred=PRO.insample(datos,meta,a,es,s,car/f'fuentes/semilla_{s}/en_muestra_{etiqueta}',progreso) if modo=='residuos_en_muestra' else pr
            rr=PRO.banco_residual(meta,pred,a,datos.q168)
            if np.array_equal(rr,((meta['y']-meta['pred_base'])/datos.q168).astype('float32')):
                d.mkdir(parents=True,exist_ok=True);guardar(d/'IDENTIDAD.json',dict(referencia='M4_N',motivo='Mismo banco, configuración y semilla; no fabricar ajuste distinto'))
                continue
            PRO.ajustar(meta,rr,a,es,ev,s,d,datos.q168,progreso)

def pred_procedencia(modo,car,etiqueta,s,meta,ix,referencia):
    d=car/('final' if etiqueta=='final' else f'desarrollo/{etiqueta}')/modo/f'semilla_{s}'
    if (d/'IDENTIDAD.json').exists():return referencia
    ck=cargar(d/'modelo.pt');r=PRO.desde_npz(d/'residuos_banco.npz')
    rr=((meta['y']-meta['pred_base'])/ck['q']).astype('float32')
    assert np.array_equal(meta['dia'][:len(r['dia'])],r['dia']);rr[:len(r['dia'])]=r['residuos']
    return PRO.predecir(ck,meta,rr,ix)

def desarrollo(datos,meta,s,work,car,textos,progreso):
    cache=car/f'desarrollo_perdidas/semilla_{s}.npz'
    if cache.exists():return PRO.desde_npz(cache)
    anio=2022 if datos.nombre=='colombia' else datos.test_anio
    partes=particiones(meta,datos.nombre,anio);ajustar_procedencias(datos,meta,s,partes,car,work,progreso)
    ac={};dd=[];cfg=BEN.configuraciones()
    for etiqueta,a,es,ev in partes:
        if not len(ev):continue
        base=work/f'desarrollo/anio_{etiqueta}';extra=car/f'desarrollo/{etiqueta}';pr={'M1':meta['pred_base'][ev]}
        for v in C.VARIANTES:
            C.ajustar_control(meta,s,a,es,ev,v,base/v/f'semilla_{s}',datos.q168,progreso)
            pr[v]=C.predecir(cargar(base/v/f'semilla_{s}/modelo.pt'),meta,ev)
        w=REC.ventanas(datos,meta['dia'])
        for v in REC.NOMBRES:
            d=extra/v/f'semilla_{s}';REC.ajustar_recuperacion(meta,w,a,es,s,v,d,datos.q168,progreso);pr[v]=REC.predecir_recuperacion(meta,w,ev,cargar(d/'modelo.pt'),datos.q168)
        FU.ajustar(meta,a,es,extra/f'fuertes/semilla_{s}',datos.q168);pr.update(FU.predecir(meta,ev,extra/f'fuertes/semilla_{s}',datos.q168))
        origen=pd.Timestamp(meta['dia'][ev[0]]);corte=origen-pd.DateOffset(months=2)
        todas=datos.pos if hasattr(datos,'pos') else datos.elegibles(np.arange(24*8,len(datos.Y)-23,24));fechas=datos.indice.tz_localize(None)[todas]
        tr=todas[fechas<corte-pd.Timedelta(days=1)];st=todas[(fechas>=corte)&(fechas<origen)];pos=posiciones(datos,meta['dia'][ev])
        for v in BEN.MODELOS:
            d=extra/'benchmark'/v/f'semilla_{s}';BEN.ajustar_benchmark(datos,v,cfg[v],s,tr,st,d,progreso);pr['benchmark_'+v]=BEN.prueba_benchmark(datos,v,s,pos,d)
        for modo in PRO.MODOS:pr[modo]=pred_procedencia(modo,car,etiqueta,s,meta,ev,pr['M4_N'])
        if datos.nombre=='colombia':
            entrenar_texto(meta,s,etiqueta,a,es,ev,extra,textos,progreso,datos.q168)
            for v in TEXTOS:
                ck=cargar(extra/v/f'semilla_{s}/modelo.pt');col='disponible_estricto' if v.endswith('_estricto') else 'disponible_declarado';pr[v]=texto_pred(meta,textos[col],ck,ev,datos.q168)
        pr.update(datos.ingenuos(pos));dd.append(meta['dia'][ev])
        y=datos.Y[pos[:,None]+np.arange(24)]
        for v,p in pr.items():ac.setdefault(v,[]).append(np.abs(p.astype('float64')-y).mean(-1)/datos.q168)
    sal={k:np.concatenate(v) for k,v in ac.items()};cache.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(cache,dia=np.concatenate(dd),**sal);return dict(dia=np.concatenate(dd),**sal)

def finales_comunes(datos,meta,s,work,progreso):
    anio=2022 if datos.nombre=='colombia' else datos.test_anio;_,a,es,_=particiones(meta,datos.nombre,anio)[-1]
    # Las primeras diez semillas reutilizan los ajustes principales sin alterar archivos.
    for v in C.VARIANTES:C.ajustar_control(meta,s,a,es,np.array([],dtype='int64'),v,work/'final'/v/f'semilla_{s}',datos.q168,progreso)
    w=REC.ventanas(datos,meta['dia'])
    for v in REC.NOMBRES:REC.ajustar_recuperacion(meta,w,a,es,s,v,work/'final'/v/f'semilla_{s}',datos.q168,progreso)
    FU.ajustar(meta,a,es,work/f'fuertes/semilla_{s}',datos.q168)
    if datos.nombre=='colombia':tr,st=datos.ajuste_final()
    else:
        inicio=pd.Timestamp(f'{anio}-01-01',tz='UTC');corte=inicio-pd.DateOffset(months=2);tr=datos.pos[datos.indice[datos.pos]<corte-pd.Timedelta(days=1)];st=datos.pos[(datos.indice[datos.pos]>=corte)&(datos.indice[datos.pos]<inicio)]
    cfg=BEN.configuraciones()
    for v in BEN.MODELOS:BEN.ajustar_benchmark(datos,v,cfg[v],s,tr,st,work/'benchmark'/v/f'semilla_{s}',progreso)
    return a,es

def probar(datos,base,s,work,car,out,textos,progreso):
    destino=car/'prueba';marca=destino/f'TERMINADO_{s}.json'
    if marca.exists():return
    meta=COL.prueba(s,work,base) if datos.nombre=='colombia' else EXT.prueba_adicional(datos,s,work,base)
    anio=2022 if datos.nombre=='colombia' else datos.test_anio;ix=np.flatnonzero(meta['anio']>=anio);pos=posiciones(datos,meta['dia'][ix]);y=datos.Y[pos[:,None]+np.arange(24)];pr={}
    if s<=110:
        with np.load(out/datos.nombre/f'prueba/M4_N_semilla_{s}.npz') as z:pr['M4_N']=z['pred'];assert np.array_equal(z['dia'],meta['dia'][ix])
    else:
        pr['M1']=meta['pred_base'][ix]
        for v in C.VARIANTES:pr[v]=C.predecir(cargar(work/'final'/v/f'semilla_{s}/modelo.pt'),meta,ix)
        w=REC.ventanas(datos,meta['dia'])
        for v in REC.NOMBRES:pr[v]=REC.predecir_recuperacion(meta,w,ix,cargar(work/'final'/v/f'semilla_{s}/modelo.pt'),datos.q168)
        pr.update(FU.predecir(meta,ix,work/f'fuertes/semilla_{s}',datos.q168))
        for v in BEN.MODELOS:pr['benchmark_'+v]=BEN.prueba_benchmark(datos,v,s,pos,work/'benchmark'/v/f'semilla_{s}')
    for modo in PRO.MODOS:pr[modo]=pred_procedencia(modo,car,'final',s,meta,ix,pr['M4_N'])
    if datos.nombre=='colombia':
        for v in TEXTOS:
            ck=cargar(car/'final'/v/f'semilla_{s}/modelo.pt');col='disponible_estricto' if v.endswith('_estricto') else 'disponible_declarado';pr[v]=texto_pred(meta,textos[col],ck,ix,datos.q168)
    for v,p in pr.items():guardar_pred(destino,v,s,meta['dia'][ix],y,p)
    guardar(marca,dict(semilla=s,modelos=list(pr),prueba=True))

def juntar(out,stage,n_semillas):
    res=[];stat=stage/'estadistica_final';stat.mkdir(exist_ok=True);metricas=[]
    for nombre in DATASETS:
        with np.load(out/f'estadistica/perdidas_{nombre}.npz') as z:err={k:z[k] for k in z.files}
        q=DatosNumericos(PAQUETE).q168 if nombre=='colombia' else EXT.DatosAdicionales(nombre).q168
        modelos=leer(stage/nombre/'prueba/TERMINADO_101.json')['modelos']
        for v in modelos:
            vv=[]
            for s in range(101,101+n_semillas):
                with np.load(stage/nombre/f'prueba/{v}_semilla_{s}.npz') as z:
                    assert np.array_equal(z['dia'],err['dia']);vv.append(np.abs(z['y']-z['pred'].astype('float64')).mean(-1)/q)
            err[v]=np.asarray(vv)
        if n_semillas>10:
            extras=leer(stage/nombre/'prueba/TERMINADO_111.json')['modelos']
            for v in extras:
                if v in modelos:continue
                assert v in err and err[v].shape[0]==10
                vv=[*err[v]]
                for s in range(111,101+n_semillas):
                    with np.load(stage/nombre/f'prueba/{v}_semilla_{s}.npz') as z:vv.append(np.abs(z['y']-z['pred'].astype('float64')).mean(-1)/q)
                err[v]=np.asarray(vv)
        if nombre=='colombia':
            indice=leer(out/'insumos/top10_adicionales.json')
            for modelo,especificacion in indice.items():
                vv=[]
                for archivo in especificacion['archivos']:
                    with np.load(archivo) as z:
                        assert np.array_equal(z['dia'],err['dia']);vv.append(np.abs(z['y']-z['pred']).mean(-1)/q)
                assert len(vv)==5;err['historico_'+modelo]=np.asarray(vv)
        # Placebo compuesto por promedio de pérdidas, no de pronósticos.
        if nombre=='colombia':err['M4_desalineado_promedio']=np.mean([err[f'M4_desalineado_{d}'] for d in [49,91,154,217,301]],axis=0)
        p=stat/f'perdidas_{nombre}.npz';np.savez_compressed(p,**err);res.append(p)
        for v,e in err.items():
            if v=='dia':continue
            metricas.append(dict(dataset=nombre,modelo=v,MASE=float(e.mean()),semillas=5 if v.startswith('historico_') else len(e),DE_semillas=float(e.mean(1).std(ddof=1)) if len(e)>1 else None))
    pd.DataFrame(metricas).to_csv(stage/'resumen_resultados.csv',index=False);DIA.supuestos(res,stat)


def ejecutar(out):
    stage=out/'complemento';stage.mkdir(exist_ok=True)
    registro=leer(stage/'REGISTRO_PREVIO.json');assert registro['autorizado_por_usuario']
    assert all(sha(REPO/p)==h for p,h in registro['fuentes'].items()),'El código complementario cambió después de registrarse'
    assert (out/'FINALIZADO.json').exists() and (out/'historicos_nbeats_timesnet/FINALIZADO.json').exists();torch.set_num_threads(4);assert torch.cuda.is_available()
    assert all(sha(REPO/p)==h for p,h in leer(out/'REGISTRO_COLA_COMPLETA.json')['fuentes'].items())
    subprocess.run([sys.executable,str(REPO/'scripts/m4_cierre_smoke.py'),'--out',str(stage/'pruebas_implementacion')],check=True)
    etapa='Desarrollo complementario';hechas=0;total=40
    def progreso(**kw):guardar(out/'progreso/estado.json',dict(estado='Ejecutando',fase=etapa,tareas=hechas,total=total,gpu=torch.cuda.get_device_name(0),actualizado=pd.Timestamp.now(tz='America/Bogota').isoformat(),**kw))
    textos={col:TEX.Texto(COL.HIST.parent,col,final=True) for col in ['disponible_declarado','disponible_estricto']}
    decisiones=[]
    for nombre in DATASETS:
        car=stage/nombre;car.mkdir(exist_ok=True);filas=[]
        for s in range(101,111):
            datos,meta,work=cargar_meta(nombre,s,out,progreso);etapa=f'{nombre}: OOF complementario y ablaciones · s{s}';progreso()
            filas.append(desarrollo(datos,meta,s,work,car,textos,progreso));hechas+=1;progreso()
        assert all(np.array_equal(filas[0]['dia'],v['dia']) for v in filas)
        errs={k:np.stack([v[k] for v in filas]) for k in filas[0] if k!='dia'}
        np.savez_compressed(car/'PILOTO_OOF_DIEZ_SEMILLAS.npz',dia=filas[0]['dia'],**errs)
        decisiones.append(DIA.simular_potencia(errs,1461 if nombre=='colombia' else 365,car))
    n=max(decisiones);guardar(stage/'PRESUPUESTO_FIJADO_DESDE_DESARROLLO.json',dict(semillas=n,decisiones_por_dataset=dict(zip(DATASETS,decisiones)),prueba_usada=False))
    etapa='Ajustes finales complementarios';hechas=0;total=4*n
    for nombre in DATASETS:
        car=stage/nombre
        for s in range(101,101+n):
            datos,meta,work=cargar_meta(nombre,s,out,progreso);progreso(etiqueta=f'{nombre} finales s{s}')
            if s>110:desarrollo(datos,meta,s,work,car,textos,progreso)
            a,es=finales_comunes(datos,meta,s,work,progreso)
            if nombre=='colombia':entrenar_texto(meta,s,'final',a,es,np.array([],dtype='int64'),car/'final',textos,progreso,datos.q168)
            # desarrollo() ya preparó los correctores de procedencia finales.
            hechas+=1;progreso()
    pesos=list(stage.rglob('modelo.pt'));congelados={str(p.relative_to(stage)):sha(p) for p in pesos}
    guardar(stage/'CONGELACION_PREPRUEBA.json',dict(fecha=pd.Timestamp.now(tz='America/Bogota').isoformat(),semillas=n,pesos=congelados))
    etapa='Prueba complementaria congelada';hechas=0;total=4*n
    for nombre in DATASETS:
        for s in range(101,101+n):
            datos,meta,work=cargar_meta(nombre,s,out,progreso);progreso(etiqueta=f'{nombre} prueba s{s}');probar(datos,meta,s,work,stage/nombre,out,textos,progreso);hechas+=1;progreso()
    assert all(sha(stage/p)==h for p,h in congelados.items())
    juntar(out,stage,n);etapa='Portabilidad ampliada y auditoría de empates';progreso()
    subprocess.run([sys.executable,str(REPO/'scripts/m4_cierre_portabilidad.py'),'--out',str(out)],check=True)
    guardar(stage/'FINALIZADO.json',dict(fecha=pd.Timestamp.now(tz='America/Bogota').isoformat(),entrenamiento=True,prueba=True,semillas=n,certificacion_E3=False))
    (stage/'INFORME.md').write_text('# Evaluación complementaria\n\nResultados: `resumen_resultados.csv`. Incluye ablaciones de procedencia residual, canales y placebos, diagnósticos y dimensionamiento OOF. Las identidades de procedimientos quedan registradas sin crear réplicas ficticias.\n\nNo se certifica potencia de los tres procedimientos F2, originalidad ni confirmación prospectiva. Las plataformas ausentes y cualquier fallo de portabilidad se conservan como pendientes o rechazos. Consultar `MATRIZ_CIERRE.json`, diagnósticos y registros.\n')
    pendientes=['Windows nativo no disponible','ONNX GPU si no existe proveedor compatible','Confirmación prospectiva no disponible','Potencia F2 >=0.90 no garantizada por número fijo de semillas','Originalidad no demostrada; RATL/FreshCast son adaptaciones']
    guardar(stage/'MATRIZ_CIERRE.json',dict(completado_operativamente=True,certificacion_E3=False,pendientes=pendientes))
    guardar(stage/'MANIFIESTO_FINAL.json',[dict(archivo=str(p.relative_to(stage)),sha256=sha(p),bytes=p.stat().st_size) for p in sorted(stage.rglob('*')) if p.is_file() and p.name!='MANIFIESTO_FINAL.json' and p.suffix not in ['.log','.tmp','.pyc']])
    from publicar_cierre_m4 import publicar
    publicar(stage);assert leer(stage/'PUBLICADO.json')['verificado']
    guardar(out/'progreso/estado.json',dict(estado='Ejecutando',fase='Cierre estadístico final49999 y Holm1536',tareas=0,total=1,gpu='No requerida; estadística CPU'))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    try:ejecutar(a.out)
    except Exception:guardar(a.out/'ERROR_COMPLEMENTO.json',dict(error=traceback.format_exc(),apagado=False));raise

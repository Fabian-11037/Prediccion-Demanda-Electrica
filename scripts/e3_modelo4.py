"""M4: memoria causal de residuos OOF, recuperación aprendida y compuerta."""
from __future__ import annotations
import itertools
from e3_gpu_comun import *
from e3_modelo1 import dividir
from e3_modelo3 import Texto, leer_base, DESPLAZAMIENTOS
from corrector.evaluacion import indices_bootstrap, holm

VARIANTES=['M4_OP','M4_N','M4_O','M4_metadatos']+[f'M4_desalineado_{d}' for d in DESPLAZAMIENTOS]+['M4_memoria_fija','M4_sin_compuerta','M4_sin_num','M4_OP_estricto']


def variante_texto(v):
    if v=='M4_N':return 'M3_N'
    if v=='M4_O':return 'M3_O'
    if v=='M4_metadatos':return 'M3_metadatos'
    if v.startswith('M4_desalineado_'):return v.replace('M4_','M3_',1)
    return 'M3_OP'


class Banco:
    """Índices de candidatos fijos; embeddings y mapas no se ajustan en inferencia."""
    def __init__(self,meta,texto,variante,mapa,media,escala,limite=None,q=215617.90361589016):
        self.meta=meta;self.texto=texto;self.variante=variante
        f=texto.filas(meta['dia'],variante_texto(variante));m,feat=texto.mascara(f,variante_texto(variante))
        self.ids=torch.as_tensor(np.maximum(texto.k['cand'][f],0).astype('int64'),device='cuda')
        self.feat=torch.as_tensor(feat,device='cuda')
        self.fte=torch.as_tensor(mapa[texto.k['fuente'][f].astype('int64')],device='cuda')
        self.mascara=torch.as_tensor(m,device='cuda')
        self.num=torch.as_tensor((meta['num']-media)/escala,device='cuda')
        self.base=torch.as_tensor(meta['pred_base'],device='cuda')
        self.residuos=torch.as_tensor((meta['y']-meta['pred_base'])/q,device='cuda')
        self.dias=torch.as_tensor(np.asarray(meta['dia'],dtype='datetime64[D]').astype('int64'),device='cuda')
        self.limite=limite

    def texto_lote(self,ix):
        e=self.texto.emb[self.ids[ix]].float()
        if self.variante=='M4_metadatos':e=torch.zeros_like(e)
        return e,self.feat[ix],self.fte[ix],self.mascara[ix]


class Memoria(nn.Module):
    def __init__(self,n,n_feat,n_fuentes,cfg,variante='M4_OP'):
        super().__init__();self.cfg=cfg;self.variante=variante;d=32
        self.consulta_texto=nn.Linear(n,d)
        self.proy=nn.ModuleList([nn.Linear(768,d) for _ in range(2)])
        self.rasgo=nn.ModuleList([nn.Linear(n_feat,d) for _ in range(2)])
        self.fuente=nn.ModuleList([nn.Embedding(n_fuentes+1,d) for _ in range(2)])
        self.norma=nn.ModuleList([nn.LayerNorm(d) for _ in range(2)])
        self.proy_q=nn.Sequential(nn.Linear(n+2*d,d),nn.LayerNorm(d))
        self.proy_k=nn.Sequential(nn.Linear(n+2*d,d),nn.LayerNorm(d))
        w=cfg['ancho_compuerta']
        self.compuerta=nn.Sequential(nn.Linear(d+4,w),nn.GELU(),nn.Dropout(.1),nn.Linear(w,1))
        nn.init.zeros_(self.compuerta[-1].weight);nn.init.constant_(self.compuerta[-1].bias,-2.)

    def codificar(self,banco,limite):
        """Relevancia aprendida por fragmento y canal; sin reducción entre fechas."""
        partes=[]
        for i in range(0,limite,64):
            ix=slice(i,min(i+64,limite));num=banco.num[ix]
            if self.variante=='M4_sin_num':num=torch.zeros_like(num)
            if self.variante=='M4_N':
                texto=num.new_zeros((len(num),64))
            else:
                emb,feat,fte,mascara=banco.texto_lote(ix);q=self.consulta_texto(num);canales=[]
                for c in range(2):
                    z=self.norma[c](self.proy[c](emb)+self.rasgo[c](feat)+self.fuente[c](fte))
                    m=mascara & (feat[...,c]>.5);hay=m.any(-1)
                    score=(z*q[:,None]).sum(-1)/(32**.5)
                    score=score.masked_fill(~m,-1e9)
                    alfa=score.softmax(-1)*m
                    alfa=alfa/alfa.sum(-1,keepdim=True).clamp_min(1e-8)
                    canales.append((alfa[:,:,None]*z).sum(1)*hay[:,None])
                texto=torch.cat(canales,1)
            partes.append(torch.cat([num,texto],1))
        x=torch.cat(partes,0)
        return self.proy_q(x),self.proy_k(x)

    def recuperar(self,q,k,dias,residuos,j,base,limite=None):
        """La máscara temporal se aplica ANTES de top-k y de consultar valores."""
        edad=dias[j,None]-dias[None,:]
        valido=(edad>=2)&(edad<=self.cfg['edad_maxima_dias'])
        if limite is not None:valido &= dias[None,:]<=limite
        scores=(q[j]@k.T)/(32**.5)
        scores=scores.masked_fill(~valido,-1e9)
        kk=min(self.cfg['k_vecinos'],k.shape[0]);valor,ids=scores.topk(kk,-1)
        ok=valido.gather(1,ids);hay=ok.any(-1)
        alfa=valor.softmax(-1)*ok
        alfa=alfa/alfa.sum(-1,keepdim=True).clamp_min(1e-8)
        # Los lugares vacíos consultan una fila centinela cero, nunca un valor
        # propio o futuro que hubiera quedado entre los top-k enmascarados.
        valores=torch.cat([residuos,residuos.new_zeros((1,24))],0)
        seguros=torch.where(ok,ids,torch.full_like(ids,len(residuos)))
        rr=valores[seguros]
        media=(alfa[:,:,None]*rr).sum(1)
        dispersion=((rr-media[:,None]).square()*alfa[:,:,None]).sum(1).mean(-1).clamp_min(1e-12).sqrt()
        apoyo=ok.sum(-1).to(q.dtype)/self.cfg['k_vecinos']
        maxsim=torch.where(hay,valor[:,0],torch.zeros_like(valor[:,0]))
        edades=edad.gather(1,ids).to(q.dtype)
        antig=(alfa*edades).sum(-1)/self.cfg['edad_maxima_dias']
        features=torch.cat([q[j],maxsim[:,None],dispersion[:,None],apoyo[:,None],antig[:,None]],1)
        g=torch.sigmoid(self.compuerta(features))[:,0]
        if self.variante=='M4_sin_compuerta':g=torch.ones_like(g)
        g=g*hay
        pred=base+215617.90361589016*g[:,None]*media
        pred=torch.where(hay[:,None],pred,base)
        assert ((~ok)|(edad.gather(1,ids)>=2)).all()
        return pred,g,ids,alfa,ok

    def forward(self,banco,j,representaciones=None):
        limite=int(j.max().item())+1
        q,k=self.codificar(banco,limite) if representaciones is None else representaciones
        n=len(k)
        return self.recuperar(q,k,banco.dias[:n],banco.residuos[:n],j,banco.base[j],banco.limite)


def construir(ck):
    return Memoria(ck['n_num'],ck['n_feat'],ck['n_fuentes'],ck['configuracion'],ck['variante']).cuda()


def ajustar_memoria(meta,texto,cfg,s,a,es,ev,v,car,q,progreso):
    if (car/'TERMINADO.json').exists():return leer(car/'TERMINADO.json')
    motor.fijar_semilla(s);mu=meta['num'][a].mean(0);sd=meta['num'][a].std(0);sd=np.where(sd>1e-6,sd,1.)
    mapa=texto.fuentes_ajuste(meta['dia'][a],variante_texto(v))
    limite=int(np.datetime64(meta['dia'][es[-1]],'D').astype('int64')) if v=='M4_memoria_fija' else None
    ck=dict(variante=v,configuracion=cfg,semilla=s,media=mu,escala=sd,mapa_fuentes=mapa,limite_memoria=limite,
            n_num=meta['num'].shape[1],n_feat=texto.k['feat'].shape[2],n_fuentes=int(mapa.max()),
            ajuste=[meta['dia'][a[0]],meta['dia'][a[-1]]],parada=[meta['dia'][es[0]],meta['dia'][es[-1]]])
    banco=Banco(meta,texto,v,mapa,mu,sd,limite,q);red=construir(ck)
    y=torch.as_tensor(meta['y'],device='cuda');aa=torch.as_tensor(a,device='cuda')
    def perdida(ix):
        j=aa[ix];return (red(banco,j)[0]-y[j]).abs().mean()/q
    def evaluar():
        rep=red.codificar(banco,int(es[-1])+1);total=0.
        for i in range(0,len(es),128):
            j=torch.as_tensor(es[i:i+128],device='cuda');pred=red(banco,j,rep)[0]
            total+=float((pred-y[j]).abs().mean()/q)*len(j)
        return total/len(es)
    info=ajustar(red,len(a),perdida,evaluar,dict(lr=.001,weight_decay=.0001,lote=64),
                 dict(max_epocas=100,min_epocas=10,paciencia=10,min_delta=.0001),s,car,progreso,f'{v} cfg{cfg["id"]} s{s} · parada{meta["anio"][es[0]]}')
    ck['modelo']=red.state_dict();checkpoint(car/'modelo.pt',ck)
    guardar(car/'escaladores.json',dict(media=mu.tolist(),escala=sd.tolist(),mapa_fuentes=mapa.tolist(),ajuste=ck['ajuste']))
    otra=construir(ck);otra.load_state_dict(cargar(car/'modelo.pt')['modelo']);otra.eval()
    j=torch.as_tensor(es[:2],device='cuda')
    with torch.no_grad():assert torch.equal(red(banco,j)[0],otra(banco,j)[0]),'Restauración M4 no idéntica'
    r=dict(variante=v,configuracion=cfg,semilla=s,n_parametros=sum(p.numel() for p in red.parameters()),
           ajuste=ck['ajuste'],parada=ck['parada'],**{k:v for k,v in info.items() if k!='historia'})
    if len(ev):
        with torch.no_grad():
            rep=red.codificar(banco,int(ev[-1])+1)
            pred=np.concatenate([red(banco,torch.as_tensor(ev[i:i+128],device='cuda'),rep)[0].cpu().numpy() for i in range(0,len(ev),128)])
        r.update(metricas(meta['y'][ev],pred,q));np.savez_compressed(car/'predicciones.npz',dia=meta['dia'][ev],y=meta['y'][ev],pred=pred)
    guardar(car/'TERMINADO.json',r);del red,otra,banco,y,aa;torch.cuda.empty_cache()
    return r


def estadistica(errores,anios,previas):
    pares=[('M4_O','M4_N'),('M4_OP','M4_N'),('M4_OP','M4_metadatos'),('M4_OP','M4_desalineado_promedio'),('M4_OP','M4_O')]
    filas=[]
    for bloque in [14,7,28]:
        idx=indices_bootstrap(anios,9999,bloque,20261122);nuevas=[]
        for a,b in pares:
            ea,eb=errores[a],errores[b];g=100*(eb.mean()-ea.mean())/eb.mean()
            gb=100*(eb[idx].mean(1)-ea[idx].mean(1))/eb[idx].mean(1);lo,hi=np.percentile(gb,[2.5,97.5])
            for m in [0.,3.]:
                p=float((1+np.sum(gb-g>=g-m))/(len(gb)+1))
                nuevas.append(dict(A=a,B=b,bloque_dias=bloque,margen_pct=m,reduccion_pct=g,ic95_inf=lo,ic95_sup=hi,p_unilateral=p))
        prev=previas[previas.bloque_dias==bloque].to_dict('records');assert len(prev)==18
        conjunto=prev+nuevas;assert len(conjunto)==28
        ajustados=holm([f['p_unilateral'] for f in conjunto])
        for f,p in zip(conjunto,ajustados):f['p_holm_global_28']=p
        filas+=conjunto
    return pd.DataFrame(filas)


def ejecutar(out,datos,diseno,run,m3,prueba_m1,progreso,completar,log):
    semillas=diseno['validacion']['semillas'];g=diseno['modelos']['M4']['rejilla']
    rejilla=[dict(id=i+1,k_vecinos=k,edad_maxima_dias=edad,ancho_compuerta=w) for i,(k,edad,w) in enumerate(itertools.product(g['k_vecinos'],g['edad_maxima_dias'],g['ancho_compuerta']))]
    guardar(out/'REJILLA_REGISTRADA.json',rejilla)
    textos={col:Texto(run,col) for col in ['disponible_declarado','disponible_estricto']};filas=[]
    for s in semillas:
        meta=leer_base(m3,s)
        for anio in diseno['validacion']['folds_externos']:
            a,es,ev=dividir(meta,anio)
            for cfg in rejilla:
                progreso(fase='busqueda_m4',etiqueta=f'M4_OP cfg{cfg["id"]} {anio} s{s}')
                car=out/'busqueda'/f'cfg_{cfg["id"]}'/f'anio_{anio}'/f'semilla_{s}'
                r=ajustar_memoria(meta,textos['disponible_declarado'],cfg,s,a,es,ev,'M4_OP',car,datos.q168,progreso)
                filas.append(dict(cfg=cfg['id'],anio=anio,semilla=s,MASE=r['MASE'],n_parametros=r['n_parametros']));completar()
    tabla=pd.DataFrame(filas);tabla.to_csv(out/'seleccion_por_fold_semilla.csv',index=False)
    sel=tabla.groupby('cfg').agg(MASE=('MASE','mean'),n_parametros=('n_parametros','mean')).reset_index().sort_values(['MASE','n_parametros','cfg'])
    sel.to_csv(out/'seleccion_resumen.csv',index=False);cfg=next(p for p in rejilla if p['id']==int(sel.iloc[0]['cfg']))
    guardar(out/'CONFIGURACION_ELEGIDA.json',dict(configuracion=cfg,criterio='MASE medio de folds/semillas, solo M4_OP; sin prueba'))
    for s in semillas:
        meta=leer_base(m3,s)
        for anio in diseno['validacion']['folds_externos']:
            a,es,ev=dividir(meta,anio)
            for v in VARIANTES[1:]:
                col='disponible_estricto' if v.endswith('_estricto') else 'disponible_declarado'
                progreso(fase='controles_m4',etiqueta=f'{v} {anio} s{s}')
                ajustar_memoria(meta,textos[col],cfg,s,a,es,ev,v,out/'controles'/v/f'anio_{anio}'/f'semilla_{s}',datos.q168,progreso);completar()
        a,es,ev=dividir(meta,2022)
        for v in VARIANTES:
            col='disponible_estricto' if v.endswith('_estricto') else 'disponible_declarado'
            progreso(fase='ajuste_final_m4',etiqueta=f'{v} final s{s}')
            ajustar_memoria(meta,textos[col],cfg,s,a,es,ev,v,out/'final'/v/f'semilla_{s}',datos.q168,progreso);completar()
    finales=list((out/'final').rglob('modelo.pt'));assert len(finales)==65
    huellas={str(p.relative_to(out)):sha(p) for p in finales}
    guardar(out/'CONGELACION_PREPRUEBA.json',dict(fecha=pd.Timestamp.now(tz='America/Bogota').isoformat(),huellas=huellas,configuracion=cfg))
    del textos;torch.cuda.empty_cache()
    textos={col:Texto(run,col,True) for col in ['disponible_declarado','disponible_estricto']}
    prueba=out/'prueba';prueba.mkdir(exist_ok=True);filas=[];anuales=[];horarios=[];todas={}
    for s in semillas:
        progreso(fase='prueba_m4',etiqueta=f'Memoria causal en prueba s{s}',epoca=None)
        hist=leer_base(m3,s)
        with np.load(m3/'base_final'/f'semilla_{s}.npz') as z:test={k:z[k] for k in z.files}
        meta={k:np.concatenate([hist[k],test[k]]) for k in test};meta['anio']=pd.DatetimeIndex(meta['dia']).year.to_numpy()
        ix=np.flatnonzero(meta['anio']>=2022);dias=meta['dia'][ix];y=meta['y'][ix]
        assert len(ix)==1461
        preds={'M1':meta['pred_base'][ix]}
        with np.load(prueba_m1/f'predicciones_semilla_{s}.npz') as z:
            assert np.array_equal(dias,z['dia']) and np.array_equal(y,z['y']);preds['NNLS']=z['NNLS']
        for v in VARIANTES:
            col='disponible_estricto' if v.endswith('_estricto') else 'disponible_declarado'
            ck=cargar(out/'final'/v/f'semilla_{s}'/'modelo.pt');red=construir(ck);red.load_state_dict(ck['modelo']);red.eval()
            banco=Banco(meta,textos[col],v,ck['mapa_fuentes'],ck['media'],ck['escala'],ck['limite_memoria'],datos.q168)
            pp=[];gg=[];jj=[];aa=[];oo=[]
            with torch.no_grad():
                # Parámetros congelados; cada consulta filtra por su propio origen.
                rep=red.codificar(banco,len(meta['dia']))
                for i in range(0,len(ix),128):
                    j=torch.as_tensor(ix[i:i+128],device='cuda');p,g,ids,alfa,ok=red(banco,j,rep)
                    pp.append(p.cpu().numpy());gg.append(g.cpu().numpy());jj.append(ids.cpu().numpy());aa.append(alfa.cpu().numpy());oo.append(ok.cpu().numpy())
            pr=np.concatenate(pp);ids=np.concatenate(jj);ok=np.concatenate(oo);gate=np.concatenate(gg);alfa=np.concatenate(aa)
            fechas=np.asarray(meta['dia'],dtype='datetime64[D]');edades=(np.asarray(dias,dtype='datetime64[D]')[:,None]-fechas[ids]).astype('timedelta64[D]').astype(int)
            assert ((~ok)|((edades>=2)&(edades<=cfg['edad_maxima_dias']))).all()
            assert np.array_equal(pr[~ok.any(-1)],preds['M1'][~ok.any(-1)])
            assert np.isfinite(pr).all();preds[v]=pr
            np.savez_compressed(prueba/f'memoria_{v}_semilla_{s}.npz',dia=dias,ids_episodios=ids,fechas_episodios=fechas[ids].astype('U10'),elegibles=ok,atencion=alfa,compuerta=gate)
            del red,banco,rep;torch.cuda.empty_cache()
        for nombre,pr in preds.items():
            todas.setdefault(nombre,[]).append(pr);filas.append(dict(modelo=nombre,semilla=s,**metricas(y,pr,datos.q168)))
            for anio in range(2022,2026):
                m=np.char.startswith(dias,str(anio));anuales.append(dict(modelo=nombre,semilla=s,anio=anio,**metricas(y[m],pr[m],datos.q168)))
            for h in range(24):horarios.append(dict(modelo=nombre,semilla=s,hora=h,**metricas(y[:,h],pr[:,h],datos.q168)))
        np.savez_compressed(prueba/f'predicciones_semilla_{s}.npz',dia=dias,y=y,**preds)
        pd.DataFrame(dict(fecha=np.repeat(dias,24),hora=np.tile(np.arange(24),len(dias)),semilla=s,y_kwh=y.ravel(),**{k:v.ravel() for k,v in preds.items()})).to_csv(prueba/f'predicciones_semilla_{s}.csv.gz',index=False,compression='gzip')
        pd.DataFrame(dict(episodio=dias,disponible_desde=(np.asarray(dias,dtype='datetime64[D]')+np.timedelta64(2,'D')).astype('U10'),semilla=s,actualizacion='incorporación no paramétrica; sin optimizador')).to_csv(prueba/f'actualizaciones_memoria_semilla_{s}.csv',index=False)
    assert all(sha(out/p)==h for p,h in huellas.items())
    tabla=pd.DataFrame(filas);tabla.to_csv(prueba/'metricas_por_semilla.csv',index=False)
    pd.DataFrame(anuales).to_csv(prueba/'metricas_por_anio_semilla.csv',index=False);pd.DataFrame(horarios).to_csv(prueba/'metricas_por_hora_semilla.csv',index=False)
    resumen=tabla.groupby('modelo').agg(MASE=('MASE','mean'),DE_MASE=('MASE','std'),MAE_kwh=('MAE_kwh','mean'),RMSE_kwh=('RMSE_kwh','mean')).sort_values('MASE')
    resumen.to_csv(prueba/'resumen_resultados.csv')
    errores={k:np.abs(np.asarray(v,dtype='float64')-y).mean(axis=(0,2))/datos.q168 for k,v in todas.items()}
    errores['M4_desalineado_promedio']=np.mean([errores[f'M4_desalineado_{d}'] for d in DESPLAZAMIENTOS],axis=0)
    progreso(fase='estadistica_m4',etiqueta='Holm conjunto de28 hipótesis e informe',epoca=None)
    c=estadistica(errores,pd.DatetimeIndex(dias).year.to_numpy(),pd.read_csv(m3/'prueba/contrastes_globales_parciales.csv'))
    c.to_csv(prueba/'contrastes_globales_28.csv',index=False)
    lineas=['# M4 · memoria causal numérica y textual','','Prueba retrospectiva2022–2025:1.461 orígenes diarios,24 horas, cinco semillas. M1 y E5 congelados. Selección de ocho configuraciones exclusivamente con M4_OP en desarrollo2018–2021; controles y ablaciones heredan la seleccionada.','','| Procedimiento | MASE | DE semillas | MAE kWh | RMSE kWh |','|---|---:|---:|---:|---:|']
    for k,r in resumen.iterrows():lineas.append(f'| {k} | {r.MASE:.6f} | {r.DE_MASE:.6f} | {r.MAE_kwh:.2f} | {r.RMSE_kwh:.2f} |')
    lineas+=['','Se reutiliza el OOF anidado de M1 preparado por M3; no se utiliza el ganador global para filas históricas. Proyección32, relevancia aprendida por fragmento y canal, atención entre episodios y compuerta según contexto, similitud, dispersión, cantidad y antigüedad del respaldo.','','La memoria es creciente y solo admite episodios d−2 o anteriores, dentro de la edad máxima. La máscara temporal se aplica antes de top-k y antes de consultar residuos. Sin vecinos, identidad exacta con M1. Los residuos corresponden a M1 fuera de muestra. Durante prueba, pesos, escalas y mapas de fuentes permanecen congelados; solo se incorporan episodios ya observables. Se guardan vecinos, fechas, atenciones, compuertas y log de disponibilidad. Las proyecciones se vectorizan por fecha sin estadísticas entre orígenes; las máscaras evitan influencia de filas futuras.','','Se incluyen control numérico, oficial, metadatos, cinco desplazamientos, memoria fija en2021, sin compuerta, sin contexto numérico y sensibilidad estricta. La variante numérica coincide con la ablación sin texto. El contrato textual específico de §5 del diseño Markdown y los vectores E5 son los mismos que M3; se registra su diferencia con el bloque genérico del JSON.','','Se completa la familia de28 hipótesis de los cuatro modelos. Holm conjunto por cada tamaño de bloque, con14 días principal y7/28 de sensibilidad;9999 réplicas pareadas dentro del año. Las semillas se promedian en pérdidas antes del bootstrap. IC95 puntuales. Los cinco desplazamientos se agregan por pérdidas. No se deduce equivalencia de ausencia de significancia.','','Diseño informado por un benchmark ya conocido; evaluación retrospectiva. No se garantiza positividad de la corrección aditiva ni mejora respecto de la base. Los resultados se guardan localmente, sin publicación o despliegue automático.','']
    (out/'INFORME.md').write_text('\n'.join(lineas),encoding='utf-8')
    guardar(out/'FINALIZADO.json',dict(fecha=pd.Timestamp.now(tz='America/Bogota').isoformat(),entrenamiento=True,prueba=True,semillas=semillas,configuracion=cfg,checkpoints_finales=65,tareas=465))
    log('M4 completado.\n'+resumen.to_string())

"""M3: selector textual acotado sobre M1 congelado, con base OOF anidada."""
from __future__ import annotations
import itertools
from e3_gpu_comun import *
from e3_modelo1 import Router, datos_meta, dividir, entrenar_router
from corrector import modelo as MO
from corrector.evaluacion import indices_bootstrap, holm

DESPLAZAMIENTOS=[49,91,154,217,301]
VARIANTES=['M3_OP','M3_O','M3_P','M3_metadatos','M3_N']+[f'M3_desalineado_{d}' for d in DESPLAZAMIENTOS]+['M3_OP_estricto']


def elegir_prefijo(filas,anio,rejilla):
    """No consulta resultados del año predicho ni de años posteriores."""
    prev=[f for f in filas if f['anio']<anio]
    if not prev:return rejilla[0],dict(bootstrap=True,regla='cfg1 predefinida antes del primer fold elegible',ultimo_anio_seleccion=None)
    t=pd.DataFrame(prev).groupby('cfg').agg(MASE=('MASE','mean'),n_parametros=('n_parametros','first')).reset_index().sort_values(['MASE','n_parametros','cfg'])
    return next(p for p in rejilla if p['id']==int(t.iloc[0]['cfg'])),dict(bootstrap=False,ultimo_anio_seleccion=max(f['anio'] for f in prev),tabla=t.to_dict('records'))


def inferir_base(meta,ix,modelo,car):
    if car.exists():
        with np.load(car) as z:return {k:z[k] for k in z.files}
    ck=cargar(modelo);red=Router(ck['n_entrada'],ck['configuracion']).cuda().eval();red.load_state_dict(ck['modelo']);red.requires_grad_(False)
    x=torch.as_tensor((meta['x'][ix]-ck['media'])/ck['escala'],device='cuda')
    pr=torch.as_tensor(meta['pred'][ix],device='cuda');pp=[];cc=[];ll=[]
    def capturar(mod,entrada,salida):ll.append(salida.detach().reshape(-1,24,3).cpu().numpy())
    h=red.logits.register_forward_hook(capturar)
    try:
        with torch.no_grad():
            for i in range(0,len(ix),256):
                p,c,_=red(x[i:i+256],pr[i:i+256]);pp.append(p.cpu().numpy());cc.append(c.cpu().numpy())
    finally:h.remove()
    r=dict(dia=meta['dia'][ix],y=meta['y'][ix],pred_base=np.concatenate(pp),c_num=np.concatenate(cc),
           logits=np.concatenate(ll),expertos=meta['pred'][ix],num=meta['x'][ix],c_real=meta['c'][ix])
    car.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(car,**r)
    del red,x,pr;torch.cuda.empty_cache()
    return r


def preparar_oof(datos,run,out,semillas,progreso,completar,log):
    rejilla=leer(run/'rejilla_registrada.json');filas=[]
    metas={s:datos_meta(datos,run,s) for s in semillas}
    # Las decisiones se escriben antes de incorporar las métricas del año Y.
    for anio in range(2013,2022):
        cfg,decision=elegir_prefijo(filas,anio,rejilla)
        guardar(out/'prefijos'/f'seleccion_{anio}.json',dict(anio_predicho=anio,configuracion=cfg,**decision))
        for s in semillas:
            meta=metas[s];a,es,ev=dividir(meta,anio)
            for p in rejilla:
                if anio>=2018:
                    car=run/'busqueda'/f'cfg_{p["id"]}'/f'anio_{anio}'/f'semilla_{s}'
                    r=leer(car/'TERMINADO.json')
                    assert r['configuracion']==p and r['semilla']==s
                else:
                    car=out/'preparacion_m1'/f'cfg_{p["id"]}'/f'anio_{anio}'/f'semilla_{s}'
                    progreso(fase='oof_anidado_m1',etiqueta=f'Prefijo M1 cfg{p["id"]} {anio} s{s}')
                    r=entrenar_router(meta,p,s,a,es,ev,car,progreso,datos.q168);completar()
                assert str(r['ajuste'][-1])<f'{anio-1}-01-01' and str(r['parada'][-1])<f'{anio}-01-01'
                filas.append(dict(cfg=p['id'],anio=anio,semilla=s,MASE=r['MASE'],n_parametros=r['n_parametros']))
            elegido=(run/'busqueda' if anio>=2018 else out/'preparacion_m1')/f'cfg_{cfg["id"]}'/f'anio_{anio}'/f'semilla_{s}'/'modelo.pt'
            progreso(fase='oof_anidado_m1',etiqueta=f'M1 OOF {anio} s{s}, selección con años anteriores',epoca=None)
            inferir_base(meta,ev,elegido,out/'base_oof'/f'anio_{anio}_semilla_{s}.npz');completar()
        pd.DataFrame(filas).to_csv(out/'seleccion_m1_por_prefijo.csv',index=False)
    guardar(out/'BASE_OOF_LISTA.json',dict(inicio='2013-01-01',fin='2021-12-31',seleccion_por_prefijo=True,ajustes_nuevos=200,modelos_oof=45))
    del metas;torch.cuda.empty_cache();log('OOF anidado de M1 completado, sin configuración elegida con años futuros.')


def leer_base(out,s):
    partes=[]
    for a in range(2013,2022):
        with np.load(out/'base_oof'/f'anio_{a}_semilla_{s}.npz') as z:partes.append({k:z[k] for k in z.files})
    r={k:np.concatenate([p[k] for p in partes]) for k in partes[0]}
    r['anio']=pd.DatetimeIndex(r['dia']).year.to_numpy()
    assert len(np.unique(r['dia']))==len(r['dia'])
    return r


class Texto:
    """Contrato específico de §5 del diseño Markdown, idéntico al corrector E3."""
    def __init__(self,run,col,final=False):
        if final:
            car=run/'corrector_final/texto'/col;res=leer(car/'LISTO.json');vp=car/'vectores.npy'
            assert res['candidatos_pre2022_identicos'] and res['disponibilidad_verificada']
        else:
            base=PAQUETE/'resultados_corrector_e3'
            car=base/('texto' if col=='disponible_declarado' else 'texto_'+col)
            res=leer(car/'resumen.json');vp=base/'texto_vectores'/f'vectores_{col}.npy'
        with np.load(car/'candidatos.npz') as z:self.k={k:z[k] for k in z.files}
        self.reservas=res.get('reservas',dict(alineados=[0,32],oficial=[32,80],prensa=[80,128]))
        self.mapa={d:i for i,d in enumerate(self.k['dias'].astype('U10'))}
        self.emb=torch.from_numpy(np.array(np.load(vp,mmap_mode='r'))).cuda()
        self.n_fuentes=len(res['fuentes']);self.col=col
        self.vectores=vp

    def filas(self,dias,variante):
        shift=int(variante.rsplit('_',1)[-1]) if variante.startswith('M3_desalineado_') else 0
        dd=(np.asarray(dias,dtype='datetime64[D]')-np.timedelta64(shift,'D')).astype('U10')
        f=np.array([self.mapa.get(d,-1) for d in dd]);assert (f>=0).all(),'Faltan candidatos para el origen/retardo'
        return f

    def mascara(self,f,variante):
        nombre='metadatos' if variante=='M3_metadatos' else 'oficial' if variante=='M3_O' else 'oficial_prensa'
        m,feat=MO.mascara_variante(nombre,self.reservas,self.k['cand'][f],self.k['feat'][f].astype('float32'))
        if variante=='M3_P':m &= feat[...,1]>.5
        if variante=='M3_N':m[:]=False
        return m,feat

    def fuentes_ajuste(self,dias,variante):
        f=self.filas(dias,variante);m,_=self.mascara(f,variante)
        conocidos=sorted(set(self.k['fuente'][f][m].tolist())-{0})
        mapa=np.zeros(self.n_fuentes+1,dtype='int64')
        for i,v in enumerate(conocidos,1):mapa[v]=i
        return mapa

    def lote(self,dias,variante,mapa):
        f=self.filas(dias,variante);m,feat=self.mascara(f,variante)
        ids=np.maximum(self.k['cand'][f],0).astype('int64')
        e=self.emb[torch.as_tensor(ids,device='cuda')].float()
        if variante in ['M3_metadatos','M3_N']:e=torch.zeros_like(e)
        ft=mapa[self.k['fuente'][f].astype('int64')]
        return e,torch.as_tensor(feat,device='cuda'),torch.as_tensor(ft,device='cuda'),torch.as_tensor(m,device='cuda')


class SelectorTexto(nn.Module):
    def __init__(self,n,n_feat,n_fuentes,cfg):
        super().__init__();d=cfg['d_texto'];self.epsilon=cfg['epsilon_logits']
        self.consulta=nn.Sequential(nn.Linear(n,d),nn.GELU(),nn.Dropout(.1),nn.Linear(d,d))
        self.hora=nn.Embedding(24,d)
        self.proy=nn.ModuleList([nn.Linear(768,d) for _ in range(2)])
        self.rasgo=nn.ModuleList([nn.Linear(n_feat,d) for _ in range(2)])
        self.fuente=nn.ModuleList([nn.Embedding(n_fuentes+1,d) for _ in range(2)])
        self.norma=nn.ModuleList([nn.LayerNorm(d) for _ in range(2)])
        self.atencion=nn.ModuleList([nn.MultiheadAttention(d,4,dropout=.1,batch_first=True) for _ in range(2)])
        self.nulo=nn.Parameter(torch.zeros(2,1,1,d))
        self.salida=nn.Sequential(nn.Linear(3*d,d),nn.GELU(),nn.Dropout(.1),nn.Linear(d,7))
        nn.init.zeros_(self.salida[-1].weight);nn.init.zeros_(self.salida[-1].bias)
        with torch.no_grad():self.salida[-1].bias[3]=-2.

    def forward(self,num,expertos,logits,base,c_num,emb,feat,fuente,mascara):
        b=len(num);q=self.consulta(num)[:,None]+self.hora.weight[None];contextos=[]
        for canal in range(2):
            m=mascara & (feat[...,canal]>.5)
            e=self.norma[canal](self.proy[canal](emb)+self.rasgo[canal](feat)+self.fuente[canal](fuente))
            e=torch.cat([self.nulo[canal].expand(b,1,-1),e],1)
            valido=torch.cat([torch.ones(b,1,dtype=torch.bool,device=num.device),m],1)
            c,_=self.atencion[canal](q,e,e,key_padding_mask=~valido,need_weights=False);contextos.append(c)
        vg=self.salida(torch.cat([q,*contextos],-1))
        hay=mascara.any(-1)
        delta=hay[:,None,None]*torch.sigmoid(vg[...,3:4])*self.epsilon*torch.tanh(vg[...,:3])
        w=(logits+delta).softmax(-1)
        pred=(w*expertos).sum(-1)
        pred=torch.where(hay[:,None],pred,base)
        c=c_num+hay[:,None,None]*vg[...,4:7]
        return pred,c,w,delta


class SelectorNumerico(nn.Module):
    """Control sin texto, capacidad ajustada al número de parámetros del selector textual."""
    def __init__(self,n,n_feat,n_fuentes,cfg):
        super().__init__();ref=SelectorTexto(n,n_feat,n_fuentes,cfg)
        objetivo=sum(p.numel() for p in ref.parameters());h=max(1,round((objetivo-168)/(n+169)))
        self.epsilon=cfg['epsilon_logits'];self.ancho_control=h
        self.mlp=nn.Sequential(nn.Linear(n,h),nn.GELU(),nn.Dropout(.1),nn.Linear(h,168))
        nn.init.zeros_(self.mlp[-1].weight);nn.init.zeros_(self.mlp[-1].bias)
        with torch.no_grad():self.mlp[-1].bias[72:96]=-2.

    def forward(self,num,expertos,logits,base,c_num,*texto):
        v=self.mlp(num);delta=self.epsilon*torch.sigmoid(v[:,72:96,None])*torch.tanh(v[:,:72].reshape(-1,24,3))
        w=(logits+delta).softmax(-1);c=c_num+v[:,96:].reshape(-1,24,3)
        return (w*expertos).sum(-1),c,w,delta


def construir(ck):
    clase=SelectorNumerico if ck['variante']=='M3_N' else SelectorTexto
    red=clase(ck['n_num'],ck['n_feat'],ck['n_fuentes'],ck['configuracion']).cuda()
    return red


def ajustar_texto(meta,texto,cfg,s,a,es,ev,variante,car,q,progreso):
    if (car/'TERMINADO.json').exists():return leer(car/'TERMINADO.json')
    motor.fijar_semilla(s);mu=meta['num'][a].mean(0);sd=meta['num'][a].std(0);sd=np.where(sd>1e-6,sd,1.)
    mapa=texto.fuentes_ajuste(meta['dia'][a],variante)
    ck=dict(variante=variante,configuracion=cfg,semilla=s,media=mu,escala=sd,mapa_fuentes=mapa,
            n_num=meta['num'].shape[1],n_feat=texto.k['feat'].shape[2],n_fuentes=int(mapa.max()),
            ajuste=[meta['dia'][a[0]],meta['dia'][a[-1]]],parada=[meta['dia'][es[0]],meta['dia'][es[-1]]])
    red=construir(ck)
    t={k:torch.as_tensor(meta[k],device='cuda') for k in ['y','pred_base','expertos','logits','c_num','c_real']}
    t['num']=torch.as_tensor((meta['num']-mu)/sd,device='cuda')
    def salida(j):
        return red(t['num'][j],t['expertos'][j],t['logits'][j],t['pred_base'][j],t['c_num'][j],*texto.lote(meta['dia'][j],variante,mapa))
    def perdida(ix):
        j=a[ix.cpu().numpy()];pr,c,w,delta=salida(j)
        principal=(pr-t['y'][j]).abs().mean()/q
        return principal+.1*nn.functional.smooth_l1_loss(c,t['c_real'][j])+cfg['lambda_penalizacion_delta']*delta.square().mean()+.001*(w[:,1:]-w[:,:-1]).square().mean()
    def evaluar():
        total=0.
        for i in range(0,len(es),128):
            j=es[i:i+128];pr=salida(j)[0];total+=float((pr-t['y'][j]).abs().mean()/q)*len(j)
        return total/len(es)
    info=ajustar(red,len(a),perdida,evaluar,dict(lr=.001,weight_decay=.0001,lote=64),
                 dict(max_epocas=100,min_epocas=10,paciencia=10,min_delta=.0001),s,car,progreso,
                 f'{variante} cfg{cfg["id"]} s{s} · parada{meta["anio"][es[0]]}')
    ck['modelo']=red.state_dict();checkpoint(car/'modelo.pt',ck)
    guardar(car/'escaladores.json',dict(media=mu.tolist(),escala=sd.tolist(),mapa_fuentes=mapa.tolist(),ajuste=ck['ajuste']))
    otra=construir(ck);otra.load_state_dict(cargar(car/'modelo.pt')['modelo']);otra.eval()
    j=es[:2];args=(t['num'][j],t['expertos'][j],t['logits'][j],t['pred_base'][j],t['c_num'][j],*texto.lote(meta['dia'][j],variante,mapa))
    with torch.no_grad():assert torch.equal(red(*args)[0],otra(*args)[0]),'Restauración M3 no idéntica'
    r=dict(variante=variante,configuracion=cfg,semilla=s,n_parametros=sum(p.numel() for p in red.parameters()),
           ajuste=ck['ajuste'],parada=ck['parada'],**{k:v for k,v in info.items() if k!='historia'})
    if len(ev):
        with torch.no_grad():pred=np.concatenate([salida(ev[i:i+128])[0].cpu().numpy() for i in range(0,len(ev),128)])
        r.update(metricas(meta['y'][ev],pred,q));np.savez_compressed(car/'predicciones.npz',dia=meta['dia'][ev],y=meta['y'][ev],pred=pred)
    guardar(car/'TERMINADO.json',r);del red,otra,t;torch.cuda.empty_cache()
    return r


def predecir(meta,texto,modelo,q):
    ck=cargar(modelo);red=construir(ck);red.load_state_dict(ck['modelo']);red.eval()
    t={k:torch.as_tensor(meta[k],device='cuda') for k in ['pred_base','expertos','logits','c_num']}
    t['num']=torch.as_tensor((meta['num']-ck['media'])/ck['escala'],device='cuda')
    pp=[];ww=[];dd=[]
    with torch.no_grad():
        for i in range(0,len(meta['dia']),128):
            j=slice(i,i+128);args=texto.lote(meta['dia'][j],ck['variante'],ck['mapa_fuentes'])
            p,c,w,d=red(t['num'][j],t['expertos'][j],t['logits'][j],t['pred_base'][j],t['c_num'][j],*args)
            if ck['variante']!='M3_N':
                sin=~args[-1].any(-1)
                assert torch.equal(p[sin],t['pred_base'][j][sin]),'M3 no preserva M1 sin texto'
            assert torch.isfinite(p).all() and torch.allclose(w.sum(-1),torch.ones_like(p),atol=2e-7)
            assert d.abs().max()<=ck['configuracion']['epsilon_logits']+1e-7
            pp.append(p.cpu().numpy());ww.append(w.cpu().numpy());dd.append(d.cpu().numpy())
    del red,t;torch.cuda.empty_cache()
    return np.concatenate(pp),np.concatenate(ww),np.concatenate(dd)


def estadistica(errores,anios,previas):
    pares=[('M3_O','M1'),('M3_OP','M1'),('M3_OP','M3_metadatos'),('M3_OP','M3_desalineado_promedio'),('M3_OP','M3_O')]
    filas=[]
    for bloque in [14,7,28]:
        idx=indices_bootstrap(anios,9999,bloque,20261122);actuales=[]
        for a,b in pares:
            ea,eb=errores[a],errores[b];g=100*(eb.mean()-ea.mean())/eb.mean()
            gb=100*(eb[idx].mean(1)-ea[idx].mean(1))/eb[idx].mean(1);lo,hi=np.percentile(gb,[2.5,97.5])
            for m in [0.,3.]:
                p=float((1+np.sum(gb-g>=g-m))/(len(gb)+1))
                actuales.append(dict(A=a,B=b,bloque_dias=bloque,margen_pct=m,reduccion_pct=g,ic95_inf=lo,ic95_sup=hi,p_unilateral=p))
        prev=previas[previas.bloque_dias==bloque].to_dict('records');assert len(prev)==8
        conjunto=prev+actuales;cotas=holm([f['p_unilateral'] for f in conjunto]+[1.]*10)
        for f,cota in zip(conjunto,cotas):f['cota_superior_p_holm_global_28']=cota
        filas+=conjunto
    return pd.DataFrame(filas)


def ejecutar(out,datos,diseno,run,prueba_m1,m2,progreso,completar,log):
    from probar_m1_gpu import rasgos
    from e3_modelo1 import complementariedad
    semillas=diseno['validacion']['semillas'];m1=run/'M1'
    preparar_oof(datos,m1,out,semillas,progreso,completar,log)
    textos={col:Texto(run,col) for col in ['disponible_declarado','disponible_estricto']}
    g=diseno['modelos']['M3']['rejilla']
    rejilla=[dict(id=i+1,d_texto=d,lambda_penalizacion_delta=l,epsilon_logits=e) for i,(d,l,e) in enumerate(itertools.product(g['d_texto'],g['lambda_penalizacion_delta'],g['epsilon_logits']))]
    guardar(out/'REJILLA_REGISTRADA.json',rejilla);filas=[]
    for s in semillas:
        meta=leer_base(out,s)
        for anio in diseno['validacion']['folds_externos']:
            a,es,ev=dividir(meta,anio)
            for cfg in rejilla:
                progreso(fase='busqueda_m3',etiqueta=f'M3_OP cfg{cfg["id"]} {anio} s{s}')
                car=out/'busqueda'/f'cfg_{cfg["id"]}'/f'anio_{anio}'/f'semilla_{s}'
                r=ajustar_texto(meta,textos['disponible_declarado'],cfg,s,a,es,ev,'M3_OP',car,datos.q168,progreso)
                filas.append(dict(cfg=cfg['id'],anio=anio,semilla=s,MASE=r['MASE'],n_parametros=r['n_parametros']));completar()
    tabla=pd.DataFrame(filas);tabla.to_csv(out/'seleccion_por_fold_semilla.csv',index=False)
    sel=tabla.groupby('cfg').agg(MASE=('MASE','mean'),n_parametros=('n_parametros','mean')).reset_index().sort_values(['MASE','n_parametros','cfg'])
    sel.to_csv(out/'seleccion_resumen.csv',index=False);cfg=next(p for p in rejilla if p['id']==int(sel.iloc[0]['cfg']))
    guardar(out/'CONFIGURACION_ELEGIDA.json',dict(configuracion=cfg,criterio='MASE medio de folds y semillas, solo M3_OP; sin prueba'))
    for s in semillas:
        meta=leer_base(out,s)
        for anio in diseno['validacion']['folds_externos']:
            a,es,ev=dividir(meta,anio)
            for v in VARIANTES[1:]:
                col='disponible_estricto' if v.endswith('_estricto') else 'disponible_declarado'
                progreso(fase='controles_m3',etiqueta=f'{v} {anio} s{s}')
                ajustar_texto(meta,textos[col],cfg,s,a,es,ev,v,out/'controles'/v/f'anio_{anio}'/f'semilla_{s}',datos.q168,progreso);completar()
        a,es,ev=dividir(meta,2022)
        for v in VARIANTES:
            col='disponible_estricto' if v.endswith('_estricto') else 'disponible_declarado'
            progreso(fase='ajuste_final_m3',etiqueta=f'{v} final s{s}')
            ajustar_texto(meta,textos[col],cfg,s,a,es,ev,v,out/'final'/v/f'semilla_{s}',datos.q168,progreso);completar()
    finales=list((out/'final').rglob('modelo.pt'));assert len(finales)==55
    huellas={str(p.relative_to(out)):sha(p) for p in finales}
    guardar(out/'CONGELACION_PREPRUEBA.json',dict(fecha=pd.Timestamp.now(tz='America/Bogota').isoformat(),huellas=huellas,configuracion=cfg))
    del textos;torch.cuda.empty_cache()
    textos={col:Texto(run,col,True) for col in ['disponible_declarado','disponible_estricto']}
    prueba=out/'prueba';prueba.mkdir(exist_ok=True);filas=[];anuales=[];horarios=[];todas={};cobertura=[]
    for s in semillas:
        progreso(fase='prueba_m3',etiqueta=f'Prueba M3 y controles s{s}',epoca=None)
        hist=datos_meta(datos,m1,s)
        with np.load(prueba_m1/f'predicciones_semilla_{s}.npz') as z:
            dias=z['dia'];y=z['y'];expertos=np.stack([z[k] for k in ['patchtst','E','periodica']],-1)
            baseref=z['M1'];preds={k:z[k] for k in ['NNLS','patchtst']}
        dd=np.concatenate([hist['dia'],dias]);yy=np.concatenate([hist['y'],y]);ee=np.concatenate([hist['pred'],expertos]);n=len(hist['dia'])
        mm=dict(dia=dd,y=yy,pred=ee,c=complementariedad(yy,ee,datos.q168),x=rasgos(datos,dd,yy,ee))
        meta=inferir_base(mm,np.arange(n,len(dd)),m1/'final'/f'semilla_{s}'/'modelo.pt',out/'base_final'/f'semilla_{s}.npz');completar()
        assert np.allclose(meta['pred_base'],baseref,rtol=1e-6,atol=3)
        preds['M1']=meta['pred_base'];pesos={};deltas={}
        for v in VARIANTES:
            col='disponible_estricto' if v.endswith('_estricto') else 'disponible_declarado'
            p,w,d=predecir(meta,textos[col],out/'final'/v/f'semilla_{s}'/'modelo.pt',datos.q168)
            preds[v]=p;pesos['pesos_'+v]=w;deltas['delta_'+v]=d
            f=textos[col].filas(dias,v);m,_=textos[col].mascara(f,v)
            cobertura.append(dict(variante=v,semilla=s,origenes=len(dias),dias_con_texto=int(m.any(-1).sum()),candidatos_medios=float(m.sum(-1).mean())))
        for nombre,pr in preds.items():
            todas.setdefault(nombre,[]).append(pr);filas.append(dict(modelo=nombre,semilla=s,**metricas(y,pr,datos.q168)))
            for anio in range(2022,2026):
                m=np.char.startswith(dias,str(anio));anuales.append(dict(modelo=nombre,semilla=s,anio=anio,**metricas(y[m],pr[m],datos.q168)))
            for h in range(24):horarios.append(dict(modelo=nombre,semilla=s,hora=h,**metricas(y[:,h],pr[:,h],datos.q168)))
        np.savez_compressed(prueba/f'predicciones_semilla_{s}.npz',dia=dias,y=y,**preds,**pesos,**deltas)
        pd.DataFrame(dict(fecha=np.repeat(dias,24),hora=np.tile(np.arange(24),len(dias)),semilla=s,y_kwh=y.ravel(),**{k:v.ravel() for k,v in preds.items()})).to_csv(prueba/f'predicciones_semilla_{s}.csv.gz',index=False,compression='gzip')
        del hist,mm,meta
    assert all(sha(out/p)==h for p,h in huellas.items())
    pd.DataFrame(cobertura).to_csv(prueba/'cobertura_textual.csv',index=False)
    tabla=pd.DataFrame(filas);tabla.to_csv(prueba/'metricas_por_semilla.csv',index=False)
    pd.DataFrame(anuales).to_csv(prueba/'metricas_por_anio_semilla.csv',index=False);pd.DataFrame(horarios).to_csv(prueba/'metricas_por_hora_semilla.csv',index=False)
    resumen=tabla.groupby('modelo').agg(MASE=('MASE','mean'),DE_MASE=('MASE','std'),MAE_kwh=('MAE_kwh','mean'),RMSE_kwh=('RMSE_kwh','mean')).sort_values('MASE')
    resumen.to_csv(prueba/'resumen_resultados.csv')
    errores={k:np.abs(np.asarray(v,dtype='float64')-y).mean(axis=(0,2))/datos.q168 for k,v in todas.items()}
    errores['M3_desalineado_promedio']=np.mean([errores[f'M3_desalineado_{d}'] for d in DESPLAZAMIENTOS],axis=0)
    progreso(fase='estadistica_m3',etiqueta='Bootstrap, Holm global parcial e informe',epoca=None)
    c=estadistica(errores,pd.DatetimeIndex(dias).year.to_numpy(),pd.read_csv(m2/'prueba/contrastes_globales_parciales.csv'))
    c.to_csv(prueba/'contrastes_globales_parciales.csv',index=False)
    lineas=['# M3 · selector de expertos condicionado por texto','','Prueba retrospectiva 2022–2025:1.461 orígenes diarios,24 horas y cinco semillas. M1 y E5 congelados. Selección exclusiva de M3_OP con desarrollo2018–2021 y aplicación de su configuración a variantes, controles y sensibilidad estricta.','','| Procedimiento | MASE | DE semillas | MAE kWh | RMSE kWh |','|---|---:|---:|---:|---:|']
    for k,r in resumen.iterrows():lineas.append(f'| {k} | {r.MASE:.6f} | {r.DE_MASE:.6f} | {r.MAE_kwh:.2f} | {r.RMSE_kwh:.2f} |')
    lineas+=['','La base de entrenamiento se obtiene por OOF anidado de M1 desde2013: cada configuración se elige solo con folds anteriores a la fila predicha. El primer prefijo usa cfg1 predefinida al no existir un fold previo. No se usa retrospectivamente el ganador global2018–2021.','','Se adopta el contrato textual específico de §5 del diseño Markdown: fragmentos1500 caracteres, solape200, máximo16; reservas32 alineados,48 oficiales y48 prensa, con ventanas72h y alineados120 días. Este detalle difiere del bloque genérico encoder del JSON; la decisión queda registrada antes de entrenar. E5 y vectores existentes se reutilizan con SHA-256. Las fuentes se mapean solo en ajuste de cada fold; las nuevas se tratan como desconocidas. No se ajusta E5.','','Dos agregadores separados con atención condicionada por entradas numéricas. Perturbación de logits acotada por epsilon y compuerta aprendida; sin texto, identidad exacta con M1. Coeficiente de complementariedad0,1 y suavidad0,001 fijos; penalización de delta según rejilla. El control numérico aproxima el número de parámetros activos mediante una MLP independiente. Las variantes desplazadas usan días anteriores reales, nunca texto futuro.','','Bootstrap por origen,9999 réplicas, bloques14 con sensibilidad7/28 dentro del año. Las semillas se promedian en pérdidas antes de remuestrear. Se conservan18 hipótesis observadas de M1/M2/M3; quedan10 de M4 pendientes. Cota conservadora de Holm global completada con unos; IC95 puntuales. El control desplazado se agrega promediando pérdidas, no predicciones.','','Los resultados del benchmark ya se conocían al diseñar: evaluación retrospectiva informada. No se afirma aporte semántico por superar únicamente una base débil. La sensibilidad estricta se mantiene separada. No se publica ni despliega automáticamente.','']
    (out/'INFORME.md').write_text('\n'.join(lineas),encoding='utf-8')
    guardar(out/'FINALIZADO.json',dict(fecha=pd.Timestamp.now(tz='America/Bogota').isoformat(),entrenamiento=True,prueba=True,configuracion=cfg,semillas=semillas,checkpoints_finales=55,tareas=665))
    log('M3 completado.\n'+resumen.to_string())

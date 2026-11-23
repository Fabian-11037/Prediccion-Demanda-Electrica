"""M2: energía diaria positiva y perfil horario sobre PatchTST congelado."""
from __future__ import annotations
import itertools
from e3_gpu_comun import *
from e3_modelo1 import dividir
from corrector.evaluacion import indices_bootstrap, holm

MODOS = ['M2', 'solo_energia', 'solo_distribucion', 'directa_24']
EPS = 1e-8
MIN_ANCLA_KWH = 1e-6


class EnergiaPerfil(nn.Module):
    def __init__(self, n, cfg, escala_energia, modo='M2'):
        super().__init__()
        w = cfg['ancho']; self.modo = modo
        self.register_buffer('escala_energia', torch.tensor(float(escala_energia), dtype=torch.float32))
        self.contexto = nn.Sequential(nn.Linear(n, w), nn.GELU(), nn.Dropout(cfg['dropout']), nn.Linear(w,w), nn.GELU())
        if modo in ['M2','solo_energia']:
            self.energia = nn.Linear(w,1)
            nn.init.zeros_(self.energia.weight); nn.init.constant_(self.energia.bias, np.log(np.expm1(1.)))
        if modo in ['M2','solo_distribucion']:
            self.perfil = nn.Linear(w,24)
            nn.init.zeros_(self.perfil.weight); nn.init.zeros_(self.perfil.bias)
        if modo == 'directa_24':
            self.directa = nn.Linear(w,24)
            nn.init.zeros_(self.directa.weight); nn.init.constant_(self.directa.bias, np.log(np.expm1(1.)))

    def forward(self, x, ancla):
        z = self.contexto(x)
        base = ancla.clamp_min(MIN_ANCLA_KWH)
        e0 = base.sum(-1, keepdim=True)
        s0 = base/e0
        if self.modo == 'directa_24':
            y = self.escala_energia/24 * nn.functional.softplus(self.directa(z))
            e = y.sum(-1, keepdim=True)
            return y, e[:,0], y/e
        e = self.escala_energia * nn.functional.softplus(self.energia(z)) if self.modo != 'solo_distribucion' else e0
        s = ((s0+EPS).log()+self.perfil(z)).softmax(-1) if self.modo != 'solo_energia' else s0
        return e*s, e[:,0], s


def contexto_congelado(red, tens):
    """Estado por canal tras los parches + calendario, antes del MLP contextual original."""
    red.eval().requires_grad_(False)
    estados, pronosticos = [], []
    def capturar(modulo, entrada):
        estados.append(entrada[0].detach().cpu().numpy())
    h = red.nucleo.contexto[0].register_forward_pre_hook(capturar)
    try:
        with torch.no_grad():
            for i in range(0,len(tens['x']),256):
                ix = slice(i,i+256)
                pronosticos.append(red(tens['x'][ix],tens['cal_p'][ix],tens['cal_f'][ix],tens['dia'][ix]).cpu().numpy())
    finally:
        h.remove()
    assert not any(p.requires_grad for p in red.parameters())
    return np.concatenate(estados), np.concatenate(pronosticos)


def extraer(datos, origen, s, car, cache, pos=None, referencia=None):
    ckpath = car/'modelo.pt'; huella = sha(ckpath)
    if cache.exists():
        with np.load(cache) as z:
            assert str(z['sha_modelo']) == huella
            return {k:z[k] for k in ['dia','y','pred','estado','nivel']}
    ck = cargar(ckpath); m = ck['metadatos']
    assert m['nombre']=='patchtst' and m['semilla']==s
    if pos is None:
        pos = RS.pliegue_anual(datos,origen)['ev']
    assert pd.Timestamp(m['parada'][-1]).year < (origen if isinstance(origen,int) else 2022)
    c = datos.construir(pos,ck['escalas']); t = motor.a_tensores(c,'cuda')
    red = construir_experto(datos,'patchtst',m['configuracion']).cuda()
    red.load_state_dict(ck['modelo'])
    estado,pred = contexto_congelado(red,t)
    pred = (pred*c.nivel[:,None]).astype('float32')
    r = dict(dia=c.origen.strftime('%Y-%m-%d').to_numpy().astype('U10'),y=c.y.astype('float32'),
             pred=pred,estado=estado.astype('float32'),nivel=c.nivel.astype('float32'))
    if referencia is not None:
        with np.load(referencia) as ref:
            assert np.array_equal(r['dia'],ref['dia']) and np.array_equal(r['y'],ref['y'])
            rp = ref['patchtst'] if 'patchtst' in ref else ref['pred']
            assert np.allclose(pred,rp,rtol=2e-6,atol=3.), 'La restauración del ancla no reproduce su corrida original'
    cache.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(cache,**r,sha_modelo=np.array(huella))
    del red,t,c;torch.cuda.empty_cache()
    return r


def unir(partes, q):
    r = {k:np.concatenate([p[k] for p in partes]) for k in partes[0]}
    assert len(np.unique(r['dia']))==len(r['dia'])
    errores = (r['y']-r['pred'])/q
    lag = lags_diarios(r['dia'],errores)
    r['x'] = np.concatenate([r['estado'],r['pred']/r['nivel'][:,None],lag],1).astype('float32')
    r['anio'] = pd.DatetimeIndex(r['dia']).year.to_numpy()
    assert np.isfinite(r['x']).all()
    return r


def diagnosticos(y,pr,q):
    ey,ep = y.sum(-1),pr.sum(-1)
    sy = y/ey[:,None]; sp = pr/ep[:,None]
    return dict(**metricas(y,pr,q),MAE_energia_kwh=float(np.abs(ey-ep).mean()),
                MASE_energia=float(np.abs(ey-ep).mean()/(24*q)),
                error_perfil_L1=float(np.abs(sy-sp).sum(-1).mean()))


def ajustar_cabezas(meta,cfg,s,a,es,ev,modo,car,q,progreso):
    if (car/'TERMINADO.json').exists():return leer(car/'TERMINADO.json')
    motor.fijar_semilla(s)
    mu=meta['x'][a].mean(0);sd=meta['x'][a].std(0);sd=np.where(sd>1e-6,sd,1.)
    escala_e=float(meta['y'][a].sum(-1).mean())
    x=torch.as_tensor((meta['x']-mu)/sd,device='cuda')
    ancla=torch.as_tensor(meta['pred'],device='cuda'); y=torch.as_tensor(meta['y'],device='cuda')
    red=EnergiaPerfil(x.shape[1],cfg,escala_e,modo).cuda()
    ia=torch.as_tensor(a,device='cuda')
    def perdida(ix):
        j=ia[ix];pr,e,_=red(x[j],ancla[j])
        return (pr-y[j]).abs().mean()/q + cfg['lambda_energia']*(e-y[j].sum(-1)).abs().mean()/(24*q)
    def evaluar():
        total=0.
        for i in range(0,len(es),256):
            j=es[i:i+256];pr=red(x[j],ancla[j])[0]
            total+=float((pr-y[j]).abs().mean()/q)*len(j)
        return total/len(es)
    info=ajustar(red,len(a),perdida,evaluar,dict(lr=.001,weight_decay=.0001,lote=64),
                 dict(max_epocas=100,min_epocas=10,paciencia=10,min_delta=.0001),s,car,progreso,
                 f'{modo} cfg{cfg["id"]} s{s} · parada {meta["anio"][es[0]]}')
    ck=dict(modelo=red.state_dict(),media=mu,escala=sd,escala_energia=escala_e,configuracion=cfg,modo=modo,
            semilla=s,n_entrada=x.shape[1],ajuste=[meta['dia'][a[0]],meta['dia'][a[-1]]],parada=[meta['dia'][es[0]],meta['dia'][es[-1]]])
    checkpoint(car/'modelo.pt',ck)
    guardar(car/'escaladores.json',dict(media=mu.tolist(),escala=sd.tolist(),escala_energia=escala_e,ajuste=ck['ajuste']))
    otra=restaurar(car/'modelo.pt')
    with torch.no_grad():
        assert torch.equal(red(x[es[:2]],ancla[es[:2]])[0],otra(x[es[:2]],ancla[es[:2]])[0]),'Restauración no idéntica'
        pr=np.concatenate([otra(x[ev[i:i+256]],ancla[ev[i:i+256]])[0].cpu().numpy() for i in range(0,len(ev),256)]) if len(ev) else None
    r=dict(configuracion=cfg,modo=modo,semilla=s,n_parametros=sum(p.numel() for p in red.parameters()),
           ajuste=ck['ajuste'],parada=ck['parada'],**{k:v for k,v in info.items() if k!='historia'})
    if pr is not None:
        r.update(diagnosticos(meta['y'][ev],pr,q))
        np.savez_compressed(car/'predicciones.npz',dia=meta['dia'][ev],y=meta['y'][ev],pred=pr)
    guardar(car/'TERMINADO.json',r)
    del red,otra,x,ancla,y;torch.cuda.empty_cache()
    return r


def restaurar(p):
    ck=cargar(p);red=EnergiaPerfil(ck['n_entrada'],ck['configuracion'],ck['escala_energia'],ck['modo']).cuda().eval()
    red.load_state_dict(ck['modelo'])
    return red


def contrastar(errores,anios,anteriores):
    filas=[]
    for b in [14,7,28]:
        idx=indices_bootstrap(anios,9999,b,20261122);f=[]
        for ref in ['patchtst','M1']:
            ea,eb=errores['M2'],errores[ref]
            g=100*(eb.mean()-ea.mean())/eb.mean()
            gb=100*(eb[idx].mean(1)-ea[idx].mean(1))/eb[idx].mean(1)
            lo,hi=np.percentile(gb,[2.5,97.5])
            for margen in [0.,3.]:
                p=float((1+np.sum(gb-g>=g-margen))/(len(gb)+1))
                f.append(dict(A='M2',B=ref,bloque_dias=b,margen_pct=margen,reduccion_pct=g,
                              ic95_inf=lo,ic95_sup=hi,p_unilateral=p))
        prev=anteriores[anteriores.bloque_dias==b].to_dict('records')
        assert len(prev)==4
        cotas=holm([r['p_unilateral'] for r in prev+f]+[1.]*20)
        for r,cota in zip(prev+f,cotas[:8]):
            r['cota_superior_p_holm_global_28']=cota
            filas.append(r)
        del idx
    return pd.DataFrame(filas)


def ejecutar(out,datos,diseno,run_m1,prueba_m1,progreso,completar,log):
    out=Path(out);semillas=diseno['validacion']['semillas']
    g=diseno['modelos']['M2']['rejilla']
    cfgs=[dict(id=i+1,ancho=w,dropout=dp,lambda_energia=lam) for i,(w,dp,lam) in enumerate(itertools.product(g['ancho'],g['dropout'],g['lambda_energia']))]
    guardar(out/'REJILLA_REGISTRADA.json',cfgs)
    filas=[]
    for s in semillas:
        partes=[]
        for anio in range(2011,2022):
            car=run_m1/'expertos/patchtst'/f'anio_{anio}'/f'semilla_{s}'
            progreso(fase='extraccion_oof',etiqueta=f'PatchTST congelado {anio} s{s}',epoca=None)
            partes.append(extraer(datos,anio,s,car,out/'cache'/f'anio_{anio}_semilla_{s}.npz',referencia=car/'predicciones.npz'))
            completar()
        meta=unir(partes,datos.q168)
        # Comprobación causal en desarrollo; no se accede a prueba.
        mod=[{**p,'y':p['y'].copy()} for p in partes]
        corte=np.flatnonzero(meta['anio']==2019)[0]
        for p in mod:
            p['y'][p['dia']>='2019-01-01']+=1e7
        alterado=unir(mod,datos.q168)
        assert np.array_equal(meta['x'][:corte+2],alterado['x'][:corte+2])
        del alterado,mod
        for anio in diseno['validacion']['folds_externos']:
            a,es,ev=dividir(meta,anio)
            for cfg in cfgs:
                progreso(fase='busqueda',etiqueta=f'M2 cfg{cfg["id"]} {anio} s{s}')
                r=ajustar_cabezas(meta,cfg,s,a,es,ev,'M2',out/'busqueda'/f'cfg_{cfg["id"]}'/f'anio_{anio}'/f'semilla_{s}',datos.q168,progreso)
                filas.append(dict(cfg=cfg['id'],anio=anio,semilla=s,MASE=r['MASE'],n_parametros=r['n_parametros']));completar()
        del meta,partes
    tabla=pd.DataFrame(filas);tabla.to_csv(out/'seleccion_por_fold_semilla.csv',index=False)
    sel=tabla.groupby('cfg').agg(MASE=('MASE','mean'),n_parametros=('n_parametros','first')).reset_index().sort_values(['MASE','n_parametros','cfg'])
    sel.to_csv(out/'seleccion_resumen.csv',index=False)
    elegida=next(c for c in cfgs if c['id']==int(sel.iloc[0]['cfg']))
    guardar(out/'CONFIGURACION_ELEGIDA.json',dict(configuracion=elegida,seleccion='Media de MASE de cuatro folds y cinco semillas, sin prueba'))
    log('Configuración seleccionada con desarrollo: '+str(elegida))
    for s in semillas:
        partes=[]
        for a0 in range(2011,2022):
            with np.load(out/'cache'/f'anio_{a0}_semilla_{s}.npz') as z:partes.append({k:z[k] for k in ['dia','y','pred','estado','nivel']})
        meta=unir(partes,datos.q168)
        for anio in diseno['validacion']['folds_externos']:
            a,es,ev=dividir(meta,anio)
            for modo in MODOS[1:]:
                progreso(fase='ablaciones',etiqueta=f'{modo} {anio} s{s}')
                ajustar_cabezas(meta,elegida,s,a,es,ev,modo,out/'controles'/modo/f'anio_{anio}'/f'semilla_{s}',datos.q168,progreso);completar()
        a,es,ev=dividir(meta,2022)
        for modo in MODOS:
            progreso(fase='ajuste_final',etiqueta=f'{modo} final s{s}')
            ajustar_cabezas(meta,elegida,s,a,es,ev,modo,out/'final'/modo/f'semilla_{s}',datos.q168,progreso);completar()
        del meta,partes
    # Todos los pesos y escaladores finales se congelan ANTES de abrir la prueba.
    finales=list((out/'final').rglob('modelo.pt'))
    assert len(finales)==20
    congelados={str(p.relative_to(out)):sha(p) for p in finales}
    guardar(out/'CONGELACION_PREPRUEBA.json',dict(fecha=pd.Timestamp.now(tz='America/Bogota').isoformat(),configuracion=elegida,huellas=congelados))
    pos=datos.prueba();assert len(pos)==1461
    prueba=out/'prueba';prueba.mkdir(exist_ok=True)
    filas=[];anuales=[];horarios=[];todas={}
    for s in semillas:
        progreso(fase='prueba',etiqueta=f'Prueba 2022–2025 s{s}',epoca=None)
        car=run_m1/'expertos_finales/patchtst'/f'semilla_{s}'
        test=extraer(datos,'final',s,car,out/'cache'/f'final_semilla_{s}.npz',pos=pos,referencia=prueba_m1/f'predicciones_semilla_{s}.npz');completar()
        partes=[]
        for a0 in range(2011,2022):
            with np.load(out/'cache'/f'anio_{a0}_semilla_{s}.npz') as z:partes.append({k:z[k] for k in ['dia','y','pred','estado','nivel']})
        partes.append(test);meta=unir(partes,datos.q168);ix=np.flatnonzero(meta['anio']>=2022)
        dias=meta['dia'][ix];y=meta['y'][ix]
        with np.load(prueba_m1/f'predicciones_semilla_{s}.npz') as z:
            assert np.array_equal(dias,z['dia']) and np.array_equal(y,z['y'])
            preds={k:z[k] for k in ['M1','NNLS','patchtst']}
        energias={};perfiles={}
        for modo in MODOS:
            p=out/'final'/modo/f'semilla_{s}'/'modelo.pt';ck=cargar(p);red=restaurar(p)
            x=torch.as_tensor((meta['x'][ix]-ck['media'])/ck['escala'],device='cuda')
            ancla=torch.as_tensor(meta['pred'][ix],device='cuda');pp=[];ee=[];ss=[]
            with torch.no_grad():
                for i in range(0,len(ix),256):
                    pr,e,sh=red(x[i:i+256],ancla[i:i+256]);pp.append(pr.cpu().numpy());ee.append(e.cpu().numpy());ss.append(sh.cpu().numpy())
            preds[modo]=np.concatenate(pp);energias['energia_'+modo]=np.concatenate(ee);perfiles['perfil_'+modo]=np.concatenate(ss)
            assert np.isfinite(preds[modo]).all() and (preds[modo]>0).all()
            assert np.allclose(preds[modo].sum(-1),energias['energia_'+modo],rtol=3e-7)
            del red,x,ancla
        for nombre,pr in preds.items():
            todas.setdefault(nombre,[]).append(pr)
            filas.append(dict(modelo=nombre,semilla=s,**diagnosticos(y,pr,datos.q168)))
            for anio in [2022,2023,2024,2025]:
                m=np.char.startswith(dias,str(anio));anuales.append(dict(modelo=nombre,semilla=s,anio=anio,**diagnosticos(y[m],pr[m],datos.q168)))
            for h in range(24):horarios.append(dict(modelo=nombre,semilla=s,hora=h,**metricas(y[:,h],pr[:,h],datos.q168)))
        np.savez_compressed(prueba/f'predicciones_semilla_{s}.npz',dia=dias,y=y,**preds,**energias,**perfiles)
        pd.DataFrame(dict(fecha=np.repeat(dias,24),hora=np.tile(np.arange(24),len(dias)),semilla=s,y_kwh=y.ravel(),**{k:v.ravel() for k,v in preds.items()})).to_csv(prueba/f'predicciones_semilla_{s}.csv.gz',index=False,compression='gzip')
        del meta,partes
    assert all(sha(out/p)==h for p,h in congelados.items())
    tabla=pd.DataFrame(filas);tabla.to_csv(prueba/'metricas_por_semilla.csv',index=False)
    pd.DataFrame(anuales).to_csv(prueba/'metricas_por_anio_semilla.csv',index=False)
    pd.DataFrame(horarios).to_csv(prueba/'metricas_por_hora_semilla.csv',index=False)
    resumen=tabla.groupby('modelo').agg(MASE=('MASE','mean'),DE_MASE=('MASE','std'),MAE_kwh=('MAE_kwh','mean'),RMSE_kwh=('RMSE_kwh','mean'),MAE_energia_kwh=('MAE_energia_kwh','mean'),error_perfil_L1=('error_perfil_L1','mean')).sort_values('MASE')
    resumen.to_csv(prueba/'resumen_resultados.csv')
    errores={k:np.abs(np.asarray(v,dtype='float64')-y).mean(axis=(0,2))/datos.q168 for k,v in todas.items()}
    progreso(fase='estadistica',etiqueta='Bootstrap por bloques e informe',epoca=None)
    c=contrastar(errores,pd.DatetimeIndex(dias).year.to_numpy(),pd.read_csv(prueba_m1/'contrastes.csv'))
    c.to_csv(prueba/'contrastes_globales_parciales.csv',index=False)
    lineas=['# M2 · energía diaria y distribución horaria','','Prueba retrospectiva 2022–2025, 1.461 días, 24 horas y cinco semillas. Selección exclusiva en desarrollo2018–2021. Checkpoints y escaladores finales congelados antes de prueba. No se actualizan parámetros en prueba.','','| Procedimiento | MASE | DE semillas | MAE kWh | RMSE kWh |','|---|---:|---:|---:|---:|']
    for k,r in resumen.iterrows():lineas.append(f'| {k} | {r.MASE:.6f} | {r.DE_MASE:.6f} | {r.MAE_kwh:.2f} | {r.RMSE_kwh:.2f} |')
    lineas+=['','Se conservan diagnósticos separados de energía y perfil. La conservación de energía y la positividad son propiedades estructurales, no garantías de precisión. Las ablaciones mantienen la configuración seleccionada del modelo completo.','','El codificador PatchTST se restaura congelado por año/semilla. Su estado de contexto y calendario se conserva en el caché OOF; cada año procede de un codificador estimado solo en años anteriores. Las bases latentes pueden variar entre modelos anuales; es una limitación de esta implementación OOF. Las escalas de las cabezas se estiman solo en ajuste. Los errores usados como entrada llevan retraso de dos días.','','Bootstrap pareado por origen:9.999 réplicas, bloques14 y sensibilidad7/28, dentro de cada año. Las semillas se promedian en la pérdida por origen. IC95 puntuales. La familia prevista es de28 hipótesis: se reúnen las cuatro previas de M1 y las cuatro nuevas de M2; las20 restantes están pendientes. Se informa cota conservadora de Holm con20 valores uno adicionales, no un ajuste definitivo de todos los modelos. M1 y NNLS son checkpoints existentes; no se reentrenan ni se incluyen como entradas de M2. No se necesita OOF anidado de M1 para este modelo.','','Estos años ya se conocían durante el diseño: evaluación retrospectiva informada. No se afirma originalidad ni superioridad sin contraste. Resultados locales; sin publicación automática.','']
    (out/'INFORME.md').write_text('\n'.join(lineas),encoding='utf-8')
    guardar(out/'FINALIZADO.json',dict(fecha=pd.Timestamp.now(tz='America/Bogota').isoformat(),entrenamiento=True,prueba=True,ajustes_neuronales=240,extracciones_congeladas=60,configuracion=elegida))
    log('M2 completado.\n'+resumen.to_string())

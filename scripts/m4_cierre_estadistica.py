"""Inferencia temporal registrada: pérdidas diarias, semillas y multiplicidad separadas."""
from e3_gpu_comun import *
from scipy import stats
from corrector.evaluacion import holm

def hac(z,lag=13):
    z=np.asarray(z,dtype='float64');z=z-z.mean(0);n=len(z)
    if z.ndim==1:z=z[:,None]
    v=z.T@z/n
    for j in range(1,min(lag,n-2)+1):
        c=z[j:].T@z[:-j]/n;v+=(1-j/(lag+1))*(c+c.T)
    return v

def estacionario(n,B=9999,L=14,seed=20261123):
    rng=np.random.default_rng(seed);ix=np.empty((B,n),dtype=np.int32);ix[:,0]=rng.integers(n,size=B)
    for t in range(1,n):ix[:,t]=np.where(rng.random(B)<1/L,rng.integers(n,size=B),(ix[:,t-1]+1)%n)
    return ix

def potencia_previa(out):
    # Exclusivamente validación OOF 2018–2021; nunca pérdidas de prueba.
    d=REPO/'results/m4_gpu_20261122_194500';m3=REPO/'results/m3_gpu_20261122_193240';la=[];lb=[]
    for s in range(101,106):
        ea=[];eb=[]
        for a in range(2018,2022):
            with np.load(d/f'controles/M4_N/anio_{a}/semilla_{s}/predicciones.npz') as z:ea.append(np.abs(z['y']-z['pred']).mean(1)/215617.90361589016)
            with np.load(m3/f'base_oof/anio_{a}_semilla_{s}.npz') as z:eb.append(np.abs(z['y']-z['pred_base']).mean(1)/215617.90361589016)
        la.append(np.concatenate(ea));lb.append(np.concatenate(eb))
    la,lb=np.asarray(la),np.asarray(lb);dif=lb-la;n=dif.shape[1]
    vt=max(float(hac(dif.mean(0))[0,0]),1e-12);vs=float(dif.mean(1).var(ddof=1));ref=float(lb.mean())
    # Aproximación de planificación (no certificación): componente temporal HAC
    # más varianza entre inicializaciones; Bonferroni conservador de 768 H0.
    familia=768;crit=stats.norm.isf(.05/familia);filas=[]
    for ns in [10,15,20,30,50,100,1000000000]:
        for nd in [365,730,1461,2922,5844]:
            se=np.sqrt(vt/nd+vs/ns)
            for margen,alternativa in [(0.,.03),(.03,.06)]:
                senal=(alternativa-margen)*ref;p=float(stats.norm.cdf(senal/se-crit))
                filas.append(dict(semillas=ns,dias=nd,margen=margen,alternativa=alternativa,potencia_aproximada=p,alpha_bonferroni=.05/familia))
    pd.DataFrame(filas).to_csv(out/'potencia_planificacion.csv',index=False)
    guardar(out/'DIMENSIONAMIENTO.json',dict(fecha=pd.Timestamp.now(tz='America/Bogota').isoformat(),fuente='OOF2018-2021 M4_N y M1, semillas101-105; sin prueba',dias=n,varianza_temporal_HAC=vt,varianza_semillas=vs,semillas_ejecucion=10,objetivo=.9,
        alcance='Diagnóstico aproximado condicionado a cinco semillas de piloto. No certifica potencia por rival/dataset ni potencia conjunta. No se infiere potencia a priori de p-valores de prueba. El límite temporal puede impedir 0.90 aun con infinitas semillas.',
        ampliar='No añadir semillas sin evidencia de que resuelve el límite temporal; conservar mínimo10 y reportar si falta potencia. Confirmación prospectiva no disponible hoy.'))

def comparar(resultados,out):
    filas=[];metricas=[];variacion=[]
    for dataset,(dias,y,preds,q) in resultados.items():
        errores={k:np.abs(np.asarray(v,dtype='float64')-y).mean(-1)/q for k,v in preds.items()}
        # Forma siempre semillas x días; deterministas permanecen con una fila.
        for k,e in errores.items():
            for s,l in enumerate(e):metricas.append(dict(dataset=dataset,modelo=k,realizacion=s,MASE=float(l.mean())))
            variacion.append(dict(dataset=dataset,modelo=k,semillas=len(e),MASE=float(e.mean()),DE_semillas=float(e.mean(1).std(ddof=1)) if len(e)>1 else None))
        objetivo=errores['M4_N'].mean(0);n=len(dias)
        idx=estacionario(n);co=np.asarray([np.ones(n),np.r_[0.,objetivo[:-1]]]).T
        for nombre,e in errores.items():
            if nombre=='M4_N':continue
            rival=e.mean(0);ganancia=1-objetivo.mean()/rival.mean();gb=1-objetivo[idx].mean(1)/rival[idx].mean(1);ic=np.quantile(gb,[.025,.975])
            for margen in [0.,.03]:
                d=(1-margen)*rival-objetivo;z=d-d.mean();vb=max(float(hac(d)[0,0]),1e-30)
                t=d.mean()/np.sqrt(vb/n)*np.sqrt((n-1)/n);pdm=float(stats.t.sf(t,n-1))
                pboot=float((1+np.sum(z[idx].mean(1)>=d.mean()))/10000)
                # GW bilateral con instrumentos prefijados constante y pérdida
                # M4 del origen anterior. HAC13. Rechazo no establece dirección.
                g=co*d[:,None];media=g.mean(0);cov=hac(g);rango=np.linalg.matrix_rank(cov)
                w=float(n*media@np.linalg.pinv(cov)@media);pgw=float(stats.chi2.sf(w,rango)) if rango else 1.
                for metodo,p in [('bootstrap_estacionario',pboot),('DM_HLN',pdm),('GW_bilateral',pgw)]:
                    filas.append(dict(dataset=dataset,A='M4_N',B=nombre,metodo=metodo,margen_relativo=margen,ganancia_relativa=ganancia,ic95_inf=float(ic[0]),ic95_sup=float(ic[1]),p=p,
                        favorable=ganancia>margen,alcance='retrospectivo' if dataset=='colombia' else 'evaluación adicional; validez condicional a supuestos'))
        np.savez_compressed(out/f'perdidas_{dataset}.npz',dia=dias,**errores)
    assert len(filas)<=768,'La familia excede el máximo registrado'
    # Mantener comparaciones no ejecutadas con p=1 para una cota conservadora;
    # no presentarlas como p observados ni reducir multiplicidad por conveniencia.
    ajustados=holm([f['p'] for f in filas]+[1.]*(768-len(filas)))[:len(filas)]
    for f,p in zip(filas,ajustados):f['p_holm_cota_familia768']=float(p)
    pd.DataFrame(filas).to_csv(out/'contrastes_F2.csv',index=False)
    pd.DataFrame(metricas).to_csv(out/'metricas_por_semilla.csv',index=False);pd.DataFrame(variacion).to_csv(out/'resumen_resultados.csv',index=False)
    guardar(out/'LIMITACIONES_INFERENCIA.json',dict(confirmacion=False,razones=['Colombia2022-2025 previamente inspeccionado','La potencia de cinco semillas piloto no acredita0.90 en todos los contrastes','Supuestos de estacionariedad y estabilidad de las reglas de memoria deben verificarse; GW no prueba dirección','Top10 histórico conserva cinco semillas y versiones nativas; no es el benchmark numérico adicional'],familias_historicas_intactas=[28,220],familia_nueva_maxima=768))
    return variacion

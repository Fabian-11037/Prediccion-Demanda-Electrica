"""Diagnósticos y dimensionamiento con OOF; nunca se elige presupuesto con prueba."""
from e3_gpu_comun import *
from scipy import stats
from m4_cierre_estadistica import hac

FAMILIA=1536

def simular_potencia(errores,dias_prueba,car,replicas=2000):
    """Simulación jerárquica de planificación condicionada al piloto OOF.

    Alternativas predefinidas 3% vs cero y 6% vs margen3%. La potencia usa
    un umbral unilateral Bonferroni (conservador frente a Holm), no p de prueba.
    No acredita potencia conjunta de los tres contrastes F2 ni supuestos.
    """
    nombres=sorted(k for k in errores if k!='M4_N');A=np.asarray(errores['M4_N']);S,T=A.shape
    assert S>=10 and T>=60
    assert all(np.asarray(errores[k]).shape==A.shape for k in nombres)
    ref=np.asarray([np.asarray(errores[k]).mean() for k in nombres]);assert (ref>0).all()
    dif=np.stack([np.asarray(errores[k])-A for k in nombres],-1).astype('float64');assert np.isfinite(dif).all()
    # Quitar únicamente el efecto medio para imponer las alternativas fijadas.
    centered=dif-dif.mean(axis=(0,1),keepdims=True)
    temporal=np.diag(hac(centered.mean(0)))/dias_prueba
    between=centered.mean(1).var(0,ddof=1);crit=stats.norm.isf(.05/FAMILIA)
    rng=np.random.default_rng(20261123);filas=[];conjuntas=[];candidatos=[10,15,20,30];seleccion=None
    # Mismos índices por rivales preservan sus dependencias. Muestreo temporal
    # estacionario dentro del soporte OOF, longitud esperada14.
    for ns in candidatos:
        se=np.sqrt(np.maximum(temporal+between/ns,1e-20));rechazos=[]
        for _ in range(replicas):
            ids=np.empty(dias_prueba,dtype='int64');ids[0]=rng.integers(T)
            renovar=rng.random(dias_prueba)<1/14;inicios=rng.integers(T,size=dias_prueba)
            for j in range(1,dias_prueba):ids[j]=inicios[j] if renovar[j] else (ids[j-1]+1)%T
            semillas=rng.integers(S,size=ns);pesos=np.bincount(semillas,minlength=S)/ns
            ruido=(centered[:,ids,:].mean(1)*pesos[:,None]).sum(0)
            rechazos.append((.03*ref+ruido)/se>crit)
        re=np.asarray(rechazos);pot=re.mean(0);cuentas=re.sum(0);alpha_MC=.05/(len(nombres)+1)
        inferiores=np.array([float(stats.beta.ppf(alpha_MC,k,replicas-k+1)) if k else 0. for k in cuentas])
        for k,p,b,t,li in zip(nombres,pot,between,temporal,inferiores):
            filas.append(dict(rival=k,semillas=ns,dias=dias_prueba,potencia_simulada=float(p),limite_inferior_MC95_simultaneo=li,MC_error=float(np.sqrt(p*(1-p)/replicas)),var_temporal=float(t),var_semillas=float(b),alternativas='3% contra0%; 6% contra3%',criterio='Wald unilateral Bonferroni, piloto OOF'))
        exitos=int(re.all(1).sum());joint=exitos/replicas;li_joint=float(stats.beta.ppf(alpha_MC,exitos,replicas-exitos+1)) if exitos else 0.;conjuntas.append(dict(semillas=ns,potencia_conjunta_Wald=joint,limite_inferior_MC95_simultaneo=li_joint,MC_error=float(np.sqrt(joint*(1-joint)/replicas))))
        if seleccion is None and (inferiores>=.90).all() and li_joint>=.90:seleccion=ns
    pd.DataFrame(filas).to_csv(car/'potencia_por_contraste.csv',index=False);pd.DataFrame(conjuntas).to_csv(car/'potencia_conjunta.csv',index=False)
    # No gastar semillas adicionales si ninguno de los presupuestos alcanza el
    # objetivo bajo esta aproximación. El déficit se informa; no se maquilla.
    minimo=10 if seleccion is None else seleccion
    guardar(car/'DIMENSIONAMIENTO_COMPLEMENTARIO.json',dict(semillas_fijadas=minimo,presupuestos=candidatos,replicas=replicas,objetivo_alcanzable_en_grid=seleccion is not None,familia=FAMILIA,
        fuente='Solo pérdidas OOF de desarrollo, diez semillas independientes',alcance='Potencia condicional simulada para Wald/Bonferroni. No certifica 0.90 de bootstrap/DM/GW ni confirmación independiente. Déficit exige periodo mayor o ampliar el diseño; no se amplía por métricas de prueba.',regla='Primero10/15/20/30 con límites inferiores Monte Carlo marginales y conjunto >=0.90; si ninguno alcanza, conservar10 y reportar insuficiencia.'))
    return minimo

def supuestos(archivos,car):
    from statsmodels.tsa.stattools import adfuller,kpss
    from statsmodels.stats.diagnostic import acorr_ljungbox
    import warnings
    filas=[]
    for archivo in archivos:
        with np.load(archivo) as z:d={k:z[k] for k in z.files}
        A=d['M4_N'].mean(0);dias=pd.DatetimeIndex(d['dia'])
        for k in sorted(d):
            if k in ['dia','M4_N']:continue
            serie=d[k].mean(0)-A
            if np.std(serie)<1e-12:
                filas.append(dict(dataset=archivo.stem,comparador=k,estado='diferencia constante; contraste degenerado'));continue
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                adf=float(adfuller(serie,maxlag=min(14,len(serie)//10),regression='c',autolag='AIC')[1]);kp=float(kpss(serie,regression='c',nlags='auto')[1])
            lb=acorr_ljungbox(serie,lags=[7,14,28],return_df=True)
            # Diagnóstico de heterogeneidad; no constituye prueba de ausencia
            # de quiebres ni certifica estacionariedad al no rechazar.
            grupos=[serie[dias.year==a] for a in sorted(set(dias.year))]
            mit=len(serie)//2;drift=float(stats.ttest_ind(serie[:mit],serie[mit:],equal_var=False).pvalue)
            filas.append(dict(dataset=archivo.stem,comparador=k,ADF_p=adf,KPSS_p=kp,LjungBox7_p=float(lb.lb_pvalue.iloc[0]),LjungBox14_p=float(lb.lb_pvalue.iloc[1]),LjungBox28_p=float(lb.lb_pvalue.iloc[2]),cambio_media_mitades_p_exploratorio=drift,medias_anuales=json.dumps([float(v.mean()) for v in grupos]),estado='compatible en diagnósticos marginales; no demuestra supuestos' if adf<.05 and kp>=.05 else 'supuestos de estabilidad cuestionados; inferencia condicional'))
    pd.DataFrame(filas).to_csv(car/'diagnosticos_supuestos.csv',index=False)
    guardar(car/'ALCANCE_SUPUESTOS.json',dict(inferencia_confirmatoria=False,diagnosticos='ADF/KPSS y Ljung-Box exploratorios. Sus no-rechazos no prueban estacionariedad. La memoria creciente y heterogeneidad anual requieren cautela. GW bilateral no acredita dirección.',potencia='El dimensionamiento simulado procede exclusivamente de desarrollo; no es potencia observada a partir de p de prueba.'))

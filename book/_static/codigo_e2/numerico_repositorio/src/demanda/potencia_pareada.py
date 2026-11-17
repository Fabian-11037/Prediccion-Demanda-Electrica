"""Dimensionamiento exploratorio pareado por semilla y bloques temporales."""
import numpy as np

def bootstrap_piloto(losses,referencia,anios,replicas=10000,bloque=14,semilla=20261110):
    """losses: semillas × orígenes; años con igual peso, bloques dentro de cada año."""
    rng=np.random.default_rng(semilla);losses=np.asarray(losses);ref=np.asarray(referencia)
    delta=losses-ref[None,:];boots=np.zeros((replicas,losses.shape[0]));medias=[];ref_mean=[]
    years=np.unique(anios)
    for year in years:
        ids=np.flatnonzero(np.asarray(anios)==year);z=delta[:,ids];n=len(ids)
        medias.append(z.mean(1));ref_mean.append(ref[ids].mean())
        for start in range(0,replicas,200):
            b=min(200,replicas-start);beg=rng.integers(0,n,(b,int(np.ceil(n/bloque))))
            ix=((beg[:,:,None]+np.arange(bloque))%n).reshape(b,-1)[:,:n]
            boots[start:start+b]+=z[:,ix].mean(2).T/len(years)
    return boots,np.mean(medias,axis=0),float(np.mean(ref_mean))

def dimensionar(losses,referencia,anios,replicas=10000):
    boots,seed_means,ref=bootstrap_piloto(losses,referencia,anios,replicas=replicas)
    center=float(seed_means.mean());margin=.03*ref;rng=np.random.default_rng(20261111)
    def power(err):
        lo,hi=np.quantile(err,[.025,.975])
        return float(min(np.mean(err<lo+margin),np.mean(err>hi-margin)))
    floor_error=boots.mean(1)-center;floor_power=power(floor_error)
    base=dict(margen_relativo=.03,margen_absoluto=margin,MASE_referencia=ref,alpha=.05,potencia_objetivo=.8,
        replicas=replicas,bloque_dias=14,semillas_piloto=int(losses.shape[0]),potencia_limite_temporal=floor_power,
        alcance='Estimación bootstrap exploratoria de precisión bajo alternativa mínima; cinco semillas no certifican potencia ni distribución de nuevas inicializaciones')
    trials=[]
    # El límite temporal permanece aunque aumente el número de inicializaciones.
    if floor_power<.8:
        return dict(base,n_final=None,estado='precision_temporal_insuficiente',evaluaciones=trials)
    grid=list(range(5,101))+list(range(110,501,10))+list(range(550,2001,50))
    for n in grid:
        counts=rng.multinomial(n,np.repeat(1/losses.shape[0],losses.shape[0]),size=replicas)
        errors=(boots*counts).sum(1)/n-center;estimated=power(errors);trials.append(dict(n=n,potencia_estimada=estimated))
        if estimated>=.8:return dict(base,n_final=n,estado='dimensionado',evaluaciones=trials)
    return dict(base,n_final=None,estado='no_dimensionado_en_rejilla_hasta_2000',evaluaciones=trials)

"""Distingue capacidad de la firma de columnas nulas en el soporte observado."""
from e3_gpu_comun import *
from e3_modelo3 import leer_base
from e3_modelo1 import dividir
def ejecutar():
    out=REPO/'results/m4_cierre_gpu_20261123';filas=[]
    for s in range(101,106):
        meta=leer_base(REPO/'results/m3_gpu_20261122_193240',s)
        for anio in [2018,2019,2020,2021,2022]:
            a,es,ev=dividir(meta,anio);base=out/'colombia'/('final' if anio==2022 else f'desarrollo/anio_{anio}')
            sc=leer(base/f'M4_N/semilla_{s}/escaladores.json');x=(meta['num'][np.r_[a,es]]-sc['media'])/sc['escala'];z=int((x==0).all(0).sum());n=x.shape[1]
            objetivo=64*n+2625-64*z
            for nombre in ['MLP_CE','EWMA_MLP_CE']:
                ck=cargar(base/f'{nombre}/semilla_{s}/modelo.pt');h=ck['ancho'];p=sum(v.numel() for v in ck['modelo'].values())-h*z
                assert abs(p-objetivo)/objetivo<=.05
                filas.append(dict(semilla=s,anio=anio,control=nombre,columnas_nulas_observadas=z,M4_N_parametros_en_soporte=objetivo,control_parametros_en_soporte=p,diferencia_relativa=abs(p-objetivo)/objetivo))
    pd.DataFrame(filas).to_csv(out/'capacidad_soporte_observado.csv',index=False)
    guardar(out/'CAPACIDAD_ALCANCE.json',dict(firma_numerica_M4_N=60993,contenedor=144225,soporte_observado='Además se excluyen proyecciones de columnas normalizadas exactamente nulas en ajuste/parada. Ambos controles conservan ±5%.',limitacion='Constancia en los datos observados no prueba inactividad estructural en todo el dominio. Se separa del conteo de la firma, que excluye los módulos textuales y64 entradas forzadas a cero por código. Los conteos originales se conservan como conteos de diseño; no se reescriben metadatos ni se retocan arquitecturas.'))
if __name__=='__main__':ejecutar()

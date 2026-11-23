"""Fase de desarrollo/capacidad; nunca abre prueba ni programa apagado."""
from e3_gpu_comun import *
from m4_cierre_controles import ajustar_control, VARIANTES
from m4_cierre_colombia import preparar
from e3_modelo1 import dividir
import argparse

def ejecutar(out,semillas):
    torch.set_num_threads(4);out.mkdir(parents=True,exist_ok=True)
    fuentes=[Path(__file__),REPO/'scripts/m4_cierre_controles.py',REPO/'scripts/m4_cierre_colombia.py']
    reg=out/'REGISTRO_FASE_CAPACIDAD.json'
    actual=dict(fecha=pd.Timestamp.now(tz='America/Bogota').isoformat(),semillas=semillas,fuentes={str(p.relative_to(REPO)):sha(p) for p in fuentes},prueba=False)
    if reg.exists():assert leer(reg)['fuentes']==actual['fuentes']
    else:guardar(reg,actual)
    progreso_car=out/'progreso';progreso_car.mkdir(exist_ok=True);hechas=0;total=5*len(semillas)*len(VARIANTES)
    def progreso(**kw):
        p=dict(estado='Ejecutando',fase='Controles de capacidad · Colombia',tareas=hechas,total=total,gpu=torch.cuda.get_device_name(0),actualizado=pd.Timestamp.now(tz='America/Bogota').isoformat(),**kw)
        guardar(progreso_car/'estado.json',p)
    progreso()
    for s in semillas:
        meta=preparar(s,out/'colombia',progreso)
        for anio in [2018,2019,2020,2021,2022]:
            a,es,ev=dividir(meta,anio)
            for v in VARIANTES:
                progreso(etiqueta=f'{v} · s{s} · {anio}')
                car=out/'colombia'/('final' if anio==2022 else f'desarrollo/anio_{anio}')/v/f'semilla_{s}'
                previo=None
                if v=='M4_N' and s<=105:
                    previo=REPO/'results/m4_gpu_20261122_194500'/('final/M4_N' if anio==2022 else f'controles/M4_N/anio_{anio}')/f'semilla_{s}'/'modelo.pt'
                ajustar_control(meta,s,a,es,ev,v,car,215617.90361589016,progreso,previo)
                hechas+=1;progreso()
    guardar(out/'FASE_CAPACIDAD_TERMINADA.json',dict(semillas=semillas,tareas=hechas,prueba=False))
    progreso(estado_fase='Terminada; continúa ampliación independiente')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--semillas',type=int,nargs='+',default=list(range(101,106)));args=ap.parse_args()
    try:ejecutar(args.out,args.semillas)
    except Exception:
        import traceback
        guardar(args.out/'ERROR_CAPACIDAD.json',dict(error=traceback.format_exc()));raise

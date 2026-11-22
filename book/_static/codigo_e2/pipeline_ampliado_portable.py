"""Compatibilidad del lanzador numérico con las arquitecturas seleccionadas."""
from pathlib import Path
import argparse
import importlib.util
import json


def ejecutar(raiz,familia,semilla=1000,fase='validacion',anio=2021,
             dispositivo='auto',salida=None,permitir_prueba=False):
    if familia not in ('patchtst','nhits'):raise ValueError('Familia desconocida')
    if fase=='prueba' and not permitir_prueba:raise PermissionError('La prueba requiere permiso explícito')
    if salida is None:raise ValueError('Indique una carpeta nueva de salida')
    raiz=Path(raiz).resolve()
    spec=importlib.util.spec_from_file_location('entrenamiento_portable',raiz/'scripts/entrenar_arquitectura.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    destino=Path(salida)/familia/f'semilla_{semilla}'/(str(anio) if fase=='validacion' else 'prueba')
    # PatchTST = Modelo6 N, configuración 18; N-HiTS = configuración N14.
    return mod.ejecutar(familia,destino,raiz,semilla,fase,anio,dispositivo,permitir_prueba)['metricas']


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--raiz',type=Path,default=Path.cwd())
    ap.add_argument('--familia',choices=['patchtst','nhits'],required=True)
    ap.add_argument('--semilla',type=int,default=1000)
    ap.add_argument('--fase',choices=['validacion','prueba'],default='validacion')
    ap.add_argument('--anio',type=int,default=2021)
    ap.add_argument('--dispositivo',choices=['auto','cpu','cuda'],default='auto')
    ap.add_argument('--salida',type=Path,required=True)
    ap.add_argument('--allow-test',action='store_true')
    a=ap.parse_args()
    print(json.dumps(ejecutar(a.raiz,a.familia,a.semilla,a.fase,a.anio,a.dispositivo,a.salida,a.allow_test),indent=2))


if __name__=='__main__':main()

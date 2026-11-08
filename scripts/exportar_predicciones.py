"""Exporta predicciones de un ajuste a CSV largo con fechas y semilla."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('carpeta', type=Path, help='Carpeta del ajuste en results/base_v2/ajustes')
    p.add_argument('salida', type=Path)
    a = p.parse_args()
    identidad = json.loads((a.carpeta / 'identidad.json').read_text())
    with np.load(a.carpeta / 'predicciones.npz') as z:
        y, pred = z['y'], z['yhat']
        origen = pd.to_datetime(z['origen'], unit='ns', utc=True).tz_convert('America/Bogota')
    h = pred.shape[1]
    fechas = origen.repeat(h)
    pasos = np.tile(np.arange(h), len(origen))
    df = pd.DataFrame(dict(etapa=identidad['etapa'], modelo=identidad['modelo'], semilla=identidad['semilla'],
        origen=fechas, fecha_objetivo=fechas + pd.to_timedelta(pasos, unit='h'), horizonte=pasos + 1,
        observado=y.reshape(-1), prediccion=pred.reshape(-1)))
    df.to_csv(a.salida, index=False)

if __name__ == '__main__':
    main()

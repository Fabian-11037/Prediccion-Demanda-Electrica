"""Exporta escaladores e índice verificable de los ajustes completos."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/base_v2'

def main():
    filas = []
    for resultado in sorted((OUT / 'ajustes').glob('*/*/resultado.json')):
        carpeta = resultado.parent
        identidad = json.loads((carpeta / 'identidad.json').read_text())
        checkpoint = torch.load(carpeta / 'modelo.pt', map_location='cpu', weights_only=False)
        escala = {k: checkpoint[k] for k in ('media', 'std', 'escala_mase', 'datos', 'columnas_x', 'columnas_cal')}
        (carpeta / 'escaladores.json').write_text(json.dumps(escala, ensure_ascii=False, indent=2, default=str))
        with np.load(carpeta / 'predicciones.npz') as pred:
            assert pred['y'].shape == pred['yhat'].shape
            assert len(pred['origen']) == len(pred['y'])
            assert np.isfinite(pred['yhat']).all()
            n, h = pred['yhat'].shape
        historia = json.loads((carpeta / 'historia.json').read_text())
        assert historia and 'train_loss' in historia[0] and 'val_loss' in historia[0]
        for nombre in ('identidad.json', 'resultado.json', 'historia.json', 'modelo.pt', 'predicciones.npz', 'escaladores.json'):
            p = carpeta / nombre
            with p.open('rb') as f: digest = hashlib.file_digest(f, 'sha256').hexdigest()
            filas.append(dict(etapa=identidad['etapa'], modelo=identidad['modelo'], semilla=identidad['semilla'], id=carpeta.name,
                inicio_evaluacion=identidad['inicio_evaluacion'], fin_evaluacion=identidad['fin_evaluacion'], origenes=n, horizonte=h,
                archivo=str(p.relative_to(OUT)), bytes=p.stat().st_size, sha256=digest))
    with (OUT / 'indice_artefactos.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]), lineterminator='\n'); w.writeheader(); w.writerows(filas)
    print(f'{len(filas)//6} ajustes verificados; {len(filas)} artefactos indexados.')

if __name__ == '__main__':
    main()

"""Clasificación de candidatos con cobertura completa de años y semillas."""
import numpy as np

def clasificar(filas,configuraciones,anios,semillas):
    ids={c['id']:c for c in configuraciones};scores=[]
    for ident,cfg in ids.items():
        valores=[]
        for year in anios:
            for seed in semillas:
                match=[r for r in filas if r['configuracion']==cfg and r['fold']==year and r['semilla']==seed]
                if len(match)!=1:raise ValueError(f'Cobertura incompleta o duplicada: {ident}, {year}, {seed}')
                value=float(match[0]['MASE'])
                if not np.isfinite(value):raise ValueError('MASE no finito')
                valores.append(value)
        scores.append((float(np.mean(valores)),cfg))
    return sorted(scores,key=lambda t:(t[0],t[1]['id']))

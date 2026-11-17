"""Reejecuta las figuras de la evaluación desde artefactos numéricos conservados."""
from pathlib import Path
import nbformat as nb
from reorganizar_capitulo10 import ejecutar
from redaccion_investigacion import aplicar

def main():
    root=Path(__file__).resolve().parents[1]
    aplicar(root)
    path=root/'book/10_evaluacion.ipynb'
    n=nb.read(path,4)
    count=ejecutar(n.cells)
    nb.write(n,path)
    print(f'Evaluación reconstruida: {count} celdas; sin entrenamiento ni inferencia de redes.')

if __name__=='__main__':main()

"""Reanuda la cola con su lista ORIGINAL de fuentes, sin incorporar fases nuevas."""
import argparse,hashlib,json,runpy,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();out=a.out.resolve()
reg=json.loads((out/'REGISTRO_COLA_COMPLETA.json').read_text());archivos=reg['fuentes']
assert all(hashlib.sha256((ROOT/k).read_bytes()).hexdigest()==v for k,v in archivos.items())
# El arranque original descubre archivos con un glob. Las fases complementarias
# añadidas después no son fuentes de aquel proceso. Respetar su manifiesto evita
# mezclar versiones al restaurar, manteniendo todas las comprobaciones originales.
glob_original=Path.glob
def glob_registrado(self,pattern):
    if self==ROOT/'scripts' and pattern=='m4_cierre_*.py':return iter([ROOT/k for k in archivos if k.startswith('scripts/m4_cierre_') and k.endswith('.py')])
    return glob_original(self,pattern)
Path.glob=glob_registrado
sys.path.insert(0,str(ROOT/'scripts'));sys.argv=[str(ROOT/'scripts/ejecutar_cierre_m4.py'),'--out',str(out),'--publicar','--apagar']
runpy.run_path(sys.argv[0],run_name='__main__')

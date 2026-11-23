"""Concatena fragmentos exactos publicados y verifica el SHA256 original."""
import argparse, hashlib, json
from pathlib import Path
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def reconstruir(indice):
    d=json.loads(indice.read_text());dest=indice.parent/d['original']
    assert dest.parent.resolve()==indice.parent.resolve()
    if dest.exists():assert sha(dest)==d['sha256'];return
    tmp=dest.with_name(dest.name+'.reconstruyendo')
    with tmp.open('wb') as f:
        for parte in d['partes']:
            p=indice.parent/parte['archivo'];assert p.parent.resolve()==indice.parent.resolve();assert sha(p)==parte['sha256']
            with p.open('rb') as r:
                while b:=r.read(2**20):f.write(b)
    assert sha(tmp)==d['sha256'] and tmp.stat().st_size==d['bytes'];tmp.replace(dest)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('raiz',type=Path);a=ap.parse_args()
    for p in a.raiz.rglob('*.partes.json'):reconstruir(p)

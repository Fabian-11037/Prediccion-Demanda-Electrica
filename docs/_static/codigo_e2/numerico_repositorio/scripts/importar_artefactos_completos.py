"""Exporta todos los artefactos del commit remoto sin cambiar el checkout."""
import hashlib,json,subprocess,tarfile
from pathlib import Path

root=Path.cwd()
sha='02ff447d94fa78e1cde323fa2382da5bce24fb6e'
base=root/'results/repositorio'/sha
base.mkdir(parents=True,exist_ok=True)
paths=['results','src','scripts','tests','requirements.txt','CONTEXTO_PROYECTO.md',
       'PREREGISTRO.md','PREREGISTRO_CNN1D.md','ADENDA_LIMITE_LSTM.md']
proceso=subprocess.Popen(['git','-c','core.autocrlf=false','archive','--format=tar',sha,*paths],stdout=subprocess.PIPE)
n=0
with tarfile.open(fileobj=proceso.stdout,mode='r|') as archive:
    for item in archive:
        target=(base/item.name).resolve()
        assert target.is_relative_to(base.resolve()),item.name
        archive.extract(item,base,filter='data')
        if item.isfile():
            n+=1
            if n%2500==0:print('Archivos extraídos:',n,flush=True)
assert proceso.wait()==0
files=[]
for p in sorted(base.rglob('*')):
    if p.is_file() and p.name!='IMPORTACION.json':
        h=hashlib.sha256()
        with p.open('rb') as f:
            for bloque in iter(lambda:f.read(1024*1024),b''):h.update(bloque)
        files.append(dict(archivo=p.relative_to(base).as_posix(),bytes=p.stat().st_size,sha256=h.hexdigest()))
(base/'IMPORTACION.json').write_text(json.dumps(dict(commit=sha,
    repositorio='https://github.com/Fabian-11037/Prediccion-Demanda-Electrica',
    archivos=files),ensure_ascii=False,indent=2),encoding='utf-8')
print('Importación completa:',len(files),'archivos',sum(r['bytes'] for r in files),'bytes',flush=True)

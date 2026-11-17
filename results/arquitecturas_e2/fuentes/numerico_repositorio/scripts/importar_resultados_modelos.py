"""Importa una instantánea remota sin mezclar fuentes locales."""
import hashlib,json,subprocess,zipfile
from pathlib import Path

root=Path.cwd()
sha='e9b70ed0a80002a324eaf6f09923498df6a3c64e'
base=root/'results/repositorio'/sha
base.mkdir(parents=True,exist_ok=True)
archive=root/'results/repositorio'/(sha+'.zip')
paths=['results','src','scripts','tests','requirements.txt','CONTEXTO_PROYECTO.md',
       'PREREGISTRO.md','PREREGISTRO_CNN1D.md','ADENDA_LIMITE_LSTM.md']
subprocess.run(['git','-c','core.autocrlf=false','archive','--format=zip','--output',str(archive),sha,*paths],check=True)
with zipfile.ZipFile(archive) as z:
    for item in z.infolist():
        target=(base/item.filename).resolve()
        assert target.is_relative_to(base.resolve()),item.filename
    z.extractall(base)
files=[]
for p in sorted(base.rglob('*')):
    if p.is_file() and p.name!='IMPORTACION.json':
        files.append(dict(archivo=p.relative_to(base).as_posix(),bytes=p.stat().st_size,
                          sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
(base/'IMPORTACION.json').write_text(json.dumps(dict(commit=sha,
    repositorio='https://github.com/Fabian-11037/Prediccion-Demanda-Electrica',
    archivos=files),ensure_ascii=False,indent=2),encoding='utf-8')
print('Instantánea:',base)
print('Archivos:',len(files))

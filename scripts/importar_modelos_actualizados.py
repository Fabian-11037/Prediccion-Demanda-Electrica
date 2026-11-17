"""Descarga una instantánea de los modelos consolidados sin modificar fuentes activas."""
from pathlib import Path
import hashlib,json,subprocess,tarfile,sys

root=Path.cwd()
sha=subprocess.check_output(['git','rev-parse','origin/main'],text=True).strip()
base=root/'results/repositorio'/sha;base.mkdir(parents=True,exist_ok=True)
paths=['results/base_v3','results/base_v4','results/consolidado','src','scripts','tests',
       'requirements.txt','CONTEXTO_PROYECTO.md','PLAN_REENTRENAMIENTO_V3.md','PLAN_VARIANTES_V4.md',
       'data/processed/feature_registry.json']
proc=subprocess.Popen(['git','-c','core.autocrlf=false','archive','--format=tar',sha,*paths],stdout=subprocess.PIPE)
with tarfile.open(fileobj=proc.stdout,mode='r|') as archive:
    for item in archive:
        assert (base/item.name).resolve().is_relative_to(base.resolve()),item.name
        archive.extract(item,base,filter='data')
assert proc.wait()==0
files=[]
for p in sorted(base.rglob('*')):
    if p.is_file() and p.name!='IMPORTACION.json':
        files.append(dict(archivo=p.relative_to(base).as_posix(),bytes=p.stat().st_size,
                          sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
(base/'IMPORTACION.json').write_text(json.dumps(dict(commit=sha,
    repositorio='https://github.com/Fabian-11037/Prediccion-Demanda-Electrica',archivos=files),ensure_ascii=False,indent=2),encoding='utf-8')
(root/'results/jbook/modelos_actualizados.json').write_text(json.dumps({'commit':sha,'registro':'results/consolidado/modelos.json'},indent=2),encoding='utf-8')
print('Instantánea:',sha,'archivos:',len(files),'bytes:',sum(f['bytes'] for f in files),flush=True)

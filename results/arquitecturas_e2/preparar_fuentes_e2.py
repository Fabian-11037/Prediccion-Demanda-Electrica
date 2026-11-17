from pathlib import Path,PurePosixPath
import json,zipfile,hashlib,shutil,ast
R=Path(r'D:\Documents\Maestria\Proyecto Final\Prediccion-Demanda-Electrica-JBook')
C=R.parent/'Consolidado-Entrenamientos-20261115'
OUT=R/'results/arquitecturas_e2';OUT.mkdir(exist_ok=True)
FU=OUT/'fuentes';FU.mkdir(exist_ok=True)
shutil.copy2(__file__,OUT/'preparar_fuentes_e2.py')
regs=[]
for nombre in ['Modelo3_TFT','Modelo4_TCN','Modelo5_iTransformer','Modelo7_TGForecaster','FusionTardia','Modelo6_TaTS']:
    p=Path(r'C:\Users\Work\Downloads')/(nombre+'.zip')
    with zipfile.ZipFile(p) as z:
        for i in z.infolist():
            if i.is_dir():continue
            parts=PurePosixPath(i.filename).parts
            assert '..' not in parts and not PurePosixPath(i.filename).is_absolute()
            # Fuentes y configuraciones; las tablas se reutilizan desde el dataset.
            if 'data' in parts:continue
            dest=FU.joinpath(*parts);assert dest.resolve().is_relative_to(FU.resolve())
            dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(z.read(i))
            regs.append(dict(paquete=nombre,ruta=str(dest.relative_to(R)),sha256=hashlib.sha256(dest.read_bytes()).hexdigest()))
    code=next((FU/nombre/'src').glob('*/redes.py'))
    tree=ast.parse(code.read_text(encoding='utf-8-sig'))
    print(nombre,'clases',[n.name for n in tree.body if isinstance(n,ast.ClassDef)])
    if nombre!='Modelo7_TGForecaster':
        mapa={'Modelo3_TFT':'modelo3_20261115_0419/resultados_modelo3',
              'Modelo4_TCN':'modelo4_20261114_1224/resultados_modelo4',
              'Modelo5_iTransformer':'modelo5_20261114_1327/resultados_modelo5',
              'Modelo6_TaTS':'modelo6_20261112_1035/resultados_modelo6',
              'FusionTardia':'fusion_tardia_20261113_1728/resultados_fusion_tardia'}
        reg=json.loads((C/'fuentes/adjuntos'/mapa[nombre]/'REGISTRO.json').read_text())
        fallos=[]
        for rel,h in reg['huellas_sha256'].items():
            f=FU/nombre/rel
            if f.is_file():
                raw=f.read_bytes()
                opciones=[hashlib.sha256(x).hexdigest() for x in [raw,raw.replace(b'\r\n',b'\n'),raw.replace(b'\r\n',b'\n').replace(b'\n',b'\r\n')]]
                if h not in opciones:fallos.append(rel)
        print('huellas_pendientes',fallos)

repo=next((C/'fuentes').glob('repositorio_*[0-9a-f]'))
for folder in ['src','scripts','config']:
    shutil.copytree(repo/folder,FU/'numerico_repositorio'/folder,dirs_exist_ok=True)
(OUT/'manifest_fuentes.json').write_text(json.dumps(regs,ensure_ascii=False,indent=2),encoding='utf-8')
print('Fuentes verificadas:',len(regs))

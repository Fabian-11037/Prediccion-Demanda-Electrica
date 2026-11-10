"""OCR local en español; conserva cobertura de páginas y no entrena modelos.

Windows: requiere winrt-Windows.Media.Ocr, Storage, Storage.Streams,
Graphics.Imaging, Globalization, Foundation y Foundation.Collections 3.2.1,
además de pdftoppm. Los resultados JSON pueden consumirse en cualquier sistema.
"""
from pathlib import Path
import argparse,asyncio,json,re,shutil,subprocess,sys

ROOT=Path(__file__).resolve().parents[2]
RAW=ROOT/'data/raw/documentos_oficiales'

def preparar():
    trabajos=[]
    for r in map(json.loads,(RAW/'documentos_extraidos.jsonl').read_text(encoding='utf-8').splitlines()):
        if r['estado_extraccion']!='requiere_ocr':continue
        trabajos.append({'id':r['id'],'original':r['original_utilizado']['archivo'],'sha256_original':r['original_utilizado']['sha256'],'paginas':r['paginas'],'salida':(RAW/'ocr'/(r['id']+'.json')).relative_to(ROOT).as_posix(),'imagenes':(RAW/'imagenes_ocr').relative_to(ROOT).as_posix()})
    (RAW/'trabajos_ocr.json').write_text(json.dumps(trabajos,indent=2),encoding='utf-8')
    print('Trabajos OCR',len(trabajos),flush=True)

def resolver(p):
    p=Path(p)
    return p if p.is_absolute() else ROOT/p

async def ejecutar(lista,lote,lotes,max_paginas):
    from winrt.windows.media.ocr import OcrEngine
    from winrt.windows.globalization import Language
    from winrt.windows.storage import StorageFile,FileAccessMode
    from winrt.windows.graphics.imaging import BitmapDecoder
    motor=OcrEngine.try_create_from_language(Language('es-MX'))
    if motor is None:raise RuntimeError('Reconocedor OCR es-MX no disponible')
    render=shutil.which('pdftoppm')
    if not render:raise RuntimeError('pdftoppm no está disponible')
    trabajos=json.loads(Path(lista).read_text(encoding='utf-8'));contador=0
    for i,t in enumerate(trabajos):
        if i%lotes!=lote:continue
        salida=resolver(t['salida'])
        if salida.exists():
            anterior=json.loads(salida.read_text(encoding='utf-8'))
            if anterior.get('estado')=='ok' and anterior.get('sha256_original')==t['sha256_original'] and len(anterior.get('paginas',[]))>=min(t['paginas'],max_paginas):continue
        imagenes=resolver(t['imagenes']);imagenes.mkdir(parents=True,exist_ok=True)
        prefijo=imagenes/t['id']
        r={'id':t['id'],'sha256_original':t['sha256_original'],'idioma':'es-MX','motor':'Windows.Media.Ocr','paginas_totales':t['paginas'],'max_paginas':max_paginas,'escala_render':1800,'paginas':[],'estado':'ok'}
        try:
            resultado=subprocess.run([render,'-f','1','-l',str(min(t['paginas'],max_paginas)),'-scale-to','1800','-png',str(resolver(t['original'])),str(prefijo)],capture_output=True,timeout=180)
            if resultado.returncode:raise RuntimeError(resultado.stderr.decode(errors='replace')[:500])
            for png in sorted(imagenes.glob(t['id']+'-*.png')):
                archivo=await StorageFile.get_file_from_path_async(str(png.resolve()))
                flujo=await archivo.open_async(FileAccessMode.READ)
                decoder=await BitmapDecoder.create_async(flujo)
                bitmap=await decoder.get_software_bitmap_async()
                reconocido=await motor.recognize_async(bitmap)
                texto='\n'.join(linea.text for linea in reconocido.lines)
                pagina=int(re.search(r'-(\d+)$',png.stem)[1])
                r['paginas'].append({'pagina':pagina,'texto':texto})
                bitmap.close();flujo.close()
                assert png.resolve().parent==imagenes.resolve()
                png.unlink()
        except Exception as e:r.update(estado='error',error=str(e))
        salida.parent.mkdir(parents=True,exist_ok=True)
        salida.write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf-8')
        contador+=1
        if contador%10==0:print('OCR',lote,contador,flush=True)
    print('OCR terminado',lote,contador,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('lista',nargs='?',default=str(RAW/'trabajos_ocr.json'));p.add_argument('--preparar',action='store_true');p.add_argument('--lote',type=int,default=0);p.add_argument('--lotes',type=int,default=1);p.add_argument('--max-paginas',type=int,default=4);a=p.parse_args()
    if a.preparar:preparar()
    else:asyncio.run(ejecutar(a.lista,a.lote,a.lotes,a.max_paginas))

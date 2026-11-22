"""Construye y sirve el JBook, o sirve el dashboard, en Windows y Linux."""
from pathlib import Path
from functools import partial
import argparse, http.server, importlib.metadata, os, subprocess, sys, sysconfig

def main():
    p=argparse.ArgumentParser()
    p.add_argument('sitio',choices=['jbook','dashboard'])
    p.add_argument('--puerto',type=int,default=8000)
    p.add_argument('--solo-construir',action='store_true')
    args=p.parse_args();root=Path(__file__).resolve().parent
    if args.sitio=='jbook':
        try:version=importlib.metadata.version('jupyter-book')
        except importlib.metadata.PackageNotFoundError:raise SystemExit('Instala primero: python -m pip install -r requirements-jbook.txt')
        if version.split('.')[0]!='1':raise SystemExit('Este libro requiere Jupyter Book 1.x. Instala requirements-jbook.txt.')
        executable=Path(sysconfig.get_path('scripts'))/('jupyter-book.exe' if os.name=='nt' else 'jupyter-book')
        subprocess.run([str(executable),'build',str(root/'book'),'--warningiserror','--keep-going'],check=True)
        directory=root/'book/_build/html';entry='intro.html'
    else:directory=root/'dashboard/site';entry='index.html'
    if args.solo_construir:return
    if not 1<=args.puerto<=65535:raise SystemExit('El puerto debe estar entre 1 y 65535.')
    handler=partial(http.server.SimpleHTTPRequestHandler,directory=str(directory))
    server=http.server.ThreadingHTTPServer(('127.0.0.1',args.puerto),handler)
    print(f'Abre http://127.0.0.1:{args.puerto}/{entry}',flush=True)
    print('Para detener: Ctrl+C.',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()

if __name__=='__main__':main()

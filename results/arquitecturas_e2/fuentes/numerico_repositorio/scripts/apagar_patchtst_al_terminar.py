"""Apagado autorizado tras cerrar una ejecución y sincronizar sus resultados."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

OUT = Path(__file__).resolve().parents[1] / 'results/patchtst_numerico_r1'
COMANDOS = (b'entrenar_patchtst_numerico.py', b'probar_patchtst_numerico.py')
TERMINALES = {'Completado', 'Requiere decisión', 'Presupuesto temporal agotado'}


def leer(nombre):
    return json.loads((OUT / nombre).read_text())


def registrar(estado, **datos):
    destino = OUT / 'apagado_automatico.json'
    temporal = destino.with_suffix('.tmp')
    temporal.write_text(json.dumps(dict(estado=estado, pid=os.getpid(),
                                       fecha=time.time(), **datos), indent=2,
                                   ensure_ascii=False))
    temporal.replace(destino)


def proceso(pid):
    """Identifica la instancia, incluso si Linux reutiliza posteriormente el PID."""
    try:
        stat = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        if stat[0] == 'Z':
            return None
        comando = Path(f'/proc/{pid}/cmdline').read_bytes()
        if not any(nombre in comando for nombre in COMANDOS):
            return None
        return stat[19]
    except (OSError, ValueError, IndexError):
        return None


def verificar(estado):
    requeridos = ['plan.json', 'hardware.json', 'busqueda.csv']
    if estado['estado'] == 'Completado':
        requeridos += ['ejecucion_completa.json', 'resumen_final.json',
                       'final_runs.csv', 'referencias_test.json']
        assert leer('ejecucion_completa.json')['ajustes'] == estado['completados']
    elif estado['estado'] == 'Requiere decisión':
        requeridos += ['dimensionamiento.json', 'seleccion_cerrada.json', 'piloto.csv']
        assert leer('dimensionamiento.json')['n_final'] is None
    for nombre in requeridos:
        assert (OUT / nombre).stat().st_size > 0, nombre
    # Cada ajuste declarado completo debe conservar sus artefactos esenciales.
    resultados = list((OUT / 'ajustes').rglob('resultado.json'))
    assert len(resultados) >= estado['completados']
    for resultado in resultados:
        for nombre in ['modelo.pt', 'predicciones.npz', 'historia.json',
                       'escaladores.json', 'identidad.json']:
            assert (resultado.parent / nombre).stat().st_size > 0, str(resultado.parent / nombre)
    inventario = {str(p.relative_to(OUT)): p.stat().st_size
                  for p in OUT.rglob('*') if p.is_file()
                  and p.suffix in {'.json', '.csv', '.pt', '.npz', '.md'}
                  and p.name not in {'apagado_automatico.json', 'resultados_guardados.json'}}
    destino = OUT / 'resultados_guardados.json'
    temporal = destino.with_suffix('.tmp')
    temporal.write_text(json.dumps(dict(estado_final=estado, archivos_bytes=inventario,
                                       verificado=time.time()), indent=2, ensure_ascii=False))
    temporal.replace(destino)


def main():
    with (OUT / 'apagado.lock').open('a') as bloqueo:
        fcntl.flock(bloqueo, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pid = leer('estado.json')['pid']
        instancia = proceso(pid)
        assert instancia is not None, 'El entrenamiento esperado no está activo'
        registrar('esperando_finalizacion', pid_entrenamiento=pid)
        while True:
            estado = leer('estado.json')
            activo = proceso(pid) == instancia
            if estado['estado'] in TERMINALES and not activo:
                with (OUT / 'gpu.lock').open('a') as gpu:
                    fcntl.flock(gpu, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    estado = leer('estado.json')
                    assert estado['estado'] in TERMINALES
                    verificar(estado)
                    registrar('apagado_solicitado', estado_entrenamiento=estado['estado'])
                    os.sync()
                    resultado = subprocess.run([
                        'busctl', 'call', 'org.freedesktop.login1',
                        '/org/freedesktop/login1', 'org.freedesktop.login1.Manager',
                        'PowerOff', 'b', 'false'], capture_output=True, text=True, timeout=60)
                    if resultado.returncode:
                        registrar('error_apagado', detalle=resultado.stderr)
                    return
            if not activo:
                registrar('cancelado_por_error_o_salida_inesperada', estado_entrenamiento=estado)
                return
            time.sleep(30)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelo', choices=['patchtst', 'nhits', 'tft'], default='patchtst')
    args = parser.parse_args()
    if args.modelo == 'nhits':
        OUT = Path(__file__).resolve().parents[1] / 'results/nhits_numerico_r1'
        COMANDOS = (b'entrenar_nhits_numerico.py',)
    elif args.modelo == 'tft':
        OUT = Path(__file__).resolve().parents[1] / 'results/tft_numerico_r1'
        COMANDOS = (b'entrenar_tft_numerico.py', b'entrenar_tft_reducido.py')
    try:
        main()
    except BaseException as error:
        registrar('apagado_bloqueado', detalle=str(error))
        raise

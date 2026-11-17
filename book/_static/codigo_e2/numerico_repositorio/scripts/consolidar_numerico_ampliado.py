"""Consolida resultados y artefactos publicables sin reentrenar ni seleccionar por prueba."""
import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/consolidado_numerico_ampliado'
CORRIDAS = ['patchtst_numerico_r1', 'nhits_numerico_r1', 'tft_numerico_r1']
EXCLUIDOS = {'servicios_locales.json', 'apagado_automatico.json', 'cola.json',
             'resultados_guardados.json', 'historia_en_curso.json'}


def guardar_csv(path, filas):
    with path.open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]), lineterminator='\n')
        w.writeheader()
        w.writerows(filas)


def publicable(p):
    if p.name in EXCLUIDOS or p.suffix in {'.lock', '.log', '.tmp', '.pyc'}:
        return False
    if '__pycache__' in p.parts:
        return False
    if 'ajustes' in p.parts and not (p.parent / 'resultado.json').exists():
        return False
    if p.suffix == '.pt' and '/ajustes/final/' not in p.as_posix():
        return False
    return p.suffix in {'.json', '.csv', '.npz', '.pt', '.md', '.py', '.html'}


def main():
    OUT.mkdir(exist_ok=True)
    modelos = {}
    filas = []
    seleccion = []
    for nombre in CORRIDAS:
        carpeta = ROOT / 'results' / nombre
        resumen = json.loads((carpeta / 'resumen_final.json').read_text())
        completo = json.loads((carpeta / 'ejecucion_completa.json').read_text())
        assert json.loads((carpeta / 'estado.json').read_text())['estado'] == 'Completado'
        assert completo['semillas'] == 5
        with (carpeta / 'final_runs.csv').open() as f:
            finales = list(csv.DictReader(f))
        assert sorted(int(r['semilla']) for r in finales) == list(range(1000, 1005))
        referencias = json.loads((carpeta / 'referencias_test.json').read_text())
        fila = dict(ejecucion=nombre, modelo=resumen['modelo'], semillas=5,
                    MASE_validacion=resumen['MASE_validacion'],
                    MASE_prueba=resumen['MASE_prueba'], MASE_sd=resumen['MASE_sd'],
                    MAE_kwh=resumen['MAE_prueba'], RMSE_kwh=resumen['RMSE_prueba'],
                    sMAPE_pct=resumen['sMAPE_prueba'],
                    mejora_ingenuo_semanal_pct=resumen['mejora_ingenuo_semanal_pct'],
                    ajustes=completo['ajustes'], exploratorio=True)
        filas.append(fila)
        modelos[nombre] = dict(resumen=resumen, referencias=referencias, artefactos=[])
        predicciones = []
        for identidad in sorted((carpeta / 'ajustes/final').glob('*/identidad.json')):
            ajuste = identidad.parent
            if not (ajuste / 'resultado.json').exists():
                continue
            meta = json.loads(identidad.read_text())
            artefactos = dict(semilla=meta['semilla'], configuracion=meta['configuracion'])
            for archivo in ['modelo.pt', 'escaladores.json', 'predicciones.npz', 'historia.json', 'resultado.json', 'identidad.json']:
                p = ajuste / archivo
                assert p.is_file() and p.stat().st_size > 0
                artefactos[archivo] = str(p.relative_to(ROOT))
            modelos[nombre]['artefactos'].append(artefactos)
            with np.load(ajuste / 'predicciones.npz') as z:
                y, pred = z['y'], z['yhat']
                origen = pd.to_datetime(z['origen'], unit='ns', utc=True).tz_convert('America/Bogota')
                pasos = np.tile(np.arange(pred.shape[1]), len(origen))
                fechas = origen.repeat(pred.shape[1])
                predicciones.append(pd.DataFrame(dict(modelo=resumen['modelo'], semilla=meta['semilla'],
                    origen=fechas, fecha_objetivo=fechas + pd.to_timedelta(pasos, unit='h'),
                    horizonte=pasos + 1, observado=y.reshape(-1), prediccion=pred.reshape(-1))))
        assert len(modelos[nombre]['artefactos']) == 5
        with gzip.GzipFile(filename=str(OUT / (nombre + '_predicciones.csv.gz')), mode='wb', mtime=0) as f:
            f.write(pd.concat(predicciones, ignore_index=True).to_csv(index=False).encode())
        seleccion.extend(p for p in carpeta.rglob('*') if p.is_file() and publicable(p))
    filas.sort(key=lambda r: r['MASE_prueba'])
    guardar_csv(OUT / 'metricas.csv', filas)
    (OUT / 'modelos_finales.json').write_text(json.dumps(modelos, ensure_ascii=False, indent=2))
    md = ['# Resultados numéricos ampliados · noviembre de2026', '',
          'Prueba retrospectiva 2022–2025, historia de 168 h, horizonte de 24 h y origen diario a las 00:00 de Colombia. Los modelos usan las mismas 30 variables numéricas históricas, 23 de calendario y 30 máscaras, con escalado por tramo de optimización. Se evaluaron cinco semillas finales (1000–1004) por modelo.', '',
          '| Modelo | MASE validación | MASE prueba | DE entre semillas | MAE (kWh) | sMAPE (%) | Mejora frente a ingenuo semanal |',
          '|---|---:|---:|---:|---:|---:|---:|']
    for r in filas:
        md.append(f"| {r['modelo']} | {r['MASE_validacion']:.4f} | {r['MASE_prueba']:.4f} | {r['MASE_sd']:.4f} | {r['MAE_kwh']:.0f} | {r['sMAPE_pct']:.2f} | {r['mejora_ingenuo_semanal_pct']:.2f}% |")
    md += ['', '## Presupuesto y alcance', '',
           '- PatchTST: P20, 130 ajustes; N-HiTS: N14, 130 ajustes. Cada corrida incluye 105 de búsqueda, 20 de piloto y cinco finales. La selección usa confirmación 2019–2021 con tres semillas.',
           '- TFT: T02, 62 ajustes tras una reducción por coste: 51 de filtro, seis de confirmación y cinco finales. La confirmación usa una semilla (43) en 2019–2021; se omitió el piloto.',
           '- Los tres usan una arquitectura fija adaptada; no se afirma réplica literal de los artículos ni exención automática del requisito académico de 1.000 arquitecturas. Se conservan los planes, adendas y fuentes.',
           '- El piloto no permitió dimensionar potencia suficiente para el margen relativo del 3 % en PatchTST y N-HiTS. Las cinco semillas de prueba son un presupuesto exploratorio; TFT tampoco acredita potencia. La desviación estándar entre semillas no es un intervalo de confianza temporal. No se afirma significancia, equivalencia ni superioridad confirmatoria.',
           '- La prueba se observó repetidamente en el proyecto. Las diferencias entre modelos son descriptivas y los presupuestos de selección no son iguales. Los baselines v3/v4 conservados usan demanda y calendario; no constituyen una comparación de arquitectura con información idéntica.',
           '- Referencias comunes: ingenuo semanal, MASE 1,3549064443; ingenuo diario, MASE 2,0767245486. El denominador MASE fijo es 215.617,90361589016 kWh.', '',
           '## Artefactos publicados', '',
           '`modelos_finales.json` referencia los 15 checkpoints finales, escaladores, historiales, predicciones y métricas. Cada modelo se reconstruye con su clase en `demanda.patchtst_numerico`, `demanda.nhits_numerico` o `demanda.tft_numerico`, usando la configuración y sigma_demanda del checkpoint. Las dependencias incluidas y sus licencias acompañan al código.', '',
           'Se publican predicciones NPZ por origen UTC y semilla de todos los ajustes completos, historiales completos por época, escaladores, identidades y métricas. Los CSV comprimidos de esta carpeta exportan las predicciones finales con origen y fecha objetivo en America/Bogota, semilla y horizonte.', '',
           'Los pesos intermedios de búsqueda y piloto se conservan localmente y se excluyen de Git por volumen (cerca de 9 GB). Los 15 pesos finales sí están publicados. Los estados provisionales, bloqueos y servicios locales no forman parte de los resultados científicos publicados. `indice_publicado.csv` contiene tamaños y SHA-256 de los artefactos incluidos en esta publicación.', '']
    (OUT / 'README.md').write_text('\n'.join(md))
    seleccion.extend(p for p in OUT.iterdir() if p.is_file() and p.name != 'indice_publicado.csv')
    inventario = [dict(ruta=str(p.relative_to(ROOT)), bytes=p.stat().st_size,
        sha256=hashlib.file_digest(p.open('rb'), 'sha256').hexdigest()) for p in sorted(set(seleccion))]
    guardar_csv(OUT / 'indice_publicado.csv', inventario)
    seleccion.append(OUT / 'indice_publicado.csv')
    Path('/tmp/demanda-resultados-publicar.nul').write_bytes(b'\0'.join(str(p.relative_to(ROOT)).encode() for p in sorted(set(seleccion))) + b'\0')
    print(f'Preparados{len(seleccion)}archivos,{len(modelos)}modelos y15checkpoints finales;{sum(p.stat().st_size for p in seleccion)/1e6:.1f}MB.')


if __name__ == '__main__':
    main()

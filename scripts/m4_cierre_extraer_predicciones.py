"""Extrae pronósticos agregados publicados; no reconstruye semillas individuales."""
import argparse
import csv
import datetime as dt
import gzip
import hashlib
import json
from pathlib import Path


def extraer(out):
    destino = out / 'predicciones_historicas_disponibles'
    destino.mkdir(exist_ok=True)
    fuentes = []
    for modelo, clave in [('TimesNet', 'nuevo_timesnet__N'), ('NBEATS', 'nuevo_nbeats__N')]:
        fuente = out / 'insumos' / (clave + '.json')
        datos = json.loads(fuente.read_text())
        for campo in ['mean', 'p10', 'p90']:
            assert len(datos[campo]) == 1461
            assert all(len(fila) == 24 for fila in datos[campo])
        archivo = destino / f'{modelo}_promedio_cinco_semillas.csv.gz'
        with gzip.open(archivo, 'wt', encoding='utf-8', newline='') as f:
            escritor = csv.writer(f)
            escritor.writerow(['fecha_origen', 'fecha_objetivo', 'horizonte',
                              'predicho_promedio_kwh', 'percentil10_semillas_kwh',
                              'percentil90_semillas_kwh', 'numero_semillas'])
            for dia in range(1461):
                origen = dt.datetime(2022, 1, 1) + dt.timedelta(days=dia)
                for hora in range(24):
                    escritor.writerow([origen.isoformat(), (origen + dt.timedelta(hours=hora)).isoformat(),
                                       hora + 1, *[f'{datos[k][dia][hora] * 1e6:.2f}'
                                                   for k in ['mean', 'p10', 'p90']], 5])
        with gzip.open(archivo, 'rt', encoding='utf-8') as f:
            assert sum(1 for _ in f) - 1 == 35064
        fuentes.append(dict(modelo=modelo, filas=35064, archivo=archivo.name,
                            sha256=hashlib.sha256(archivo.read_bytes()).hexdigest(),
                            fuente=f'dashboard/site/data/predictions/{clave}.json',
                            sha256_fuente=hashlib.sha256(fuente.read_bytes()).hexdigest()))
    registro = dict(archivos=fuentes, periodo=['2022-01-01', '2025-12-31'],
                    zona_horaria='America/Bogota', horizonte='24 intervalos horarios del día de origen',
                    unidades='kWh', precision_publicada_kwh=0.01,
                    alcance='Promedio y percentiles entre cinco semillas; no predicciones individuales.',
                    advertencia='Los percentiles entre semillas no constituyen intervalos predictivos. '
                                'Los agregados no permiten reconstruir las cinco realizaciones. '
                                'El error del promedio difiere del promedio de errores usado en el benchmark.')
    (destino / 'MANIFIESTO.json').write_text(json.dumps(registro, ensure_ascii=False, indent=2))
    (destino / 'README.md').write_text(
        '# Pronósticos históricos disponibles\n\n'
        'Se extraen 35.064 pronósticos horarios por modelo para 2022–2025, con origen diario '
        'a las 00:00 de America/Bogota. El horizonte 1 corresponde al intervalo 00:00–01:00. '
        'Los CSV conservan el promedio y los percentiles 10/90 de las cinco semillas publicados '
        'en el dashboard, convertidos de GWh a kWh, con precisión de 0,01 kWh.\n\n'
        'No contienen las cinco realizaciones individuales: estas no se recuperan de los agregados. '
        'Los percentiles describen dispersión entre semillas, no cobertura predictiva. '
        'Los contrastes del benchmark utilizan las pérdidas individuales promediadas por origen, '
        'disponibles en los JSON originales; no el error del pronóstico promedio.\n', encoding='utf-8')
    return registro


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    print(json.dumps(extraer(parser.parse_args().out), ensure_ascii=False))

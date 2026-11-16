"""Evaluación retrospectiva autorizada, con cinco semillas fijadas antes de prueba."""
import fcntl
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import entrenar_patchtst_numerico as base

SEMILLAS = list(range(1000, 1005))
NOTA = ('Evaluación exploratoria retrospectiva autorizada con cinco semillas. '
        'El piloto no acreditó la potencia objetivo del 80 % para el margen del 3 %. '
        'Configuración y selección congeladas antes de evaluar 2022–2025.')


def main():
    with (base.OUT / 'gpu.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (base.OUT / 'ejecucion_completa.json').exists():
            return
        assert base.torch.cuda.is_available(), 'Se requiere GPU'
        plan = json.loads((base.OUT / 'plan.json').read_text())
        for nombre, digest in plan['sha256'].items():
            assert hashlib.sha256((base.ROOT / nombre).read_bytes()).hexdigest() == digest, nombre
        seleccion = json.loads((base.OUT / 'seleccion_cerrada.json').read_text())
        assert seleccion['dimensionamiento']['n_final'] is None
        assert seleccion['mejor'] == json.loads((base.OUT / 'mejor_config.json').read_text())['configuracion']
        cfg = seleccion['mejor']
        base.CONTRACT = json.loads((base.ROOT / 'config/diseno_entrenamiento_texto.json').read_text())
        base.DEADLINE = json.loads((base.OUT / 'limite_tiempo.json').read_text())['deadline']
        if time.time() >= base.DEADLINE:
            raise TimeoutError('Plazo original agotado; no se renueva automáticamente')
        decision = dict(configuracion=cfg, semillas=SEMILLAS, exploratorio=True,
                        potencia_acreditada=False, nota=NOTA, plazo_original=base.DEADLINE,
                        fuentes_sha256=plan['sha256'],
                        seleccion_sha256=hashlib.sha256((base.OUT / 'seleccion_cerrada.json').read_bytes()).hexdigest(),
                        ejecutor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        destino = base.OUT / 'plan_prueba_exploratoria.json'
        if destino.exists():
            anterior = json.loads(destino.read_text())
            assert {k: anterior[k] for k in decision} == decision, 'Plan de prueba modificado'
        else:
            decision['fecha'] = base.pd.Timestamp.now(tz='America/Bogota').isoformat()
            base.save(destino, decision)
        previo = base.OUT / 'estado_preprueba.json'
        if not previo.exists():
            base.save(previo, json.loads((base.OUT / 'estado.json').read_text()))
        base.DONE = 125
        base.TOTAL = 125 + len(SEMILLAS)
        estado_original = base.state

        def estado(**datos):
            estado_original(**datos)
            pagina = base.OUT / 'progreso.html'
            contenido = pagina.read_text().replace(
                'Evaluación final: R semillas, pendientes del dimensionamiento del piloto.',
                'Evaluación exploratoria autorizada: cinco semillas (1000–1004); potencia objetivo no acreditada.')
            temp = pagina.with_suffix('.tmp')
            temp.write_text(contenido)
            temp.replace(pagina)

        base.state = estado
        base.state(estado='Entrenando', etapa='final', detalle=NOTA)
        base.PANEL = base.pd.read_parquet(base.ROOT / 'data/processed/panel_horario.parquet')
        base.REG = json.loads((base.ROOT / 'data/processed/feature_registry.json').read_text())
        filas = []
        for semilla in SEMILLAS:
            filas.append(base.fit('final', cfg, semilla, 'final'))
            base.table('final_runs.csv', filas)
        _, _, _, evaluacion = base.pack('final')
        referencias = {nombre: base.evaluar(evaluacion.y, pred, base.CONTRACT['metrica']['q168_kwh'])
                       for nombre, pred in base.references(evaluacion).items()}
        base.save(base.OUT / 'referencias_test.json', referencias)
        valores = base.np.array([fila['MASE'] for fila in filas])
        resumen = dict(modelo='PatchTST_adaptado', n_final=len(SEMILLAS), semillas=SEMILLAS,
                       MASE_validacion=seleccion['MASE_cv'], anios_confirmacion=[2019, 2020, 2021],
                       MASE_prueba=float(valores.mean()), MASE_sd=float(valores.std(ddof=1)),
                       dimensionamiento=seleccion['dimensionamiento'], nota=NOTA,
                       exploratorio=True, potencia_acreditada=False)
        for metrica in ['MAE', 'RMSE', 'sMAPE', 'MAPE']:
            resumen[metrica + '_prueba'] = float(base.np.mean([fila[metrica] for fila in filas]))
        resumen['mejora_ingenuo_semanal_pct'] = 100 * (1 - resumen['MASE_prueba'] / referencias['ingenuo_168h']['MASE'])
        base.save(base.OUT / 'resumen_final.json', resumen)
        (base.OUT / 'RESUMEN.md').write_text('# PatchTST numérico: evaluación retrospectiva exploratoria\n\n' +
                                           json.dumps(resumen, indent=2, ensure_ascii=False) + '\n')
        base.save(base.OUT / 'ejecucion_completa.json', dict(ajustes=base.DONE, semillas=len(SEMILLAS),
                  exploratorio=True, potencia_acreditada=False,
                  fecha=base.pd.Timestamp.now(tz='America/Bogota').isoformat()))
        base.state(estado='Completado', detalle='Prueba exploratoria y artefactos guardados. ' + NOTA)
        print(json.dumps(resumen, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    try:
        main()
    except TimeoutError as error:
        base.state(estado='Presupuesto temporal agotado', detalle=str(error))
    except BaseException as error:
        base.state(estado='Error', detalle=str(error))
        traceback.print_exc()
        raise

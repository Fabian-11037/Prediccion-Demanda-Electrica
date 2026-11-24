"""Cierre TFT autorizado:51filtro,6confirmación,5prueba;sin renovar plazo."""
import fcntl
import hashlib
import html
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import entrenar_tft_numerico as B

CONFIG = B.CONFIG[:17]
SEMILLAS_FINALES = list(range(1000, 1005))


def congelar():
    original = json.loads((B.OUT / 'plan.json').read_text())
    for nombre, huella in original['sha256'].items():
        assert hashlib.sha256((B.ROOT / nombre).read_bytes()).hexdigest() == huella, nombre
    fuentes = ['scripts/entrenar_tft_reducido.py', 'ADENDA_TFT_PRESUPUESTO_REDUCIDO.md']
    plan = dict(configuraciones=CONFIG, filtro_anios=[2018], semillas_filtro=[43, 44, 45],
                finalistas=2, confirmacion_anios=[2019, 2020, 2021], semillas_confirmacion=[43],
                piloto=False, semillas_finales=SEMILLAS_FINALES, total=62, exploratorio=True,
                plan_original_sha256=hashlib.sha256((B.OUT / 'plan.json').read_bytes()).hexdigest(),
                sha256={n: hashlib.sha256((B.ROOT / n).read_bytes()).hexdigest() for n in fuentes})
    destino = B.OUT / 'plan_reducido.json'
    if destino.exists():
        anterior = json.loads(destino.read_text())
        assert {k: anterior[k] for k in plan} == plan, 'Plan reducido modificado'
    else:
        assert not (B.OUT / 'plan_prueba.json').exists(), 'La prueba ya fue preparada'
        plan['fecha'] = B.pd.Timestamp.now(tz='America/Bogota').isoformat()
        B.save(destino, plan)
    return plan


def elegir_final(filas, finalistas):
    return B.clasificar(filas, finalistas, [2019, 2020, 2021], [43])[0]


def estado(**datos):
    B.S.update(datos)
    B.S.update(completados=B.DONE, total=62, pid=B.os.getpid(),
               actualizado=B.pd.Timestamp.now(tz='America/Bogota').isoformat())
    B.save(B.OUT / 'estado.json', B.S)
    detalle = ' · '.join(f'{k}: {v}' for k, v in B.S.items() if k in
               ['estado', 'etapa', 'configuracion', 'semilla', 'fold', 'epoca'])
    texto = f'''<!doctype html><html lang="es"><meta charset="utf-8"><meta http-equiv="refresh" content="8"><title>TFT · Presupuesto reducido</title><style>body{{font:18px system-ui;background:#14202c;color:#eef5fa;max-width:950px;margin:50px auto;padding:20px}}progress{{width:100%;height:32px}}p{{line-height:1.7}}</style><h1>TFT compacto · GPU</h1><progress value="{B.DONE}" max="62"></progress><p>{B.DONE}/62 ajustes · {100*B.DONE/62:.1f}%</p><p>{html.escape(detalle)}</p><p>Filtro2018:17configuraciones × 3semillas =51.<br>Confirmación2019–2021:2finalistas × 1semilla × 3años =6.<br>Prueba2022–2025:5semillas =5.<br>Piloto omitido; evaluación exploratoria, sin potencia acreditada.</p><p>{html.escape(B.S.get('detalle',''))}</p><p>Se conserva el plazo original de24h y el apagado tras guardar resultados.</p><small>{B.S['actualizado']}</small></html>'''
    tmp = B.OUT / 'progreso.tmp'
    tmp.write_text(texto)
    tmp.replace(B.OUT / 'progreso.html')


def main():
    congelar()
    if '--preparar' in sys.argv:
        print('Plan reducido congelado; prueba no leída.', flush=True)
        return
    with (B.OUT / 'gpu.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (B.OUT / 'ejecucion_completa.json').exists():
            return
        assert B.torch.cuda.is_available(), 'Se requiere GPU'
        B.DEADLINE = json.loads((B.OUT / 'limite_tiempo.json').read_text())['deadline']
        if time.time() >= B.DEADLINE:
            raise TimeoutError('Se alcanzó el plazo original; no se renueva')
        B.CONTRACT = json.loads((B.ROOT / 'config/diseno_entrenamiento_texto.json').read_text())
        B.PANEL = B.pd.read_parquet(B.ROOT / 'data/processed/panel_horario.parquet')
        B.REG = json.loads((B.ROOT / 'data/processed/feature_registry.json').read_text())
        B.TOTAL = 62
        B.DONE = 0
        B.state = estado
        estado(estado='Entrenando', detalle='Plan reducido autorizado:62ajustes, sin renovar el plazo.')
        filas = []
        for cfg in CONFIG:
            for seed in (43, 44, 45):
                filas.append(B.fit('busqueda', cfg, seed, 2018))
                B.table('busqueda.csv', filas)
        ranking = B.clasificar(filas, CONFIG, [2018], [43, 44, 45])
        finalistas = [cfg for _, cfg in ranking[:2]]
        B.save(B.OUT / 'seleccion_2018.json', dict(finalistas=finalistas,
            ranking=[dict(MASE=score, configuracion=cfg) for score, cfg in ranking], presupuesto_reducido=True))
        for year in (2019, 2020, 2021):
            for cfg in finalistas:
                filas.append(B.fit('busqueda', cfg, 43, year))
                B.table('busqueda.csv', filas)
        score, best = elegir_final(filas, finalistas)
        dim = dict(estado='piloto_omitido_por_presupuesto_autorizado', n_final=None,
                   potencia_acreditada=False, semillas_fijadas_exploratorias=5)
        B.save(B.OUT / 'mejor_config.json', dict(configuracion=best, MASE_cv=score,
            anios_confirmacion=[2019, 2020, 2021], semillas_confirmacion=[43]))
        B.save(B.OUT / 'seleccion_cerrada.json', dict(mejor=best, MASE_cv=score,
            dimensionamiento=dim, presupuesto_reducido=True, fecha=B.S['actualizado']))
        prueba = dict(configuracion=best, semillas=SEMILLAS_FINALES,
            exploratorio=True, potencia_acreditada=False, piloto_realizado=False)
        archivo = B.OUT / 'plan_prueba.json'
        if archivo.exists():
            assert json.loads(archivo.read_text()) == prueba
        else:
            B.save(archivo, prueba)
        estado(etapa='final', detalle='Prueba retrospectiva exploratoria con cinco semillas;sin potencia acreditada.')
        finales = []
        for seed in SEMILLAS_FINALES:
            finales.append(B.fit('final', best, seed, 'final'))
            B.table('final_runs.csv', finales)
        _, _, _, ev = B.pack('final')
        referencia = {n: B.evaluar(ev.y, p, B.CONTRACT['metrica']['q168_kwh']) for n, p in B.references(ev).items()}
        B.save(B.OUT / 'referencias_test.json', referencia)
        valores = B.np.array([r['MASE'] for r in finales])
        resumen = dict(modelo='TFT_adaptado', n_final=5, semillas=SEMILLAS_FINALES,
            MASE_validacion=score, anios_confirmacion=[2019, 2020, 2021], semillas_confirmacion=[43],
            MASE_prueba=float(valores.mean()), MASE_sd=float(valores.std(ddof=1)),
            presupuesto_reducido=True, ajustes=62, dimensionamiento=dim, exploratorio=True,
            potencia_acreditada=False, nota='Selección reducida autorizada antes de prueba. Evaluación retrospectiva exploratoria;sin piloto ni potencia acreditada.')
        for metrica in ['MAE', 'RMSE', 'sMAPE', 'MAPE']:
            resumen[metrica + '_prueba'] = float(B.np.mean([r[metrica] for r in finales]))
        resumen['mejora_ingenuo_semanal_pct'] = 100 * (1 - valores.mean() / referencia['ingenuo_168h']['MASE'])
        B.save(B.OUT / 'resumen_final.json', resumen)
        (B.OUT / 'RESUMEN.md').write_text('# TFT compacto:presupuesto reducido\n\n' + json.dumps(resumen, ensure_ascii=False, indent=2) + '\n')
        B.save(B.OUT / 'ejecucion_completa.json', dict(ajustes=B.DONE, semillas=5,
            presupuesto_reducido=True, exploratorio=True, fecha=B.S['actualizado']))
        estado(estado='Completado', detalle='Resultados y artefactos guardados.')


if __name__ == '__main__':
    try:
        main()
    except TimeoutError as error:
        B.state = estado
        estado(estado='Presupuesto temporal agotado', detalle=str(error))
    except BaseException as error:
        B.state = estado
        estado(estado='Error', detalle=str(error))
        traceback.print_exc()
        raise

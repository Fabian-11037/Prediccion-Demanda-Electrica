"""Prueba retrospectiva de M1: restauración congelada, sin ajuste ni selección."""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import shutil
from zoneinfo import ZoneInfo

from e3_gpu_comun import *
from e3_modelo1 import Router, EXPERTOS, datos_meta
from corrector.evaluacion import indices_bootstrap, holm


def rasgos(datos, dias, y, pred):
    """Receta idéntica al entrenamiento; etiquetas solo para errores d-2 o anteriores."""
    fechas = pd.DatetimeIndex(dias).tz_localize('America/Bogota')
    pos = datos.indice.get_indexer(fechas)
    assert (pos >= datos.L).all()
    xx = datos.X[pos[:, None] + np.arange(-datos.L, 0)]
    valid = np.isfinite(xx)
    limpio = np.where(valid, xx, 0.)
    medias = limpio.sum(1) / np.maximum(valid.sum(1), 1)
    ultimo = limpio[:, -1]
    m24 = limpio[:, -24:].sum(1) / np.maximum(valid[:, -24:].sum(1), 1)
    ausente = 1 - valid.mean(1)
    nivel = xx[:, :, 0].mean(1)
    assert (nivel > 0).all()
    cf = datos.K[pos[:, None] + np.arange(24)]
    resumen = np.concatenate([ultimo, medias, m24, ausente], 1)
    resumen[:, [0, len(datos.hist), 2 * len(datos.hist)]] /= nivel[:, None]
    errores = (y[:, :, None] - pred) / datos.q168
    lag = lags_diarios(dias, errores)
    x = np.concatenate([resumen, cf.reshape(len(dias), -1),
                        (pred / nivel[:, None, None]).reshape(len(dias), -1),
                        pred.std(-1) / nivel[:, None], lag], 1).astype('float32')
    assert np.isfinite(x).all()
    return x


def inferir_router(ck, x, pred):
    red = Router(ck['n_entrada'], ck['configuracion']).cuda().eval()
    red.load_state_dict(ck['modelo'])
    assert next(red.parameters()).is_cuda
    salidas, pesos = [], []
    with torch.inference_mode():
        for i in range(0, len(x), 256):
            xx = torch.as_tensor((x[i:i+256] - ck['media']) / ck['escala'], device='cuda')
            pp = torch.as_tensor(pred[i:i+256], device='cuda')
            yy, _, ww = red(xx, pp)
            salidas.append(yy.cpu().numpy())
            pesos.append(ww.cpu().numpy())
    return np.concatenate(salidas), np.concatenate(pesos)


def estadistica(errores, anios):
    """Semillas promediadas dentro del origen; conserva la familia global de 28 H0."""
    filas = []
    for bloque in (14, 7, 28):
        idx = indices_bootstrap(anios, 9999, bloque, 20261122)
        por_bloque = []
        for ref in ('NNLS', 'patchtst'):
            ea, eb = errores['M1'], errores[ref]
            g = 100 * (eb.mean() - ea.mean()) / eb.mean()
            gb = 100 * (eb[idx].mean(1) - ea[idx].mean(1)) / eb[idx].mean(1)
            lo, hi = np.percentile(gb, [2.5, 97.5])
            for margen in (0., 3.):
                p = float((1 + np.sum(gb - g >= g - margen)) / (len(gb) + 1))
                por_bloque.append(dict(A='M1', B=ref, bloque_dias=bloque, margen_pct=margen,
                                      MASE_A=ea.mean(), MASE_B=eb.mean(), reduccion_pct=g,
                                      ic95_inf=lo, ic95_sup=hi, p_unilateral=p))
        # Las 24 hipótesis de modelos no ejecutados permanecen pendientes. Rellenar
        # con unos produce una cota conservadora de Holm, no sus p-valores medidos.
        cotas = holm([r['p_unilateral'] for r in por_bloque] + [1.] * 24)[:4]
        for r, cota in zip(por_bloque, cotas):
            r['cota_superior_p_holm_global_28'] = cota
            r['rechazo_conservador_005'] = cota < .05
        filas.extend(por_bloque)
        del idx
    return pd.DataFrame(filas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--entrenamiento', type=Path, required=True)
    ap.add_argument('--salida', type=Path, required=True)
    args = ap.parse_args()
    run, out = args.entrenamiento.resolve(), args.salida.resolve()
    out.mkdir(parents=True, exist_ok=True)
    lock = (out/'proceso.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (out/'REGISTRO_PREVIO.json').exists(), 'No sobrescribir una evaluación previa'
    m1 = run/'M1'
    semillas = [101, 102, 103, 104, 105]
    config = leer(m1/'CONFIGURACION_ELEGIDA.json')['configuracion']
    diseno = leer(REPO/'continuidad/DISENO_CUATRO_MODELOS_GPU.json')
    assert config == dict(id=8, ancho=64, dropout=.1, lambda_complementariedad=.1)
    manifiesto = {r['archivo']: r['sha256'] for r in leer(run/'MANIFIESTO_FINAL.json')}
    entradas = []
    for s in semillas:
        entradas += [m1/'final'/f'semilla_{s}'/'modelo.pt',
                     m1/'controles_finales'/f'semilla_{s}'/'pesos.json']
        entradas += [m1/'expertos_finales'/k/f'semilla_{s}'/'modelo.pt' for k in EXPERTOS]
        entradas += [m1/'expertos'/k/f'anio_{a}'/f'semilla_{s}'/'predicciones.npz'
                     for a in range(2011, 2022) for k in EXPERTOS]
    huellas = {str(p): sha(p) for p in entradas}
    assert all(huellas[str(p)] == manifiesto[str(p.relative_to(run))] for p in entradas)
    for nombre in ('e3_gpu_comun.py', 'e3_modelo1.py'):
        assert sha(REPO/'scripts'/nombre) == sha(run/'fuentes/scripts'/nombre)
    for p, h in leer(run/'REGISTRO_PREVIO.json')['huellas'].items():
        if p.startswith(str(PAQUETE/'src')) and p.endswith('.py'):
            assert sha(p) == h, 'El código original cambió desde el entrenamiento: ' + p
    assert sha(PAQUETE/'data/processed/panel_horario.parquet') == diseno['datos']['hashes_entrada']['data/processed/panel_horario.parquet']
    fuente = out/'fuentes'
    fuente.mkdir(exist_ok=True)
    for p in (Path(__file__), REPO/'scripts/e3_modelo1.py', REPO/'scripts/e3_gpu_comun.py'):
        shutil.copy2(p, fuente/p.name)
    plan = dict(fecha=dt.datetime.now(ZoneInfo('America/Bogota')).isoformat(),
                autorizacion='Solicitud expresa del usuario: haz la prueba de m1',
                periodo=['2022-01-01', '2025-12-31'], origenes=1461, horizonte=24, rezago=168,
                configuracion=config, semillas=semillas, entrenamiento=str(run), huellas=huellas,
                seleccion_adicional=False, reajuste=False, actualizacion_pesos=False,
                rezago_errores_dias=2, inicializacion_errores='OOF 2011–2021 y resultados disponibles de prueba d-2',
                comparadores_primarios=['NNLS', 'patchtst (mejor individual de desarrollo)'],
                comparadores_descriptivos=['E', 'periodica', 'convexa_fija', 'media'],
                estadistica=dict(remuestreos=9999, bloques=[14,7,28], semilla=20261122,
                                  margenes_pct=[0,3], familia_total=28,
                                  multiplicidad='Holm conjunto pendiente de los otros modelos; se informa cota superior conservadora con 24 p=1 no observados',
                                  unidad='error horario medio por origen, promediado entre semillas; semillas no independientes temporalmente'),
                alcance='Evaluación retrospectiva informada por el benchmark ya conocido; no holdout nunca consultado.')
    guardar(out/'REGISTRO_PREVIO.json', plan)
    assert torch.cuda.is_available(), 'Se exige GPU; no hay fallback CPU'
    torch.set_num_threads(8)
    guardar(out/'ENTORNO_GPU.json', dict(nombre=torch.cuda.get_device_name(0), torch=torch.__version__, rocm=torch.version.hip))
    datos = DatosNumericos(PAQUETE)
    pos = datos.prueba()  # Único acceso final autorizado, después de congelar el registro.
    dias = datos.indice[pos].strftime('%Y-%m-%d').to_numpy().astype('U10')
    assert len(dias) == 1461 and dias[0] == '2022-01-01' and dias[-1] == '2025-12-31'
    todas, filas, anuales, horarios = {}, [], [], []
    verificaciones = []
    for s in semillas:
        motor.fijar_semilla(s)
        pp, yy = [], None
        for k in EXPERTOS:
            print(f'GPU: restaurando {k}, semilla {s}', flush=True)
            ck = cargar(m1/'expertos_finales'/k/f'semilla_{s}'/'modelo.pt')
            met = ck['metadatos']
            assert met['semilla'] == s and pd.Timestamp(met['parada'][-1]).year <= 2021
            assert pd.Timestamp(ck['escalas']['fin_ajuste']).year <= 2021
            conj = datos.construir(pos, ck['escalas'])
            if yy is None:
                yy = conj.y.astype('float32')
            else:
                assert np.array_equal(yy, conj.y.astype('float32'))
            red = construir_experto(datos, k, met['configuracion']).cuda().eval()
            red.load_state_dict(ck['modelo'])
            assert next(red.parameters()).is_cuda
            tens = motor.a_tensores(conj, 'cuda')
            pred = (motor.predecir(red, tens, 256) * conj.nivel[:, None]).astype('float32')
            assert np.isfinite(pred).all()
            pp.append(pred)
            del ck, conj, red, tens
            torch.cuda.empty_cache()
        expertos = np.stack(pp, -1)
        meta = datos_meta(datos, m1, s)
        # La adaptación de inferencia reproduce bit a bit los rasgos de desarrollo.
        assert np.array_equal(meta['x'], rasgos(datos, meta['dia'], meta['y'], meta['pred']))
        nd = len(meta['dia'])
        dd = np.concatenate([meta['dia'], dias])
        y = np.concatenate([meta['y'], yy])
        pred = np.concatenate([meta['pred'], expertos])
        xx = rasgos(datos, dd, y, pred)[nd:]
        # Perturbar objetivos desde un corte no cambia entradas hasta corte+1:
        # ni el objetivo actual ni el del día previo pueden afectar la predicción.
        corte = nd + 365
        alterado = y.copy()
        alterado[corte:] += 1e7
        xa = rasgos(datos, dd, alterado, pred)[nd:]
        assert np.array_equal(xx[:367], xa[:367])
        assert not np.array_equal(xx[367:], xa[367:])
        ck = cargar(m1/'final'/f'semilla_{s}'/'modelo.pt')
        n_entrada = ck['n_entrada']
        assert ck['configuracion'] == config and ck['semilla'] == s
        assert ck['parada'][-1].startswith('2021') and ck['ajuste'][-1] < '2021'
        pm1, pesos = inferir_router(ck, xx, expertos)
        assert np.isfinite(pm1).all() and np.isfinite(pesos).all()
        assert np.allclose(pesos.sum(-1), 1, atol=2e-7) and pesos.min() >= 0
        assert (pm1 >= expertos.min(-1) - 4).all() and (pm1 <= expertos.max(-1) + 4).all()
        controles = leer(m1/'controles_finales'/f'semilla_{s}'/'pesos.json')
        preds = dict(M1=pm1, NNLS=(expertos * np.asarray(controles['NNLS'])).sum(-1),
                     convexa_fija=(expertos * np.asarray(controles['convexa'])).sum(-1), media=expertos.mean(-1))
        preds.update({k: expertos[:, :, j] for j, k in enumerate(EXPERTOS)})
        if todas:
            assert np.array_equal(y_real, yy)
        y_real = yy
        for nombre, pr in preds.items():
            todas.setdefault(nombre, []).append(pr)
            filas.append(dict(modelo=nombre, semilla=s, **metricas(yy, pr, datos.q168)))
            for a in (2022, 2023, 2024, 2025):
                mask = np.char.startswith(dias, str(a))
                anuales.append(dict(modelo=nombre, semilla=s, anio=a, **metricas(yy[mask], pr[mask], datos.q168)))
            for h in range(24):
                horarios.append(dict(modelo=nombre, semilla=s, hora=h, **metricas(yy[:,h], pr[:,h], datos.q168)))
        np.savez_compressed(out/f'predicciones_semilla_{s}.npz', dia=dias, y=yy, pesos_M1=pesos, **preds)
        # CSV largo para análisis pareados por fecha, semilla y horizonte.
        tabla = pd.DataFrame(dict(fecha=np.repeat(dias,24), semilla=s, hora=np.tile(np.arange(24),len(dias)),
                                  y_kwh=yy.ravel(), **{k: v.ravel() for k,v in preds.items()}))
        tabla.to_csv(out/f'predicciones_semilla_{s}.csv.gz', index=False, compression='gzip')
        verificaciones.append(dict(semilla=s, rasgos_identicos_a_entrenamiento=True,
                                    invariancia_objetivos_presentes_futuros=True, restauracion_gpu=True))
        print(f'Semilla {s}: MASE M1={filas[-7]["MASE"]:.6f}', flush=True)
        del meta, xx, xa, y, pred, ck
    tabla = pd.DataFrame(filas)
    tabla.to_csv(out/'metricas_por_semilla.csv', index=False)
    pd.DataFrame(anuales).to_csv(out/'metricas_por_anio_semilla.csv', index=False)
    pd.DataFrame(horarios).to_csv(out/'metricas_por_hora_semilla.csv', index=False)
    resumen = tabla.groupby('modelo').agg(MASE=('MASE','mean'), DE_MASE=('MASE','std'),
                                          MAE_kwh=('MAE_kwh','mean'), RMSE_kwh=('RMSE_kwh','mean')).sort_values('MASE')
    resumen.to_csv(out/'resumen_resultados.csv')
    errores = {k: np.abs(np.asarray(v, dtype='float64') - y_real).mean(axis=(0,2)) / datos.q168 for k,v in todas.items()}
    anios = pd.DatetimeIndex(dias).year.to_numpy()
    contrastes = estadistica(errores, anios)
    contrastes.to_csv(out/'contrastes.csv', index=False)
    assert all(sha(p) == huellas[str(p)] for p in entradas), 'Los artefactos congelados cambiaron'
    guardar(out/'VERIFICACION.json', dict(semillas=verificaciones, origenes=1461,
              expertos_y_router_en_gpu=True, originales_conservados=True, q168=datos.q168,
              n_entrada=n_entrada))
    texto = '# M1 · evaluación retrospectiva 2022–2025\n\n'
    texto += '1.461 orígenes diarios, 24 horas, cinco semillas (101–105). Pesos y escaladores restaurados en GPU y congelados; sin reentrenamiento ni selección adicional. Configuración 8: ancho 64, dropout 0,1 y coeficiente auxiliar 0,1.\n\n'
    texto += resumen.to_string(float_format=lambda v:f'{v:.6f}') + '\n\n'
    texto += 'MASE medio entre semillas; DE_MASE describe inicialización, no incertidumbre temporal. RMSE es la media del RMSE de cada semilla.\n\n'
    texto += '## Contrastes primarios\n\n' + contrastes[contrastes.bloque_dias==14].to_string(index=False) + '\n\n'
    texto += ('Ganancia positiva indica menor MASE de M1. Bootstrap pareado: 9.999 réplicas, bloques circulares de 14 días dentro de cada año; sensibilidades 7 y 28. Los errores se promedian entre semillas antes del remuestreo. IC95 puntuales. Se conserva la familia global de 28 hipótesis: las otras 24 están pendientes; el ajuste informado es una cota superior conservadora de Holm obtenida completándolas con unos, no un ajuste final de todos los modelos. No se declara equivalencia a partir de ausencia de significancia.\n\n'
              'Los errores que alimentan M1 se incorporan con retraso de dos días, inicializados con predicciones OOF anteriores a 2022; no se actualizan parámetros ni escaladores en prueba. Los rasgos reproducen exactamente los de entrenamiento y superan la prueba de invariancia a objetivos contemporáneos y futuros.\n\n'
              'El benchmark de estos años ya era conocido durante el diseño: la evaluación es retrospectiva e informada, no un conjunto nunca consultado. El control sin pérdida auxiliar no dispone de checkpoint final y no se evalúa ni se reentrena aquí. Los resultados no se publican automáticamente.\n')
    (out/'INFORME.md').write_text(texto, encoding='utf-8')
    guardar(out/'FINALIZADO.json', dict(fecha=dt.datetime.now(ZoneInfo('America/Bogota')).isoformat(), prueba_m1_ejecutada=True,
                                       entrenamiento_adicional=False, semillas=semillas, origenes=1461))
    guardar(out/'MANIFIESTO.json', [dict(archivo=str(p.relative_to(out)), bytes=p.stat().st_size, sha256=sha(p))
                                     for p in sorted(out.rglob('*')) if p.is_file() and p.name not in ('MANIFIESTO.json','proceso.lock','ejecucion.log')])
    print(resumen.to_string(), flush=True)
    print('FINALIZADO: ' + str(out), flush=True)


if __name__ == '__main__':
    main()

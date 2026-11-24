"""Colombia, diez semillas y ablaciones finales; sustituye la cola extensa.

No busca hiperparámetros ni ajusta nuevos comparadores del Top10. Las fuentes
registradas se verifican antes de ejecutar; la prueba no cambia los pesos.
"""
from e3_gpu_comun import *
import argparse
import fcntl
import shutil
import subprocess
import traceback
import m4_cierre_colombia as COL
import m4_cierre_controles as C
import m4_cierre_recuperacion as REC
import m4_cierre_fuertes as FU
import m4_cierre_procedencia as PRO
import m4_cierre_complemento as EXTRA
import m4_cierre_diagnosticos as DIA
import m4_cierre_estadistica as EST
from e3_modelo1 import dividir
from ejecutar_cierre_m4 import guardar_pred, traza

SEMILLAS = list(range(101, 111))
ARCHIVOS_AJUSTE = {'modelo.pt', 'historia.csv', 'escaladores.json',
                   'TERMINADO.json', 'controles.npz', 'contrato.json'}


def copiar_ajuste(origen, destino):
    """Reutiliza únicamente ajustes terminados; conserva optimizadores locales."""
    if not (origen / 'TERMINADO.json').exists():
        return
    destino.mkdir(parents=True, exist_ok=True)
    for p in origen.iterdir():
        if p.name in ARCHIVOS_AJUSTE and not (destino / p.name).exists():
            shutil.copy2(p, destino / p.name)


def huellas_pesos(run, work):
    pesos = [p for p in (run / 'colombia').rglob('modelo.pt')
             if 'base_M1_nueva' not in p.parts]
    for carpeta in ['expertos_finales', 'final']:
        pesos += list((work / 'M1' / carpeta).rglob('modelo.pt'))
    pesos += list((COL.HIST / 'expertos_finales').rglob('modelo.pt'))
    pesos += list((COL.HIST / 'final').rglob('modelo.pt'))
    return {str(p.relative_to(REPO)): sha(p) for p in sorted(set(pesos))}


def guardar_manifiesto(run):
    guardar(run / 'MANIFIESTO_FINAL.json', [
        dict(archivo=str(p.relative_to(run)), sha256=sha(p), bytes=p.stat().st_size)
        for p in sorted(run.rglob('*'))
        if p.is_file() and p.name not in ['MANIFIESTO_FINAL.json', 'cola.lock']
        and p.suffix not in ['.log', '.lock', '.tmp', '.pyc']
        and 'progreso' not in p.relative_to(run).parts
        and p.name != 'reanudar.pt'
    ])


def resumir(run, out, datos):
    destino = run / 'colombia/prueba'
    modelos = leer(destino / 'TERMINADO_101.json')['modelos']
    errores, metricas_semilla, resumen = {}, [], []
    dias = y = None
    for v in modelos:
        perdidas = []
        for s in SEMILLAS:
            with np.load(destino / f'{v}_semilla_{s}.npz') as z:
                if dias is None:
                    dias, y = z['dia'], z['y']
                assert np.array_equal(dias, z['dia']) and np.array_equal(y, z['y'])
                pred = z['pred'].astype('float64')
                perdidas.append(np.abs(y - pred).mean(-1) / datos.q168)
                metricas_semilla.append(dict(modelo=v, semilla=s,
                                              **metricas(y, pred, datos.q168)))
        errores[v] = np.asarray(perdidas)
    pos = EXTRA.posiciones(datos, dias)
    for v, pred in datos.ingenuos(pos).items():
        guardar_pred(destino, v, 0, dias, y, pred)
        errores[v] = (np.abs(y - pred).mean(-1) / datos.q168)[None, :]
        metricas_semilla.append(dict(modelo=v, semilla=0,
                                      **metricas(y, pred, datos.q168)))
    historicos, info = EST.historicos_colombia(dias, y)
    assert len(historicos) == 10, 'Faltan comparaciones históricas verificadas'
    errores.update(historicos)
    errores['M4_desalineado_promedio'] = np.mean([
        errores[f'M4_desalineado_{d}'] for d in [49, 91, 154, 217, 301]], axis=0)
    for v, e in errores.items():
        resumen.append(dict(dataset='colombia', modelo=v, MASE=float(e.mean()),
                            semillas=info.get(v, {}).get('semillas', len(e)),
                            DE_semillas=info.get(v, {}).get('DE_semillas',
                                float(e.mean(1).std(ddof=1)) if len(e) > 1 else None)))
    estad = run / 'estadistica'
    estad.mkdir(exist_ok=True)
    p = estad / 'perdidas_colombia.npz'
    np.savez_compressed(p, dia=dias, **errores)
    pd.DataFrame(metricas_semilla).to_csv(run / 'metricas_por_semilla.csv', index=False)
    tabla = pd.DataFrame(resumen).sort_values('MASE')
    tabla.to_csv(run / 'resumen_resultados.csv', index=False)
    DIA.supuestos([p], estad)
    from m4_cierre_completar_top10 import completar_familia
    completar_familia(run)
    pendientes = [
        'UCI, GEFCom y OPSD: diferidos por presupuesto temporal',
        'Nuevos ajustes y validación ampliada del Top10: diferidos',
        'Validación externa forward chaining de las nuevas ablaciones residuales y textuales: diferida',
        'Ampliación de semillas según potencia: diferida; no se acredita potencia 0,90',
        'Confirmación prospectiva y originalidad metodológica: pendientes',
        'Windows nativo y ONNX GPU: no ejecutados; se conserva el rechazo histórico en empates'
    ]
    guardar(run / 'MATRIZ_CIERRE.json', dict(opcion='B', datasets=['colombia'],
            semillas=SEMILLAS, certificacion_E3=False, pendientes=pendientes))
    lineas = ['# Evaluación reducida B: Colombia', '',
        'Diez semillas 101–110, configuración 6 fija y pesos restaurados en su mejor época interna. '
        'Se reutilizan los ajustes completos compatibles. Las nuevas ablaciones se ajustan '
        'una sola vez por semilla: optimización hasta 2020, parada interna en 2021 y prueba '
        '2022–2025. No se realiza una nueva búsqueda ni una nueva validación externa de esas ablaciones.', '',
        'La base conserva la construcción OOF por prefijos. La memoria operativa solo '
        'incorpora residuos observables con retraso de dos días y antigüedad máxima de 365 días. '
        'Las retropredicciones en muestra solo sustituyen los residuos históricos de la ablación correspondiente.', '',
        'TimesNet y N-BEATS se evalúan desde cinco pesos históricos sin entrenamiento; '
        'los demás comparadores históricos conservan sus pérdidas publicadas y sus cinco semillas. '
        'Las realizaciones no se emparejan artificialmente por identificadores de semilla. '
        'La unidad inferencial es el origen diario, promediando las pérdidas entre semillas.', '',
        '| Procedimiento | MASE | Semillas |', '|---|---:|---:|']
    lineas += [f'| {r.modelo} | {r.MASE:.6f} | {r.semillas} |'
               for r in tabla.itertuples()]
    lineas += ['', 'Los contrastes finales utilizan 49.999 réplicas, bloques esperados de 14 días, '
        'DM–HLN con HAC 13, GW bilateral con instrumento observable de rezago dos y Holm '
        'con cota familiar de 1.536 hipótesis. Las no ejecutadas no se presentan como resultados; '
        'se conserva la cota familiar previamente registrada. Un rechazo GW no establece dirección.', '',
        'Colombia conserva carácter retrospectivo por exposición previa a prueba. No se '
        'certifica potencia, originalidad ni cumplimiento completo del E3.', '',
        '## Alcance pendiente', ''] + ['- ' + t for t in pendientes]
    (run / 'INFORME.md').write_text('\n'.join(lineas) + '\n', encoding='utf-8')
    (run / 'RESULTADOS_FINALES.md').write_text(
        '# Resultados de la opción B\n\n'
        '- [Informe y alcance](INFORME.md).\n'
        '- [Métricas por semilla](metricas_por_semilla.csv).\n'
        '- [Resumen](resumen_resultados.csv).\n'
        '- [Contrastes temporales finales](ampliacion_top10/contrastes_F2_Top10_completo.csv).\n'
        '- [Pendientes explícitos](MATRIZ_CIERRE.json).\n\n'
        'Esta ejecución sustituye operativamente la cola extensa, que queda diferida.\n', encoding='utf-8')


def ejecutar(out, publicar, apagar):
    run = out / 'reducido_B'
    run.mkdir(exist_ok=True)
    lock = (run / 'cola.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    registro = leer(run / 'REGISTRO_PREVIO.json')
    assert registro['opcion'] == 'B' and registro['semillas'] == SEMILLAS
    assert leer(run / 'LISTO_PARA_COLA.json')['verificado']
    for seccion in ['fuentes', 'entradas']:
        assert all(sha(REPO / p) == h for p, h in registro[seccion].items()), seccion
    assert all(sha(p) == h for p, h in registro['paquete_fuentes'].items())
    hechas, total, etapa = 0, 44, 'Preparación de la opción B'
    gpu = 'AMD Radeon · inferencia y entrenamiento GPU obligatorios'

    def progreso(**kw):
        estado = dict(estado='Ejecutando', fase=etapa, tareas=hechas, total=total,
                      gpu=gpu, actualizado=pd.Timestamp.now(tz='America/Bogota').isoformat())
        estado.update(kw)
        guardar(out / 'progreso/estado.json', estado)

    progreso()
    etapa = 'Predicciones del Top10 desde pesos originales; sin reentrenar'
    progreso(etiqueta='TimesNet y N-BEATS · diez checkpoints')
    leg = out / 'historicos_nbeats_timesnet'
    if not (leg / 'FINALIZADO.json').exists():
        # El proceso padre todavía no inicializa GPU: un único consumidor de cómputo.
        subprocess.run([sys.executable, str(REPO / 'scripts/m4_cierre_predecir_historicos.py'),
                        '--out', str(out)], check=True)
    assert leer(leg / 'FINALIZADO.json')['pesos_congelados']
    hechas += 1
    etapa = 'Entrenamiento final de controles numéricos'
    torch.set_num_threads(4)
    assert torch.cuda.is_available()
    gpu = torch.cuda.get_device_name(0)
    datos = DatosNumericos(PAQUETE)
    datos.nombre = 'colombia'
    assert datos.hash_panel == registro['panel_sha256']
    car, work = run / 'colombia', out / 'colombia'
    car.mkdir(exist_ok=True)
    metas = {}
    for s in SEMILLAS:
        progreso(etiqueta=f'Controles numéricos · semilla {s}')
        if s > 105:
            assert (work / f'base_oof/semilla_{s}.npz').exists(), 'Base nueva incompleta; no ampliar la cola'
            assert len(list((work / 'M1/expertos_finales').glob(f'*/semilla_{s}/TERMINADO.json'))) == 3
        meta = COL.preparar(s, work, progreso)
        metas[s] = meta
        a, es, ev = dividir(meta, 2022)
        assert not len(ev) and meta['dia'][a[-1]] < meta['dia'][es[0]] < '2022-01-01'
        for v in C.VARIANTES:
            d = car / 'final' / v / f'semilla_{s}'
            copiar_ajuste(work / 'final' / v / f'semilla_{s}', d)
            previo = (REPO / 'results/m4_gpu_20261122_194500/final/M4_N' /
                      f'semilla_{s}/modelo.pt') if v == 'M4_N' and s <= 105 else None
            C.ajustar_control(meta, s, a, es, ev, v, d, datos.q168, progreso, previo)
        w = REC.ventanas(datos, meta['dia'])
        for v in REC.NOMBRES:
            d = car / 'final' / v / f'semilla_{s}'
            copiar_ajuste(work / 'final' / v / f'semilla_{s}', d)
            REC.ajustar_recuperacion(meta, w, a, es, s, v, d, datos.q168, progreso)
        FU.ajustar(meta, a, es, car / f'fuertes/semilla_{s}', datos.q168)
        hechas += 1
        progreso()
    etapa = 'Ablaciones de procedencia residual · ajuste final únicamente'
    for s in SEMILLAS:
        progreso(etiqueta=f'Residuos en muestra y OOF fijo · semilla {s}')
        a, es, ev = dividir(metas[s], 2022)
        EXTRA.ajustar_procedencias(datos, metas[s], s, [('2022', a, es, ev)],
                                  car, work, progreso)
        hechas += 1
        progreso()
    etapa = 'Ablaciones de texto y placebos · ajuste final únicamente'
    textos = {col: EXTRA.TEX.Texto(COL.HIST.parent, col, final=True)
              for col in ['disponible_declarado', 'disponible_estricto']}
    for s in SEMILLAS:
        progreso(etiqueta=f'Canales oficiales/prensa, metadatos y placebos · semilla {s}')
        a, es, ev = dividir(metas[s], 2022)
        EXTRA.entrenar_texto(metas[s], s, 'final', a, es, ev, car / 'final',
                             textos, progreso, datos.q168)
        hechas += 1
        progreso()
    pesos = huellas_pesos(run, work)
    congelacion = run / 'CONGELACION_PREPRUEBA.json'
    if congelacion.exists():
        assert leer(congelacion)['pesos'] == pesos
    else:
        guardar(congelacion, dict(fecha=pd.Timestamp.now(tz='America/Bogota').isoformat(),
                                 semillas=SEMILLAS, pesos=pesos))
    # Referencias completas y copias de las bases nuevas necesarias para inferencia.
    for s in SEMILLAS:
        np.savez_compressed(car / f'base_oof_semilla_{s}.npz', **metas[s])
    for carpeta in ['expertos_finales', 'final']:
        origen = work / 'M1' / carpeta
        for p in origen.rglob('*'):
            if p.is_file() and p.name in ARCHIVOS_AJUSTE:
                d = car / 'base_M1_nueva' / carpeta / p.relative_to(origen)
                d.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, d)
    etapa = 'Prueba Colombia 2022–2025 · todos los pesos congelados'
    destino = car / 'prueba'
    for s in SEMILLAS:
        progreso(etiqueta=f'Inferencia de todos los controles y ablaciones · semilla {s}')
        marca = destino / f'TERMINADO_{s}.json'
        if not marca.exists():
            meta = COL.prueba(s, work, metas[s])
            ix = np.flatnonzero(meta['anio'] >= 2022)
            dias = meta['dia'][ix]
            pos = EXTRA.posiciones(datos, dias)
            y = datos.Y[pos[:, None] + np.arange(24)]
            assert y.shape == (1461, 24)
            assert np.array_equal(y.astype('float32'), meta['y'][ix])
            pr = {'M1': meta['pred_base'][ix]}
            for v in C.VARIANTES:
                ck = cargar(car / 'final' / v / f'semilla_{s}/modelo.pt')
                pr[v] = C.predecir(ck, meta, ix)
                if v == 'M4_N':
                    destino.mkdir(exist_ok=True)
                    traza(meta, ix, ck, destino / f'memoria_semilla_{s}.npz')
            w = REC.ventanas(datos, meta['dia'])
            for v in REC.NOMBRES:
                ck = cargar(car / 'final' / v / f'semilla_{s}/modelo.pt')
                pr[v] = REC.predecir_recuperacion(meta, w, ix, ck, datos.q168)
            pr.update(FU.predecir(meta, ix, car / f'fuertes/semilla_{s}', datos.q168))
            for modo in PRO.MODOS:
                pr[modo] = EXTRA.pred_procedencia(modo, car, 'final', s, meta, ix, pr['M4_N'])
            for v in EXTRA.TEXTOS:
                ck = cargar(car / 'final' / v / f'semilla_{s}/modelo.pt')
                col = 'disponible_estricto' if v.endswith('_estricto') else 'disponible_declarado'
                pr[v] = EXTRA.texto_pred(meta, textos[col], ck, ix, datos.q168)
            for v, pred in pr.items():
                guardar_pred(destino, v, s, dias, y, pred)
            guardar(marca, dict(semilla=s, modelos=list(pr), prueba=True))
        hechas += 1
        progreso()
    assert all(sha(REPO / p) == h for p, h in pesos.items()), 'Pesos modificados durante prueba'
    etapa = 'Resumen e inferencia estadística temporal'
    progreso()
    insumos = run / 'insumos'
    insumos.mkdir(exist_ok=True)
    for nombre in ['nuevo_timesnet__N.json', 'nuevo_nbeats__N.json', 'top10_adicionales.json']:
        shutil.copy2(out / 'insumos' / nombre, insumos / nombre)
    resumir(run, out, datos)
    for seccion in ['fuentes', 'entradas']:
        assert all(sha(REPO / p) == h for p, h in registro[seccion].items()), seccion
    assert all(sha(p) == h for p, h in registro['paquete_fuentes'].items())
    guardar(run / 'FINALIZADO.json', dict(fecha=pd.Timestamp.now(tz='America/Bogota').isoformat(),
        opcion='B', entrenamiento=True, prueba=True, datasets=['colombia'], semillas=SEMILLAS,
        certificacion_E3=False))
    hechas += 1
    guardar_manifiesto(run)
    if publicar:
        from publicar_cierre_m4 import publicar as subir
        etapa = 'Publicación y verificación de predicciones históricas'
        progreso()
        if not (leg / 'PUBLICADO.json').exists():
            subir(leg)
        assert leer(leg / 'PUBLICADO.json')['verificado']
        hechas += 1
        etapa = 'Publicación y verificación de resultados reducidos B'
        progreso()
        if not (run / 'PUBLICADO.json').exists():
            subir(run)
        assert leer(run / 'PUBLICADO.json')['verificado']
        hechas += 1
        progreso()
    if apagar:
        assert publicar and leer(run / 'PUBLICADO.json')['verificado']
        assert leer(leg / 'PUBLICADO.json')['verificado']
        r = subprocess.run(['shutdown', '-h', '+2'], capture_output=True, text=True)
        guardar(run / 'APAGADO.json', dict(retorno=r.returncode, salida=r.stdout, error=r.stderr))
        assert r.returncode == 0, r.stderr
    etapa = 'Opción B finalizada y publicada; apagado en dos minutos' if apagar else 'Opción B finalizada'
    progreso(estado='Finalizado', tareas=total)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--publicar', action='store_true')
    ap.add_argument('--apagar', action='store_true')
    args = ap.parse_args()
    try:
        ejecutar(args.out.resolve(), args.publicar, args.apagar)
    except Exception:
        guardar(args.out / 'reducido_B/ERROR.json', dict(error=traceback.format_exc(), apagado=False))
        guardar(args.out / 'progreso/estado.json', dict(estado='Error', fase='Cola B detenida; apagado cancelado',
            tareas=0, total=1, gpu='Consultar reducido_B/ERROR.json',
            actualizado=pd.Timestamp.now(tz='America/Bogota').isoformat()))
        raise

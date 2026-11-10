"""Importancia agrupada mediante inferencia de checkpoints congelados."""
from pathlib import Path
from types import SimpleNamespace
import argparse
import hashlib
import json
import platform
import sys
import time
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODELOS = ['mlp', 'arx_lineal', 'cnn1d', 'lstm']
REPETICIONES = 10
SEMILLA_PERMUTACION = 11037
GRUPOS_CAL = {
    'semanal_festivos': ['cal_dow', 'cal_es_festivo', 'cal_es_domingo',
        'cal_es_sabado', 'cal_es_no_laboral', 'cal_tipo_dia',
        'cal_dias_al_siguiente_festivo', 'cal_dias_desde_festivo_previo',
        'cal_es_vispera_festivo', 'cal_es_dia_post_festivo',
        'cal_es_fin_semana_largo', 'cal_dow_sin', 'cal_dow_cos'],
    'anual_temporadas': ['cal_mes', 'cal_dia_anio', 'cal_es_semana_santa',
        'cal_es_temporada_navidena', 'cal_es_fin_de_anio', 'cal_doy_sin', 'cal_doy_cos'],
    'hora': ['cal_hora', 'cal_hora_sin', 'cal_hora_cos'],
}
GRUPOS = ['demanda_completa', 'perfil_demanda', 'nivel_ventana',
          'calendario_completo', 'semanal_festivos', 'anual_temporadas'] + [f'dia_{i}' for i in range(1, 8)]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def indices_donantes(fechas, repeticion):
    """Traslada circularmente orígenes completos dentro de cada año."""
    rng = np.random.default_rng(SEMILLA_PERMUTACION + repeticion)
    perm = np.arange(len(fechas))
    for anio in sorted(set(fechas.year)):
        ix = np.flatnonzero(fechas.year == anio)
        if len(ix) < 57:
            raise ValueError('Se requieren al menos 57 orígenes diarios por año')
        desplazamiento = int(rng.integers(28, len(ix) - 27))
        perm[ix] = np.roll(ix, desplazamiento)
    assert np.array_equal(np.sort(perm), np.arange(len(fechas)))
    assert np.array_equal(fechas.year[perm], fechas.year)
    assert np.all(perm != np.arange(len(fechas)))
    return perm


def perturbar(xp, cf, nivel, donantes, grupo, columnas_cal):
    """Copia solo el grupo perturbado; las etiquetas nunca entran aquí."""
    if grupo not in GRUPOS:
        raise ValueError(f'Grupo no reconocido: {grupo}')
    x, c, n = xp, cf, nivel
    if grupo in ('demanda_completa', 'perfil_demanda'):
        x = xp.copy()
        x[:, :, 0] = xp[donantes, :, 0]
    if grupo in ('demanda_completa', 'nivel_ventana'):
        n = nivel[donantes]
    if grupo.startswith('dia_'):
        dia = int(grupo.split('_')[1])
        lo, hi = xp.shape[1] - 24 * dia, xp.shape[1] - 24 * (dia - 1)
        x = xp.copy()
        x[:, lo:hi, 0] = xp[donantes, lo:hi, 0]
    if grupo in ('calendario_completo', 'semanal_festivos', 'anual_temporadas'):
        cols = columnas_cal if grupo == 'calendario_completo' else GRUPOS_CAL[grupo]
        ix = [columnas_cal.index(k) for k in cols]
        x = xp.copy()
        c = cf.copy()
        # xp concatena demanda y calendario pasado; cf conserva calendario futuro.
        x[:, :, np.asarray(ix) + 1] = xp[donantes][:, :, np.asarray(ix) + 1]
        c[:, :, ix] = cf[donantes][:, :, ix]
    return x, c, n


def restaurar_entradas(panel, fechas, escalador):
    """Usa solo escaladores publicados; no ajusta estadísticas."""
    cols = escalador['columnas_cal']
    assert escalador['columnas_x'] == ['x_demanda_real_kwh']
    assert escalador['datos']['ventana'] == 168
    assert escalador['datos']['normalizacion'] == 'nivel'
    assert pd.Timestamp(escalador['fin_ajuste']) < pd.Timestamp('2022-01-01', tz='America/Bogota')
    pos = panel.index.get_indexer(fechas)
    assert (pos >= 168).all()
    pasada = pos[:, None] - np.arange(168, 0, -1)
    futura = pos[:, None] + np.arange(24)
    assert np.all(panel.index.asi8[pasada[:, -1]] < fechas.asi8)
    historia = panel['x_demanda_real_kwh'].to_numpy(dtype=np.float64)[pasada]
    nivel = historia.mean(axis=1)
    x = (historia / nivel[:, None]).astype('float32')
    if escalador['secuencial']:
        x = (x - 1.) / escalador['sigma_demanda']
    media = np.array([escalador['media'][k] for k in cols])
    std = np.array([escalador['std'][k] for k in cols])
    cal = panel[cols].to_numpy(dtype=np.float64)
    cp = ((cal[pasada] - media) / std).astype('float32')
    cf = ((cal[futura] - media) / std).astype('float32')
    y = panel['y_demanda_real_kwh'].to_numpy()[futura]
    xp = np.concatenate([x[:, :, None], cp], axis=2)
    assert xp.dtype == cf.dtype == np.float32
    assert np.isfinite(xp).all() and np.isfinite(cf).all() and (nivel > 0).all()
    indices_hora = [cols.index(k) for k in GRUPOS_CAL['hora']]
    assert all(np.array_equal(cp[:, :, j], np.broadcast_to(cp[0:1, :, j], cp[:, :, j].shape)) for j in indices_hora)
    assert all(np.array_equal(cf[:, :, j], np.broadcast_to(cf[0:1, :, j], cf[:, :, j].shape)) for j in indices_hora)
    return xp, cf, nivel, y, cols


def predecir(red, xp, cf, nivel, torch):
    salida = []
    for i in range(0, len(nivel), 256):
        yhat = red(torch.from_numpy(xp[i:i+256]), torch.from_numpy(cf[i:i+256]))
        salida.append(yhat.cpu().numpy())
    pred = np.concatenate(salida).astype(np.float64) * nivel[:, None]
    assert pred.shape == (len(nivel), 24) and np.isfinite(pred).all()
    return pred


def ejecutar(verificar=False):
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(SEMILLA_PERMUTACION)
    sha = json.loads((ROOT/'results/jbook/modelos_actualizados.json').read_text())['commit']
    snapshot = ROOT/'results/repositorio'/sha
    sys.path.insert(0, str(snapshot/'src'))
    from demanda import modelos_base as M
    from demanda.reentrenamiento_v3 import Secuencial as CNN
    from demanda.variantes_v4 import Secuencial as LSTM
    registro = json.loads((snapshot/'results/consolidado/modelos.json').read_text(encoding='utf-8'))
    huellas = {r['archivo']:r['sha256'] for r in json.loads((snapshot/'IMPORTACION.json').read_text(encoding='utf-8'))['archivos']}
    out = ROOT/'results/jbook/importancia_variables'
    out.mkdir(parents=True, exist_ok=True)
    protocolo = dict(version=1, escenario='diagnostico_retrospectivo_prueba_2022_2025',
        modelos=MODELOS, semillas=list(range(1000,1005)), repeticiones=REPETICIONES,
        semilla_permutacion=SEMILLA_PERMUTACION, grupos=GRUPOS,
        grupos_calendario=GRUPOS_CAL, donantes='desplazamiento_circular_por_anio',
        desplazamiento_minimo_dias=28, denominador_mase=215617.90361589016,
        dispositivo='cpu', hilos=1, batch=256, entrenamiento=False,
        reajuste_escaladores=False, seleccion_con_resultados=False,
        hora='invariante_entre_origenes_no_identificable_con_esta_permutacion')
    p = out/'protocolo.json'
    if p.exists():
        assert json.loads(p.read_text(encoding='utf-8'))['protocolo'] == protocolo
    else:
        p.write_text(json.dumps(dict(registrado=pd.Timestamp.now(tz='America/Bogota').isoformat(),
            protocolo=protocolo), ensure_ascii=False, indent=2), encoding='utf-8')
    primer = registro['mlp']['artefactos'][0]
    with np.load(snapshot/primer['predicciones'], allow_pickle=False) as z:
        fechas = pd.to_datetime(z['origen'], unit='ns', utc=True).tz_convert('America/Bogota')
    assert fechas.equals(pd.date_range('2022-01-01','2025-12-31',freq='D',tz='America/Bogota'))
    cols = json.loads((snapshot/primer['escaladores']).read_text(encoding='utf-8'))['columnas_cal']
    assert sorted(sum(GRUPOS_CAL.values(), [])) == sorted(cols)
    panel = pd.read_parquet(ROOT/'data/processed/panel_horario.parquet', columns=['x_demanda_real_kwh','y_demanda_real_kwh',*cols])
    donantes = [indices_donantes(fechas,r) for r in range(REPETICIONES)]
    np.savez_compressed(out/'donantes.npz', fechas_utc=fechas.asi8, indices=np.stack(donantes))
    benchmark_path = ROOT/'results/jbook/modelos_actuales/benchmark.csv'
    benchmark_hash = sha256(benchmark_path)
    filas, validaciones = [], []
    inicio = time.perf_counter()
    with torch.inference_mode():
        for nombre in MODELOS:
            for a in sorted(registro[nombre]['artefactos'], key=lambda x:x['semilla']):
                for campo in ('pesos','escaladores','predicciones'):
                    assert sha256(snapshot/a[campo]) == huellas[a[campo]]
                esc = json.loads((snapshot/a['escaladores']).read_text(encoding='utf-8'))
                xp, cf, nivel, y, cols = restaurar_entradas(panel, fechas, esc)
                shapes = SimpleNamespace(x=xp[:, :, :1], cal_p=xp[:, :, 1:], cal_f=cf)
                params = registro[nombre]['configuracion']['params']
                red = (LSTM(nombre, params, shapes) if nombre=='lstm' else
                       CNN(nombre, params, shapes) if nombre=='cnn1d' else
                       M.construir_red(nombre, params, shapes))
                ck = torch.load(snapshot/a['pesos'], map_location='cpu', weights_only=True)
                red.load_state_dict(ck['state_dict'], strict=True)
                red.eval()
                esperado = np.load(snapshot/a['predicciones'], allow_pickle=False)
                assert np.array_equal(esperado['y'], y)
                assert np.array_equal(esperado['origen'], fechas.asi8)
                base = predecir(red, xp, cf, nivel, torch)
                error = np.abs(base-y)
                mae = error.mean()
                diferencia = float(np.max(np.abs(base-esperado['yhat'])))
                diferencia_mae = float(abs(mae-np.abs(esperado['yhat']-y).mean()))
                assert diferencia < 100 and diferencia_mae < 5, (nombre, diferencia, diferencia_mae)
                validaciones.append(dict(modelo=nombre, semilla=a['semilla'],
                    max_diferencia_pred_kWh=diferencia, diferencia_MAE_kWh=diferencia_mae,
                    MAE_cpu_kWh=float(mae), checkpoint_sha256=huellas[a['pesos']]))
                if not verificar:
                    for grupo in GRUPOS:
                        if nombre=='arx_lineal' and grupo.startswith('dia_') and int(grupo[4:])>2:
                            for r in range(REPETICIONES):
                                filas.append(dict(modelo=nombre,semilla=a['semilla'],grupo=grupo,
                                    repeticion=r,usa_grupo=False,MAE_base=mae,MAE_perturbado=mae,
                                    delta_MAE=0.,delta_MASE=0.,delta_rel_pct=0.))
                            continue
                        for r, perm in enumerate(donantes):
                            x, c, n = perturbar(xp, cf, nivel, perm, grupo, cols)
                            pred = predecir(red, x, c, n, torch)
                            nuevo = float(np.abs(pred-y).mean())
                            filas.append(dict(modelo=nombre,semilla=a['semilla'],grupo=grupo,
                                repeticion=r,usa_grupo=True,MAE_base=mae,MAE_perturbado=nuevo,
                                delta_MAE=nuevo-mae,delta_MASE=(nuevo-mae)/esc['escala_mase'],
                                delta_rel_pct=100*(nuevo/mae-1)))
                pd.DataFrame(validaciones).to_csv(out/'reproduccion_baseline.csv', index=False)
                if filas:
                    pd.DataFrame(filas).to_csv(out/'permutaciones.csv', index=False)
                print(f"{nombre} semilla {a['semilla']}: {'verificado' if verificar else 'completo'}; {time.perf_counter()-inicio:.1f} s", flush=True)
    assert sha256(benchmark_path) == benchmark_hash
    reporte = dict(completo=True,solo_verificacion=verificar,horas=35064,origenes=1461,
        modelos=4,checkpoints=20,repeticiones=REPETICIONES,grupos=len(GRUPOS),
        filas=len(filas),entrenamiento=False,publicado=False,benchmark_intacto=True,
        max_diferencia_pred_kWh=max(r['max_diferencia_pred_kWh'] for r in validaciones),
        max_diferencia_MAE_kWh=max(r['diferencia_MAE_kWh'] for r in validaciones),
        duracion_segundos=time.perf_counter()-inicio,torch=torch.__version__,
        numpy=np.__version__,pandas=pd.__version__,sistema=platform.platform(),
        codigo_sha256=sha256(Path(__file__)))
    (out/('validacion.json' if verificar else 'ejecucion.json')).write_text(json.dumps(reporte, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(reporte,ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--verificar', action='store_true')
    ejecutar(parser.parse_args().verificar)

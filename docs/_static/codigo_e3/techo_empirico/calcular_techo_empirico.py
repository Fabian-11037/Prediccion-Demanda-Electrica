"""Techo empírico y gamma desde pérdidas conservadas; no ejecuta modelos."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_NAME = 'techo_empirico_20261124'
Q = 215617.90361589016
BASE = 'ingenuo_168h'
M4 = 'm4_n'


def calcular(root=ROOT):
    root = Path(root)
    out = root / 'results' / OUT_NAME
    out.mkdir(parents=True, exist_ok=True)
    sources = [root / 'results/evaluacion_e2/perdidas_y_bootstrap.npz',
               root / 'results/evaluacion_e2/benchmark.csv',
               root / 'results/cierre_e3_sin_entrenar_20261124/perdidas_diarias.csv']
    benchmark = pd.read_csv(sources[1]).set_index('clave')
    with np.load(sources[0], allow_pickle=False) as z:
        days = pd.to_datetime(z['origen'], utc=True).tz_convert('America/Bogota')
        losses = pd.DataFrame(z['loss'], columns=z['nombres'], index=days)
    losses = losses.rename(columns={r.modelo: k for k, r in benchmark.iterrows()})
    assert set(losses) == set(benchmark.index)
    dates = days.strftime('%Y-%m-%d').to_numpy()
    assert len(days) == 1461 and days.is_unique and days.is_monotonic_increasing
    assert (np.diff(days.asi8) == 86400 * 10**9).all()
    assert (dates[0], dates[-1]) == ('2022-01-01', '2025-12-31')
    for key in benchmark.index:
        assert abs(losses[key].mean() - benchmark.loc[key, 'MASE']) < 2e-6, key

    # Reconstrucción de las diez pérdidas individuales de M4, antes de promediar.
    per_seed = []
    truth = None
    for seed in range(101, 111):
        path = root / f'results/m4_cierre_gpu_20261123/reducido_B/colombia/prueba/M4_N_semilla_{seed}.npz'
        sources.append(path)
        with np.load(path, allow_pickle=False) as z:
            assert np.array_equal(z['dia'], dates)
            y, pred = z['y'].astype(float), z['pred'].astype(float)
        assert y.shape == pred.shape == (1461, 24)
        if truth is None:
            truth = y
        else:
            assert np.array_equal(y, truth)
        per_seed.append(np.abs(pred - y).mean(axis=1) / Q)
    losses[M4] = np.mean(per_seed, axis=0)
    saved = pd.read_csv(sources[2])
    assert set(saved) == set(losses)
    assert np.allclose(saved[losses.columns], losses.to_numpy(), rtol=1e-11, atol=1e-12)
    assert abs(losses[M4].mean() - .5770369087419158) < 1e-10

    order = losses.mean().sort_values().index.tolist()
    losses = losses[order]
    values = losses.to_numpy()
    assert values.shape == (1461, 18) and np.isfinite(values).all() and (values >= 0).all()
    oracle = values.min(axis=1)
    oracle_without_m4 = losses.drop(columns=M4).min(axis=1).to_numpy()
    means = values.mean(axis=0)
    ref = oracle.mean()
    base_i = order.index(BASE)
    best_i = int(means.argmin())
    assert order[best_i] == M4 and means[base_i] > ref
    gamma = (means - ref) / (means[base_i] - ref)
    assert np.allclose(ref + gamma * (means[base_i] - ref), means)
    assert np.all(oracle[:, None] <= values + 1e-12)

    # Bootstrap estacionario pareado: el mismo día/bloque para los 18 métodos
    # y para su mínimo retrospectivo. Las semillas conservadas permanecen fijas.
    B, block, seed = 9999, 14, 20261124
    rng = np.random.default_rng(seed)
    indices = np.empty((B, len(days)), dtype=np.int32)
    indices[:, 0] = rng.integers(len(days), size=B)
    for t in range(1, len(days)):
        restart = rng.random(B) < 1 / block
        indices[:, t] = np.where(restart, rng.integers(len(days), size=B),
                                 (indices[:, t - 1] + 1) % len(days))
    combined = np.column_stack([values, oracle, oracle_without_m4])
    boot = np.empty((B, combined.shape[1]))
    for start in range(0, B, 128):
        boot[start:start + 128] = combined[indices[start:start + 128]].mean(axis=1)
    boot_ref = boot[:, len(order)]
    boot_gamma = (boot[:, :len(order)] - boot_ref[:, None]) / (boot[:, base_i] - boot_ref)[:, None]
    assert np.isfinite(boot_gamma).all() and np.all(boot[:, base_i] > boot_ref)
    ci_error = np.quantile(boot[:, :len(order)], [.025, .975], axis=0)
    ci_gamma = np.quantile(boot_gamma, [.025, .975], axis=0)
    ci_ref = np.quantile(boot_ref, [.025, .975])
    # Reparto fraccional de empates; no se favorece el orden de las columnas.
    tied = np.isclose(values, oracle[:, None], atol=1e-12, rtol=0)
    wins = (tied / tied.sum(axis=1, keepdims=True)).sum(axis=0)
    names = benchmark.modelo.to_dict() | {M4: 'M4ₙ'}
    seeds = benchmark.semillas.to_dict() | {M4: 10}
    rows = pd.DataFrame([dict(clave=k, modelo=names[k], semillas=int(seeds[k]),
        MASE=float(means[i]), MAE_kWh=float(means[i] * Q),
        MASE_ic_inf=float(ci_error[0, i]), MASE_ic_sup=float(ci_error[1, i]),
        gamma=float(gamma[i]), gamma_ic_inf=float(ci_gamma[0, i]), gamma_ic_sup=float(ci_gamma[1, i]),
        distancia_MASE=float(means[i] - ref), dias_minimo=float(wins[i]),
        porcentaje_dias_minimo=float(100 * wins[i] / len(days))) for i, k in enumerate(order)])
    rows.to_csv(out / 'metricas_gamma.csv', index=False)
    daily = losses.copy()
    daily.insert(0, 'dia', dates)
    daily['referencia_empirica'] = oracle
    daily['referencia_sin_M4'] = oracle_without_m4
    daily.to_csv(out / 'perdidas_diarias.csv', index=False)
    annual = pd.DataFrame({'anio': days.year, 'referencia_empirica': oracle,
        'M4_N': losses[M4].to_numpy(), 'TimesNet': losses['nuevo_timesnet__N'].to_numpy(),
        'ingenuo_semanal': losses[BASE].to_numpy()}).groupby('anio').mean().reset_index()
    annual.to_csv(out / 'resultados_anuales.csv', index=False)
    losses.corr(method='spearman').to_csv(out / 'correlaciones_perdidas.csv', index_label='clave')
    np.savez_compressed(out / 'bootstrap.npz', claves=np.array(order), MASE=boot[:, :len(order)],
                        referencia=boot_ref, gamma=boot_gamma, referencia_sin_M4=boot[:, -1])
    record = dict(fecha='2026-11-24', periodo=['2022-01-01', '2025-12-31'], dias=len(days),
        procedimientos=len(order), escala_Q_kWh=Q, baseline=BASE, mejor=order[best_i],
        referencia_MASE=float(ref), referencia_MAE_kWh=float(ref * Q),
        referencia_ic95_MASE=ci_ref.tolist(), referencia_sin_M4_MASE=float(oracle_without_m4.mean()),
        mejor_MASE=float(means[best_i]), baseline_MASE=float(means[base_i]),
        gamma_mejor=float(gamma[best_i]), gamma_mejor_ic95=ci_gamma[:, best_i].tolist(),
        fraccion_brecha_cerrada=float(1 - gamma[best_i]),
        reduccion_adicional_relativa=float((means[best_i] - ref) / means[best_i]),
        diferencia_MAE_kWh=float((means[best_i] - ref) * Q))
    protocol = dict(**record, horizonte_horas=24, unidad='origen diario',
        perdida='Media de MAE por semilla y origen, dividida por Q fijo de entrenamiento',
        referencia='Media temporal del mínimo entre las 18 pérdidas diarias promedio de semillas',
        formula_gamma='(E_modelo-E_referencia)/(E_ingenuo_semanal-E_referencia)',
        sentido='Menor gamma indica menor margen restante; baseline semanal=1; referencia empírica=0',
        bootstrap=dict(tipo='estacionario pareado', replicas=B, longitud_media_bloque_dias=block,
                       semilla=seed, nivel=.95, intervalo='percentil puntual', remuestreo_semillas=False),
        alcance='Referencia retrospectiva condicionada a los procedimientos, pesos, semillas y fechas conservados; no estima el error de Bayes ni constituye un predictor implementable',
        entrenamiento=False, inferencia_neuronal=False, seleccion_arquitecturas=False)
    hashes = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    checks = dict(fechas_unicas_y_completas=True, paridad_benchmark=True, paridad_perdidas_M4=True,
        minimo_por_origen_verificado=True, identidad_gamma=True, n_semillas_M4=10,
        n_comparadores_aprendidos=15, n_referencias_deterministas=2,
        empates=int((tied.sum(axis=1) > 1).sum()), fuentes_sha256=hashes,
        resultados_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob('*.csv')})
    for name, value in [('resumen.json', record), ('protocolo.json', protocol), ('verificacion.json', checks)]:
        (out / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(record, ensure_ascii=False, indent=2))
    return record


if __name__ == '__main__':
    calcular()

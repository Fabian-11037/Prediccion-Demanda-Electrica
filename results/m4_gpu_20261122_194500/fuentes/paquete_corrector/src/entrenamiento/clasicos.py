"""Baselines clásicos (§3.2, §7.4 S5): LightGBM directo multihorizonte y MSTL con ETS o ARIMA.

LightGBM recibe las mismas ventanas, escalas y particiones que las redes (``DatosNumericos``): demanda
normalizada de las 168 h, último valor y media de 24 h de cada otra entrada histórica e indicador, y el
calendario conocido de la hora pronosticada; una fila por (origen, hora). Objetivo: demanda / nivel - 1,
pérdida L1, parada temprana con la parada interna. Protocolo idéntico al de las redes: 20 configuraciones
en 2018 (semilla 43), 5 finalistas en 2019-2021 y 5 semillas finales.

MSTL (estacionalidades de 24 y 168 h) con tendencia ETS automática o ARIMA automática (statsforecast), se
ajusta en cada origen con las últimas W semanas de demanda disponibles (el dato de la hora previa al origen no
está publicado: se pronostican 25 pasos y se descarta el primero). W se elige en validación entre {4, 8, 12}.
Es determinista: una sola realización, como los pronósticos ingenuos.
"""
from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from demanda import metricas as MET

from .protocolo import guardar_json, leer_json


# ------------------------------------------------------------------------------------ LightGBM
def _tabla(conj) -> tuple[np.ndarray, np.ndarray]:
    n, L, C = conj.x.shape
    H = conj.cal_f.shape[1]
    base = np.concatenate([conj.x[:, :, 0], conj.x[:, -1, 1:], conj.x[:, -24:, 1:].mean(1)], axis=1)       # (n, 168 + 2 (C-1))
    filas = np.concatenate([np.repeat(base, H, axis=0), conj.cal_f.reshape(n * H, -1), np.tile(np.arange(H), n)[:, None]], axis=1)
    return filas.astype("float32"), (conj.y_norm - 1.0).reshape(-1).astype("float32")


def _parametros(p: dict, semilla: int, hilos: int) -> dict:
    return dict(objective="l1", learning_rate=p["learning_rate"], num_leaves=p["num_leaves"], min_child_samples=p["min_child_samples"],
                feature_fraction=p["feature_fraction"], bagging_fraction=p["bagging_fraction"], bagging_freq=1 if p["bagging_fraction"] < 1 else 0,
                lambda_l2=p["lambda_l2"], seed=int(semilla), deterministic=True, force_row_wise=True, num_threads=hilos, verbose=-1)


def corrida_lgbm(ctx, p: dict, semilla: int, clave: str, pos: dict, carpeta: Path, nombre_ev: str, guardar: bool = False) -> dict:
    import lightgbm as lgb
    carpeta = Path(carpeta)
    rr = carpeta / "resultado.json"
    if rr.exists():
        if ctx.barra is not None:
            ctx.barra.update(1)
        return leer_json(rr)
    carpeta.mkdir(parents=True, exist_ok=True)
    esc, conj = ctx.conjuntos(clave, pos)
    Xtr, ytr = _tabla(conj["tr"])
    Xes, yes = _tabla(conj["es"])
    Xev, _ = _tabla(conj["ev"])
    t0 = time.time()
    rondas = 30 if ctx.rapido else int(p["max_arboles"])
    m = lgb.train(_parametros(p, semilla, ctx.hilos), lgb.Dataset(Xtr, ytr), num_boost_round=rondas, valid_sets=[lgb.Dataset(Xes, yes)],
                  callbacks=[lgb.early_stopping(10 if ctx.rapido else int(p["parada"]), verbose=False)])
    seg = time.time() - t0
    d = ctx.datos
    pred = lambda c, X: (1.0 + m.predict(X, num_iteration=m.best_iteration).reshape(len(c), d.H)) * c.nivel[:, None]   # noqa: E731
    t1 = time.time()
    yhat = pred(conj["ev"], Xev)
    lat = 1000 * (time.time() - t1) / len(conj["ev"])
    met = MET.evaluar(conj["ev"].y, yhat, d.q168)
    res = dict(variante="N", configuracion=p, semilla=int(semilla), evaluacion=nombre_ev, dispositivo="cpu", n_parametros=int(m.num_trees()),
               n_parametros_definicion="número de árboles (los bosques no tienen parámetros entrenables comparables)", epocas=int(m.best_iteration),
               mejor_epoca=int(m.best_iteration), mejor_perdida_parada=float(m.best_score["valid_0"]["l1"]), segundos=round(seg, 1),
               n_ajuste=len(conj["tr"]), n_parada=len(conj["es"]), n_evaluacion=len(conj["ev"]),
               MASE_ajuste=MET.evaluar(conj["tr"].y, pred(conj["tr"], Xtr), d.q168)["MASE"], MASE_parada=MET.evaluar(conj["es"].y, pred(conj["es"], Xes), d.q168)["MASE"],
               metricas=met, mase_por_horizonte=[float(v) for v in MET.mase_por_horizonte(conj["ev"].y, yhat, d.q168)],
               costes=dict(segundos_entrenamiento=round(seg, 1), latencia_ms_por_origen_lote=lat, dispositivo="cpu"), diagnostico={})
    np.savez_compressed(carpeta / f"pred_{nombre_ev}.npz", origen=np.asarray(conj["ev"].origen.strftime("%Y-%m-%d"), dtype="U10"), yhat_kwh=yhat.astype("float32"),
                        y_kwh=conj["ev"].y.astype("float32"), mase_por_origen=d.mase_por_origen(conj["ev"].y, yhat))
    if guardar:
        m.save_model(str(carpeta / "modelo_lightgbm.txt"))
    guardar_json(rr, res)
    if ctx.barra is not None:
        ctx.barra.update(1)
    ctx.log(f"LightGBM cfg {p['id']:02d} semilla {semilla} {nombre_ev}: MASE {met['MASE']:.4f} | {res['epocas']} árboles | {seg:.0f} s", consola=False)
    return res


def lightgbm(ctx, configs: list[dict]) -> dict:
    b, sal = ctx.cfg["busqueda"], ctx.salida
    if ctx.rapido:
        configs = [c for c in configs if c["id"] in ctx.rapido["configuraciones"]]
    pl = {f["anio"]: f for f in ctx.datos.pliegues()}
    pos0 = ctx.recortar(dict(tr=pl[b["pliegue_inicial"]]["tr"], es=pl[b["pliegue_inicial"]]["es"], ev=pl[b["pliegue_inicial"]]["val"]))
    filas = [dict(id=p["id"], pliegue=b["pliegue_inicial"], MASE=corrida_lgbm(ctx, p, b["semilla"], f"pliegue_{b['pliegue_inicial']}", pos0,
                                                                              sal / "busqueda" / f"pliegue_{b['pliegue_inicial']}" / f"cfg_{p['id']:02d}", "val")["metricas"]["MASE"])
             for p in configs]
    t = pd.DataFrame(filas).sort_values(["MASE", "id"])
    nfin = ctx.rapido["finalistas"] if ctx.rapido else b["finalistas"]
    fin = t.id.head(nfin).tolist()
    for anio in b["pliegues_restantes"]:
        pos = ctx.recortar(dict(tr=pl[anio]["tr"], es=pl[anio]["es"], ev=pl[anio]["val"]))
        for i in fin:
            p = next(c for c in configs if c["id"] == i)
            filas.append(dict(id=i, pliegue=anio, MASE=corrida_lgbm(ctx, p, b["semilla"], f"pliegue_{anio}", pos, sal / "busqueda" / f"pliegue_{anio}" / f"cfg_{i:02d}", "val")["metricas"]["MASE"]))
    t = pd.DataFrame(filas)
    t.to_csv(sal / "busqueda" / "tabla_busqueda.csv", index=False)
    r = t[t.id.isin(fin)].groupby("id").MASE.mean().sort_values()
    p = next(c for c in configs if c["id"] == int(r.index[0]))
    guardar_json(sal / "busqueda" / "configuracion_elegida.json", dict(configuracion=p, MASE_medio_validacion=float(r.iloc[0])))
    tr, es = ctx.datos.ajuste_final()
    pos = ctx.recortar(dict(tr=tr, es=es, ev=ctx.datos.prueba()))
    semillas = ctx.rapido["semillas_finales"] if ctx.rapido else ctx.cfg["final"]["semillas"]
    res = [corrida_lgbm(ctx, p, s, "final", pos, sal / "final" / f"semilla_{s}", "prueba", guardar=True) for s in semillas]
    out = dict(configuracion=p, MASE_por_semilla=[r["metricas"]["MASE"] for r in res], MASE_media=float(np.mean([r["metricas"]["MASE"] for r in res])))
    guardar_json(sal / "final" / "resumen_prueba.json", out)
    return out


# ------------------------------------------------------------------------------------ MSTL
def _mstl_lote(args):
    serie, h, modelo, semanas = args
    from statsforecast.models import MSTL, AutoARIMA, AutoETS
    tendencia = AutoETS(model="ZZN") if modelo == "ets" else AutoARIMA()
    out = np.empty((len(serie), h))
    for i, s in enumerate(serie):
        s = pd.Series(s).interpolate(limit_direction="both").to_numpy()
        m = MSTL(season_length=[24, 168], trend_forecaster=tendencia)
        out[i] = m.forecast(y=s, h=h)["mean"]
    return out


def _pronosticar(datos, pos: np.ndarray, modelo: str, semanas: int, hilos: int) -> np.ndarray:
    W = semanas * 168
    rz = int(getattr(datos, "rezago_demanda", 1))          # 1 en el SIN (x = y con una hora de rezago); 0 en los adicionales
    series = [datos.X[p - W:p, 0] for p in pos]            # demanda disponible al origen
    lotes = [series[i:i + 25] for i in range(0, len(series), 25)]
    h = datos.H + rz
    if hilos > 1:
        with ProcessPoolExecutor(hilos) as ex:
            partes = list(ex.map(_mstl_lote, [(l, h, modelo, semanas) for l in lotes]))
    else:
        partes = [_mstl_lote((l, h, modelo, semanas)) for l in lotes]
    return np.concatenate(partes)[:, rz:]                    # se descarta la hora previa al origen si aún no está publicada


def mstl(ctx, plan: dict) -> dict:
    d, sal = ctx.datos, ctx.salida
    semanas_l = plan["ventanas_semanas"]
    pl = {f["anio"]: f for f in d.pliegues()}
    resultados = {}
    for modelo in plan["modelos"]:
        filas = []
        for W in semanas_l:
            for anio in plan["pliegues_validacion"]:
                c = sal / modelo / "validacion" / f"W{W}_{anio}.npz"
                pos = pl[anio]["val"]
                if ctx.rapido:
                    pos = pos[:ctx.rapido["n_ev"]]
                if not c.exists():
                    c.parent.mkdir(parents=True, exist_ok=True)
                    t0 = time.time()
                    yhat = _pronosticar(d, pos, modelo, W, ctx.hilos)
                    y = d.Y[pos[:, None] + np.arange(d.H)]
                    np.savez_compressed(c, yhat_kwh=yhat, y_kwh=y, segundos=time.time() - t0)
                z = np.load(c)
                filas.append(dict(modelo=modelo, semanas=W, pliegue=anio, MASE=MET.evaluar(z["y_kwh"], z["yhat_kwh"], d.q168)["MASE"]))
                if ctx.barra is not None:
                    ctx.barra.update(1)
        t = pd.DataFrame(filas)
        t.to_csv(sal / modelo / "validacion.csv", index=False)
        W = int(t.groupby("semanas").MASE.mean().idxmin())
        cp = sal / modelo / f"prueba_W{W}.npz"
        pos = d.prueba()
        if ctx.rapido:
            pos = pos[:ctx.rapido["n_ev"]]
        if not cp.exists():
            t0 = time.time()
            yhat = _pronosticar(d, pos, modelo, W, ctx.hilos)
            y = d.Y[pos[:, None] + np.arange(d.H)]
            seg = time.time() - t0
            np.savez_compressed(cp, origen=np.asarray(d.indice[pos].strftime("%Y-%m-%d"), dtype="U10"), yhat_kwh=yhat.astype("float32"), y_kwh=y.astype("float32"),
                                mase_por_origen=d.mase_por_origen(y, yhat), segundos=seg)
        z = np.load(cp)
        resultados[modelo] = dict(semanas_elegidas=W, MASE_prueba=MET.evaluar(z["y_kwh"], z["yhat_kwh"], d.q168)["MASE"],
                                  latencia_ms_por_origen=1000 * float(z["segundos"]) / len(z["y_kwh"]))
        if ctx.barra is not None:
            ctx.barra.update(1)
    guardar_json(sal / "resumen_mstl.json", resultados)
    return resultados

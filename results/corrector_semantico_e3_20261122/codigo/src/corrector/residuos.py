"""Etapas residuos y cr: pronósticos fuera de muestra de E año por año y corrección diaria C+R temporal.

residuos  Para cada año Y del plan (2011-2021) y cada semilla, E (configuración del plan) se entrena con los orígenes
          cuyo objetivo termina antes del 1-ene de Y (el último 10 % para la parada interna, con purga de 24 h) y
          pronostica Y. Es el mismo forward chaining de src/demanda/particion.py, ampliado a años anteriores a 2018
          (todos dentro de la partición de entrenamiento y validación; la prueba 2022-2025 no se toca).
cr        Corrección diaria C+R: ridge del residuo relativo medio del día sobre el calendario del día objetivo y los
          residuos ya observables (r_{d-2} y media de r_{d-8..d-2}). Para cada año Y se ajusta solo con años anteriores
          (alfa elegido con el último de ellos como validación); el pronóstico corregido es ŷ (1 + r̂).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from demanda import particion as P
from entrenamiento import protocolo as PR
from entrenamiento.protocolo import guardar_json

TZ = "America/Bogota"
COLS_DIA = ["cal_es_festivo", "cal_es_no_laboral", "cal_es_vispera_festivo", "cal_es_dia_post_festivo", "cal_es_fin_semana_largo",
            "cal_es_semana_santa", "cal_es_temporada_navidena", "cal_es_fin_de_anio", "cal_dias_al_siguiente_festivo", "cal_dias_desde_festivo_previo"]


def calendario_diario(datos) -> pd.DataFrame:
    """Calendario determinista del día objetivo (fila de las 12:00), con día de la semana y mes en indicadores."""
    panel = pd.read_parquet(Path(datos.raiz) / "data" / "processed" / "panel_horario.parquet",
                            columns=COLS_DIA + ["cal_dow", "cal_mes"])
    idx = panel.index.tz_convert(TZ) if panel.index.tz is not None else panel.index.tz_localize(TZ)
    cal = panel[idx.hour == 12].copy()
    cal.index = idx[idx.hour == 12].tz_localize(None).normalize()
    C = pd.concat([cal[COLS_DIA].astype(float), pd.get_dummies(cal.cal_dow.astype(int), prefix="dow", dtype=float),
                   pd.get_dummies(cal.cal_mes.astype(int), prefix="mes", dtype=float)], axis=1)
    for c in ("cal_dias_al_siguiente_festivo", "cal_dias_desde_festivo_previo"):
        C[c] = np.log1p(C[c].clip(0, 60))
    return C


def pliegue_anual(datos, anio: int) -> dict:
    v = datos.contrato["validacion"]
    f = P.folds_forward_chaining(datos.indice, [anio], datos.H, datos.L, 24, 0, v["fraccion_parada"])[0]
    return dict(tr=datos.elegibles(f["tr"]), es=datos.elegibles(f["es"]), ev=datos.elegibles(f["val"]))


def etapa_residuos(raiz: Path, salida: Path, plan: dict, datos, dispositivo: str, log, rapido: dict | None, args, barra=None) -> pd.DataFrame:
    b = plan["base"]
    anios = plan["residuos"]["anios"] if not rapido else rapido["anios_residuos"]
    semillas = plan["semillas"] if not rapido else rapido["semillas"]
    cfg = dict(version=plan["version"], paquete=dict(clave="E_anual", nombre="E", red=b["red"], carpeta_resultados="residuos"),
               variantes=["N"], busqueda=plan["comun"]["busqueda"], final=dict(semillas=[]), entrenamiento=dict(plan["comun"]["entrenamiento"]), inferencia={})
    filas = []
    for anio in anios:
        ctx = PR.Contexto(raiz, salida / "residuos", cfg, datos, dispositivo, log, rapido)
        ctx.hilos, ctx.mlflow, ctx.barra = args.hilos, False, barra
        if rapido:
            ctx.opciones["lote_evaluacion"] = 16
        pos = ctx.recortar(pliegue_anual(datos, anio))
        for s in semillas:
            car = salida / "residuos" / f"anio_{anio}" / f"semilla_{s}"
            r = PR.corrida(ctx, "N", dict(b["config"]), s, f"anio_{anio}", pos, car, f"val_{anio}")
            filas.append(dict(anio=anio, semilla=s, MASE=r["metricas"]["MASE"], epocas=r["epocas"], segundos=r["segundos"]))
            log(f"Residuos E {anio} semilla {s}: MASE {r['metricas']['MASE']:.4f}", consola=False)
    t = pd.DataFrame(filas)
    t.to_csv(salida / "residuos" / "resumen.csv", index=False)
    return t


def leer_residuos(salida: Path, anios: list[int], semilla: int) -> pd.DataFrame:
    partes = []
    for a in anios:
        z = np.load(salida / "residuos" / f"anio_{a}" / f"semilla_{semilla}" / f"pred_val_{a}.npz")
        partes.append(pd.DataFrame(dict(dia=pd.to_datetime(z["origen"]), anio=a, y=list(z["y_kwh"].astype("float64")),
                                        yhat=list(z["yhat_kwh"].astype("float64")))))
    t = pd.concat(partes, ignore_index=True).sort_values("dia").reset_index(drop=True)
    if t.dia.duplicated().any():
        raise RuntimeError("Orígenes repetidos entre años")
    return t


def _rasgos_cr(t: pd.DataFrame, C: pd.DataFrame, r: np.ndarray) -> np.ndarray:
    """Calendario del día objetivo + r_{d-2} y media de r_{d-8..d-2}, con días faltantes alineados por fecha."""
    s = pd.Series(r, index=pd.DatetimeIndex(t.dia)).asfreq("D")
    lag2 = s.shift(2)
    m7 = s.shift(2).rolling(7, min_periods=1).mean()
    R = pd.DataFrame(dict(r_lag2=lag2, r_m7=m7)).reindex(pd.DatetimeIndex(t.dia)).fillna(0.0).to_numpy()
    return np.hstack([C.loc[pd.DatetimeIndex(t.dia)].to_numpy(dtype=float), R])


def etapa_cr(salida: Path, plan: dict, datos, log, rapido: dict | None) -> dict:
    anios = plan["residuos"]["anios"] if not rapido else rapido["anios_residuos"]
    semillas = plan["semillas"] if not rapido else rapido["semillas"]
    C = calendario_diario(datos)
    alfas = np.logspace(-1, 5, 25)
    info = {}
    (salida / "cr").mkdir(parents=True, exist_ok=True)
    for s in semillas:
        t = leer_residuos(salida, anios, s)
        Y, Yh = np.stack(t.y.to_numpy()), np.stack(t.yhat.to_numpy())
        r = ((Y - Yh) / Yh).mean(1)
        X = _rasgos_cr(t, C, r)
        rhat = np.full(len(t), np.nan)
        elegidos = {}
        for a in anios[1:]:
            pasado, ahora = (t.anio < a).to_numpy(), (t.anio == a).to_numpy()
            previos = sorted(t.anio[pasado].unique())
            if len(previos) >= 2:
                ajuste, valid = (t.anio < previos[-1]).to_numpy(), (t.anio == previos[-1]).to_numpy()
                sc = StandardScaler().fit(X[ajuste])
                err = [np.mean((Ridge(alpha=al).fit(sc.transform(X[ajuste]), r[ajuste]).predict(sc.transform(X[valid])) - r[valid]) ** 2) for al in alfas]
                alfa = float(alfas[int(np.argmin(err))])
            else:
                alfa = float(plan["cr"]["alfa_sin_historial"])
            sc = StandardScaler().fit(X[pasado])
            rhat[ahora] = Ridge(alpha=alfa).fit(sc.transform(X[pasado]), r[pasado]).predict(sc.transform(X[ahora]))
            elegidos[str(a)] = alfa
        ok = ~np.isnan(rhat)
        ycr = Yh[ok] * (1 + rhat[ok, None])
        dias = t.dia[ok].dt.strftime("%Y-%m-%d").to_numpy()
        np.savez_compressed(salida / "cr" / f"semilla_{s}.npz", dia=dias.astype("U10"), anio=t.anio[ok].to_numpy(), y=Y[ok].astype("float32"),
                            yhat_e=Yh[ok].astype("float32"), yhat_cr=ycr.astype("float32"), rhat=rhat[ok].astype("float32"))
        por_anio = {}
        for a in anios[1:]:
            m = (t.anio[ok] == a).to_numpy()
            b0, b1 = np.abs(Y[ok][m] - Yh[ok][m]).mean() / datos.q168, np.abs(Y[ok][m] - ycr[m]).mean() / datos.q168
            por_anio[str(a)] = dict(MASE_E=float(b0), MASE_CR=float(b1), reduccion_pct=float(100 * (b0 - b1) / b0))
        info[str(s)] = dict(alfas=elegidos, por_anio=por_anio)
        log(f"C+R semilla {s}: " + ", ".join(f"{a} {v['reduccion_pct']:+.2f} %" for a, v in por_anio.items()), consola=False)
    guardar_json(salida / "cr" / "resumen.json", info)
    return info

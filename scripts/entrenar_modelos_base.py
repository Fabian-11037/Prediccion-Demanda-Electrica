"""Entrena los tres modelos base en PyTorch (ARX lineal ridge/lasso, MLP, LSTM) con el mismo pipeline.

Usa GPU si está disponible (--dispositivo auto). LightGBM es opcional (--modelos lightgbm) y corre en CPU.

Etapas (en este orden, cada una con su subcomando):

  listar    muestra los espacios de búsqueda y las opciones
  buscar    búsqueda aleatoria, misma estrategia y mismo presupuesto (--n-config) para los tres
            modelos, 1 semilla. Cada configuración se evalúa con VALIDACIÓN CRUZADA FORWARD
            CHAINING (ventana expansiva; un pliegue por año de --cv-anios; purga = horizonte;
            parada temprana con los últimos --frac-es del entrenamiento de cada pliegue).
            Criterio de selección: MASE medio entre pliegues.
  piloto    mejor configuración de cada modelo x semillas del piloto (mismos pliegues); estima
            el número de semillas del benchmark (potencia, delta relativo)
  final     mejor configuración x n semillas: reajusta con todo lo anterior a la prueba (parada
            temprana con el último tramo) y evalúa en PRUEBA. Exige --allow-test y que exista el
            preregistro. Úsala una sola vez.

Ejemplos:
  python scripts/entrenar_modelos_base.py listar
  python scripts/entrenar_modelos_base.py buscar --n-config 20 --semillas-por-config 3 --salida results/base_v2
  python scripts/entrenar_modelos_base.py buscar --modelos lstm --fijar lstm.unidades=64 --max-epocas 40
  python scripts/entrenar_modelos_base.py buscar --dispositivo cuda --amp --determinismo rapido
  python scripts/entrenar_modelos_base.py piloto
  python scripts/entrenar_modelos_base.py final --allow-test --prereg PREREGISTRO.md

Prueba rápida de humo:
  python scripts/entrenar_modelos_base.py buscar --n-config 2 --max-origenes-train 300 --max-epocas 3 --max-arboles 50 --salida /tmp/humo
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from demanda import config as C  # noqa: E402
from demanda import modelos_base as M  # noqa: E402
from demanda.metricas import evaluar, n_semillas_requerido  # noqa: E402

SEMILLAS_PILOTO = [100, 101, 102, 103, 104]
SEMILLA_FINAL_BASE = 1000


# ------------------------------------------------------------------------ utilidades
def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("etapa", choices=["listar", "buscar", "piloto", "final"])
    g = ap.add_argument_group("datos y partición")
    g.add_argument("--covariables", choices=M.COVARIABLES, default="calendario",
                   help="serie: solo demanda | calendario: + festivos/calendario | completo: + precio, hidrología y clima ponderado")
    g.add_argument("--clima-por-ciudad", action="store_true", help="con 'completo': 24 columnas de clima en vez del clima ponderado por habitantes")
    g.add_argument("--ventana", type=int, default=C.VENTANA, help="horas de historia L (por defecto %(default)s)")
    g.add_argument("--horizonte", type=int, default=C.HORIZONTE, help="intervalos pronosticados H (por defecto %(default)s)")
    g.add_argument("--paso-h", type=int, default=24, help="separación entre orígenes en horas; divisor o múltiplo de 24 (por defecto %(default)s)")
    g.add_argument("--hora-origen", type=int, default=0, help="hora del día del origen (por defecto %(default)s)")
    g.add_argument("--normalizacion", choices=M.NORMALIZACIONES, default="nivel",
                   help="nivel: demanda dividida por su media en la ventana | global: estandarizada con entrenamiento")
    g.add_argument("--escala-mase-lag", type=int, choices=[24, 168], default=168, help="rezago del ingenuo para la escala de MASE")
    g.add_argument("--max-origenes-train", type=int, default=None, help="(sin efecto en validación cruzada) submuestreo de la partición fija de entrenamiento")
    g.add_argument("--cv-anios", type=int, nargs="+", default=C.CV_ANIOS,
                   help="años de validación del forward chaining (entre 2018 y 2021); por defecto %(default)s")
    g.add_argument("--frac-es", type=float, default=C.FRACCION_PARADA_TEMPRANA,
                   help="fracción final (cronológica) del entrenamiento reservada para la parada temprana")

    b = ap.add_argument_group("búsqueda y modelos")
    b.add_argument("--modelos", nargs="+", choices=list(M.CLASES), default=list(M.MODELOS_PYTORCH))
    b.add_argument("--n-config", type=int, default=20, help="configuraciones por modelo (mismo presupuesto para todos; 20 desde la v2 por el espacio ampliado)")
    b.add_argument("--semillas-por-config", type=int, default=3, help="semillas por configuración en la búsqueda (se promedian; 3 desde la v2, 1 en la v1)")
    b.add_argument("--semilla-busqueda", type=int, default=1, help="semilla del muestreo de configuraciones y del entrenamiento en la búsqueda")
    b.add_argument("--fijar", action="append", default=[], metavar="modelo.param=valor",
                   help="fija un hiperparámetro (sale de la búsqueda); repetible. Ej.: lstm.unidades=64")
    b.add_argument("--espacio", type=Path, default=None, help="JSON que reemplaza espacios de búsqueda: {modelo: {param: spec}}")

    t = ap.add_argument_group("entrenamiento (no se buscan)")
    t.add_argument("--max-epocas", type=int, default=200, help="modelos PyTorch: épocas máximas (la parada temprana con --paciencia decide cuándo termina)")
    t.add_argument("--paciencia", type=int, default=8, help="épocas (PyTorch, si no se busca) o rondas (LightGBM) sin mejora en validación; en la v2 la paciencia de los modelos PyTorch es hiperparámetro")
    t.add_argument("--max-arboles", type=int, default=1500, help="LightGBM: árboles máximos (parada temprana)")
    t.add_argument("--lr", type=float, default=1e-3, help="tasa de aprendizaje por defecto si no se busca (en la v2 es hiperparámetro)")
    t.add_argument("--lr-arboles", type=float, default=0.05, help="LightGBM: tasa de aprendizaje")
    t.add_argument("--perdida", choices=M.PERDIDAS, default="mse", help="pérdida por defecto si no se busca (en la v2 es hiperparámetro: mse/mae/huber)")
    t.add_argument("--huber-delta", type=float, default=None, help="delta de Huber en unidades normalizadas (por defecto 0.05 con normalización nivel, 0.5 con global)")
    t.add_argument("--min-delta", type=float, default=1e-6, help="mejora mínima de la pérdida de validación para reiniciar la paciencia")
    t.add_argument("--clip", type=float, default=1.0, help="recorte de la norma del gradiente")
    t.add_argument("--hilos", type=int, default=1, help="hilos de CPU (1 = determinismo más estricto)")
    d = ap.add_argument_group("dispositivo")
    d.add_argument("--dispositivo", default="auto", help="auto (GPU si hay) | cpu | cuda | cuda:N")
    d.add_argument("--determinismo", choices=["estricto", "rapido"], default="estricto",
                   help="estricto: torch.use_deterministic_algorithms(True) (reproducible, más lento en GPU) | rapido: sin garantías bit a bit")
    d.add_argument("--amp", action="store_true", help="precisión mixta bfloat16 en GPU (más rápido; cambia levemente los resultados)")
    d.add_argument("--max-gb-gpu", type=float, default=2.0, help="si los datos de entrenamiento superan este tamaño se mantienen en CPU y se envían por lotes")

    s = ap.add_argument_group("piloto y final")
    s.add_argument("--piloto-ultimo-pliegue", action="store_true", help="el piloto usa solo el último pliegue (más rápido)")
    s.add_argument("--semillas-piloto", type=int, nargs="+", default=SEMILLAS_PILOTO)
    s.add_argument("--delta", type=float, default=0.03, help="diferencia relativa mínima de interés para la potencia (definir en el preregistro)")
    s.add_argument("--alfa", type=float, default=0.05)
    s.add_argument("--potencia", type=float, default=0.80)
    s.add_argument("--n-semillas", type=int, default=None, help="semillas de la corrida final (por defecto, el valor del piloto; mínimo 5)")
    s.add_argument("--allow-test", action="store_true", help="autoriza leer la PRUEBA (solo etapa final)")
    s.add_argument("--prereg", type=Path, default=C.RAIZ / "PREREGISTRO.md", help="archivo de preregistro requerido para la etapa final")
    s.add_argument("--salida", type=Path, default=C.RAIZ / "results" / "base")
    return ap


def _valor(txt: str):
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        return txt


def espacios(a) -> dict:
    esp = {k: dict(v) for k, v in M.ESPACIOS.items()}
    if a.espacio:
        for mod, d in json.loads(a.espacio.read_text()).items():
            esp[mod] = d
    return esp


def fijos(a) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for f in a.fijar:
        clave, _, val = f.partition("=")
        mod, _, par = clave.partition(".")
        if mod not in M.CLASES or not par or not _:
            sys.exit(f"--fijar inválido: {f} (formato modelo.param=valor)")
        out.setdefault(mod, {})[par] = _valor(val)
    return out


def opciones(a) -> M.Opciones:
    return M.Opciones(a.max_epocas, a.paciencia, a.max_arboles, a.lr, a.min_delta, a.huber_delta, a.lr_arboles, a.perdida, a.clip,
                      a.hilos, a.dispositivo, a.determinismo, a.amp, a.max_gb_gpu)


def cfg_datos(a) -> M.ConfigDatos:
    return M.ConfigDatos(a.covariables, a.ventana, a.horizonte, a.paso_h, a.hora_origen, a.normalizacion,
                         a.clima_por_ciudad, a.max_origenes_train, a.escala_mase_lag)


def cargar_datos(a) -> M.Datos:
    panel = pd.read_parquet(C.PROCESADO / "panel_horario.parquet")
    reg = json.loads((C.PROCESADO / "feature_registry.json").read_text())
    return M.Datos(panel, reg, cfg_datos(a))


def hacer_folds(a, datos: M.Datos):
    from demanda.particion import folds_forward_chaining

    c = datos.cfg
    return folds_forward_chaining(datos.panel.index, a.cv_anios, c.horizonte, c.ventana, c.paso_h, c.hora_origen, a.frac_es)


def registrar_entorno(a, datos: M.Datos, etapa: str):
    a.salida.mkdir(parents=True, exist_ok=True)
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=C.RAIZ).stdout.strip()
    except Exception:
        commit = ""
    import torch
    try:
        import lightgbm
        lgb_v = lightgbm.__version__
    except ImportError:
        lgb_v = None
    dev = M.resolver_dispositivo(a.dispositivo)
    gpu = {"dispositivo": str(dev), "nombre": torch.cuda.get_device_name(dev) if dev.type == "cuda" else None,
           "cuda": torch.version.cuda, "cuda_disponible": torch.cuda.is_available()}
    env = {"etapa": etapa, "fecha": pd.Timestamp.now(tz=C.TZ).isoformat(), "python": sys.version, "plataforma": platform.platform(),
           "cpu": platform.processor(), "nucleos": __import__("os").cpu_count(), "hilos": a.hilos, "gpu": gpu,
           "versiones": {"numpy": np.__version__, "pandas": pd.__version__, "torch": torch.__version__,
                         "lightgbm": lgb_v},
           "commit": commit, "datos": asdict(datos.cfg), "opciones": asdict(opciones(a)), "n_config": a.n_config,
           "semilla_busqueda": a.semilla_busqueda, "semillas_por_config": a.semillas_por_config, "cv": {"tipo": "forward chaining (ventana expansiva)", "anios_validacion": a.cv_anios, "purga_h": datos.cfg.horizonte, "frac_parada_temprana": a.frac_es}, "columnas_x": datos.cols_x, "columnas_cal": datos.cols_cal,
           "escala_mase": datos.escala_mase, "determinismo": a.determinismo,
           "particion": C.PARTICION}
    (a.salida / f"entorno_{etapa}.json").write_text(json.dumps(env, indent=2, ensure_ascii=False, default=str))


def guardar_hist(a, etapa, nombre, semilla, hist):
    d = a.salida / "hist"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{etapa}_{nombre}_{semilla}.json").write_text(json.dumps(hist))


def fila(r: dict) -> dict:
    return {k: (json.dumps(v) if k == "params" else v) for k, v in r.items() if not k.startswith("_")}


# ---------------------------------------------------------------------------- etapas
def etapa_listar(a):
    esp = espacios(a)
    print(json.dumps(esp, indent=1, ensure_ascii=False))
    print("\nOpciones de entrenamiento:", asdict(opciones(a)))
    print("Fijados:", fijos(a))


def etapa_buscar(a):
    datos = cargar_datos(a)
    registrar_entorno(a, datos, "buscar")
    esp, fj, op = espacios(a), fijos(a), opciones(a)
    folds = hacer_folds(a, datos)
    print("Pliegues (forward chaining):", [(f["anio"], len(f["tr"]), len(f["es"]), len(f["val"])) for f in folds], flush=True)
    ruta = a.salida / "busqueda.csv"
    filas = pd.read_csv(ruta).to_dict("records") if ruta.exists() else []
    for nombre in a.modelos:
        rng = np.random.default_rng(a.semilla_busqueda)
        configs = [M.muestrear(esp[nombre], rng, fj.get(nombre)) for _ in range(a.n_config)]
        for i, p in enumerate(configs):
            if any(f["modelo"] == nombre and f["config"] == i and json.loads(f["params"]) == p for f in filas):
                continue
            semillas = [a.semilla_busqueda + k for k in range(a.semillas_por_config)]
            r = M.validacion_cruzada_semillas(nombre, p, semillas, datos, op, folds)
            filas.append({"config": i, **fila(r)})
            guardar_hist(a, "busqueda", f"{nombre}_c{i}", a.semilla_busqueda, {str(k): v for k, v in r["_historia"].items()})
            pd.DataFrame(filas).to_csv(ruta, index=False)
            por_fold = {k: round(v, 3) for k, v in r.items() if k.startswith("MASE_2")}
            print(f"[{nombre} {i + 1}/{a.n_config}] MASE_cv={r['MASE']:.4f} {por_fold} ({r['segundos']}s) {p}", flush=True)
    df = pd.DataFrame(filas)
    mejores = {}
    for nombre in a.modelos:
        d = df[df.modelo == nombre].sort_values("MASE")
        mejores[nombre] = {"params": json.loads(d.iloc[0]["params"]), "MASE_cv": float(d.iloc[0]["MASE"]), "n_config_evaluadas": int(len(d))}
    (a.salida / "mejor_config.json").write_text(json.dumps(mejores, indent=2))
    ref = {}
    for f in folds:
        ev = datos.conjunto_pos(f["val"])
        for k, v in datos.referencias_pos(f["val"], ev).items():
            ref.setdefault(k, []).append(evaluar(ev.y, v, datos.escala_mase)["MASE"])
    ref = {k: {"MASE_cv": float(np.mean(v)), "por_pliegue": dict(zip([f["anio"] for f in folds], map(float, v)))} for k, v in ref.items()}
    (a.salida / "referencias_cv.json").write_text(json.dumps(ref, indent=2))
    print("Mejores:", json.dumps(mejores, indent=1))
    print("Referencias (MASE medio en los pliegues):", {k: round(v["MASE_cv"], 4) for k, v in ref.items()})


def _mejores(a) -> dict:
    p = a.salida / "mejor_config.json"
    if not p.exists():
        sys.exit("Falta mejor_config.json: ejecutar primero la etapa 'buscar'")
    return json.loads(p.read_text())


def etapa_piloto(a):
    datos = cargar_datos(a)
    registrar_entorno(a, datos, "piloto")
    mej, op = _mejores(a), opciones(a)
    folds = hacer_folds(a, datos)
    if a.piloto_ultimo_pliegue:
        folds = folds[-1:]
    filas = []
    for nombre in a.modelos:
        for s in a.semillas_piloto:
            r = M.validacion_cruzada(nombre, mej[nombre]["params"], s, datos, op, folds)
            filas.append({"etapa": "piloto", **fila(r)})
            guardar_hist(a, "piloto", nombre, s, {str(k): v for k, v in r["_historia"].items()})
            print(f"[piloto {nombre} semilla={s}] MASE_cv={r['MASE']:.4f}", flush=True)
            pd.DataFrame(filas).to_csv(a.salida / "piloto.csv", index=False)
    df = pd.DataFrame(filas)
    res = {}
    for nombre in a.modelos:
        v = df[df.modelo == nombre]["MASE"].to_numpy()
        res[nombre] = {"media": float(v.mean()), "sd": float(v.std(ddof=1)), "n_piloto": int(len(v)),
                       "n_requerido": n_semillas_requerido(float(v.std(ddof=1)), float(v.mean()), a.delta, a.alfa, a.potencia),
                       "determinista": False}
    n_final = max(5, max(r["n_requerido"] for r in res.values()))
    out = {"delta_rel": a.delta, "alfa": a.alfa, "potencia": a.potencia, "n_final": n_final, "pliegues": [f["anio"] for f in folds],
           "por_modelo": res, "nota": "Estimación del piloto (normal, dos muestras); no es definitiva. Mínimo 5; 6 si se usan pruebas por rangos."}
    (a.salida / "piloto_resumen.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))


def etapa_final(a):
    from demanda.particion import pretest

    if not a.allow_test:
        sys.exit("La etapa final lee la PRUEBA: requiere --allow-test (úsala una sola vez, tras el preregistro)")
    if not a.prereg.exists():
        sys.exit(f"No existe el preregistro {a.prereg}: escribirlo y versionarlo antes de tocar la prueba")
    datos = cargar_datos(a)
    registrar_entorno(a, datos, "final")
    mej, op = _mejores(a), opciones(a)
    n = a.n_semillas
    if n is None:
        p = a.salida / "piloto_resumen.json"
        n = json.loads(p.read_text())["n_final"] if p.exists() else 5
    n = max(n, 5)
    c = datos.cfg
    pos_tr, pos_es = pretest(datos.panel.index, c.horizonte, c.ventana, c.paso_h, c.hora_origen, a.frac_es)
    tr, es, te = datos.conjunto_pos(pos_tr), datos.conjunto_pos(pos_es), datos.conjunto("test", True)
    print(f"Ajuste final: {len(tr)} orígenes de entrenamiento, {len(es)} de parada temprana, {len(te)} de prueba; n semillas = {n}", flush=True)
    filas, preds = [], {}
    for nombre in a.modelos:
        for i in range(n):
            s = SEMILLA_FINAL_BASE + i
            r = M.entrenar_y_evaluar(nombre, mej[nombre]["params"], s, datos, op, tr, es, te, guardar_pred=True)
            filas.append({"etapa": "final", **fila(r)})
            preds[f"{nombre}_{s}"] = r["_yhat"].astype("float32")
            guardar_hist(a, "final", nombre, s, r["_historia"])
            print(f"[final {nombre} semilla={s}] MASE_test={r['MASE']:.4f}", flush=True)
            pd.DataFrame(filas).to_csv(a.salida / "final_runs.csv", index=False)
    refs = {k: evaluar(te.y, v, datos.escala_mase) for k, v in datos.referencias("test", True).items()}
    (a.salida / "referencias_test.json").write_text(json.dumps(refs, indent=2))
    pdir = a.salida / "preds"
    pdir.mkdir(exist_ok=True)
    np.savez_compressed(pdir / "test.npz", y=te.y, origen=te.origen.astype("int64"), **preds)
    from scipy import stats

    df = pd.DataFrame(filas)
    resumen = []
    for nombre in a.modelos:
        d = df[df.modelo == nombre]
        v = d["MASE"].to_numpy()
        h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else float("nan")
        fila_r = {"modelo": nombre, "n": len(v), "MASE_media": v.mean(), "MASE_sd": v.std(ddof=1) if len(v) > 1 else 0.0,
                  "IC95_inf": v.mean() - h, "IC95_sup": v.mean() + h}
        for k in ("RMSE", "MAE", "sMAPE", "MAPE", "MASE_h1", "MASE_h12", "MASE_h24"):
            fila_r[k + "_media"] = d[k].mean()
        resumen.append(fila_r)
    for k, v in refs.items():
        resumen.append({"modelo": k, "n": 1, "MASE_media": v["MASE"], **{kk + "_media": v[kk] for kk in ("RMSE", "MAE", "sMAPE", "MAPE", "MASE_h1", "MASE_h12", "MASE_h24")}})
    pd.DataFrame(resumen).to_csv(a.salida / "resumen_final.csv", index=False)
    print(pd.DataFrame(resumen).to_string(index=False))


if __name__ == "__main__":
    args = parser().parse_args()
    t0 = time.time()
    {"listar": etapa_listar, "buscar": etapa_buscar, "piloto": etapa_piloto, "final": etapa_final}[args.etapa](args)
    print(f"Listo en {time.time() - t0:.0f}s")

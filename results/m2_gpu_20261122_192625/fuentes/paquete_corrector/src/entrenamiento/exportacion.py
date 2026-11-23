"""Bloque B del Entregable 3: exportación a ONNX, prueba de ida y vuelta, metadatos, cuantización y latencia en CPU.

Para cada procedimiento de la familia (y el original) se exporta la primera semilla final (1000), la misma política
para todos. Prueba de ida y vuelta: sobre 64 orígenes fijos de la prueba, error absoluto máximo de la salida normalizada
menor que la tolerancia declarada (1e-4) y MASE del modelo exportado en toda la prueba frente al modelo en memoria. Si el
archivo supera el límite por archivo (95 MB), se cuantiza a INT8 dinámico con onnxruntime y se contrasta la pérdida con
TOST por bootstrap de bloques (margen δ = 3 %). Si la exportación falla o no reproduce las salidas (por ejemplo,
TimesNet fija sus periodos al trazar), el modelo se marca como «precalculado» y se documenta el motivo.
"""
from __future__ import annotations

import hashlib
import json
import platform
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from demanda import metricas as MET

from . import motor
from . import protocolo_e2 as P2
from .protocolo import guardar_json, leer_json


class _Envoltura(torch.nn.Module):
    def __init__(self, red):
        super().__init__()
        self.red = red

    def forward(self, x, cal_p, cal_f, dia):
        return self.red(x, cal_p, cal_f, dia)


def _sha(ruta: Path) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _versiones() -> dict:
    out = dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__)
    for m in ("onnx", "onnxruntime", "lightgbm", "statsforecast", "pandas"):
        try:
            out[m] = __import__(m).__version__
        except Exception:
            out[m] = None
    return out


def _latencia(fn, reps: int) -> dict:
    fn()
    t = []
    for _ in range(reps):
        a = time.perf_counter()
        fn()
        t.append(1000 * (time.perf_counter() - a))
    return dict(mediana_ms=float(np.median(t)), p95_ms=float(np.percentile(t, 95)), repeticiones=reps)


def _tost_bloques(d: np.ndarray, base: np.ndarray, margen: float, bloque: int = 14, b: int = 2000, semilla: int = 20261118) -> dict:
    g = np.random.default_rng(semilla)
    n = len(d)
    nb = int(np.ceil(n / bloque))
    ix = ((g.integers(0, n, size=(b, nb))[:, :, None] + np.arange(bloque)[None, None, :]) % n).reshape(b, -1)[:, :n]
    rel = 100 * d[ix].mean(1) / base[ix].mean(1)
    lo, hi = np.percentile(rel, [5, 95])
    return dict(ic90_diferencia_relativa_pct=[float(lo), float(hi)], equivalente=bool(lo > -100 * margen and hi < 100 * margen), margen_pct=100 * margen)


def exportar(e3, carpeta_artefactos: Path) -> list[dict]:
    """``e3``: instancia de ``protocolo_e3.E3`` con la etapa familia terminada."""
    import onnxruntime as ort
    pe = e3.plan["exportacion"]
    datos = e3.datos
    pos = datos.prueba()
    if e3.rapido:
        pos = pos[:e3.rapido["n_ev"]]
    ida = pos[:pe["origenes_ida_vuelta"]]
    carpeta_artefactos = Path(carpeta_artefactos)
    filas = []
    for m in e3.miembros():
        dst = carpeta_artefactos / m["clave"]
        dst.mkdir(parents=True, exist_ok=True)
        base = e3.salida / "familia" / m["clave"]
        meta = dict(clave=m["clave"], nombre=m.get("nombre"), grupo=m.get("grupo"), versiones=_versiones(), ventana_h=datos.L, horizonte_h=datos.H,
                    q168_kwh=datos.q168, panel_sha256=datos.hash_panel)
        if m["tipo"] == "mstl":
            r = leer_json(base / "determinista" / "resultado.json")
            meta.update(formato="precalculado", motivo="MSTL se reajusta en cada origen (no tiene pesos que exportar); se publican sus pronósticos de prueba",
                        configuracion=r["configuracion"], metricas_prueba=r["metricas"])
            shutil.copy(base / "determinista" / "pred_prueba.npz", dst / "pronosticos_prueba.npz")
            guardar_json(dst / "metadata.json", meta)
            filas.append(dict(clave=m["clave"], formato="precalculado", MASE_prueba=r["metricas"]["MASE"]))
            continue
        carpetas = sorted(base.glob("semilla_*"), key=lambda p: int(p.name.split("_")[1]))
        if not carpetas:
            continue
        car = carpetas[0]
        r = leer_json(car / "resultado.json")
        meta.update(semilla=r["semilla"], configuracion=r["configuracion"], variante=r["variante"], metricas_prueba=r["metricas"],
                    MASE_ajuste=r.get("MASE_ajuste"), MASE_parada=r.get("MASE_parada"), presupuesto=r.get("presupuesto"))
        if m["tipo"] == "lightgbm":
            shutil.copy(car / "modelo_lightgbm.txt", dst / "modelo_lightgbm.txt")
            meta.update(formato="lightgbm_txt", motivo="Árboles: se cargan con lightgbm.Booster en CPU; no se convierte a ONNX",
                        sha256_modelo=_sha(dst / "modelo_lightgbm.txt"))
            guardar_json(dst / "metadata.json", meta)
            filas.append(dict(clave=m["clave"], formato="lightgbm_txt", MASE_prueba=r["metricas"]["MASE"]))
            continue
        esc_js = leer_json(car / "escaladores.json")
        esc = P2._escalas(esc_js, datos)
        conj = datos.construir(pos, esc)
        red, _ = e3.cargar_red(m, car, conj)
        red = red.cpu().eval()
        meta.update(preprocesamiento=esc_js, esquema_entrada=dict(
            x=dict(forma=["n", datos.L, conj.x.shape[2]], tipo="float32", canales=datos.hist + datos.nombres_mascara,
                   rango_entrenamiento=[float(np.percentile(conj.x, 0.1)), float(np.percentile(conj.x, 99.9))]),
            cal_p=dict(forma=["n", datos.L, conj.cal_p.shape[2]], tipo="float32", canales=datos.cal), cal_f=dict(forma=["n", datos.H, conj.cal_f.shape[2]], tipo="float32"),
            dia=dict(forma=["n"], tipo="int64", descripcion="días desde " + str(datos.dia_base.date()))),
            salida=dict(nombre="o", forma=["n", datos.H], descripcion="demanda / nivel de la ventana; kWh = o * media de la demanda de las 168 h"))
        if m["variante"] == "N+O" and e3.texto is not None:
            meta["texto"] = dict(codificador=datos.contrato["texto"]["encoder"], revision=datos.contrato["texto"]["revision"],
                                 sha256_vectores=hashlib.sha256(np.ascontiguousarray(e3.texto["vectores"]).tobytes()).hexdigest(),
                                 sha256_listas=hashlib.sha256(np.ascontiguousarray(e3.texto["lista"]).tobytes()).hexdigest())
        t = motor.a_tensores(conj, "cpu")
        args = tuple(t[k][:2] for k in ("x", "cal_p", "cal_f", "dia"))
        ruta = dst / "modelo.onnx"
        estado = dict(clave=m["clave"])
        with torch.no_grad():
            ref = torch.cat([red(t["x"][i:i + 256], t["cal_p"][i:i + 256], t["cal_f"][i:i + 256], t["dia"][i:i + 256]) for i in range(0, len(t["y"]), 256)]).numpy()
        try:
            torch.onnx.export(_Envoltura(red).eval(), args, str(ruta), opset_version=pe["opset"], input_names=["x", "cal_p", "cal_f", "dia"], output_names=["o"],
                              dynamic_axes={k: {0: "n"} for k in ("x", "cal_p", "cal_f", "dia", "o")}, dynamo=False)
            so = ort.SessionOptions()
            so.intra_op_num_threads = pe["hilos_latencia"]
            ses = ort.InferenceSession(str(ruta), so, providers=["CPUExecutionProvider"])
            nombres = [i.name for i in ses.get_inputs()]              # el exportador omite las entradas que la red no usa
            feed = lambda ix: {k: t[k][ix].numpy() for k in nombres}   # noqa: E731
            n_ida = len(ida)
            out_ida = ses.run(None, feed(slice(0, n_ida)))[0]
            err = float(np.abs(out_ida - ref[:n_ida]).max())
            full = np.concatenate([ses.run(None, feed(slice(i, i + 256)))[0] for i in range(0, len(ref), 256)])
            mase_onnx = MET.evaluar(conj.y, full * conj.nivel[:, None], datos.q168)["MASE"]
            mase_mem = MET.evaluar(conj.y, ref * conj.nivel[:, None], datos.q168)["MASE"]
            estado.update(formato="onnx", error_absoluto_maximo=err, tolerancia=pe["tolerancia_absoluta"], ida_y_vuelta_correcta=bool(err < pe["tolerancia_absoluta"]),
                          MASE_memoria=mase_mem, MASE_onnx=mase_onnx, diferencia_MASE=mase_onnx - mase_mem, tamano_mb=ruta.stat().st_size / 2**20)
            uno = feed(slice(0, 1))
            estado["latencia_cpu_onnx"] = _latencia(lambda: ses.run(None, uno), 20 if e3.rapido else pe["repeticiones_latencia"])
            if estado["tamano_mb"] > pe["limite_mb"]:
                from onnxruntime.quantization import QuantType, quantize_dynamic
                q = dst / "modelo_int8.onnx"
                quantize_dynamic(str(ruta), str(q), weight_type=QuantType.QInt8)
                sq = ort.InferenceSession(str(q), so, providers=["CPUExecutionProvider"])
                fq = np.concatenate([sq.run(None, feed(slice(i, i + 256)))[0] for i in range(0, len(ref), 256)])
                lq, lm = datos.mase_por_origen(conj.y, fq * conj.nivel[:, None]), datos.mase_por_origen(conj.y, ref * conj.nivel[:, None])
                estado["cuantizacion"] = dict(tipo="INT8 dinámico (onnxruntime.quantization)", tamano_mb=q.stat().st_size / 2**20, MASE=float(lq.mean()),
                                              **_tost_bloques(lq - lm, lm, pe["margen_tost"]))
                if not estado["ida_y_vuelta_correcta"] or not estado["cuantizacion"]["equivalente"] or estado["cuantizacion"]["tamano_mb"] > pe["limite_mb"]:
                    estado["formato"] = "precalculado"
                    estado["motivo"] = "el modelo no cabe bajo el límite sin perder más que δ; se publican sus pronósticos y el archivo va fuera de GitHub (almacenamiento externo o Git LFS)"
                ruta.unlink()
            if not estado["ida_y_vuelta_correcta"]:
                estado["formato"] = "precalculado"
                estado["motivo"] = "el modelo exportado no reproduce las salidas dentro de la tolerancia (operación dependiente de los datos fijada al trazar)"
        except Exception as e:
            estado.update(formato="precalculado", motivo=f"falló la exportación a ONNX: {type(e).__name__}: {str(e)[:200]}")
        if estado["formato"] == "precalculado":
            np.savez_compressed(dst / "pronosticos_prueba.npz", origen=np.asarray(conj.origen.strftime("%Y-%m-%d"), dtype="U10"), yhat_kwh=(ref * conj.nivel[:, None]).astype("float32"))
        torch.save({k: v.cpu() for k, v in red.state_dict().items()}, dst / "modelo_pesos.pt")
        if (car / "optimizador_y_configuracion.pt").exists():
            shutil.copy(car / "optimizador_y_configuracion.pt", dst / "optimizador_y_configuracion.pt")
        uno_t = tuple(t[k][:1] for k in ("x", "cal_p", "cal_f", "dia"))
        torch.set_num_threads(pe["hilos_latencia"])
        with torch.no_grad():
            estado["latencia_cpu_torch"] = _latencia(lambda: red(*uno_t), 20 if e3.rapido else pe["repeticiones_latencia"])
        meta.update(exportacion=estado, n_parametros=red.n_parametros())
        guardar_json(dst / "metadata.json", meta)
        guardar_json(dst / "ida_y_vuelta.json", estado)
        filas.append(dict(clave=m["clave"], formato=estado["formato"], ida_y_vuelta=estado.get("ida_y_vuelta_correcta"), error_maximo=estado.get("error_absoluto_maximo"),
                          MASE_memoria=estado.get("MASE_memoria"), MASE_onnx=estado.get("MASE_onnx"), tamano_mb=estado.get("tamano_mb"),
                          latencia_onnx_mediana_ms=(estado.get("latencia_cpu_onnx") or {}).get("mediana_ms"), latencia_onnx_p95_ms=(estado.get("latencia_cpu_onnx") or {}).get("p95_ms"),
                          latencia_torch_mediana_ms=estado["latencia_cpu_torch"]["mediana_ms"], latencia_torch_p95_ms=estado["latencia_cpu_torch"]["p95_ms"],
                          n_parametros=red.n_parametros(), motivo=estado.get("motivo")))
        e3.log(f"Exportación {m['clave']}: {estado['formato']} (error máximo {estado.get('error_absoluto_maximo')}).")
        del red, t, conj
        import gc
        gc.collect()
    import pandas as pd
    pd.DataFrame(filas).to_csv(carpeta_artefactos / "resumen_exportacion.csv", index=False)
    return filas

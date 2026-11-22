"""Corrector semántico por hora sobre E + C+R (Entregable 3, partición de desarrollo). Ver LEEME.txt y config/plan_corrector.json.

No se llama directamente: use CORRECTOR_WINDOWS.bat o corrector_linux.sh, que preparan el entorno.

    python ejecutar_corrector.py [--corpus RUTA]          todas las etapas -> resultados_corrector_e3 (reanudable)
    python ejecutar_corrector.py --prueba-rapida          todo en minutos con pocos datos y un codificador de prueba
Etapas: verificar residuos cr texto texto_estricto busqueda variantes placebo estricto evaluacion
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import subprocess
import sys
import time
import zipfile
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
PLAN = json.loads((RAIZ / "config" / "plan_corrector.json").read_text(encoding="utf-8"))
ETAPAS = ["verificar", "residuos", "cr", "texto", "texto_estricto", "busqueda", "variantes", "placebo", "estricto", "evaluacion"]

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, str(RAIZ / "src"))

import numpy as np          # noqa: E402
import torch                # noqa: E402
from tqdm import tqdm       # noqa: E402

from corrector import protocolo as CP                                # noqa: E402
from corrector import residuos as RS                                 # noqa: E402
from corrector import texto as TX                                    # noqa: E402
from entrenamiento import protocolo as PR                            # noqa: E402
from entrenamiento import pruebas                                    # noqa: E402
from entrenamiento.datos import DatosNumericos, sha256               # noqa: E402

ASCII = os.name == "nt"
RAPIDO = dict(max_epocas=2, n_tr=96, n_es=24, n_ev=150, semillas=[101, 102], anios_residuos=[2016, 2017, 2018, 2019, 2020, 2021],
              primer_anio_ajuste=2017, anio_seleccion=2019, anios_evaluacion=[2020, 2021], desplazamientos=[49, 91], docs_por_anio=300, lote_evaluacion=16)


class Bitacora:
    def __init__(self, ruta: Path):
        self.ruta = ruta

    def __call__(self, mensaje: str, consola: bool = True):
        linea = f"[{dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {mensaje}"
        with open(self.ruta, "a", encoding="utf-8") as f:
            f.write(linea + "\n")
        if consola:
            tqdm.write(linea)


def huellas() -> dict:
    rutas = sorted((RAIZ / "config").glob("*.json")) + sorted((RAIZ / "data").rglob("*.*")) + [RAIZ / "ejecutar_corrector.py"]
    rutas += sorted(p for p in (RAIZ / "src").rglob("*.py") if "__pycache__" not in p.parts) + sorted((RAIZ / "tests").glob("*.py"))
    return {str(r.relative_to(RAIZ)).replace("\\", "/"): sha256(r) for r in rutas if r.is_file()}


def entorno(dispositivo: str) -> dict:
    e = dict(python=platform.python_version(), sistema=platform.platform(), torch=torch.__version__, numpy=np.__version__, dispositivo=dispositivo,
             cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    for m in ("pandas", "scipy", "sklearn", "transformers"):
        try:
            e[m] = __import__(m).__version__
        except Exception:
            e[m] = None
    return e


def empaquetar(salida: Path, log) -> Path:
    sello = dt.datetime.now().strftime("%Y%m%d_%H%M")
    z = RAIZ / f"ENVIAR_resultados_corrector_{sello}.zip"
    pesados = lambda r: r.suffix in (".pt", ".onnx") or "texto_vectores" in r.parts or r.name.endswith(".parcial.npz")   # noqa: E731
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as f:
        for r in sorted(salida.rglob("*")):
            if r.is_file() and not pesados(r):
                f.write(r, Path(salida.name) / r.relative_to(salida))
    log(f"Archivo para enviar: {z.name} ({z.stat().st_size / 2**20:.1f} MB). Los vectores de texto quedan en {salida.name}\\texto_vectores (no hace falta enviarlos).")
    return z


def main() -> int:
    ap = argparse.ArgumentParser(description="Corrector semántico por hora sobre E + C+R (solo desarrollo)")
    ap.add_argument("--corpus", default=PLAN["texto"]["ruta_por_defecto"], help="ruta de corpus_texto.parquet (corpus ampliado del 20 de noviembre)")
    ap.add_argument("--etapas", nargs="+", choices=ETAPAS, default=None)
    ap.add_argument("--prueba-rapida", action="store_true")
    ap.add_argument("--dispositivo", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--hilos", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)))
    ap.add_argument("--codificador", default=None, help="ruta local del codificador E5 si no hay internet")
    ap.add_argument("--sin-empaquetar", action="store_true")
    a = ap.parse_args()
    a.etapas = a.etapas or list(ETAPAS)
    rapido = dict(RAPIDO) if a.prueba_rapida else None
    salida = RAIZ / ("resultados_prueba_rapida_corrector" if rapido else "resultados_corrector_e3")
    salida.mkdir(exist_ok=True)
    log = Bitacora(salida / "bitacora.log")
    dispositivo = ("cuda" if torch.cuda.is_available() else "cpu") if a.dispositivo == "auto" else a.dispositivo
    torch.set_num_threads(a.hilos)
    log("=" * 78)
    log(f"{'PRUEBA RÁPIDA' if rapido else 'CORRECTOR'} -> {salida.name} | etapas {a.etapas} | dispositivo {dispositivo}"
        + (f" ({torch.cuda.get_device_name(0)})" if dispositivo == "cuda" else ""))
    corpus = Path(a.corpus)
    if rapido and not corpus.exists():
        alt = RAIZ / "data" / "processed" / "corpus_texto.parquet"
        corpus = alt if alt.exists() else corpus
    if set(a.etapas) & {"texto", "texto_estricto"} and not corpus.exists():
        log(f"ERROR: no encuentro el corpus en {corpus}. Pase la ruta con --corpus \"RUTA\\corpus_texto.parquet\".")
        return 2
    h = huellas()
    rreg = salida / "REGISTRO.json"
    if rreg.exists():
        reg = PR.leer_json(rreg)
        if reg["huellas_sha256"] != h:
            cambiados = sorted(k for k in set(h) | set(reg["huellas_sha256"]) if reg["huellas_sha256"].get(k) != h.get(k))
            log(f"ERROR: estos archivos cambiaron después de iniciar: {cambiados}. Para empezar de cero, renombre la carpeta {salida.name}.")
            return 2
        reg.setdefault("reanudaciones", []).append(dt.datetime.now().astimezone().isoformat(timespec="seconds"))
    else:
        reg = dict(registro="Escrito antes de entrenar: plan, regla, semillas y código quedan fijados por las huellas.", plan=PLAN["version"],
                   creado=dt.datetime.now().astimezone().isoformat(timespec="seconds"), huellas_sha256=h, prueba_2022_2025="no se lee (ni su texto)",
                   reanudaciones=[], prueba_rapida=bool(rapido))
    PR.guardar_json(rreg, reg)
    PR.guardar_json(salida / f"entorno_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.json", entorno(dispositivo))
    t0 = time.time()
    if "verificar" in a.etapas:
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", str(RAIZ / "tests"), "-p", "no:cacheprovider"], capture_output=True, text=True, cwd=RAIZ)
        (salida / "pytest.txt").write_text(r.stdout + "\n" + r.stderr, encoding="utf-8")
        log(f"Pruebas unitarias: {(r.stdout.strip().splitlines() or ['sin salida'])[-1]}")
        if r.returncode != 0:
            log("ERROR: fallaron las pruebas unitarias (ver pytest.txt).")
            return 5
    datos = DatosNumericos(RAIZ)
    comp = pruebas.numericas(datos)
    if not all(comp.values()):
        log(f"ERROR: fallaron comprobaciones previas {[k for k, v in comp.items() if not v]}.")
        return 4
    cfg_texto = dict(datos.contrato["texto"])
    if a.codificador:
        cfg_texto["encoder"] = a.codificador
    cp = None
    for etapa in [e for e in ETAPAS if e in a.etapas and e != "verificar"]:
        log(f"---- Etapa: {etapa}")
        with tqdm(total=None, desc=f"Corrector {etapa}", unit="paso", dynamic_ncols=True, ascii=ASCII) as barra:
            if etapa == "residuos":
                RS.etapa_residuos(RAIZ, salida, PLAN, datos, dispositivo, log, rapido, a, barra)
            elif etapa == "cr":
                info = RS.etapa_cr(salida, PLAN, datos, log, rapido)
                log("C+R por año (semilla " + str(next(iter(info))) + "): " + ", ".join(f"{k} {v['reduccion_pct']:+.2f} %" for k, v in next(iter(info.values()))["por_anio"].items()))
            elif etapa in ("texto", "texto_estricto"):
                col = PLAN["texto"]["disponibilidad_principal" if etapa == "texto" else "disponibilidad_sensibilidad"]
                TX.etapa_texto(salida, PLAN, cfg_texto, corpus, col, dispositivo, log, rapido, barra)
            else:
                cp = cp or CP.Corrida(salida, PLAN, datos, dispositivo, log, rapido)
                cp.barra = barra
                r = getattr(cp, etapa)()
                if etapa == "evaluacion":
                    log("VEREDICTO: " + json.dumps(r["veredicto"], ensure_ascii=False))
    log(f"Etapas terminadas en {(time.time() - t0) / 3600:.2f} h.")
    if not a.sin_empaquetar and "evaluacion" in a.etapas:
        empaquetar(salida, log)
    return 0


if __name__ == "__main__":
    sys.exit(main())

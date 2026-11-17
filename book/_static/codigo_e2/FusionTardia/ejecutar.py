"""Entrenamiento completo del modelo descrito en config/modelo.json, variantes N y N+O.

Uso normal: no se llama a este archivo directamente; se usa ENTRENAR_WINDOWS.bat o
entrenar_linux_mac.sh, que preparan el entorno y lo ejecutan.

    python ejecutar.py                  corrida completa (reanuda si ya hay resultados)
    python ejecutar.py --prueba-rapida  verificación de pocos minutos; no produce resultados válidos
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import sys
import time
import traceback
import zipfile
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
CFG = json.loads((RAIZ / "config" / "modelo.json").read_text(encoding="utf-8"))
PAQ = CFG["paquete"]


def _cache_codificador() -> Path:
    """La descarga del codificador queda dentro del paquete; si un paquete vecino ya la tiene, se comparte."""
    propia = RAIZ / ".cache_hf"
    if propia.exists():
        return propia
    for otra in sorted(RAIZ.parent.glob("*/.cache_hf")):
        if (otra / "hub").exists():
            return otra
    return propia


os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # reproducibilidad en GPU
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HOME", str(_cache_codificador()))
sys.path.insert(0, str(RAIZ / "src"))

import numpy as np          # noqa: E402
import torch                # noqa: E402
from tqdm import tqdm       # noqa: E402

from entrenamiento import protocolo as PR      # noqa: E402
from entrenamiento import pruebas, texto       # noqa: E402
from entrenamiento.datos import DatosNumericos, sha256   # noqa: E402

ASCII = os.name == "nt"     # barra de avance con caracteres simples en la consola de Windows
RAPIDO = dict(max_epocas=2, n_tr=320, n_es=48, n_ev=48, finalistas=1, semillas_finales=[1000, 1001], remuestreos=500, documentos_codificador=24)
SALIDA_REAL, SALIDA_RAPIDA = PAQ["carpeta_resultados"], "resultados_prueba_rapida"


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
    rutas = sorted((RAIZ / "config").glob("*.json")) + [RAIZ / "data" / "processed" / n for n in ("panel_horario.parquet", "corpus_oficial.parquet", "feature_registry.json")]
    rutas += [RAIZ / "ejecutar.py"] + sorted((RAIZ / "src").rglob("*.py"))
    return {str(r.relative_to(RAIZ)).replace("\\", "/"): sha256(r) for r in rutas}


def entorno(dispositivo: str) -> dict:
    e = dict(python=platform.python_version(), sistema=platform.platform(), procesador=platform.processor() or platform.machine(), nucleos=os.cpu_count(),
             torch=torch.__version__, numpy=np.__version__, dispositivo=dispositivo, cuda_disponible=torch.cuda.is_available())
    if torch.cuda.is_available():
        e.update(gpu=torch.cuda.get_device_name(0), memoria_gpu_gb=round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1), cuda=torch.version.cuda)
    for m in ("pandas", "pyarrow", "transformers", "tqdm"):
        try:
            e[m] = __import__(m).__version__
        except Exception:
            e[m] = None
    return e


def empaquetar(salida: Path, log) -> list[Path]:
    """Dos archivos para enviar: resultados (liviano) y pesos + vectores de texto (pesado)."""
    sello = dt.datetime.now().strftime("%Y%m%d_%H%M")
    pesados = lambda r: r.name in ("modelo.pt", "vectores_documento.npy", "parcial.npz")   # noqa: E731
    z1, z2 = RAIZ / f"ENVIAR_resultados_{PAQ['clave']}_{sello}.zip", RAIZ / f"ENVIAR_pesos_y_vectores_{PAQ['clave']}_{sello}.zip"
    for destino, criterio, nivel in ((z1, lambda r: not pesados(r), zipfile.ZIP_DEFLATED), (z2, pesados, zipfile.ZIP_STORED)):
        with zipfile.ZipFile(destino, "w", nivel) as z:
            for r in sorted(salida.rglob("*")):
                if r.is_file() and criterio(r):
                    z.write(r, Path(salida.name) / r.relative_to(salida))
    log(f"Archivos para enviar: {z1.name} ({z1.stat().st_size / 2**20:.1f} MB) y {z2.name} ({z2.stat().st_size / 2**20:.1f} MB)")
    return [z1, z2]


def resumen_texto(salida: Path, res: dict, segundos: float, rapido: bool) -> str:
    L = [PAQ["nombre"].upper() + " - RESUMEN" + (" DE LA PRUEBA RÁPIDA (sin validez)" if rapido else ""), ""]
    L.append(f"Prueba retrospectiva: {res['periodo_prueba'][0]} a {res['periodo_prueba'][1]} ({res['n_origenes_prueba']} orígenes diarios), semillas {res['semillas']}")
    L.append("Menor MASE es mejor. Referencias: " + ", ".join(f"{k} = {v:.4f}" for k, v in res["referencias"].items()))
    for v, d in res["variantes"].items():
        c = d["configuracion"]
        desv = "" if d["MASE_desv_semillas"] is None else f" (desv. entre semillas {d['MASE_desv_semillas']:.4f})"
        detalle = ", ".join(f"{k} {c[k]}" for k in ("nucleo", "d_model", "capas", "capas_mezcla") if k in c)
        L.append(f"  {v:4s}: MASE {d['MASE_media_semillas']:.4f}{desv} | configuración {c['id']} ({detalle})")
    k = res.get("contraste_N+O_menos_N")
    if k:
        p = k["primario"]
        L.append(f"  N+O frente a N: {p['diferencia_relativa_pct']:+.2f} % (IC 95 % por bloques de {p['bloque_dias']} días: {p['ic95_diferencia_relativa_pct'][0]:+.2f} a "
                 f"{p['ic95_diferencia_relativa_pct'][1]:+.2f}); margen práctico {k['margen_practico_pct']:.0f} %. Negativo favorece al texto.")
    L += ["", f"Duración de esta ejecución: {segundos / 3600:.2f} h", "Detalle: carpetas busqueda/ y final/ ; desviaciones declaradas en REGISTRO.json"]
    s = "\n".join(L)
    (salida / "RESUMEN.txt").write_text(s, encoding="utf-8")
    return s


def main() -> int:
    ap = argparse.ArgumentParser(description="Entrenamiento: " + PAQ["nombre"])
    ap.add_argument("--prueba-rapida", action="store_true", help="verifica la instalación en pocos minutos; no produce resultados válidos")
    ap.add_argument("--dispositivo", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--codificador", default=None, help="ruta local del codificador de texto, si ya está descargado")
    ap.add_argument("--sin-descarga", action="store_true", help="solo con --prueba-rapida: no intenta descargar el codificador")
    ap.add_argument("--ensayo", action="store_true", help=argparse.SUPPRESS)   # recorrido completo (texto real y archivos de envío) con tamaños mínimos
    a = ap.parse_args()

    cfg = json.loads(json.dumps(CFG))
    rapido = dict(RAPIDO, configuraciones=cfg["prueba_rapida"]["configuraciones"]) if (a.prueba_rapida or a.ensayo) else None
    texto_de_prueba = a.prueba_rapida          # la prueba rápida usa vectores seudoaleatorios; el ensayo, el codificador
    salida = RAIZ / (SALIDA_RAPIDA if a.prueba_rapida else "resultados_ensayo" if a.ensayo else SALIDA_REAL)
    salida.mkdir(exist_ok=True)
    log = Bitacora(salida / "bitacora.log")
    dispositivo = ("cuda" if torch.cuda.is_available() else "cpu") if a.dispositivo == "auto" else a.dispositivo
    if dispositivo == "cuda" and not torch.cuda.is_available():
        log("Se pidió GPU pero PyTorch no la detecta; se usa CPU.")
        dispositivo = "cpu"
    t_ini = time.time()
    log("=" * 78)
    log(("PRUEBA RÁPIDA" if a.prueba_rapida else "ENSAYO" if a.ensayo else "ENTRENAMIENTO COMPLETO") + f": {PAQ['nombre']} | dispositivo: {dispositivo}"
        + (f" ({torch.cuda.get_device_name(0)})" if dispositivo == "cuda" else ""))
    if dispositivo == "cpu" and not rapido:
        log("AVISO: sin GPU NVIDIA el entrenamiento completo puede tardar varios días. Se puede interrumpir y reanudar.")

    # ---------------------------------------------------------------- registro previo al entrenamiento
    h = huellas()
    rreg = salida / "REGISTRO.json"
    if rreg.exists():
        reg = PR.leer_json(rreg)
        if reg["huellas_sha256"] != h:
            cambiados = [k for k in set(h) | set(reg["huellas_sha256"]) if reg["huellas_sha256"].get(k) != h.get(k)]
            log(f"ERROR: estos archivos cambiaron después de iniciar el entrenamiento: {sorted(cambiados)}. "
                f"Para empezar de cero, cambie el nombre de la carpeta {salida.name} y vuelva a ejecutar.")
            return 2
        reg.setdefault("reanudaciones", []).append(dt.datetime.now().astimezone().isoformat(timespec="seconds"))
        log("Se encontraron resultados anteriores: se reanuda donde quedó.")
    else:
        reg = dict(registro="Escrito antes de entrenar. Fija configuraciones, semillas, reglas de selección y desviaciones.",
                   creado=dt.datetime.now().astimezone().isoformat(timespec="seconds"), paquete=cfg["version"], modelo=PAQ["nombre"], prueba_rapida=bool(rapido),
                   plan=dict(variantes=cfg["variantes"], busqueda=cfg["busqueda"], final=cfg["final"], entrenamiento=cfg["entrenamiento"], inferencia=cfg["inferencia"],
                             arquitectura_fija=cfg["arquitectura_fija"], configuraciones=cfg["configuraciones"]),
                   desviaciones_declaradas=cfg["desviaciones_declaradas"], huellas_sha256=h, reanudaciones=[])
        if rapido:
            reg["parametros_prueba_rapida"] = rapido
    PR.guardar_json(rreg, reg)
    PR.guardar_json(salida / "entorno.json", entorno(dispositivo))

    # ---------------------------------------------------------------- datos y comprobaciones
    log("Cargando el panel y verificando su SHA-256...")
    datos = DatosNumericos(RAIZ)
    clima = [c for c in datos.hist if c.startswith("x_obs_")]
    log(f"Entradas numéricas: {len(datos.hist)} históricas (demanda, precio, 4 de hidrología y {len(clima)} de clima con rezago de "
        f"{datos.contrato['numericas']['rezago_clima_h']} h) + {len(datos.cal)} de calendario.")
    comprobaciones = dict(numericas=pruebas.numericas(datos))
    configs = cfg["configuraciones"]
    if rapido:
        configs = [p for p in configs if p["id"] in rapido["configuraciones"]]
        cfg["busqueda"]["finalistas"] = rapido["finalistas"]
    ctx = PR.Contexto(RAIZ, salida, cfg, datos, dispositivo, log, rapido)

    # ---------------------------------------------------------------- texto
    ct = datos.contrato["texto"]
    con_documentos, con_descripciones = "N+O" in cfg["variantes"], bool(PAQ.get("usa_descripciones"))
    canal = None
    if texto_de_prueba and (con_documentos or con_descripciones) and not a.sin_descarga:
        log("Verificando la descarga del codificador de texto con unos pocos documentos...")
        corpus = texto.cargar_corpus_oficial(RAIZ)
        corpus["sha_contenido"] = [hashlib.sha256(texto.contenido(f).encode("utf-8")).hexdigest() for f in corpus.itertuples()]
        r = texto.codificar(corpus, ct, salida / "texto_verificacion", dispositivo, log, a.codificador, limite=rapido["documentos_codificador"])
        log(f"Codificador correcto: {r['documentos_codificados']} documentos en {r['segundos']} s.")
    if con_descripciones:
        ctx.descripciones = texto.preparar_descripciones(RAIZ, salida / "texto" / "descripciones", ct, datos.hist + datos.nombres_mascara, dispositivo, log,
                                                         modo_prueba=texto_de_prueba, codificador=a.codificador)
        log(f"Descripciones de canal codificadas: {len(ctx.descripciones)}.")
    if con_documentos:
        n_dias = len(datos.indice) // 24
        if texto_de_prueba:
            canal = texto.preparar_canal(RAIZ, salida / "texto", ct, datos.dia_base, n_dias, dispositivo, log, modo_prueba=True)
        else:
            corpus_n = len(texto.cargar_corpus_oficial(RAIZ))
            with tqdm(total=corpus_n, desc="Vectores de texto", unit="doc", dynamic_ncols=True, ascii=ASCII) as barra:
                rp = salida / "texto" / "parcial.npz"
                if rp.exists():
                    try:
                        barra.update(int(np.load(rp)["hecho"]))
                    except Exception:
                        pass
                elif (salida / "texto" / "vectores_documento.npy").exists():
                    barra.update(corpus_n)
                canal = texto.preparar_canal(RAIZ, salida / "texto", ct, datos.dia_base, n_dias, dispositivo, log, codificador=a.codificador, barra=barra)
        ctx.texto = dict(vectores=canal["vectores"], lista=canal["lista"])
        comprobaciones["texto"] = pruebas.texto(datos, canal, ct)
        log(f"Canal oficial: {len(canal['corpus'])} documentos; {int((canal['cobertura'].documentos_usados > 0).mean() * 100)} % de los días tienen al menos uno en 72 h.")
    comprobaciones["redes"] = pruebas.redes(datos, PAQ["red"], configs[0], canal, ctx.descripciones)
    PR.guardar_json(salida / "comprobaciones_previas.json", comprobaciones)
    fallas = [f"{g}.{k}" for g, d in comprobaciones.items() for k, v in d.items() if not v]
    if fallas:
        log(f"ERROR: fallaron las comprobaciones previas {fallas}. No se entrena. Envíe la carpeta {salida.name} a quien le compartió el paquete.")
        return 3
    log(f"Comprobaciones previas correctas ({sum(len(d) for d in comprobaciones.values())}).")

    # ---------------------------------------------------------------- entrenamientos
    nv, b = len(cfg["variantes"]), cfg["busqueda"]
    n_fin = len(rapido["semillas_finales"] if rapido else cfg["final"]["semillas"])
    total = nv * (len(configs) + b["finalistas"] * len(b["pliegues_restantes"]) + n_fin)
    with tqdm(total=total, desc="Entrenamientos", unit="modelo", dynamic_ncols=True, smoothing=0.1, ascii=ASCII) as barra:
        ctx.barra = barra
        log(f"Etapa 1 de 3: {len(configs)} configuraciones por variante en el pliegue {b['pliegue_inicial']} (semilla {b['semilla']}).")
        finalistas = PR.etapa_busqueda(ctx, configs)
        log(f"Finalistas: {finalistas}")
        log(f"Etapa 2 de 3: finalistas en los pliegues {b['pliegues_restantes']}.")
        elegidas = PR.etapa_pliegues(ctx, configs, finalistas)
        log(f"Configuración elegida por variante: {elegidas}")
        log("Etapa 3 de 3: ajuste final y prueba retrospectiva 2022-2025.")
        res = PR.etapa_final(ctx, configs, elegidas)
    seg = time.time() - t_ini
    reg = PR.leer_json(rreg)
    reg["terminado"] = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    PR.guardar_json(rreg, reg)
    tqdm.write("\n" + resumen_texto(salida, res, seg, bool(rapido)) + "\n")
    if a.prueba_rapida:
        log("PRUEBA RÁPIDA CORRECTA. El equipo está listo: ahora ejecute el entrenamiento completo.")
    else:
        zs = empaquetar(salida, log)
        log("ENTRENAMIENTO TERMINADO. Envíe estos dos archivos (están en la carpeta del paquete, junto a ejecutar.py):")
        for z in zs:
            log(f"   {z.name}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrumpido. No se pierde lo ya entrenado: vuelva a ejecutar y continúa donde quedó.")
        sys.exit(130)
    except Exception:
        print("\n" + "=" * 78 + "\nERROR. Copie el texto siguiente y envíelo junto con la carpeta de resultados:\n")
        traceback.print_exc()
        try:
            carpeta = RAIZ / (SALIDA_RAPIDA if "--prueba-rapida" in sys.argv else "resultados_ensayo" if "--ensayo" in sys.argv else SALIDA_REAL)
            with open(carpeta / "bitacora.log", "a", encoding="utf-8") as f:
                f.write(traceback.format_exc())
        except Exception:
            pass
        sys.exit(1)

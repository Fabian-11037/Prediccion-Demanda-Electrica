"""Canal oficial de texto: vectores por documento y listas de documentos elegibles por día.

Contrato (config/diseno_entrenamiento_texto.json):
  * escenario declarado; un documento es elegible si su disponibilidad cae en
    [origen - 72 h, origen]; como máximo los 64 más recientes;
  * codificador congelado intfloat/multilingual-e5-base en la revisión fijada, prefijo
    ``query: ``, promedio de tokens con máscara y normalización L2;
  * máximo 512 tokens por fragmento (incluye prefijo y tokens especiales), solapamiento
    de 64 y hasta 8 fragmentos por documento, en orden de lectura;
  * no se concatenan URL, identificadores ni fechas de recuperación al contenido.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

DIM = 768


def cargar_corpus_oficial(raiz: Path) -> pd.DataFrame:
    t = pd.read_parquet(Path(raiz) / "data" / "processed" / "corpus_oficial.parquet")
    t["disponible_desde"] = pd.to_datetime(t["disponible_desde"], utc=True)
    if not t["disponible_desde"].is_monotonic_increasing:
        raise RuntimeError("El corpus debe estar ordenado por disponibilidad")
    return t.reset_index(drop=True)


def contenido(fila) -> str:
    cuerpo = (fila.texto or "").strip()
    return (fila.titulo.strip() + "\n" + cuerpo) if cuerpo else fila.titulo.strip()


def listas_por_dia(disponible: pd.Series, dia_base: pd.Timestamp, n_dias: int, ventana_h: int, max_n: int):
    """Para cada día d (00:00 de Bogotá) devuelve los documentos con disponibilidad en [d - ventana, d].

    Misma regla que ``demanda.corpus.documentos_en_origen``: extremos inclusivos y, si hay más de
    ``max_n``, se conservan los más recientes. Devuelve (lista, total_en_ventana): ``lista`` es
    (n_dias, max_n) con posiciones del corpus y -1 como relleno.
    """
    valores = pd.DatetimeIndex(disponible).as_unit("ns").asi8
    dias = pd.DatetimeIndex([dia_base + pd.Timedelta(days=i) for i in range(n_dias)]).tz_convert("UTC").as_unit("ns").asi8
    hi = np.searchsorted(valores, dias, side="right")
    lo = np.searchsorted(valores, dias - int(ventana_h) * 3600 * 10**9, side="left")
    total = hi - lo
    lo2 = np.maximum(lo, hi - max_n)
    lista = np.full((n_dias, max_n), -1, dtype="int64")
    for i in range(n_dias):
        k = hi[i] - lo2[i]
        if k:
            lista[i, :k] = np.arange(lo2[i], hi[i])
    return lista, total


# --------------------------------------------------------------------------- codificación
def fragmentar(ids: list[int], capacidad: int, solape: int, max_frag: int):
    """Fragmentos en orden de lectura. Devuelve (fragmentos conservados, número total de fragmentos)."""
    if len(ids) <= capacidad:
        return [ids], 1
    paso = capacidad - solape
    inicios = list(range(0, len(ids) - solape, paso))
    return [ids[i:i + capacidad] for i in inicios[:max_frag]], len(inicios)


def _ficha(cfg_texto: dict, corpus: pd.DataFrame, codificador: str) -> dict:
    h = hashlib.sha256()
    for t in corpus["sha_contenido"]:
        h.update(t.encode())
    return dict(encoder=codificador, revision=cfg_texto["revision"], prefijo=cfg_texto["prefijo"], pooling=cfg_texto["pooling"],
                normalizacion=cfg_texto["normalizacion"], tokens_max=cfg_texto["tokens_max"], solapamiento_tokens=cfg_texto["solapamiento_tokens"],
                max_fragmentos_documento=cfg_texto["max_fragmentos_documento"], n_documentos=int(len(corpus)), sha256_contenidos=h.hexdigest())


def vectores_de_prueba(corpus: pd.DataFrame) -> np.ndarray:
    """Vectores seudoaleatorios deterministas por documento. SOLO para verificar la instalación."""
    out = np.empty((len(corpus), DIM), dtype="float32")
    for i, s in enumerate(corpus["sha_contenido"]):
        g = np.random.default_rng(int(s[:16], 16))
        v = g.standard_normal(DIM).astype("float32")
        out[i] = v / np.linalg.norm(v)
    return out


def _copiar_de_vecino(vecinos: Path | None, carpeta: Path, ficha: dict, log) -> dict | None:
    """Si otro paquete extraído en la misma carpeta ya calculó estos mismos vectores, se copian."""
    if vecinos is None:
        return None
    for rf2 in sorted(Path(vecinos).glob("*/resultados_*/texto/ficha.json")):
        try:
            r = json.loads(rf2.read_text(encoding="utf-8"))
            origen = rf2.parent
            if r.get("ficha") != ficha or r.get("codificador_de_prueba") or not (origen / "vectores_documento.npy").exists() or origen.resolve() == carpeta.resolve():
                continue
            for nombre in ("vectores_documento.npy", "indice.parquet", "ficha.json"):
                if (origen / nombre).exists():
                    shutil.copyfile(origen / nombre, carpeta / nombre)
            log(f"Vectores de texto copiados del paquete vecino {origen.parent.parent.name}: no se recalculan.")
            return r
        except Exception:
            continue
    return None


def codificar(corpus: pd.DataFrame, cfg_texto: dict, carpeta: Path, dispositivo: str, log, codificador: str | None = None,
              limite: int | None = None, barra=None, vecinos: Path | None = None) -> dict:
    """Calcula (o reanuda) los vectores por documento y los guarda en ``carpeta``.

    ``codificador`` permite una ruta local; por defecto se descarga el modelo fijado en el contrato.
    ``limite`` codifica solo los primeros documentos (verificación de instalación).
    """
    import torch
    import transformers
    from transformers import AutoModel, AutoTokenizer

    transformers.logging.set_verbosity_error()   # el aviso de "más de 512 tokens" no aplica: aquí se fragmenta
    try:
        transformers.utils.logging.disable_progress_bar()   # la barra de carga de pesos se mezclaría con la de avance
    except Exception:
        pass

    carpeta = Path(carpeta); carpeta.mkdir(parents=True, exist_ok=True)
    nombre = codificador or cfg_texto["encoder"]
    es_local = Path(nombre).exists()
    ficha = _ficha(cfg_texto, corpus, nombre)
    rf, rv, ri, rp = carpeta / "ficha.json", carpeta / "vectores_documento.npy", carpeta / "indice.parquet", carpeta / "parcial.npz"
    if limite is None and rf.exists() and rv.exists() and json.loads(rf.read_text(encoding="utf-8")).get("ficha") == ficha:
        log("Vectores de texto ya calculados; se reutilizan.")
        return json.loads(rf.read_text(encoding="utf-8"))
    if limite is None and not rp.exists():
        r = _copiar_de_vecino(vecinos, carpeta, ficha, log)
        if r is not None:
            if barra is not None:
                barra.update(len(corpus))
            return r

    log(f"Cargando el codificador de texto {nombre} (la primera vez descarga cerca de 1 GB)...")
    kw = {} if es_local else {"revision": cfg_texto["revision"]}
    try:
        tok = AutoTokenizer.from_pretrained(nombre, **kw)
        modelo = AutoModel.from_pretrained(nombre, **kw).to(dispositivo).eval()
    except Exception as e:
        raise RuntimeError(f"No se pudo cargar el codificador de texto {nombre} (revisión {cfg_texto['revision']}). Revise la conexión a internet y vuelva a "
                           f"ejecutar. Si el mensaje original dice que la revisión o el modelo no existen, avise al autor del paquete. Mensaje original: {e}") from e
    dim = int(modelo.config.hidden_size)
    ids_prefijo = tok(cfg_texto["prefijo"], add_special_tokens=False)["input_ids"]
    # Tokens especiales de una secuencia simple, deducidos del propio tokenizador (válido en transformers 4 y 5).
    completo = tok(cfg_texto["prefijo"], add_special_tokens=True)["input_ids"]
    k = next(i for i in range(len(completo) - len(ids_prefijo) + 1) if completo[i:i + len(ids_prefijo)] == ids_prefijo)
    inicio, cierre = completo[:k], completo[k + len(ids_prefijo):]
    capacidad = int(cfg_texto["tokens_max"]) - len(inicio) - len(cierre) - len(ids_prefijo)
    solape, max_frag = int(cfg_texto["solapamiento_tokens"]), int(cfg_texto["max_fragmentos_documento"])
    n = len(corpus) if limite is None else min(limite, len(corpus))

    vec = np.zeros((n, dim), dtype="float32")
    meta = np.zeros((n, 3), dtype="int64")  # tokens, fragmentos totales, fragmentos usados
    hecho = 0
    if limite is None and rp.exists():
        z = np.load(rp, allow_pickle=False)
        if str(z["sha"]) == ficha["sha256_contenidos"] and z["vec"].shape[1] == dim:
            hecho = int(z["hecho"]); vec[:hecho] = z["vec"][:hecho]; meta[:hecho] = z["meta"][:hecho]
            log(f"Se reanuda la codificación desde el documento {hecho} de {n}.")
    lote = 32 if dispositivo != "cpu" else 8
    bloque = 256
    t0 = time.time()
    for a in range(hecho, n, bloque):
        b = min(a + bloque, n)
        frags, dueno = [], []
        for i in range(a, b):
            ids = tok(contenido(corpus.iloc[i]), add_special_tokens=False)["input_ids"]
            fr, total = fragmentar(ids, capacidad, solape, max_frag)
            meta[i] = (len(ids), total, len(fr))
            for f in fr:
                frags.append(inicio + ids_prefijo + f + cierre); dueno.append(i - a)
        orden = np.argsort([len(f) for f in frags])
        emb = np.zeros((len(frags), dim), dtype="float32")
        for j in range(0, len(orden), lote):
            sel = orden[j:j + lote]
            largo = max(len(frags[k]) for k in sel)
            ids_t = torch.full((len(sel), largo), tok.pad_token_id, dtype=torch.long)
            att = torch.zeros((len(sel), largo), dtype=torch.long)
            for r, k in enumerate(sel):
                ids_t[r, :len(frags[k])] = torch.tensor(frags[k]); att[r, :len(frags[k])] = 1
            ids_t, att = ids_t.to(dispositivo), att.to(dispositivo)
            with torch.no_grad():                                  # precisión simple, sin autocast
                h = modelo(input_ids=ids_t, attention_mask=att).last_hidden_state.float()
            m = att.unsqueeze(-1).float()
            v = (h * m).sum(1) / m.sum(1).clamp(min=1.0)          # promedio de tokens con máscara
            v = torch.nn.functional.normalize(v, dim=-1)           # L2 por fragmento
            emb[sel] = v.cpu().numpy()
        dueno = np.asarray(dueno)
        for i in range(a, b):
            d = emb[dueno == (i - a)].mean(axis=0)
            vec[i] = d / max(float(np.linalg.norm(d)), 1e-12)       # vector del documento: promedio de fragmentos, L2
        if barra is not None:
            barra.update(b - a)
        if limite is None and (b == n or (b // bloque) % 4 == 0):
            np.savez(rp, vec=vec, meta=meta, hecho=b, sha=ficha["sha256_contenidos"])
    del modelo
    if dispositivo.startswith("cuda"):
        torch.cuda.empty_cache()
    resumen = dict(ficha=ficha, dimension=dim, documentos_codificados=int(n), segundos=round(time.time() - t0, 1), dispositivo=dispositivo,
                   documentos_truncados=int((meta[:n, 1] > meta[:n, 2]).sum()),
                   porcentaje_documentos_truncados=round(100 * float((meta[:n, 1] > meta[:n, 2]).mean()), 2),
                   porcentaje_tokens_descartados=round(100 * float(1 - np.minimum(meta[:n, 0], meta[:n, 2] * capacidad).sum() / max(meta[:n, 0].sum(), 1)), 2),
                   capacidad_tokens_por_fragmento=int(capacidad), tokens_especiales=[len(inicio), len(cierre)], tokens_prefijo=len(ids_prefijo),
                   transformers=transformers.__version__, codificador_de_prueba=False)
    if limite is None:
        np.save(rv, vec.astype("float16"))
        pd.DataFrame(dict(id=corpus["id"].to_numpy()[:n], fuente=corpus["fuente"].to_numpy()[:n], sha_contenido=corpus["sha_contenido"].to_numpy()[:n],
                          tokens=meta[:n, 0], fragmentos_totales=meta[:n, 1], fragmentos_usados=meta[:n, 2])).to_parquet(ri, index=False)
        rf.write_text(json.dumps(resumen, ensure_ascii=False, indent=1), encoding="utf-8")
        if rp.exists():
            rp.unlink()
    return resumen


def preparar_descripciones(raiz: Path, carpeta: Path, cfg_texto: dict, canales: list[str], dispositivo: str, log, modo_prueba: bool = False,
                           codificador: str | None = None, descripciones: dict | None = None) -> np.ndarray:
    """Vectores de las descripciones de canal (TGForecaster), con el mismo codificador y prefijo que los documentos.
    ``descripciones`` permite pasar las de un conjunto adicional (por omisión, config/descripciones_canal.json)."""
    desc = descripciones or json.loads((Path(raiz) / "config" / "descripciones_canal.json").read_text(encoding="utf-8"))["canales"]
    faltan = [c for c in canales if c not in desc]
    if faltan:
        raise RuntimeError(f"Faltan descripciones de canal: {faltan}")
    t = pd.DataFrame(dict(id=canales, fuente="descripcion_canal", titulo=[desc[c] for c in canales], texto=""))
    t["sha_contenido"] = [hashlib.sha256(contenido(f).encode("utf-8")).hexdigest() for f in t.itertuples()]
    carpeta = Path(carpeta); carpeta.mkdir(parents=True, exist_ok=True)
    if modo_prueba:
        vec = vectores_de_prueba(t)
        (carpeta / "ficha.json").write_text(json.dumps(dict(ficha=_ficha(cfg_texto, t, "VECTORES_DE_PRUEBA"), codificador_de_prueba=True), ensure_ascii=False, indent=1), encoding="utf-8")
    else:
        codificar(t, cfg_texto, carpeta, dispositivo, log, codificador)
        vec = np.load(carpeta / "vectores_documento.npy").astype("float32")
    t[["id", "titulo"]].rename(columns=dict(id="canal", titulo="descripcion")).to_csv(carpeta / "descripciones_usadas.csv", index=False)
    return vec


def preparar_canal(raiz: Path, carpeta: Path, cfg_texto: dict, dia_base, n_dias: int, dispositivo: str, log, modo_prueba: bool = False,
                   codificador: str | None = None, barra=None) -> dict:
    """Devuelve los vectores por documento, las listas por día y la cobertura del canal oficial."""
    corpus = cargar_corpus_oficial(raiz)
    corpus["sha_contenido"] = [hashlib.sha256(contenido(f).encode("utf-8")).hexdigest() for f in corpus.itertuples()]
    carpeta = Path(carpeta); carpeta.mkdir(parents=True, exist_ok=True)
    if modo_prueba:
        vec = vectores_de_prueba(corpus)
        resumen = dict(ficha=_ficha(cfg_texto, corpus, "VECTORES_DE_PRUEBA"), codificador_de_prueba=True, dimension=DIM)
        (carpeta / "ficha.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=1), encoding="utf-8")
    else:
        resumen = codificar(corpus, cfg_texto, carpeta, dispositivo, log, codificador, barra=barra, vecinos=Path(raiz).resolve().parent)
        vec = np.load(carpeta / "vectores_documento.npy").astype("float32")
    lista, total = listas_por_dia(corpus["disponible_desde"], dia_base, n_dias, cfg_texto["ventana_h"], cfg_texto["max_documentos_canal"])
    dias = pd.DatetimeIndex([dia_base + pd.Timedelta(days=i) for i in range(n_dias)])
    usados = (lista >= 0).sum(axis=1)
    cob = pd.DataFrame(dict(dia=dias.strftime("%Y-%m-%d"), documentos_en_ventana=total, documentos_usados=usados, descartados_por_limite=total - usados))
    cob.to_csv(carpeta / "cobertura_por_dia.csv", index=False)
    return dict(vectores=vec, lista=lista, resumen=resumen, corpus=corpus, cobertura=cob)

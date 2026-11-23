"""Etapa texto: fragmentos, intervalos anunciados, candidatos por día y codificación de los fragmentos usados.

Regla de disponibilidad: un documento entra en el día d solo si su disponibilidad (escenario del plan) es <= d 00:00
de Bogotá. El texto posterior al 31-dic-2021 no se lee (la prueba queda cerrada también para el texto).

Fragmentos: contenido = título + cuerpo, en trozos de 1.500 caracteres con 200 de solape, hasta 16 por documento
(unos 24.000 caracteres, frente a los 5.000 del diagnóstico); el codificador trunca en 512 tokens.

Candidatos de cada día objetivo d (como máximo 128, en tres reservas):
  alineados  hasta 32 fragmentos que contienen la mención de un intervalo anunciado que cubre d, de documentos
             disponibles en [d - 120 días, d]; los más recientes primero
  oficial    hasta 48 fragmentos de documentos oficiales disponibles en [d - 72 h, d]; documentos más recientes
             primero y, dentro de cada uno, en orden de lectura
  prensa     hasta 48 titulares disponibles en [d - 72 h, d]; los más recientes primero
Rasgos por candidato: canal oficial, canal prensa, alineado, reciente, log(1 + antigüedad en días), días desde el
inicio anunciado / 30, días hasta el fin anunciado / 30 (cero fuera de la reserva alineada), e índice de fuente.
Solo se codifican los fragmentos que aparecen en algún candidato.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import fechas as F

TZ = "America/Bogota"
DIM = 768
RASGOS = ["oficial", "prensa", "alineado", "reciente", "log_antiguedad", "desde_inicio_30", "hasta_fin_30"]


def sha256(ruta: Path) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def cargar_corpus(ruta: Path, col_disp: str, desde: str, hasta: str, limite_por_anio: int | None = None, max_caracteres: int = 21000) -> pd.DataFrame:
    """Lee solo las columnas y filas necesarias (sin cuarentena, con disponibilidad en [desde, hasta)) y recorta cada cuerpo
    a ``max_caracteres`` (lo que cabe en los fragmentos conservados), para no cargar en memoria texto que no se usa."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    ini, fin = pd.Timestamp(desde, tz=TZ).tz_convert("UTC"), pd.Timestamp(hasta, tz=TZ).tz_convert("UTC")
    partes = []
    for lote in pq.ParquetFile(ruta).iter_batches(batch_size=20000, columns=["id", "fuente", "tipo_fuente", "titulo", "texto", "cuarentena", col_disp]):
        d = lote.column(col_disp)
        ok = pc.and_(pc.and_(pc.invert(pc.fill_null(lote.column("cuarentena"), True)), pc.greater_equal(d, pa.scalar(ini))), pc.less(d, pa.scalar(fin)))
        lote = lote.filter(pc.fill_null(ok, False))
        if lote.num_rows:
            tx = pc.utf8_slice_codeunits(pc.fill_null(lote.column("texto"), ""), 0, max_caracteres)
            df = lote.set_column(lote.schema.get_field_index("texto"), "texto", tx).to_pandas()
            if limite_por_anio:                                 # solo la prueba rápida: pocos documentos por año y canal
                df = df.groupby([pd.to_datetime(df[col_disp], utc=True).dt.year, df.tipo_fuente == "news"], group_keys=False).head(
                    max(1, limite_por_anio // 10))
            partes.append(df)
    c = pd.concat(partes, ignore_index=True).rename(columns={col_disp: "disp"})
    del partes
    c["disp"] = pd.to_datetime(c.disp, utc=True)
    c["canal"] = np.where(c.tipo_fuente == "news", "prensa", "oficial")
    if limite_por_anio:                                         # solo la prueba rápida
        c = c.groupby([c.disp.dt.year, "canal"], group_keys=False).head(limite_por_anio)
    c = c.sort_values("disp", kind="stable").reset_index(drop=True)
    return c.drop(columns=["cuarentena", "tipo_fuente"])


def fragmentos(c: pd.DataFrame, largo: int = 1500, solape: int = 200, max_frag: int = 16) -> pd.DataFrame:
    filas = []
    for i, (t, x) in enumerate(zip(c.titulo, c.texto)):
        cont = (t or "").strip() + ("\n" + x.strip() if x and x.strip() and x.strip() != (t or "").strip() else "")
        paso = largo - solape
        n = 1 if len(cont) <= largo else min(max_frag, 1 + int(np.ceil((len(cont) - largo) / paso)))
        for k in range(n):
            a = k * paso
            filas.append((i, k, a, min(len(cont), a + largo)))
    return pd.DataFrame(filas, columns=["doc", "orden", "c0", "c1"])


def contenido(c: pd.DataFrame, i: int) -> str:
    t, x = c.titulo.iat[i] or "", c.texto.iat[i] or ""
    return t.strip() + ("\n" + x.strip() if x.strip() and x.strip() != t.strip() else "")


def intervalos_corpus(c: pd.DataFrame, frag: pd.DataFrame, log, largo: int = 1500, solape: int = 200) -> pd.DataFrame:
    """Intervalos anunciados de cada documento y el fragmento (posición global) que contiene la mención."""
    pub = (c.disp.dt.tz_convert(TZ) - pd.Timedelta(minutes=1)).dt.date.to_numpy()   # "fecha sin hora -> día siguiente 00:00"
    inicio_doc = frag.groupby("doc").indices
    paso = largo - solape
    filas = []
    t0 = time.time()
    for i in range(len(c)):
        cont = contenido(c, i)
        if not any(ch.isdigit() for ch in cont):
            continue
        for ini, fin, pos in F.intervalos(cont, pub[i]):
            ids = inicio_doc[i]
            k = pos // paso
            if k < len(ids):                                   # la mención cae en un fragmento conservado
                filas.append((i, ini, fin, pos, int(ids[k])))
    t = pd.DataFrame(filas, columns=["doc", "inicio", "fin", "pos", "fragmento"])
    log(f"Intervalos anunciados: {len(t)} en {t.doc.nunique() if len(t) else 0} documentos ({time.time() - t0:.0f} s).", consola=False)
    return t


def candidatos(c: pd.DataFrame, frag: pd.DataFrame, inter: pd.DataFrame, dias: pd.DatetimeIndex, fuentes: list[str],
               max_al: int = 32, max_of: int = 48, max_pr: int = 48, ventana_h: int = 72, antig_al_dias: int = 120) -> dict:
    M = max_al + max_of + max_pr
    n = len(dias)
    cand = np.full((n, M), -1, dtype="int64")
    feat = np.zeros((n, M, len(RASGOS)), dtype="float32")
    fte = np.zeros((n, M), dtype="int16")
    disp = pd.DatetimeIndex(c.disp).as_unit("ns").asi8
    origen = pd.DatetimeIndex(dias).tz_localize(TZ).tz_convert("UTC").as_unit("ns").asi8
    es_of = (c.canal == "oficial").to_numpy()
    idx_f = c.fuente.map({f: k + 1 for k, f in enumerate(fuentes)}).fillna(0).astype("int16").to_numpy()
    frag_de = frag.groupby("doc").indices                       # doc -> posiciones de sus fragmentos (en orden)
    of_docs, pr_docs = np.flatnonzero(es_of), np.flatnonzero(~es_of)
    of_disp, pr_disp = disp[of_docs], disp[pr_docs]
    dia_ns = 86400 * 10**9
    # reserva alineada: (día, fragmento, disp) por cada día cubierto por un intervalo
    al = {}
    base = pd.DatetimeIndex(dias).as_unit("ns").asi8
    if len(inter):
        d_ini = pd.DatetimeIndex(pd.to_datetime(inter.inicio)).as_unit("ns").asi8
        d_fin = pd.DatetimeIndex(pd.to_datetime(inter.fin)).as_unit("ns").asi8
        pos_dia = {v: k for k, v in enumerate(base)}
        for j in range(len(inter)):
            doc = int(inter.doc.iat[j])
            for v in range(d_ini[j], d_fin[j] + 1, dia_ns):
                k = pos_dia.get(v)
                if k is None:
                    continue
                if disp[doc] <= origen[k] and disp[doc] >= origen[k] - antig_al_dias * dia_ns:
                    al.setdefault(k, []).append((disp[doc], int(inter.fragmento.iat[j]), doc, d_ini[j], d_fin[j]))
    for k in range(n):
        o = origen[k]
        m = 0
        for dsp, fr, doc, a, b in sorted(set(al.get(k, [])), key=lambda z: -z[0])[:max_al]:
            cand[k, m] = fr
            feat[k, m] = [es_of[doc], not es_of[doc], 1, 0, np.log1p((o - dsp) / dia_ns), (base[k] - a) / dia_ns / 30, (b - base[k]) / dia_ns / 30]
            fte[k, m] = idx_f[doc]
            m += 1
        m = max_al
        for docs, dd, cupo, oficial in ((of_docs, of_disp, max_of, True), (pr_docs, pr_disp, max_pr, False)):
            hi = np.searchsorted(dd, o, side="right")
            lo = np.searchsorted(dd, o - ventana_h * 3600 * 10**9, side="left")
            usados = 0
            for doc in docs[lo:hi][::-1]:
                for fr in frag_de[doc]:
                    if usados >= cupo:
                        break
                    cand[k, m] = fr
                    feat[k, m] = [oficial, not oficial, 0, 1, np.log1p((o - disp[doc]) / dia_ns), 0, 0]
                    fte[k, m] = idx_f[doc]
                    m += 1
                    usados += 1
                if usados >= cupo:
                    break
            m = max_al + (max_of if oficial else max_of + max_pr)
    return dict(cand=cand, feat=feat, fuente=fte, M=M, reservas=dict(alineados=(0, max_al), oficial=(max_al, max_al + max_of),
                                                                      prensa=(max_al + max_of, M)))


# ------------------------------------------------------------------ codificación
def codificador_de_prueba(textos: list[str]) -> np.ndarray:
    """Solo la prueba rápida: bolsa de palabras con hash a 768 dimensiones, normalizada (no descarga ningún modelo)."""
    from sklearn.feature_extraction.text import HashingVectorizer
    v = HashingVectorizer(n_features=DIM, alternate_sign=True, norm="l2").transform([F.plano(t) for t in textos])
    return v.toarray().astype("float32")


def codificar(textos: list[str], cfg_texto: dict, ruta: Path, dispositivo: str, log, prueba: bool = False, barra=None) -> np.ndarray:
    """Vectores E5 (prefijo, promedio con máscara, L2) de cada texto, reanudable por bloques."""
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    huella = hashlib.sha256(("\x1f".join(textos) + cfg_texto["encoder"] + cfg_texto["revision"] + str(prueba)).encode("utf-8")).hexdigest()
    parcial = ruta.with_suffix(".parcial.npz")
    if ruta.exists() and json.loads(ruta.with_suffix(".json").read_text(encoding="utf-8")).get("huella") == huella:
        return np.load(ruta)
    if prueba:
        v = codificador_de_prueba(textos).astype("float16")
        np.save(ruta, v)
        ruta.with_suffix(".json").write_text(json.dumps(dict(huella=huella, codificador="prueba", n=len(textos))), encoding="utf-8")
        return v
    import torch
    import transformers
    from transformers import AutoModel, AutoTokenizer
    transformers.logging.set_verbosity_error()
    log(f"Cargando {cfg_texto['encoder']} (revisión {cfg_texto['revision'][:8]}; la primera vez descarga cerca de 1 GB)...")
    tok = AutoTokenizer.from_pretrained(cfg_texto["encoder"], revision=cfg_texto["revision"])
    modelo = AutoModel.from_pretrained(cfg_texto["encoder"], revision=cfg_texto["revision"]).to(dispositivo).eval()
    vec = np.zeros((len(textos), DIM), dtype="float16")
    hecho = 0
    if parcial.exists():
        z = np.load(parcial)
        if str(z["huella"]) == huella:
            hecho = int(z["hecho"])
            vec[:hecho] = z["vec"][:hecho]
            log(f"Se reanuda la codificación en el fragmento {hecho} de {len(textos)}.")
    lote = 32 if dispositivo != "cpu" else 8
    bloque = 4096
    t0 = time.time()
    for a in range(hecho, len(textos), bloque):
        b = min(len(textos), a + bloque)
        orden = np.argsort([len(t) for t in textos[a:b]]) + a
        for j in range(0, len(orden), lote):
            sel = orden[j:j + lote]
            enc = tok([cfg_texto["prefijo"] + textos[k] for k in sel], truncation=True, max_length=int(cfg_texto["tokens_max"]), padding=True,
                      return_tensors="pt").to(dispositivo)
            with torch.no_grad():
                h = modelo(**enc).last_hidden_state.float()
            msk = enc["attention_mask"].unsqueeze(-1).float()
            v = torch.nn.functional.normalize((h * msk).sum(1) / msk.sum(1).clamp(min=1.0), dim=-1)
            vec[sel] = v.cpu().numpy().astype("float16")
        if barra is not None:
            barra.update(b - a)
        np.savez(parcial, vec=vec, hecho=b, huella=huella)
        log(f"Codificados {b} de {len(textos)} fragmentos ({time.time() - t0:.0f} s).", consola=False)
    del modelo
    np.save(ruta, vec)
    ruta.with_suffix(".json").write_text(json.dumps(dict(huella=huella, codificador=cfg_texto["encoder"], revision=cfg_texto["revision"],
                                                         n=len(textos), segundos=round(time.time() - t0, 1))), encoding="utf-8")
    parcial.unlink(missing_ok=True)
    return vec


def etapa_texto(salida: Path, plan: dict, cfg_texto: dict, ruta_corpus: Path, col_disp: str, dispositivo: str, log, rapido: dict | None,
                barra=None) -> dict:
    tx = plan["texto"]
    car = salida / ("texto" if col_disp == tx["disponibilidad_principal"] else f"texto_{col_disp}")
    car.mkdir(parents=True, exist_ok=True)
    if (car / "candidatos.npz").exists() and (car / "resumen.json").exists():
        log(f"Candidatos de texto ({col_disp}) ya construidos; se reutilizan.")
        return json.loads((car / "resumen.json").read_text(encoding="utf-8"))
    h = sha256(ruta_corpus)
    if h != tx["corpus_sha256"] and not rapido:
        raise RuntimeError(f"El corpus {ruta_corpus} no es el preregistrado (SHA-256 {h[:12]}..., se esperaba {tx['corpus_sha256'][:12]}...)")
    t0 = time.time()
    c = cargar_corpus(ruta_corpus, col_disp, tx["desde"], tx["hasta"], rapido.get("docs_por_anio") if rapido else None,
                      tx["max_fragmentos_documento"] * (tx["caracteres_fragmento"] - tx["solape_caracteres"]) + tx["solape_caracteres"])
    log(f"Corpus ({col_disp}): {len(c)} documentos hasta {tx['hasta']} ({c.canal.value_counts().to_dict()}).")
    frag = fragmentos(c, tx["caracteres_fragmento"], tx["solape_caracteres"], tx["max_fragmentos_documento"])
    inter = intervalos_corpus(c, frag, log, tx["caracteres_fragmento"], tx["solape_caracteres"])
    dias = pd.date_range(tx["primer_dia_candidatos"], tx["ultimo_dia"], freq="D")
    fuentes = sorted(c.fuente.unique())
    k = candidatos(c, frag, inter, dias, fuentes, tx["max_alineados"], tx["max_oficial"], tx["max_prensa"], tx["ventana_h"], tx["antiguedad_alineados_dias"])
    usados = np.unique(k["cand"][k["cand"] >= 0])
    mapa = np.full(len(frag), -1, dtype="int64")
    mapa[usados] = np.arange(len(usados))
    compacto = np.where(k["cand"] >= 0, mapa[np.maximum(k["cand"], 0)], -1)
    textos = [contenido(c, int(frag.doc.iat[u]))[int(frag.c0.iat[u]):int(frag.c1.iat[u])] for u in usados]
    log(f"Fragmentos: {len(frag)} en total, {len(usados)} usados por algún día; se codifican esos.")
    emb = codificar(textos, cfg_texto, salida / "texto_vectores" / f"vectores_{col_disp}.npy", dispositivo, log, prueba=bool(rapido), barra=barra)
    np.savez_compressed(car / "candidatos.npz", dias=dias.strftime("%Y-%m-%d").to_numpy().astype("U10"), cand=compacto.astype("int32"),
                        feat=k["feat"].astype("float16"), fuente=k["fuente"], rasgos=np.array(RASGOS))
    frag.assign(doc_id=c.id.to_numpy()[frag.doc], fuente=c.fuente.to_numpy()[frag.doc]).iloc[usados].to_parquet(car / "fragmentos_usados.parquet", index=False)
    inter.assign(doc_id=c.id.to_numpy()[inter.doc]).to_parquet(car / "intervalos.parquet", index=False)
    (car / "vectores.txt").write_text(str(salida / "texto_vectores" / f"vectores_{col_disp}.npy"), encoding="utf-8")
    al = (compacto[:, :tx["max_alineados"]] >= 0).sum(1)
    res = dict(disponibilidad=col_disp, corpus_sha256=h, documentos=int(len(c)), por_canal=c.canal.value_counts().to_dict(), fragmentos=int(len(frag)),
               fragmentos_usados=int(len(usados)), intervalos=int(len(inter)), dias=int(len(dias)), fuentes=fuentes,
               dias_con_alineados_pct={str(a): float(100 * (al[dias.year == a] > 0).mean()) for a in sorted(set(dias.year))},
               candidatos_medios_por_reserva={r: float((compacto[:, a:b] >= 0).sum(1).mean()) for r, (a, b) in k["reservas"].items()},
               reservas=k["reservas"], segundos=round(time.time() - t0, 1))
    (car / "resumen.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"Texto listo: días con anuncio alineado por año {res['dias_con_alineados_pct']}.")
    return res

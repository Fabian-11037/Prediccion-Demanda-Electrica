"""Ejecuta las cadenas de la matriz de novedad de M4N en OpenAlex, arXiv y Semantic Scholar y guarda conteos y resultados."""  # Propósito
import json  # Lectura y escritura JSON
import time  # Pausas entre peticiones
import datetime  # Fecha de ejecución
import urllib.parse  # Construcción de URL
import urllib.request  # Peticiones HTTP
import xml.etree.ElementTree as ET  # Lectura del XML de arXiv

UA = {"User-Agent": "revision-novedad-m4n"}  # Agente genérico, sin datos personales
CADENAS = {  # Cadena lógica: (OpenAlex, arXiv, Semantic Scholar); None si no aplica
    "C1": ('("load forecasting" OR "electricity demand forecasting" OR "power demand forecasting") AND ("retrieval" OR "retrieval-augmented" OR "nearest neighbor" OR "nearest neighbour") AND ("residual" OR "forecast error" OR "error correction")',  # OpenAlex
           '(abs:"load forecasting" OR abs:"electricity demand forecasting") AND (abs:retrieval OR abs:"nearest neighbor") AND (abs:residual OR abs:"error correction" OR abs:"forecast error")',  # arXiv
           '("load forecasting" | "electricity demand forecasting" | "power demand forecasting") + (retrieval | "retrieval-augmented" | "nearest neighbor") + (residual | "forecast error" | "error correction")'),  # Semantic Scholar
    "C2": ('("load forecasting" OR "electricity demand forecasting") AND ("analog ensemble" OR "analogue ensemble" OR "analog forecasting" OR "similar day") AND ("error correction" OR "bias correction" OR "residual")',  # OpenAlex
           '(abs:"load forecasting" OR abs:"electricity demand forecasting") AND (abs:"analog ensemble" OR abs:"similar day" OR abs:analog) AND (abs:"error correction" OR abs:residual OR abs:"bias correction")',  # arXiv
           '("load forecasting" | "electricity demand forecasting") + ("analog ensemble" | "analogue ensemble" | "similar day") + ("error correction" | "bias correction" | residual)'),  # Semantic Scholar
    "C3": ('("load forecasting" OR "electricity demand forecasting") AND ("memory-augmented" OR "external memory" OR "memory bank" OR "episodic memory")',  # OpenAlex
           '(abs:"load forecasting" OR abs:"electricity demand forecasting") AND (abs:"memory-augmented" OR abs:"external memory" OR abs:"memory bank" OR abs:"episodic memory")',  # arXiv
           '("load forecasting" | "electricity demand forecasting") + ("memory-augmented" | "external memory" | "memory bank" | "episodic memory")'),  # Semantic Scholar
    "C4": ('("load forecasting" OR "electricity demand forecasting") AND ("residual correction" OR "error correction" OR "error compensation") AND ("attention" OR "gate" OR "gating")',  # OpenAlex
           '(abs:"load forecasting" OR abs:"electricity demand forecasting") AND (abs:"residual correction" OR abs:"error correction" OR abs:"error compensation") AND (abs:attention OR abs:gate OR abs:gating)',  # arXiv
           '("load forecasting" | "electricity demand forecasting") + ("residual correction" | "error correction" | "error compensation") + (attention | gate | gating)'),  # Semantic Scholar
    "C5": ('("pronóstico de demanda" OR "pronóstico de carga" OR "predicción de carga" OR "predicción de la demanda eléctrica") AND ("corrección de errores" OR "residuos" OR "memoria")',  # OpenAlex, en español
           None,  # arXiv no indexa resúmenes en español de forma útil
           '("pronóstico de demanda" | "pronóstico de carga" | "predicción de carga") + ("corrección de errores" | residuos | memoria)'),  # Semantic Scholar
    "C6": ('("analog ensemble" OR "analogue ensemble") AND ("forecast") AND ("error" OR "bias")',  # Origen del mecanismo: análogos en meteorología y renovables
           '(abs:"analog ensemble" OR abs:"analogue ensemble") AND abs:forecast',  # arXiv
           '("analog ensemble" | "analogue ensemble") + forecast + (error | bias)'),  # Semantic Scholar
    "C7": ('("retrieval-augmented" OR "retrieval augmented") AND ("time series forecasting")',  # Origen: recuperación en series de otros dominios
           '(abs:"retrieval-augmented" OR abs:"retrieval augmented") AND abs:"time series forecasting"',  # arXiv
           '("retrieval-augmented" | "retrieval augmented") + "time series forecasting"'),  # Semantic Scholar
    "C8": ('("nearest neighbor language model" OR "kNN-LM" OR "retrieval-augmented language model" OR "memorizing transformer")',  # Origen: memoria de vecinos en lenguaje
           '(abs:"nearest neighbor language model" OR abs:"kNN-LM" OR abs:"memorizing transformer")',  # arXiv
           '("nearest neighbor language model" | "kNN-LM" | "memorizing transformer")'),  # Semantic Scholar
}  # Fin de las cadenas


def get(u):  # Descarga con reintentos cortos
    for k in range(4):  # Hasta cuatro intentos
        try:  # Intento
            return urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=60).read()  # Contenido
        except Exception:  # Error de red o de límite de peticiones
            time.sleep(4 * (k + 1))  # Espera creciente
    raise RuntimeError(u)  # Falla definitiva


def openalex(q):  # OpenAlex: título y resumen, 2015–2026
    u = "https://api.openalex.org/works?" + urllib.parse.urlencode({"filter": f"title_and_abstract.search:{q},publication_year:2015-2026", "per-page": 50, "sort": "relevance_score:desc", "select": "id,doi,display_name,publication_year,abstract_inverted_index,cited_by_count"})  # URL
    d = json.loads(get(u))  # Respuesta
    out = []  # Resultados
    for w in d["results"]:  # Cada trabajo
        inv = w.get("abstract_inverted_index") or {}  # Resumen en forma invertida
        pos = sorted((p, t) for t, ps in inv.items() for p in ps)  # Reconstruye el orden de las palabras
        out.append({"titulo": w["display_name"], "anio": w["publication_year"], "doi": w.get("doi"), "id": w["id"], "citas": w.get("cited_by_count"), "resumen": " ".join(t for _, t in pos)[:1500]})  # Registro
    return d["meta"]["count"], out  # Conteo y resultados


def arxiv(q):  # arXiv: búsqueda en resumen
    u = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode({"search_query": q, "max_results": 50, "sortBy": "relevance"})  # URL
    r = ET.fromstring(get(u))  # XML de respuesta
    ns = {"a": "http://www.w3.org/2005/Atom", "o": "http://a9.com/-/spec/opensearch/1.1/"}  # Espacios de nombres
    out = [{"titulo": " ".join(e.find("a:title", ns).text.split()), "anio": int(e.find("a:published", ns).text[:4]), "doi": e.find("a:id", ns).text, "resumen": " ".join(e.find("a:summary", ns).text.split())[:1500]} for e in r.findall("a:entry", ns)]  # Entradas
    time.sleep(3)  # Pausa recomendada por arXiv
    return int(r.find("o:totalResults", ns).text), out  # Conteo y resultados


def s2(q):  # Semantic Scholar: búsqueda masiva con sintaxis booleana
    u = "https://api.semanticscholar.org/graph/v1/paper/search/bulk?" + urllib.parse.urlencode({"query": q, "fields": "title,year,externalIds,abstract,citationCount", "year": "2015-2026"})  # URL
    d = json.loads(get(u))  # Respuesta
    time.sleep(2)  # Pausa por límite de peticiones
    out = [{"titulo": p["title"], "anio": p.get("year"), "doi": (p.get("externalIds") or {}).get("DOI"), "id": p.get("paperId"), "citas": p.get("citationCount"), "resumen": (p.get("abstract") or "")[:1500]} for p in d.get("data", [])[:50]]  # Primeros 50
    return d.get("total", 0), out  # Conteo y resultados


if __name__ == "__main__":  # Ejecución directa
    res = {"fecha": datetime.date.today().isoformat(), "consultas": []}  # Contenedor
    for c, (qa, qx, qs) in CADENAS.items():  # Cada cadena
        for base, f, q in [("OpenAlex", openalex, qa), ("arXiv", arxiv, qx), ("Semantic Scholar", s2, qs)]:  # Cada base
            if q is None:  # Sin versión para esta base
                continue  # Se omite
            try:  # Ejecuta
                n, out = f(q)  # Conteo y registros
            except Exception as e:  # Error definitivo
                n, out = None, [{"error": str(e)}]  # Registra el error
            res["consultas"].append({"cadena": c, "base": base, "consulta": q, "resultados": n, "registros": out})  # Guarda
            print(c, base, n)  # Avance
    with open(f"busqueda_novedad_{res['fecha'].replace('-', '')}.json", "w", encoding="utf-8") as fh:  # Archivo de resultados
        json.dump(res, fh, ensure_ascii=False, indent=1)  # Escribe

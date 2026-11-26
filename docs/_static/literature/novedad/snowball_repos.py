"""Búsqueda por citación (hacia atrás y hacia adelante) de los trabajos más cercanos a M4N y revisión de repositorios de código."""  # Propósito
import json  # Lectura y escritura JSON
import re  # Filtros de texto
import time  # Pausas
import datetime  # Fecha
import urllib.parse  # URL
import urllib.request  # HTTP

UA = {"User-Agent": "revision-novedad-m4n"}  # Agente genérico
CERCANOS = {  # Trabajos semilla: nombre → búsqueda por título en OpenAlex
    "RATL (2026)": "RATL: Learning from Retrieved Residuals for Robust Multivariate Time-Series Forecasting",  # Residuos recuperados
    "WeL-RAC (2026)": "A Model-Agnostic Correction Framework for Short-Term Building Load Forecasting",  # Corrección por recuperación en carga de edificios
    "RAFT (2025)": "Retrieval Augmented Time Series Forecasting",  # Recuperación en series
    "TE-AG-GRU-RC (2026)": "Two-stage transformer encoder and attention-guided GRU-based residual correction for short-term load forecasting",  # Corrección de residuos en demanda
    "kNN-LM (2020)": "Generalization through Memorization: Nearest Neighbor Language Models",  # Origen: memoria de vecinos interpolada con un modelo base
}  # Fin de semillas
FILTRO = re.compile(r"residual|retriev|analog|nearest|memory|error correct|error compens", re.I)  # Mecanismo
DOMINIO = re.compile(r"load|demand|electric|power|energy", re.I)  # Problema


def get(u):  # Descarga con reintentos
    for k in range(4):  # Intentos
        try:  # Petición
            return json.loads(urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=60).read())  # JSON
        except Exception:  # Error
            time.sleep(5 * (k + 1))  # Espera
    return {}  # Sin respuesta


def obra(titulo):  # Busca la obra por título exacto aproximado
    d = get("https://api.openalex.org/works?" + urllib.parse.urlencode({"search": titulo, "per-page": 1, "select": "id,display_name,publication_year,referenced_works,cited_by_count"}))  # Consulta
    return (d.get("results") or [None])[0]  # Primer resultado


def titulos(ids):  # Títulos de una lista de identificadores OpenAlex
    out = []  # Resultados
    for i in range(0, len(ids), 50):  # Lotes de 50
        lote = "|".join(x.split("/")[-1] for x in ids[i:i + 50])  # Filtro por identificadores
        d = get("https://api.openalex.org/works?" + urllib.parse.urlencode({"filter": f"openalex_id:{lote}", "per-page": 50, "select": "id,display_name,publication_year,doi"}))  # Consulta
        out += d.get("results", [])  # Acumula
    return out  # Lista


res = {"fecha": datetime.date.today().isoformat(), "snowballing": [], "repositorios": []}  # Contenedor
for nombre, titulo in CERCANOS.items():  # Cada semilla
    w = obra(titulo)  # Obra encontrada
    if not w:  # No indexada
        res["snowballing"].append({"semilla": nombre, "indexada": False}); continue  # Registra y sigue
    atras = titulos(w.get("referenced_works") or [])  # Referencias (hacia atrás)
    citan = get("https://api.openalex.org/works?" + urllib.parse.urlencode({"filter": f"cites:{w['id'].split('/')[-1]}", "per-page": 200, "select": "id,display_name,publication_year,doi"})).get("results", [])  # Citas recibidas (hacia adelante)
    rel = lambda L: [{"titulo": x["display_name"], "anio": x["publication_year"], "doi": x.get("doi")} for x in L if x.get("display_name") and FILTRO.search(x["display_name"]) and DOMINIO.search(x["display_name"])]  # Relevantes: mecanismo y dominio en el título
    res["snowballing"].append({"semilla": nombre, "indexada": True, "titulo_indexado": w["display_name"], "referencias": len(atras), "citas": len(citan), "relevantes_atras": rel(atras), "relevantes_adelante": rel(citan)})  # Guarda
    print(nombre, "| ref", len(atras), "| citas", len(citan), "| rel", len(rel(atras)), len(rel(citan)))  # Avance

for q in ["residual retrieval forecasting", "retrieval augmented load forecasting", "analog ensemble load forecasting", "residual memory forecasting", "forecast error memory nearest neighbor"]:  # Consultas en GitHub
    d = get("https://api.github.com/search/repositories?" + urllib.parse.urlencode({"q": q, "per_page": 10}))  # Búsqueda de repositorios
    res["repositorios"].append({"fuente": "GitHub", "consulta": q, "resultados": d.get("total_count"), "primeros": [{"repo": x["full_name"], "descripcion": (x.get("description") or "")[:160]} for x in d.get("items", [])[:10]]})  # Guarda
    print("GitHub", q, d.get("total_count")); time.sleep(7)  # Pausa por límite sin autenticación
for q in ["retrieval forecasting residual", "load forecasting retrieval", "analog ensemble"]:  # Consultas en Hugging Face Hub
    for tipo in ["models", "datasets"]:  # Modelos y conjuntos
        try:  # Petición
            d = json.loads(urllib.request.urlopen(urllib.request.Request(f"https://huggingface.co/api/{tipo}?" + urllib.parse.urlencode({"search": q, "limit": 20}), headers=UA), timeout=60).read())  # Lista
        except Exception:  # Error
            d = []  # Vacío
        res["repositorios"].append({"fuente": f"Hugging Face ({tipo})", "consulta": q, "resultados": len(d), "primeros": [x.get("id") for x in d[:10]]})  # Guarda
        print("HF", tipo, q, len(d))  # Avance
with open(f"snowball_repos_{res['fecha'].replace('-', '')}.json", "w", encoding="utf-8") as fh:  # Archivo
    json.dump(res, fh, ensure_ascii=False, indent=1)  # Escribe

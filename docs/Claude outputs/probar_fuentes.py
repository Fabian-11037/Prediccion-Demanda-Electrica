"""Prueba de acceso y de fechas de las fuentes de texto (correr en el equipo local).

Uso:  pip install requests beautifulsoup4
      python probar_fuentes.py            ->  prueba_fuentes.csv  y  prueba_fuentes.json

No descarga nada masivo: 1 solicitud por prueba, con pausa y user agent identificable.
Para cada fuente registra: estado HTTP, si hay fecha visible, si hay hora, si hay fecha
de modificación y la primera captura de Wayback (CDX), que es la prueba de disponibilidad.
"""
import csv, json, re, time
import requests
from bs4 import BeautifulSoup

UA = {"User-Agent": "Investigacion-academica-Uninorte (contacto: fajose97@gmail.com)"}
PAUSA = 2.0
MESES = "enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre"
RE_FECHA = re.compile(rf"\b(\d{{1,2}}) de ({MESES}) de (\d{{4}})\b", re.I)
RE_ISO = re.compile(r"\b\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?(?:Z|[+-]\d{2}:?\d{2})?)?\b")

PRUEBAS = {
    "creg_listado": "https://gestornormativo.creg.gov.co/gestor/entorno/novedades_circulares.html",
    "creg_documento": "https://gestornormativo.creg.gov.co/gestor/entorno/docs/circular_creg_0223_2025.htm",
    "upme_pdf": "https://docs.upme.gov.co/SalaPrensa/ComunicadosPrensa/Comunicado_65_UPME_Gobierno_acaba_proyectos_de_papel_12-11-2025.pdf",
    "minenergia_listado": "https://www.minenergia.gov.co/es/sala-de-prensa/noticias-index/",
    "minenergia_nota": "https://www.minenergia.gov.co/es/sala-de-prensa/noticias-index/gobierno-acaba-con-proyectos-de-papel-y-libera-mas-de-cinco-mil-megavatios-de-la-red-para-impulsar-las-energias-limpias-en-colombia/",
    "xm_portal": "https://www.xm.com.co",
    "ideam_portal": "https://www.ideam.gov.co",
    "gdelt_doc": "https://api.gdeltproject.org/api/v2/doc/doc?query=energia%20sourcecountry:colombia&mode=artlist&maxrecords=5&format=json&timespan=1month",
}


def cdx_primera_captura(url):
    q = {"url": url, "output": "json", "limit": 1, "fl": "timestamp,digest,statuscode", "filter": "statuscode:200"}
    r = requests.get("https://web.archive.org/cdx/search/cdx", params=q, headers=UA, timeout=60)
    filas = r.json() if r.ok and r.text.strip() else []
    return filas[1] if len(filas) > 1 else None   # filas[0] es el encabezado


def analizar(nombre, url):
    t0 = time.time()
    out = {"fuente": nombre, "url": url}
    try:
        r = requests.get(url, headers=UA, timeout=60)
        out.update(http=r.status_code, bytes=len(r.content), seg=round(time.time() - t0, 2),
                   content_type=r.headers.get("content-type", ""), last_modified_header=r.headers.get("last-modified", ""))
        if r.ok and "html" in out["content_type"]:
            s = BeautifulSoup(r.text, "html.parser")
            meta = {m.get("property") or m.get("name"): m.get("content") for m in s.find_all("meta")}
            txt = s.get_text(" ", strip=True)
            out["meta_fechas"] = {k: v for k, v in meta.items() if k and re.search("date|time|modified|published", k, re.I)}
            out["fechas_texto"] = sorted({m.group(0) for m in RE_FECHA.finditer(txt)})[:5]
            out["iso_en_html"] = sorted(set(RE_ISO.findall(r.text)))[:5]
            out["tiene_hora"] = any("T" in x or ":" in x for x in out["iso_en_html"])
            out["enlaces"] = len(s.find_all("a"))
        elif r.ok:
            out["nota"] = "no HTML (PDF/JSON): revisar con pdftotext o json"
            if "json" in out["content_type"] or url.endswith("json"):
                out["muestra"] = r.text[:400]
    except Exception as e:  # noqa: BLE001
        out["error"] = repr(e)
    time.sleep(PAUSA)
    try:
        cap = cdx_primera_captura(url)
        out["wayback_primera_captura"] = cap[0] if cap else None
        out["wayback_digest"] = cap[1] if cap else None
    except Exception as e:  # noqa: BLE001
        out["wayback_error"] = repr(e)
    time.sleep(PAUSA)
    return out


if __name__ == "__main__":
    res = [analizar(n, u) for n, u in PRUEBAS.items()]
    json.dump(res, open("prueba_fuentes.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    cols = ["fuente", "http", "bytes", "tiene_hora", "wayback_primera_captura", "last_modified_header", "error"]
    with open("prueba_fuentes.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader(); w.writerows(res)
    print(open("prueba_fuentes.csv", encoding="utf-8").read())

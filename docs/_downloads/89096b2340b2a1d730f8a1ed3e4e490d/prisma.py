"""Une exportaciones de Scopus, WoS e IEEE, prepara el cribado de dos revisores y dibuja el diagrama PRISMA 2020."""  # Propósito del módulo
import argparse  # Lee los subcomandos de la línea de comandos
import json  # Guarda los conteos en un archivo legible
import re  # Normaliza títulos y DOI con expresiones regulares
import sys  # Permite terminar con un mensaje de error claro
from pathlib import Path  # Maneja rutas sin depender del sistema operativo

import pandas as pd  # Lee, une y escribe las tablas de registros

AQUI = Path(__file__).resolve().parent  # Carpeta donde vive este script
EXPORT = AQUI / "exportaciones"  # Carpeta donde se dejan los archivos descargados
CRIBADO = AQUI / "cribado.csv"  # Hoja de cribado que llenan los revisores
SALIDA = AQUI / "resultados"  # Carpeta de conteos y figuras
ANIO_MIN = 2018  # Primer año del periodo elegible del protocolo

MOTIVOS = {  # Texto de cada código de exclusión del protocolo
    "E1": "otro dominio u otra variable",  # Código E1
    "E2": "horizonte mediano o largo",  # Código E2
    "E3": "sin modelo neuronal",  # Código E3
    "E4": "sin comparación o sin métricas",  # Código E4
    "E5": "revisión o sin resultados primarios",  # Código E5
    "E6": "texto completo no recuperado",  # Código E6
    "E7": "retractado",  # Código E7
}  # Fin del diccionario de motivos

COLUMNAS_REVISION = [  # Columnas que llenan los revisores, en orden
    "titulo_A", "motivo_titulo_A", "titulo_B", "motivo_titulo_B",  # Cribado por título y resumen de cada revisor
    "consenso_titulo", "motivo_titulo",  # Decisión de consenso de la primera fase
    "recuperado",  # Si se obtuvo el texto completo (si/no)
    "tc_A", "tc_B", "consenso_tc", "motivo_tc",  # Evaluación de texto completo y consenso
]  # Fin de las columnas de revisión


def leer_texto(ruta: Path, sep: str) -> pd.DataFrame:  # Lee un archivo de texto probando codificaciones habituales
    for cod in ("utf-8-sig", "utf-16", "latin-1"):  # Scopus usa UTF-8; WoS antiguo usa UTF-16
        try:  # Intenta con la codificación actual
            return pd.read_csv(ruta, sep=sep, dtype=str, encoding=cod, quoting=0 if sep == "," else 3, index_col=False)  # WoS no usa comillas
        except (UnicodeError, pd.errors.ParserError):  # Si falla, prueba la siguiente
            continue  # Pasa a la siguiente codificación
    sys.exit(f"No pude leer {ruta.name}")  # Ninguna codificación funcionó


def normalizar_doi(doi) -> str:  # Deja el DOI en minúsculas y sin prefijo de URL
    if not isinstance(doi, str):  # Celdas vacías llegan como NaN
        return ""  # Sin DOI
    return re.sub(r"^https?://(dx\.)?doi\.org/", "", doi.strip().lower())  # Quita el prefijo y espacios


def normalizar_titulo(titulo) -> str:  # Reduce el título a letras y números para comparar
    if not isinstance(titulo, str):  # Título ausente
        return ""  # Devuelve cadena vacía
    return re.sub(r"[^a-z0-9]", "", titulo.lower())  # Quita espacios, tildes no ASCII y puntuación


def estandarizar(df: pd.DataFrame, mapa: dict, base: str) -> pd.DataFrame:  # Renombra columnas al esquema común
    faltan = [c for c in mapa if c not in df.columns]  # Columnas esperadas que no están en el archivo
    if faltan:  # La exportación no trae lo necesario
        sys.exit(f"La exportación de {base} no tiene las columnas {faltan}; revisa los campos exportados")  # Explica qué falta
    out = df[list(mapa)].rename(columns=mapa)  # Toma y renombra las columnas útiles
    out["fuente"] = base  # Anota la base de origen
    return out  # Devuelve la tabla estandarizada


def cargar_exportaciones() -> pd.DataFrame:  # Lee todas las exportaciones de la carpeta
    partes = []  # Acumula una tabla por archivo
    for ruta in sorted(EXPORT.glob("*")):  # Recorre los archivos en orden alfabético
        nombre = ruta.name.lower()  # Nombre en minúsculas para reconocer la base
        if nombre.startswith("scopus") and nombre.endswith(".csv"):  # Exportación CSV de Scopus
            df = estandarizar(leer_texto(ruta, ","), {"Title": "titulo", "Year": "anio", "DOI": "doi", "Abstract": "resumen", "Document Type": "tipo", "Authors": "autores"}, "Scopus")  # Campos de Scopus
        elif (nombre.startswith("savedrecs") or nombre.startswith("web of science")) and nombre.endswith(".txt"):  # Exportación tabulada de WoS
            df = estandarizar(leer_texto(ruta, "\t"), {"TI": "titulo", "PY": "anio", "DI": "doi", "AB": "resumen", "DT": "tipo", "AU": "autores"}, "Web of Science")  # Campos de WoS
        elif (nombre.startswith("savedrecs") or nombre.startswith("web of science")) and nombre.endswith(".xls"):  # Exportación Excel de WoS (requiere xlrd)
            df = estandarizar(pd.read_excel(ruta, dtype=str), {"Article Title": "titulo", "Publication Year": "anio", "DOI": "doi", "Abstract": "resumen", "Document Type": "tipo", "Authors": "autores"}, "Web of Science")  # Campos de WoS en Excel
        elif nombre.startswith("iee") and nombre.endswith(".csv"):  # Exportación CSV de IEEE Xplore
            df = estandarizar(leer_texto(ruta, ","), {"Document Title": "titulo", "Publication Year": "anio", "DOI": "doi", "Abstract": "resumen", "Authors": "autores"}, "IEEE Xplore")  # Campos de IEEE
            df["tipo"] = ""  # IEEE no trae el tipo en la misma columna
        elif nombre == "otras_fuentes.csv":  # Registros hallados por otros métodos
            df = estandarizar(leer_texto(ruta, ","), {"titulo": "titulo", "anio": "anio", "doi": "doi"}, "Otros métodos")  # Esquema mínimo propio
            df["resumen"] = ""  # Sin resumen exportado
            df["tipo"] = ""  # Sin tipo exportado
            df["autores"] = ""  # Sin autores exportados
        else:  # Archivo que no corresponde a ninguna base
            continue  # Se ignora
        df["archivo"] = ruta.name  # Guarda el archivo de origen para trazabilidad
        partes.append(df)  # Añade la tabla leída
    if not partes:  # La carpeta está vacía
        sys.exit(f"No hay exportaciones en {EXPORT}")  # Indica dónde dejarlas
    return pd.concat(partes, ignore_index=True)  # Une todas las tablas


def clave(fila) -> str:  # Clave de duplicado: DOI si existe, si no título y año
    doi = normalizar_doi(fila["doi"])  # DOI normalizado
    return "doi:" + doi if doi else "ti:" + normalizar_titulo(fila["titulo"]) + ":" + str(fila["anio"])  # Clave final


def unir() -> None:  # Subcomando que crea cribado.csv
    if CRIBADO.exists():  # Evita borrar decisiones ya tomadas
        sys.exit("cribado.csv ya existe; muévelo o bórralo a mano si quieres regenerarlo")  # Protege el trabajo de los revisores
    todos = cargar_exportaciones()  # Lee todas las bases
    anio = pd.to_numeric(todos["anio"], errors="coerce")  # Año como número
    fuera = todos[(todos["fuente"] != "Otros métodos") & (anio < ANIO_MIN)]  # Registros de bases anteriores al periodo
    fuera_por_base = fuera.groupby("fuente").size().to_dict()  # Cuántos quedan fuera en cada base
    por_base_total = todos[todos["fuente"] != "Otros métodos"].groupby("fuente").size().to_dict()  # Registros exportados por base
    todos = todos.drop(fuera.index)  # Retira los registros fuera del periodo antes del cribado
    todos["clave"] = todos.apply(clave, axis=1)  # Calcula la clave de cada registro
    todos["via"] = todos["fuente"].map(lambda f: "otros" if f == "Otros métodos" else "bases")  # Columna del diagrama a la que pertenece
    todos = todos.sort_values("via")  # Las bases van primero para que ganen en los duplicados
    unicos = todos.groupby("clave", sort=False).agg(  # Un registro por clave
        via=("via", "first"),  # Vía del primer registro (bases si aparece en alguna)
        fuentes=("fuente", lambda s: "; ".join(sorted(set(s)))),  # Todas las bases donde aparece
        anio=("anio", "first"),  # Año del primer registro
        titulo=("titulo", "first"),  # Título del primer registro
        doi=("doi", "first"),  # DOI del primer registro
        tipo=("tipo", "first"),  # Tipo de documento
        autores=("autores", "first"),  # Autores
        resumen=("resumen", "first"),  # Resumen para el cribado
    ).reset_index(drop=True)  # Quita la clave como índice
    unicos.insert(0, "id", [f"R{i:04d}" for i in range(1, len(unicos) + 1)])  # Identificador estable de cada registro
    for col in COLUMNAS_REVISION:  # Agrega las columnas vacías de los revisores
        unicos[col] = ""  # Celda vacía para llenar a mano
    unicos.to_csv(CRIBADO, index=False, encoding="utf-8-sig")  # UTF-8 con BOM para que Excel muestre las tildes
    otros = int((todos["via"] == "otros").sum())  # Registros de otros métodos
    SALIDA.mkdir(exist_ok=True)  # Crea la carpeta de resultados
    ident = {"por_base": por_base_total, "fuera_periodo": fuera_por_base, "otros_metodos": otros, "unicos_bases": int((unicos["via"] == "bases").sum()), "unicos": len(unicos)}  # Conteos de identificación
    (SALIDA / "identificacion.json").write_text(json.dumps(ident, ensure_ascii=False, indent=2), encoding="utf-8")  # Guarda los conteos de identificación
    print(json.dumps(ident, ensure_ascii=False, indent=2))  # Resumen en pantalla
    print(f"Hoja de cribado: {CRIBADO}")  # Indica qué archivo deben llenar


def kappa(a: pd.Series, b: pd.Series) -> float:  # Kappa de Cohen entre dos revisores
    a, b = a.str.strip().str.lower(), b.str.strip().str.lower()  # Normaliza las decisiones
    po = float((a == b).mean())  # Acuerdo observado
    cats = set(a) | set(b)  # Categorías usadas por cualquiera
    pe = sum(float((a == c).mean()) * float((b == c).mean()) for c in cats)  # Acuerdo esperado por azar
    return float("nan") if pe == 1 else (po - pe) / (1 - pe)  # Fórmula de Cohen


def exigir(df: pd.DataFrame, cols: list, etapa: str) -> None:  # Verifica que no haya celdas vacías
    vacias = df[df[cols].isna().any(axis=1) | (df[cols].astype(str).apply(lambda s: s.str.strip()) == "").any(axis=1)]  # Filas incompletas
    if len(vacias):  # Hay decisiones pendientes
        sys.exit(f"Faltan decisiones de {etapa} en {len(vacias)} registros, por ejemplo {list(vacias['id'][:5])}")  # Detiene con ejemplos


def contar_motivos(serie: pd.Series) -> dict:  # Cuenta exclusiones por código
    return serie.str.strip().str.upper().value_counts().sort_index().to_dict()  # Diccionario código → número


def diagrama() -> None:  # Subcomando que cuenta etapas y dibuja el PRISMA
    if not CRIBADO.exists():  # Aún no se ha corrido unir
        sys.exit("No existe cribado.csv; corre primero: py -3.14 prisma.py unir")  # Indica el paso previo
    ident = json.loads((SALIDA / "identificacion.json").read_text(encoding="utf-8"))  # Conteos de identificación
    df = pd.read_csv(CRIBADO, dtype=str, encoding="utf-8-sig").fillna("")  # Hoja llenada por los revisores
    bases = df[df["via"] == "bases"].copy()  # Registros de bases de datos
    otros = df[df["via"] == "otros"].copy()  # Registros de otros métodos
    exigir(bases, ["titulo_A", "titulo_B", "consenso_titulo"], "título y resumen")  # Primera fase completa
    excl_ti = bases[bases["consenso_titulo"].str.lower() == "excluir"]  # Excluidos por título y resumen
    exigir(excl_ti, ["motivo_titulo"], "motivo de título")  # Cada exclusión necesita motivo
    pasan = pd.concat([bases[bases["consenso_titulo"].str.lower() == "incluir"], otros])  # Buscados para recuperación
    exigir(pasan, ["recuperado"], "recuperación")  # Cada uno debe decir si se recuperó
    no_rec = pasan[pasan["recuperado"].str.lower() == "no"]  # No recuperados
    evaluados = pasan[pasan["recuperado"].str.lower() == "si"]  # Evaluados en texto completo
    exigir(evaluados, ["tc_A", "tc_B", "consenso_tc"], "texto completo")  # Segunda fase completa
    excl_tc = evaluados[evaluados["consenso_tc"].str.lower() == "excluir"]  # Excluidos en texto completo
    exigir(excl_tc, ["motivo_tc"], "motivo de texto completo")  # Cada exclusión necesita motivo
    incluidos = evaluados[evaluados["consenso_tc"].str.lower() == "incluir"]  # Estudios incluidos

    def columna(p: str) -> dict:  # Conteos de una vía ("bases" u "otros")
        return {  # Diccionario de conteos
            "buscados": int(len(pasan[pasan["via"] == p])),  # Buscados para recuperación
            "no_recuperados": int(len(no_rec[no_rec["via"] == p])),  # No recuperados
            "evaluados": int(len(evaluados[evaluados["via"] == p])),  # Evaluados en texto completo
            "excluidos_tc": contar_motivos(excl_tc[excl_tc["via"] == p]["motivo_tc"]),  # Exclusiones por motivo
        }  # Fin del diccionario

    total_bases = sum(ident["por_base"].values())  # Registros identificados en bases
    c = {  # Conteos completos del flujo
        "identificados_por_base": ident["por_base"],  # Por base
        "fuera_periodo": sum(ident["fuera_periodo"].values()),  # Anteriores a 2018 retirados antes del cribado
        "duplicados_eliminados": total_bases - sum(ident["fuera_periodo"].values()) - len(bases),  # Duplicados entre y dentro de bases
        "cribados": len(bases),  # Registros cribados por título y resumen
        "excluidos_titulo": contar_motivos(excl_ti["motivo_titulo"]),  # Excluidos en la primera fase por motivo
        "bases": columna("bases"),  # Flujo de bases
        "otros": {"identificados": ident["otros_metodos"], "ya_en_bases": ident["otros_metodos"] - len(otros), **columna("otros")},  # Flujo de otros métodos
        "incluidos": len(incluidos),  # Estudios incluidos en la síntesis
        "kappa_titulo": round(kappa(bases["titulo_A"], bases["titulo_B"]), 3),  # Acuerdo en la primera fase
        "kappa_texto_completo": round(kappa(evaluados["tc_A"], evaluados["tc_B"]), 3),  # Acuerdo en la segunda fase
    }  # Fin de los conteos
    (SALIDA / "conteos_prisma.json").write_text(json.dumps(c, ensure_ascii=False, indent=2), encoding="utf-8")  # Guarda los conteos
    incluidos[["id", "fuentes", "anio", "titulo", "doi"]].to_csv(SALIDA / "incluidos.csv", index=False, encoding="utf-8-sig")  # Lista de incluidos
    dibujar(c)  # Dibuja el diagrama
    print(json.dumps(c, ensure_ascii=False, indent=2))  # Muestra los conteos


def motivos_texto(d: dict) -> str:  # Convierte el diccionario de motivos en líneas legibles
    return "\n".join(f"  {k} · {MOTIVOS.get(k, k)}: {v}" for k, v in d.items())  # Una línea por motivo


def dibujar(c: dict) -> None:  # Dibuja el diagrama PRISMA 2020 con matplotlib
    import matplotlib  # Biblioteca de gráficos
    matplotlib.use("Agg")  # Dibuja sin ventana
    import matplotlib.pyplot as plt  # Interfaz de figuras
    from matplotlib.patches import FancyBboxPatch  # Cajas con esquinas redondeadas

    fig, ax = plt.subplots(figsize=(14, 11))  # Lienzo amplio
    ax.set_xlim(0, 100)  # Coordenadas horizontales de 0 a 100
    ax.set_ylim(0, 100)  # Coordenadas verticales de 0 a 100
    ax.axis("off")  # Sin ejes

    def caja(x, y, w, h, texto, color="#ffffff"):  # Dibuja una caja con texto centrado
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4", fc=color, ec="#333333", lw=1))  # Rectángulo
        ax.text(x + w / 2, y + h / 2, texto, ha="center", va="center", fontsize=9, wrap=True)  # Texto centrado

    def flecha(x1, y1, x2, y2):  # Dibuja una flecha entre dos puntos
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", color="#333333", lw=1))  # Flecha simple

    b, o = c["bases"], c["otros"]  # Atajos para las dos vías
    por_base = "\n".join(f"  {k} (n = {v})" for k, v in c["identificados_por_base"].items())  # Conteo por base
    ax.text(2, 96, "Identificación de estudios mediante bases de datos", fontsize=11, weight="bold")  # Título de la vía principal
    ax.text(70, 96, "Otros métodos", fontsize=11, weight="bold")  # Título de la vía secundaria
    for y, etiqueta in [(80, "Identificación"), (50, "Cribado"), (8, "Incluidos")]:  # Etiquetas de fase a la izquierda
        ax.text(0.5, y, etiqueta, rotation=90, va="center", fontsize=10, weight="bold", color="#1f4e79")  # Texto vertical
    caja(4, 76, 28, 16, f"Registros identificados:\n{por_base}")  # Identificados por base
    caja(38, 76, 26, 16, f"Registros eliminados antes del cribado:\n  Duplicados (n = {c['duplicados_eliminados']})\n  Anteriores a {ANIO_MIN} (n = {c['fuera_periodo']})")  # Duplicados
    flecha(32, 84, 38, 84)  # Identificados → eliminados
    caja(4, 60, 28, 8, f"Registros cribados\n(n = {c['cribados']})")  # Cribados
    flecha(18, 76, 18, 68)  # Identificados → cribados
    caja(38, 56, 26, 16, f"Registros excluidos\n(n = {sum(c['excluidos_titulo'].values())}):\n{motivos_texto(c['excluidos_titulo'])}")  # Excluidos por título
    flecha(32, 64, 38, 64)  # Cribados → excluidos
    caja(4, 44, 28, 8, f"Publicaciones buscadas para recuperación\n(n = {b['buscados']})")  # Buscadas
    flecha(18, 60, 18, 52)  # Cribados → buscadas
    caja(38, 44, 26, 8, f"No recuperadas\n(n = {b['no_recuperados']})")  # No recuperadas
    flecha(32, 48, 38, 48)  # Buscadas → no recuperadas
    caja(4, 28, 28, 8, f"Publicaciones evaluadas en texto completo\n(n = {b['evaluados']})")  # Evaluadas
    flecha(18, 44, 18, 36)  # Buscadas → evaluadas
    caja(38, 22, 26, 18, f"Publicaciones excluidas\n(n = {sum(b['excluidos_tc'].values())}):\n{motivos_texto(b['excluidos_tc'])}")  # Excluidas en texto completo
    flecha(32, 32, 38, 32)  # Evaluadas → excluidas
    caja(70, 76, 26, 16, f"Registros identificados por\nbúsqueda pública y referencias\n(n = {o['identificados']})\nYa identificados en bases (n = {o['ya_en_bases']})", "#f3f3f3")  # Otros métodos
    caja(70, 44, 26, 8, f"Buscadas para recuperación (n = {o['buscados']})\nNo recuperadas (n = {o['no_recuperados']})", "#f3f3f3")  # Recuperación de otros
    flecha(83, 76, 83, 52)  # Otros → recuperación
    caja(70, 22, 26, 18, f"Evaluadas (n = {o['evaluados']})\nExcluidas (n = {sum(o['excluidos_tc'].values())}):\n{motivos_texto(o['excluidos_tc'])}", "#f3f3f3")  # Evaluación de otros
    flecha(83, 44, 83, 40)  # Recuperación → evaluación
    caja(4, 4, 28, 10, f"Estudios incluidos en la síntesis\n(n = {c['incluidos']})", "#e8f0e0")  # Incluidos
    flecha(18, 28, 18, 14)  # Evaluadas → incluidos
    flecha(83, 22, 32, 10)  # Otros evaluados → incluidos
    ax.text(50, 1, f"Kappa de Cohen: título y resumen {c['kappa_titulo']}; texto completo {c['kappa_texto_completo']}", ha="center", fontsize=9, color="#555555")  # Acuerdo entre revisores
    for ext in ("png", "svg"):  # Guarda en mapa de bits y vectorial
        fig.savefig(SALIDA / f"prisma_2020.{ext}", dpi=200, bbox_inches="tight")  # Escribe el archivo
    plt.close(fig)  # Libera memoria
    print(f"Diagrama: {SALIDA / 'prisma_2020.png'}")  # Indica dónde quedó


if __name__ == "__main__":  # Punto de entrada al ejecutar el script
    p = argparse.ArgumentParser(description=__doc__)  # Analizador de argumentos
    p.add_argument("accion", choices=["unir", "diagrama"])  # Subcomando requerido
    a = p.parse_args()  # Lee la línea de comandos
    unir() if a.accion == "unir" else diagrama()  # Ejecuta la acción pedida

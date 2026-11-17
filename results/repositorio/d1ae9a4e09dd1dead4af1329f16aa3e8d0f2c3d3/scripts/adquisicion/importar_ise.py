"""Importa el ISE del DANE (xlsx descargado) a data/external/actividad_economica/ise_dane.csv.

SIN PROBAR con el archivo real (el contenedor de desarrollo no alcanza dane.gov.co).
El parser busca, en todas las hojas, filas cuya primera celda sea una fecha o texto
«mes año» y toma las tres primeras columnas numéricas como original, desestacionalizada
y tendencia-ciclo. Debe revisarse a mano contra el Excel antes de usar los datos.

Descarga manual: sección «Indicador de Seguimiento a la Economía (ISE)» del DANE
(cuentas nacionales), archivo de anexo de series. Guardar la fecha de descarga: es el
``vintage`` (el DANE revisa las cifras históricas en cada publicación).

Uso:
    python scripts/adquisicion/importar_ise.py ruta/anex-ISE-....xlsx --vintage 2026-10-19
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from demanda import config as C  # noqa: E402

MESES = {"enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6, "julio": 7, "agosto": 8,
         "septiembre": 9, "setiembre": 9, "octubre": 10, "noviembre": 11, "diciembre": 12}


def _a_periodo(x):
    if isinstance(x, (pd.Timestamp,)) or hasattr(x, "year") and hasattr(x, "month"):
        return pd.Period(year=x.year, month=x.month, freq="M")
    if isinstance(x, str):
        m = re.match(r"^\s*([A-Za-záéíóú]+)\.?\s*(?:de\s*)?(\d{4})\s*(?:\(?[a-z]{1,3}\)?)?\s*$", x.lower())
        if m and m.group(1) in MESES:
            return pd.Period(year=int(m.group(2)), month=MESES[m.group(1)], freq="M")
        m = re.match(r"^(\d{4})[-/](\d{1,2})$", x.strip())
        if m:
            return pd.Period(year=int(m.group(1)), month=int(m.group(2)), freq="M")
    return None


def extraer(df_hoja: pd.DataFrame) -> pd.DataFrame:
    filas = []
    for _, r in df_hoja.iterrows():
        per = _a_periodo(r.iloc[0])
        if per is None:
            continue
        nums = [v for v in r.iloc[1:] if isinstance(v, (int, float)) and pd.notna(v)]
        if len(nums) >= 3:
            filas.append((per, *nums[:3]))
    return pd.DataFrame(filas, columns=["mes", "ise_original", "ise_desestacionalizado", "ise_tendencia_ciclo"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xlsx")
    ap.add_argument("--vintage", required=True, help="fecha de descarga AAAA-MM-DD")
    ap.add_argument("--hoja", default=None, help="nombre de hoja; por defecto la primera con >= 12 meses")
    a = ap.parse_args()
    hojas = pd.read_excel(a.xlsx, sheet_name=None, header=None)
    cand = {k: extraer(v) for k, v in hojas.items()} if a.hoja is None else {a.hoja: extraer(hojas[a.hoja])}
    cand = {k: v for k, v in cand.items() if len(v) >= 12}
    if not cand:
        sys.exit("No se encontró ninguna hoja con >= 12 meses reconocibles; revisar el archivo")
    hoja, d = next(iter(cand.items()))
    d = d.drop_duplicates("mes", keep="last").sort_values("mes")
    d["mes"] = d["mes"].astype(str)
    d["vintage"] = a.vintage
    destino = C.EXTERNO / "actividad_economica" / "ise_dane.csv"
    d.to_csv(destino, index=False)
    print(f"Hoja usada: {hoja}; {len(d)} meses ({d.mes.iloc[0]} a {d.mes.iloc[-1]}) -> {destino}")
    print(d.tail(3).to_string(index=False))
    print("Revisar manualmente el orden de columnas (original / desestacionalizada / tendencia-ciclo).")


if __name__ == "__main__":
    main()

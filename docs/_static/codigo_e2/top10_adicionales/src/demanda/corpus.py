"""Corpus de prensa y documentos oficiales con disponibilidad por versión."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from . import config as C

RUTA=C.PROCESADO/'corpus_texto.parquet'
ESCENARIOS=('declarado','estricto')

def cargar_corpus(escenario: str, ruta: Path | None=None, fuentes: list[str] | None=None) -> pd.DataFrame:
    """Carga texto utilizable; los metadatos de auditoría no se concatenan al contenido."""
    if escenario not in ESCENARIOS:raise ValueError(f'Escenario válido: {ESCENARIOS}')
    t=pd.read_parquet(ruta or RUTA)
    t=t.loc[~t.cuarentena.fillna(True)].copy()
    if fuentes is not None:t=t.loc[t.fuente.isin(fuentes)].copy()
    columna='disponible_'+escenario
    t['disponible_desde']=pd.to_datetime(t[columna],utc=True,errors='coerce')
    t=t.loc[t.disponible_desde.notna() & t.titulo.fillna('').str.strip().ne('')].copy()
    if escenario=='estricto':
        t=t.loc[t.prueba_disponibilidad.isin(['wayback_payload_digest_and_capture','wayback_cdx_digest_matches_downloaded_pdf','download_payload_sha256_and_retrieval'])].copy()
        cdx=t.prueba_disponibilidad.eq('wayback_cdx_digest_matches_downloaded_pdf')
        if 'titulo_verificado' not in t:
            t=t.loc[~cdx].copy()
        if 'titulo_verificado' in t:
            t=t.loc[~cdx | t.titulo_verificado.fillna('').str.strip().ne('')].copy()
            t['titulo']=t.titulo_verificado.fillna(t.titulo)
    columnas=['id','fuente','tipo_fuente','familia_documental','titulo','texto','texto_alcance','disponible_desde']
    t=t[columnas].sort_values(['disponible_desde','id']).reset_index(drop=True)
    return t

def documentos_en_origen(documentos: pd.DataFrame, origen: pd.Timestamp, ventana_horas: int=24, max_n: int | None=None) -> pd.DataFrame:
    """Selecciona versiones disponibles en [origen−ventana, origen], en UTC.

    `documentos` debe proceder de cargar_corpus y conservar su orden temporal.
    Una fecha sin zona se interpreta en America/Bogota. No se arrastran textos
    anteriores a la ventana ni se incorporan documentos disponibles después del origen.
    """
    if ventana_horas<=0:raise ValueError('La ventana debe ser positiva')
    if max_n is not None and max_n<=0:raise ValueError('max_n debe ser positivo')
    o=pd.Timestamp(origen)
    if o.tzinfo is None:o=o.tz_localize(C.TZ)
    o=o.tz_convert('UTC')
    valores=pd.DatetimeIndex(documentos.disponible_desde).as_unit('ns')
    if not valores.is_monotonic_increasing:raise ValueError('El corpus debe estar ordenado por disponibilidad')
    hi=np.searchsorted(valores.asi8,o.value,side='right')
    lo=np.searchsorted(valores.asi8,(o-pd.Timedelta(hours=ventana_horas)).value,side='left')
    seleccion=documentos.iloc[lo:hi]
    return seleccion.tail(max_n) if max_n is not None else seleccion

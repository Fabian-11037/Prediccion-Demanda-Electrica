"""Selectores de datos descargados; no modifican el pipeline entrenado."""
from __future__ import annotations
import numpy as np
import pandas as pd

TZ='America/Bogota'


def _origen(fecha):
    o=pd.Timestamp(fecha)
    return o.tz_localize(TZ) if o.tzinfo is None else o.tz_convert(TZ)


def preparar_pronosticos(d,modelo):
    """Selecciona un proveedor explícito e indexa por validez, sin promediar modelos."""
    if not modelo or modelo not in set(d.modelo):
        raise ValueError('Debe seleccionarse un modelo meteorológico existente')
    p=d.loc[d.modelo.eq(modelo)].copy()
    if p.duplicated(['valido','ciudad','lead_dias']).any():
        raise ValueError('Hay versiones duplicadas para una misma clave meteorológica')
    return p.sort_values(['valido','ciudad','lead_dias']).set_index('valido',drop=False)


def meteorologia_en_origen(preparados,origen,ciudades,variables,horizonte=24,
                           aceptar_disponibilidad_estimada=False):
    """Último adelanto utilizable completo por hora/ciudad; ausencia queda como NaN.

    Previous Runs no identifica la inicialización exacta. Activar explícitamente
    la regla proxy exige declararla en el protocolo; no certifica publicación real.
    """
    if not aceptar_disponibilidad_estimada:
        raise ValueError('La emisión de Previous Runs es estimada; debe declararse su uso')
    o=_origen(origen)
    fechas=pd.date_range(o,periods=horizonte,freq='h')
    p=preparados.loc[o:fechas[-1]].copy()
    p=p.loc[p.disponible_estimado.le(o) & p[variables].notna().all(axis=1)]
    p=p.sort_values(['disponible_estimado','lead_dias'],ascending=[False,True])
    p=p.drop_duplicates(['valido','ciudad'],keep='first')
    indice=pd.MultiIndex.from_product([fechas,ciudades],names=['valido','ciudad'])
    p=p.reset_index(drop=True).set_index(['valido','ciudad']).reindex(indice)
    valores=p[variables].to_numpy(dtype='float32').reshape(horizonte,len(ciudades),len(variables))
    mascara=np.isfinite(valores).all(axis=2)
    leads=p.lead_dias.fillna(-1).to_numpy(dtype='int16').reshape(horizonte,len(ciudades))
    return dict(valores=valores,mascara=mascara,lead_dias=leads,
                auditoria=p[['disponible_estimado','emision_estimada']])


def ise_en_origen(versiones,origen,base_indice='2015',rezago_dias=60):
    """Último mes de la base elegida y última edición declarada disponible.

    Excluye fechas de publicación desconocidas y conserva la advertencia de
    versión histórica no verificada. No encadena bases distintas del índice.
    """
    o=_origen(origen)
    p=versiones.loc[versiones.base_indice.astype(str).eq(str(base_indice))].copy()
    pub=pd.to_datetime(p.vintage_publicacion,errors='coerce').dt.tz_localize(TZ)
    referencia=pd.to_datetime(p.mes+'-01').dt.tz_localize(TZ)+pd.offsets.MonthEnd(0)
    p['disponible_declarado']=pub+pd.Timedelta(days=1)
    p=p.loc[pub.notna() & p.disponible_declarado.le(o) &
            (referencia+pd.Timedelta(days=rezago_dias)).le(o) & p.ise_original.notna()]
    if p.empty:return None
    return p.sort_values(['mes','disponible_declarado','archivo_fuente']).iloc[-1].to_dict()

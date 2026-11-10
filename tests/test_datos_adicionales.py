import numpy as np
import pandas as pd
import pytest
from demanda.datos_adicionales import preparar_pronosticos,meteorologia_en_origen,ise_en_origen


def test_no_selecciona_un_pronostico_posterior_al_origen_y_conserva_ausencias():
    o=pd.Timestamp('2025-01-02',tz='America/Bogota')
    d=pd.DataFrame([dict(modelo='gfs_global',valido=o,ciudad='bogota',lead_dias=n,
        temperature_2m_c=v,emision_estimada=o-pd.Timedelta(days=n),disponible_estimado=fecha)
        for n,v,fecha in [(1,99.,o+pd.Timedelta(hours=1)),(2,20.,o-pd.Timedelta(hours=1))]])
    p=preparar_pronosticos(d,'gfs_global')
    with pytest.raises(ValueError):meteorologia_en_origen(p,o,['bogota'],['temperature_2m_c'])
    x=meteorologia_en_origen(p,o,['bogota'],['temperature_2m_c'],aceptar_disponibilidad_estimada=True)
    assert x['valores'][0,0,0]==20 and x['lead_dias'][0,0]==2
    assert x['mascara'].sum()==1 and np.isnan(x['valores'][1:,0,0]).all()
    with pytest.raises(ValueError):preparar_pronosticos(d,'otro')


def test_rechaza_duplicados_antes_de_formar_una_ventana():
    d=pd.DataFrame(dict(modelo=['gfs_global']*2,valido=[pd.Timestamp('2025-01-01')]*2,
        ciudad=['bogota']*2,lead_dias=[1]*2))
    with pytest.raises(ValueError):preparar_pronosticos(d,'gfs_global')


def test_ise_no_usa_una_revision_futura_ni_una_base_distinta():
    filas=[dict(mes='2024-01',ise_original=v,base_indice=b,vintage_publicacion=f,
        archivo_fuente=str(i),version_historica_no_verificada=True)
        for i,(v,b,f) in enumerate([(100.,'2015','2024-03-15'),(999.,'2015','2024-09-01'),(9.,'2005','2024-03-01')])]
    d=pd.DataFrame(filas)
    assert ise_en_origen(d,'2024-03-01') is None
    r=ise_en_origen(d,'2024-04-01')
    assert r['ise_original']==100. and r['version_historica_no_verificada']

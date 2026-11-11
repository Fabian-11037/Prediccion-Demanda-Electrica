import pytest
from demanda.seleccion_escalonada import clasificar

def test_filtro_y_confirmacion_usan_anios_distintos_y_cobertura_completa():
    cc=[{'id':'A'},{'id':'B'}];rows=[]
    for year in (2018,2019,2020,2021):
        for seed in (43,44,45):
            rows.extend([dict(configuracion=cc[0],fold=year,semilla=seed,MASE=.1 if year==2018 else 2.),dict(configuracion=cc[1],fold=year,semilla=seed,MASE=.2 if year==2018 else 1.)])
    assert clasificar(rows,cc,[2018],[43,44,45])[0][1]['id']=='A'
    assert clasificar(rows,cc,[2019,2020,2021],[43,44,45])[0][1]['id']=='B'
    with pytest.raises(ValueError):clasificar(rows[:-1],cc,[2019,2020,2021],[43,44,45])

"""Contratos de las perturbaciones temporales, sin cargar ni ajustar modelos."""
from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location('importancia_variables', Path(__file__).resolve().parents[1]/'scripts/importancia_variables.py')
I = importlib.util.module_from_spec(spec)
spec.loader.exec_module(I)


def entradas():
    cols = sum(I.GRUPOS_CAL.values(), [])
    x = np.arange(3*168*24, dtype='float32').reshape(3,168,24)
    c = np.arange(3*24*23, dtype='float32').reshape(3,24,23)
    return x,c,np.array([10.,20.,30.]),np.array([2,0,1]),cols


def test_donantes_conservan_anos_orden_interno_y_distancia_temporal():
    fechas = pd.date_range('2022-01-01','2025-12-31',freq='D',tz='America/Bogota')
    p = I.indices_donantes(fechas,0)
    assert np.array_equal(np.sort(p),np.arange(len(fechas)))
    for anio in sorted(set(fechas.year)):
        ix=np.flatnonzero(fechas.year==anio)
        assert np.all(fechas.year[p[ix]]==anio)
        assert np.all(np.mod(np.diff(p[ix]),len(ix))==1)
        distancia=np.abs(p[ix]-ix)
        assert np.all(np.minimum(distancia,len(ix)-distancia)>=28)
    assert np.array_equal(p,I.indices_donantes(fechas,0))
    assert not np.array_equal(p,I.indices_donantes(fechas,1))


def test_bloque_de_rezagos_no_altera_otros_dias_calendario_ni_nivel():
    x,c,n,p,cols=entradas();original=x.copy()
    xx,cc,nn=I.perturbar(x,c,n,p,'dia_2',cols)
    assert np.array_equal(x,original)
    assert np.array_equal(xx[:,:120,:],x[:,:120,:])
    assert np.array_equal(xx[:,144:,:],x[:,144:,:])
    assert np.array_equal(xx[:,120:144,1:],x[:,120:144,1:])
    assert np.array_equal(xx[:,120:144,0],x[p,120:144,0])
    assert np.array_equal(cc,c) and np.array_equal(nn,n)


def test_calendario_se_transfiere_conjuntamente_en_pasado_y_horizonte():
    x,c,n,p,cols=entradas();original=x.copy();original_c=c.copy()
    xx,cc,nn=I.perturbar(x,c,n,p,'calendario_completo',cols)
    assert np.array_equal(xx[:,:,0],x[:,:,0])
    assert np.array_equal(xx[:,:,1:],x[p,:,1:])
    assert np.array_equal(cc,c[p]) and np.array_equal(nn,n)
    assert np.array_equal(x,original) and np.array_equal(c,original_c)


def test_nivel_es_independiente_del_perfil_y_rechaza_grupos_inexistentes():
    x,c,n,p,cols=entradas()
    xx,cc,nn=I.perturbar(x,c,n,p,'nivel_ventana',cols)
    assert np.array_equal(xx,x) and np.array_equal(cc,c)
    assert np.array_equal(nn,n[p])
    with pytest.raises(ValueError):I.perturbar(x,c,n,p,'no_existe',cols)

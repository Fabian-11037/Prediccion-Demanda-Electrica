"""La planificación usa desarrollo y variación, sin resultados de prueba."""
from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('precision_parada',ROOT/'scripts/precision_parada_jbook.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def inputs():
    return pd.read_csv(ROOT/'results/evaluacion_e2/entradas_precision_parada.csv')


def test_precision_cubre_procedimientos_sin_piloto():
    x=inputs();summary,pairs,matrix=module.calcular_precision_parada(x)
    design=pd.read_csv(ROOT/'results/evaluacion_e2/diseno_seleccion.csv')
    assert set(summary.clave)==set(design.loc[design.piloto.eq('Sin piloto independiente'),'clave'])
    assert len(x)==40 and matrix.shape==(5,8) and len(pairs)==28
    assert (summary.IC95_inf<summary.MASE_media).all()
    assert (summary.IC95_sup>summary.MASE_media).all()
    assert pairs.referencia.eq(module.REFERENCIA).all()
    assert pairs.delta_relativo.eq(.03).all()
    assert (pairs.R_Bonferroni>=pairs.R_individual).all()


def test_cambiar_metricas_de_prueba_no_cambia_la_precision():
    x=inputs();before=module.calcular_precision_parada(x)
    x['MASE_prueba']=np.linspace(1e6,1e9,len(x))
    x['MASE_ajuste']=np.arange(len(x))*123
    after=module.calcular_precision_parada(x)
    for a,b in zip(before,after):pd.testing.assert_frame_equal(a,b)


def test_dimensiona_el_margen_fijado_y_no_la_diferencia_observada():
    x=inputs();_,before,_=module.calcular_precision_parada(x)
    target='adj_modelo3_20261115_0419__N'
    x.loc[x.clave.eq(target),'MASE_parada']+=10
    _,after,_=module.calcular_precision_parada(x)
    assert not np.allclose(before.diferencia_MASE,after.diferencia_MASE)
    assert np.array_equal(before[['R_individual','R_Bonferroni']],after[['R_individual','R_Bonferroni']])


def test_no_mezcla_reservas_ni_duplica_semillas():
    x=inputs();x.loc[0,'fin_parada']='2020-01-01'
    with pytest.raises(AssertionError):module.calcular_precision_parada(x)
    x=inputs();x=pd.concat([x,x.iloc[[0]]],ignore_index=True)
    with pytest.raises(AssertionError):module.calcular_precision_parada(x)


def test_presentacion_completa_sin_huecos_y_con_fuentes_explicitas():
    out=ROOT/'results/evaluacion_e2'
    summary=pd.read_csv(out/'precision_desarrollo_resumen.csv').set_index('clave')
    seeds=pd.read_csv(out/'precision_desarrollo_semillas.csv')
    bench=pd.read_csv(out/'benchmark.csv').set_index('clave')
    assert set(summary.index)==set(bench.index[bench.semillas.eq(5)])
    assert len(seeds)==65 and summary.Semillas.eq(5).all()
    assert summary.Fuente.value_counts().to_dict()=={'Piloto externo':5,'Parada interna':8}
    assert summary[['MASE_media','DE','CV_pct','IC95_inf','IC95_sup']].notna().all().all()
    stop=summary[summary.Fuente.eq('Parada interna')]
    expected=inputs().groupby('clave').MASE_parada.mean().reindex(stop.index)
    assert np.allclose(stop.MASE_media,expected)
    general=pd.read_csv(out/'generalizacion_integrada.csv').set_index('clave')
    assert general.notna().all().all() and len(general)==13
    assert np.allclose(general['Precisión desarrollo'],summary.MASE_media.reindex(general.index))
    assert np.allclose(general.Prueba,bench.MASE.reindex(general.index))

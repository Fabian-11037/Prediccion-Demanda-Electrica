"""Precisión retrospectiva con métricas conservadas de parada, sin inferencia."""
from pathlib import Path
import hashlib
import inspect
import json
import shutil
import numpy as np
import pandas as pd
import nbformat as nb
from scipy import stats
from itertools import combinations

REFERENCIA='adj_modelo6_20261112_1035__N'


def calcular_precision_parada(entradas,referencia='adj_modelo6_20261112_1035__N'):
    """Usa solo MASE de desarrollo; no estima potencia con el efecto observado."""
    required=['clave','semilla','MASE_parada','n_parada','inicio_parada','fin_parada','q168_kwh']
    x=entradas[required].copy()
    assert not x.isna().any().any()
    assert not x.duplicated(['clave','semilla']).any()
    assert np.isfinite(x.MASE_parada).all() and x.MASE_parada.ge(0).all()
    assert x[['n_parada','inicio_parada','fin_parada','q168_kwh']].drop_duplicates().shape[0]==1
    for _,g in x.groupby('clave'):
        assert sorted(g.semilla.tolist())==list(range(1000,1005))
    matriz=x.pivot(index='semilla',columns='clave',values='MASE_parada').sort_index()
    assert matriz.notna().all().all() and referencia in matriz
    n=len(matriz);alpha=.05;objetivo=.80;delta=.03
    tcrit=stats.t.ppf(1-alpha/2,n-1)
    resumen=matriz.agg(['mean','std','count']).T.rename(columns={'mean':'MASE_media','std':'DE','count':'semillas'})
    resumen['CV_pct']=100*resumen.DE/resumen.MASE_media
    resumen['IC95_inf']=resumen.MASE_media-tcrit*resumen.DE/np.sqrt(n)
    resumen['IC95_sup']=resumen.MASE_media+tcrit*resumen.DE/np.sqrt(n)
    mu_ref=float(resumen.loc[referencia,'MASE_media'])
    assert mu_ref>0
    parejas=list(combinations(matriz.columns,2))
    z=stats.norm.ppf(1-alpha/2)+stats.norm.ppf(objetivo)
    z_bonf=stats.norm.ppf(1-alpha/(2*len(parejas)))+stats.norm.ppf(objetivo)
    filas=[]
    for a,b in parejas:
        d=matriz[a]-matriz[b]
        media=float(d.mean());sd=float(d.std(ddof=1));sd_rel=sd/mu_ref
        margen=tcrit*sd/np.sqrt(n)
        filas.append(dict(A=a,B=b,semillas=n,diferencia_MASE=media,DE_diferencia=sd,
            IC95_diferencia_inf=media-margen,IC95_diferencia_sup=media+margen,
            diferencia_pct_referencia=100*media/mu_ref,DE_relativa=sd_rel,
            referencia=referencia,MASE_referencia=mu_ref,delta_relativo=delta,
            alpha=alpha,potencia_objetivo=objetivo,parejas_familia=len(parejas),
            delta_sobre_DE=delta/sd_rel if sd_rel>0 else np.inf,
            R_individual=max(n,int(np.ceil((z*sd_rel/delta)**2))),
            R_Bonferroni=max(n,int(np.ceil((z_bonf*sd_rel/delta)**2))),
            MDD_5_pct_referencia=100*z*sd_rel/np.sqrt(n)))
    return resumen.reset_index(),pd.DataFrame(filas),matriz


def preparar_entradas(root):
    """Extrae 30 métricas con fechas y huellas; no lee predicciones de prueba."""
    root=Path(root);out=root/'results/evaluacion_e2';path=out/'entradas_precision_parada.csv'
    source=root.parent/'Consolidado-Entrenamientos-20261115'
    design=pd.read_csv(out/'diseno_seleccion.csv')
    keys=set(design.loc[design.piloto.eq('Sin piloto independiente'),'clave'])
    assert len(keys)>=2
    if (source/'consolidado/indice_modelos.json').is_file():
        index=json.loads((source/'consolidado/indice_modelos.json').read_text(encoding='utf-8'))
        rows=[]
        for model in index['modelos']:
            key=model['experimento']+'__'+model['variante']
            if not model['principal'] or key not in keys:continue
            p=source/model['artefactos']['resultado']
            result=json.loads(p.read_text(encoding='utf-8'))
            scales=json.loads((source/model['artefactos']['escaladores']).read_text(encoding='utf-8'))
            assert result['semilla']==model['semilla']
            rows.append(dict(clave=key,semilla=model['semilla'],MASE_parada=result['MASE_parada'],
                n_parada=result['n_parada'],inicio_parada=result['origen_parada'][0],fin_parada=result['origen_parada'][1],
                q168_kwh=scales['q168_kwh'],fuente_resultado=model['artefactos']['resultado'],
                sha256_resultado=hashlib.sha256(p.read_bytes()).hexdigest()))
        pd.DataFrame(rows).sort_values(['clave','semilla']).to_csv(path,index=False)
    inputs=pd.read_csv(path)
    assert len(inputs)==5*len(keys) and set(inputs.clave)==keys
    assert inputs.n_parada.eq(803).all()
    assert inputs.inicio_parada.eq('2019-10-21 00:00:00-05:00').all()
    assert inputs.fin_parada.eq('2021-12-31 00:00:00-05:00').all()
    assert np.allclose(inputs.q168_kwh,215617.90361589016,rtol=0,atol=1e-9)
    meta=pd.read_csv(out/'metadatos_procedimientos.csv').set_index(['clave','semilla'])
    for row in inputs.itertuples():
        assert np.isclose(row.MASE_parada,meta.loc[(row.clave,row.semilla),'MASE_parada'],rtol=1e-12)
    summary,pairs,_=calcular_precision_parada(inputs)
    names=pd.read_csv(out/'benchmark.csv').set_index('clave').modelo
    summary.insert(1,'Procedimiento',summary.clave.map(names))
    pairs.insert(2,'Comparación',pairs.A.map(names)+' − '+pairs.B.map(names))
    summary.to_csv(out/'precision_parada_resumen.csv',index=False)
    pairs.to_csv(out/'precision_parada_comparaciones.csv',index=False)
    audit=dict(procedimientos=len(keys),realizaciones=len(inputs),semillas=list(range(1000,1005)),parejas=len(pairs),
        n_origenes_parada=803,inicio=str(inputs.inicio_parada.iloc[0]),fin=str(inputs.fin_parada.iloc[0]),
        referencia=REFERENCIA,MASE_referencia=float(pairs.MASE_referencia.iloc[0]),delta_relativo=.03,
        alpha=.05,potencia_objetivo=.80,R_max_individual=int(pairs.R_individual.max()),
        R_max_Bonferroni=int(pairs.R_Bonferroni.max()),
        fuente_error='MASE_parada · evaluación del checkpoint restaurado',
        prueba_usada_para_dimensionar=False,piloto_externo_reconstruido=False,
        entrenamiento=False,publicacion=False)
    (out/'precision_parada_protocolo.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    for name in ['entradas_precision_parada.csv','precision_parada_resumen.csv','precision_parada_comparaciones.csv','precision_parada_protocolo.json']:
        shutil.copy2(out/name,root/'book/_static/resultados/evaluacion_e2'/name)
    return summary,pairs,audit


def integrar_precision_parada(root,cells):
    summary,pairs,audit=preparar_entradas(root)
    md=nb.v4.new_markdown_cell;code=nb.v4.new_code_cell
    best=summary.loc[summary.MASE_media.idxmin()]
    low=summary.loc[summary.CV_pct.idxmin()];high=summary.loc[summary.CV_pct.idxmax()]
    nuevo=[md('''### Precisión en la reserva interna

TFT, TCN, iTransformer, el control numérico N, TaTS con texto oficial, fusión tardía y las variantes N y N+O de TGForecaster conservan métricas de parada de sus cinco entrenamientos finales, semillas 1000–1004. Se utilizan para completar el análisis de precisión de desarrollo. Los ocho procedimientos comparten 803 orígenes de reserva interna, del 21 de octubre de 2019 al 31 de diciembre de 2021, y el denominador MASE q168 de 215.617,90361589016 kWh.

Se calcula media, DE, CV e IC95 % t entre semillas sobre el MASE de parada del checkpoint restaurado. La reserva seleccionó también la época de parada; los resultados son descriptivos y retrospectivos. Los intervalos condicionan ese periodo y suponen independencia y normalidad aproximada entre inicializaciones, difíciles de evaluar con cinco realizaciones.'''),
        code('''# Métricas de desarrollo guardadas; no se utilizan errores de prueba.
from itertools import combinations
p_inputs=pd.read_csv(g_result/'entradas_precision_parada.csv')
assert len(p_inputs)==5*p_inputs.clave.nunique() and p_inputs.n_parada.eq(803).all()
'''+inspect.getsource(calcular_precision_parada)+'''
p_summary,p_pairs,p_matrix=calcular_precision_parada(p_inputs)
p_summary.insert(1,'Procedimiento',p_summary.clave.map(g_names))
p_pairs.insert(2,'Comparación',p_pairs.A.map(g_names)+' − '+p_pairs.B.map(g_names))
display(p_summary[['Procedimiento','semillas','MASE_media','DE','CV_pct','IC95_inf','IC95_sup']].rename(columns={
    'semillas':'Semillas','MASE_media':'MASE parada medio','CV_pct':'CV (%)',
    'IC95_inf':'IC95 inferior','IC95_sup':'IC95 superior'}).round(5))
'''),md(f'''{best.Procedimiento} registra el menor MASE de parada medio ({best.MASE_media:.3f}). El CV oscila entre {low.CV_pct:.2f} % en {low.Procedimiento} y {high.CV_pct:.2f} % en {high.Procedimiento}. La estabilidad entre semillas y el error medio describen propiedades diferentes; esta tabla no sustituye el benchmark de prueba.

La planificación conserva δ=3 %, α=0,05 y potencia objetivo de 0,80. Las diferencias pareadas se expresan respecto de la media de parada del control N ({audit['MASE_referencia']:.6f}), común a los ocho procedimientos. Se utiliza la misma aproximación normal y MDD con cinco semillas descritas antes, con Bonferroni para las veintiocho parejas. El cálculo usa la DE de las diferencias y el margen fijado, sin estimar potencia a partir de la diferencia observada.'''),code('''p_columns=['Comparación','diferencia_MASE','DE_diferencia','IC95_diferencia_inf','IC95_diferencia_sup',
    'DE_relativa','delta_sobre_DE','R_individual','R_Bonferroni','MDD_5_pct_referencia']
display(p_pairs[p_columns].rename(columns={'diferencia_MASE':'Diferencia MASE','DE_diferencia':'DE diferencia',
    'IC95_diferencia_inf':'IC95 inferior','IC95_diferencia_sup':'IC95 superior',
    'DE_relativa':'DE relativa','delta_sobre_DE':'δ / DE','R_individual':'R individual',
    'R_Bonferroni':f'R Bonferroni({len(p_pairs)})','MDD_5_pct_referencia':'MDD con 5 (% control N)'}).round(5))
assert len(p_pairs)==len(p_summary)*(len(p_summary)-1)//2 and set(p_pairs.A)|set(p_pairs.B)==set(p_summary.clave)
'''),md(f'''El máximo requerido por la aproximación individual es {audit['R_max_individual']} semillas y asciende a {audit['R_max_Bonferroni']} con Bonferroni para las veintiocho parejas. Las cinco ejecuciones disponibles permiten estimar variabilidad, pero no acreditan potencia de 0,80 para el margen del 3 % en todas las comparaciones. Las estimaciones son sensibles a cinco diferencias de desarrollo y no garantizan la potencia de una evaluación externa.

Este análisis completa la precisión entre inicializaciones de los ocho procedimientos sin piloto externo. Sus resultados no se mezclan con las medias de los cuatro años externos de los pilotos ni convierten la planificación en un registro previo a los entrenamientos.

[Métricas de parada por semilla](_static/resultados/evaluacion_e2/entradas_precision_parada.csv) · [Resumen de precisión](_static/resultados/evaluacion_e2/precision_parada_resumen.csv) · [Quince diferencias y dimensionamiento](_static/resultados/evaluacion_e2/precision_parada_comparaciones.csv) · [Especificación del análisis](_static/resultados/evaluacion_e2/precision_parada_protocolo.json).''')]
    i=next(i for i,c in enumerate(cells) if c.cell_type=='markdown' and c.source.startswith('## Curvas de aprendizaje'))
    cells[i:i]=nuevo
    for cell in cells:
        if cell.cell_type=='markdown':
            cell.source=cell.source.replace('Se incluyen los doce procedimientos en el resumen de disponibilidad y se dimensionan las quince parejas de los seis que conservan un piloto independiente.','Se resumen los seis procedimientos con piloto independiente y se dimensionan sus quince parejas. Los otros seis se analizan a continuación mediante sus métricas de parada interna.')
            cell.source=cell.source.replace('En los seis procedimientos sin piloto independiente, el número requerido permanece sin estimar.','La precisión de los seis procedimientos sin piloto independiente se estima por separado a partir de su reserva interna.')
            cell.source=cell.source.replace('Los otros seis procedimientos no conservan un piloto independiente y se evalúan con sus cinco realizaciones finales.','Los otros seis procedimientos utilizan métricas de parada de sus cinco realizaciones finales para el análisis retrospectivo de precisión presentado a continuación.')
        elif cell.cell_type=='code' and "display(g_pilot_summary[['Disponibilidad'" in cell.source:
            cell.source=cell.source.replace("display(g_pilot_summary[['Disponibilidad','Semillas piloto','MASE medio','DE','CV (%)']]",
                "display(g_pilot_summary.loc[g_pilot_summary['Semillas piloto'].notna(),['Semillas piloto','MASE medio','DE','CV (%)']]")
        elif cell.cell_type=='code' and 'g_coverage.append' in cell.source:
            cell.source=cell.source.replace("'Piloto independiente':'Disponible' if g_k in set(g_pilots.clave) else 'No disponible',",
                "'Piloto independiente':'Disponible' if g_k in set(g_pilots.clave) else 'No disponible',\n        'Precisión de desarrollo':'Piloto externo' if g_k in set(g_pilots.clave) else 'Parada interna · retrospectiva',")
            cell.source=cell.source.replace("'Piloto independiente':'No disponible','Métricas ajuste/parada'",
                "'Piloto independiente':'No disponible','Precisión de desarrollo':'Sin resultados','Métricas ajuste/parada'")
    from presentacion_precision_jbook import integrar_presentacion_precision
    return integrar_presentacion_precision(root,cells)

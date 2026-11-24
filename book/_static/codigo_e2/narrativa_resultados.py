"""Actualiza narrativa desde artefactos: no importa ni ejecuta modelos."""
from pathlib import Path
from math import comb
from datetime import date
import json,re,shutil,hashlib
import numpy as np
import pandas as pd
import nbformat as nb


def palabras(n):
    return {0:'cero',1:'uno',2:'dos',3:'tres',4:'cuatro',5:'cinco',6:'seis',7:'siete',8:'ocho',9:'nueve',10:'diez',11:'once',12:'doce',13:'trece',14:'catorce',15:'quince',16:'dieciséis',17:'diecisiete',18:'dieciocho',19:'diecinueve',20:'veinte'}.get(int(n),str(int(n)))

def dec(x,n=4):return f'{float(x):.{n}f}'.replace('.',',')
def pvalor(x):
    return '<1e-300' if x==0 else (f'{float(x):.3e}' if x<1e-4 else f'{float(x):.4f}').replace('.',',')
def lista(items):
    items=list(items)
    return items[0] if len(items)==1 else ', '.join(items[:-1])+' y '+items[-1] if items else 'ninguno'


def cargar(root):
    root=Path(root);out=root/'results/evaluacion_e2'
    archivos=['benchmark','comparaciones_temporales','comparaciones_semillas','nemenyi','mcs','contrastes_texto','diseno_seleccion',
              'metadatos_procedimientos','pilotos_disponibles','entradas_precision_parada','precision_parada_resumen',
              'precision_parada_comparaciones','dimensionamiento_piloto_integrado','curvas_aprendizaje','resultados_horizonte']
    t={name:pd.read_csv(out/(name+'.csv')) for name in archivos}
    p=json.loads((out/'protocolo.json').read_text(encoding='utf8'))
    b=t['benchmark'];models=b[b.semillas.gt(1)];keys=set(models.clave);names=b.set_index('clave').modelo.to_dict()
    meta=t['metadatos_procedimientos'];stop=t['entradas_precision_parada'];pilot=t['pilotos_disponibles']
    assert set(meta.clave)==keys and not meta.duplicated(['clave','semilla']).any()
    assert len(t['comparaciones_temporales'])==comb(len(b),2)
    assert len(t['comparaciones_semillas'])==comb(len(models),2)
    assert set(stop.clave).isdisjoint(pilot.clave) and set(stop.clave)|set(pilot.clave)==keys
    imp=pd.read_csv(root/'results/diagnosticos_ampliados/importancia_semillas.csv');imp=imp[imp.clave.isin(keys)]
    rep=pd.read_csv(root/'results/diagnosticos_ampliados/reproduccion_semillas.csv');rep=rep[rep.clave.isin(keys)]
    lat=pd.read_csv(root/'results/diagnosticos_ampliados/latencia_semillas.csv');lat=lat[lat.clave.isin(keys)]
    ls=pd.read_csv(root/'results/diagnosticos_ampliados/latencia_resumen.csv');ls=ls[ls.clave.isin(keys)]
    ranking=models.sort_values('MASE').clave.tolist();seedcounts=meta.groupby('clave').semilla.nunique().unique();assert len(seedcounts)==1
    s=dict(procedimientos=len(models),realizaciones=len(meta),referencias=len(b)-len(models),metodos=len(b),semillas=int(seedcounts[0]),
           parejas_temporales=len(t['comparaciones_temporales']),parejas_semillas=len(t['comparaciones_semillas']),
           pilotos=pilot.clave.nunique(),procedimientos_parada=stop.clave.nunique(),realizaciones_parada=len(stop),
           parejas_parada=len(t['precision_parada_comparaciones']),parejas_piloto=len(t['dimensionamiento_piloto_integrado']),
           origenes=int(p['n_origenes']),origenes_parada=int(stop.n_parada.iloc[0]),anios_piloto=pilot.fold.nunique(),
           demanda_calendario=int(models.entradas.eq('demanda_calendario').sum()),ampliados=int(models.entradas.ne('demanda_calendario').sum()),
           importancia_ampliada=imp.clave.nunique(),fechas_importancia=int(imp.origenes.iloc[0]),repeticiones_importancia=imp.repeticion.nunique(),
           grupos_importancia=imp.grupo.nunique(),basales_completos=len(rep),procedimientos_basales=rep.clave.nunique(),
           latencia_procedimientos=lat.clave.nunique(),latencia_realizaciones=int(lat[['clave','semilla']].drop_duplicates().shape[0]),
           latencia_registros=len(lat),latencia_origenes=lat.origen.nunique(),latencia_repeticiones=int(lat.repeticiones.iloc[0]),
           latencia_calentamiento=int(lat.calentamiento.iloc[0]),contrastes_textuales=len(t['contrastes_texto']),
           ranking=ranking,mcs=t['mcs'].loc[t['mcs'].retenido_90,'clave'].tolist(),nombres=names,
           _tablas=t,_p=p,_imp=imp,_rep=rep,_lat=lat,_ls=ls,_root=str(root))
    s['rechazos_DM']=int(t['comparaciones_temporales'].p_DM_Holm.lt(.05).sum())
    s['rechazos_t']=int(t['comparaciones_semillas'].p_t_Holm.lt(.05).sum())
    s['rechazos_Wilcoxon']=int(t['comparaciones_semillas'].p_Wilcoxon_Holm.lt(.05).sum())
    s['rechazos_Nemenyi']=int(t['nemenyi'][next(c for c in t['nemenyi'] if c.startswith('p'))].lt(.05).sum())
    plan=json.loads((root/'book/_static/codigo_e2/top10_adicionales/config/plan_e2.json').read_text(encoding='utf8'))
    def buscar(x):
        if isinstance(x,dict):
            if 'lote_evaluacion' in x:return x['lote_evaluacion']
            for value in x.values():
                found=buscar(value)
                if found is not None:return found
    s['horizonte']=int(t['resultados_horizonte'].groupby('clave').horizonte.nunique().iloc[0])
    canonico=next((out/'entradas/adicionales').glob('*/final/N/semilla_*/predicciones_canonicas.npz'))
    with np.load(canonico,allow_pickle=False) as dates:
        s['inicio_prueba']=str(dates['origen'][0]);s['fin_prueba']=str(dates['origen'][-1])
    s['timesnet_lote_desarrollo']=int(buscar(plan))
    s['timesnet_lote_prueba']=int(json.loads((root/'results/reevaluacion_timesnet_lote1_20261120/protocolo.json').read_text(encoding='utf8'))['lote'])
    return s


def mcs(s):
    return f"El MCS al 90 % retiene {lista(s['nombres'][key] for key in s['mcs'])}. El conjunto está condicionado al periodo de prueba, a las {palabras(s['semillas'])} inicializaciones y al protocolo de remuestreo."


def texto(s):
    resultado=[]
    for row in s['_tablas']['contrastes_texto'].itertuples():
        verbo='reduce' if row.relativa_pct<0 else 'aumenta'
        frase=f"{s['nombres'][row.A]} {verbo} el MAE {dec(abs(row.relativa_pct),2)} % frente a {s['nombres'][row.B]} (IC95 % relativo [{dec(row.IC95_inf,2)} %; {dec(row.IC95_sup,2)} %]; DM–HLN con Holm({s['contrastes_textuales']}): p={pvalor(row.p_DM_Holm_texto)})."
        frase+=' El contraste detecta una diferencia temporal.' if row.p_DM_Holm_texto<.05 else ' El contraste no demuestra una diferencia temporal.'
        frase+=' TOST acredita equivalencia dentro del margen relativo del 3 %.' if row.p_TOST_Holm_texto<.05 else ' TOST no acredita equivalencia dentro del margen relativo del 3 %.'
        resultado.append(frase)
    return '\n\n'.join(resultado)+'\n\nLa inferencia condiciona las inicializaciones y el corpus evaluados. Las configuraciones se seleccionaron por separado; la comparación de procedimientos y las intervenciones con pesos fijos responden a preguntas diferentes.'


def textos(s):
    w=palabras;n=s['procedimientos'];t=s['_tablas'];p=s['_p'];names=s['nombres'];best=t['benchmark'].set_index('clave').loc[s['ranking'][0]]
    ranking=lista(names[key] for key in s['ranking'][:3]);seeds=s['semillas']
    r={}
    r['familias']=f"Se incluyen {s['parejas_temporales']} parejas temporales entre {w(n)} procedimientos aprendidos y {w(s['referencias'])} referencias, y {s['parejas_semillas']} parejas entre los {w(n)} procedimientos con {w(seeds)} semillas. Holm se aplica por tipo de prueba a su familia completa. Los {w(s['contrastes_textuales'])} contrastes textuales constituyen una familia exploratoria específica."
    r['busqueda']=f"La selección comprende {w(n)} procedimientos: {w(s['demanda_calendario'])} con demanda y calendario y {w(s['ampliados'])} con entradas ampliadas. Los presupuestos, semillas y periodos de cada procedimiento se conservan en la [tabla de selección](_static/resultados/evaluacion_e2/diseno_seleccion.csv). La comparación mantiene esas diferencias de diseño."
    r['ejecucion']=f"Los {w(n)} procedimientos conservan {w(seeds)} realizaciones finales cada uno. La [tabla de selección](_static/resultados/evaluacion_e2/diseno_seleccion.csv) registra máximos de épocas, paciencia y semillas por etapa. Se restaura el mejor estado de la reserva interna. Hay pilotos independientes para {w(s['pilotos'])} procedimientos y precisión retrospectiva de parada para {w(s['procedimientos_parada'])}. El preprocesamiento se ajusta al tramo de optimización de cada pliegue."
    r['precision_cobertura']=f"Los {w(n)} procedimientos tienen una estimación de precisión de desarrollo: {w(s['pilotos'])} a partir del piloto externo y {w(s['procedimientos_parada'])} a partir de la parada interna. Los puntos muestran {w(seeds)} inicializaciones por procedimiento. Las fuentes cubren periodos y mecanismos de selección distintos; sus medias no constituyen un ranking común de validación. El control de permutación conserva las familias para las que fue calculado."
    r['precision_intro']=f"### Precisión de desarrollo y dimensionamiento\n\nEl resumen identifica fuente y periodo para los {w(n)} procedimientos. Los {w(s['pilotos'])} pilotos externos y las {w(s['procedimientos_parada'])} reservas internas se dimensionan por separado. Cada semilla piloto resume {w(s['anios_piloto'])} años externos; cada realización de parada resume {s['origenes_parada']} días. Se presentan media de MASE, DE, CV e IC95 % entre inicializaciones. Los errores de prueba no intervienen en el dimensionamiento."
    d=t['dimensionamiento_piloto_integrado'];col=next(c for c in d if c.startswith('R Bonferroni'))
    r['piloto_dimensionamiento']=f"La aproximación individual requiere entre {int(d['R individual'].min())} y {int(d['R individual'].max())} semillas por procedimiento, según la pareja; el máximo con Bonferroni para las {w(s['parejas_piloto'])} parejas es {int(d[col].max())}. Las {w(seeds)} semillas ejecutadas no acreditan potencia de 0,80 para el margen del 3 % en todas las comparaciones. Los {w(s['procedimientos_parada'])} procedimientos sin piloto independiente conservan su dimensionamiento retrospectivo de parada, presentado por separado."
    r['parada_intro']=f"### Precisión en la reserva interna\n\n{lista(names[k] for k in sorted(set(t['entradas_precision_parada'].clave)))} conservan métricas de parada de sus realizaciones finales. Los {w(s['procedimientos_parada'])} procedimientos reúnen {s['realizaciones_parada']} registros sobre {s['origenes_parada']} orígenes y el denominador q168={dec(p['q168'],8)} kWh.\n\nSe calcula media, DE, CV e IC95 % t entre las {w(seeds)} semillas. La reserva seleccionó la época de parada; el análisis es descriptivo y retrospectivo. Los intervalos condicionan ese periodo y suponen independencia y normalidad aproximada entre inicializaciones."
    d=t['precision_parada_resumen'];low=d.loc[d.CV_pct.idxmin()];high=d.loc[d.CV_pct.idxmax()];bs=d.loc[d.MASE_media.idxmin()];ref=t['precision_parada_comparaciones'].iloc[0]
    r['parada_precision']=f"{names[bs.clave]} registra el menor MASE de parada medio ({dec(bs.MASE_media,3)}). El CV oscila entre {dec(low.CV_pct,2)} % en {names[low.clave]} y {dec(high.CV_pct,2)} % en {names[high.clave]}. La estabilidad entre semillas y el error medio describen propiedades distintas.\n\nEl dimensionamiento conserva δ=3 %, α=0,05 y potencia objetivo 0,80. Las diferencias se expresan respecto de la media de parada de {names[ref.referencia]} ({dec(ref.MASE_referencia,6)}), común a los {w(s['procedimientos_parada'])} procedimientos. Bonferroni considera las {s['parejas_parada']} parejas."
    d=t['precision_parada_comparaciones']
    r['parada_dimensionamiento']=f"El máximo requerido por la aproximación individual es {int(d.R_individual.max())} semillas y asciende a {int(d.R_Bonferroni.max())} con Bonferroni para las {s['parejas_parada']} parejas. Las {w(seeds)} realizaciones permiten estimar variabilidad, sin acreditar potencia de 0,80 para el margen del 3 % en todas las comparaciones.\n\nEste análisis completa la precisión retrospectiva de los {w(s['procedimientos_parada'])} procedimientos sin piloto externo. Sus resultados permanecen separados de los pilotos externos.\n\n[Métricas de parada por semilla](_static/resultados/evaluacion_e2/entradas_precision_parada.csv) · [Resumen de precisión](_static/resultados/evaluacion_e2/precision_parada_resumen.csv) · [{s['parejas_parada']} diferencias y dimensionamiento de parada](_static/resultados/evaluacion_e2/precision_parada_comparaciones.csv)."
    r['temporal_intro']=f"## Comparaciones temporales de todos los procedimientos\n\nLa matriz incluye las {s['parejas_temporales']} parejas de los {n} procedimientos aprendidos y las {w(s['referencias'])} referencias. Muestra p de DM–HLN con Holm({s['parejas_temporales']}), sobre diferencias diarias con HAC de {p['lag_HAC']} rezagos. Las diferencias e intervalos completos se conservan en la tabla descargable."
    r['temporal_resultado']=f"DM–HLN con Holm rechaza en {s['rechazos_DM']} de las {s['parejas_temporales']} parejas. Los rechazos son específicos de cada comparación y no convierten el orden de medias en un ranking de diferencias significativas. Un valor p elevado no establece equivalencia. La inferencia temporal condiciona las inicializaciones ejecutadas."
    r['semillas_intro']=f"## Comparaciones entre semillas y tamaño del efecto\n\nSe incluyen las {s['parejas_semillas']} parejas de los {n} procedimientos, emparejadas por índice de semilla. Se reportan diferencias de MASE, Shapiro–Wilk, t pareada y Wilcoxon exacto con Holm({s['parejas_semillas']}), y Cohen d_z con IC95 % por inversión de t no central. La t es una sensibilidad; {w(seeds)} pares no acreditan normalidad porque Shapiro no rechace."
    r['wilcoxon']=f"Con {w(seeds)} pares, el mínimo p bilateral de Wilcoxon exacto sin empates es {dec(2/2**seeds,4)}; {s['rechazos_Wilcoxon']} de las {s['parejas_semillas']} comparaciones rechazan tras Holm. Las pruebas entre semillas y temporales responden a fuentes distintas de incertidumbre. Cohen d_z estandariza la diferencia por su dispersión entre inicializaciones y no expresa un porcentaje de mejora."
    r['t_pareada']=f"La t pareada con Holm rechaza en {s['rechazos_t']} de las {s['parejas_semillas']} parejas. Los valores negativos favorecen al procedimiento de la fila; los asteriscos identifican rechazos sujetos a sus supuestos. El tamaño del efecto estandarizado y la relevancia práctica se valoran por separado."
    r['qq']=f"### Distribución de diferencias entre inicializaciones\n\nLos gráficos Q–Q comparan cada procedimiento con PatchTST numérico: {n-1} contrastes diagnósticos sobre {w(seeds)} diferencias. La tabla anterior conserva las {s['parejas_semillas']} parejas completas."
    r['friedman']=f"## Friedman, post hoc y conjunto de confianza\n\nFriedman utiliza {w(seeds)} bloques de semillas y {w(n)} procedimientos, condicionados a una misma serie. Nemenyi emplea rangos medios y una distancia crítica de {dec(p['CD_Nemenyi'],2)} posiciones. Wilcoxon–Holm se conserva como alternativa post hoc."
    r['nemenyi_intro']=f"### Post hoc completo de Nemenyi\n\nLa matriz presenta las {s['parejas_semillas']} comparaciones entre los {w(n)} procedimientos. Nemenyi controla su familia mediante la distribución del rango studentizado; no se añade Holm a esos valores p."
    r['nemenyi']=f"Nemenyi rechaza en {s['rechazos_Nemenyi']} de las {s['parejas_semillas']} parejas. El rechazo global de Friedman no se extiende automáticamente a cada par. Las celdas sin rechazo no acreditan equivalencia."
    r['mcs']=f"El MCS evalúa los {w(n)} procedimientos aprendidos y las {w(s['referencias'])} referencias. "+mcs(s)
    r['ranking']=f"{ranking} presentan los menores errores medios de prueba. La cercanía de barras e intervalos no sustituye la comparación pareada sobre los mismos orígenes."
    r['observado']=f"{ranking} presentan los menores errores agregados. La distancia a la diagonal describe errores de nivel y amplitud; las nubes muestran dónde se concentran las desviaciones. La densidad reúne fechas e inicializaciones y no implica independencia entre observaciones."
    nuevos=set(s['_imp'].clave)-set(s['_rep'].clave)
    r['importancia']=f"## Importancia de variables en los modelos ampliados\n\nSe trasladan circularmente grupos de entradas entre orígenes del mismo año con separación mínima de 28 días, manteniendo pesos, escaladores, orden interno de la historia y etiqueta. Los {w(s['grupos_importancia'])} grupos incluyen perfil y nivel de demanda, precio, reservas, aportes, temperatura, humedad, precipitación y calendario; cada grupo climático reúne las ocho ciudades.\n\nSe utilizan {w(seeds)} inicializaciones de los {w(s['importancia_ampliada'])} procedimientos ampliados y {s['fechas_importancia']} fechas: días 1, 8, 15 y 22 de cada mes de 2022–2025. Los donantes proceden de los {s['origenes']} orígenes del periodo; se evalúan {w(s['repeticiones_importancia'])} desplazamientos y se promedian primero las semillas.\n\nEl registro de reproducción conserva {s['basales_completos']} comprobaciones de {w(s['procedimientos_basales'])} procedimientos sobre los orígenes completos. {lista(names[k] for k in sorted(nuevos))} incorporan sus basales CPU en las {s['fechas_importancia']} fechas del diagnóstico. TimesNet utiliza lote {s['timesnet_lote_prueba']} tanto en el basal como en cada perturbación. La referencia de toda perturbación es la predicción del mismo checkpoint y motor. La hora del día es invariante entre estos orígenes diarios y su importancia no queda identificada."
    r['coste']=f"## Coste de inferencia medido y coste de ajuste registrado\n\nSe presentan parámetros, duración y épocas de ajuste de los {w(n)} procedimientos. La latencia CPU comprende {s['latencia_realizaciones']} realizaciones y {s['latencia_registros']} registros: {s['latencia_origenes']} orígenes por semilla, un hilo, lote uno, {s['latencia_calentamiento']} calentamientos y {s['latencia_repeticiones']} mediciones. Se informa la mediana de las medianas y la mediana de sus p95; esta última no es el p95 combinado. Los dispositivos y protocolos de entrenamiento distintos limitan la comparación del coste de ajuste."
    d=s['_ls'];low=d.loc[d.mediana_ms.idxmin()];high=d.loc[d.mediana_ms.idxmax()]
    r['latencia']=f"La menor latencia mediana corresponde a {names[low.clave]} ({dec(low.mediana_ms,3)} ms) y la mayor a {names[high.clave]} ({dec(high.mediana_ms,3)} ms). Las mediciones describen el forward en CPU con entradas y vectores preparados; no incluyen lectura, escalado ni codificación de documentos."
    r['cobertura']=f"Los {w(n)} procedimientos tienen benchmark, curvas, observado–predicho, contrastes, importancia y latencia. La importancia ampliada utiliza {s['fechas_importancia']} fechas y {s['repeticiones_importancia']} desplazamientos en {w(s['importancia_ampliada'])} procedimientos; las {w(s['demanda_calendario'])} familias base conservan sus diagnósticos específicos. Hay {w(s['pilotos'])} pilotos independientes y {w(s['procedimientos_parada'])} análisis retrospectivos de parada."
    r['texto']=texto(s)
    cuerpo=f"En el benchmark de referencia, {best.modelo} obtiene el menor MASE medio ({dec(best.MASE,4)}); su MAE es {dec(best.MAE/1e6,4)} GWh por intervalo. El orden descriptivo comienza por {ranking}. "+mcs(s)+'\n\n'+texto(s)+f"\n\nLa comparación reúne {w(n)} procedimientos aprendidos, {s['realizaciones']} realizaciones y {w(s['referencias'])} referencias. Las diferencias de entradas, máscaras y presupuestos limitan su atribución exclusiva a la arquitectura. La dispersión de las semillas no acredita por sí sola la potencia objetivo de 0,80."
    r['discusion']='## Discusión\n\n'+cuerpo;r['sintesis']='## Evaluación conjunta de arquitecturas\n\n'+cuerpo
    r['timesnet']=f"### Protocolo de inferencia de TimesNet\n\nLa selección y la parada interna conservan las métricas registradas con lote {s['timesnet_lote_desarrollo']}; la prueba y las perturbaciones se calculan por origen con lote {s['timesnet_lote_prueba']}. La selección de frecuencias de TimesBlock depende del lote. Por tanto, la diferencia entre errores de desarrollo y prueba combina cambio de periodo y cambio de condiciones de inferencia. Las curvas describen el ajuste registrado."
    r['seleccion']=f"## Entrenamiento, selección y precisión del piloto\n\nLa tabla reúne los presupuestos de los {w(n)} procedimientos y las semillas finales. Los filtros, finalistas, años externos y condiciones de parada se conservan en los registros de selección. Los {w(s['pilotos'])} pilotos externos y los {w(s['procedimientos_parada'])} análisis de parada interna se identifican por separado; la selección utiliza exclusivamente desarrollo."
    r['generalizacion_intro']=f"## Validación y generalización\n\nLos {w(n)} procedimientos incluyen validación de selección, precisión de desarrollo y prueba. La barra central utiliza piloto externo en {w(s['pilotos'])} procedimientos y parada interna en {w(s['procedimientos_parada'])}; colores y tramas identifican su fuente. Los periodos y protocolos de selección se detallan en la tabla de diseño."
    cv=t['metadatos_procedimientos'].groupby('clave').validacion.first()
    observed=t['benchmark'].set_index('clave').MASE.reindex(cv.index)
    higher=int(observed.gt(cv).sum())
    r['generalizacion_resultado']=f"El MASE de prueba supera al de validación en {higher} de los {n} procedimientos. La barra de precisión de desarrollo conserva la fuente disponible para cada procedimiento. La parada interna participó en la selección de época y puede ser optimista; la diferencia hasta prueba no se atribuye únicamente a cambios de distribución."
    fdecision='rechaza igualdad global de rangos' if p['p_Friedman']<.05 else 'no demuestra diferencias globales de rangos'
    ldecision='rechaza igualdad de varianzas' if p['p_Levene']<.05 else 'no demuestra desigualdad de varianzas'
    r['friedman_resultado']=f"Friedman {fdecision} (p={pvalor(p['p_Friedman'])}); su resultado no se extiende automáticamente a cada par. Levene centrado en mediana {ldecision} (p={pvalor(p['p_Levene'])}). Los rangos, contrastes pareados y tamaños del efecto se presentan por separado. El MCS utiliza pérdidas diarias y eliminación secuencial con bootstrap circular."
    changes=[]
    for row in t['contrastes_texto'].itertuples():
        status='reduce' if row.relativa_pct<0 else 'aumenta'
        verdict='con diferencia temporal detectada' if row.p_DM_Holm_texto<.05 else 'sin diferencia temporal demostrada'
        changes.append(f"{names[row.A]} {status} el MAE {dec(abs(row.relativa_pct),2)} % frente a su control, {verdict}")
    r['sintesis']=f"## Evaluación conjunta de arquitecturas\n\nEl benchmark de referencia incluye {w(n)} procedimientos aprendidos y {s['realizaciones']} realizaciones. El menor MASE medio corresponde a {best.modelo} ({dec(best.MASE,4)}). "+mcs(s)+'\n\n'+'; '.join(changes)+'. Las entradas y los presupuestos distintos limitan la atribución exclusiva a la arquitectura. La sensibilidad al contenido y la comparación entre procedimientos mantienen interpretaciones separadas.'
    if 'nuevo_nbeats__N' in names and 'mlp' in names:
        bench=t['benchmark'].set_index('clave');nv=bench.loc['nuevo_nbeats__N'];mv=bench.loc['mlp'];rel=100*(nv.MASE/mv.MASE-1)
        direction='mayor' if rel>0 else 'menor'
        detalle=f"N-BEATS presenta MASE {dec(nv.MASE,4)}, MAE {dec(nv.MAE/1e6,4)} GWh y RMSE {dec(nv.RMSE/1e6,4)} GWh por intervalo. Su MAE es {dec(abs(rel),2)} % {direction} que el de MLP. La complejidad de la arquitectura no garantiza menor error bajo este conjunto de entradas y presupuesto."
        r['ranking']+='\n\n'+detalle
        r['discusion']+='\n\n'+detalle
    from coherencia_documental import textos as textos_cobertura
    r.update(textos_cobertura(s))
    return r

PREFIJOS={
 '## Entrenamiento, selección y precisión del piloto':'seleccion',
 '## Validación y generalización':'generalizacion_intro','El error de prueba supera al de validación':'generalizacion_resultado',
 'El MASE de prueba supera al de validación':'generalizacion_resultado','Friedman rechaza igualdad global':'friedman_resultado',
 '### Precisión de desarrollo y dimensionamiento':'precision_intro',
 'La aproximación individual exige':'piloto_dimensionamiento','La aproximación individual requiere':'piloto_dimensionamiento',
 '### Precisión en la reserva interna':'parada_intro','El máximo requerido por la aproximación individual':'parada_dimensionamiento',
 '## Comparaciones temporales de todos los procedimientos':'temporal_intro','DM–HLN con Holm rechaza':'temporal_resultado',
 '## Comparaciones entre semillas y tamaño del efecto':'semillas_intro','Las pruebas entre semillas y las temporales':'wilcoxon',
 'Con cinco pares, el mínimo p bilateral':'wilcoxon','La t pareada con Holm rechaza':'t_pareada',
 '### Distribución de diferencias entre inicializaciones':'qq','## Friedman, post hoc y conjunto de confianza':'friedman',
 '### Post hoc completo de Nemenyi':'nemenyi_intro','Nemenyi rechaza en':'nemenyi',
 'El MCS evalúa':'mcs','El procedimiento evaluó los':'mcs','## Importancia de variables en los modelos ampliados':'importancia',
 '## Coste de inferencia medido y coste de ajuste registrado':'coste','La menor latencia mediana corresponde':'latencia',
 '## Discusión':'discusion','TaTS reduce el error medio':'texto','TaTS adaptado · N+O reduce el MAE':'texto',
 'TimesNet presenta el menor error medio':'ranking','TaTS con texto oficial y su control numérico presentan':'ranking',
 'TimesNet, TaTS y PatchTST presentan':'observado','TaTS y su control numérico muestran':'observado',
 '### Protocolo de inferencia de TimesNet':'timesnet'}


def actualizar_celdas(root,cells,capitulo='10_evaluacion',estado=None):
    s=estado or cargar(root);v=textos(s);w=palabras;n=s['procedimientos'];p=s['_p']
    from coherencia_documental import PREFIJOS as prefijos_cobertura,sin_repeticiones,actualizar_descargas,actualizar_codigo
    prefijos={**PREFIJOS,**prefijos_cobertura}
    import yaml
    toc=yaml.safe_load((Path(root)/'book/_toc.yml').read_text(encoding='utf-8'))
    model_sections=next(c['sections'] for c in toc['chapters'] if c['file']=='10_modelos')
    evalnum=next(i for i,section in enumerate(model_sections,1) if section['file']=='10_evaluacion')
    for c in cells:
        if c.cell_type=='code':
            c.source=actualizar_codigo(c.source)
            continue
        if c.cell_type!='markdown':continue
        c.source=sin_repeticiones(c.source)
        if capitulo=='10_evaluacion' and c.source.startswith('### Código de restauración y diagnóstico'):
            actual=(Path(root)/'scripts/diagnosticos_ampliados.py').read_text(encoding='utf-8')
            def sincronizar(m):
                return '```python\n'+actual.rstrip()+'\n```' if 'from inferencia_ampliada import' in m[1] else m[0]
            c.source=re.sub(r'```python\n(.*?)```',sincronizar,c.source,flags=re.S)
        source=c.source;role=c.metadata.get('resultado_documental')
        if not role:
            if capitulo=='10_evaluacion':
                if source.startswith('(evaluacion-e2)=') or re.match(r'# 10\.\d+\. Resultados y evaluación integrada',source):role='evaluacion_intro'
                else:role=next((value for key,value in prefijos.items() if source.startswith(key)),None)
                if re.match(r'Los \w+ procedimientos tienen una estimación de precisión',source):role='precision_cobertura'
                if re.match(r'Los \w+ procedimientos tienen benchmark',source):role='cobertura'
                if 'registra el menor MASE de parada medio' in source:role='parada_precision'
            elif capitulo=='01_datos' and source.startswith('**Tipo de problema:**'):role='datos_pregunta'
            elif capitulo=='12_reproducibilidad':
                if source.startswith('## Procedencia de modelos y reconstrucción'):role='repro_procedencia'
                elif source.startswith(('Los análisis se ejecutan por CPU.','Los análisis documentales se ejecutan por CPU.')):role='repro_entorno'
            elif capitulo=='09_validacion':
                role={'## Búsqueda y prevención de fuga':'busqueda','## Unidad de evaluación, bloques y comparaciones':'unidad',
                      '## Ejecución y alcance estadístico':'ejecucion'}.get(source.split('\n')[0])
            elif capitulo=='11_sintesis' and source.startswith('## Evaluación conjunta de arquitecturas'):role='sintesis'
            elif capitulo=='10_modelos' and source.startswith('# 10. Modelos'):role='portada'
        if role:
            if role=='evaluacion_intro':
                c.source=f'''(evaluacion-e2)=
# 10.{evalnum}. Resultados y evaluación integrada

El benchmark de referencia compara {w(n)} procedimientos aprendidos con {w(s['semillas'])} semillas cada uno ({s['realizaciones']} realizaciones), además de {w(s['referencias'])} referencias deterministas. PatchTST se contabiliza una vez y TGForecaster incorpora las variantes N y N+O.

La evaluación cubre {s['origenes']} orígenes diarios, de {s['inicio_prueba']} a {s['fin_prueba']}. La pérdida por origen promedia el error absoluto de sus {s['horizonte']} horas, dividido por q168={dec(p['q168'],8)} kWh. Se promedian las pérdidas de las {w(s['semillas'])} semillas, conservando el estimando de error individual.

$$L_{{m,t}}=\\frac{{1}}{{{s['semillas']}\\times{s['horizonte']}\\times q_{{168}}}}\\sum_{{s=1}}^{{{s['semillas']}}}\\sum_{{h=1}}^{{{s['horizonte']}}}|y_{{t,h}}-\\widehat y_{{m,s,t,h}}|.$$

Los {w(s['demanda_calendario'])} procedimientos de demanda/calendario utilizan esas entradas; los {w(s['ampliados'])} ampliados incorporan clima, hidrología y precio, con o sin documentos. Las máscaras y los presupuestos distintos limitan la atribución de diferencias a la arquitectura.

{v['familias']}

La incertidumbre temporal utiliza bootstrap circular pareado de {p['bloque_dias']} días y {p['B']} réplicas, con sensibilidad de 7 y 28 días; condiciona las inicializaciones ejecutadas. Los diagnósticos de precisión, generalización, observado–predicho, segmentos, atípicos, importancia y latencia abarcan los procedimientos según sus conjuntos de entrada.'''
            elif role in ['busqueda','ejecucion']:
                partes=source.split('\n\n');partes[1]=v[role];c.source='\n\n'.join(partes)
            elif role=='unidad':
                partes=source.split('\n\n')
                for i,para in enumerate(partes):
                    if para.startswith(('Se conservan ','Se incluyen ')):
                        partes[i]=v['familias']+' El margen relativo es ±3 % respecto de la referencia de la pareja, o del control N en los contrastes textuales. No se infiere equivalencia por ausencia de significación.'
                    elif 'procedimientos sin piloto externo' in para:
                        partes[i]=f"Los {w(s['procedimientos_parada'])} procedimientos sin piloto externo reúnen {s['realizaciones_parada']} métricas de parada y {s['parejas_parada']} parejas, sobre {s['origenes_parada']} días anteriores a 2022. Esta precisión es retrospectiva: la reserva seleccionó la época. El dimensionamiento mantiene δ=3 % y no utiliza errores de prueba."
                c.source='\n\n'.join(partes)
            elif role=='repro_procedencia':
                partes=source.split('\n\n')
                for i,para in enumerate(partes):
                    if para.startswith('La evaluación integra '):partes[i]=v[role]
                c.source='\n\n'.join(partes)
            elif role=='portada':
                c.source=f'''# 10. Modelos, arquitecturas y evaluación

El capítulo presenta primero la composición matemática de los modelos y sus pipelines de entrenamiento y prueba. Después reúne resultados, incertidumbre y diagnósticos en una evaluación integrada.

El benchmark de referencia reúne {w(n)} procedimientos completados, {s['realizaciones']} realizaciones y {w(s['referencias'])} referencias estacionales. Las fichas distinguen entradas, mecanismos y presupuestos de ajuste.

## Secuencia de lectura

La {{ref}}`notación y las operaciones compartidas <fundamentos-arquitecturas>` introducen entradas, normalización, atención y optimización. Las fichas desarrollan cada arquitectura, su esquema y código. La {{ref}}`evaluación integrada <evaluacion-e2>` reúne selección, aprendizaje, precisión, inferencia, diagnóstico e importancia.'''
            else:c.source=v[role]
            c.metadata['resultado_documental']=role;c.metadata.pop('plantilla_resultados',None)
        if capitulo=='10_evaluacion' and c.source.startswith('## Tablas y reproducción'):
            c.source=actualizar_descargas(c.source,s)
            c.source=re.sub(r'\[\d+ comparaciones temporales\]',f"[{s['parejas_temporales']} comparaciones temporales]",c.source)
            c.source=re.sub(r'\[\d+ comparaciones entre semillas\]',f"[{s['parejas_semillas']} comparaciones entre semillas]",c.source)
    if capitulo=='10_evaluacion' and not any(c.metadata.get('resultado_documental')=='timesnet' for c in cells):
        i=next((i for i,c in enumerate(cells) if c.cell_type=='markdown' and c.source.startswith('## Validación y generalización')),len(cells))
        cells.insert(i,nb.v4.new_markdown_cell(v['timesnet'],metadata={'resultado_documental':'timesnet'}))
    from revision_e3_documental import armonizar_celdas
    return armonizar_celdas(root,cells,capitulo)


def revision(root,s):
    from revision_integral_jbook import inventariar
    inv=inventariar(root);counts=inv.tipo.value_counts().to_dict()
    report={k:value for k,value in s.items() if not k.startswith('_')}
    report.update(inventario=counts,entrenamiento=False,publicacion=False)
    sources=[Path(root)/'results/evaluacion_e2'/f'{name}.csv' for name in s['_tablas']]
    sources += [Path(root)/'results/evaluacion_e2/protocolo.json',
                Path(root)/'results/diagnosticos_ampliados/importancia_semillas.csv',
                Path(root)/'results/diagnosticos_ampliados/reproduccion_semillas.csv',
                Path(root)/'results/diagnosticos_ampliados/latencia_semillas.csv',
                Path(root)/'results/diagnosticos_ampliados/latencia_resumen.csv',
                Path(root)/'book/_static/codigo_e2/top10_adicionales/config/plan_e2.json',
                Path(root)/'results/reevaluacion_timesnet_lote1_20261120/protocolo.json']
    report['fuentes_sha256']={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    out=Path(root)/'results/revision_documental_20261120';out.mkdir(exist_ok=True)
    (out/'resumen_documental.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,default=lambda value:value.item()),encoding='utf8')
    nbs=list((Path(root)/'book').glob('*.ipynb'));ncode=sum(sum(c.cell_type=='code' for c in nb.read(p,4).cells) for p in nbs)
    body=f'''# Revisión de consistencia y completitud del JBook

Estado local verificado el {date.today().isoformat()}. Este informe sustituye el inventario de la revisión del 16 de noviembre de 2026. Los resultados proceden de artefactos conservados; no se ejecuta entrenamiento ni publicación.

## Inventario vigente

{len(nbs)} cuadernos, {ncode} celdas analíticas, {counts.get('figura',0)} gráficas, {counts.get('tabla_calculada',0)+counts.get('tabla_markdown',0)} tablas y {counts.get('esquema',0)} esquemas. La evaluación comprende {s['procedimientos']} procedimientos aprendidos, {s['realizaciones']} realizaciones, {s['referencias']} referencias, {s['parejas_temporales']} parejas temporales y {s['parejas_semillas']} parejas entre semillas.

La precisión distingue {s['pilotos']} pilotos externos y {s['procedimientos_parada']} reservas internas, con {s['realizaciones_parada']} registros y {s['parejas_parada']} parejas. La importancia ampliada comprende {s['importancia_ampliada']} procedimientos; la latencia, {s['latencia_procedimientos']}.

## Conclusiones y procedencia

{mcs(s)} El menor MASE medio corresponde a {s['nombres'][s['ranking'][0]]}: {dec(s['_tablas']['benchmark'].set_index('clave').loc[s['ranking'][0],'MASE'],8)}.

Conteos, orden descriptivo, decisiones de contrastes, MCS y conclusiones textuales se generan mediante `scripts/narrativa_resultados.py` a partir de CSV y JSON. Las celdas Markdown identifican su función con `resultado_documental` y se regeneran al reconstruir el libro. Las fuentes y sus huellas están en `results/revision_documental_20261120/resumen_documental.json`.

## Alcance metodológico

TimesNet conserva desarrollo con lote {s['timesnet_lote_desarrollo']} y prueba con lote {s['timesnet_lote_prueba']}; esa diferencia se declara en la evaluación. Los resultados mantienen alcance retrospectivo. La equiparación de presupuestos, la potencia objetivo, la priorización reproducible del Top 10, la revisión sistemática, la réplica del paper guía, la robustez por severidad, las ablaciones, la cobertura predictiva y el techo de desempeño requieren su evidencia específica. La corrección documental no los presenta como experimentos completados.
'''
    from coherencia_documental import resumen_revision
    body+=resumen_revision(s)
    (Path(root)/'REVISION_CONSISTENCIA_JBOOK.md').write_text(body,encoding='utf8')
    dest=Path(root)/'book/_static/resultados/revision_documental';dest.mkdir(parents=True,exist_ok=True)
    shutil.copy2(out/'resumen_documental.json',dest/'resumen_documental.json');shutil.copy2(Path(root)/'REVISION_CONSISTENCIA_JBOOK.md',dest/'revision_consistencia.md')
    return report


def verificar_celdas(root,cells,capitulo,estado=None):
    import copy
    esperado=actualizar_celdas(root,copy.deepcopy(cells),capitulo,estado)
    assert len(esperado)==len(cells),(capitulo,'Faltan bloques documentales derivados de resultados')
    for i,(actual,canonico) in enumerate(zip(cells,esperado)):
        if canonico.metadata.get('resultado_documental'):
            assert actual.source==canonico.source,(capitulo,i,canonico.metadata['resultado_documental'],'Narrativa desactualizada respecto de los resultados')


def verificar(root):
    root=Path(root);s=cargar(root);managed=0
    for stem in ['01_datos','09_validacion','10_modelos','10_evaluacion','11_sintesis','12_reproducibilidad']:
        cells=nb.read(root/f'book/{stem}.ipynb',4).cells
        verificar_celdas(root,cells,stem,s)
        managed+=sum(bool(c.metadata.get('resultado_documental')) for c in cells)
    from coherencia_documental import sin_repeticiones
    for path in (root/'book').glob('*.ipynb'):
        for i,c in enumerate(nb.read(path,4).cells):
            if c.cell_type=='markdown':assert c.source==sin_repeticiones(c.source),(path.name,i,'Párrafos duplicados')
    assert managed>0
    return dict(celdas_documentales=managed,fuentes='CSV y JSON conservados',verificacion=True)


def actualizar_contexto(root,s,reporte):
    root=Path(root);p=root/'CONTEXTO_PROYECTO.md';text=p.read_text(encoding='utf8')
    today=date.today();meses=['enero','febrero','marzo','abril','mayo','junio','julio','agosto','septiembre','octubre','noviembre','diciembre'];fecha=f'{today.day} de {meses[today.month-1]} de {today.year}'
    text=text.replace('Verificación local: 89 pruebas correctas','Verificación de la integración de §36: 89 pruebas correctas')
    text=re.sub(r'Actualizado el .*?\. Idioma:',f'Actualizado el {fecha}. Idioma:',text,count=1)
    text=re.sub(r'## Estado vigente · [^\n]+',f'## Estado vigente · {fecha}',text,count=1)
    inv=reporte['inventario'];tables=inv.get('tabla_calculada',0)+inv.get('tabla_markdown',0)
    current=f"La evaluación conserva {s['procedimientos']} procedimientos aprendidos, {s['realizaciones']} realizaciones y {s['referencias']} referencias estacionales. El menor MASE medio corresponde a {s['nombres'][s['ranking'][0]]}: {dec(s['_tablas']['benchmark'].set_index('clave').loc[s['ranking'][0],'MASE'],7)}. El MCS al 90 % retiene {lista(s['nombres'][k] for k in s['mcs'])}. Hay {inv.get('figura',0)} gráficas, {tables} tablas y {inv.get('esquema',0)} esquemas. Los conteos y las conclusiones se regeneran desde resultados; ver §37. Esta actualización es local; no entrenar ni publicar sin solicitud expresa."
    text=re.sub(r'La evaluación conserva .*?(?=\n\n)',current,text,count=1,flags=re.S)
    heading='## 37. Consistencia documental y generación desde resultados'
    posteriores=''
    if heading in text:
        start=text.index(heading)
        next_section=re.search(r'\n## ',text[start+len(heading):])
        if next_section:posteriores=text[start+len(heading)+next_section.start()+1:]
        text=text[:start].rstrip()
    text+='\n\n'+heading+f''' · {fecha}

Las inconsistencias de conteo, precisión, importancia y MCS se corrigen en los capítulos 9, 10 y 11. La revisión vigente se genera en `REVISION_CONSISTENCIA_JBOOK.md`; el capítulo 12 enlaza esa revisión y el resumen de procedencia.

`scripts/narrativa_resultados.py` es la fuente de presentación de conteos y conclusiones. Lee el benchmark, contrastes, MCS, metadatos, pilotos, parada, importancia y latencia. Las celdas Markdown se identifican con `resultado_documental` y no contienen código de generación de texto. `integrar_top10.ajustar_celdas`, `reorganizar_capitulo10.py`, `reconstruir_evaluacion.py`, `actualizar_diagnosticos_jbook.py` y la armonización final de `redaccion_investigacion.py` aplican esa fuente. No volver a corregir los números mediante sustituciones de palabras o cifras antiguas.

Estado derivado: {s['parejas_temporales']} parejas temporales, {s['parejas_semillas']} entre semillas, {s['pilotos']} pilotos externos y {s['procedimientos_parada']} procedimientos con precisión de parada ({s['realizaciones_parada']} registros; {s['parejas_parada']} parejas). La importancia ampliada comprende {s['importancia_ampliada']} procedimientos; sus basales se distinguen de las {s['basales_completos']} comprobaciones de reproducción completa ya conservadas. La latencia comprende {s['latencia_procedimientos']} procedimientos y {s['latencia_registros']} registros de medición.

TimesNet mantiene validación y parada con lote {s['timesnet_lote_desarrollo']}, y prueba y perturbaciones con lote {s['timesnet_lote_prueba']}. La comparación desarrollo–prueba declara esa diferencia; los checkpoints, predicciones y métricas se conservan. Los vacíos metodológicos identificados en la revisión no se presentan como corregidos por esta actualización documental.

El verificador rechaza las celdas gestionadas si su contenido no coincide con los resultados, incluso si el libro construye correctamente. Las pruebas incluyen idempotencia, detección de párrafos obsoletos y cambios simulados de conteos, ranking y MCS. `results/revision_documental_20261120/resumen_documental.json` conserva los valores y huellas de origen. Regenerar con `python scripts/reorganizar_capitulo10.py`; construir con `jupyter-book build book --warningiserror --keep-going` y revisar con `python scripts/verificar_jbook.py`. Ninguno de estos pasos entrena ni despliega.
'''
    if posteriores:text=text.rstrip()+'\n\n'+posteriores
    p.write_text(text,encoding='utf8')


def aplicar(root):
    root=Path(root);s=cargar(root)
    for path in (root/'book').glob('*.ipynb'):
        n=nb.read(path,4);antes=nb.writes(n);n.cells=actualizar_celdas(root,n.cells,path.stem,s)
        if path.stem=='12_reproducibilidad':
            for c in n.cells:
                if c.cell_type=='markdown' and c.source.startswith('## Reconstrucción y revisión de consistencia'):
                    c.source='''## Reconstrucción y revisión de consistencia

La narrativa de resultados se genera mediante `scripts/narrativa_resultados.py` a partir del benchmark, contrastes, MCS, selección y diagnósticos conservados. Las celdas Markdown identifican su función mediante metadatos; las cifras y conclusiones se actualizan durante la reconstrucción.

La {download}`revisión vigente <_static/resultados/revision_documental/revision_consistencia.md>` y el [resumen con huellas de procedencia](_static/resultados/revision_documental/resumen_documental.json) documentan el inventario y las fuentes utilizadas.'''
        if path.stem=='08_literatura':
            for c in n.cells:
                if c.cell_type=='markdown' and c.source.startswith('### Extensiones respaldadas por la revisión'):
                    partes=c.source.split('\n\n');partes[-1]='Las fichas desarrollan las arquitecturas numéricas y documentales evaluadas, incluidas N-BEATS y TimesNet. Cada ficha distingue el mecanismo implementado de su artículo orientador y enlaza el pipeline. Estas incorporaciones no alteran el denominador de estudios empíricos utilizado para contar comparadores.';c.source='\n\n'.join(partes)
        if nb.writes(n)!=antes:nb.write(n,path)
    intro=root/'book/intro.md';text=intro.read_text(encoding='utf8');text=re.sub(r'(?<=junto con )\w+ procedimientos de pronóstico',f"{palabras(s['procedimientos'])} procedimientos de pronóstico",text);intro.write_text(text,encoding='utf8')
    for script in ['narrativa_resultados.py','integrar_top10.py','coherencia_documental.py']:shutil.copy2(root/'scripts'/script,root/'book/_static/codigo_e2'/script)
    from integrar_m4_resultados import aplicar as integrar_propuesta_m4
    integrar_propuesta_m4(root)
    reporte=revision(root,s)
    actualizar_contexto(root,s,reporte)
    from revision_e3_documental import aplicar as revisar_e3
    revisar_e3(root)
    from integrar_cierre_e3 import aplicar as completar_cierre_e3
    completar_cierre_e3(root)
    return reporte

if __name__=='__main__':
    s=aplicar(Path(__file__).resolve().parents[1]);print(json.dumps({k:s[k] for k in ['procedimientos','realizaciones','parejas_temporales','parejas_semillas','mcs']},ensure_ascii=False))

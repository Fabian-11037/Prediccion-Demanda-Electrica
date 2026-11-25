"""Presentación de la robustez desde resultados guardados; no ejecuta redes."""
from __future__ import annotations
import base64,hashlib,json,os,re,shutil,sys
from pathlib import Path
import nbformat as nb
import numpy as np
import pandas as pd
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output
from IPython.core.pylabtools import print_figure
from matplotlib.figure import Figure

ROOT=Path(__file__).resolve().parents[1]
TAG='robustez_congelada'

CURVAS=r'''from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

rob_dir = Path("_static/resultados/robustez")
if not rob_dir.exists():
    rob_dir = Path("book") / rob_dir
rob_resumen = pd.read_csv(rob_dir / "resumen.csv")
rob_original = rob_resumen.query("nivel == 0").sort_values("MASE")
rob_orden = rob_original.clave.tolist()
rob_nombres = rob_original.set_index("clave").modelo.to_dict()
rob_tipos = ["ruido", "interrupcion", "atipicos", "perdida_texto"]
rob_titulos = ["Ruido en la historia numérica", "Interrupción de datos", "Impulsos en las entradas", "Pérdida de documentos"]
rob_ejes = ["Amplitud / DE de entrenamiento", "Horas finales sin observación", "Impulso / DE de entrenamiento", "Probabilidad de pérdida documental"]
rob_colores = {k: plt.get_cmap("tab20")(i) for i, k in enumerate(rob_orden)}
rob_colores["m4_n"] = "#0b3f54"
rob_colores["nuevo_timesnet__N"] = "#af5846"
rob_curvas, rob_axes = plt.subplots(2, 2, figsize=(15, 11.5))
for rob_ax, rob_tipo, rob_titulo, rob_eje in zip(rob_axes.flat, rob_tipos, rob_titulos, rob_ejes):
    for rob_key in rob_orden:
        rob_r = pd.concat([rob_original.loc[rob_original.clave.eq(rob_key)],
                          rob_resumen.loc[rob_resumen.clave.eq(rob_key) & rob_resumen.tipo.eq(rob_tipo)]])
        rob_gruesa = rob_key == "m4_n"
        rob_ax.plot(rob_r.severidad, rob_r.MASE, marker="o", markersize=3,
                    linewidth=2.5 if rob_gruesa else 1.2, alpha=1 if rob_gruesa else .8,
                    color=rob_colores[rob_key], label=rob_nombres[rob_key],
                    linestyle="--" if rob_key.startswith("naive_") else "-")
        if rob_key in ["m4_n", "nuevo_timesnet__N"]:
            rob_ax.fill_between(rob_r.severidad, rob_r.MASE_ic_inf, rob_r.MASE_ic_sup,
                                color=rob_colores[rob_key], alpha=.09)
    rob_ax.set(title=rob_titulo, xlabel=rob_eje, ylabel="MASE")
    rob_ax.set_xticks(sorted(rob_resumen.loc[rob_resumen.tipo.eq(rob_tipo), "severidad"].unique().tolist() + [0]))
    rob_ax.grid(alpha=.2)
rob_handles, rob_labels = rob_axes.flat[0].get_legend_handles_labels()
rob_curvas.legend(rob_handles, rob_labels, loc="lower center", ncol=3, frameon=False, fontsize=9)
rob_curvas.subplots_adjust(bottom=.22, hspace=.3, wspace=.22)
display(rob_curvas)
plt.close(rob_curvas)
'''

MAPA=r'''from matplotlib.colors import SymLogNorm

rob_condiciones = [(tipo, nivel) for tipo in rob_tipos for nivel in [1, 2, 3]]
rob_matriz = np.array([[rob_resumen.loc[(rob_resumen.clave == clave) &
    (rob_resumen.tipo == tipo) & (rob_resumen.nivel == nivel), "degradacion_pct"].iloc[0]
    for tipo, nivel in rob_condiciones] for clave in rob_orden])
rob_max = max(5., float(np.abs(rob_matriz).max()))
rob_mapa, rob_ax = plt.subplots(figsize=(14, 10))
rob_im = rob_ax.imshow(rob_matriz, aspect="auto", cmap="RdBu_r",
    norm=SymLogNorm(linthresh=5, vmin=-rob_max, vmax=rob_max))
rob_ax.set_xticks(range(12), [f"{titulo}\n{nivel}" for titulo in
    ["Ruido", "Interrupción", "Impulsos", "Documentos"] for nivel in [1, 2, 3]])
rob_ax.set_yticks(range(len(rob_orden)), [rob_nombres[k] for k in rob_orden])
rob_ax.tick_params(axis="x", labelsize=9)
for rob_i in range(len(rob_orden)):
    for rob_j in range(12):
        rob_v = rob_matriz[rob_i, rob_j]
        rob_ax.text(rob_j, rob_i, f"{rob_v:+.1f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(rob_v) > max(20, rob_max / 4) else "black")
rob_ax.set(title="Cambio relativo del MASE respecto a la entrada original", xlabel="Tipo de perturbación y severidad")
rob_mapa.colorbar(rob_im, ax=rob_ax, label="Cambio del MASE (%) · escala logarítmica simétrica fuera de ±5 %")
rob_mapa.tight_layout()
display(rob_mapa)
plt.close(rob_mapa)
'''

TABLA=r'''rob_contrastes = pd.read_csv(rob_dir / "contrastes_M4.csv")
rob_vs_timesnet = rob_contrastes.loc[rob_contrastes.comparador.eq("nuevo_timesnet__N")].copy()
rob_tabla = pd.DataFrame({
    "Perturbación": rob_vs_timesnet.tipo.map(dict(zip(rob_tipos, rob_titulos))),
    "Nivel": rob_vs_timesnet.nivel,
    "Mejora de M4 (%)": rob_vs_timesnet.mejora_pct.map(lambda v: f"{v:.2f}"),
    "IC 95 % (%)": [f"[{a:.2f}; {b:.2f}]" for a,b in zip(rob_vs_timesnet.mejora_ic_inf, rob_vs_timesnet.mejora_ic_sup)],
    "p ajustado Holm · δ=3 %": rob_vs_timesnet.p_ajustado.map(lambda v: f"{v:.4f}"),
    "Mejora >3 % respaldada": rob_vs_timesnet.supera_3pct.map({True:"Sí",False:"No"})
})
display(rob_tabla.reset_index(drop=True))
'''


def export_dashboard(root=ROOT):
    root=Path(root);out=root/'results/robustez_congelada_20261124';site=root/'dashboard/site'
    protocol=json.loads((out/'protocolo.json').read_text(encoding='utf-8'))
    quality=json.loads((out/'calidad_entradas_originales.json').read_text(encoding='utf-8'))
    summary=pd.read_csv(out/'resumen.csv');comp=pd.read_csv(out/'contrastes_M4.csv');deg=pd.read_csv(out/'contrastes_degradacion.csv')
    payload=dict(quality=quality,units={'MAE':'kWh por intervalo','RMSE':'kWh por intervalo','MASE':'adimensional','degradacion_pct':'porcentaje relativo'},protocol=protocol,summary=json.loads(summary.to_json(orient='records')),
                 comparisons=json.loads(comp.to_json(orient='records')),degradation=json.loads(deg.to_json(orient='records')),
                 book_section='10_evaluacion.html#robustez-congelada',
                 sources_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in [out/'resumen.csv',out/'contrastes_M4.csv',out/'protocolo.json']})
    (site/'data/robustez.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    for name in ['resumen.csv','contrastes_M4.csv','contrastes_degradacion.csv','protocolo.json']:
        shutil.copy2(out/name,site/'downloads'/('robustez_'+name))
    return payload


def execute(cells,root):
    shell=InteractiveShell.instance();figs=[]
    def figure(fig):
        figs.append(fig)
        shell.display_pub.publish(data={'image/png':print_figure(fig,fmt='png',base64=True),'text/plain':'Robustez calculada desde predicciones congeladas.'})
    shell.display_formatter.ipython_display_formatter.for_type(Figure,figure)
    old=Path.cwd();os.chdir(root)
    try:
        for i,c in enumerate(cells,1):
            if c.cell_type!='code':continue
            with capture_output()as cap:result=shell.run_cell(c.source,store_history=False)
            if result.error_in_exec or result.error_before_exec:raise RuntimeError(result.error_in_exec or result.error_before_exec)
            c.execution_count=i;c.outputs=[]
            if cap.stdout:c.outputs.append(nb.v4.new_output('stream',name='stdout',text=cap.stdout))
            if cap.stderr:c.outputs.append(nb.v4.new_output('stream',name='stderr',text=cap.stderr))
            for o in cap.outputs:c.outputs.append(nb.v4.new_output('display_data',data=o.data,metadata=o.metadata))
    finally:os.chdir(old)
    return shell.user_ns,figs


def aplicar(root=ROOT):
    root=Path(root);out=root/'results/robustez_congelada_20261124'
    if not(out/'resumen.csv').exists():return
    assets=root/'book/_static/resultados/robustez';assets.mkdir(parents=True,exist_ok=True)
    for name in ['resumen.csv','contrastes_M4.csv','contrastes_degradacion.csv','protocolo.json','verificacion.json','paridad_predicciones.csv','metricas_por_origen.parquet','calidad_entradas_originales.json']:
        shutil.copy2(out/name,assets/name)
    code=root/'book/_static/codigo_e3/robustez';code.mkdir(parents=True,exist_ok=True)
    for name in ['evaluar_robustez_congelada.py','analizar_robustez_congelada.py','integrar_robustez_congelada.py']:
        shutil.copy2(root/'scripts'/name,code/name)
    summary=pd.read_csv(out/'resumen.csv');comp=pd.read_csv(out/'contrastes_M4.csv')
    md=lambda s:nb.v4.new_markdown_cell(s.strip(),metadata={TAG:True})
    cc=lambda s:nb.v4.new_code_cell(s.strip(),metadata={TAG:True})
    quality=json.loads((out/'calidad_entradas_originales.json').read_text(encoding='utf-8'))
    cells=[md('''(robustez-congelada)=
## Robustez ante perturbaciones controladas

La evaluación utiliza 96 orígenes diarios de 2022–2025: los días 1 y 15 de cada mes. Se conservan cinco realizaciones por procedimiento aprendido, dos referencias estacionales, las arquitecturas y los escaladores. En M4 se restauran las semillas 101–105 con sus tres expertos y M1; los comparadores utilizan 1000–1004. Cada salida contiene las 24 horas siguientes. TimesNet se ejecuta con lote uno.

Las perturbaciones representan escenarios de estrés impuestos. La frecuencia natural de fallos y extremos no se estima con esta simulación; el benchmark sobre datos originales conserva su interpretación principal.

Se aplican cuatro perturbaciones con tres severidades: ruido gaussiano de amplitud 0,10, 0,25 y 0,50 veces la desviación de entrenamiento; impulsos de 1, 3 y 5 desviaciones en nueve de las 168 horas; interrupciones de las últimas 6, 24 y 48 horas; y pérdida documental con probabilidades 0,25, 0,50 y 0,75. El ruido, los impulsos y la pérdida de documentos tienen tres réplicas. Las realizaciones se comparten entre modelos y son anidadas entre severidades. Se conservan los límites físicos de las variables.

Las desviaciones se calculan con datos de 2008–2017 y solo definen la magnitud de las perturbaciones. En una interrupción, la demanda se mantiene en su último valor anterior y las covariables siguen el tratamiento de ausencia del checkpoint. El calendario permanece conocido. En M4 se recalculan expertos, M1 y consulta de memoria; el archivo histórico y los errores ya observables se mantienen disponibles. La pérdida textual retira documentos sin cambiar las fechas ni el contenido de los restantes.

El MAE y el RMSE de las descargas se expresan en kWh por intervalo. El MASE es adimensional y conserva la escala semanal de entrenamiento de 215.617,90 kWh. Primero se agregan los 24 errores del origen y las réplicas; después se promedian fechas e inicializaciones. Los intervalos al 95 % se obtienen con 9.999 remuestreos conjuntos: bloques circulares de tres meses y remuestreo de cinco semillas dentro de cada modelo. Cada mes contiene los dos orígenes prefijados. Este diagnóstico corresponde al panel de fechas y realizaciones indicado.
'''),cc(CURVAS)]
    quality_text=f"Las ventanas originales contienen {quality['celdas']:,} celdas numéricas y {quality['celdas_ausentes']} ausencias ({quality['porcentaje_ausente']:.2f} %). Los impulsos se imponen en nueve de las 168 horas ({quality['porcentaje_horas_impulsadas']:.2f} %) de cada ventana y sobre las variables numéricas, una densidad fijada para el estrés."
    cells[0].source=cells[0].source.replace('Las perturbaciones representan escenarios de estrés impuestos.',quality_text+'\n\nLas perturbaciones representan escenarios de estrés impuestos.')
    worst=summary.query('nivel==3').sort_values('degradacion_pct',ascending=False).iloc[0]
    m4=summary.query("clave=='m4_n'and nivel==3").set_index('tipo')
    f=lambda v:f'{v:.2f}'.replace('.',',')
    tipo_nombre={'ruido':'ruido numérico','interrupcion':'interrupciones','atipicos':'valores atípicos','perdida_texto':'pérdida documental'}
    descriptions='; '.join(f'{name}: {f(m4.loc[k,"degradacion_pct"])} %'for k,name in [('ruido','ruido'),('interrupcion','interrupción'),('atipicos','atípicos'),('perdida_texto','documentos')])
    cells.append(md(f'''Las curvas muestran la evolución del error con la severidad para los 18 procedimientos. Las bandas corresponden a M4 y TimesNet. En el nivel mayor, los cambios relativos de M4 son: {descriptions}. El archivo histórico conservado forma parte de su información disponible durante la perturbación de la consulta. Las redes numéricas mantienen su salida cuando solo se retiran documentos.'''))
    cells += [cc(MAPA),md(f'''El mapa expresa aumentos y reducciones respecto a la entrada original de cada procedimiento. El mayor cambio en el nivel alto corresponde a {worst.modelo}, bajo {tipo_nombre[worst.tipo]}, con {f(worst.degradacion_pct)} %. El cero de los controles numéricos en pérdida documental verifica que esa intervención no modifica sus entradas. Los ejes de severidad representan magnitudes distintas según el tipo de perturbación.'''),
              md('''### Contrastes de degradación y conservación de la ventaja

La degradación se contrasta con la entrada original mediante 216 hipótesis unilaterales, ajustadas conjuntamente con Holm. La comparación de M4 con 17 procedimientos define otra familia de 204 hipótesis sobre una mejora relativa superior al 3 %. Los intervalos son puntuales; los valores p se ajustan por familia. La siguiente tabla detalla el contraste frente a TimesNet. Los resultados de todas las parejas se conservan en las descargas.'''),cc(TABLA)]
    t=comp.query("comparador=='nuevo_timesnet__N'");n=int(t.supera_3pct.sum())
    inversiones=int(((t.tipo!='perdida_texto') & t.mejora_pct.lt(0)).sum())
    hallazgo=(f'El panel no acredita una mejora superior al 3 % de M4 frente a TimesNet después del ajuste de Holm.' if n==0 else f'M4 conserva una mejora superior al 3 % frente a TimesNet en {n} de las 12 condiciones después del ajuste de Holm.')
    comparacion=f'Bajo las perturbaciones impuestas, la ventaja descriptiva de M4 frente a TimesNet se invierte en {inversiones} de las nueve condiciones numéricas. Las interrupciones y los valores atípicos de mayor amplitud incrementan especialmente su error.'
    cells.append(md(f'''{hallazgo} {comparacion} La magnitud observada y el respaldo estadístico se interpretan por separado. El remuestreo mantiene emparejadas las fechas y no trata las 24 horas como réplicas independientes.

[Métricas completas](_static/resultados/robustez/resumen.csv) · [Contrastes frente a M4](_static/resultados/robustez/contrastes_M4.csv) · [Degradación](_static/resultados/robustez/contrastes_degradacion.csv) · [Pérdidas por origen](_static/resultados/robustez/metricas_por_origen.parquet) · [Protocolo](_static/resultados/robustez/protocolo.json) · [Paridad de inferencia](_static/resultados/robustez/paridad_predicciones.csv).

### Código de evaluación y análisis

La [restauración compartida de redes y entradas](13_inferencia.md) complementa los siguientes módulos de perturbación, inferencia y análisis estadístico.

```{{literalinclude}} _static/codigo_e3/robustez/evaluar_robustez_congelada.py
:language: python
```

```{{literalinclude}} _static/codigo_e3/robustez/analizar_robustez_congelada.py
:language: python
```

La reproducción de la evaluación y su presentación utiliza:

```console
python scripts/evaluar_robustez_congelada.py
python scripts/analizar_robustez_congelada.py
python scripts/integrar_robustez_congelada.py
```
'''))
    for i,c in enumerate(cells):c.id=f'robustez-congelada-{i}'
    ns,figs=execute(cells,root)
    figdir=out/'figuras';figdir.mkdir(exist_ok=True)
    for fig,name in zip(figs,['curvas_robustez','mapa_degradacion']):
        fig.savefig(figdir/(name+'.png'),dpi=180,bbox_inches='tight')
    p=root/'book/10_evaluacion.ipynb';book=nb.read(p,4)
    book.cells=[c for c in book.cells if not c.metadata.get(TAG)]
    j=next((i for i,c in enumerate(book.cells)if c.cell_type=='markdown'and c.source.startswith('## Coste de inferencia')),len(book.cells))
    book.cells[j:j]=cells;nb.validate(book);nb.write(book,p)
    p=root/'book/11_sintesis.ipynb';conclusion=nb.read(p,4)
    conclusion.cells=[c for c in conclusion.cells if not c.metadata.get(TAG)]
    conclusion.cells.append(md(f'''## Robustez de las entradas

El diagnóstico de estrés de 96 fechas y cinco realizaciones por procedimiento cuantifica la respuesta a ruido, interrupciones, valores atípicos y pérdida documental en tres severidades. {hallazgo} {comparacion} Las curvas y los intervalos delimitan la estabilidad del pipeline; los procedimientos sin canal textual mantienen su salida al retirar documentos. La memoria utiliza el archivo histórico disponible mientras se perturba la consulta. El benchmark con datos originales constituye la evaluación principal; la frecuencia real de fallos no se deduce de los escenarios simulados. El {{ref}}`análisis de robustez <robustez-congelada>` desarrolla estos resultados y su alcance temporal.'''))
    conclusion.cells[-1].id='robustez-conclusion'
    nb.write(conclusion,p)
    export_dashboard(root)
    title='## 74. Robustez con tres severidades y pesos congelados · 24 de noviembre de 2026'
    context=f'''Resultados en results/robustez_congelada_20261124. Se evalúan 18 procedimientos, 96 orígenes (días 1 y 15 de cada mes de 2022–2025), cuatro tipos de perturbación y tres severidades por tipo. Cinco realizaciones completas por modelo aprendido (1000–1004; M4 101–105) y dos referencias deterministas: 82 archivos. En M4 no se reemplazan expertos entre semillas ni se simulan las cinco bases completas no disponibles para 106–110. El benchmark original de diez semillas de M4 y cinco de comparadores permanece intacto. Las {quality['celdas']} celdas de entrada del panel tienen {quality['celdas_ausentes']} ausencias. Las perturbaciones son escenarios de estrés hipotéticos, no estimaciones de fallos habituales; nueve horas con impulsos equivalen al 5,36 % de cada ventana. La frecuencia natural de atípicos no se cuantifica aquí.

Ruido: 0,10/0,25/0,50 DE de 2008–2017; atípicos: ±1/3/5 DE en nueve horas; interrupciones: 6/24/48 h (demanda por último valor y covariables ausentes); pérdida documental Bernoulli: 25/50/75 %. Tres réplicas estocásticas, mismas perturbaciones entre modelos y niveles anidados. Calendario y banco histórico de memoria conservados; se reejecuta el pipeline completo de M4 sobre la consulta perturbada. TimesNet utiliza lote uno. No confundir con corrupción del archivo histórico completo, sensibilidad de hiperparámetros, ablación con reajuste o robustez al tamaño de muestra.

Métricas de 24 horas agregadas primero por origen y réplica; MASE con Q=215617.90361589016 kWh. Bootstrap jerárquico de 9999 réplicas: bloques circulares de tres meses (dos orígenes por mes) y remuestreo de cinco semillas. Dos familias Holm: 216 contrastes de deterioro frente a entrada limpia y 204 comparaciones de M4 frente a 17 comparadores con margen 3 %. IC95% puntuales. M4 supera el margen frente a TimesNet en {n}/12 condiciones; su ventaja descriptiva se invierte en {inversiones}/9 condiciones numéricas. Los textos de resultados se calculan desde CSV, sin introducir cifras manuales.

Código canónico: scripts/evaluar_robustez_congelada.py, analizar_robustez_congelada.py e integrar_robustez_congelada.py. El integrador agrega el apartado robustez-congelada de 10_evaluacion.ipynb con figuras, interpretaciones integradas, tabla de contraste y código completo. Exporta dashboard/site/data/robustez.json y las descargas. La armonización y exportación del dashboard conservan este bloque al regenerar. Los pesos y escaladores se comprueban por hash, y la inferencia limpia reproduce las predicciones guardadas. No entrenar, publicar ni subir por esta actualización. La evaluación se limita al panel prefijado y no acredita por sí sola el cierre integral de robustez del E3.
'''
    for rel in ['CONTEXTO_PROYECTO.md','dashboard/CONTEXTO_DASHBOARD.md']:
        p=root/rel;s=p.read_text(encoding='utf-8')
        if title in s:s=re.sub(re.escape(title)+r'.*?(?=\n## \d+\.|\Z)',lambda _:title+'\n\n'+context,s,flags=re.S,count=1)
        else:s=s.rstrip()+'\n\n'+title+'\n\n'+context
        p.write_text(s,encoding='utf-8',newline='\n')
    shutil.copy2(root/'CONTEXTO_PROYECTO.md',root/'continuidad/CONTEXTO_PROYECTO.md')
    print('Robustez integrada desde resultados: JBook, dashboard y contexto.')


if __name__=='__main__':
    if '--solo-dashboard'in sys.argv:export_dashboard()
    else:aplicar()

"""Armonización editorial y control de contratos entre capítulos del JBook."""
from pathlib import Path
import csv,json,re


def armonizar(books,md,root):
    for cells in books.values():
        for cell in cells:
            s=cell.source
            s=s.replace('GWh/h','GWh por intervalo')
            if cell.cell_type=='code':
                s=s.replace('Cambio semanal (GWh por intervalo)', 'Cambio semanal'+chr(92)+'n(GWh por intervalo)').replace('MAD móvil, 90 días (GWh por intervalo)', 'MAD móvil, 90 días'+chr(92)+'n(GWh por intervalo)')
                if "p_lb=float(lb.loc[14,'lb_pvalue'])" in s:
                    s+="\np_lb_texto='<1e-300' if p_lb==0 else f'{p_lb:.3g}'\n"
                    s=s.replace('display(diagnosticos)',"display(diagnosticos.style.format({'p_sin_ajustar':lambda p:'<1e-300' if p==0 else f'{p:.3g}'},precision=4))")
                if 'adf_diferencia=float(registros[1]' in s:
                    s+="\nadf_diferencia_texto='<0.001' if adf_diferencia==0 else f'{adf_diferencia:.3g}'\n"
                if 'display(pd.DataFrame(registros))' in s:
                    s=s.replace('display(pd.DataFrame(registros))',"display(pd.DataFrame(registros).style.format({'ADF_p':lambda p:'<0.001' if p==0 else f'{p:.3g}','KPSS_p_tabulado':'{:.2f}'},precision=4))")
                if "ciudad_peso=geo.loc[geo.peso.idxmax(),'ciudad']" in s:
                    s+="\nciudad_peso={'bogota':'Bogotá','medellin':'Medellín','cucuta':'Cúcuta'}.get(ciudad_peso,ciudad_peso.title())\n"
                    s=s.replace('barh(geo.ciudad,',"barh(geo.ciudad.str.title().replace({'Bogota':'Bogotá','Medellin':'Medellín','Cucuta':'Cúcuta'}),")
                if "groupby(['hora','no_laboral'])" in s:
                    s=s.replace('fig.tight_layout(); display(fig)',"axes[0].legend(['Resto de días','Domingo/festivo']); fig.tight_layout(); display(fig)")
                    s+='\nreduccion_no_laboral=-diferencia\n'
            else:
                s=s.replace('Se formula un pronóstico de series de tiempo multivariado con salida directa de 24 intervalos y 168 horas de historia.',
                    'Se formula un pronóstico de una serie objetivo con covariables, salida directa de 24 intervalos y 168 horas de historia.')
                s=s.replace('con pronóstico multivariado directo de 24 pasos y rezagos del objetivo.','con pronóstico directo de 24 pasos, covariables y rezagos del objetivo.')
                s=s.replace('p = {p_lb:.3g}','p {p_lb_texto}').replace('ADF p={adf_diferencia:.3g}','ADF p{adf_diferencia_texto}')
                s=s.replace('sin descartar rupturas ni garantizar estabilidad futura.', 'sin descartar rupturas ni garantizar estabilidad futura. En ADF, el cero devuelto fuera del rango de la aproximación de MacKinnon se informa conservadoramente como p<0,001.')
                s=s.replace('La heterocedasticidad se revisará después sobre errores predictivos genuinos.','Los diagnósticos de heterocedasticidad sobre residuos predictivos se presentan en §10.9.')
                s=s.replace('La demanda media en días no laborales difiere un {diferencia:.1f} % de la de días laborales.',
                    'La demanda media de domingos y festivos es un {reduccion_no_laboral:.1f} % inferior a la del resto de días.')
                s=s.replace('La prueba 2022–2025 queda fuera de este análisis. La escala semanal de MASE procede de entrenamiento; con demanda creciente no representa una escala intercambiable entre particiones.',
                    'El diagrama utiliza únicamente desarrollo 2000–2021; las métricas de prueba 2022–2025 se presentan en el capítulo 10. El denominador de MASE permanece fijo en la referencia histórica 2000–2017: permite comparar valores en la misma escala, aunque los periodos pueden tener distinto nivel y dificultad de pronóstico.')
                s=s.replace('La escala $q_{168}$ procede exclusivamente de entrenamiento 2000–2017.',
                    'La escala $q_{168}$ procede del bloque de referencia histórico 2000–2017, anterior a los años externos de validación.')
                s=s.replace('Huber utiliza δ=0,05 en la escala normalizada; este parámetro de pérdida es distinto del margen relativo de mejora del 3 %.',
                    'Huber utiliza κ=0,05 en la escala normalizada.')
                s=s.replace('El piloto existe en los artefactos; el cálculo de dimensionamiento que se presenta en el informe es retrospectivo y no acredita que determinara las cinco semillas ejecutadas. ', '')
                s=s.replace('No se declara un δ de 0,0163 o de 0,0226 MASE; esas cantidades eran conversiones de escala dependientes del periodo y no son el margen del contraste actual. ', '')
                s=s.replace('se excluyen resultados de entrenamiento.','los resultados de modelos se identifican por separado mediante su registro de artefactos.')
                s=s.replace('## Identificación de los archivos','## Manifiesto de entradas del análisis')
                s=s.replace('El EDA no demuestra todavía superioridad predictiva, causalidad ni una ganancia atribuible al texto.',
                    'El EDA caracteriza asociaciones y cobertura, sin establecer causalidad ni una ganancia atribuible al texto.')
                s=s.replace('La similitud NMF media es {sim_media:.3f} y la mínima {sim_min:.3f}.',
                    'La similitud NMF media es {sim_media:.3f} y la mínima {sim_min:.3f}, a la precisión mostrada.')
                # La trazabilidad detallada permanece en CSV; no repetir rutas de exportación por fila.
                s=re.sub(r' \(Abstract exportado; fuente original IEEE Xplore Citation Plain Text Download [^)]*\)\.', '', s)
                s=s.replace('| contexto_revision |','| revisión |').replace('| mercado_relacionado |','| mercado relacionado |').replace('| probabilistico |','| probabilístico |')
                s=s.replace('Time-MMD, NSW-EPNews y los estudios de precio se conservan como trabajos de otros autores con aportaciones metodológicas complementarias; no se cuentan como evidencia empírica de demanda.',
                    'Time-MMD, de Liu et al., y NSW-EPNews, de Bi et al., aportan referencias metodológicas para integrar series y documentos; sus resultados no se agregan a la evidencia empírica de demanda.')
                # Mantener títulos, enlaces, identificadores y código en su forma original.
                fragmentos=re.split(r'(```[\s\S]*?```|`[^`]*`|\[[^\]]*\]\([^)]+\))',s)
                for i in range(0,len(fragmentos),2):
                    fragmentos[i]=re.sub(r'\babstracts\b','resúmenes',fragmentos[i],flags=re.I)
                    fragmentos[i]=re.sub(r'\bAbstract\b','Resumen',fragmentos[i])
                    fragmentos[i]=re.sub(r'\babstract\b','resumen',fragmentos[i])
                s=''.join(fragmentos)
            cell.source=s
            if cell.metadata.get('plantilla_resultados'):cell.metadata['plantilla_resultados']=s
    # Consolidar la discusión y el alcance en una sección única.
    cells=books['10_modelos'];nuevo=[]
    for cell in cells:
        if cell.cell_type=='markdown' and cell.source.startswith('## Discusión crítica y margen disponible'):continue
        if cell.cell_type=='markdown' and cell.source.startswith('## 10.8.'):
            cell=md('''## 10.8. Discusión crítica, alcance y margen disponible

La MLP establece el baseline empírico de estas configuraciones y obtiene el menor error agregado. Las t pareadas con Holm y los contrastes temporales detectan diferencias entre las tres redes en los datos evaluados; los contrastes por rangos no distinguen todas las parejas. ARX y CNN1D presentan medias próximas, sin una diferencia temporal significativa ni equivalencia demostrada por TOST. Los [resultados por método](10_evaluacion.ipynb) delimitan estas conclusiones.

La representación directa de rezagos y calendario de la MLP es coherente con los ciclos observados. La recurrencia de LSTM y los filtros compartidos de CNN1D no garantizan menor error: arquitectura, regularización, escalado y presupuesto intervienen conjuntamente. La generalización se evalúa por años, horas y tipos de día, además de las curvas de aprendizaje. El presupuesto desigual impide atribuir diferencias únicamente a la familia.

El margen de relevancia práctica es relativo: δ=3 %. La planificación aproximada exige más de cinco semillas para detectar ese margen con potencia del 80 %. Detectar diferencias observadas mayores no contradice esa limitación. La cronología de selección y la exposición al conjunto de prueba, documentadas en el capítulo 9, limitan la interpretación confirmatoria; los resultados son exploratorios.

La MLP contiene más parámetros que las otras redes, mientras LSTM requiere más tiempo en las condiciones registradas. El ajuste se ejecutó con PyTorch 2.9.1, ROCm 6.4 y GPU AMD de aproximadamente 12 GB. La latencia de inferencia se midió por separado en CPU; ambos tiempos describen procedimientos y dispositivos distintos. Los checkpoints, escaladores, historiales y predicciones se identifican mediante SHA-256 y el [pipeline completo](10_pipeline.md).

El ingenuo semanal constituye una referencia operativa, no una cota de ruido irreducible. Sin mediciones replicadas ni un modelo validado de error instrumental, esa cota no se identifica a partir de esta única serie; tampoco la determina el residuo del mejor modelo. Las mejoras publicadas en otros sistemas no son transferibles numéricamente al SIN. La siguiente etapa debe evaluar covariables y documentos disponibles al origen mediante ablaciones, presupuesto comparable y un protocolo prospectivo.''')
        nuevo.append(cell)
    books['10_modelos']=nuevo
    # El alcance del código y los archivos derivados queda trazado en reproducción.
    books['12_reproducibilidad'].append(md('## Reconstrucción y revisión de consistencia\n\nEl generador integra modelos_actualizados_jbook.py, correcciones_e1_jbook.py y consistencia_jbook.py. La construcción del informe utiliza artefactos guardados; la medición de CPU se conserva en latencia_inferencia.csv y entorno_inferencia.json. Para repetir esa medición se utiliza un entorno separado con requirements-inferencia.lock.txt y scripts/medir_inferencia.py. No se ajustan pesos durante el análisis. La revisión global contrasta terminología, unidades, particiones, margen, cifras, inferencia, referencias y enlaces con los archivos fuente; sus resultados se registran en REVISION_CONSISTENCIA_JBOOK.md.'))


def armonizar_archivos(root):
    p=root/'book/intro.md';s=p.read_text(encoding='utf-8')
    s=s.replace('El libro presenta los datos, su calidad, distribución, estacionalidad, dependencia, covariables, titulares y protocolo temporal.',
        'El libro presenta los datos, su calidad, distribución, estacionalidad, dependencia, covariables, titulares y protocolo temporal, junto con el benchmark numérico de MLP, LSTM y CNN1D y las referencias ARX y estacionales.')
    p.write_text(s,encoding='utf-8')
    p=root/'book/diccionario.md';s=p.read_text(encoding='utf-8').replace('de el día','del día');p.write_text(s,encoding='utf-8')
    p=root/'data/processed/diccionario_datos.json';r=json.loads(p.read_text(encoding='utf-8'))
    for item in r:item['descripcion']=item['descripcion'].replace('de el día','del día')
    p.write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf-8')
    p=root/'data/processed/diccionario_datos.csv';rows=list(csv.DictReader(p.open(encoding='utf-8-sig')))
    for item in rows:item['descripcion']=item['descripcion'].replace('de el día','del día')
    with p.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

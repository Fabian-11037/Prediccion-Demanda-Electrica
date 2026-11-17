"""Integra metodología y resultados conservados; no ejecuta entrenamiento."""
from pathlib import Path
import ast,base64,copy,hashlib,json,re,shutil
import nbformat as nb
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output
from IPython.core.pylabtools import print_figure
from matplotlib.figure import Figure


def archivos(root):
    out=root/'results/reorganizacion_capitulo10';src=out/'fuentes';src.mkdir(parents=True,exist_ok=True)
    for nombre in ['10_modelos.ipynb','10_evaluacion.ipynb','10_estadistica_e2.ipynb','10_pipeline.md']:
        p=root/'book'/nombre;q=src/nombre
        if not q.exists():
            if not p.exists():raise FileNotFoundError(p)
            shutil.copy2(p,q)
    p=src/'fundamentos_original.md'
    if not p.exists():shutil.copy2(root/'book/arquitecturas/fundamentos.md',p)
    return out,src


def separar(cells):
    """Separa secciones Markdown sin interpretar encabezados dentro de código."""
    partes={'_intro':[]};clave='_intro'
    for c in copy.deepcopy(cells):
        if c.cell_type!='markdown':partes[clave].append(c);continue
        bloque=[];fence=False
        for line in c.source.splitlines(keepends=True):
            if line.lstrip().startswith('```'):fence=not fence
            if not fence and line.startswith('## '):
                if ''.join(bloque).strip():partes[clave].append(nb.v4.new_markdown_cell(''.join(bloque).strip()))
                clave=line[3:].strip();partes.setdefault(clave,[]);bloque=[]
            bloque.append(line)
        if ''.join(bloque).strip():partes[clave].append(nb.v4.new_markdown_cell(''.join(bloque).strip()))
    return partes


class Privadas(ast.NodeTransformer):
    def __init__(self,names):self.names=names
    def visit_Name(self,n):
        if n.id in self.names:n.id='e2_'+n.id
        return n
    def visit_arg(self,n):
        if n.arg in self.names:n.arg='e2_'+n.arg
        return n


class SoloLectura(ast.NodeTransformer):
    def visit_Expr(self,n):
        # Conservar cálculos y presentación sin sobrescribir artefactos derivados.
        if isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute) and n.value.func.attr in {'to_csv','write_text'}:
            return None
        return self.generic_visit(n)
    def visit_For(self,n):
        n=self.generic_visit(n)
        return n if n.body else None


def privadas(cells):
    result=copy.deepcopy(cells);names=set();imports=set()
    for c in result:
        if c.cell_type!='code':continue
        for n in ast.walk(ast.parse(c.source)):
            if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Store):names.add(n.id)
            if isinstance(n,(ast.Import,ast.ImportFrom)):
                imports.update(a.asname or a.name.split('.')[0] for a in n.names)
    names-=imports
    for c in result:
        if c.cell_type=='code':c.source=ast.unparse(ast.fix_missing_locations(Privadas(names).visit(ast.parse(c.source))))
    return result


def base_cells(cells):
    result=copy.deepcopy(cells)
    for c in result:
        if c.cell_type=='code':c.source=ast.unparse(ast.fix_missing_locations(SoloLectura().visit(ast.parse(c.source))))
        else:
            c.source=re.sub(r'(?m)^## 10\.\d+\. ', '## ',c.source)
            c.metadata.pop('plantilla_resultados',None)
    return result


def portada(root):
    return [nb.v4.new_markdown_cell('''# 10. Modelos, arquitecturas y evaluación

El capítulo presenta primero la composición matemática de los modelos y sus pipelines de entrenamiento y prueba. Después reúne los resultados, la incertidumbre y los diagnósticos en una evaluación integrada.

Las familias ARX, MLP, LSTM y CNN1D reciben demanda y calendario. PatchTST, N-HiTS, TFT, TCN e iTransformer utilizan el conjunto numérico ampliado; TaTS, TGForecaster y la fusión tardía evalúan además documentos oficiales. Se comparan catorce procedimientos completados y dos referencias estacionales.

## Secuencia de lectura

La {ref}`notación y las operaciones compartidas <fundamentos-arquitecturas>` introducen las entradas, la normalización, la atención y la optimización. Las fichas que siguen desarrollan cada arquitectura, su esquema y el código correspondiente. La {ref}`evaluación integrada <evaluacion-e2>` reúne luego selección, aprendizaje, precisión, inferencia, diagnóstico e importancia de variables.

La comparación distingue los conjuntos de entrada y los presupuestos de ajuste. Las diferencias de error entre procedimientos no se atribuyen exclusivamente a la arquitectura.''')]


def construir(root):
    out,src=archivos(root)
    model=nb.read(src/'10_modelos.ipynb',4);legacy=nb.read(src/'10_evaluacion.ipynb',4)
    nuevo=nb.read(src/'10_estadistica_e2.ipynb',4)
    parts=separar(privadas(nuevo.cells))
    intro=parts['_intro']
    intro[0].source=intro[0].source.replace('# 10.23. Resultados y evaluación estadística','# 10.14. Resultados y evaluación integrada')
    intro[0].source+='\n\nLos diagnósticos de precisión, generalización, observado–predicho, segmentos y atípicos abarcan los catorce procedimientos disponibles. La importancia por perturbación y la latencia CPU conservan el alcance de las cuatro familias medidas.'
    setup=base_cells([model.cells[3]])
    c=nb.v4.new_markdown_cell
    cells=intro+setup
    cells.append(c('''## Entrenamiento, selección y precisión del piloto

La selección utiliza validación cronológica y una reserva interna para parada. Las fichas documentan la arquitectura y el pipeline antes de presentar resultados. PatchTST y N-HiTS numéricos conservan 130 ajustes cada uno; los paquetes adjuntos, 40 por variante. Los presupuestos y las máscaras de ausencia no son idénticos entre fuentes.

En las cuatro familias de demanda/calendario, el filtro evaluó 20 candidatos de MLP, 20 de ARX, 24 de LSTM y 26 de CNN1D. Se confirmaron finalistas con dos semillas y cuatro años externos, y se realizaron pilotos con cinco semillas. Los siguientes diagnósticos caracterizan ese grupo; no constituyen un dimensionamiento común para todas las arquitecturas.'''))
    # Curvas de cinco semillas de todos los procedimientos reemplazan la figura de una semilla.
    cells+=base_cells(model.cells[10:15])
    cells+=parts['Curvas de aprendizaje']
    cells+=parts['Desempeño y precisión temporal']
    cells+=base_cells(model.cells[24:33])
    for x in cells:
        if x.cell_type=='markdown':
            x.source=x.source.replace('## Validación y generalización','## Validación y generalización del grupo demanda/calendario')
            x.source=x.source.replace('## 10.6. Validación y generalización','## Validación y generalización del grupo demanda/calendario')
            x.source=x.source.replace('## 10.7. Demanda observada frente a predicción','## Demanda observada frente a predicción: demanda/calendario')
    cells+=parts['Error por horizonte']
    seg=base_cells(legacy.cells[3:6])
    seg[0].source='''## Segmentos de evaluación: demanda y calendario

El análisis complementario distingue hora, año, tipo de día y cuartil de demanda para las cuatro familias con las mismas entradas. Los errores se promedian primero entre inicializaciones y después entre días del segmento. Estos segmentos describen heterogeneidad del error; no se utilizan para volver a seleccionar modelos con la prueba.'''
    seg[1].source='''panel = pd.read_parquet(ROOT/'data/processed/panel_horario.parquet',columns=['cal_es_festivo'])
festivo = panel.loc[fechas,'cal_es_festivo'].to_numpy().astype(bool)
tipo = np.where((fechas.day_of_week==6)|festivo,'Domingo/festivo',np.where(fechas.day_of_week==5,'Sábado','Lun–vie'))
segmentos = {'Año':fechas.year.astype(str),'Tipo de día':tipo,'Nivel de demanda':pd.qcut(y.mean(axis=1),4,labels=['Q1','Q2','Q3','Q4']).astype(str)}
fig, axes = plt.subplots(1,3,figsize=(14,4))
for ax,(nombre,grupos) in zip(axes,segmentos.items()):
    orden = sorted(set(grupos)) if nombre!='Tipo de día' else ['Lun–vie','Sábado','Domingo/festivo']
    for m in modelos:
        valores = [loss_d[m][np.asarray(grupos)==g].mean() for g in orden]
        ax.plot(orden,valores,marker='o',label=etiquetas[m])
    ax.set(xlabel=nombre,ylabel='MASE');ax.legend(fontsize=8)
fig.tight_layout();display(fig);plt.close(fig)
pico_hora = int(np.argmax(loss_h['mlp'].mean(axis=0)))
anos_mlp = pd.Series(loss_d['mlp'],index=fechas).groupby(fechas.year).mean()
peor_ano, mejor_ano = int(anos_mlp.idxmax()), int(anos_mlp.idxmin())
'''
    cells+=seg
    for key in ['Aporte de texto oficial y equivalencia','Comparaciones entre semillas y tamaño del efecto','Friedman, post hoc y conjunto de confianza','Diagnóstico de residuos y calibración']:
        cells+=parts[key]
    # Mantener la información única de variación sin repetir rangos parciales.
    varcode='''componentes = []
for m in modelos:
    valores = np.abs(pred[m] - y[None, :, :]).mean(axis=2) / escala
    media = valores.mean()
    semilla = np.mean((valores.mean(axis=1) - media)**2)
    fecha = np.mean((valores.mean(axis=0) - media)**2)
    interaccion = np.mean((valores-valores.mean(axis=1)[:,None]-valores.mean(axis=0)[None,:]+media)**2)
    componentes.append({'Modelo':etiquetas[m], 'Inicialización':semilla,'Fecha':fecha,'Interacción':interaccion})
comp = pd.DataFrame(componentes).set_index('Modelo')
fig, ax = plt.subplots(figsize=(9,4))
comp.div(comp.sum(axis=1),axis=0).plot.bar(stacked=True,ax=ax,rot=0)
ax.set(ylabel='Fracción de variación descriptiva',xlabel='');ax.legend(fontsize=9)
fig.tight_layout();display(fig);plt.close(fig)
'''
    cells+=[c('''## Variación entre fechas e inicializaciones

La descomposición de pérdida de las cuatro familias con demanda/calendario separa efectos descriptivos de fecha, semilla e interacción. No trata las semillas como nuevas fechas ni presupone independencia entre días.'''),nb.v4.new_code_cell(varcode),c('''La variación asociada a fechas describe días difíciles compartidos por las inicializaciones. La interacción recoge cambios de error que no se explican aditivamente por fecha o semilla. Estas componentes son descriptivas y no un ANOVA con fechas independientes; complementan los intervalos temporales y entre semillas.''')]
    outlier=base_cells(legacy.cells[32:35])
    outlier[1].source='''observado = pd.read_parquet(ROOT/'data/processed/panel_horario.parquet',columns=['y_demanda_real_kwh'])['y_demanda_real_kwh']
tiempos = pd.DatetimeIndex([d+pd.Timedelta(hours=h) for d in fechas for h in range(24)])
'''+outlier[1].source
    cells+=outlier
    cells.append(c('## Coste de inferencia medido\n\n'+model.cells[34].source.split('\n\n',1)[1]))
    cost=base_cells([model.cells[22]])[0]
    # Eliminar la repetición de la tabla de métricas: conservar exclusivamente coste.
    tree=ast.parse(cost.source);tree.body=tree.body[1:];cost.source=ast.unparse(tree)
    cells+=[cost,c('''La tabla compara coste entre las cuatro familias bajo el protocolo de inferencia indicado. La arquitectura y el número de parámetros no determinan por sí solos la latencia. Las mediciones de entrenamiento en dispositivos distintos no se interpretan como una comparación controlada de eficiencia de todas las arquitecturas.''')]
    cells+=base_cells(legacy.cells[38:46])
    cells+=parts['Discusión y alcance del Entregable 2']
    cells+=parts['Tablas y reproducción']
    # Homogeneizar referencias y eliminar conclusiones parciales que aparenten ser globales.
    for x in cells:
        if x.cell_type=='markdown':
            x.source=x.source.replace('La MLP presenta el menor MAE agregado','En el grupo demanda/calendario, la MLP presenta el menor MAE agregado')
            x.source=x.source.replace('MLP registra el menor MAE semanal','En este grupo, MLP registra el menor MAE semanal')
            x.source=x.source.replace('## Importancia de variables y bloques de rezagos','## Importancia de variables del grupo demanda/calendario')
            x.metadata.pop('plantilla_resultados',None)
    from completar_evaluacion_jbook import ampliar
    from seleccion_patchtst import ajustar_celdas
    cells=ajustar_celdas(root,ampliar(root,cells))
    for cell in cells:
        if cell.cell_type=='code':
            cell.source=cell.source.replace('for fila in indice.itertuples():\n    p = SNAP / fila.archivo',"for fila in indice.itertuples():\n    if fila.archivo.endswith(('.pt', '.pth')): continue\n    p = SNAP / fila.archivo")
    return cells


def ejecutar(cells):
    shell=InteractiveShell.instance();shell.reset(new_session=True)
    shell.display_formatter.formatters['image/png'].for_type(Figure,lambda f:base64.b64encode(print_figure(f,fmt='png')).decode())
    count=0
    for c in cells:
        if c.cell_type=='code':ast.parse(c.source)
    for i,c in enumerate(cells):
        if c.cell_type!='code':continue
        count+=1
        with capture_output() as cap:result=shell.run_cell(c.source,store_history=False)
        if result.error_in_exec or result.error_before_exec:
            raise RuntimeError(f'Celda integrada {i}: {result.error_in_exec or result.error_before_exec}')
        c.execution_count=count;c.outputs=[]
        if cap.stdout:c.outputs.append(nb.v4.new_output('stream',name='stdout',text=cap.stdout))
        if cap.stderr:c.outputs.append(nb.v4.new_output('stream',name='stderr',text=cap.stderr))
        for o in cap.outputs:c.outputs.append(nb.v4.new_output('display_data',data=o.data,metadata=o.metadata))
    return count


def fundamentos(root):
    _,src=archivos(root)
    text=(src/'fundamentos_original.md').read_text(encoding='utf-8')
    text=text.replace('# 10.10.1. Notación y operaciones compartidas','# 10.1. Fundamentos y preprocesamiento compartido')
    partes=separar([nb.v4.new_markdown_cell((src/'10_pipeline.md').read_text(encoding='utf-8'))])
    text+='\n\n## Código compartido de preparación\n\nLas fichas muestran cada red y su ajuste. Los siguientes bloques reúnen las operaciones reutilizadas de partición, construcción de ventanas, escalado y evaluación.\n\n'
    for k in ['Partición temporal y parada interna','Datos, ventanas, escalas y referencias','Métricas']:
        text+='\n\n'.join(c.source for c in partes[k])+'\n\n'
    opciones=next(v for k,v in partes.items() if k.startswith('Modelos densos y opciones'))
    codigo='\n'.join(c.source for c in opciones)
    fragmentos=re.findall(r'```python\s*\n(.*?)```',codigo,re.S)
    seleccion=[]
    for frag in fragmentos:
        for node in ast.parse(frag).body:
            if isinstance(node,(ast.Import,ast.ImportFrom)) or (
                isinstance(node,(ast.ClassDef,ast.FunctionDef)) and node.name in {'Opciones','resolver_dispositivo','fijar_semillas'}):
                seleccion.append(ast.unparse(node))
    if seleccion:
        text+='## Opciones de ejecución compartidas\n\nLas clases específicas están en sus fichas; este bloque conserva la selección de dispositivo, la fijación de semillas y las opciones reutilizadas por los pipelines.\n\n```python\n'+'\n\n'.join(seleccion)+'\n```\n'
    (root/'book/arquitecturas/fundamentos.md').write_text(text,encoding='utf-8')


def escribir_toc(root):
    p=root/'book/_toc.yml';text=p.read_text(encoding='utf-8')
    start=text.index('  - file: 10_modelos\n');end=text.index('  - file: 11_sintesis\n',start)
    mods=json.loads((root/'results/arquitecturas_e2/indice_documentacion.json').read_text(encoding='utf-8'))
    seccion='  - file: 10_modelos\n    sections:\n      - file: arquitecturas/fundamentos\n'
    seccion+=''.join('      - file: arquitecturas/'+m['key']+'\n' for m in mods)
    seccion+='      - file: 10_evaluacion\n'
    p.write_text(text[:start]+seccion+text[end:],encoding='utf-8')


def reorganizar(root,actualizar_eda=False):
    root=Path(root).resolve();out,src=archivos(root)
    cells=construir(root);n=ejecutar(cells)
    meta={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'},'capitulo10_integrado':True}
    nb.write(nb.v4.new_notebook(cells=cells,metadata=meta),root/'book/10_evaluacion.ipynb')
    cobertura=root/'results/evaluacion_e2/cobertura_diagnosticos.csv'
    shutil.copy2(cobertura,root/'book/_static/resultados/evaluacion_e2'/cobertura.name)
    for nombre in ['resumen_piloto_integrado.csv','dimensionamiento_piloto_integrado.csv',
                   'precision_desarrollo_resumen.csv','precision_desarrollo_semillas.csv','generalizacion_integrada.csv']:
        shutil.copy2(root/'results/evaluacion_e2'/nombre,root/'book/_static/resultados/evaluacion_e2'/nombre)
    nb.write(nb.v4.new_notebook(cells=portada(root),metadata=meta),root/'book/10_modelos.ipynb')
    fundamentos(root)
    mods=json.loads((root/'results/arquitecturas_e2/indice_documentacion.json').read_text(encoding='utf-8'))
    for i,m in enumerate(mods,2):
        p=root/'book/arquitecturas'/(m['key']+'.md');t=p.read_text(encoding='utf-8')
        t=re.sub(r'(?m)^# 10\.\d+\. ',f'# 10.{i}. ',t)
        t=t.replace('{doc}`../10_pipeline`','{ref}`preprocesamiento compartido <fundamentos-arquitecturas>`')
        p.write_text(t,encoding='utf-8')
    escribir_toc(root)
    for p in (root/'book').glob('*.ipynb'):
        x=nb.read(p,4);changed=False
        for c in x.cells:
            if c.cell_type!='markdown':continue
            for field in ['source']:
                s=c.source
                s=s.replace('§10.9','{ref}`evaluación integrada <evaluacion-e2>`')
                s=s.replace('(10_pipeline.md)','(arquitecturas/fundamentos.md)')
                s=s.replace('(10_estadistica_e2.ipynb)','(10_evaluacion.ipynb)')
                if s!=c.source:c.source=s;changed=True
            if 'plantilla_resultados' in c.metadata:
                c.metadata.plantilla_resultados=c.metadata.plantilla_resultados.replace('§10.9','{ref}`evaluación integrada <evaluacion-e2>`')
        if p.stem=='11_sintesis':
            for c in x.cells:
                if c.cell_type=='markdown' and c.source.startswith('## Resultado del benchmark'):
                    c.source=c.source.replace('## Resultado del benchmark','## Referencia con demanda y calendario').replace('La MLP presenta el menor MASE','Entre las cuatro familias con demanda y calendario, la MLP presenta el menor MASE')
                    changed=True
        if changed:nb.write(x,p)
    # Retirar páginas duplicadas sólo después de conservar una copia verificable.
    retiros=out/'paginas_retiradas';retiros.mkdir(exist_ok=True)
    for nombre in ['10_pipeline.md','10_estadistica_e2.ipynb']:
        p=root/'book'/nombre
        if not p.is_file():continue
        assert p.resolve().parent==(root/'book').resolve()
        h=hashlib.sha256(p.read_bytes()).hexdigest()
        copia=retiros/(p.stem+'_'+h[:12]+p.suffix)
        if not copia.exists():shutil.copy2(p,copia)
        assert hashlib.sha256(copia.read_bytes()).hexdigest()==h
        p.unlink()
    from completar_evaluacion_jbook import armonizar_secciones
    armonizar_secciones(root)
    if actualizar_eda:
        from revision_integral_jbook import actualizar_capitulos
        actualizar_capitulos(root)
    from esquemas_presentacion import instalar as instalar_esquemas_presentacion
    instalar_esquemas_presentacion(root)
    from seleccion_patchtst import armonizar as armonizar_patchtst
    armonizar_patchtst(root)
    from redaccion_investigacion import aplicar as revisar_redaccion
    revisar_redaccion(root)
    audit=dict(celdas_integradas=n,arquitecturas=len(mods),fuentes=[p.name for p in src.iterdir()],
               orden='fundamentos, doce arquitecturas y pipelines, evaluación integrada',
               paginas_retiradas=['10_pipeline','10_estadistica_e2'],entrenamiento=False,publicacion=False)
    (out/'integracion.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(audit,ensure_ascii=False),flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--actualizar-eda',action='store_true')
    args=parser.parse_args()
    reorganizar(Path(__file__).resolve().parents[1],actualizar_eda=args.actualizar_eda)

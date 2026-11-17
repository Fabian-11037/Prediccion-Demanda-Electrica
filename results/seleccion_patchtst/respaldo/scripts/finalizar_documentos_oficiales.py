"""Informe de cobertura, manifiestos y contexto de la ampliación documental."""
from pathlib import Path
import hashlib,importlib.metadata as metadata,json,re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1];AUD=ROOT/'results/documentos_oficiales';RAW=ROOT/'data/raw/documentos_oficiales'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    a=json.loads((AUD/'auditoria_integracion.json').read_text(encoding='utf-8'))
    o=pd.read_parquet(RAW/'documentos_oficiales.parquet');c=pd.read_parquet(ROOT/'data/processed/corpus_texto.parquet')
    m=pd.read_csv(AUD/'cobertura_mensual_fuente.csv');of=pd.read_csv(AUD/'oficiales_cobertura_mensual_fuente.csv')
    descarga=json.loads((AUD/'descargas_resumen.json').read_text(encoding='utf-8'));paquetes=json.loads((AUD/'paquetes_expandidos.json').read_text(encoding='utf-8'))
    assert a['ids_duplicados']==0 and not a['archivos_previos_modificados']
    # Una captura tardía prueba disponibilidad desde la captura, no novedad del contenido.
    histmask=(~o.cuarentena)&o.prueba_disponibilidad.eq('wayback_cdx_digest_matches_downloaded_pdf')
    retardo=o.loc[histmask,['id','fuente','fecha_declarada','precision_fecha','disponible_estricto']].copy()
    dia=pd.to_datetime(retardo.fecha_declarada.where(retardo.precision_fecha.eq('day')),errors='coerce').dt.tz_localize('America/Bogota').dt.tz_convert('UTC')
    retardo['dias_desde_inicio_fecha_declarada']=(retardo.disponible_estricto-dia).dt.total_seconds()/86400
    retardo['captura_hasta_72h_fecha_diaria']=retardo.dias_desde_inicio_fecha_declarada.between(0,3)
    retardo.to_csv(AUD/'antiguedad_edicion_al_verificar.csv',index=False)
    frescos=int(retardo.captura_hasta_72h_fecha_diaria.sum());con_fecha=int(dia.notna().sum());mediana=float(retardo.dias_desde_inicio_fecha_declarada.median())
    # Contraste de la cobertura por origen con los periodos vigentes; no se leen etiquetas de demanda.
    cuadros=[]
    for nombre,datos in [('Corpus integrado',m),('Documentos oficiales',of)]:
        for escenario,g in datos.loc[datos.fuente.eq('TODAS')].groupby('escenario'):
            for periodo,mask in [('Entrenamiento 2000–2017',g.mes.lt('2018-01')),('Validación 2018–2021',g.mes.ge('2018-01')&g.mes.lt('2022-01')),('Prueba 2022–2025',g.mes.ge('2022-01'))]:
                d=g.loc[mask];r={'corpus':nombre,'escenario':escenario,'periodo':periodo,'origenes':int(d.origenes.sum())}
                for h in [24,72,168]:r['cobertura_'+str(h)+'h']=float(d['origenes_con_texto_'+str(h)+'h'].sum()/d.origenes.sum())
                cuadros.append(r)
    cobertura=pd.DataFrame(cuadros);cobertura.to_csv(AUD/'cobertura_particiones.csv',index=False)
    # Identifica meses sin nuevas disponibilidades y gaps reales, sin imputar documentos.
    diario=pd.read_parquet(AUD/'oficiales_cobertura_origenes_diaria.parquet');continuidad=[]
    for (fuente,escenario),g in diario.groupby(['fuente','escenario']):
        g=g.sort_values('origen');v=g.con_texto_24h.to_numpy();i=np.flatnonzero(v)
        def gap(v):
            z=np.r_[False,~v,False].astype(int);cortes=np.diff(z);s=np.flatnonzero(cortes==1);e=np.flatnonzero(cortes==-1)
            return int(np.max(e-s)) if len(s) else 0
        continuidad.append({'fuente':fuente,'escenario':escenario,'primer_origen_con_texto':str(g.iloc[i[0]].origen) if len(i) else None,'ultimo_origen_con_texto':str(g.iloc[i[-1]].origen) if len(i) else None,'mayor_gap_dias_periodo_total':gap(v),'mayor_gap_dias_entre_primero_y_ultimo':gap(v[i[0]:i[-1]+1]) if len(i) else None})
    pd.DataFrame(continuidad).to_csv(AUD/'continuidad_fuente.csv',index=False)
    # Mapa mensual del texto oficial utilizable en las últimas 72 horas.
    meses=sorted(of.mes.unique());fuentes=sorted(o.fuente.unique())
    fig,axes=plt.subplots(2,1,figsize=(15,5),sharex=True,layout='constrained')
    for ax,escenario in zip(axes,['declarado','estricto']):
        matriz=of.loc[of.escenario.eq(escenario)&of.fuente.isin(fuentes)].pivot(index='fuente',columns='mes',values='cobertura_72h').reindex(index=fuentes,columns=meses)
        im=ax.imshow(matriz.to_numpy(),aspect='auto',vmin=0,vmax=1,cmap='viridis');ax.set_yticks(range(len(fuentes)),[{'MinEnergia':'MinEnergía'}.get(f,f) for f in fuentes]);ax.set_title('Escenario '+escenario+' · ventana de 72 horas',fontsize=10)
        ax.axvline(meses.index('2022-01')-.5,color='white',linestyle='--',linewidth=1)
    marcas=[i for i,x in enumerate(meses) if x.endswith('-01') and int(x[:4])%2==0]
    axes[-1].set_xticks(marcas,[meses[i][:4] for i in marcas]);fig.colorbar(im,ax=axes,label='Fracción de orígenes con documentos disponibles',shrink=.8)
    fig.savefig(AUD/'cobertura_oficiales_mensual.png',dpi=170);plt.close(fig)
    # Diccionario independiente: no modifica el diccionario numérico del E1.
    descripciones={
    'id':'Identificador de versión. Conserva el id original al enriquecer un titular; para nuevos documentos deriva de fuente y SHA-256 binario.',
    'fuente':'Entidad oficial o medio editor.', 'tipo_fuente':'Prensa u organismo oficial; metadato de procedencia.',
    'familia_documental':'Serie editorial o clase documental del repositorio, sin clasificación temática aprendida o manual.',
    'titulo':'Título o nombre editorial declarado por el catálogo actual.',
    'titulo_verificado':'En PDF corroborados por CDX, encabezado derivado del propio PDF; evita usar un título editorial actual como si estuviera archivado.',
    'texto':'Cuerpo extraído, texto de diapositivas, resumen oficial o fragmento OCR según texto_alcance.',
    'texto_alcance':'Alcance real de la extracción: titular, cuerpo, cuerpo parcial, OCR o texto de diapositivas.',
    'url':'Enlace de procedencia; los fragmentos identifican miembros de ZIP. No se concatena como texto de entrada.',
    'fecha_declarada':'Fecha de publicación o edición declarada; puede tener precisión anual, mensual o diaria.',
    'precision_fecha':'Precisión de la fecha; no añade una hora ficticia de publicación.',
    'base_fecha':'Evidencia o supuesto que respalda la fecha declarada.',
    'disponible_declarado':'Disponibilidad supuesta en UTC; día sin hora pasa al siguiente día a las 00:00 de Bogotá, año sin mes al año siguiente. Puede adelantarse hasta una captura que pruebe esos mismos bytes.',
    'disponible_estricto':'Captura histórica con identidad de contenido corroborada o recuperación actual si no existe esa evidencia.',
    'primera_verificacion':'Primera captura corroborada de esta versión o recuperación actual.',
    'recuperado_en':'Instante UTC de recuperación local del documento actual.',
    'prueba_disponibilidad':'Método de evidencia: captura archivada del titular, digest CDX idéntico al PDF o recuperación actual.',
    'sha256_original':'SHA-256 de los bytes originales utilizados.',
    'sha256_texto':'SHA-256 del cuerpo extraído en UTF-8.',
    'archivo_original':'Ruta del original descargado o PDF descomprimido, relativa al proyecto.',
    'archivo_texto':'Ruta del texto derivado, relativa al proyecto.',
    'cuarentena':'Excluye registros cuya fecha, edición o formato no supera las verificaciones.',
    'motivo_cuarentena':'Razón reproducible de exclusión; no es una etiqueta temática.',
    'alertas':'Advertencias sobre versión histórica, OCR, extracción o fechas.',
    'paginas':'Páginas PDF o diapositivas contabilizadas; cero para titulares y HTML.',
    'paginas_con_texto':'Páginas o diapositivas con texto extraído; permite detectar extracciones parciales.',
    'estado_extraccion':'Resultado de la extracción automática, incluido OCR pendiente o error.',
    'version':'Versión de integración o conservación del titular.',
    'grupo_titulo_fecha':'Huella de fuente, título normalizado y fecha para diagnosticar repetición, sin borrar automáticamente publicaciones de días distintos.'}
    assert set(c.columns)<=set(descripciones),set(c.columns)-set(descripciones)
    pd.DataFrame([{'variable':col,'tipo':str(c[col].dtype),'descripcion':descripciones[col]} for col in c.columns]).to_csv(ROOT/'data/processed/diccionario_corpus_texto.csv',index=False)
    insumos=['data/processed/panel_horario.parquet','data/processed/corpus_texto.parquet','data/raw/noticias/titulares.parquet','data/raw/documentos_oficiales/documentos_oficiales.parquet','data/processed/diccionario_corpus_texto.csv','scripts/adquisicion/descargar_documentos_oficiales.py','scripts/adquisicion/ocr_documentos_oficiales.py','scripts/integrar_documentos_oficiales.py','scripts/verificar_disponibilidad_oficial.py','scripts/finalizar_documentos_oficiales.py','src/demanda/corpus.py']
    manifiesto={'version':'corpus_oficial_20261109','periodo':['2000-01-01','2025-12-31'],'tabla_numerica':'data/processed/panel_horario.parquet','tabla_texto':'data/processed/corpus_texto.parquet','cargador':'demanda.corpus.cargar_corpus','alineacion':'demanda.corpus.documentos_en_origen','ventanas_horas_auditadas':[24,72,168],'regla_alineacion':'origen - ventana <= disponibilidad <= origen; zona America/Bogota','archivos':[{'archivo':s,'bytes':(ROOT/s).stat().st_size,'sha256':sha(ROOT/s)} for s in insumos]}
    (ROOT/'data/processed/dataset_texto_oficial_manifest.json').write_text(json.dumps(manifiesto,ensure_ascii=False,indent=2),encoding='utf-8')
    columnas=['fuente','documentos','con_cuerpo','con_texto_parcial','cuarentena','habilitados_declarado_2000_2025','habilitados_estricto_2000_2025']
    resumen=pd.DataFrame(a['por_fuente'])[columnas]
    tabla='| '+' | '.join(columnas)+' |\n|'+ '|'.join(['---']*len(columnas))+'|\n'+'\n'.join('| '+' | '.join(str(x) for x in r)+' |' for r in resumen.itertuples(index=False,name=None))
    hist=int(resumen.habilitados_estricto_2000_2025.sum());primera=o.loc[(~o.cuarentena)&o.disponible_estricto.lt(pd.Timestamp('2026-01-01',tz='America/Bogota')),'disponible_estricto'].min()
    particiones='\n'.join(f"- {r.periodo}, {r.escenario}: {r.cobertura_72h:.1%} de orígenes con texto oficial en 72 horas." for r in cobertura.loc[cobertura.corpus.eq('Documentos oficiales')].itertuples())
    texto=f'''# Ampliación documental oficial · 9 de noviembre de 2026

Se consultaron los catálogos públicos de IDEAM, XM, CREG, UPME y MinEnergía para 2000–2025. Se recuperaron {descarga['correctos']} de {descarga['descubiertos']} entradas de descarga; {len(descarga['fallos'])} enlaces siguen fallando después de reintentar rutas públicas equivalentes. Los {len(paquetes['paquetes'])} ZIP contienen {paquetes['documentos_pdf_miembros']} PDF con CRC comprobado; cada miembro se fecha por su propio encabezado, nunca por la fecha del ZIP. CREG conserva además los originales PDF enlazados, cuando existen.

El conjunto incorpora {a['oficiales_unicos']} documentos oficiales únicos y enriquece {a['titulares_enriquecidos']} titulares con contenido. El corpus integrado contiene {a['corpus_filas']} registros y cero identificadores duplicados. Las repeticiones de título y fecha se registran; los titulares originales permanecen intactos.

{tabla}

## Disponibilidad temporal

{hist} documentos oficiales son utilizables históricamente en el escenario estricto, con primera disponibilidad corroborada {primera}. La corroboración compara los bytes PDF completos con el digest SHA-1 Base32 de una captura CDX con respuesta 200; conserva SHA-256 local, URL, timestamp e índice CDX. No se afirma haber redescargado el payload de Internet Archive: la reproducción consultada respondió HTTP 429. No se promueven capturas de un título a evidencia de un cuerpo.

El título editorial actual tampoco queda verificado por el digest del PDF: en el escenario estricto se sustituye por un encabezado derivado del propio PDF. Los archivos sin prueba histórica se habilitan estrictamente desde su recuperación en 2026. El escenario declarado conserva fechas editoriales y supuestos explícitos; no acredita ausencia de fuga. La fecha de creación o modificación de XM no se presenta como una hora certificada de publicación, y un periodo referido en un nombre de archivo no se convierte automáticamente en fecha editorial.

La corroboración histórica tampoco acredita novedad: de {con_fecha} documentos corroborados con fecha diaria, {frescos} tienen una captura en las primeras 72 horas desde el inicio de esa fecha. La mediana de ese desfase es {mediana:.1f} días. Las capturas muy posteriores a la edición sirven como contexto histórico; no convierten un pronóstico antiguo en un anuncio reciente. El detalle está en antiguedad_edicion_al_verificar.csv. No se equipara esta cobertura documental con cobertura de información anticipatoria vigente.

{particiones}

Los CSV incluyen todos los meses de 2000–2025, también aquellos sin documentos. Las ventanas de 24, 72 y 168 horas describen disponibilidad y antigüedad, no garantizan vigencia meteorológica ni señal predictiva. No se imputan boletines ni se prolonga indefinidamente el último texto.

## Calidad y alcance

El OCR local en español se aplica a un máximo de cuatro páginas por PDF escaneado; texto_alcance y paginas_con_texto distinguen fragmentos y cuerpos completos. Los originales completos se conservan. Las diapositivas aportan sus cuadros de texto; los valores contenidos únicamente en mapas o imágenes no se convierten en variables numéricas. No hubo selección por palabras clave ni clasificación manual por temas. Las fechas o formatos incompatibles se conservan en cuarentena.

Los datos numéricos, titulares originales, cuadernos y HTML conservaron sus SHA-256. No se entrenaron modelos, no se modificaron pesos ni particiones y no se publicó contenido.

## Archivos de uso

- data/processed/corpus_texto.parquet: tabla consolidada de texto.
- data/raw/documentos_oficiales/documentos_oficiales.parquet: documentos oficiales y evidencia.
- data/processed/dataset_texto_oficial_manifest.json: tablas, reglas y hashes.
- data/processed/diccionario_corpus_texto.csv: significado de cada campo.
- results/documentos_oficiales/cobertura_mensual_fuente.csv: corpus completo por mes, fuente y escenario.
- results/documentos_oficiales/oficiales_cobertura_mensual_fuente.csv: cobertura exclusiva de la ampliación oficial.
- results/documentos_oficiales/cobertura_particiones.csv y continuidad_fuente.csv: disponibilidad por partición y gaps.
- results/documentos_oficiales/verificaciones_cdx.json: correspondencias de contenido con capturas históricas.

![Disponibilidad mensual oficial](cobertura_oficiales_mensual.png)

La disponibilidad declarada es más densa en los años recientes. La evidencia archivada permanece dispersa y su demora respecto de la edición limita el uso de boletines de vigencia corta.
'''
    (AUD/'INFORME_COBERTURA.md').write_text(texto,encoding='utf-8')
    instrucciones='''# Corpus de documentos oficiales

La tabla consolidada está en data/processed/corpus_texto.parquet. El inventario, originales HTTP, miembros PDF de ZIP, textos, OCR e índices CDX se conservan en esta carpeta. Consultar results/documentos_oficiales/INFORME_COBERTURA.md para cifras, cobertura y limitaciones.

Secuencia reproducible, desde la raíz del proyecto:

```bash
python scripts/adquisicion/descargar_documentos_oficiales.py descubrir
python scripts/adquisicion/descargar_documentos_oficiales.py ampliar_xm
python scripts/adquisicion/descargar_documentos_oficiales.py descargar
python scripts/adquisicion/descargar_documentos_oficiales.py reintentar
python scripts/adquisicion/descargar_documentos_oficiales.py originales_creg
python scripts/adquisicion/descargar_documentos_oficiales.py expandir_paquetes
python scripts/verificar_disponibilidad_oficial.py --descargar-indices
python scripts/integrar_documentos_oficiales.py --solo-extraer
python scripts/adquisicion/ocr_documentos_oficiales.py --preparar
python scripts/adquisicion/ocr_documentos_oficiales.py --max-paginas 4
python scripts/integrar_documentos_oficiales.py --solo-integrar
python scripts/finalizar_documentos_oficiales.py
```

El OCR requiere Windows, idioma es-MX y pdftoppm; se ejecuta en un entorno separado con requirements-ocr-windows.txt. El resto de la adquisición y los resultados OCR guardados se pueden consumir sin ese motor. Cambiar max-paginas permite ampliar OCR de anexos; las páginas parciales no se presentan como documentos íntegramente transcritos.

Uso por origen:

```python
from demanda.corpus import cargar_corpus, documentos_en_origen
corpus = cargar_corpus('estricto')
documentos = documentos_en_origen(corpus, '2023-08-15 00:00', ventana_horas=72)
```

El origen sin zona se interpreta en America/Bogota. Elegir declarado o estricto explícitamente y aplicar el mismo escenario en todas las particiones. No utilizar fecha_declarada para saltarse disponible_desde. URLs, hashes, pruebas y metadatos no se concatenan automáticamente al texto. No ejecutar entrenamiento ni publicar durante la reproducción de esta integración.
'''
    (RAW/'README.md').write_text(instrucciones,encoding='utf-8')
    (ROOT/'requirements-oficiales.txt').write_text('\n'.join(p+'=='+metadata.version(p) for p in ['requests','beautifulsoup4','pypdf','pandas','pyarrow','numpy','matplotlib'])+'\n',encoding='utf-8')
    paquetes_ocr=['winrt-runtime','winrt-Windows.Foundation','winrt-Windows.Foundation.Collections','winrt-Windows.Media.Ocr','winrt-Windows.Storage','winrt-Windows.Storage.Streams','winrt-Windows.Graphics.Imaging','winrt-Windows.Globalization']
    (ROOT/'requirements-ocr-windows.txt').write_text('\n'.join(p+'==3.2.1; sys_platform == "win32"' for p in paquetes_ocr)+'\n',encoding='utf-8')
    contexto=ROOT/'CONTEXTO_PROYECTO.md';s=contexto.read_text(encoding='utf-8');s=s.replace('Actualizado el 8 de noviembre de 2026.','Actualizado el 9 de noviembre de 2026.')
    s=s.replace('6. **Texto:** solo titulares.','6. **Texto:** el corpus original contiene titulares; la ampliación oficial incorpora cuerpos y OCR según §13.')
    fila_inventario=f"| Corpus integrado de prensa y documentos oficiales | {a['corpus_filas']} filas; {a['oficiales_unicos']} documentos oficiales únicos | data/processed/corpus_texto.parquet; disponibilidad y alcance en §13 |\n"
    if '| Corpus integrado de prensa y documentos oficiales |' not in s:
        s=s.replace('| Festivos de Colombia |',fila_inventario+'| Festivos de Colombia |',1)
    marca='\n## 13. Corpus oficial y disponibilidad temporal · 9 de noviembre de 2026'
    if marca in s:s=s.split(marca)[0]
    s+=marca+f'''

Adquisición local, sin entrenamiento ni publicación. Fuente operativa: results/documentos_oficiales/INFORME_COBERTURA.md y auditoria_integracion.json. Se descargaron {descarga['correctos']} entradas de {descarga['descubiertos']} del inventario; {len(descarga['fallos'])} fallos HTTP permanecen documentados. Los {len(paquetes['paquetes'])} paquetes contienen {paquetes['documentos_pdf_miembros']} PDF individuales. El conjunto oficial conserva {a['oficiales_unicos']} versiones únicas; {a['titulares_enriquecidos']} titulares se enriquecen y el corpus consolidado tiene {a['corpus_filas']} filas, con ids únicos.

{tabla}

Tablas: data/processed/corpus_texto.parquet y data/raw/documentos_oficiales/documentos_oficiales.parquet. Manifiesto: data/processed/dataset_texto_oficial_manifest.json. Diccionario: data/processed/diccionario_corpus_texto.csv. Los originales numéricos y titulares, cuadernos, HTML y modelos del E1 no se modifican. El corpus ampliado se integra mediante src/demanda/corpus.py: cargar_corpus requiere un escenario explícito y documentos_en_origen selecciona origen−ventana <= disponibilidad <= origen, con zona de Colombia y ventanas auditadas de 24/72/168 horas.

La disponibilidad estricta de un PDF solo se adelanta a una captura cuando su SHA-1 Base32 coincide exactamente con el digest de una respuesta 200 en CDX. Se conservan el índice completo, timestamp, URL y SHA-256 local. No se declara una recuperación del payload archivado cuando la reproducción no fue accesible. Son {hist} documentos oficiales históricos utilizables por esta regla; la primera captura utilizable es {primera}. El título editorial actual no queda verificado por ese digest: el cargador estricto utiliza el encabezado derivado del propio PDF. Una captura de un titular no verifica un cuerpo recuperado después. Sin prueba histórica, disponible_estricto es la recuperación en 2026 y el registro no entra antes en los orígenes de 2000–2025.

Antigüedad al verificar: {con_fecha} documentos corroborados tienen fecha diaria y {frescos} presentan una captura en las primeras 72 horas desde el inicio de esa fecha; desfase mediano {mediana:.1f} días. Detalle: antiguedad_edicion_al_verificar.csv. Un boletín antiguo capturado después no constituye un pronóstico reciente para ese origen. Antes de evaluar señal anticipatoria, comprobar su vigencia y fecha de edición; las ventanas de captura por sí solas no acreditan frescura. No se aplicó un filtro temático ni se entrenó una compuerta.

El escenario declarado mantiene la advertencia de versión histórica no verificada. Fecha diaria sin hora: día siguiente 00:00 de Bogotá; año de publicación sin mes: año siguiente. Un año de referencia del archivo no acredita publicación. XM conserva el supuesto de fecha de creación/modificación del repositorio como tal, sin llamarlo hora oficial de publicación. Los miembros de ZIP usan su encabezado de edición, no la fecha del contenedor. Fechas desconocidas o incompatibles quedan en cuarentena, salvo evidencia de disponibilidad de esa misma versión por captura.

Cobertura de la ampliación oficial en ventana de 72 horas:
{particiones}

La cobertura mensual incluye meses con cero documentos y se guarda por fuente y escenario. cobertura_particiones.csv conserva los cortes vigentes y continuidad_fuente.csv identifica días sin texto. Estos gaps no se rellenan ni se interpretan automáticamente como publicaciones perdidas: la periodicidad de las series es distinta. Los orígenes y métricas numéricas permanecen congelados.

OCR local Windows.Media.Ocr, idioma es-MX, render máximo 1800 píxeles y hasta cuatro páginas por escaneo. El campo texto_alcance distingue cuerpo, fragmento, texto de diapositivas y titular; las páginas restantes de documentos largos permanecen en el PDF original. No se asignan temas ni se filtra por relevancia manual. Toda selección aprendida, vocabulario o ajuste posterior debe usar únicamente entrenamiento. No se inició entrenamiento, selección de arquitecturas, evaluación predictiva nueva, reconstrucción del libro ni despliegue.

Reproducción: comandos en data/raw/documentos_oficiales/README.md; scripts de adquisición, expansión segura de ZIP, extracción, verificación CDX e integración incluidos. requirements-oficiales.txt y requirements-ocr-windows.txt separan el procesamiento general del OCR. Los JSON OCR permiten reutilizar el texto sin el motor Windows. No confundir el corpus nuevo con el archivo original de titulares al continuar E2/E3; utilizar el manifiesto y el cargador por origen.
'''
    contexto.write_text(s,encoding='utf-8');continuidad=ROOT/'continuidad/CONTEXTO_PROYECTO.md';continuidad.parent.mkdir(exist_ok=True);continuidad.write_text(s,encoding='utf-8')
    print(json.dumps({'corpus':a['corpus_filas'],'oficiales':a['oficiales_unicos'],'estrictos_historicos':hist,'primer_estricto':str(primera),'contexto_actualizado':True,'entrenamiento':False,'publicacion':False},ensure_ascii=False),flush=True)

if __name__=='__main__':main()

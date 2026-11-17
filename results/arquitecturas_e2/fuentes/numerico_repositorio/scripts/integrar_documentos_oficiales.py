"""Extrae documentos, integra el corpus y audita disponibilidad sin entrenar."""
from __future__ import annotations
from pathlib import Path
from datetime import datetime,timezone
import concurrent.futures,hashlib,json,logging,re,sys,unicodedata,warnings,zipfile
from xml.etree import ElementTree as ET
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'data/raw/documentos_oficiales';AUD=ROOT/'results/documentos_oficiales';TZ='America/Bogota'
MESES={x:i+1 for i,x in enumerate('enero febrero marzo abril mayo junio julio agosto septiembre octubre noviembre diciembre'.split())}

def limpiar(s):return re.sub(r'[ \t]+',' ',str(s)).strip()
def canonico(s):return re.sub(r'\n+','\n',s.replace('\r\n','\n').replace('\r','\n')).strip()
def normalizar(s):return ''.join(c for c in unicodedata.normalize('NFKD',str(s)) if not unicodedata.combining(c))
def fecha_es(s):
    s=normalizar(s).lower()
    m=re.search(r'\b([0-3]?\d)\s*(?:de\s+)?('+'|'.join(MESES)+r')\s*,?\s*(?:de\s+)?((?:19|20)\d{2})\b',s)
    if m:
        try:return pd.Timestamp(year=int(m[3]),month=MESES[m[2]],day=int(m[1])).date().isoformat()
        except ValueError:return None
    return None

def disponibilidad_declarada(fecha,precision):
    if not fecha:return pd.NaT
    if precision=='day':d=pd.Timestamp(fecha)+pd.Timedelta(days=1)
    elif precision=='month':d=pd.Timestamp(str(fecha)[:7]+'-01')+pd.offsets.MonthBegin(1)
    elif precision=='year':d=pd.Timestamp(str(int(str(fecha)[:4])+1)+'-01-01')
    else:return pd.NaT
    return d.tz_localize(TZ).tz_convert('UTC')

def fecha_miembro(body,nombre):
    """Fecha de edición del encabezado, sin usar fecha ZIP ni fechas del cuerpo."""
    lineas=[limpiar(x) for x in re.sub(r'\[Página[^\]]*\]','',body).splitlines()[:30]]
    for linea in lineas:
        f=fecha_es(linea)
        if f and (len(linea)<100 or re.search(r'bogot[aá]|fecha\s+de\s+(?:emisi[oó]n|publicaci[oó]n)',linea,re.I)):
            return f,'day','document_header_edition_date',linea
    f=fecha_es(nombre.replace('_',' ').replace('-',' '))
    if f:return f,'day','archive_member_filename_edition_date',nombre
    return None,'unknown','archive_member_date_not_identified',''

def extraer(r):
    """Texto completo con separación de páginas; las notas jurídicas actuales se excluyen."""
    logging.getLogger('pypdf').setLevel(logging.ERROR)
    meta=r['descarga'];aux=r.get('original_creg')
    usado=aux['descarga'] if aux and aux.get('estado')=='ok' else meta
    path=ROOT/usado['archivo'];ident=hashlib.sha256((r['fuente']+'\0'+usado['sha256']).encode()).hexdigest()
    cache=RAW/'extracciones'/(ident+'.json');txt=RAW/'textos'/(ident+'.txt')
    cache.parent.mkdir(parents=True,exist_ok=True);txt.parent.mkdir(parents=True,exist_ok=True)
    if cache.exists():
        old=json.loads(cache.read_text(encoding='utf-8'))
        if old.get('version_extractor')==2 and txt.exists() and old['url']==r['url'] and (not r['titulo'].lower().endswith('.pptx') or old.get('texto_alcance')=='all_slide_text') and (r['fuente']!='XM' or r['familia_documental']!='comunicado' or old.get('revision_html')==1):return old
    out={**r,'id':ident,'original_utilizado':usado,'version_extractor':2,'revision_html':1,'texto_alcance':'title_only','paginas':0,'paginas_con_texto':0,'estado_extraccion':'sin_cuerpo','alertas':[]}
    fecha=r['fecha_declarada'];precision=r['precision_fecha'];base=r['base_fecha']
    cuerpo='';fragmentos=[]
    try:
        payload=path.read_bytes();es_pdf=payload.lstrip().startswith(b'%PDF-')
        if es_pdf:
            from pypdf import PdfReader
            with warnings.catch_warnings():
                warnings.simplefilter('ignore');reader=PdfReader(path,strict=False)
                out['paginas']=len(reader.pages)
                for i,page in enumerate(reader.pages,1):
                    texto=page.extract_text() or '';texto=limpiar(texto)
                    if len(texto)>30:out['paginas_con_texto']+=1
                    fragmentos.append(texto)
            cuerpo='\n\n'.join(f'[Página {i+1}]\n{s}' for i,s in enumerate(fragmentos))
            chars=sum(map(len,fragmentos))
            if chars<100:
                cuerpo='';out['estado_extraccion']='requiere_ocr';out['texto_alcance']='title_only_requires_ocr'
            else:
                out['estado_extraccion']='ok';out['texto_alcance']='full_body' if out['paginas_con_texto']==out['paginas'] else 'partial_body'
                if out['paginas_con_texto']<out['paginas']:out['alertas'].append('paginas_sin_texto_extraible')
            pdfmeta=reader.metadata or {}
            out['pdf_creation_date']=str(pdfmeta.get('/CreationDate',''))
            cubierta=' '.join(fragmentos[:3])
            if base=='archive_member_edition_date_pending':
                fecha,precision,base,evi=fecha_miembro(fragmentos[0] if fragmentos else '',r['titulo']);out['evidencia_fecha_miembro']=evi
            # Fechas de versión explícitas: los años del horizonte proyectado no se interpretan como edición.
            patron=r'(?:revisi[oó]n|versi[oó]n|actualizaci[oó]n|bogot[aá](?:\s*d\.?\s*c\.?)?)\s*[:.,–-]?\s*(?:[0-3]?\d\s+de\s+)?(?:'+'|'.join(MESES)+r')\s*(?:de\s+)?(20\d{2})'
            versiones=re.findall(patron,cubierta.lower())
            if fecha and versiones and max(map(int,versiones))>int(str(fecha)[:4]):
                out['alertas'].append('edicion_posterior_a_fecha_del_indice');out['edicion_detectada']=max(map(int,versiones))
            # Una fecha de creación de PDF es metadato de versión, nunca fecha de publicación.
            mc=re.search(r'D:((?:19|20)\d{2})',out['pdf_creation_date'])
            if fecha and mc and int(mc[1])>int(str(fecha)[:4]):out['alertas'].append('pdf_creado_despues_del_anio_declarado')
        elif payload.startswith(b'PK\x03\x04'):
            with zipfile.ZipFile(path) as z:
                slides=sorted([n for n in z.namelist() if re.fullmatch(r'ppt/slides/slide\d+\.xml',n)],key=lambda n:int(re.search(r'(\d+)\.xml$',n)[1]))
                for n in slides:
                    tree=ET.fromstring(z.read(n));lineas=[]
                    for parrafo in tree.iter('{http://schemas.openxmlformats.org/drawingml/2006/main}p'):
                        texto=''.join(t.text or '' for t in parrafo.iter('{http://schemas.openxmlformats.org/drawingml/2006/main}t'))
                        if texto.strip():lineas.append(texto)
                    fragmentos.append('\n'.join(lineas))
                if not slides and 'word/document.xml' in z.namelist():
                    tree=ET.fromstring(z.read('word/document.xml'))
                    fragmentos=['\n'.join(''.join(t.text or '' for t in p.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t')) for p in tree.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p'))]
                cuerpo='\n\n'.join(f'[Diapositiva {i+1}]\n{s}' for i,s in enumerate(fragmentos))
                out['paginas']=len(fragmentos);out['paginas_con_texto']=sum(len(s)>30 for s in fragmentos)
                out['estado_extraccion']='ok' if len(cuerpo)>100 else 'sin_cuerpo';out['texto_alcance']='all_slide_text'
        else:
            s=BeautifulSoup(payload,'html.parser')
            if r['fuente']=='MinEnergia':
                partes=s.select('[class*="block-paragraph_block"], [class*="block-heading_block"], [class*="block-quote_block"]')
                cuerpo='\n\n'.join(x.get_text(' ',strip=True) for x in partes)
                out['texto_alcance']='full_body' if cuerpo else 'official_summary'
                if not cuerpo:cuerpo=r.get('resumen_oficial','')
            elif r['fuente']=='XM':
                partes=s.select('article.node--article .article__content .body.text-long') or s.select('article.node--article .field-body')
                cuerpo='\n\n'.join(x.get_text(' ',strip=True) for x in partes)
                out['texto_alcance']='full_body' if cuerpo else 'title_only'
                if r['familia_documental']!='comunicado':out['alertas'].append('respuesta_no_pdf')
            elif r['fuente']=='CREG':
                # El HTML contiene anotaciones agregadas con posterioridad; solo se usa como evidencia de fecha.
                cuerpo='';out['alertas'].append('html_juridico_anotado_excluido_del_cuerpo')
            else:out['alertas'].append('respuesta_no_pdf')
            out['estado_extraccion']='ok' if len(cuerpo)>100 else 'sin_cuerpo'
        if r['fuente']=='CREG':
            h=BeautifulSoup((ROOT/meta['archivo']).read_bytes(),'html.parser')
            panel=h.select_one('.panel-documento');cab=(panel.get_text(' ',strip=True) if panel else '')[:2500]
            do=re.search(r'Diario Oficial.{0,230}',cab,re.I)
            fecha_do=fecha_es(do[0]) if do else None
            if fecha_do:fecha=fecha_do;precision='day';base='official_gazette_publication_date_in_document';out['evidencia_fecha_diario_oficial']=do[0]
            else:
                # Año de expedición no acredita el día de publicación.
                fecha=None;out['alertas'].append('fecha_publicacion_no_identificada')
        if base=='reference_year_not_publication':
            fecha=None;out['alertas'].append('periodo_referido_no_es_publicacion')
    except Exception as e:
        out['estado_extraccion']='error';out['error_extraccion']=str(e);cuerpo=''
    cuerpo=canonico(cuerpo)
    out.update({'fecha_utilizable':fecha,'precision_utilizable':precision,'base_fecha_utilizable':base,'archivo_texto':txt.relative_to(ROOT).as_posix(),'caracteres_cuerpo':len(cuerpo),'sha256_texto':hashlib.sha256(cuerpo.encode()).hexdigest()})
    txt.write_text(cuerpo,encoding='utf-8',newline='\n');cache.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    return out

def extraer_todos():
    indice=RAW/'descargas_expandidas.jsonl'
    if not indice.exists():indice=RAW/'descargas.jsonl'
    rows=[json.loads(x) for x in indice.read_text(encoding='utf-8').splitlines()]
    aux={}
    if (RAW/'creg_originales.jsonl').exists():aux={x['url_documento']:x for x in map(json.loads,(RAW/'creg_originales.jsonl').read_text(encoding='utf-8').splitlines())}
    validos=[]
    for x in rows:
        if x['estado_descarga']=='ok':
            if x['fuente']=='CREG':x['original_creg']=aux.get(x['url'])
            validos.append(x)
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as ex,(RAW/'documentos_extraidos.jsonl').open('w',encoding='utf-8') as f:
        for i,r in enumerate(ex.map(extraer,validos,chunksize=4),1):
            f.write(json.dumps(r,ensure_ascii=False)+'\n');f.flush()
            if i%100==0:print('Extraídos',i,'/',len(validos),flush=True)

def oficiales():
    datos=[]
    prueba_path=AUD/'verificaciones_cdx.json'
    pruebas={r['sha256_original']:r for r in json.loads(prueba_path.read_text(encoding='utf-8'))} if prueba_path.exists() else {}
    for line in (RAW/'documentos_extraidos.jsonl').read_text(encoding='utf-8').splitlines():
        r=json.loads(line);m=r['original_utilizado'];body=(ROOT/r['archivo_texto']).read_text(encoding='utf-8')
        ocr_path=RAW/'ocr'/(r['id']+'.json')
        if ocr_path.exists() and r['estado_extraccion']=='requiere_ocr':
            ocr=json.loads(ocr_path.read_text(encoding='utf-8'))
            assert ocr['sha256_original']==m['sha256']
            pp=[p for p in ocr['paginas'] if len(p['texto'].strip())>30]
            if ocr['estado']=='ok' and pp:
                body='\n\n'.join('[Página '+str(p['pagina'])+' · OCR]\n'+p['texto'] for p in pp)
                r['paginas_con_texto']=len(pp);r['estado_extraccion']='ok_ocr'
                r['texto_alcance']='full_body_ocr' if len(pp)==r['paginas'] else 'partial_body_ocr'
                r['alertas'].append('ocr_automatico_es_MX')
                if len(pp)<r['paginas']:r['alertas'].append('ocr_limitado_a_primeras_paginas')
                r['archivo_texto']=(RAW/'textos'/(r['id']+'_ocr.txt')).relative_to(ROOT).as_posix()
        body=canonico(body)
        r['sha256_texto']=hashlib.sha256(body.encode()).hexdigest()
        (ROOT/r['archivo_texto']).write_text(body,encoding='utf-8',newline='\n')
        fecha=r['fecha_utilizable']
        if not fecha and r['base_fecha']=='archive_member_edition_date_pending':
            fecha,precision,base,evi=fecha_miembro(body,r['titulo']);r['precision_utilizable']=precision;r['base_fecha_utilizable']=base
        disp=disponibilidad_declarada(fecha,r['precision_utilizable']);razones=[]
        if fecha and not '2000'<=str(fecha)[:4]<='2025':razones.append('fecha_fuera_periodo_2000_2025')
        if pd.isna(disp):razones.append('fecha_disponibilidad_no_identificada')
        if not r['titulo'].strip():razones.append('titulo_vacio')
        if 'edicion_posterior_a_fecha_del_indice' in r['alertas']:razones.append('edicion_posterior_a_fecha_del_indice')
        if r['estado_extraccion']=='error':razones.append('error_extraccion')
        if 'respuesta_no_pdf' in r['alertas']:razones.append('respuesta_no_pdf')
        # Una recuperación actual solo prueba esta versión desde la recuperación.
        recuperacion=max(pd.Timestamp(m['retrieved_at']),pd.Timestamp(r['descarga']['retrieved_at']))
        datos.append(dict(id=r['id'],fuente=r['fuente'],tipo_fuente='official_report',familia_documental=r['familia_documental'],titulo=r['titulo'],texto=body,texto_alcance=r['texto_alcance'],url=r['url'],fecha_declarada=fecha,precision_fecha=r['precision_utilizable'],base_fecha=r['base_fecha_utilizable'],disponible_declarado=disp,disponible_estricto=recuperacion,primera_verificacion=recuperacion,recuperado_en=recuperacion,prueba_disponibilidad='download_payload_sha256_and_retrieval',sha256_original=m['sha256'],sha256_texto=r['sha256_texto'],archivo_original=m['archivo'],archivo_texto=r['archivo_texto'],cuarentena=bool(razones),motivo_cuarentena=';'.join(razones),alertas=';'.join(r['alertas']),paginas=r['paginas'],paginas_con_texto=r['paginas_con_texto'],estado_extraccion=r['estado_extraccion'],version='oficial_20261109'))
        datos[-1]['titulo_verificado']=r['titulo']
        datos[-1]['alertas']=';'.join([*r['alertas'],'version_historica_no_verificada'])
        if m['sha256'] in pruebas and len(body.strip())>100:
            prueba=pruebas[m['sha256']]
            captura=pd.to_datetime(prueba['timestamp'],format='%Y%m%d%H%M%S',utc=True)
            assert captura<=recuperacion
            fila=datos[-1];fila['disponible_estricto']=captura;fila['primera_verificacion']=captura
            fila['prueba_disponibilidad']='wayback_cdx_digest_matches_downloaded_pdf'
            fila['alertas']=';'.join([*r['alertas'],'identidad_pdf_corroborada_por_digest_cdx_sin_replay'])
            # El índice actual no verifica su título editorial. El encabezado estricto se deriva del propio PDF.
            fila['titulo_verificado']=limpiar(re.sub(r'\[Página[^\]]*\]','',body)[:300])
            fila['disponible_declarado']=min(disp,captura) if pd.notna(disp) else captura
            # La captura permite usar una versión aunque no tenga fecha editorial identificada.
            razones=[x for x in razones if x!='fecha_disponibilidad_no_identificada']
            fila['cuarentena']=bool(razones);fila['motivo_cuarentena']=';'.join(razones)
    t=pd.DataFrame(datos)
    duplicados=t.loc[t.duplicated('id',keep=False)].copy();duplicados.to_csv(AUD/'alias_documentos.csv',index=False)
    # Para una misma versión binaria, la elección de alias nunca adelanta su disponibilidad.
    t=t.sort_values('disponible_declarado',na_position='first').drop_duplicates('id',keep='last').reset_index(drop=True)
    for c in ['disponible_declarado','disponible_estricto','primera_verificacion','recuperado_en']:t[c]=pd.to_datetime(t[c],utc=True)
    t.to_parquet(RAW/'documentos_oficiales.parquet',index=False,compression='zstd')
    return t

def integrar(o):
    t=pd.read_parquet(ROOT/'data/raw/noticias/titulares.parquet')
    verificado=(t.availability_status.eq('verified_archived_version') & t.availability_proof.eq('wayback_payload_digest_and_capture'))
    p=pd.DataFrame({'id':t.id,'fuente':t.source,'tipo_fuente':t.source_kind,'familia_documental':np.where(t.source_kind.eq('news'),'prensa','boletin_ideam_titular'),'titulo':t.titular,'texto':'','texto_alcance':'title_only','url':t.url,'fecha_declarada':t.publication_date_claimed,'precision_fecha':t.date_precision,'base_fecha':t.assumed_availability_basis,'disponible_declarado':pd.to_datetime(t.available_at_assumed,utc=True,errors='coerce',format='mixed'),'disponible_estricto':pd.to_datetime(t.available_at_verified.where(verificado),utc=True,errors='coerce',format='mixed'),'primera_verificacion':pd.to_datetime(t.capture_at_verified.where(verificado),utc=True,errors='coerce',format='mixed'),'recuperado_en':pd.NaT,'prueba_disponibilidad':t.availability_proof.where(verificado),'sha256_original':None,'sha256_texto':None,'archivo_original':None,'archivo_texto':None,'cuarentena':t.quarantined,'motivo_cuarentena':t.quarantine_reason,'alertas':np.where(verificado,'','version_historica_no_verificada'),'paginas':0,'paginas_con_texto':0,'estado_extraccion':'title_only','version':'titulares_conservados'})
    p['recuperado_en']=pd.Series(pd.NaT,index=p.index,dtype='datetime64[ns, UTC]')
    p['titulo_verificado']=p['titulo'].copy()
    # Enriquecer los boletines ya presentes cuando coinciden fuente, título de archivo y fecha.
    def key(titulo):return re.sub(r'[^a-z0-9]+','',normalizar(str(titulo).lower().removesuffix('.pdf')))
    idx={('IDEAM',key(row.titulo),str(row.fecha_declarada)):i for i,row in p.loc[p.fuente.eq('IDEAM')].iterrows()}
    nuevos=[];enriquecidos=[];reemplazos={}
    for _,row in o.iterrows():
        k=(row.fuente,key(row.titulo),str(row.fecha_declarada))
        if row.fuente=='IDEAM' and k in idx:
            i=idx[k];anterior=p.loc[i,'id']
            prueba_titular=p.loc[i,'disponible_estricto']
            if pd.notna(prueba_titular) and prueba_titular<row.disponible_estricto:
                nuevos.append(row.to_dict());continue
            res=row.copy();res['id']=anterior;res['version']='titular_enriquecido_oficial_20261109'
            # No mejorar retroactivamente una prueba archivada del titular usando un cuerpo recuperado hoy.
            reemplazos[i]=res.to_dict()
            enriquecidos.append({'id_titular':anterior,'id_documento':row.id,'archivo_original':row.archivo_original})
        else:nuevos.append(row.to_dict())
    # Una sola concatenación evita copiar las columnas completas de titulares por cada reemplazo.
    p['_orden']=np.arange(len(p))
    for i,r in reemplazos.items():r['_orden']=i
    for i,r in enumerate(nuevos,len(p)):r['_orden']=i
    c=pd.concat([p.loc[~p.index.isin(reemplazos)],pd.DataFrame(reemplazos.values()),pd.DataFrame(nuevos)],ignore_index=True).sort_values('_orden').drop(columns='_orden').reset_index(drop=True)
    for f in ['disponible_declarado','disponible_estricto','primera_verificacion','recuperado_en']:c[f]=pd.to_datetime(c[f],utc=True,errors='coerce')
    c['cuarentena']=c.cuarentena.fillna(True).astype(bool)
    c['texto']=c.texto.fillna('');c['titulo']=c.titulo.fillna('')
    c['grupo_titulo_fecha']=[hashlib.sha256((str(a)+'\0'+key(b)+'\0'+str(d)).encode()).hexdigest() for a,b,d in zip(c.fuente,c.titulo,c.fecha_declarada)]
    assert c.id.is_unique
    c.to_parquet(ROOT/'data/processed/corpus_texto.parquet',index=False,compression='zstd')
    pd.DataFrame(enriquecidos).to_csv(AUD/'titulares_enriquecidos.csv',index=False)
    return c,len(t),len(enriquecidos)

def cobertura(c,prefijo=''):
    meses=pd.period_range('2000-01','2025-12',freq='M');origenes=pd.date_range('2000-01-01','2025-12-31',freq='D',tz=TZ)
    ori=origenes.tz_convert('UTC').as_unit('ns').asi8;fechas=origenes.tz_localize(None).to_period('M').astype(str)
    diario=[];mensual=[]
    fuentes=sorted(c.fuente.unique());versiones=['declarado','estricto']
    for escenario in versiones:
        for fuente in fuentes+['TODAS']:
            d=c.loc[~c.cuarentena & c['disponible_'+escenario].notna()]
            if fuente!='TODAS':d=d.loc[d.fuente.eq(fuente)]
            vv=pd.DatetimeIndex(d['disponible_'+escenario]).sort_values().as_unit('ns');v=vv.asi8
            meses_documentos=vv.tz_convert(TZ).tz_localize(None).to_period('M').astype(str)
            conteos_mes=pd.Series(meses_documentos).value_counts()
            dias_mes=pd.DataFrame({'mes':meses_documentos,'dia':vv.tz_convert(TZ).date}).groupby('mes').dia.nunique()
            hi=np.searchsorted(v,ori,side='right');frame=pd.DataFrame({'origen':origenes,'fuente':fuente,'escenario':escenario,'mes':fechas})
            for h in [24,72,168]:
                lo=np.searchsorted(v,ori-pd.Timedelta(hours=h).value,side='left');frame['documentos_'+str(h)+'h']=hi-lo;frame['con_texto_'+str(h)+'h']=(hi-lo)>0
            last=np.where(hi>0,np.clip(hi-1,0,max(len(v)-1,0)),0)
            frame['antiguedad_ultimo_h']=np.where(hi>0,(ori-v[last])/3.6e12,np.nan) if len(v) else np.nan
            diario.append(frame)
            for mes,g in frame.groupby('mes',sort=True):
                r={'mes':mes,'fuente':fuente,'escenario':escenario,'origenes':len(g),'documentos_habilitados':int(conteos_mes.get(mes,0)),'dias_nueva_disponibilidad':int(dias_mes.get(mes,0))}
                for h in [24,72,168]:
                    r['origenes_con_texto_'+str(h)+'h']=int(g['con_texto_'+str(h)+'h'].sum());r['cobertura_'+str(h)+'h']=float(g['con_texto_'+str(h)+'h'].mean())
                mensual.append(r)
    pd.concat(diario,ignore_index=True).to_parquet(AUD/(prefijo+'cobertura_origenes_diaria.parquet'),index=False,compression='zstd')
    m=pd.DataFrame(mensual);m.to_csv(AUD/(prefijo+'cobertura_mensual_fuente.csv'),index=False)
    anual=m.assign(anio=m.mes.str[:4]).groupby(['anio','fuente','escenario'],as_index=False).sum(numeric_only=True)
    for h in [24,72,168]:anual['cobertura_'+str(h)+'h']=anual['origenes_con_texto_'+str(h)+'h']/anual.origenes
    anual.to_csv(AUD/(prefijo+'cobertura_anual_fuente.csv'),index=False)
    return m

def main():
    AUD.mkdir(parents=True,exist_ok=True)
    if '--solo-extraer' in sys.argv:
        extraer_todos();return
    if '--solo-integrar' not in sys.argv:extraer_todos()
    o=oficiales();c,n,enriquecidos=integrar(o);m=cobertura(c);cobertura(o,'oficiales_')
    detalle=o.groupby('fuente').agg(documentos=('id','size'),con_cuerpo=('caracteres_cuerpo','sum')) if 'caracteres_cuerpo' in o else o.groupby('fuente').agg(documentos=('id','size'))
    resumen=[]
    for fuente,g in o.groupby('fuente'):
        u=g.loc[~g.cuarentena]
        resumen.append({'fuente':fuente,'documentos':len(g),'con_cuerpo':int(g.texto.str.strip().ne('').sum()),'con_texto_parcial':int(g.texto_alcance.str.startswith('partial_body').sum()),'requieren_ocr':int(g.estado_extraccion.eq('requiere_ocr').sum()),'cuarentena':int(g.cuarentena.sum()),'habilitados_declarado_2000_2025':int((u.disponible_declarado<pd.Timestamp('2026-01-01',tz=TZ)).sum()),'habilitados_estricto_2000_2025':int((u.disponible_estricto<pd.Timestamp('2026-01-01',tz=TZ)).sum()),'primera_disponibilidad_declarada':str(u.disponible_declarado.min()),'ultima_disponibilidad_declarada':str(u.disponible_declarado.max())})
    pd.DataFrame(resumen).to_csv(AUD/'resumen_fuentes.csv',index=False)
    anterior=json.loads((AUD/'estado_previo.json').read_text(encoding='utf-8'))
    cambios=[f for f,sha in anterior.items() if hashlib.sha256((ROOT/f).read_bytes()).hexdigest()!=sha]
    assert not cambios,cambios
    r={'fecha':datetime.now(timezone.utc).isoformat(),'periodo':['2000-01-01','2025-12-31'],'titulares_originales':n,'oficiales_unicos':len(o),'titulares_enriquecidos':enriquecidos,'corpus_filas':len(c),'ids_duplicados':int(c.id.duplicated().sum()),'titulos_fecha_repetidos':int(c.grupo_titulo_fecha.duplicated().sum()),'por_fuente':resumen,'fuentes_cobertura':sorted(c.fuente.unique()),'filas_cobertura_mensual':len(m),'archivos_previos_modificados':cambios,'entrenamiento_ejecutado':False,'publicacion_ejecutada':False,'nota_estricto':'Los PDF con bytes idénticos al digest SHA-1 Base32 de una captura CDX de respuesta 200 se habilitan desde esa captura. Se conserva SHA-256 local e índice CDX completo; no se afirma haber redescargado el payload archivado. El resto se verifica solo desde la recuperación en 2026. Una captura del titular no verifica retroactivamente un cuerpo.'}
    (AUD/'auditoria_integracion.json').write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(r,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()

"""Adquisición reproducible de documentos oficiales, sin ajuste de modelos."""
from __future__ import annotations
from pathlib import Path
from urllib.parse import urljoin,urldefrag,urlencode,urlsplit,unquote
from datetime import datetime,timezone
import argparse,base64,concurrent.futures,hashlib,json,re,threading,time,unicodedata,zipfile
from urllib.parse import quote
import requests
from bs4 import BeautifulSoup

ROOT=Path(__file__).resolve().parents[2]
RAW=ROOT/'data/raw/documentos_oficiales'
AUD=ROOT/'results/documentos_oficiales'
TLS=threading.local();LOCK=threading.Lock()
MESES={x:i+1 for i,x in enumerate('enero febrero marzo abril mayo junio julio agosto septiembre octubre noviembre diciembre'.split())}

def normalizar(s):return ''.join(c for c in unicodedata.normalize('NFKD',str(s)) if not unicodedata.combining(c))
def fecha_es(s):
    s=normalizar(s).lower()
    m=re.search(r'\b([0-3]?\d)\s*(?:de\s+)?('+'|'.join(MESES)+r')\s*,?\s*(?:de\s+)?((?:19|20)\d{2})\b',s)
    if m:
        try:return datetime(int(m[3]),MESES[m[2]],int(m[1])).date().isoformat()
        except ValueError:return None
    m=re.search(r'\b((?:19|20)\d{2})-(\d{2})-(\d{2})\b',s)
    if m:
        try:return datetime(*map(int,m.groups())).date().isoformat()
        except ValueError:return None
    return None

def obtener(url,clase='indices',payload=None):
    """Conserva bytes y respuesta HTTP; Last-Modified nunca se interpreta como publicación."""
    key=hashlib.sha256((url+(json.dumps(payload,sort_keys=True) if payload else '')).encode()).hexdigest();folder=RAW/clase;folder.mkdir(parents=True,exist_ok=True)
    path=folder/(key+'.bin');meta=folder/(key+'.json')
    if path.exists() and meta.exists():return path,json.loads(meta.read_text(encoding='utf-8'))
    if not hasattr(TLS,'session'):
        TLS.session=requests.Session();TLS.session.headers['User-Agent']='DemandaSIN-Research/1.0 (academic public-document archive)'
    for intento in range(2):
        try:
            peticion=TLS.session.get if payload is None else TLS.session.post
            with peticion(url,**({'json':payload} if payload else {}),timeout=(15,70),stream=True) as r:
                r.raise_for_status();tmp=path.with_suffix('.part')
                with tmp.open('wb') as f:
                    for chunk in r.iter_content(1024*256):f.write(chunk)
                tmp.replace(path)
                info={'url':url,'final_url':r.url,'status':r.status_code,'retrieved_at':datetime.now(timezone.utc).isoformat(),'content_type':r.headers.get('Content-Type'),'last_modified_http':r.headers.get('Last-Modified'),'etag':r.headers.get('ETag'),'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'archivo':path.relative_to(ROOT).as_posix()}
                if payload:info.update({'method':'POST','request_json':payload})
                meta.write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8');return path,info
        except Exception:
            if intento:raise
            time.sleep(1)

def sopa(url):
    p,m=obtener(url);return BeautifulSoup(p.read_bytes(),'html.parser'),m
def paralelo(func,items,n=4):
    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as ex:
        for x in ex.map(func,items):yield x

def fusionar_xm(data):
    extra=RAW/'inventario_xm_archivos.json'
    if not extra.exists():return data
    unicos={x['url']:x for x in data}
    for x in json.loads(extra.read_text(encoding='utf-8')):
        e=json.loads(x['evidencia_fecha']);fechas=[f[:10] for f in [e.get('fechaCreacion'),e.get('fechaModificacion')] if f]
        if fechas:
            x['fecha_declarada']=max(fechas);x['base_fecha']='repository_creation_or_modification_not_certified_publication'
        if len(x['fecha_declarada'])==10 and x['fecha_declarada']>'2025-12-31':continue
        unicos.setdefault(x['url'],x)
    return sorted(unicos.values(),key=lambda x:(x['fuente'],x['fecha_declarada'],x['url']))

def descubrir():
    RAW.mkdir(parents=True,exist_ok=True);AUD.mkdir(parents=True,exist_ok=True)
    documentos={};fallos=[];paginas=[]
    def guardar(rec):
        u=urldefrag(rec['url'])[0];rec['url']=u
        if not u.startswith('https://'):return
        documentos.setdefault(u,rec)
    def error(fuente,u,e):fallos.append({'fuente':fuente,'url':u,'error':str(e)})
    # IDEAM: listado paginado oficial y fecha de publicación por archivo.
    base='https://ideam.gov.co'
    catalogo,_=sopa(base+'/sala-de-prensa/boletines')
    familias=['Boletín de Pronóstico Nacional del Tiempo para Tres Días (BPTD)','Boletín de predicción climática','Boletín de seguimiento al ciclo ENOS','Comunicados Especiales (CE)']
    repos=[]
    for a in catalogo.select('a[href]'):
        u=urljoin(base,a['href']);slug=normalizar(unquote(u)).lower()
        if '/ultimo/' not in u and '/boletines/' in u and any(normalizar(f).lower().replace(' ','-') in slug for f in familias):repos.append(u)
    def pagina_ideam(u):
        try:
            s,m=sopa(u);salida=[]
            for a in s.select('a[href*="file-download/download/public/"]'):
                parent=a
                for _ in range(10):
                    if 'Fecha de publicación' in parent.get_text(' ',strip=True):break
                    if parent.parent is None:break
                    parent=parent.parent
                texto=parent.get_text(' ',strip=True)
                f=fecha_es(texto.split('Fecha de publicación',1)[-1]);title=a.get_text(' ',strip=True)
                if f and '2000-01-01'<=f<='2025-12-31':
                    salida.append({'fuente':'IDEAM','familia_documental':unquote(urlsplit(u).path.rsplit('/',1)[-1]),'url':urljoin(base,a['href']),'titulo':title,'fecha_declarada':f,'precision_fecha':'day','base_fecha':'publisher_listing_publication_date','evidencia_fecha':texto[:900],'indice_url':u,'indice_archivo':m['archivo'],'mime_declarado':a.get('type')})
            pages=[int(x) for a in s.select('a[href]') for x in re.findall(r'[?&]page=(\d+)',a['href'])]
            return salida,max(pages,default=0)
        except Exception as e:error('IDEAM',u,e);return [],0
    for repo in dict.fromkeys(repos):
        u=repo+'?'+urlencode({'desde':'2000-01-01','hasta':'2025-12-31'})
        inicial,ultimo=pagina_ideam(u)
        for x in inicial:guardar(x)
        for rows,_ in paralelo(pagina_ideam,[u+'&page='+str(i) for i in range(1,ultimo+1)]):
            for x in rows:guardar(x)
        paginas.append({'fuente':'IDEAM','serie':repo,'paginas_esperadas':ultimo+1,'documentos_acumulados':sum(x['fuente']=='IDEAM' for x in documentos.values())})
        print('Descubierto IDEAM',repo.rsplit('/',1)[-1],ultimo+1,flush=True)
    # UPME publica el inventario histórico en arreglos JavaScript del sitio oficial.
    upme=['https://www.upme.gov.co/simec/energia-electrica/generacion-de-energia-electrica/seguimiento-a-proyectos-de-generacion-en-desarrollo/','https://www.upme.gov.co/simec/planeacion-energetica/proyeccion-de-demanda-v2/']
    for u in upme:
        try:
            s,m=sopa(u);cantidad=0
            for script in s.find_all('script'):
                for bloque in re.findall(r'\{[^{}]*URL\s*:[^{}]*\}',script.get_text()):
                    campos={k:v for k,v in re.findall(r'(URL|TITULO|ANO_PUBLICACION)\s*:\s*"([^"]*)"',bloque)}
                    if not all(k in campos for k in ['URL','TITULO','ANO_PUBLICACION']):continue
                    if not campos['URL'].lower().split('?')[0].endswith('.pdf'):continue
                    if not '2000'<=campos['ANO_PUBLICACION']<='2025':continue
                    guardar({'fuente':'UPME','familia_documental':'seguimiento_generacion' if 'generacion-en' in u else 'proyecciones_demanda','url':campos['URL'],'titulo':campos['TITULO'].strip(),'fecha_declarada':campos['ANO_PUBLICACION'],'precision_fecha':'year','base_fecha':'publisher_listing_publication_year','evidencia_fecha':bloque.strip(),'indice_url':u,'indice_archivo':m['archivo']});cantidad+=1
            paginas.append({'fuente':'UPME','serie':u,'documentos_listados':cantidad})
        except Exception as e:error('UPME',u,e)
    print('Descubierto UPME',sum(x['fuente']=='UPME' for x in documentos.values()),flush=True)
    # CREG: proyectos publicados por orden cronológico. No se usan textos consolidados de normas.
    creg='https://gestornormativo.creg.gov.co/gestor/entorno/'
    def pagina_creg(ano):
        u=creg+f'proyectos_resolucion_por_orden_cronologico_{ano}.html'
        try:
            s,m=sopa(u);rows=[]
            for a in s.select('a[href*="docs/resolucion_creg_"]'):
                link=urljoin(u,a['href'])
                if re.search(r'_'+str(ano)+r'\.htm$',link):rows.append({'fuente':'CREG','familia_documental':'proyecto_resolucion','url':link,'titulo':a.get_text(' ',strip=True).replace('menu_book','').replace('Consultar navigate_next','').strip(),'fecha_declarada':str(ano),'precision_fecha':'year','base_fecha':'publisher_year_index','evidencia_fecha':a.get_text(' ',strip=True),'indice_url':u,'indice_archivo':m['archivo']})
            return rows
        except Exception as e:error('CREG',u,e);return []
    for rows in paralelo(pagina_creg,range(2000,2026),4):
        for x in rows:guardar(x)
    paginas.append({'fuente':'CREG','serie':'proyectos_resolucion_por_orden_cronologico','anios_consultados':list(range(2000,2026))})
    print('Descubierto CREG',sum(x['fuente']=='CREG' for x in documentos.values()),flush=True)
    # XM: comunicados con fecha diaria declarada en el listado, paginación completa.
    xm='https://www.xm.com.co/sala-de-prensa/comunicados'
    def pagina_xm(i):
        u=xm+'?page='+str(i)
        try:
            s,m=sopa(u);rows=[]
            for row in s.select('.view-xm-noticias .views-row'):
                a=row.select_one('h3 a[href]');t=row.find('time')
                if not a or not t:continue
                f=t.get('datetime','')[:10]
                if '2000-01-01'<=f<='2025-12-31':rows.append({'fuente':'XM','familia_documental':'comunicado','url':urljoin(xm,a['href']),'titulo':a.get_text(' ',strip=True),'fecha_declarada':f,'precision_fecha':'day','base_fecha':'publisher_listing_publication_day_time_not_certified','evidencia_fecha':str(t),'indice_url':u,'indice_archivo':m['archivo']})
            return rows,max([int(v) for v in re.findall(r'[?&]page=(\d+)',str(s))],default=0)
        except Exception as e:error('XM',u,e);return [],0
    rows,last=pagina_xm(0)
    for x in rows:guardar(x)
    for rows,_ in paralelo(pagina_xm,range(1,last+1),4):
        for x in rows:guardar(x)
    paginas.append({'fuente':'XM','serie':xm,'paginas_esperadas':last+1})
    print('Descubierto XM',sum(x['fuente']=='XM' for x in documentos.values()),flush=True)
    # Ministerio: API pública usada por el propio histórico; después se recupera el cuerpo HTML.
    api='https://www.minenergia.gov.co/api/v2/pages/'
    u=api+'?'+urlencode({'type':'sala_prensa.NoticiaPage','fields':'*','limit':100,'offset':0})
    def pagina_min(offset):
        link=api+'?'+urlencode({'type':'sala_prensa.NoticiaPage','fields':'*','limit':100,'offset':offset})
        try:
            p,m=obtener(link);j=json.loads(p.read_bytes());rows=[]
            for item in j['items']:
                f=item.get('date_published','');url=item['meta']['html_url'].replace('http://','https://')
                if '2000-01-01'<=f<='2025-12-31':rows.append({'fuente':'MinEnergia','familia_documental':'comunicado','url':url,'titulo':item['title'],'fecha_declarada':f,'precision_fecha':'day','base_fecha':'publisher_api_date_published','evidencia_fecha':json.dumps({'date_published':f,'cms_first_published_at':item['meta'].get('first_published_at')},ensure_ascii=False),'resumen_oficial':item.get('introduction',''),'cms_first_published_at':item['meta'].get('first_published_at'),'indice_url':link,'indice_archivo':m['archivo']})
            return rows,j['meta']['total_count'],len(j['items'])
        except Exception as e:error('MinEnergia',link,e);return [],0,100
    rows,total,n=pagina_min(0)
    for x in rows:guardar(x)
    for rows,_,_ in paralelo(pagina_min,range(n,total,n or 100),4):
        for x in rows:guardar(x)
    paginas.append({'fuente':'MinEnergia','serie':api,'total_api':total,'tamano_pagina':n})
    data=fusionar_xm(sorted(documentos.values(),key=lambda x:(x['fuente'],x['fecha_declarada'],x['url'])))
    (RAW/'inventario_descubierto.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    (AUD/'descubrimiento.json').write_text(json.dumps({'fecha':datetime.now(timezone.utc).isoformat(),'periodo':['2000-01-01','2025-12-31'],'series':paginas,'fallos':fallos,'por_fuente':{s:sum(x['fuente']==s for x in data) for s in sorted({x['fuente'] for x in data})}},ensure_ascii=False,indent=2),encoding='utf-8')
    print('INVENTARIO',len(data),'fallos',len(fallos),flush=True)

def descargar():
    inv=json.loads((RAW/'inventario_descubierto.json').read_text(encoding='utf-8'));registros=[]
    def uno(rec):
        try:
            p,m=obtener(rec['url'],'originales',rec.get('request_json'));return {**rec,**{'descarga':m,'estado_descarga':'ok'}}
        except Exception as e:return {**rec,'estado_descarga':'error','error_descarga':str(e)}
    with (RAW/'descargas.jsonl').open('w',encoding='utf-8') as salida:
        for i,r in enumerate(paralelo(uno,inv,6),1):
            registros.append(r);salida.write(json.dumps(r,ensure_ascii=False)+'\n');salida.flush()
            if i%25==0:print('Descargados',i,'/',len(inv),'correctos',sum(x['estado_descarga']=='ok' for x in registros),flush=True)
    (AUD/'descargas_resumen.json').write_text(json.dumps({'descubiertos':len(inv),'correctos':sum(x['estado_descarga']=='ok' for x in registros),'fallos':[{k:x.get(k) for k in ['fuente','url','error_descarga']} for x in registros if x['estado_descarga']!='ok'],'bytes':sum(x.get('descarga',{}).get('bytes',0) for x in registros)},ensure_ascii=False,indent=2),encoding='utf-8')
    print('DESCARGA TERMINADA',len(registros),flush=True)

def ampliar_xm():
    """Recorre exclusivamente los directorios públicos enlazados por XM."""
    api='https://api-portalxm.xm.com.co/v2/administracion-archivos/ficheros/'
    b64=lambda s:base64.urlsafe_b64encode(s.encode()).decode().rstrip('=')
    paginas=[('informe_mensual_mercado','https://www.xm.com.co/nuestra-empresa/informes/informes-de-la-operacion-y-el-mercado/informes-mensuales-de-analisis-del-mercado'),('boletin_energetico','https://www.xm.com.co/nuestra-empresa/informes/informes-de-la-operacion-y-el-mercado/boletin-energetico'),('recomendacion_mantenimientos','https://www.xm.com.co/operacion/planeacion/planeacion-mediano-plazo/recomendacion-de-mantenimientos')]
    cola=[];vistas=set();extras=[];errores=[];consultas=[]
    for familia,u in paginas:
        s,m=sopa(u)
        for t in s.find_all('explorador-archivos-component'):
            cola.append((t.get('directorio'),familia,u,m))
    def consultar(item):
        ruta,familia,indice,meta=item
        try:
            body={'ruta':b64(ruta),'nombre':'','contenedor':'storageportalxm','ordenarPor':'nombre','orden':'desc','pagina':1,'resultadosPorPagina':1000}
            p,m=obtener(api+'consulta?directorio='+b64(ruta),'indices',body);j=json.loads(p.read_bytes())
            fs=j.get('ficheros',[])
            total=max([f.get('totalRegistros',0) for f in fs],default=0)
            assert total<=1000,(ruta,total)
            return item,fs,m,None
        except Exception as e:return item,[],None,str(e)
    while cola:
        pendientes=cola;cola=[]
        for (ruta,familia,indice,meta),fs,m,error in paralelo(consultar,pendientes,4):
            vistas.add(ruta)
            if error:errores.append({'ruta':ruta,'error':error});continue
            consultas.append({'ruta':ruta,'entradas':len(fs),'indice_archivo':m['archivo']})
            for f in fs:
                nombre=f['nombre'];path=ruta.rstrip('/')+'/'+nombre
                if f['idTipoContenido']==2:
                    years=re.findall(r'(?<!\d)((?:19|20)\d{2})(?!\d)',nombre)
                    if years and any(int(y)>2025 or int(y)<2000 for y in years):continue
                    if path not in vistas:cola.append((path,familia,indice,meta))
                    continue
                if not nombre.lower().endswith('.pdf'):continue
                creada=f.get('fechaCreacion');fecha=creada[:10] if creada else None
                if fecha and not '2000-01-01'<=fecha<='2025-12-31':continue
                if not fecha:
                    years=re.findall(r'(?<!\d)((?:19|20)\d{2})(?!\d)',path)
                    if not years or not 2000<=int(years[-1])<=2025:continue
                    fecha=years[-1]
                payload={'ruta':b64(path.lstrip('/')),'nombreBlobContainer':'storageportalxm'}
                extras.append({'fuente':'XM','familia_documental':familia,'url':api+'descarga-archivo?archivo='+b64(path),'request_json':payload,'titulo':nombre,'fecha_declarada':fecha,'precision_fecha':'day' if creada else 'year','base_fecha':'repository_creation_date_assumed_publication' if creada else 'reference_year_not_publication','evidencia_fecha':json.dumps({k:f.get(k) for k in ['nombre','fechaCreacion','fechaModificacion','metadata']},ensure_ascii=False),'indice_url':indice,'indice_archivo':m['archivo'],'ruta_repositorio':path})
        print('XM directorios',len(vistas),'PDF',len(extras),'pendientes',len(cola),flush=True)
    # Inventario separado: evita sustituir una descarga que esté en curso.
    (RAW/'inventario_xm_archivos.json').write_text(json.dumps(extras,ensure_ascii=False,indent=2),encoding='utf-8')
    (AUD/'xm_archivos_descubrimiento.json').write_text(json.dumps({'consultas':consultas,'fallos':errores,'pdf':len(extras)},ensure_ascii=False,indent=2),encoding='utf-8')
    inventario=RAW/'inventario_descubierto.json'
    if inventario.exists():inventario.write_text(json.dumps(fusionar_xm(json.loads(inventario.read_text(encoding='utf-8'))),ensure_ascii=False,indent=2),encoding='utf-8')

def originales_creg():
    """Conserva el PDF original enlazado; evita incorporar anotaciones jurídicas actuales."""
    filas=[]
    for line in (RAW/'descargas.jsonl').read_text(encoding='utf-8').splitlines():
        r=json.loads(line)
        if r['fuente']=='CREG' and r['estado_descarga']=='ok':filas.append(r)
    def uno(r):
        s=BeautifulSoup((ROOT/r['descarga']['archivo']).read_bytes(),'html.parser')
        urls=[urljoin(r['url'],a['href']) for a in s.select('a[href]') if '.pdf' in a['href'].lower() and '$file' in a['href'].lower()]
        if not urls:return {'url_documento':r['url'],'estado':'sin_pdf_original'}
        try:
            p,m=obtener(urls[0],'originales');return {'url_documento':r['url'],'estado':'ok','descarga':m}
        except Exception as e:return {'url_documento':r['url'],'estado':'error','error':str(e),'url_original':urls[0]}
    with (RAW/'creg_originales.jsonl').open('w',encoding='utf-8') as f:
        for i,r in enumerate(paralelo(uno,filas,4),1):
            f.write(json.dumps(r,ensure_ascii=False)+'\n');f.flush()
            if i%50==0:print('PDF originales CREG',i,'/',len(filas),flush=True)

def reintentar():
    """Reintenta fallos HTTP mediante rutas públicas equivalentes de la misma fuente."""
    rows=[json.loads(s) for s in (RAW/'descargas.jsonl').read_text(encoding='utf-8').splitlines()]
    for r in rows:
        if r['estado_descarga']=='ok':continue
        candidatos=[r['url']]
        if r['fuente']=='MinEnergia':candidatos.append(r['url'].replace('https://minenergia.gov.co/','https://www.minenergia.gov.co/'))
        if r['fuente']=='XM' and 'descarga-archivo' in r['url']:candidatos.append(r['url'].replace('descarga-archivo','mostrar-url'))
        errores=[]
        for u in dict.fromkeys(candidatos):
            try:
                p,m=obtener(u,'originales',r.get('request_json'));r['descarga']=m;r['estado_descarga']='ok';r['url_descarga_efectiva']=u;break
            except Exception as e:errores.append(str(e))
        r['errores_reintentos']=errores
        print('Reintento',r['fuente'],r['estado_descarga'],flush=True)
    (RAW/'descargas.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    (AUD/'descargas_resumen.json').write_text(json.dumps({'descubiertos':len(rows),'correctos':sum(x['estado_descarga']=='ok' for x in rows),'fallos':[{k:x.get(k) for k in ['fuente','url','error_descarga','errores_reintentos']} for x in rows if x['estado_descarga']!='ok'],'bytes':sum(x.get('descarga',{}).get('bytes',0) for x in rows)},ensure_ascii=False,indent=2),encoding='utf-8')

def expandir_paquetes():
    rows=[json.loads(s) for s in (RAW/'descargas.jsonl').read_text(encoding='utf-8').splitlines()]
    salida=[];paquetes=[];resumenes=0;dest=RAW/'descomprimidos';dest.mkdir(exist_ok=True)
    for r in rows:
        if r['estado_descarga']!='ok':
            if r['fuente']=='MinEnergia' and r.get('resumen_oficial','').strip():
                parent=ROOT/r['indice_archivo'];pagina=json.loads(parent.read_bytes());pm=json.loads(parent.with_suffix('.json').read_text(encoding='utf-8'))
                item=next((i for i in pagina['items'] if unquote(urlsplit(i['meta']['html_url']).path)==unquote(urlsplit(r['url']).path)),None)
                if item:
                    data=json.dumps(item,ensure_ascii=False,sort_keys=True).encode();huella=hashlib.sha256(data).hexdigest();carpeta=RAW/'extractos_api';carpeta.mkdir(exist_ok=True);f=carpeta/(huella+'.json');f.write_bytes(data)
                    meta={**pm,'url':r['url'],'content_type':'application/json','bytes':len(data),'sha256':huella,'archivo':f.relative_to(ROOT).as_posix(),'method':'api_item','parent_api_file':pm['archivo'],'parent_api_sha256':pm['sha256']}
                    salida.append({**r,'estado_descarga':'ok','descarga':meta,'modo_recuperacion':'resumen_api_html_no_disponible'});resumenes+=1
            continue
        m=r['descarga'];p=ROOT/m['archivo']
        with p.open('rb') as f:cabeza=f.read(8)
        if not cabeza.startswith(b'PK\x03\x04'):
            salida.append(r);continue
        registro={'url':r['url'],'sha256_paquete':m['sha256'],'miembros_pdf':0,'errores':[]}
        try:
            with zipfile.ZipFile(p) as z:
                if 'ppt/presentation.xml' in z.namelist() or 'word/document.xml' in z.namelist():
                    salida.append(r);continue
                for miembro in z.infolist():
                    if miembro.is_dir() or not miembro.filename.lower().endswith('.pdf'):continue
                    try:
                        assert miembro.file_size<=512*1024*1024,'PDF miembro mayor de 512 MB'
                        data=z.read(miembro)
                        assert b'%PDF-' in data[:1024],'El miembro no es PDF'
                        sha=hashlib.sha256(data).hexdigest();pdf=dest/(sha+'.pdf')
                        if not pdf.exists():pdf.write_bytes(data)
                        u=r['url']+'#miembro='+quote(miembro.filename,safe='')
                        meta={**m,'url':u,'content_type':'application/pdf','bytes':len(data),'sha256':sha,'archivo':pdf.relative_to(ROOT).as_posix(),'method':'zip_member','parent_archive_sha256':m['sha256'],'parent_archive_file':m['archivo'],'archive_member':miembro.filename}
                        (dest/(sha+'.json')).write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
                        salida.append({**r,'url':u,'titulo':Path(miembro.filename).name,'fecha_archive_declarada':r['fecha_declarada'],'fecha_declarada':None,'precision_fecha':'unknown','base_fecha':'archive_member_edition_date_pending','evidencia_fecha':'Fecha del paquete no aplicada al documento miembro','descarga':meta,'miembro_paquete':miembro.filename})
                        registro['miembros_pdf']+=1
                    except Exception as e:registro['errores'].append({'miembro':miembro.filename,'error':str(e)})
        except Exception as e:registro['errores'].append({'error':str(e)})
        paquetes.append(registro)
    (RAW/'descargas_expandidas.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in salida),encoding='utf-8')
    (AUD/'paquetes_expandidos.json').write_text(json.dumps({'paquetes':paquetes,'documentos_pdf_miembros':sum(p['miembros_pdf'] for p in paquetes),'documentos_totales':len(salida),'resumenes_api_recuperados':resumenes},ensure_ascii=False,indent=2),encoding='utf-8')
    print('Paquetes',len(paquetes),'documentos',len(salida),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('fase',choices=['descubrir','descargar','ampliar_xm','originales_creg','reintentar','expandir_paquetes']);a=p.parse_args()
    {'descubrir':descubrir,'descargar':descargar,'ampliar_xm':ampliar_xm,'originales_creg':originales_creg,'reintentar':reintentar,'expandir_paquetes':expandir_paquetes}[a.fase]()

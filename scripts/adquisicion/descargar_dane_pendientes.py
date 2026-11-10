"""Anexos DANE y población: conserva originales, versiones y fechas declaradas."""
from pathlib import Path
from urllib.parse import urljoin,urlparse,unquote
import hashlib
import io
import json
import re
import time
import unicodedata
import zipfile
import pandas as pd
import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[2]
ISE=ROOT/'data/external/actividad_economica'
POB=ROOT/'data/external/poblacion_dane'
HIST='https://www.dane.gov.co/index.php/estadisticas-por-tema/cuentas-nacionales/indicador-de-seguimiento-a-la-economia-ise/historicos-ise-comunicados-y-boletines'
POP='https://www.dane.gov.co/index.php/estadisticas-por-tema/demografia-y-poblacion/proyecciones-de-poblacion'
MESES={v:i+1 for i,v in enumerate('enero febrero marzo abril mayo junio julio agosto septiembre octubre noviembre diciembre'.split())}
MESES.update({k[:3]:v for k,v in list(MESES.items())})
CODIGOS={'05001':'medellin','08001':'barranquilla','11001':'bogota','13001':'cartagena',
         '54001':'cucuta','66001':'pereira','68001':'bucaramanga','76001':'cali'}


def norm(x):
    return ''.join(c for c in unicodedata.normalize('NFKD',str(x)) if not unicodedata.combining(c)).lower().strip()


def descargar(url,carpeta):
    carpeta.mkdir(parents=True,exist_ok=True)
    nombre=unquote(Path(urlparse(url).path).name)
    p=carpeta/nombre
    meta=p.with_suffix(p.suffix+'.json')
    if p.exists() and meta.exists():
        m=json.loads(meta.read_text(encoding='utf-8'))
        assert hashlib.sha256(p.read_bytes()).hexdigest()==m['sha256']
        return p,m
    if p.exists() and p.suffix.lower()=='.xlsx' and zipfile.is_zipfile(p):
        m=dict(url=url,recuperado_utc=pd.Timestamp(p.stat().st_mtime,unit='s',tz='UTC').isoformat(),
            bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
            archivo=str(p.relative_to(ROOT)).replace('\\','/'),
            procedencia='Descarga HTTP 200 previa de esta sesion; XLSX validado')
        meta.write_text(json.dumps(m,ensure_ascii=False,indent=2),encoding='utf-8')
        return p,m
    for intento in range(4):
        r=requests.get(url,timeout=70)
        if r.status_code==200:break
        if r.status_code==429:
            demora=max(30,int(r.headers.get('Retry-After','30')) if r.headers.get('Retry-After','').isdigit() else 30)
            print('DANE: pausa por limite de solicitudes',demora,flush=True)
            time.sleep(min(demora,60))
        elif r.status_code>=500:time.sleep(5*(intento+1))
        else:break
    r.raise_for_status()
    p.write_bytes(r.content)
    m=dict(url=url,recuperado_utc=pd.Timestamp.now(tz='UTC').isoformat(),bytes=len(r.content),
        sha256=hashlib.sha256(r.content).hexdigest(),archivo=str(p.relative_to(ROOT)).replace('\\','/'))
    meta.write_text(json.dumps(m,ensure_ascii=False,indent=2),encoding='utf-8')
    time.sleep(1.5)
    return p,m


def fecha_boletin(path):
    try:
        texto=norm(PdfReader(path).pages[0].extract_text())
        for day,mes,year in re.findall(r'(\d{1,2})\s*(?:de\s+)?('+ '|'.join(MESES)+r')\s*(?:de\s+)?(20\d{2})',texto):
            return pd.Timestamp(year=int(year),month=MESES[mes],day=int(day)).date().isoformat()
    except Exception:
        return None


def extraer_horizontal(path):
    salida={};bases=set()
    hojas=pd.read_excel(path,sheet_name=None,header=None)
    for nombre,d in hojas.items():
        vertical=next((i for i in range(min(20,len(d)-1))
            if norm(d.iat[i,0])=='ano' and norm(d.iat[i,1])=='mes'),None)
        if vertical is not None:
            grupo='';mapa={}
            for col in range(2,d.shape[1]):
                if pd.notna(d.iat[vertical,col]):grupo=norm(d.iat[vertical,col])
                sub=norm(d.iat[vertical+1,col])
                if 'indice' in grupo and 'variacion' not in grupo:
                    etiqueta=sub
                elif 'serie' in grupo and 'indice' in sub:
                    etiqueta=grupo
                else:continue
                tipo=('ise_original' if 'original' in etiqueta else
                    'ise_desestacionalizado' if ('desestacional' in etiqueta or 'corregida' in etiqueta or 'ajustada' in etiqueta) else
                    'ise_tendencia_ciclo' if 'tendencia' in etiqueta else None)
                if tipo:mapa[tipo]=col
            year=None;datos={k:{} for k in mapa}
            for row in d.iloc[vertical+2:].itertuples(index=False,name=None):
                z=re.match(r'^(20\d{2})(?:\D.*)?$',str(row[0]))
                if z:year=int(z[1])
                mes=norm(row[1]).rstrip('.')
                if year is None or mes not in MESES:continue
                for tipo,col in mapa.items():
                    valor=pd.to_numeric(row[col],errors='coerce')
                    if pd.notna(valor):datos[tipo][f'{year:04d}-{MESES[mes]:02d}']=float(valor)
            for tipo,serie in datos.items():
                if len(serie)>=12:salida[tipo]=pd.Series(serie)
            cab=' '.join(norm(v) for v in d.iloc[:14].to_numpy().ravel() if pd.notna(v))
            b=re.search(r'(?:referencia|base)\s*(20\d{2})',cab)
            if b:bases.add(b[1])
            continue
        cab=' '.join(norm(v) for v in d.iloc[:14].to_numpy().ravel() if pd.notna(v))
        tipo=('ise_original' if 'datos originales' in cab else
              'ise_desestacionalizado' if (('ajustad' in cab and 'estacional' in cab) or 'desestacional' in cab) else
              'ise_tendencia_ciclo' if ('tendencia' in cab and 'ciclo' in cab) else None)
        if tipo is None:continue
        encabezado=None
        for i in range(min(24,len(d)-1)):
            meses=[norm(v).rstrip('.') for v in d.iloc[i+1]]
            if sum(v in MESES for v in meses)>=12:
                encabezado=i;break
        if encabezado is None:continue
        objetivo=[]
        for i in range(encabezado+2,min(len(d),encabezado+40)):
            s=norm(d.iat[i,0])
            if 'indicador de seguimiento' in s and '(ise)' not in s and pd.to_numeric(d.iloc[i,1:],errors='coerce').notna().sum()>=12:
                objetivo.append(i)
        if not objetivo:continue
        fila=objetivo[0];year=None;serie={}
        for col in range(1,d.shape[1]):
            z=re.match(r'^(20\d{2})(?:\D.*)?$',str(d.iat[encabezado,col]))
            if z:year=int(z[1])
            mes=norm(d.iat[encabezado+1,col]).rstrip('.')
            valor=pd.to_numeric(d.iat[fila,col],errors='coerce')
            if year is not None and mes in MESES and pd.notna(valor):
                serie[f'{year:04d}-{MESES[mes]:02d}']=float(valor)
        if serie:salida[tipo]=pd.Series(serie)
        b=re.search(r'(?:referencia|base)\s*(20\d{2})',cab)
        if b:bases.add(b[1])
    if 'ise_original' not in salida:raise ValueError(f'No se identificó la serie original de índices: {list(salida)}')
    d=pd.DataFrame(salida).rename_axis('mes').reset_index()
    for col in ['ise_original','ise_desestacionalizado','ise_tendencia_ciclo']:
        if col not in d:d[col]=float('nan')
    assert not d.mes.duplicated().any()
    return d,','.join(sorted(bases)) or 'no_identificada'


def adquirir_ise():
    ISE.mkdir(exist_ok=True,parents=True)
    p,m=descargar(HIST,ISE/'originales')
    soup=BeautifulSoup(p.read_text(encoding='utf-8'),'html.parser')
    registros=[];versiones=[];errores=[];vistos=set()
    for a in soup.find_all('a',href=True):
        href=urljoin(HIST,a['href'])
        if not re.search(r'\.xlsx?(?:\?|$)',href,re.I) or href in vistos:continue
        vistos.add(href)
        try:
            file,meta=descargar(href,ISE/'originales')
            row=a.find_parent('tr')
            boletin=row.find('a',href=re.compile(r'\.pdf',re.I)) if row else None
            fecha=None;pdfmeta=None
            if boletin:
                pdf,pdfmeta=descargar(urljoin(HIST,boletin['href']),ISE/'boletines')
                fecha=fecha_boletin(pdf)
            info=dict(**meta,publicacion_declarada=fecha,boletin=pdfmeta,
                      version_historica_no_verificada=True)
            registros.append(info)
            try:
                d,base=extraer_horizontal(file)
                d['vintage_publicacion']=fecha
                d['vintage_recuperacion']='2026-11-10'
                d['archivo_fuente']=file.name
                d['base_indice']=base
                d['version_historica_no_verificada']=True
                versiones.append(d)
                info['meses_extraidos']=len(d)
            except Exception as exc:
                info['extraccion_pendiente']=str(exc)[:220]
        except Exception as exc:
            errores.append(dict(url=href,error=str(exc)[:220]))
        if len(registros)%20==0:print('ISE anexos conservados',len(registros),flush=True)
    pd.DataFrame([{k:v for k,v in r.items() if k!='boletin'} for r in registros]).to_csv(ISE/'inventario_versiones.csv',index=False)
    if versiones:
        total=pd.concat(versiones,ignore_index=True)
        assert not total.duplicated(['archivo_fuente','mes']).any()
        total.to_parquet(ISE/'ise_vintages.parquet',index=False,compression='zstd')
        ultimo=total.loc[total.archivo_fuente.eq('anex-ISE-9actividades-ago2026.xlsx')].copy()
        assert len(ultimo)==260 and ultimo.mes.iloc[0]=='2005-01' and ultimo.mes.iloc[-1]=='2026-08'
        assert round(float(ultimo.ise_original.iloc[-1]),2)==127.27
        ultimo['vintage']=ultimo.vintage_publicacion
        ultimo.to_csv(ISE/'ise_dane.csv',index=False)
    report=dict(anexos_identificados=len(vistos),anexos_descargados=len(registros),
        versiones_interpretadas=len(versiones),versiones_sin_parser=sum('extraccion_pendiente' in r for r in registros),
        fechas_publicacion_identificadas=sum(r['publicacion_declarada'] is not None for r in registros),
        errores=errores,originales=registros,
        nota='Las versiones se recuperaron hoy. Una fecha de boletín es declarada; no verifica por sí sola la versión histórica del Excel. No mezclar bases del índice ni usar el vintage final como si fuera original.')
    (ISE/'manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('ISE', {k:v for k,v in report.items() if k not in ('originales',)},flush=True)


def adquirir_poblacion():
    base='https://www.dane.gov.co/files/censo2018/proyecciones-de-poblacion/Municipal/'
    archivos=[('DCD-area-proypoblacion-Mun-1995-2004.xlsx','Municipal','2023-03-22'),
              ('DCD-area-proypoblacion-Mun-2005-2017_VP.xlsx','NuevaMpal','2025-08-08'),
              ('PPED-AreaMun-2018-2042_VP.xlsx','PobMunicipalxÁrea','2025-08-08')]
    filas=[];origen=[]
    for nombre,hoja,fecha in archivos:
        p,m=descargar(base+nombre,POB/'originales');origen.append(m)
        d=pd.read_excel(p,sheet_name=hoja,header=None)
        hdr=next(i for i,r in d.head(25).iterrows() if any(norm(v) in ('ano','año') for v in r))
        nombres=[norm(v) for v in d.iloc[hdr]]
        candidatos=[i for i,v in enumerate(nombres) if v in ('mpio','dpmp','codigo municipio','cod_mpio','codmpio')]
        # Los archivos históricos intercambian MPIO/DPMP entre código y nombre.
        codigo=max(candidatos,key=lambda i:d.iloc[hdr+1:,i].map(lambda v:str(v).split('.')[0].zfill(5)).isin(CODIGOS).sum())
        m['columna_codigo_original']=nombres[codigo]
        anio=next(i for i,v in enumerate(nombres) if v in ('ano','año'))
        area=next(i for i,v in enumerate(nombres) if 'area' in v)
        valor=next(i for i,v in enumerate(nombres) if v in ('total','poblacion'))
        for r in d.iloc[hdr+1:].itertuples(index=False,name=None):
            cod=str(r[codigo]).split('.')[0].zfill(5)
            if cod not in CODIGOS:continue
            y=pd.to_numeric(r[anio],errors='coerce');n=pd.to_numeric(r[valor],errors='coerce')
            if pd.isna(y) or pd.isna(n) or not 2000<=int(y)<=2026:continue
            ar=norm(r[area]);ar='total' if ar=='total' else 'cabecera' if 'cabecera' in ar else 'rural'
            filas.append(dict(ciudad=CODIGOS[cod],codigo_divipola=cod,anio=int(y),area=ar,poblacion=int(n),
                publicacion_declarada=fecha,archivo_fuente=nombre))
    t=pd.DataFrame(filas)
    assert not t.duplicated(['ciudad','anio','area']).any()
    w=t.pivot(index=['ciudad','codigo_divipola','anio','publicacion_declarada','archivo_fuente'],columns='area',values='poblacion').reset_index()
    assert len(w)==216 and not w[['total','cabecera','rural']].isna().any().any()
    assert (w.total==w.cabecera+w.rural).all()
    w['peso_total_8_ciudades']=w.total/w.groupby('anio').total.transform('sum')
    w.to_csv(POB/'poblacion_ciudades_2000_2026.csv',index=False)
    (POB/'manifest.json').write_text(json.dumps(dict(filas=len(w),ciudades=8,inicio=2000,fin=2026,
        versiones_actuales_revisadas=True,originales=origen,pagina=POP,
        nota='No sustituye automáticamente pesos congelados. Las retroproyecciones usan censos/revisiones posteriores; respetar su vintage en experimentos históricos.'),ensure_ascii=False,indent=2),encoding='utf-8')
    print('Población',len(w),'filas verificadas',flush=True)


if __name__=='__main__':
    adquirir_ise()
    adquirir_poblacion()

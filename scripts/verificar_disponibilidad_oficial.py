"""Contrasta los bytes PDF descargados con huellas históricas del índice CDX."""
from pathlib import Path
import base64,hashlib,json,re,sys
from urllib.parse import urlencode

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'data/raw/documentos_oficiales';AUD=ROOT/'results/documentos_oficiales'

def digest_pdf(path):
    h=hashlib.sha1()
    with path.open('rb') as f:
        if not f.read(16).lstrip().startswith(b'%PDF-'):return None
        f.seek(0)
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return base64.b32encode(h.digest()).decode()

def indice_cdx():
    indice={};archivos=[]
    for p in sorted((RAW/'cdx').glob('*.json')):
        try:j=json.loads(p.read_bytes())
        except (ValueError,UnicodeError):continue
        if not isinstance(j,list) or not j or not isinstance(j[0],list):continue
        head=j[0]
        if not {'timestamp','original','digest','statuscode','mimetype'}<=set(head):continue
        assert len(j)-1<50000,'La consulta CDX requiere paginación'
        sha=hashlib.sha256(p.read_bytes()).hexdigest();archivos.append({'archivo':p.relative_to(ROOT).as_posix(),'sha256':sha,'registros':len(j)-1})
        for row in j[1:]:
            d=dict(zip(head,row));ts=d.get('timestamp','')
            if d.get('statuscode')!='200' or d.get('mimetype')!='application/pdf' or not re.fullmatch(r'\d{14}',ts) or ts>'20251231235959':continue
            if not re.fullmatch(r'[A-Z2-7]{32}',d.get('digest','')):continue
            old=indice.get(d['digest'])
            if old is None or ts<old['timestamp']:indice[d['digest']]={**d,'indice_archivo':p.relative_to(ROOT).as_posix(),'indice_sha256':sha}
    (AUD/'manifiesto_indices_cdx.json').write_text(json.dumps(archivos,indent=2),encoding='utf-8')
    return indice

def descargar_indices():
    import requests
    carpeta=RAW/'cdx';carpeta.mkdir(exist_ok=True)
    fuentes={'ideam_nuevo':'ideam.gov.co/file-download/download/public/*','ideam_legado':'ideam.gov.co/documents/*','upme':'docs.upme.gov.co/*','upme_legado':'www1.upme.gov.co/*','xm':'xm.com.co/*','creg':'gestornormativo.creg.gov.co/Publicac.nsf/*','creg_legado':'apolo.creg.gov.co/Publicac.nsf/*'}
    registros=[]
    for nombre,patron in fuentes.items():
        params={'url':patron,'output':'json','filter':['statuscode:200','mimetype:application/pdf'],'collapse':'digest','to':'20251231','fl':'timestamp,original,mimetype,statuscode,digest','limit':50000}
        url='https://web.archive.org/cdx/search/cdx?'+urlencode(params,doseq=True)
        try:
            r=requests.get(url,timeout=(15,90));r.raise_for_status();j=r.json()
            assert len(j)-1<50000,'Consulta truncada: requiere paginación'
            archivo=carpeta/(nombre+'.json');archivo.write_bytes(r.content)
            registros.append({'fuente':nombre,'url':url,'status':r.status_code,'registros':len(j)-1,'sha256':hashlib.sha256(r.content).hexdigest()})
        except Exception as e:registros.append({'fuente':nombre,'url':url,'error':str(e)})
    (AUD/'consulta_cdx.json').write_text(json.dumps(registros,indent=2),encoding='utf-8')

def main():
    if '--descargar-indices' in sys.argv:descargar_indices()
    indice=indice_cdx();coincidencias=[];cachepath=AUD/'huellas_sha1_pdf.json'
    cache=json.loads(cachepath.read_text(encoding='utf-8')) if cachepath.exists() else {}
    parcial=AUD/'coincidencias_cdx_parcial.json'
    if parcial.exists():
        for x in json.loads(parcial.read_text(encoding='utf-8')):cache[x['sha256']]=x['digest_sha1_base32']
    examinados=0
    for f in sorted([*(RAW/'originales').glob('*.json'),*(RAW/'descomprimidos').glob('*.json')]):
        m=json.loads(f.read_text(encoding='utf-8'));p=ROOT/m['archivo'];sha=m['sha256']
        digest=cache.get(sha)
        if digest is None:
            digest=digest_pdf(p)
            if digest:cache[sha]=digest
        if not digest:continue
        examinados+=1
        if digest in indice:
            coincidencias.append({'archivo_original':m['archivo'],'sha256_original':sha,'sha1_base32_calculado':digest,'prueba_disponibilidad':'wayback_cdx_digest_matches_downloaded_pdf','alcance':'bytes_pdf_completos','payload_archivado_redescargado':False,**indice[digest]})
        if examinados%200==0:print('PDF contrastados',examinados,'coincidencias',len(coincidencias),flush=True)
    cachepath.write_text(json.dumps(cache,indent=2),encoding='utf-8')
    (AUD/'verificaciones_cdx.json').write_text(json.dumps(coincidencias,ensure_ascii=False,indent=2),encoding='utf-8')
    print('CDX',examinados,len(coincidencias),flush=True)

if __name__=='__main__':main()

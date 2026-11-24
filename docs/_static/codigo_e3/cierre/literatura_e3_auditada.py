"""Conserva búsquedas públicas y documentos primarios para la revisión de M4."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json,hashlib,datetime,sys,requests,re
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/cierre_e3_sin_entrenar_20261124/literatura'
OUT.mkdir(parents=True,exist_ok=True)
queries=[
 ('mecanismo','("load forecasting" OR "time series forecasting") AND ("residual memory" OR "residual retrieval" OR "retrieval augmented")'),
 ('aplicacion','("electricity demand" OR "load forecasting") AND (Colombia OR "Sistema Interconectado Nacional") AND (residual OR memory OR retrieval)'),
 ('compuerta','"load forecasting" AND (memory OR retrieval) AND (gating OR gate OR "mixture of experts")')]
sources=[
 ('RATL','https://arxiv.org/html/2609.03937v1','texto completo'),
 ('FreshCast','https://arxiv.org/html/2610.07834v3','texto completo'),
 ('ResMem','https://proceedings.neurips.cc/paper_files/paper/2023/file/bf0857cb9a41c73639f028a80301cdf0-Paper-Conference.pdf','texto completo'),
 ('TE_AG_GRU_RC','https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2026.1935983/full','texto completo'),
 ('Colombia','https://www.econjournals.com/index.php/ijeep/article/view/17528','resumen editorial y enlace al texto'),
 ('MMEMformer','https://www.sciencedirect.com/science/article/pii/S0360544225014045','secciones públicas; texto completo no acreditado')]
def download(record):
 name,url,scope=record
 try:
  r=requests.get(url,timeout=60,headers={'User-Agent':'Research project reproducibility/1.0'})
  suffix='.pdf' if 'application/pdf' in r.headers.get('Content-Type','') else '.html'
  p=OUT/(name+suffix);p.write_bytes(r.content)
  return dict(nombre=name,url=url,fecha='2026-11-24',estado_http=r.status_code,archivo=p.name,bytes=len(r.content),sha256=hashlib.sha256(r.content).hexdigest(),alcance=scope,acceso_valido=r.ok)
 except Exception as e:return dict(nombre=name,url=url,error=str(e),alcance=scope)
def search(record):
 base,key,q=record
 translated={'mecanismo':'residual memory retrieval time series load forecasting','aplicacion':'Colombia electricity demand residual memory forecasting','compuerta':'load forecasting memory retrieval gating'}[key]
 if base=='Crossref':url='https://api.crossref.org/works';params={'query':translated,'rows':25,'filter':'until-pub-date:2026-11-24'}
 elif base=='OpenAlex':url='https://api.openalex.org/works';params={'search':translated,'per-page':25,'filter':'to_publication_date:2026-11-24'}
 else:
  native={'mecanismo':'(all:"time series forecasting" OR all:"load forecasting") AND (all:"residual memory" OR all:"residual retrieval" OR all:"retrieval augmented")',
   'aplicacion':'(all:Colombia OR all:"Sistema Interconectado Nacional") AND (all:"electricity demand" OR all:"load forecasting")',
   'compuerta':'all:"load forecasting" AND (all:memory OR all:retrieval) AND (all:gating OR all:gate)'}
  url='https://export.arxiv.org/api/query';params={'search_query':native[key],'start':0,'max_results':25,'sortBy':'relevance'}
 record=dict(base=base,fecha='2026-11-24',consulta_booleana=q,traduccion=translated,parametros=params)
 try:
  r=requests.get(url,params=params,timeout=60);p=OUT/f'{base}_{key}.txt';p.write_text(r.text,encoding='utf-8')
  record.update(estado_http=r.status_code,url=r.url,archivo=p.name,sha256=hashlib.sha256(r.content).hexdigest())
  if r.ok and base=='Crossref':record.update(total=r.json()['message']['total-results'],recibidos=len(r.json()['message']['items']))
  if r.ok and base=='OpenAlex':record.update(total=r.json()['meta']['count'],recibidos=len(r.json()['results']))
  if r.ok and base=='arXiv':record['recibidos']=r.text.count('<entry>')
 except Exception as e:record['error']=str(e)
 return record
if __name__=='__main__':
 with ThreadPoolExecutor(max_workers=4) as pool:
  searches=list(pool.map(search,[(base,key,q) for base in ['Crossref','OpenAlex','arXiv'] for key,q in queries]))
  documents=list(pool.map(download,sources))
 audit=dict(fecha='2026-11-24',intervalo='Desde inicio de cada índice hasta2026-11-24',consultas=searches,documentos=documents,
  criterios_inclusion=['Memoria explícita o corrección residual de pronósticos continuos','Antecedentes directos de carga eléctrica y demanda colombiana','Descripción verificable del mecanismo en fuente primaria'],
  criterios_exclusion=['Memoria solo como estado LSTM sin recuperación explícita','ResNet sin banco de episodios: antecedente arquitectónico, no equivalente a memoria residual','Tareas ajenas sin mecanismo transferible','Resultados posteriores al corte o duplicados por DOI/título'],
  evaluacion='Un extractor; requiere revisión independiente por tercero',
  bases_requeridas='Las consultas públicas a Crossref, OpenAlex y arXiv no sustituyen exportaciones de Scopus, Web of Science e IEEE Xplore.',
  busqueda_web_complementaria=[{'fecha':'2026-11-24','consulta':q,'alcance':'Índice web restringido al dominio; no búsqueda nativa acreditada'} for q in [
   'site.ieeexplore.ieee.org ("load forecasting") ("residual" "memory")',
   'site.scopus.com ("load forecasting") ("retrieval" OR "memory")',
   'site.webofscience.com ("load forecasting") ("residual" OR "memory")']],
  conclusion='La recuperación y corrección de residuos tiene antecedentes directos. La contribución candidata de M4 reside en la integración temporal OOF de una base de expertos, proyecciones aprendidas y compuerta contextual para el SIN; no se acredita invención del principio general ni prioridad mundial.')
 (OUT/'registro.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps({r['base']+'_'+str(i):r.get('estado_http',r.get('error')) for i,r in enumerate(searches)},ensure_ascii=False),flush=True)

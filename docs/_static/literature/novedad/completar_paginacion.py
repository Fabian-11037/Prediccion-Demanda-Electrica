"""Completa las consultas truncadas del anexo sin alterar la captura histórica."""
from pathlib import Path
import concurrent.futures,datetime,json,time,urllib.parse,urllib.request
ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'book/_static/literature/novedad/busqueda_novedad_20261125.json'
OUT=ROOT/'literature/novedad/paginacion_20261126.json'

def fetch(url):
 for attempt in range(3):
  try:
   with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'SIN-demand-literature-audit/1.0'}),timeout=25)as r:return json.load(r)
  except Exception:
   if attempt==2:raise
   time.sleep(3*(attempt+1))

def complete(row):
 records=[];calls=[];seen=set();cursor='*';page=1;total=None
 try:
  while cursor:
   if row['base']=='OpenAlex':
    params={'filter':f"title_and_abstract.search:{row['consulta']},publication_year:2015-2026",'per-page':200,'page':page,'sort':'relevance_score:desc','select':'id,doi,display_name,publication_year,abstract_inverted_index,cited_by_count'};url='https://api.openalex.org/works?'+urllib.parse.urlencode(params)
   else:
    params={'query':row['consulta'],'fields':'title,year,externalIds,abstract,citationCount','year':'2015-2026'}
    if cursor!='*':params['token']=cursor
    url='https://api.semanticscholar.org/graph/v1/paper/search/bulk?'+urllib.parse.urlencode(params)
   d=fetch(url);calls.append({'url':url,'recibidos':len(d.get('results',d.get('data',[])))})
   if row['base']=='OpenAlex':
    total=d['meta']['count'];batch=d['results'];next_cursor=str(page+1)if batch and page*200<total else None;page+=1
    for w in batch:
     inv=w.get('abstract_inverted_index')or{};abstract=' '.join(t for _,t in sorted((p,t)for t,positions in inv.items()for p in positions))
     records.append({'id':w['id'],'doi':w.get('doi'),'titulo':w['display_name'],'anio':w.get('publication_year'),'resumen':abstract})
   else:
    total=d.get('total');next_cursor=d.get('token')
    for w in d.get('data',[]):records.append({'id':w['paperId'],'doi':(w.get('externalIds')or{}).get('DOI'),'titulo':w['title'],'anio':w.get('year'),'resumen':w.get('abstract')or''})
   if not next_cursor:break
   if next_cursor in seen:raise ValueError('Cursor repetido')
   seen.add(cursor);cursor=next_cursor
   if len(calls)>=100:raise ValueError('Límite operativo; consulta incompleta declarada')
   time.sleep(1)
  unique={w['id']:w for w in records}
  return {'cadena':row['cadena'],'base':row['base'],'consulta':row['consulta'],'total_indice':total,'recuperados':len(unique),'paginacion_agotada':True,'total_estimado':row['base']=='Semantic Scholar','llamadas':calls,'registros':list(unique.values()),'fecha':'2026-11-26'}
 except Exception as e:return {'cadena':row['cadena'],'base':row['base'],'consulta':row['consulta'],'error':str(e),'recuperados':len(records),'paginacion_agotada':False,'llamadas':calls,'registros':records,'fecha':'2026-11-26'}

def main():
 source=json.loads(SOURCE.read_text(encoding='utf-8'));todo=[r for r in source['consultas']if isinstance(r.get('resultados'),int)and r['resultados']>len(r.get('registros',[]))]
 with concurrent.futures.ThreadPoolExecutor(max_workers=2)as pool:results=list(pool.map(complete,todo))
 OUT.write_text(json.dumps({'fecha':'2026-11-26','fuente_historica':SOURCE.name,'alcance':'Repetición paginada de cinco consultas truncadas; no modifica las exportaciones de Scopus/WoS/IEEE ni acredita revisión independiente','consultas':results},ensure_ascii=False,indent=2),encoding='utf-8')
 for row in results:print(row['cadena'],row['base'],row.get('total_indice'),row['recuperados'],row['paginacion_agotada'],row.get('error',''),flush=True)
if __name__=='__main__':main()

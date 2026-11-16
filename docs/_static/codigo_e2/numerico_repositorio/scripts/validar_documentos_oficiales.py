"""Comprueba identidad, cobertura y disponibilidad del corpus integrado."""
from pathlib import Path
import hashlib,json,re,sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from demanda.corpus import cargar_corpus,documentos_en_origen

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    audit=ROOT/'results/documentos_oficiales';raw=ROOT/'data/raw/documentos_oficiales'
    c=pd.read_parquet(ROOT/'data/processed/corpus_texto.parquet');o=pd.read_parquet(raw/'documentos_oficiales.parquet');t=pd.read_parquet(ROOT/'data/raw/noticias/titulares.parquet')
    assert c.id.is_unique and o.id.is_unique and set(t.id)<=set(c.id)
    previo=json.loads((audit/'estado_previo.json').read_text(encoding='utf-8'))
    assert all(sha(ROOT/p)==v for p,v in previo.items())
    for r in o.itertuples():
        assert hashlib.sha256(r.texto.encode()).hexdigest()==r.sha256_texto,r.id
        assert hashlib.sha256((r.fuente+'\0'+r.sha256_original).encode()).hexdigest()==r.id
    pruebas=json.loads((audit/'verificaciones_cdx.json').read_text(encoding='utf-8'));pmap={r['sha256_original']:r for r in pruebas}
    q=o.loc[o.prueba_disponibilidad.eq('wayback_cdx_digest_matches_downloaded_pdf')]
    for r in q.itertuples():
        p=pmap[r.sha256_original]
        assert r.disponible_estricto==pd.to_datetime(p['timestamp'],format='%Y%m%d%H%M%S',utc=True)
        assert p['digest']==p['sha1_base32_calculado'] and p['alcance']=='bytes_pdf_completos'
        assert sha(ROOT/p['indice_archivo'])==p['indice_sha256']
        assert r.titulo_verificado==re.sub(r'[ \t]+',' ',re.sub(r'\[Página[^\]]*\]','',r.texto)[:300]).strip()
    for nombre,nfuentes in [('cobertura_mensual_fuente.csv',13),('oficiales_cobertura_mensual_fuente.csv',6)]:
        m=pd.read_csv(audit/nombre)
        assert len(m)==312*nfuentes*2 and not m.duplicated(['mes','fuente','escenario']).any()
        assert m.groupby(['fuente','escenario']).mes.nunique().eq(312).all()
        for h in [24,72,168]:
            assert m['cobertura_'+str(h)+'h'].between(0,1).all()
            assert np.allclose(m['cobertura_'+str(h)+'h'],m['origenes_con_texto_'+str(h)+'h']/m.origenes)
        assert (m.cobertura_24h<=m.cobertura_72h).all() and (m.cobertura_72h<=m.cobertura_168h).all()
    estricto=cargar_corpus('estricto');origen=pd.Timestamp('2023-08-15',tz='America/Bogota');sel=documentos_en_origen(estricto,origen,72)
    assert sel.disponible_desde.le(origen).all() and sel.disponible_desde.ge(origen-pd.Timedelta(hours=72)).all()
    nohist=o.loc[o.prueba_disponibilidad.eq('download_payload_sha256_and_retrieval')]
    assert nohist.disponible_estricto.ge(pd.Timestamp('2026-01-01',tz='UTC')).all()
    manifiesto=json.loads((ROOT/'data/processed/dataset_texto_oficial_manifest.json').read_text(encoding='utf-8'))
    assert all(sha(ROOT/r['archivo'])==r['sha256'] for r in manifiesto['archivos'])
    assert (ROOT/'CONTEXTO_PROYECTO.md').read_bytes()==(ROOT/'continuidad/CONTEXTO_PROYECTO.md').read_bytes()
    r={'resultado':'correcto','corpus_filas':len(c),'oficiales_unicos':len(o),'cuerpos_sha256_comprobados':len(o),'identificadores_unicos':True,'titulares_originales_conservados':len(t),'archivos_previos_sin_cambios':len(previo),'meses_por_fuente_escenario':312,'regla_asof_verificada':True,'entrenamiento_ejecutado':False,'publicacion_ejecutada':False}
    (audit/'validacion_final.json').write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(r,ensure_ascii=False),flush=True)

if __name__=='__main__':main()

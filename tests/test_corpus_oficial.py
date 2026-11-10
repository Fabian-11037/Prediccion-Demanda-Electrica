import pandas as pd
import pytest
from demanda.corpus import cargar_corpus,documentos_en_origen

def fila(id,fecha,verificada=None,prueba=None,cuarentena=False):
    return dict(id=id,fuente='IDEAM',tipo_fuente='official_report',familia_documental='boletin',titulo='Pronóstico',texto='Contenido',texto_alcance='full_body',disponible_declarado=fecha,disponible_estricto=verificada,prueba_disponibilidad=prueba,cuarentena=cuarentena)

def test_una_edicion_antigua_recuperada_hoy_no_entra_en_el_pasado_estricto(tmp_path):
    ruta=tmp_path/'corpus.parquet';pd.DataFrame([fila('a','2010-01-02T05:00Z','2026-11-09T10:00Z','download_payload_sha256_and_retrieval')]).to_parquet(ruta)
    declarado=cargar_corpus('declarado',ruta);estricto=cargar_corpus('estricto',ruta)
    assert len(documentos_en_origen(declarado,pd.Timestamp('2010-01-03T00:00Z')))==1
    assert documentos_en_origen(estricto,pd.Timestamp('2010-01-03T00:00Z')).empty
    assert len(documentos_en_origen(estricto,pd.Timestamp('2026-11-10T00:00Z')))==1

def test_ventana_incluye_lo_disponible_al_origen_excluye_futuro_y_respeta_bogota(tmp_path):
    fechas=['2022-01-01T04:59:59Z','2022-01-01T05:00:00Z','2022-01-02T04:59:59Z','2022-01-02T05:00:00Z','2022-01-02T05:00:01Z']
    ruta=tmp_path/'corpus.parquet';pd.DataFrame([fila(str(i),f) for i,f in enumerate(fechas)]).to_parquet(ruta)
    corpus=cargar_corpus('declarado',ruta)
    assert documentos_en_origen(corpus,pd.Timestamp('2022-01-02')).id.tolist()==['1','2','3']
    assert documentos_en_origen(corpus,pd.Timestamp('2022-01-02'),max_n=1).id.tolist()==['3']

def test_cuarentena_y_prueba_invalida_no_habilitan_texto(tmp_path):
    ruta=tmp_path/'corpus.parquet';pd.DataFrame([fila('a','2020-01-02','2020-01-02','fecha_declarada'),fila('b','2020-01-02','2020-01-02','wayback_payload_digest_and_capture',True)]).to_parquet(ruta)
    assert cargar_corpus('estricto',ruta).empty
    c=cargar_corpus('declarado',ruta)
    assert c.id.tolist()==['a']
    assert not {'url','prueba_disponibilidad','fecha_declarada'} & set(c.columns)

def test_rechaza_corpus_desordenado_y_ventanas_invalidas():
    c=pd.DataFrame({'disponible_desde':pd.to_datetime(['2020-01-02','2020-01-01'],utc=True)})
    with pytest.raises(ValueError):documentos_en_origen(c,pd.Timestamp('2020-01-03'))
    with pytest.raises(ValueError):documentos_en_origen(c,pd.Timestamp('2020-01-03'),ventana_horas=0)

def test_un_digest_del_pdf_no_verifica_un_titulo_editorial_actual(tmp_path):
    r=fila('pdf','2020-01-01','2020-01-01','wayback_cdx_digest_matches_downloaded_pdf')
    r['titulo']='Título editorial añadido después';r['titulo_verificado']='Encabezado extraído del PDF capturado'
    ruta=tmp_path/'corpus.parquet';pd.DataFrame([r]).to_parquet(ruta)
    assert cargar_corpus('estricto',ruta).titulo.tolist()==['Encabezado extraído del PDF capturado']
    assert cargar_corpus('declarado',ruta).titulo.tolist()==['Título editorial añadido después']

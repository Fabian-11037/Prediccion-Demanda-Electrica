"""Diccionario semántico del panel, el corpus y el inventario de ciudades."""
import csv
import json
from pathlib import Path
import pyarrow.parquet as pq


CAL = {
    'dow': ('Día de la semana: 0 lunes, 1 martes, 2 miércoles, 3 jueves, 4 viernes, 5 sábado y 6 domingo.', 'categoría 0–6'),
    'mes': ('Mes del año: 1 enero hasta 12 diciembre.', 'categoría 1–12'),
    'dia_anio': ('Posición del día dentro del año; el 29 de febrero se conserva en años bisiestos.', 'día 1–366'),
    'es_festivo': ('Indica si la fecha corresponde a un festivo del calendario colombiano implementado.', '0/1'),
    'es_domingo': ('Indica si la fecha es domingo.', '0/1'),
    'es_sabado': ('Indica si la fecha es sábado, sea o no festivo.', '0/1'),
    'es_no_laboral': ('Indica domingo o festivo. Un sábado no festivo toma valor 0 en esta definición operativa.', '0/1'),
    'tipo_dia': ('Tipo de día: 0 lunes a viernes no festivo; 1 sábado no festivo; 2 domingo o festivo.', 'categoría 0–2'),
    'dias_al_siguiente_festivo': ('Días hasta el festivo estrictamente posterior a la fecha actual; distancias mayores de 30 se truncan a 30.', 'días'),
    'dias_desde_festivo_previo': ('Días transcurridos desde el festivo estrictamente anterior a la fecha actual; distancias mayores de 30 se truncan a 30.', 'días'),
    'es_vispera_festivo': ('Indica que el día siguiente es festivo.', '0/1'),
    'es_dia_post_festivo': ('Indica que el día anterior es festivo.', '0/1'),
    'es_fin_semana_largo': ('Marca viernes, sábado, domingo y lunes del bloque cuyo viernes o lunes es festivo.', '0/1'),
    'es_semana_santa': ('Indica pertenencia al intervalo entre Domingo de Ramos y Domingo de Pascua, ambos incluidos.', '0/1'),
    'es_temporada_navidena': ('Indica pertenencia al periodo del 16 de diciembre al 6 de enero, ambos incluidos.', '0/1'),
    'es_fin_de_anio': ('Indica pertenencia al periodo del 24 de diciembre al 1 de enero, ambos incluidos.', '0/1'),
    'hora': ('Hora local de inicio del intervalo horario en America/Bogota.', 'hora 0–23'),
}
for nombre, base, periodo, significado in [
    ('hora', 'hora', 24, 'la hora local'), ('dow', 'dow', 7, 'el día de la semana'),
    ('doy', 'dia_anio', 366, 'el día del año')]:
    for sufijo, funcion in [('sin', 'seno'), ('cos', 'coseno')]:
        CAL[f'{nombre}_{sufijo}'] = (f'Codificación cíclica de {significado}: {funcion}(2π × {base} / {periodo}).', 'adimensional [−1, 1]')

TEXT = {
    'id': 'Identificador del registro documental para trazabilidad y deduplicación.',
    'source': 'Medio de comunicación o entidad que publica el titular.',
    'source_kind': 'Tipo de fuente registrado en el corpus, por ejemplo prensa o entidad institucional.',
    'title_kind': 'Tipo de pieza o encabezado registrado durante la recopilación.',
    'titular': 'Encabezado de la noticia o informe; contenido textual empleado en la exploración léxica.',
    'url': 'Dirección de la pieza original, utilizada para trazabilidad.',
    'publication_date_claimed': 'Fecha de publicación declarada por la fuente; no acredita por sí sola la disponibilidad de esa versión.',
    'published_at_claimed': 'Marca temporal de publicación declarada, con la precisión conservada por el archivo.',
    'date_precision': 'Precisión temporal de la fecha declarada, por ejemplo fecha diaria o marca con hora.',
    'availability_status': 'Estado de verificación de la disponibilidad histórica de la versión del titular.',
    'available_at_verified': 'Instante desde el que la versión archivada tiene disponibilidad verificable; determina el escenario estricto.',
    'capture_at_verified': 'Instante de captura archivística verificada que respalda la versión documental.',
    'availability_proof': 'Tipo de evidencia de disponibilidad; el escenario estricto exige wayback_payload_digest_and_capture.',
    'available_at_assumed': 'Disponibilidad asignada en el escenario declarado: se conserva la hora declarada cuando existe; con fecha sin hora se usa el día siguiente a las 00:00 de Colombia.',
    'assumed_availability_basis': 'Regla o evidencia utilizada para asignar available_at_assumed.',
    'quarantined': 'Indicador de cuarentena; los registros con valor verdadero se excluyen de los escenarios de análisis.',
    'quarantine_reason': 'Motivo documentado de la cuarentena del registro.',
    'provenance_source': 'Archivo o proceso de recuperación del que procede el registro.',
}
CITY = {
    'ciudad': ('Identificador de la ciudad utilizado en los nombres de las columnas meteorológicas.', 'texto'),
    'lat_solicitada': ('Latitud geográfica solicitada para consultar la serie de la ciudad.', 'grados decimales'),
    'lon_solicitada': ('Longitud geográfica solicitada para consultar la serie de la ciudad.', 'grados decimales'),
    'lat_malla': ('Latitud del punto de malla efectivamente utilizado por el proveedor meteorológico.', 'grados decimales'),
    'lon_malla': ('Longitud del punto de malla efectivamente utilizado por el proveedor meteorológico.', 'grados decimales'),
    'elevacion_m': ('Elevación reportada para el punto de consulta meteorológica.', 'm'),
    'habitantes_censo_2018_ajustado': ('Población del censo 2018 ajustada por omisión censal; determina los pesos urbanos estáticos. La cifra procede de una fuente secundaria que cita al DANE.', 'habitantes'),
}


def construir(root):
    root = Path(root)
    registry = json.loads((root/'data/processed/feature_registry.json').read_text(encoding='utf-8'))
    schema = pq.read_schema(root/'data/processed/panel_horario.parquet')
    ciudades = {'bogota':'Bogotá','medellin':'Medellín','cali':'Cali','barranquilla':'Barranquilla',
                'cartagena':'Cartagena','bucaramanga':'Bucaramanga','pereira':'Pereira','cucuta':'Cúcuta'}
    especiales = {
        'inicio': ('Inicio del intervalo horario al que corresponde la fila; índice temporal del panel.', 'fecha-hora America/Bogota'),
        'y_demanda_real_kwh': ('Energía eléctrica demandada por el SIN durante el intervalo horario objetivo.', 'kWh/intervalo horario'),
        'x_demanda_real_kwh': ('Demanda real horaria del SIN desplazada una fila horaria respecto de la serie original.', 'kWh/intervalo horario'),
        'x_precio_bolsa_cop_kwh': ('Precio horario de la energía en la bolsa de XM, desplazado 24 filas horarias respecto de la serie original.', 'COP/kWh'),
        'x_hid_llenado_valor_original': ('Indicador de llenado conservado en su escala original, correspondiente a la razón entre reserva útil y capacidad útil. No se multiplica por 100 ni se recortan valores superiores a 1.', 'razón adimensional'),
        'x_hid_reserva_util_kwh': ('Energía equivalente almacenada en la reserva útil de los embalses del SIN para la fecha hidráulica disponible.', 'kWh'),
        'x_hid_capacidad_util_kwh': ('Capacidad útil de almacenamiento de los embalses del SIN, expresada como energía equivalente.', 'kWh'),
        'x_hid_aportes_kwh_dia': ('Aportes hídricos diarios al SIN expresados como energía equivalente de los caudales recibidos.', 'kWh/día'),
        'aux_hid_fecha_dato': ('Fecha del dato hidrológico diario seleccionado por la unión temporal; permite auditar su antigüedad.', 'fecha'),
    }
    clima = {
        'temperature_2m_c': ('Temperatura del aire a 2 m sobre la superficie', '°C'),
        'relative_humidity_2m_pct': ('Humedad relativa del aire a 2 m sobre la superficie', '%'),
        'precipitation_preceding_hour_mm': ('Precipitación acumulada durante la hora precedente a la marca meteorológica', 'mm/hora precedente'),
    }
    filas = []
    for campo in schema:
        nombre = campo.name
        meta = registry.get(nombre, {})
        if nombre in especiales:
            significado, unidad = especiales[nombre]
        elif nombre.startswith('cal_'):
            significado, unidad = CAL[nombre[4:]]
        elif nombre.startswith('x_obs_'):
            coincidencias = [(v, u, sufijo) for prefijo,(v,u) in clima.items()
                             if nombre.startswith('x_obs_'+prefijo+'_')
                             for sufijo in [nombre[len('x_obs_'+prefijo+'_'):]]]
            assert len(coincidencias) == 1, nombre
            texto, unidad, ciudad = coincidencias[0]
            significado = (texto + ('; promedio de las ocho ciudades ponderado por población estática del censo 2018.'
                           if ciudad == 'ponderada' else f' para el punto de malla asociado a {ciudades[ciudad]}.'))
            significado += ' Reanálisis observado con desplazamiento de una fila horaria.'
        else:
            raise ValueError('Falta una definición específica: '+nombre)
        clase = meta.get('clase', 'indice')
        disponible = {'target':'Sólo etiqueta del horizonte; no es una entrada.',
            'known_future':'Conocida para la historia y el horizonte.',
            'past_only':'Sólo filas anteriores al origen; la fila t reúne información disponible hasta t+1 h.',
            'auditoria':'Sólo auditoría; excluida de entradas.',
            'indice':'Índice temporal; no equivale a una observación disponible al inicio del intervalo.'}[clase]
        if nombre.startswith('x_hid_'):
            disponible += ' Unión as-of: cierre del día hidráulico + 24 h (supuesto).'
        filas.append(dict(bloque='panel',variable=nombre,descripcion=significado,unidad=unidad,
            tipo=str(campo.type),clase=clase,rol=meta.get('rol','indice_temporal'),
            fuente=meta.get('fuente','Índice construido del panel'),rezago_h=meta.get('rezago_h'),disponibilidad=disponible))
    assert set(schema.names) == {f['variable'] for f in filas}
    assert set(k for k in registry if not k.startswith('_')) == set(schema.names)-{'inicio'}
    schema_t = pq.read_schema(root/'data/raw/noticias/titulares.parquet')
    assert set(schema_t.names) == set(TEXT)
    for campo in schema_t:
        filas.append(dict(bloque='titulares',variable=campo.name,descripcion=TEXT[campo.name],
            unidad='booleano' if campo.name=='quarantined' else 'texto / marca temporal conservada',
            tipo=str(campo.type),clase='texto' if campo.name=='titular' else 'metadato',
            rol='contenido_textual' if campo.name=='titular' else 'auditoria_y_filtrado',fuente='Archivo de titulares',
            rezago_h=None,disponibilidad='El titular se admite según el escenario declarado o estricto; los metadatos sólo filtran y auditan.'))
    with (root/'data/raw/numeric/ciudades.csv').open(encoding='utf-8-sig',newline='') as f:
        campos = csv.DictReader(f).fieldnames
    assert set(campos) == set(CITY)
    for nombre in campos:
        significado, unidad = CITY[nombre]
        filas.append(dict(bloque='ciudades',variable=nombre,descripcion=significado,unidad=unidad,
            tipo='texto' if nombre=='ciudad' else 'numérico',clase='metadato_estatico',rol='localizacion_y_ponderacion',
            fuente='Inventario de ciudades; población citada por Portafolio a partir del DANE',rezago_h=None,
            disponibilidad='Metadato estático; los pesos del censo 2018 se aplican también a años anteriores.'))
    destino=root/'data/processed/diccionario_datos.csv'
    with destino.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(filas[0]));w.writeheader();w.writerows(filas)
    (destino.with_suffix('.json')).write_text(json.dumps(filas,ensure_ascii=False,indent=2),encoding='utf-8')
    textos=['# Diccionario de datos',
        'Cada fila del panel representa un intervalo horario del SIN. Las variables se documentan por su significado físico u operativo, unidad y disponibilidad temporal. Las categorías del calendario describen la codificación implementada.',
        'Las columnas `x_*` incorporan los rezagos del panel y se consultan únicamente en filas anteriores al origen. Una fila con índice `t` puede contener información disponible hasta `t + 1 h`; el índice no es la fecha de publicación. Los rezagos de disponibilidad son supuestos del protocolo. Las columnas `cal_*` pueden consultarse en el horizonte; `aux_*` se reservan para auditoría.',
        '## Panel horario']
    grupos=[('Índice y objetivo',lambda f:f['variable'] in ('inicio','y_demanda_real_kwh')),
            ('Demanda y precio históricos',lambda f:f['variable'] in ('x_demanda_real_kwh','x_precio_bolsa_cop_kwh')),
            ('Meteorología',lambda f:f['variable'].startswith('x_obs_')),
            ('Hidrología y auditoría',lambda f:f['variable'].startswith(('x_hid_','aux_'))),
            ('Calendario',lambda f:f['variable'].startswith('cal_'))]
    def tabla(rows,cols):
        return '\n'.join(['| '+' | '.join(t for _,t in cols)+' |','| '+' | '.join('---' for _ in cols)+' |']+
                         ['| '+' | '.join(str(r[k] if r[k] is not None else '—').replace('|','/').replace('\n',' ') for k,_ in cols)+' |' for r in rows])
    for titulo,seleccionar in grupos:
        textos.extend(['### '+titulo,tabla([f for f in filas if f['bloque']=='panel' and seleccionar(f)],
                      [('variable','Variable'),('descripcion','Significado'),('unidad','Unidad')])])
    textos.extend(['## Disponibilidad y fuentes',tabla([
        {'familia':'Demanda real','fuente':'XM','regla':'Objetivo horario; x_demanda desplazada 1 h.'},
        {'familia':'Precio de bolsa','fuente':'XM','regla':'Precio desplazado 24 h; sólo historia anterior al origen.'},
        {'familia':'Meteorología','fuente':'ERA5 vía Open-Meteo','regla':'Reanálisis observado desplazado 1 h; no es pronóstico meteorológico.'},
        {'familia':'Hidrología','fuente':'XM diario','regla':'Dato diario disponible tras el cierre + 24 h; unión temporal hacia atrás.'},
        {'familia':'Calendario','fuente':'Calendario colombiano implementado','regla':'Conocido para el horizonte.'}],
        [('familia','Familia'),('fuente','Fuente'),('regla','Regla aplicada')]),
        'La demanda comercial se excluye como entrada por su relación con la demanda real y su liquidación posterior. El precio contemporáneo al objetivo y el clima observado del horizonte no son entradas conocidas al origen.',
        '## Corpus de titulares',
        'El contenido analítico es `titular`. Identificadores, fuentes, fechas, URLs y evidencias permiten seleccionar registros y auditar cobertura; no son atributos textuales del pronóstico.',
        tabla([f for f in filas if f['bloque']=='titulares'],[('variable','Campo'),('descripcion','Significado')]),
        '## Inventario de ciudades',tabla([f for f in filas if f['bloque']=='ciudades'],[('variable','Campo'),('descripcion','Significado'),('unidad','Unidad')]),
        'El diccionario estructurado se conserva en `data/processed/diccionario_datos.csv` y `data/processed/diccionario_datos.json`. Su cobertura se verifica contra los esquemas de los archivos, sin consultar valores del conjunto de prueba.'])
    (root/'book/diccionario.md').write_text('\n\n'.join(textos)+'\n',encoding='utf-8')
    return filas


if __name__=='__main__':
    construir(Path(__file__).resolve().parents[1])

"""Instala los esquemas suministrados sin alterar modelos ni resultados."""
from pathlib import Path
import hashlib,json,shutil

MODELOS={
 'arx':(3,4,'La historia se divide por su nivel y se seleccionan 48 rezagos. La concatenación con el calendario produce 600 atributos; una matriz genera las 24 salidas.',
        'Cada hora utiliza una fila distinta de la matriz de pesos. El sesgo completa la combinación lineal; la pérdida incorpora penalización lasso y el pronóstico se restituye a kWh con el nivel de la ventana.'),
 'mlp':(5,6,'Los 168 rezagos y el calendario futuro forman 720 atributos. Dos capas de 128 neuronas combinan esas entradas antes de producir simultáneamente las 24 horas.',
        'Cada neurona aplica una combinación afín, ReLU y dropout durante el ajuste. La pérdida Huber se calcula en unidades normalizadas; el dropout se desactiva en evaluación.'),
 'cnn1d':(7,8,'La convolución extrae patrones locales de la demanda y del calendario pasado. La agrupación reduce la historia; su representación se combina con el calendario futuro para obtener 24 correcciones.',
        'Cada filtro consulta cinco posiciones de la historia y los 24 canales de entrada. AvgPool combina pares de horas; los 1.344 atributos resultantes y los 552 del calendario alimentan la lectura lineal.'),
 'lstm':(9,10,'Una misma celda recurrente procesa las 168 horas. El estado oculto final de 48 componentes se concatena con el calendario del horizonte para generar las 24 salidas.',
        'Las compuertas de olvido, entrada y salida regulan la memoria y el estado oculto. La demanda se expresa respecto del nivel y de la dispersión de ajuste; la lectura utiliza el estado final y el calendario conocido.'),
 'patchtst':(11,12,'Cada canal se divide en parches y pasa por el mismo codificador de tres capas. La cabeza genera 24 valores por canal; una lectura afín reúne esos valores con el calendario futuro para pronosticar demanda.',
        'La atención y la red interna se combinan con conexiones residuales y BatchNorm. La lectura de cada hora utiliza 83 pronósticos de canal y 23 atributos de calendario, con un total de 106 entradas.'),
 'nhits':(13,14,'Los tres bloques explican progresivamente la historia y suman contribuciones al pronóstico. Las salidas interpoladas de 6, 12 y 24 nudos se añaden al último nivel observado.',
        'Cada bloque agrupa historia y exógenas, calcula coeficientes con un MLP y separa backcast y pronóstico. El backcast se resta de la historia; la interpolación fija lleva los nudos a las 24 horas del horizonte.'),
 'tft':(15,16,'La selección de variables transforma las 56 entradas históricas y las 23 de calendario futuro a representaciones de dimensión 160. La recurrencia, las compuertas y la atención de cuatro cabezas producen una salida por hora.',
        'Las GRN combinan transformaciones no lineales con conexiones residuales y compuertas. La selección softmax pondera las representaciones de las variables en cada instante; esos pesos describen el mecanismo interno y no efectos causales.'),
 'tcn':(1,2,'Los 56 canales históricos atraviesan tres bloques causales con dilataciones 1, 4 y 16. Su campo receptivo de 169 horas cubre la ventana de 168 horas; el último estado se combina con el calendario futuro para producir las 24 salidas.',
        'Cada bloque contiene dos convoluciones de cinco posiciones, recorte del relleno derecho y una conexión residual. El bloque central aplica dropout de 0,2; el atajo proyecta los canales cuando cambia su dimensión. La lectura combina 64 atributos temporales y 64 de calendario.'),
 'itransformer':(3,4,'Cada variable histórica se representa mediante un token de dimensión 128. El calendario aporta otros 23 tokens construidos con sus 168 horas pasadas y 24 futuras. Dos capas de atención relacionan los 56 tokens; la lectura del token de demanda genera el horizonte.',
        'La proyección se comparte entre variables del mismo tipo, mientras una identidad aprendida distingue cada canal. Cada capa combina atención de ocho cabezas, conexiones residuales, normalización y una red interna de dimensión 256. La atención opera entre variables.'),
 'tats':(5,6,'El texto oficial se resume mediante relevancia aprendida en siete cierres diarios. Doce atributos y un indicador de presencia por día se repiten sobre las horas correspondientes y se añaden a los 33 canales numéricos. PatchTST procesa los 46 canales y combina la lectura de demanda con una lectura de contexto.',
        'Los embeddings E5 permanecen congelados; las proyecciones de consulta, clave y valor aprenden a ponderar documentos disponibles. Las dos lecturas producen 24 correcciones: una utiliza el canal de demanda y otra reúne los demás canales con el calendario. Sin documentos, los canales textuales son cero.'),
 'fusion':(7,8,'Una rama PatchTST procesa la historia numérica y el calendario, mientras otra transforma un resumen de documentos oficiales en 24 correcciones. Ambas ramas se ajustan conjuntamente; un escalar aprendido pondera la contribución textual antes de sumar las salidas.',
        'La rama documental utiliza una consulta numérica para ponderar los embeddings disponibles y un MLP de dimensiones 64, 128 y 24. Su última capa inicia en cero; el escalar comienza en 0,5 y se aprende sin acotación. El indicador de presencia anula exactamente la contribución documental cuando falta texto.'),
 'tgforecaster':(9,10,'La historia se divide en parches y se combina con descripciones estáticas de las 33 variables. El calendario pasado completa una memoria de 700 tokens. Cinco consultas incorporan el calendario del horizonte, la descripción de demanda y el resumen documental; la atención cruzada genera parches de pronóstico superpuestos.',
        'La identidad de cada variable se proyecta sobre sus parches históricos. Las consultas del horizonte atienden esa memoria y producen cinco parches de ocho horas, desplazados cuatro horas. Pesos aprendidos, normalizados entre los parches que cubren cada hora, reconstruyen las 24 salidas. La variante numérica conserva las descripciones estáticas y omite los documentos.')}

def huella(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def instalar(root):
    root=Path(root);source=root/'results/esquemas_presentacion'
    if not (source/'manifest.json').is_file():return
    manifest=json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    dest=root/'book/_static/arquitecturas_presentacion';dest.mkdir(exist_ok=True)
    for key,(_,_,globaltext,detailtext) in MODELOS.items():
        for kind in ['global','detalle']:
            entry=manifest['imagenes'][key][kind];src=source/entry['archivo']
            assert huella(src)==entry['sha256'];shutil.copy2(src,dest/f'{key}_{kind}.png')
        p=root/'book/arquitecturas'/f'{key}.md'
        if not p.is_file():continue
        t=p.read_text(encoding='utf-8');a=t.find('### Esquema global')
        if a<0:a=t.index('![Flujo de ')
        b=t.index('### Código de la arquitectura',a)
        gp=f'../_static/arquitecturas_presentacion/{key}_global.png';dt=f'../_static/arquitecturas_presentacion/{key}_detalle.png'
        block=f'''### Esquema global

[![Arquitectura global de {key}]({gp})]({gp})

{globaltext}

### Detalle de las operaciones

[![Detalle de la arquitectura de {key}]({dt})]({dt})

{detailtext}

El {{ref}}`código de exportación de los esquemas <exportacion-esquemas>` forma parte del preprocesamiento compartido.

'''
        p.write_text(t[:a]+block+t[b:],encoding='utf-8')
    ps=root/'scripts/exportar_esquemas_presentacion.ps1';shutil.copy2(ps,root/'book/_static/codigo_e2'/ps.name)
    js=root/'scripts/crear_esquemas_ampliados.mjs';shutil.copy2(js,root/'book/_static/codigo_e2'/js.name)
    p=root/'book/arquitecturas/fundamentos.md';t=p.read_text(encoding='utf-8');marker='\n(exportacion-esquemas)=\n## Exportación de los esquemas'
    t=t.split('\n(exportacion-esquemas)=')[0].split('\n## Exportación de los esquemas')[0]+marker+'''

Los diagramas globales y de detalle se conservan como imágenes de alta resolución; al pulsarlos se abre el archivo completo. Las dimensiones y las operaciones corresponden a las configuraciones evaluadas. Los diagramas de TCN, iTransformer, TaTS, fusión tardía y TGForecaster se construyen con objetos editables, siguiendo el formato gráfico de la presentación.

El siguiente código genera sus bloques, conexiones, ecuaciones y archivos editables a partir de las dimensiones documentadas.

```{literalinclude} ../_static/codigo_e2/crear_esquemas_ampliados.mjs
:language: javascript
:linenos:
```

La extracción utiliza PowerPoint en Windows. La construcción del JBook usa las imágenes incluidas y no necesita PowerPoint ni ejecutar modelos.

```{literalinclude} ../_static/codigo_e2/exportar_esquemas_presentacion.ps1
:language: powershell
:linenos:
```
'''
    p.write_text(t,encoding='utf-8')

if __name__=='__main__':instalar(Path(__file__).resolve().parents[1])

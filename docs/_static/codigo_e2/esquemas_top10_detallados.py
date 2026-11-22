"""Esquemas vectoriales de las arquitecturas congeladas, sin ejecutar redes."""
from pathlib import Path
import json,hashlib,shutil
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch,Circle,Rectangle

ROOT=Path(__file__).resolve().parents[1]
P={
 'data':('#A9DDD6','#1CA497'),'layer':('#E2F0D9','#7BA85B'),
 'weight':('#FFE6C2','#FF9817'),'op':('#FBCACA','#F34C54'),
 'act':('#FFD5B3','#F58025'),'cal':('#FFF2C6','#EFB916'),
 'vector':('#E7E8E9','#949494'),'union':('#0C4052','#0C4052'),
 'output':('#EEE5F5','#B59BD5')}
CARD=[('#E3EFFB','#A8C8ED'),('#E8F2DE','#AED197'),('#EEE5F5','#B69BD3'),('#FFF8DD','#E9CB5B')]
INK='#25333F';WIRE='#43515F';TEXT=[];LAYOUT=[]
fonts=set(font_manager.get_font_names());SANS='Calibri' if 'Calibri' in fonts else 'DejaVu Sans';SERIF='Cambria' if 'Cambria' in fonts else 'DejaVu Serif'
plt.rcParams.update({'font.family':SANS,'mathtext.fontset':'stix','svg.fonttype':'none'})


def text(ax,x,y,label,fs=11,bold=False,ha='center',color=INK):
    for old,new in {'rᵦ₊₁':r'$r_{b+1}$','rᵦ':r'$r_b$','uᵦ':r'$u_b$','b̂ᵦ':r'$\widehat b_b$','f̂ᵦ':r'$\widehat f_b$','Fᵦ₋₁':r'$F_{b-1}$','Fᵦ':r'$F_b$'}.items():label=label.replace(old,new)
    return ax.text(x,y,label,fontsize=fs,fontweight='bold' if bold else 'normal',ha=ha,va='center',color=color,zorder=5,linespacing=1.25)


def box(ax,x,y,w,h,label,kind='layer',fs=11,bold=False):
    fc,ec=P[kind];ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=10',fc=fc,ec=ec,lw=1.4,zorder=3))
    t=text(ax,x+w/2,y+h/2,label,fs,bold,color='white' if kind=='union' else INK);TEXT.append((t,x,y,w,h));return t


def circle(ax,x,y,label,r=21,kind='op',fs=20):
    fc,ec=P[kind];ax.add_patch(Circle((x,y),r,fc=fc,ec=ec,lw=1.6,zorder=3));return text(ax,x,y,label,fs)


def pipe(ax,points,dash=False,color=WIRE,arrow=True,lw=1.5):
    xs,ys=zip(*points);ax.plot(xs,ys,color=color,lw=lw,ls=(0,(5,3)) if dash else '-',zorder=2)
    if arrow:ax.add_patch(FancyArrowPatch(points[-2],points[-1],arrowstyle='-|>',mutation_scale=12,lw=lw,color=color,shrinkA=0,shrinkB=0,zorder=2))


def frame(ax,x,y,w,h,label):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=18',fc='white',ec=WIRE,lw=1.5,ls=(0,(5,4)),zorder=0))
    text(ax,x+w/2,y+29,label,14,bold=True)


def canvas(family,kind,info):
    global TEXT
    TEXT=[];fig,ax=plt.subplots(figsize=(16,9),dpi=200);fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
    ax.set(xlim=(0,1600),ylim=(900,0));ax.axis('off')
    title=('N-BEATS' if family=='nbeats' else 'TimesNet')+(': arquitectura global' if kind=='global' else ': detalle de bloques')
    ax.text(58,75,title,fontfamily=SERIF,fontsize=31,fontweight='bold',color=P['union'][0],va='center')
    params=f"{info['parametros']:,}".replace(',',' ');mase=f"{info['MASE']:.3f}".replace('.',',')
    text(ax,60,149,f'N · {params} parámetros · MASE {mase}'+(' · prueba: lote 1' if family=='timesnet' else ''),11.5,ha='left',color='#61717B')
    for i,(kind2,label) in enumerate([('data','Datos'),('layer','Capa'),('weight','Pesos'),('op','Operación'),('act','Activación'),('cal','Calendario'),('vector','Vector'),('union','Unión')]):
        x=635+i*112;fc,ec=P[kind2]
        if kind2 in ['data','op']:ax.add_patch(Circle((x,149),8,fc=fc,ec=ec,lw=1))
        else:ax.add_patch(Rectangle((x-8,141),16,16,fc=fc,ec=ec,lw=1))
        text(ax,x+14,149,label,9,ha='left',color='#61717B')
    return fig,ax


def cards(ax,entries):
    for i,(heading,formula,note) in enumerate(entries):
        x=45+i*385;fc,ec=CARD[i]
        ax.add_patch(FancyBboxPatch((x,665),360,192,boxstyle='round,pad=0,rounding_size=12',fc=fc,ec=ec,lw=1.4,zorder=0))
        text(ax,x+180,698,f'{i+1}. {heading}',13.5,bold=True)
        t=text(ax,x+180,757,formula,17);TEXT.append((t,x+8,725,344,70))
        t=text(ax,x+180,824,note,10.5);TEXT.append((t,x+10,800,340,45))


def save(fig,ax,family,kind,root):
    # Ajustar la tipografía al cuadro y comprobar que el texto no rebasa su área.
    for _ in range(3):
        fig.canvas.draw();renderer=fig.canvas.get_renderer();changed=False
        for item,x,y,w,h in TEXT:
            bb=item.get_window_extent(renderer);a=ax.transData.transform((x+7,y+5));b=ax.transData.transform((x+w-7,y+h-5))
            width,height=abs(b[0]-a[0]),abs(b[1]-a[1]);ratio=min(width/max(bb.width,1),height/max(bb.height,1))
            if ratio<.98:item.set_fontsize(max(8.8,item.get_fontsize()*ratio*.98));changed=True
        if not changed:break
    fig.canvas.draw();renderer=fig.canvas.get_renderer();over=[]
    for item,x,y,w,h in TEXT:
        bb=item.get_window_extent(renderer);a=ax.transData.transform((x,y));b=ax.transData.transform((x+w,y+h))
        if bb.width>abs(b[0]-a[0])+1 or bb.height>abs(b[1]-a[1])+1:over.append(item.get_text())
    assert not over,(family,kind,'Texto fuera del bloque',over)
    out=Path(root)/'book/_static/arquitecturas_presentacion';out.mkdir(exist_ok=True,parents=True)
    for ext in ['png','svg']:fig.savefig(out/f'{family}_{kind}.{ext}',dpi=200,facecolor='white')
    plt.close(fig);LAYOUT.append(dict(modelo=family,tipo=kind,bloques_de_texto=len(TEXT),textos_fuera_de_bloque=0,tamano_png=[3200,1800]))


def nbeats_global(root,info):
    fig,ax=canvas('nbeats','global',info);frame(ax,45,205,1510,430,'Treinta bloques: reconstruir la historia y sumar contribuciones al horizonte')
    box(ax,80,282,180,58,'Exógenas + máscaras\n24 × 32 = 768','vector')
    box(ax,305,282,175,58,'Calendario futuro\n24 × 23 = 552','cal')
    box(ax,515,280,85,62,'Unir\n1320','union',11,True)
    pipe(ax,[(260,311),(280,311),(280,262),(557,262),(557,280)])
    pipe(ax,[(480,311),(515,311)])
    pipe(ax,[(600,311),(620,311),(620,365),(1160,365)],arrow=False,color='#458F86')
    pipe(ax,[(620,365),(425,365)],arrow=False,color='#458F86')
    text(ax,920,346,'Contexto compartido C · idéntico en cada bloque',10.5,color='#367E76')
    box(ax,80,400,180,75,'r₀: demanda\n168 valores\norden inverso','vector',11)
    for x,k in [(340,'1'),(660,'2'),(1075,'30')]:
        box(ax,x,400,170,75,f'Bloque {k}\n4 densas × 512\nReLU · dropout 0,1','layer',10.8)
        pipe(ax,[(x+85,365),(x+85,400)],color='#458F86')
        pipe(ax,[(x+85,475),(x+85,530)],arrow=True)
        text(ax,x+100,502,f'f{k}: 24',10,ha='left')
    pipe(ax,[(260,438),(340,438)])
    for start,x,nextx,label in [(290,560,660,'r₁'),(625,880,950,'r₂')]:
        pipe(ax,[(start,438),(start,385),(x,385),(x,418)],dash=True)
        circle(ax,x,438,'−',20)
        pipe(ax,[(510 if x==560 else 830,438),(x-20,438)])
        pipe(ax,[(x+20,438),(nextx,438)])
        text(ax,(x+20+nextx)/2,462,label,10.5)
    text(ax,1008,438,r'$\cdots$',27);pipe(ax,[(1045,438),(1075,438)])
    pipe(ax,[(1045,438),(1045,385),(1285,385),(1285,418)],dash=True)
    text(ax,1045,463,'r₂₉',10.5)
    circle(ax,1285,438,'−',20)
    pipe(ax,[(1245,438),(1265,438)])
    pipe(ax,[(1305,438),(1360,438)])
    text(ax,1385,438,'r₃₀',11)
    pipe(ax,[(425,530),(1290,530)],arrow=True)
    circle(ax,1310,530,'Σ',20);pipe(ax,[(1330,530),(1370,530)])
    circle(ax,1390,530,'1+',20,fs=14);pipe(ax,[(1410,530),(1430,530)])
    box(ax,1430,507,55,46,'×ℓ','weight',16);pipe(ax,[(1485,530),(1500,530)])
    circle(ax,1520,530,'ŷ',20,'output',20)
    text(ax,1320,580,'24 horas · corrección acumulada y restitución del nivel',11)
    text(ax,620,593,'Cada residuo r se actualiza; el contexto C se conserva. Los bloques no comparten parámetros.',11)
    cards(ax,[('Entrada',r'$u_b=[r_b;z_{24};c_f]$','1488 = 168 + 768 + 552'),
              ('Representación',r'$h_b=\mathrm{MLP}_b(u_b)$','4 capas, ancho 512 y dropout 0,1'),
              ('Reconstrucción',r'$r_{b+1}=r_b-\widehat b_b$','Backcast: 168 valores de historia'),
              ('Pronóstico',r'$\widehat{\mathbf{y}}=\ell[1+\sum_{b=1}^{30}\widehat f_b]$','Suma de 30 contribuciones de 24 horas')])
    save(fig,ax,'nbeats','global',root)


def nbeats_detalle(root,info):
    fig,ax=canvas('nbeats','detalle',info)
    frame(ax,45,205,920,430,'Dentro de un bloque: capas densas y dos salidas')
    frame(ax,990,205,565,430,'Actualización de historia y pronóstico')
    text(ax,500,274,'Entrada: residuo (168) + exógenas (768) + calendario (552)',11)
    box(ax,75,335,105,75,'uᵦ\n1488','vector',15)
    for i,x in enumerate([215,395,575,755]):
        box(ax,x,335,145,75,('1488 → 512' if i==0 else '512 → 512')+'\nReLU\nDropout 0,1','layer',10.5)
        if i==0:pipe(ax,[(180,372),(215,372)])
        else:pipe(ax,[(x-35,372),(x,372)])
    pipe(ax,[(827,410),(827,463)])
    box(ax,745,463,165,64,'Wθ: 512 → 192\nBase genérica','weight',11)
    pipe(ax,[(745,495),(650,495)],arrow=False)
    box(ax,390,452,190,58,'Backcast b̂ᵦ\n168 valores','vector',11.5)
    box(ax,390,545,190,58,'Pronóstico f̂ᵦ\n24 valores','vector',11.5)
    pipe(ax,[(650,495),(625,495),(625,481),(580,481)])
    pipe(ax,[(650,495),(650,574),(580,574)])
    text(ax,792,571,'Un bloque: 1 648 832 parámetros\n30 bloques independientes',10.5)
    box(ax,1030,300,150,65,'Residuo rᵦ\n168 valores','vector',11)
    circle(ax,1280,332,'−',21);pipe(ax,[(1180,332),(1259,332)])
    box(ax,1380,300,140,65,'rᵦ₊₁\n168 valores','vector',12);pipe(ax,[(1301,332),(1380,332)])
    box(ax,1210,392,140,53,'Backcast b̂ᵦ','vector',11);pipe(ax,[(1280,392),(1280,353)])
    box(ax,1030,525,150,65,'Acumulado Fᵦ₋₁\n24 valores','vector',10.5)
    circle(ax,1280,557,'+',21);pipe(ax,[(1180,557),(1259,557)])
    box(ax,1380,525,140,65,'Fᵦ\n24 valores','vector',12);pipe(ax,[(1301,557),(1380,557)])
    box(ax,1210,458,140,53,'Pronóstico f̂ᵦ','vector',11);pipe(ax,[(1280,511),(1280,536)])
    cards(ax,[('Capas ocultas',r'$h^{(j)}=D_{0,1}[\mathrm{ReLU}(W_jh^{(j-1)}+a_j)]$','Cuatro capas; dropout sólo durante ajuste'),
              ('Separación',r'$\theta_b=W_\theta h^{(4)}+a_\theta$','192 coeficientes: 168 + 24'),
              ('Historia residual',r'$r_{b+1}=r_b-\widehat b_b$','El siguiente bloque recibe el residuo'),
              ('Suma de salidas',r'$F_b=F_{b-1}+\widehat f_b$','F₀ = 0; salida final ℓ(1 + F₃₀)')])
    save(fig,ax,'nbeats','detalle',root)


def timesnet_global(root,info):
    fig,ax=canvas('timesnet','global',info)
    frame(ax,45,205,1510,430,'De la historia multivariada a periodos, representaciones 2D y pronóstico')
    circle(ax,100,368,'X',27,'data',22)
    box(ax,170,320,160,96,'Normalización\npor ventana\n168 × 33','op',11)
    pipe(ax,[(127,368),(170,368)])
    text(ax,105,425,'168 h\n33 canales',10.5)
    ax.add_patch(FancyBboxPatch((365,282),290,184,boxstyle='round,pad=0,rounding_size=10',fc='#F1F6FB',ec='#A8C8ED',lw=1.2,zorder=0))
    text(ax,510,300,'Representación · dimensión 16',10.8,bold=True)
    box(ax,380,320,150,37,'Conv1D: 33 → 16','layer',10)
    box(ax,380,365,150,37,'Posición sinusoidal','vector',10)
    box(ax,380,410,150,42,'Calₚ: 23 → 16','cal',10)
    circle(ax,615,371,'Σ',17,fs=16)
    pipe(ax,[(330,368),(350,368),(350,338),(380,338)])
    pipe(ax,[(530,338),(565,338),(565,371),(598,371)])
    pipe(ax,[(530,383),(580,383),(580,371),(598,371)])
    pipe(ax,[(530,431),(615,431),(615,388)])
    pipe(ax,[(632,371),(690,371)])
    text(ax,665,445,'Dropout 0,1',9.8)
    box(ax,690,326,150,90,'Extensión lineal\n168 → 192\n192 × 16','layer',11)
    box(ax,880,326,150,90,'TimesBlock 1\n5 periodos\nLayerNorm','layer',11)
    box(ax,1070,326,150,90,'TimesBlock 2\n5 periodos\nLayerNorm','layer',11)
    box(ax,1260,326,180,90,'Proyección: 16 → 33\nDesnormalización\n192 × 33','layer',10.8)
    for left,right in [(840,880),(1030,1070),(1220,1260)]:pipe(ax,[(left,371),(right,371)])
    pipe(ax,[(250,320),(250,266),(1350,266),(1350,326)],dash=True)
    text(ax,1010,254,'Media μ y desviación σ de cada canal en su ventana',10,color='#61717B')
    pipe(ax,[(1350,416),(1350,476),(730,476),(730,545),(750,545)])
    box(ax,750,515,155,60,'Últimas 24 h\nDemanda: 24 × 1','vector',10.8)
    pipe(ax,[(905,545),(960,545)])
    box(ax,960,515,150,60,'Unir\n24 × (1 + 23)','union',11,True)
    box(ax,750,590,180,36,'F: 24 h × 23','cal',10.5)
    pipe(ax,[(930,608),(1035,608),(1035,575)])
    pipe(ax,[(1110,545),(1160,545)])
    box(ax,1160,515,135,60,'Lectura compartida\n24 → 1 por hora','weight',10.5)
    pipe(ax,[(1295,545),(1329,545)]);circle(ax,1350,545,'1+',21,fs=14)
    pipe(ax,[(1371,545),(1400,545)]);box(ax,1400,522,60,46,'×ℓ','weight',16)
    pipe(ax,[(1460,545),(1499,545)]);circle(ax,1520,545,'ŷ',21,'output',21)
    cards(ax,[('Representación',r'$E=\mathrm{Drop}[\mathrm{Conv}(Z)+P+W_cC_p]$','Valor, posición y calendario histórico'),
              ('Periodos',r'$p_i=\lfloor192/f_i\rfloor,\quad i=1,\ldots,5$','Dos bloques vuelven a calcular sus periodos'),
              ('Salida numérica',r'$D=\mathrm{Linear}(E)\odot\sigma+\mu$','Se retienen 24 horas del canal de demanda'),
              ('Pronóstico',r'$\widehat y_{t+h}=\ell[1+W_o(d_{t+h};c_{t+h})]$','Calendario futuro y una lectura por hora')])
    save(fig,ax,'timesnet','global',root)


def timesnet_detalle(root,info):
    fig,ax=canvas('timesnet','detalle',info)
    frame(ax,45,205,430,430,'1. Identificar cinco periodos')
    frame(ax,495,205,680,430,'2. Procesar cada representación periódica')
    frame(ax,1195,205,360,430,'3. Ponderar y añadir el residuo')
    box(ax,75,300,125,64,'E\n192 × 16','vector',14)
    box(ax,270,300,170,64,'FFT temporal\nAmplitudes |FFT(E)|','layer',10.8)
    pipe(ax,[(200,332),(270,332)])
    box(ax,270,402,170,64,'Promedio del lote\ny de canales\nDC = 0; top 5','op',10.4)
    pipe(ax,[(355,364),(355,402)])
    box(ax,270,518,170,65,r'$p_i=\lfloor192/f_i\rfloor$'+'\nCinco periodos','vector',11.2)
    pipe(ax,[(355,466),(355,518)])
    text(ax,150,433,'f₁  f₂  f₃  f₄  f₅\nFrecuencias\nseleccionadas',11)
    text(ax,150,548,'Inferencia:\nun origen\npor llamada',11)
    text(ax,830,271,'Mismos pesos convolucionales en las cinco ramas de un bloque',10.7)
    box(ax,780,322,195,270,'Inception 2D\n\n6 Conv2D → media\n16 → 16\n\nGELU\n\n6 Conv2D → media\n16 → 16','layer',11)
    pipe(ax,[(440,550),(483,550)],arrow=False)
    pipe(ax,[(483,348),(483,564)],arrow=False)
    for i,y in enumerate([328,382,436,490,544],1):
        pipe(ax,[(483,y+20),(515,y+20)])
        box(ax,515,y,55,40,f'p{i}','vector',12)
        # Rejilla simbólica: no representa valores empíricos de una ventana.
        for row in range(2):
            for col in range(4):ax.add_patch(Rectangle((610+col*22,y+3+row*13),20,11,fc='#DDECF5' if col%2==0 else '#C5DCEB',ec='#96B4CD',lw=.5,zorder=3))
        text(ax,654,y+39,'cᵢ × pᵢ',8.8)
        pipe(ax,[(570,y+20),(610,y+20)])
        pipe(ax,[(698,y+20),(780,y+20)])
        box(ax,1020,y,115,40,f'V{i}: 192 × 16','vector',10)
        pipe(ax,[(975,y+20),(1020,y+20)])
        pipe(ax,[(1135,y+20),(1183,y+20)],arrow=False)
    # Colector: las cinco salidas llegan a la suma ponderada.
    pipe(ax,[(1183,348),(1183,564)],arrow=False)
    pipe(ax,[(1183,420),(1230,420)])
    text(ax,830,614,'Rellenar a múltiplo de pᵢ → reordenar → convolucionar → restaurar y recortar a 192',10)
    box(ax,1230,290,110,55,'Amplitudes Aᵢ','vector',10)
    box(ax,1380,290,140,55,'Softmax\nw₁,…,w₅','op',11)
    pipe(ax,[(1340,317),(1380,317)])
    box(ax,1230,390,290,65,'Suma ponderada: Σᵢ wᵢ Vᵢ\n192 × 16','union',12,True)
    pipe(ax,[(1450,345),(1450,390)])
    pipe(ax,[(1375,455),(1375,499)])
    circle(ax,1375,520,'+',21)
    box(ax,1230,495,100,50,'Residuo E','vector',10.5);pipe(ax,[(1330,520),(1354,520)],dash=True)
    box(ax,1435,492,85,56,'LayerNorm','layer',9.5);pipe(ax,[(1396,520),(1435,520)])
    pipe(ax,[(1477,548),(1477,580)]);box(ax,1395,580,125,30,'E′: 192 × 16','vector',10)
    cards(ax,[('Selección espectral',r'$p_i=\lfloor T/f_i\rfloor,\quad T=192$','Sin frecuencia cero; cinco amplitudes dominantes'),
              ('Convolución 2D',r'$V_i=\mathrm{Restore}[I_2(\mathrm{GELU}(I_1(U_i)))]$','6 núcleos: 1, 3, 5, 7, 9 y 11 en cada I'),
              ('Pesos por periodo',r'$w=\mathrm{softmax}(A_1,\ldots,A_5)$','Amplitudes de cada origen evaluado'),
              ('Salida del bloque',r'$E\prime=\mathrm{LN}(E+\sum_iw_iV_i)$','Se aplican dos bloques, con normalización')])
    save(fig,ax,'timesnet','detalle',root)


def anadir_fuentes(root):
    root=Path(root);dest=root/'book/_static/codigo_e2';dest.mkdir(exist_ok=True,parents=True)
    for script in ['esquemas_top10_detallados.py','arquitecturas_top10.py']:
        shutil.copy2(root/'scripts'/script,dest/script)
    page=root/'book/arquitecturas/fundamentos.md'
    if page.exists():
        marker='(esquemas-top10-detallados)='
        source=page.read_text(encoding='utf8')
        if marker not in source:
            source+='''\n\n(esquemas-top10-detallados)=
## Código de los esquemas de N-BEATS y TimesNet

El código representa las dimensiones y conexiones de las configuraciones conservadas. Las cifras del encabezado proceden del benchmark y de los metadatos; la representación gráfica no ejecuta las redes.

```{literalinclude} ../_static/codigo_e2/esquemas_top10_detallados.py
:language: python
:linenos:
```
'''
            page.write_text(source,encoding='utf8')


def generar(root=ROOT):
    root=Path(root);out=root/'book/_static/arquitecturas_presentacion';audit=root/'results/esquemas_presentacion/top10_detallados';audit.mkdir(exist_ok=True)
    backup=audit/'respaldo';backup.mkdir(exist_ok=True)
    for f in ['nbeats','timesnet']:
        for kind in ['global','detalle']:
            for ext in ['png','svg']:
                path=out/f'{f}_{kind}.{ext}'
                if path.exists() and not (backup/path.name).exists():shutil.copy2(path,backup/path.name)
    benchmark=pd.read_csv(root/'results/evaluacion_e2/benchmark.csv').set_index('clave')
    meta=pd.read_csv(root/'results/evaluacion_e2/metadatos_procedimientos.csv')
    info={f:dict(MASE=float(benchmark.loc[f'nuevo_{f}__N','MASE']),parametros=int(meta.loc[meta.clave.eq(f'nuevo_{f}__N'),'parametros'].iloc[0])) for f in ['nbeats','timesnet']}
    # Este trazado describe las configuraciones guardadas: verificar su contrato antes de dibujar.
    original=root/'results/evaluacion_e2/entradas/adicionales'
    ncfg=json.loads((original/'nuevo_nbeats/busqueda/configuracion_elegida.json').read_text(encoding='utf8'))['N']['configuracion']
    tcfg=json.loads((original/'nuevo_timesnet/busqueda/configuracion_elegida.json').read_text(encoding='utf8'))['N']['configuracion']
    assert (ncfg['pilas'],ncfg['capas_fc'],ncfg['ancho'],ncfg['dropout'])==(30,4,512,.1)
    assert (tcfg['e_layers'],tcfg['top_k'],tcfg['num_kernels'],tcfg['d_model'],tcfg['d_ff'])==(2,5,6,16,16)
    assert info['nbeats']['parametros']==30*((1488+1)*512+3*(512+1)*512+(512+1)*192)
    LAYOUT.clear()
    nbeats_global(root,info['nbeats']);nbeats_detalle(root,info['nbeats'])
    timesnet_global(root,info['timesnet']);timesnet_detalle(root,info['timesnet'])
    for f in info:
        shutil.copy2(out/f'{f}_global.svg',root/f'book/_static/arquitecturas/{f}.svg')
        for kind in ['global','detalle']:shutil.copy2(out/f'{f}_{kind}.png',root/f'dashboard/site/architectures/{f}_{kind}.png')
    anadir_fuentes(root)
    images={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for f in info for p in out.glob(f'{f}_*') if p.suffix in ['.png','.svg']}
    (audit/'manifest.json').write_text(json.dumps(dict(modelos=info,configuraciones=dict(nbeats=ncfg,timesnet=tcfg),formato='16:9 · PNG 3200×1800 y SVG',verificacion=LAYOUT,imagenes_sha256=images,entrenamiento=False,publicacion=False),ensure_ascii=False,indent=2),encoding='utf8')
    print('Cuatro esquemas detallados generados y sincronizados localmente.',flush=True)

if __name__=='__main__':generar()

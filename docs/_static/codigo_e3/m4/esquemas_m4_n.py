"""Esquemas de M4_N con el formato del resto de arquitecturas; no ejecuta redes."""
from pathlib import Path
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle

P={'data':('#A9DDD6','#1CA497'),'layer':('#E2F0D9','#7BA85B'),
   'weight':('#FFE6C2','#FF9817'),'op':('#FBCACA','#F34C54'),
   'act':('#FFD5B3','#F58025'),'cal':('#FFF2C6','#EFB916'),
   'vector':('#E7E8E9','#949494'),'union':('#0C4052','#0C4052'),
   'output':('#EEE5F5','#B59BD5')}
CARDS=[('#E3EFFB','#A8C8ED'),('#E8F2DE','#AED197'),
       ('#EEE5F5','#B69BD3'),('#FFF8DD','#E9CB5B')]
INK='#25333F';WIRE='#43515F';TEXT=[]
fonts=set(font_manager.get_font_names())
SANS='Calibri' if 'Calibri' in fonts else 'DejaVu Sans'
SERIF='Cambria' if 'Cambria' in fonts else 'DejaVu Serif'
plt.rcParams.update({'font.family':SANS,'mathtext.fontset':'stix','svg.fonttype':'none'})

def text(ax,x,y,label,fs=11,bold=False,ha='center',color=INK):
    return ax.text(x,y,label,fontsize=fs,fontweight='bold' if bold else 'normal',
                   ha=ha,va='center',color=color,zorder=5,linespacing=1.28)

def box(ax,x,y,w,h,label,kind='layer',fs=11,bold=False):
    fc,ec=P[kind]
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=8',
                 fc=fc,ec=ec,lw=1.4,zorder=3))
    t=text(ax,x+w/2,y+h/2,label,fs,bold,color='white' if kind=='union' else INK)
    TEXT.append((t,x,y,w,h))

def circle(ax,x,y,label,r=19,kind='op',fs=18):
    fc,ec=P[kind];ax.add_patch(Circle((x,y),r,fc=fc,ec=ec,lw=1.5,zorder=3))
    text(ax,x,y,label,fs)

def pipe(ax,points,dash=False,color=WIRE,arrow=True):
    xs,ys=zip(*points);ax.plot(xs,ys,color=color,lw=1.45,
                 ls=(0,(5,3)) if dash else '-',zorder=2)
    if arrow:ax.add_patch(FancyArrowPatch(points[-2],points[-1],arrowstyle='-|>',
            mutation_scale=11,lw=1.45,color=color,shrinkA=0,shrinkB=0,zorder=2))

def group(ax,x,y,w,h,label,color=0):
    fc,ec=CARDS[color]
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=10',
                 fc=fc,ec=ec,lw=1.2,zorder=0))
    text(ax,x+9,y+15,label,11,bold=True,ha='left')

def frame(ax,x,y,w,h,label):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=15',
                 fc='white',ec=WIRE,lw=1.4,ls=(0,(5,4)),zorder=0))
    text(ax,x+w/2,y+23,label,14,bold=True)

def canvas(kind):
    TEXT.clear();fig,ax=plt.subplots(figsize=(16,9),dpi=200)
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
    ax.set(xlim=(0,1600),ylim=(900,0));ax.axis('off')
    title=r'M4$_{\mathrm{N}}$: '+('arquitectura global' if kind=='global' else 'detalle de la memoria y la compuerta')
    text(ax,58,72,title,30,bold=True,ha='left',color='#0C4052').set_fontfamily(SERIF)
    text(ax,60,146,'Propuesta retrospectiva · 60 993 parámetros activos del corrector',11,ha='left',color='#61717B')
    for i,(kind2,label) in enumerate([('data','Datos'),('layer','Capa'),('weight','Pesos'),
                 ('op','Operación'),('act','Activación'),('cal','Disponibilidad'),('vector','Vector'),('union','Unión')]):
        x=656+i*114;fc,ec=P[kind2]
        if kind2 in ['data','op']:ax.add_patch(Circle((x,146),7,fc=fc,ec=ec,lw=1))
        else:ax.add_patch(Rectangle((x-7,139),14,14,fc=fc,ec=ec,lw=1))
        text(ax,x+12,146,label,8.3,ha='left',color='#61717B')
    return fig,ax

def cards(ax,entries):
    for i,(heading,formula,note) in enumerate(entries):
        x=45+i*385;fc,ec=CARDS[i]
        ax.add_patch(FancyBboxPatch((x,665),360,194,boxstyle='round,pad=0,rounding_size=11',
                    fc=fc,ec=ec,lw=1.3,zorder=0))
        t=text(ax,x+180,697,heading,13,bold=True);TEXT.append((t,x+7,674,346,45))
        t=text(ax,x+180,755,formula,17);TEXT.append((t,x+8,722,344,70))
        t=text(ax,x+180,821,note,10.5);TEXT.append((t,x+8,796,344,50))

def save(fig,ax,out,kind):
    for _ in range(4):
        fig.canvas.draw();renderer=fig.canvas.get_renderer();changed=False
        for t,x,y,w,h in TEXT:
            bb=t.get_window_extent(renderer)
            a=ax.transData.transform((x+6,y+4));b=ax.transData.transform((x+w-6,y+h-4))
            ratio=min(abs(b[0]-a[0])/max(bb.width,1),abs(b[1]-a[1])/max(bb.height,1))
            if ratio<.98:t.set_fontsize(max(8.5,t.get_fontsize()*ratio*.98));changed=True
        if not changed:break
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    for t,x,y,w,h in TEXT:
        bb=t.get_window_extent(renderer);a=ax.transData.transform((x,y));b=ax.transData.transform((x+w,y+h))
        assert bb.width<=abs(b[0]-a[0])+1 and bb.height<=abs(b[1]-a[1])+1,t.get_text()
    for ext in ['png','svg']:fig.savefig(Path(out)/f'm4_n_{kind}.{ext}',dpi=200,facecolor='white')
    plt.close(fig)

def global_diagram(out):
    fig,ax=canvas('global')
    frame(ax,45,193,1510,447,'Base congelada + recuperación causal de residuos + corrección aprendida')
    group(ax,66,239,570,128,'Base M1: expertos y combinador congelados',0)
    for x,label in [(80,'PatchTST'),(213,'E: demanda\n+ exógenas'),(346,'Periódico\nConv2D 7 × 24')]:
        box(ax,x,278,115,35,label,'layer',10.5)
        pipe(ax,[(x+57,313),(x+57,330),(510,330)],arrow=False)
    box(ax,499,275,120,65,'Combinación\nconvexa\npor hora','union',10.5)
    pipe(ax,[(510,330),(499,330)])
    pipe(ax,[(619,305),(660,305),(660,248),(1541,248),(1541,403),(1446,403),(1446,433)],dash=True)
    text(ax,1040,236,r'$b_t\in\mathbb{R}^{24}$'+' · trayectoria base, sin ajuste de pesos',10.5)

    group(ax,870,253,665,118,'Compuerta: consulta + evidencia del respaldo',2)
    box(ax,884,291,132,52,'Concat\n32 + 4 = 36','union',10.5)
    box(ax,1032,291,121,52,'Linear\n36 → 64','weight',10.5)
    box(ax,1169,291,108,52,'GELU\nDropout 0,1','act',10.5)
    box(ax,1293,291,112,52,'Linear\n64 → 1','weight',10.5)
    box(ax,1421,291,98,52,'Sigmoid\n0 ≤ g ≤ 1','act',10.5)
    for a,b in [(1016,1032),(1153,1169),(1277,1293),(1405,1421)]:pipe(ax,[(a,317),(b,317)])
    pipe(ax,[(1470,343),(1470,394),(1373,394),(1373,433)])
    text(ax,1411,382,r'$g_t$'+' · escalar',10)

    box(ax,67,411,165,85,'Contexto numérico\n912 atributos\nEscalar con ajuste','vector',11)
    box(ax,258,424,86,60,r'$W_Q,c_Q$'+'\n912 → 32','weight',11)
    box(ax,369,424,94,60,'LayerNorm\n32','layer',11)
    box(ax,507,424,135,60,'Producto interno\n÷ √32','op',10.5)
    box(ax,674,424,138,60,'Máscara causal\n2 ≤ edad ≤ 365','cal',10.5)
    box(ax,844,424,90,60,'Top-k\nk = 16','op',11)
    box(ax,961,424,140,60,'Softmax válido\n+ renormalizar','act',10.5)
    box(ax,1131,424,139,60,'Mezcla residual\n16 × 24 → 24','layer',10.5)
    circle(ax,1373,454,'×');circle(ax,1446,454,'+')
    circle(ax,1521,454,r'$\widehat y_t$',24,'output',20)
    for a,b in [(232,258),(344,369),(463,507),(642,674),(812,844),(934,961),(1101,1131),(1270,1354),(1392,1427),(1465,1497)]:
        pipe(ax,[(a,454),(b,454)])
    text(ax,487,439,r'$a_t$',10);text(ax,1116,439,r'$\alpha_t$',10)
    text(ax,1312,438,r'$q\,m_t$',11)
    text(ax,1521,498,'24 horas\nkWh',10)

    group(ax,66,531,748,88,'Banco de episodios · predicciones fuera de muestra / OOF anidado',1)
    box(ax,80,565,120,39,r'$z_i$'+' · contexto','vector',10.5)
    box(ax,219,565,130,39,r'$W_K,c_K$'+' → LN','weight',10.5)
    box(ax,371,565,113,39,r'$k_i\in\mathbb{R}^{32}$','vector',12)
    box(ax,505,565,124,39,'Fecha del\nepisodio i','cal',10)
    box(ax,651,565,148,39,r'$r_i=(y_i-b_i)/q$','vector',11)
    for a,b in [(200,219),(349,371)]:pipe(ax,[(a,584),(b,584)])
    pipe(ax,[(428,565),(428,517),(575,517),(575,484)])
    pipe(ax,[(567,565),(567,524),(743,524),(743,484)])
    pipe(ax,[(725,604),(725,628),(1200,628),(1200,484)])
    text(ax,980,616,'Residuos recuperados solo de vecinos válidos',10)
    pipe(ax,[(416,424),(416,387),(849,387),(849,317),(884,317)])
    pipe(ax,[(1200,424),(1200,388),(950,388),(950,343)])
    text(ax,682,377,'Consulta de 32 + semejanza, dispersión, respaldo y edad',9.6)
    text(ax,1390,551,'Sin vecinos válidos:',10.5,bold=True)
    text(ax,1390,577,r'$g_t=0\quad\Longrightarrow\quad\widehat y_t=b_t$',15)
    cards(ax,[
      ('Entrada y representación',r'$z_t=(u_t-\mu)/s$','912 atributos numéricos\nConsulta y claves de dimensión 32'),
      ('Memoria observable',r'$r_i=(y_i-b_i)/q$','Residuos OOF de 24 horas\nRetraso 2 días; edad máxima 365'),
      ('Atención entre episodios',r'$m_t=\sum_i\alpha_{ti}r_i$','Hasta 16 vecinos\nPeso nulo para posiciones inválidas'),
      ('Pronóstico corregido',r'$\widehat y_t=b_t+qg_tm_t$','La compuerta es escalar\nLa corrección cambia en cada hora')])
    save(fig,ax,out,'global')

def detail_diagram(out):
    fig,ax=canvas('detalle')
    frame(ax,45,193,748,446,'Proyecciones, disponibilidad y atención')
    frame(ax,815,193,740,446,'Trayectoria residual y compuerta aprendida')
    for y,who in [(267,'t'),(348,'i')]:
        box(ax,66,y,94,49,f'$z_{who}$\n912','vector',11)
        box(ax,186,y,115,49,('$W_Q,c_Q$' if who=='t' else '$W_K,c_K$')+'\n912 → 32','weight',11)
        box(ax,325,y,111,49,'LayerNorm\n32','layer',11)
        box(ax,462,y,70,49,('$a_t$' if who=='t' else '$k_i$'),'vector',16)
        for a,b in [(160,186),(301,325),(436,462)]:pipe(ax,[(a,y+24),(b,y+24)])
    circle(ax,580,332,'·',19)
    pipe(ax,[(532,291),(580,291),(580,313)])
    pipe(ax,[(532,372),(580,372),(580,351)])
    box(ax,628,304,142,56,'Dividir por √32\nPuntaje s(t,i)','op',10.5)
    pipe(ax,[(599,332),(628,332)])
    box(ax,66,458,133,59,'Edad y validez\n2 ≤ t − i ≤ 365','cal',10.5)
    box(ax,230,458,144,59,'Enmascarar\ninválidos: −10⁹','op',10.5)
    box(ax,404,458,104,59,'Top-k\n16 lugares','op',11)
    box(ax,538,458,146,59,'Softmax\n× validez\nRenormalizar','act',10.5)
    circle(ax,743,487,r'$\alpha_t$',23,'data',18)
    for a,b in [(199,230),(374,404),(508,538),(684,720)]:pipe(ax,[(a,487),(b,487)])
    pipe(ax,[(699,360),(699,423),(302,423),(302,458)])
    text(ax,411,568,'La máscara precede a top-k; las posiciones vacías',11)
    text(ax,411,595,'leen un residuo centinela cero, no un objetivo futuro.',11)

    box(ax,839,265,125,57,'Residuos vecinos\nhasta 16 × 24','vector',10.5)
    circle(ax,1000,294,'Σ',20)
    pipe(ax,[(964,294),(980,294)])
    pipe(ax,[(766,487),(807,487),(807,239),(1000,239),(1000,274)])
    text(ax,893,231,r'$\alpha_t$'+' · pesos de episodios',9.7)
    box(ax,1035,266,106,56,r'$m_t$'+'\n24 valores','vector',11)
    pipe(ax,[(1020,294),(1035,294)])
    box(ax,1172,266,126,56,'Dispersión d(t)\nentre residuos','op',10.5)
    pipe(ax,[(1141,294),(1172,294)])
    box(ax,1322,266,209,56,'Semejanza máxima\nRespaldo: n/16 · Edad: /365','cal',10.5)
    text(ax,1187,349,'Consulta 32 + cuatro indicadores de evidencia',10.8)
    box(ax,839,389,120,57,'Concat\n36 valores','union',11)
    box(ax,984,389,97,57,'Linear\n36 → 64','weight',10.5)
    box(ax,1106,389,87,57,'GELU','act',11)
    box(ax,1218,389,99,57,'Dropout\np = 0,1','act',10.5)
    box(ax,1342,389,93,57,'Linear\n64 → 1','weight',10.5)
    box(ax,1460,389,72,57,'Sigmoid','act',10.5)
    for a,b in [(959,984),(1081,1106),(1193,1218),(1317,1342),(1435,1460)]:pipe(ax,[(a,417),(b,417)])
    pipe(ax,[(1235,322),(1235,367),(899,367),(899,389)])
    pipe(ax,[(1426,322),(1426,367),(899,367)],arrow=False)
    pipe(ax,[(497,267),(497,242),(798,242),(798,367),(899,367)],dash=True,arrow=False)
    pipe(ax,[(456,517),(456,538),(782,538),(782,252),(1426,252),(1426,266)],dash=True)
    text(ax,1158,468,'En evaluación, dropout se desactiva',10)

    box(ax,839,540,121,55,r'$q\,m_t$'+'\n24 valores','vector',12)
    pipe(ax,[(1088,322),(1088,337),(824,337),(824,567),(839,567)],dash=True)
    circle(ax,1040,567,'×',20)
    pipe(ax,[(960,567),(1020,567)])
    pipe(ax,[(1496,446),(1496,503),(1040,503),(1040,547)])
    text(ax,1272,492,r'$g_t\in[0,1]$'+' · una intensidad por origen',10.5)
    circle(ax,1255,567,'+',20)
    box(ax,1190,512,130,29,r'$b_t$'+' · base M1','vector',11)
    pipe(ax,[(1255,541),(1255,547)])
    pipe(ax,[(1060,567),(1235,567)])
    circle(ax,1481,567,r'$\widehat y_t$',27,'output',20)
    pipe(ax,[(1275,567),(1454,567)])
    text(ax,1370,602,'24 pronósticos en kWh',10.5)
    cards(ax,[
      ('Evidencia de los vecinos',r'$c_t=[a_t;s_{\max};d_t;n/16;\bar a_t/365]$',
       '32 + 4 = 36 valores\nDispersión y antigüedad ponderadas'),
      ('Compuerta de dos capas',r'$g_t=\sigma(W_2D_{.1}(\mathrm{GELU}(W_1c_t+b_1))+b_2)$',
       'Inicialización: W₂ = 0; b₂ = −2\ng inicial ≈ 0,119 si hay vecinos'),
      ('Aprendizaje temporal',r'$\mathcal{L}=\mathrm{MAE}(y,\widehat y)/q$',
       'AdamW: lr 0,001 · lote 64\nParada interna y restauración del mínimo'),
      ('Capacidad numérica activa',r'$P=64n+2625=60993$',
       'n = 912; se excluye texto inactivo\nM1 y el banco se contabilizan aparte')])
    save(fig,ax,out,'detalle')

def experto_e_diagram(out):
    fig,ax=canvas('experto_e')
    title=ax.texts[0];title.set_text('Experto E: demanda y corrección exógena acotada')
    ax.texts[1].set_text('Configuración seleccionada · 242 048 parámetros · horizonte de 24 horas')
    frame(ax,45,193,1510,448,'Una base de demanda y calendario recibe una contribución exógena con amplitud limitada')
    group(ax,65,240,938,169,'Rama de demanda y calendario',0)
    box(ax,80,289,120,72,'Demanda\n168 horas','vector',11)
    box(ax,224,289,120,72,'Parches 16 h\nPaso 8 h\n21 parches','layer',10.5)
    box(ax,368,289,112,72,'Proyección\n16 → 32\n+ posición','weight',10.5)
    box(ax,504,278,255,95,'3 bloques Transformer\nAtención: 4 cabezas\nBatchNorm + FF 32→64→32\n+ conexiones residuales','layer',10.5)
    box(ax,783,278,204,95,'Lecturas: demanda 672→24\nContexto 745→128→24\nCalendario pasado/futuro\n+ nivel normalizado 1','union',10.2)
    for a,b in [(200,224),(344,368),(480,504),(759,783)]:pipe(ax,[(a,325),(b,325)])
    circle(ax,1180,325,'+',21)
    pipe(ax,[(987,325),(1159,325)])
    text(ax,1075,305,r'$o_{E,0}\in\mathbb{R}^{24}$',11)
    circle(ax,1375,325,'×',21)
    circle(ax,1495,325,r'$p^{(E)}_t$',29,'output',20)
    pipe(ax,[(1201,325),(1354,325)]);pipe(ax,[(1396,325),(1466,325)])
    box(ax,1307,247,136,42,'Nivel histórico ℓ(t)\nMedia de 168 h','vector',10.5)
    pipe(ax,[(1375,289),(1375,304)])
    text(ax,1495,378,'24 horas\nkWh',10.5)

    group(ax,65,431,938,193,'Rama exógena: valor propuesto y compuerta por hora',1)
    box(ax,80,484,135,91,'29 covariables\n+ 3 máscaras\nSolo historia','vector',10.8)
    box(ax,239,484,185,91,'Últimas 24 h\n+ 7 medias diarias\n+ calendario 24×23\nVector u(E): 1544','union',10.5)
    pipe(ax,[(215,529),(239,529)])
    box(ax,477,469,190,55,'MLP de valor v(E)\n1544 → 32 → 24','layer',10.5)
    box(ax,477,549,190,55,'MLP de compuerta g(E)\n1544 → 32 → 24','layer',10.2)
    box(ax,698,469,92,55,'tanh\n[−1,1]','act',11)
    box(ax,698,549,92,55,'Sigmoid\n[0,1]','act',11)
    pipe(ax,[(424,529),(449,529),(449,496),(477,496)])
    pipe(ax,[(449,529),(449,576),(477,576)])
    pipe(ax,[(667,496),(698,496)]);pipe(ax,[(667,576),(698,576)])
    circle(ax,842,534,r'$\odot$',20)
    pipe(ax,[(790,496),(842,496),(842,514)])
    pipe(ax,[(790,576),(842,576),(842,554)])
    box(ax,891,507,95,55,'Amplitud ρ\n0,02','weight',11)
    pipe(ax,[(862,534),(891,534)])

    circle(ax,1180,534,'×',21)
    pipe(ax,[(986,534),(1159,534)])
    box(ax,1051,438,258,61,'Disponibilidad η(t)\n1 − media de máscaras\nÚltimas 24 horas','cal',10.5)
    pipe(ax,[(1180,499),(1180,513)])
    pipe(ax,[(1201,534),(1320,534),(1320,410),(1180,410),(1180,346)])
    text(ax,1247,395,r'$\eta_t\,\delta_{E,t}$',12)
    text(ax,1071,578,'MLP: GELU y dropout 0,1',10.3)
    text(ax,1320,604,'Sin aporte exógeno si η(t) = 0',10.3)
    cards(ax,[
      ('Resumen exógeno',r'$u_E\in\mathbb{R}^{1544}$',
       '32×24 + 32×7 + 24×23\nCovariables, máscaras y calendario'),
      ('Corrección por hora',r'$\delta_E=0.02\,\sigma(g_E(u_E))\odot\tanh(v_E(u_E))$',
       'Dos MLP con ancho 32 y salida 24\nLa rama de valor inicia en cero'),
      ('Disponibilidad y cota',r'$|\eta_t\delta_{E,t,h}|\leq0.02\,\eta_t$',
       'Se acota la corrección normalizada\nLa cota no es un límite del error'),
      ('Pronóstico del experto',r'$p^{(E)}_{t,h}=\ell_t(o_{E,0,t,h}+\eta_t\delta_{E,t,h})$',
       'El experto E aporta 24 valores a M1\nM4$_{N}$ corrige después la combinación')])
    save(fig,ax,out,'experto_e')


def experto_periodico_diagram(out):
    fig,ax=canvas('experto_periodico')
    ax.texts[0].set_text('Experto periódico: convoluciones sobre la semana 7 × 24')
    ax.texts[1].set_text('Configuración seleccionada · 33 601 parámetros · horizonte de 24 horas')
    frame(ax,45,193,1510,447,'Organizar días y horas, aprender patrones locales y leer una predicción por hora')
    box(ax,67,267,177,74,'Historia: 168 × 33\n30 variables + 3 máscaras\nCalendario: 168 × 23','vector',10.3)
    box(ax,269,267,113,74,'Concatenar\n168 × 56','union',11)
    box(ax,407,267,151,74,'Reorganizar\n56 × 7 × 24\ndías × horas','op',11)
    box(ax,583,267,157,74,'Conv2D 1 × 1\n56 → 32 canales','weight',11)
    box(ax,765,267,132,74,'Mapas\n32 × 7 × 24','vector',11)
    box(ax,922,267,164,74,'Bloque residual 1\nancho 32','layer',11)
    box(ax,1111,267,164,74,'Bloque residual 2\nancho 32','layer',11)
    box(ax,1300,267,178,74,r'$Z_B$'+'\n32 × 7 × 24','vector',13)
    for a,b in [(244,269),(382,407),(558,583),(740,765),(897,922),(1086,1111),(1275,1300)]:
        pipe(ax,[(a,304),(b,304)])

    group(ax,65,371,1035,135,'Detalle del bloque residual · dos repeticiones con parámetros propios',1)
    circle(ax,99,432,r'$Z_b$',20,'data',15)
    box(ax,139,407,131,51,'Conv2D 3 × 3\n32 → 32','weight',10.5)
    box(ax,295,407,85,51,'GELU','act',11)
    box(ax,405,407,105,51,'Dropout\np = 0,1','act',10.5)
    box(ax,535,407,131,51,'Conv2D 1 × 5\n32 → 32','weight',10.5)
    circle(ax,725,432,'+',19)
    box(ax,778,407,154,51,'GroupNorm\n1 grupo, 32 canales','layer',10.5)
    circle(ax,1037,432,r'$Z_{b+1}$',24,'output',16)
    for a,b in [(119,139),(270,295),(380,405),(510,535),(666,706),(744,778),(932,1013)]:
        pipe(ax,[(a,432),(b,432)])
    pipe(ax,[(99,452),(99,484),(725,484),(725,451)],dash=True)
    text(ax,440,476,'Conexión residual',9.4)

    group(ax,1123,371,390,135,'Lectura para cada hora del horizonte',2)
    box(ax,1140,408,356,72,'Último día: 32 componentes\nMedia de 7 días: 32 componentes\nCalendario futuro: 23 componentes','vector',11)
    pipe(ax,[(1389,341),(1389,408)])
    pipe(ax,[(1318,480),(1318,520),(175,520),(175,541)])
    box(ax,67,541,208,61,'Concatenar por hora\n32 + 32 + 23 = 87','union',11)
    box(ax,315,541,135,61,'Linear\n87 → 32','weight',11)
    box(ax,490,541,95,61,'GELU','act',11)
    box(ax,625,541,120,61,'Dropout\np = 0,1','act',11)
    box(ax,785,541,135,61,'Linear\n32 → 1','weight',11)
    circle(ax,1005,571,'+1',21)
    circle(ax,1215,571,'×',21)
    circle(ax,1455,571,r'$p_t^{(P)}$',27,'output',20)
    for a,b in [(275,315),(450,490),(585,625),(745,785),(920,984),(1026,1194),(1236,1428)]:
        pipe(ax,[(a,571),(b,571)])
    text(ax,1095,552,'Razón respecto del nivel',9.8)
    text(ax,1455,615,'24 horas · kWh',10.5)
    text(ax,1215,620,r'$\ell_t$'+' · nivel de la ventana',10.5)
    pipe(ax,[(1215,608),(1215,592)])
    cards(ax,[
      ('Organización periódica',r'$U\in\mathbb{R}^{56\times7\times24}$',
       'Periodos fijos de 24 y 168 horas\nCalendario e historia anteriores al origen'),
      ('Bloque convolucional',r'$Z_{b+1}=\mathrm{GN}(Z_b+\mathrm{Conv}_{1\times5}(D(\mathrm{GELU}(\mathrm{Conv}_{3\times3}(Z_b)))))$',
       'Filtros entre días y horas\nPadding conserva la forma 7 × 24'),
      ('Lectura horaria',r'$f_{t,h}=[Z_{B,-1,h};\overline{Z}_{B,h};c_{t,h}]$',
       '87 atributos por hora\nLa misma MLP se aplica a las 24 horas'),
      ('Pronóstico del experto',r'$p_{t,h}^{(P)}=\ell_t(1+\mathrm{MLP}(f_{t,h}))$',
       'Última capa inicializada en cero\nSu trayectoria se combina dentro de M1')])
    save(fig,ax,out,'experto_periodico')


def main(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    global_diagram(out);detail_diagram(out);experto_e_diagram(out);experto_periodico_diagram(out)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--salida',type=Path,required=True)
    main(parser.parse_args().salida)

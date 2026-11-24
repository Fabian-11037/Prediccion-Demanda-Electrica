from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,Circle,FancyArrowPatch
STATIC=Path(__file__).resolve().parent.parent/"arquitecturas"

def diagrama(m):
    fig,ax=plt.subplots(figsize=(16,6.1));ax.set(xlim=(0,16),ylim=(0,6.1));ax.axis('off')
    ax.text(8,5.8,m['title'],ha='center',va='center',fontsize=17,fontweight='bold',color='#253340')
    cols=['#e5f1fb','#e8f3df','#e8f3df','#eee5f6'];edges=['#7da9ce','#7fa35e','#7fa35e','#a58bc3']
    def box(x,y,w,h,txt,c,edge,fs=11):
        p=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.10',fc=c,ec=edge,lw=1.5);ax.add_patch(p)
        ax.text(x+w/2,y+h/2,txt,ha='center',va='center',fontsize=fs,color='#253340',linespacing=1.5)
    for i,txt in enumerate(m['blocks']):
        x=1.25+i*3.45;box(x,3.55,2.75,1.13,txt,cols[i],edges[i])
        if i<3:ax.add_patch(FancyArrowPatch((x+2.85,4.12),(x+3.35,4.12),arrowstyle='-|>',mutation_scale=16,color='#405363',lw=1.6))
    for x,label,col in [(.48,r'$X$','#9ed8d0'),(15.55,r'$\hat y$','#d8c7e9')]:
        ax.add_patch(Circle((x,4.12),.38,facecolor=col,edgecolor='#687985',lw=1.4));ax.text(x,4.12,label,ha='center',va='center',fontsize=17)
    ax.add_patch(FancyArrowPatch((.88,4.12),(1.15,4.12),arrowstyle='-|>',mutation_scale=14,color='#405363'))
    ax.add_patch(FancyArrowPatch((14.45,4.12),(15.1,4.12),arrowstyle='-|>',mutation_scale=14,color='#405363'))
    textual=m['key'] in {'tats','tgforecaster','fusion'}
    txt='Documentos disponibles → E5 congelado (768) → relevancia aprendida + presencia' if textual else 'Calendario de las 24 horas del horizonte: conocido en el origen'
    box(1.3,2.25,12.9,.68,txt,'#fff5d6','#d3ae45',11)
    if textual:ax.add_patch(FancyArrowPatch((8.0,3.03),(8.0,3.47),arrowstyle='-|>',mutation_scale=16,color='#a37b19',lw=1.5))
    for i,formula in enumerate(m['cards']):
        x=.25+i*4.0;box(x,.3,3.55,1.32,'',cols[i],edges[i])
        ax.text(x+.13,1.35,f'{i+1}. '+['Entrada','Representación','Composición','Pronóstico'][i],fontsize=10.5,fontweight='bold',color='#253340')
        ax.text(x+1.77,.89,formula,fontsize=10 if len(formula)<85 else 8.5,ha='center',va='center')
    fig.savefig(STATIC/(m['key']+'.svg'),bbox_inches='tight');fig.savefig(STATIC/(m['key']+'.png'),dpi=140,bbox_inches='tight');plt.close(fig)

if __name__=="__main__":
    diagrama({'key': 'tft', 'title': 'TFT: selección de variables y atención interpretable', 'blocks': ['56 variables pasadas\n23 futuras conocidas', 'Selección GRN + softmax\nLSTM encoder / decoder', 'Compuertas + atención\ncausal, 4 cabezas', 'GRN por posición\n24 salidas escalares'], 'cards': ['$v_s=\\mathrm{softmax}(\\mathrm{GRN}([e_{s,j}]))$', '$z_s=\\sum_j v_{s,j}\\mathrm{GRN}_j(e_{s,j})$', '$A=\\mathrm{softmax}(QK^T/\\sqrt{d_k}+M)$', '$\\hat y_h=\\ell_t[1+w^T\\psi_h+b]$']})

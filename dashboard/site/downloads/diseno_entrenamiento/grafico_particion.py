"""Diagrama temporal de selección, ajuste final y prueba, sin ejecutar modelos."""
from datetime import datetime,timedelta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch


def grafico_particiones(protocolo):
 colores={'ajuste':'#3b789b','parada':'#c58a36','validacion':'#238f83','prueba':'#b65d50','reserva':'#e8edef'}
 fecha=lambda x:datetime.fromisoformat(x[:10])
 fig,axes=plt.subplots(2,1,figsize=(16,10.7),sharex=True,gridspec_kw={'height_ratios':[5,6]})
 fig.subplots_adjust(left=.26,right=.98,top=.86,bottom=.1,hspace=.32)
 corte=fecha(protocolo['prueba'][0]);fin=fecha(protocolo['prueba'][1])+timedelta(days=1)
 for ax,grupo,titulo in zip(axes,['general','m4'],['Procedimiento general: validación y ajuste final','M4: selección de memoria y ajustes finales']):
  filas=protocolo[grupo]
  for y,row in enumerate(filas):
   for nombre in ['ajuste','parada','evaluacion']:
    a,b=map(fecha,row[nombre]);b+=timedelta(days=1)
    tipo=('validacion'if row['tipo']=='validacion'else'prueba')if nombre=='evaluacion'else nombre
    ax.barh(y,mdates.date2num(b)-mdates.date2num(a),left=mdates.date2num(a),height=.59,color=colores[tipo],edgecolor='white',linewidth=1,zorder=3)
   if row['tipo']=='validacion':
    ax.barh(y,mdates.date2num(fin)-mdates.date2num(corte),left=mdates.date2num(corte),height=.59,color=colores['reserva'],edgecolor='white',hatch='///',zorder=3)
   else:
    ax.text((mdates.date2num(corte)+mdates.date2num(fin))/2,y,'2022–2025',ha='center',va='center',fontsize=10,color='white',weight='bold',zorder=4)
  ax.axvline(mdates.date2num(corte),color='#65767d',ls='--',lw=1.3,zorder=4)
  ax.set_yticks(range(len(filas)),[row['etapa']for row in filas],fontsize=11)
  ax.set_ylim(len(filas)-.4,-.7);ax.set_title(titulo,loc='left',fontsize=14,color='#24383e',pad=14,weight='bold')
  ax.grid(axis='x',color='#dde5e9',lw=.7,zorder=0);ax.tick_params(axis='both',length=0,pad=8,labelsize=10)
  for spine in ax.spines.values():spine.set_visible(False)
 axes[-1].set_xlim(datetime(2000,1,1),fin)
 axes[-1].set_xticks([datetime(y,1,1)for y in [2000,2005,2010,2013,2015,2018,2020,2022,2026]])
 axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'));axes[-1].set_xlabel('Año · la frontera de 2026 es exclusiva',fontsize=11,labelpad=12)
 fig.legend(handles=[Patch(facecolor=colores[k],label=t)for k,t in [('ajuste','Optimización'),('parada','Parada interna'),('validacion','Validación externa'),('prueba','Prueba'),('reserva','Reservado durante selección')]],loc='upper center',bbox_to_anchor=(.6,.98),ncol=3,frameon=False,fontsize=11)
 fig.text(.26,.03,'Periodos de referencia del esquema general: registros de TFT. En M4 se muestran la memoria y su base M1 por separado.',fontsize=10,color='#526973')
 return fig

if __name__ == '__main__':
 import json
 from pathlib import Path
 protocolo=json.loads(Path(__file__).with_name('protocolo_temporal.json').read_text(encoding='utf8'))
 figura=grafico_particiones(protocolo)
 figura.savefig('particion_temporal.svg',bbox_inches='tight',metadata={'Date':None})
 figura.savefig('particion_temporal.png',dpi=150,bbox_inches='tight')
 plt.close(figura)

"""Esquema de partición temporal desde las fechas y reglas conservadas."""
from pathlib import Path
from datetime import datetime,timedelta
import shutil
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch


def preparar(root,methodology,models):
 root=Path(root);dest=root/'dashboard/site/figures/metodologia';dest.mkdir(parents=True,exist_ok=True)
 colors={'ajuste':'#3b789b','parada':'#c58a36','validacion':'#238f83','prueba':'#b65d50','reserva':'#e8edef'}
 plt.rcParams['svg.hashsalt']='particion-sin-20261127'
 fig,ax=plt.subplots(figsize=(16,7.5));fig.subplots_adjust(left=.26,right=.98,top=.79,bottom=.15)
 def date(value):return datetime.fromisoformat(value[:10])
 def bar(y,start,end,color,hatch=None):
  ax.barh(y,mdates.date2num(end)-mdates.date2num(start),left=mdates.date2num(start),height=.56,color=color,edgecolor='white',linewidth=1,hatch=hatch,zorder=3)
 general=next(m['final']for m in models if m['family']=='tft')
 start=date(general['train_period'].split(' a ')[0]);cut=date('2022-01-01');finish=date('2026-01-01')
 labels=[]
 for row,year in enumerate([2018,2019,2020,2021]):
  boundary=datetime(year,1,1);dates=pd.date_range(start,boundary-timedelta(days=1),freq='D')
  n_stop=max(1,round(.1*len(dates)));stop=dates[-n_stop].to_pydatetime()
  bar(row,start,stop,colors['ajuste']);bar(row,stop,boundary,colors['parada']);bar(row,boundary,datetime(year+1,1,1),colors['validacion'])
  bar(row,cut,finish,colors['reserva'],hatch='///');labels.append(f'Pliegue externo {year}')
 ax.axhline(3.8,color='#ccd7dc',lw=1)
 stop=date(general['stop_period'].split(' a ')[0]);bar(5,start,stop,colors['ajuste']);bar(5,stop,cut,colors['parada']);bar(5,cut,finish,colors['prueba'])
 m4=next(m['final']for m in models if m['id']=='m4_n')
 bar(6,date(m4['train_period'].split(' a ')[0]),date(m4['stop_period'].split(' a ')[0]),colors['ajuste'])
 bar(6,date(m4['stop_period'].split(' a ')[0]),cut,colors['parada']);bar(6,cut,finish,colors['prueba'])
 ax.text(mdates.date2num(datetime(2001,1,1)),5,'Ajuste con configuración seleccionada',va='center',color='white',fontsize=11,zorder=5)
 ax.text(mdates.date2num(datetime(2024,1,1)),5,'2022–2025',ha='center',va='center',color='#fff',weight='bold',fontsize=11,zorder=5)
 ax.axvline(mdates.date2num(cut),color='#65767d',ls='--',lw=1.5,zorder=4)
 ax.set_yticks([0,1,2,3,5,6],labels+['Ajuste final · base y ampliados','Ajuste final · memoria M4ₙ'],fontsize=11)
 ax.set_xlim(datetime(2000,1,1),datetime(2026,1,1));ax.set_ylim(6.65,-.8)
 ax.set_xticks([datetime(y,1,1)for y in [2000,2005,2010,2015,2018,2020,2022,2026]])
 ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'));ax.tick_params(axis='both',length=0,pad=10,labelsize=11)
 ax.grid(axis='x',color='#dde5e9',linewidth=.7,zorder=0)
 for spine in ax.spines.values():spine.set_visible(False)
 fig.suptitle('Validación hacia adelante y prueba cronológica',x=.26,ha='left',y=.97,fontsize=18,weight='bold',color='#24383e')
 ax.legend(handles=[Patch(facecolor=colors[k],label=t)for k,t in [('ajuste','Optimización'),('parada','Parada interna'),('validacion','Validación externa'),('prueba','Prueba'),('reserva','Reservado durante selección')]],ncol=3,loc='lower left',bbox_to_anchor=(0,1.01),frameon=False,fontsize=11,columnspacing=1.4)
 fig.text(.26,.065,'Pliegues de referencia: reserva interna del 10 %. Purga de etiquetas de 24 h, no visible a esta escala.',fontsize=10,color='#526973')
 fig.text(.26,.025,'M4ₙ: memoria ajustada en 2013–2020 y parada en 2021. Las reservas de cada procedimiento se detallan en su diseño.',fontsize=10,color='#526973')
 fig.savefig(dest/'particion_temporal.svg',bbox_inches='tight',metadata={'Date':None});fig.savefig(dest/'particion_temporal.png',dpi=150,bbox_inches='tight');plt.close(fig)
 download=root/'dashboard/site/downloads/diseno_entrenamiento';download.mkdir(parents=True,exist_ok=True)
 shutil.copy2(Path(__file__),download/'grafico_particion.py')
 methodology['partition']={
  'title':'Partición para entrenamiento y prueba',
  'paragraphs':[
   'La separación conserva el orden temporal. Las configuraciones se seleccionan mediante validación hacia adelante en 2018–2021: cada año se evalúa con historia anterior y una reserva de parada interna. Tras fijar la configuración, se ajustan las realizaciones finales con datos anteriores a 2022 y se evalúan los mismos 1.461 orígenes de 2022–2025.',
   'Las familias base y los paquetes ampliados reservan el último 10 % del periodo previo para parada interna. La memoria de M4 utiliza 2013–2020 para ajuste y 2021 para parada. Se aplica purga de etiquetas de 24 horas y las transformaciones se estiman solo en optimización.'],
  'image':'./figures/metodologia/particion_temporal.svg','alt':'Cuatro pliegues anuales de validación expansiva, reserva interna y prueba 2022–2025; ajuste final general y de la memoria M4',
  'caption':'Las franjas verdes representan años externos distintos: la historia de ajuste crece en cada pliegue. La prueba roja corresponde a la configuración ya seleccionada. N-HiTS puntúa su confirmación de 2019–2021.',
  'source':'./downloads/diseno_entrenamiento/grafico_particion.py'}
 return methodology

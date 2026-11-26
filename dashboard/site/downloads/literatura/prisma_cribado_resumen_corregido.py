"""Extrae candidatos comparadores explícitos; no convierte menciones en baselines."""
import argparse,re
from pathlib import Path
import pandas as pd
FAMILIAS={
 'MLP':r'\bmlps?\b|\bffnns?\b|\bfnns?\b|multi[- ]?layer perceptron|feed[- ]?forward neural network',
 'ANN/DNN no especificada':r'\banns?\b|\bdnns?\b|artificial neural network|deep neural network|\bbpnn\b|back[- ]?propagation neural network',
 'ELM':r'\belm\b|extreme learning machine','RBF':r'\brbf\b|radial basis function',
 'RNN':r'\brnns?\b|recurrent neural network|\belman\b|echo state|\besn\b',
 'LSTM':r'lstm|long short[- ]term memory','GRU':r'\bgrus?\b|bigru|gated recurrent unit',
 'CNN':r'(?<![a-z])cnns?\b|convolutional neural network|\bconv1d\b',
 'TCN':r'\btcns?\b|temporal convolutional network','Transformer':r'transformer|informer|autoformer|fedformer|patchtst|\btft\b|temporal fusion',
 'N-BEATS':r'n-?beats','N-HiTS':r'n-?hits','GNN':r'\bgcns?\b|graph (neural|convolutional|attention)|\bgat\b'}
METRICAS=re.compile(r'\bmape\b|\brmse\b|\bmae\b|\bmase\b|\bsmape\b|\bnrmse\b|\bmse\b|r-squared',re.I)
REFERENCIAS=re.compile(r'(?:compared\s+(?:with|to|against)|outperforms?\s+|better\s+than\s+|superior\s+to\s+|versus\s+|\bvs\.?\s+|against\s+)([^.;]+)',re.I)
HIBRIDO=re.compile(r'\b[\w]+(?:[-+/][\w]+)+\b')

def familias_comparadas(texto):
 """Identifica familias en la cláusula que nombra referencias, preservando híbridos."""
 families=set();snippets=[]
 for match in REFERENCIAS.finditer(str(texto)):
  clause=re.split(r'\b(?:whereas|while|our|we|proposed|propose)\b',match[1],maxsplit=1,flags=re.I)[0];snippets.append(clause.strip())
  def hybrid(m):
   name=m[0];found=[f for f,p in FAMILIAS.items()if re.search(p,name.replace('-',' ').replace('/',' ').replace('+',' '),re.I)]
   if len(found)>=2:families.add(name.upper()+' (híbrido)');return ' '
   return name
  clause=HIBRIDO.sub(hybrid,clause)
  families.update(f for f,p in FAMILIAS.items()if re.search(p,clause,re.I))
 return sorted(families),snippets

def decidir(fila):
 text=str(fila.get('resumen',''));families,snippets=familias_comparadas(text)
 if len(text)<200:return 'pendiente_revision','','resumen_insuficiente','', ''
 if len(families)>=2 and METRICAS.search(text):return 'candidato_texto_completo','','referencias_explicitas',';'.join(families),' | '.join(snippets)
 return 'pendiente_revision','','roles_o_comparadores_no_confirmados',';'.join(families),' | '.join(snippets)

def main():
 p=argparse.ArgumentParser();p.add_argument('--entrada',type=Path,required=True);p.add_argument('--salida',type=Path,required=True);a=p.parse_args()
 if a.entrada.resolve()==a.salida.resolve():raise SystemExit('Conservar el registro original; indicar otra salida.')
 if a.salida.exists():raise SystemExit('La salida ya existe; no se sobrescriben decisiones.')
 df=pd.read_csv(a.entrada,dtype=str,encoding='utf-8-sig').fillna('')
 if 'resumen'not in df:raise SystemExit('Se requieren las exportaciones con resúmenes; el archivo público sin resúmenes no permite recalcular el cribado.')
 cols=['ajuste_decision','ajuste_motivo','ajuste_regla','ajuste_familias','ajuste_evidencia'];df[cols]=df.apply(decidir,axis=1,result_type='expand');df.to_csv(a.salida,index=False,encoding='utf-8-sig')
 print(df.ajuste_decision.value_counts().to_string())
if __name__=='__main__':main()

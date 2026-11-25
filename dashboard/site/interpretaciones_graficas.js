// Interpretaciones calculadas con los mismos valores que recibe cada gráfica.
// No reajusta parámetros ni utiliza información adicional a la vista mostrada.
const finite=v=>typeof v==='number'&&Number.isFinite(v);
const avg=a=>a.length?a.reduce((s,v)=>s+v,0)/a.length:NaN;
const quantile=(a,p)=>{const s=a.filter(finite).sort((a,b)=>a-b);if(!s.length)return NaN;const k=(s.length-1)*p,j=Math.floor(k);return s[j]+(s[Math.ceil(k)]-s[j])*(k-j);};
const extrema=a=>{const ix=a.map((v,i)=>[v,i]).filter(([v])=>finite(v));return ix.length?{min:ix.reduce((a,b)=>a[0]<=b[0]?a:b),max:ix.reduce((a,b)=>a[0]>=b[0]?a:b)}:null;};

export function describeChart(id,traces,{num,study,state,modelName}){
 const t=traces[0];if(!t)return '';
 if(id==='hero-trend'||id==='eda-series'){
  const pairs=t.y.map((v,i)=>[v,String(t.x[i])]).filter(([v])=>finite(v));if(!pairs.length)return 'El periodo seleccionado no contiene observaciones disponibles.';
  const years=[...new Set(pairs.map(([,d])=>d.slice(0,4)))],first=years[0],last=years.at(-1);
  const a=avg(pairs.filter(([,d])=>d.startsWith(first)).map(([v])=>v)),b=avg(pairs.filter(([,d])=>d.startsWith(last)).map(([v])=>v));
  const observed=100*pairs.length/t.y.length;
  return first!==last?`La media de los valores diarios pasa de ${num(a,3)} en ${first} a ${num(b,3)} en ${last}${a!==0?` (${num(100*(b/a-1),2)} %)` : ''}, en las unidades del eje. Hay observaciones en el ${num(observed,1)} % de los días mostrados.`:`En ${first}, los valores diarios tienen media ${num(a,3)} y se sitúan entre ${num(Math.min(...pairs.map(([v])=>v)),3)} y ${num(Math.max(...pairs.map(([v])=>v)),3)}, en las unidades del eje.`;
 }
 if(id==='eda-hist'){
  const x=t.x.filter(finite);if(!x.length)return 'No hay observaciones disponibles para describir la distribución.';
  return `La mediana es ${num(quantile(x,.5),3)}; el 80 % central de los valores diarios se encuentra entre ${num(quantile(x,.1),3)} y ${num(quantile(x,.9),3)}, en las unidades del eje. La media es ${num(avg(x),3)}.`;
 }
 if(id==='eda-calendar'){
  const cells=t.z.flatMap((row,d)=>row.map((v,h)=>({v,d,h}))).filter(r=>finite(r.v));if(!cells.length)return '';
  const high=cells.reduce((a,b)=>a.v>=b.v?a:b),low=cells.reduce((a,b)=>a.v<=b.v?a:b);
  return `El mayor promedio corresponde a ${t.y[high.d]} a las ${high.h}:00 (${num(high.v,3)} GWh) y el menor, a ${t.y[low.d]} a las ${low.h}:00 (${num(low.v,3)} GWh). El patrón respalda la inclusión de hora y día de semana en el pronóstico.`;
 }
 if(id==='eda-correlation'){
  const r=t.z[0].map((v,i)=>({v,i})).filter(r=>r.i>0&&finite(r.v));if(!r.length)return '';
  const best=r.reduce((a,b)=>Math.abs(a.v)>=Math.abs(b.v)?a:b);
  return `Entre las covariables mostradas, ${t.x[best.i]} presenta la mayor asociación absoluta con la demanda (Spearman ρ = ${num(best.v,3)}) en el periodo seleccionado. Esta relación marginal no determina su aporte una vez incluidos los rezagos y el calendario.`;
 }
 if(id==='corpus-count'){
  const totals=traces.map(t=>({name:t.name,total:t.y.reduce((a,b)=>a+b,0)})),total=totals.reduce((a,b)=>a+b.total,0);if(!total)return 'El periodo seleccionado no contiene documentos admitidos en este inventario.';
  const lead=totals.reduce((a,b)=>a.total>=b.total?a:b),annual=t.x.map((_,i)=>traces.reduce((s,t)=>s+t.y[i],0)),ex=extrema(annual);
  return `${lead.name} concentra el ${num(100*lead.total/total,1)} % de los ${num(total,0)} documentos del periodo. El mayor volumen anual se registra en ${t.x[ex.max[1]]}, con ${num(ex.max[0],0)} documentos; la composición por fuente debe considerarse al comparar periodos.`;
 }
 if(id==='corpus-window'){
  const ex=extrema(t.y);if(!ex)return '';
  return `El promedio máximo es ${num(ex.max[0],2)} documentos por cierre en ${t.x[ex.max[1]]}; el mínimo, ${num(ex.min[0],2)} en ${t.x[ex.min[1]]}. La cantidad de texto disponible para el modelo cambia entre años.`;
 }
 if(id==='forecast'){
  const observed=traces.find(t=>t.name==='Demanda observada'),pred=traces[3];if(!observed||!pred)return '';
  const ex=extrema(observed.y),bias=avg(observed.y.map((v,i)=>pred.y[i]-v)),peak=pred.y[ex.max[1]];
  return `El máximo observado de la selección es ${num(ex.max[0],3)} GWh; en esa hora, la predicción media es ${num(peak,3)} GWh. En el conjunto de horas mostradas, la predicción media ${bias>=0?'sobreestima':'subestima'} el nivel observado en ${num(Math.abs(bias),4)} GWh. Esta lectura describe la trayectoria media; las métricas superiores promedian errores individuales entre semillas.`;
 }
 if(id==='forecast-error'){
  const ex=extrema(t.y);return ex?`El mayor MAE se presenta en el horizonte ${t.x[ex.max[1]]} h (${num(ex.max[0],4)} GWh) y el menor, en ${t.x[ex.min[1]]} h (${num(ex.min[0],4)} GWh). La dificultad varía dentro de las 24 horas pronosticadas.`:'';
 }
 if(id==='benchmark'||id==='m4-benchmark'){
  const ex=extrema(t.x);if(!ex)return '';
  return `${t.y[ex.min[1]]} obtiene el menor ${id==='m4-benchmark'?'MASE':state.metric} descriptivo (${num(ex.min[0],4)}) entre los procedimientos mostrados. El extremo superior corresponde a ${t.y[ex.max[1]]} (${num(ex.max[0],4)}). Los contrastes pareados determinan el respaldo estadístico de las diferencias.`;
 }
 if(id==='ranks'){
  const ex=extrema(t.x);return ex?`${t.y[ex.min[1]]} presenta el menor rango medio (${num(ex.min[0],2)}) y ${t.y[ex.max[1]]}, el mayor (${num(ex.max[0],2)}). Un orden de rangos no implica que todas las parejas difieran en el post hoc.`:'';
 }
 if(id==='importance'){
  const ex=extrema(t.x);if(!ex)return '';
  const n=t.x.filter(v=>v<0).length;
  return `La mayor sensibilidad corresponde a ${t.y[ex.max[1]]}: trasladar este bloque cambia el MAE un ${num(ex.max[0],2)} %. ${n?`${n===1?'Un bloque muestra':`${n} bloques muestran`} cambios negativos, es decir, menor error bajo esa intervención.`:'Ningún bloque muestra una reducción del error bajo esta intervención.'} Las magnitudes describen dependencia de las entradas con los pesos conservados.`;
 }
 if(id==='learning'||id==='m4-learning'){
  const val=traces.filter(t=>t.name==='Parada interna'||t.name?.startsWith('Validación interna'));
  const chosen=val.map(t=>{const ex=extrema(t.y);return ex?{epoch:t.x[ex.min[1]],loss:ex.min[0]}:null;}).filter(Boolean);if(!chosen.length)return '';
  const epochs=chosen.map(r=>r.epoch),loss=chosen.map(r=>r.loss);
  return chosen.length===1?`En la realización mostrada, la menor pérdida interna es ${num(loss[0],4)} en la época ${num(epochs[0],0)}. El checkpoint se restaura en ese mínimo; las épocas posteriores no mejoran la pérdida de esa reserva.`:`Los mínimos internos se alcanzan entre las épocas ${num(Math.min(...epochs),0)} y ${num(Math.max(...epochs),0)}, con pérdida mediana ${num(quantile(loss,.5),4)}. La variación entre curvas refleja sensibilidad a la inicialización; el mínimo de cada reserva determina el checkpoint.`;
 }
 if(id==='text-sensitivity'){
  const cells=traces.flatMap(t=>t.y.map((v,i)=>({v,model:t.name,condition:t.x[i]}))).filter(r=>finite(r.v));if(!cells.length)return '';
  const max=cells.reduce((a,b)=>Math.abs(a.v)>=Math.abs(b.v)?a:b);
  return `La mayor variación absoluta aparece en ${max.model}, bajo «${max.condition}»: ${num(max.v,2)} % de cambio del MAE. Un valor positivo indica deterioro al modificar el texto; uno negativo indica mejora. La comparación con el control numérico sigue siendo necesaria para atribuir una ventaja al canal documental.`;
 }
 if(id==='m4-gain'){
  const ex=extrema(t.x);return ex?`La ganancia de M4ₙ va de ${num(ex.min[0],2)} % frente a ${t.y[ex.min[1]]} a ${num(ex.max[0],2)} % frente a ${t.y[ex.max[1]]}. La línea del 3 % identifica el margen práctico utilizado en los contrastes.`:'';
 }
 if(id==='m4-horizon'){
  const ref=traces[1],ex=extrema(t.y),n=t.y.filter((v,i)=>v<ref.y[i]).length;
  return `M4ₙ tiene menor MAE que TimesNet en ${n} de los ${t.y.length} horizontes. Su mayor error corresponde a ${t.x[ex.max[1]]} h (${num(ex.max[0],4)} GWh). El promedio global se complementa con este perfil de dificultad horaria.`;
 }
 if(id==='m4-year'){
  const ex=extrema(t.y),n=t.y.filter((v,i)=>v<traces[1].y[i]).length;
  return `M4ₙ presenta menor MASE que TimesNet en ${n} de los ${t.y.length} años. Su mayor MASE corresponde a ${t.x[ex.max[1]]} (${num(ex.max[0],4)}) y el menor, a ${t.x[ex.min[1]]} (${num(ex.min[0],4)}). La escala de entrenamiento es común a los cuatro años.`;
 }
 return '';
}

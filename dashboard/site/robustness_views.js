const kinds=[['ruido','Ruido numérico','Amplitud / DE de entrenamiento'],['interrupcion','Interrupción de datos','Horas finales sin observación'],['atipicos','Impulsos en las entradas','Impulso / DE de entrenamiento'],['perdida_texto','Pérdida de documentos','Probabilidad de pérdida']];

export function renderRobustness(data,{state,num,pvalue,escapeHTML:e,modelSelect,table,chart,plot,line,book}){
 const rows=data.summary,selected=state.model,other=state.robustCompare??'nuevo_timesnet__N';
 const base=rows.find(r=>r.clave===selected&&r.nivel===0),name=base.modelo;
 const clean=rows.filter(r=>r.nivel===0),names=Object.fromEntries(clean.map(r=>[r.clave,r.modelo]));
 const chosen=selected==='m4_n'?other:selected;
 const contrasts=data.comparisons.filter(r=>r.comparador===chosen);
 const lost=contrasts.filter(r=>r.tipo!=='perdida_texto'&&r.mejora_pct<0).length;
 const supported=contrasts.filter(r=>r.supera_3pct).length;
 const reading=contrasts.length?`M4ₙ registra un MASE mayor que ${e(names[chosen])} en ${lost} de las nueve condiciones numéricas. El margen de mejora del 3 % queda respaldado en ${supported} de las doce condiciones después de Holm.`:'';
 const label=Object.fromEntries(kinds.map(([id,text])=>[id,text]));
 const compareSelect=`<select id="robust-compare"><option value="">Sin comparación</option>${clean.map(r=>`<option value="${e(r.clave)}" ${r.clave===other?'selected':''}>${e(r.modelo)}</option>`).join('')}</select>`;
 const panel=kinds.map(([kind,title,xlabel])=>{
  const end=rows.find(r=>r.clave===selected&&r.tipo===kind&&r.nivel===3);
  return `<section><h3>${e(title)}</h3>${chart('robust-'+kind)}<p class="caption">En el nivel alto, ${e(name)} cambia su MASE un ${num(end.degradacion_pct,2)} % respecto a la entrada original; IC 95 %: [${num(end.degradacion_ic_inf,2)}; ${num(end.degradacion_ic_sup,2)}] %.</p></section>`;
 }).join('');
 const html=`<section aria-labelledby="robust-title"><h3 id="robust-title">Curvas de degradación</h3><p>Panel fijo de ${data.protocol.origenes.length} fechas de 2022–2025, dos por mes, y cinco realizaciones completas por modelo. La evaluación conserva pesos y escaladores, y aplica las mismas perturbaciones a cada procedimiento.</p><p>Las entradas originales de este panel presentan ${num(data.quality.celdas_ausentes,0)} ausencias en ${num(data.quality.celdas,0)} celdas. Estos escenarios de estrés hipotéticos no estiman la frecuencia real de fallos; el benchmark con los datos originales es la evaluación principal.</p><div class="control-row"><div class="field model"><label for="robust-compare">Comparar curvas con</label>${compareSelect}</div><div class="metric"><span class="metric-label">MASE · entrada original del panel</span><strong>${num(base.MASE,4)}</strong><small>${e(name)}</small></div></div><p class="caption">Ruido: 0,10/0,25/0,50 DE; atípicos: 1/3/5 DE en nueve horas; interrupciones: 6/24/48 horas; pérdida documental: 25/50/75 %. Las bandas corresponden al bootstrap de meses y semillas al 95 %. El calendario y el archivo histórico de memoria se conservan; M4 recalcula expertos, M1 y consulta.</p><div class="plot-grid">${panel}</div><section class="section"><h3>Degradación relativa de los 18 procedimientos</h3>${chart('robust-all','wide')}<p class="caption">Los valores positivos indican mayor MASE y los negativos, menor MASE frente a las entradas originales de cada procedimiento. La escala de color comprime cambios grandes; el valor exacto se muestra al situar el cursor. Los controles sin texto conservan su salida al retirar documentos.</p></section><section class="section"><h3>Margen del 3 %: M4ₙ frente a ${e(names[chosen]??'los comparadores')}</h3>${contrasts.length?table(['Perturbación','Nivel','Mejora de M4 (%)','IC 95 % (%)','p ajustado · Holm','Mejora >3 % respaldada'],contrasts.map(r=>[e(label[r.tipo]),num(r.nivel,0),num(r.mejora_pct,2),`[${num(r.mejora_ic_inf,2)}; ${num(r.mejora_ic_sup,2)}]`,pvalue(r.p_ajustado),r.supera_3pct?'Sí':'No']),['','number','number','number','number','']):'<p>Selecciona un comparador distinto de M4 para consultar los contrastes.</p>'}<p>${reading}</p><p class="caption">Se ajustan 204 hipótesis de M4 con Holm. Los 216 contrastes de deterioro frente a la entrada original forman otra familia. Las horas no se consideran réplicas independientes. El MAE y el RMSE de la descarga están en kWh por intervalo.</p><div class="priority-links"><a class="download" href="./downloads/robustez_resumen.csv" download>Métricas completas CSV ↓</a><a class="download" href="./downloads/robustez_contrastes_M4.csv" download>Contrastes de M4 CSV ↓</a><a href="${e(book+data.book_section)}" target="_blank" rel="noopener">Método, código y resultados en el JBook ↗</a></div></section></section>`;
 function draw(){
  for(const[kind,title,xlabel]of kinds){
   const traces=[];
   for(const[key,color]of [[selected,'#126c70'],[other,'#af5846']]){
    if(!key||!names[key]||(key===other&&key===selected&&traces.length))continue;
    const r=[rows.find(x=>x.clave===key&&x.nivel===0),...rows.filter(x=>x.clave===key&&x.tipo===kind).sort((a,b)=>a.nivel-b.nivel)];
    const x=r.map(d=>d.severidad);
    traces.push(line(x,r.map(d=>d.MASE_ic_sup),'',color,{showlegend:false,line:{width:0},hoverinfo:'skip'}));
    traces.push(line(x,r.map(d=>d.MASE_ic_inf),'',color,{showlegend:false,line:{width:0},fill:'tonexty',fillcolor:color==='#126c70'?'rgba(18,108,112,.1)':'rgba(175,88,70,.1)',hoverinfo:'skip'}));
    traces.push(line(x,r.map(d=>d.MASE),names[key],color,{mode:'lines+markers',hovertemplate:'Severidad: %{x}<br>MASE: %{y:.4f}<extra>%{fullData.name}</extra>'}));
   }
   plot('robust-'+kind,traces,{xaxis:{title:xlabel},yaxis:{title:'MASE'},hovermode:'closest'});
  }
  const order=[...clean].sort((a,b)=>a.MASE-b.MASE).map(r=>r.clave),conditions=kinds.flatMap(([kind,title])=>[1,2,3].map(level=>({kind,level,label:title+' · '+level})));
  const values=order.map(key=>conditions.map(c=>rows.find(r=>r.clave===key&&r.tipo===c.kind&&r.nivel===c.level).degradacion_pct));
  const transform=v=>Math.sign(v)*Math.log1p(Math.abs(v)/5),z=values.map(row=>row.map(transform));
  const max=Math.max(1,...z.flat().map(Math.abs)),ticks=[-500,-100,-25,-5,0,5,25,100,500].filter(v=>Math.abs(transform(v))<=max);
  plot('robust-all',[{z,customdata:values,x:conditions.map(c=>c.label),y:order.map(k=>names[k]),type:'heatmap',zmin:-max,zmax:max,zmid:0,colorscale:'RdBu',reversescale:true,colorbar:{title:{text:'Cambio (%)'},tickvals:ticks.map(transform),ticktext:ticks.map(v=>String(v)),thickness:12},hovertemplate:'%{y}<br>%{x}<br>Cambio del MASE: %{customdata:.2f} %<extra></extra>'}],{height:680,xaxis:{tickangle:-40},yaxis:{autorange:'reversed'},margin:{t:15,b:120,l:165,r:85}});
 }
 return {html,draw};
}

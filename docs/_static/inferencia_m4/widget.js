'use strict';
const $=id=>document.getElementById(id),number=(v,n=3)=>Number(v).toLocaleString('es-CO',{maximumFractionDigits:n,minimumFractionDigits:n});
let manifest,worker,last,custom=null;
function status(text,error=false){$('status').textContent=text;$('status').classList.toggle('error',error);}
function busy(value){for(const id of ['model','origin','change','history','reset','run','coverage','interval-scope'])$(id).disabled=value;}
function invalidate(){last=null;$('output').hidden=true;status('Entradas modificadas. Genera el pronóstico para calcular la nueva trayectoria.');}
function csvRows(text){
 const rows=[];let row=[],cell='',quoted=false;
 for(let i=0;i<text.length;i++){
  const c=text[i];if(c==='"'){if(quoted&&text[i+1]==='"'){cell+='"';i++;}else quoted=!quoted;}
  else if(c===','&&!quoted){row.push(cell);cell='';}
  else if((c==='\n'||c==='\r')&&!quoted){if(c==='\r'&&text[i+1]==='\n')i++;row.push(cell);if(row.some(x=>x.trim()))rows.push(row);row=[];cell='';}
  else cell+=c;
 }
 if(quoted)throw Error('El CSV contiene comillas sin cerrar.');
 if(cell||row.length){row.push(cell);rows.push(row);}return rows;
}
async function readHistory(file,index){
 if(file.size>500000)throw Error('El CSV debe ocupar menos de 500 kB.');
 const rows=csvRows((await file.text()).replace(/^\uFEFF/,''));
 if(!rows.length)throw Error('El CSV está vacío.');
 const header=rows.shift().map(v=>v.trim()),expected=['fecha',...manifest.variables];
 if(header.length!==31||new Set(header).size!==31||expected.some(v=>!header.includes(v)))throw Error('El CSV requiere fecha y las 30 columnas de la plantilla.');
 if(rows.length!==168)throw Error('Se requieren exactamente 168 horas de historia.');
 const origin=Date.parse(manifest.presets[index].origin+'T00:00:00-05:00'),window=new Float64Array(168*30);
 rows.forEach((row,t)=>{
  if(row.length!==header.length)throw Error('Número de columnas incorrecto en la fila '+(t+2));
  let date=row[header.indexOf('fecha')].trim().replace(' ','T');
  if(!/(Z|[+-]\d{2}:?\d{2})$/.test(date))date+='-05:00';
  if(Date.parse(date)!==origin-(168-t)*3600000)throw Error('Las fechas deben ser las 168 horas consecutivas anteriores al origen.');
  manifest.variables.forEach((name,j)=>{
   const raw=row[header.indexOf(name)].trim();
   if(!raw||/^(nan|na|null)$/i.test(raw))window[t*30+j]=NaN;
   else{const value=Number(raw);if(!Number.isFinite(value))throw Error('Valor no numérico en '+name);window[t*30+j]=value;}
  });
 });
 return window;
}
function updateTemplate(){const i=Number($('origin').value);$('template').href=`historia_${i}.csv`;$('template').download=`historia_${manifest.presets[i].origin}.csv`;}
function plot(r){
 const history=r.historia_kwh.slice(-24).map(x=>x/1e6),pred=r.prediccion_kwh.map(x=>x/1e6),base=(r.base_kwh||[]).map(x=>x/1e6);
 const interval=r.interval?.available?r.interval:null;
 $('plot-caption').textContent='Historia de entrada (gris) y pronóstico del modelo (verde). '+(r.base_kwh?'La base M1 se muestra en ocre. ':'')+(interval?'La banda representa el intervalo calibrado del checkpoint. ':'')+'La línea vertical marca el origen.';
 const all=[...history,...pred,...base,...(interval?interval.lower_kwh.map(x=>x/1e6):[]),...(interval?interval.upper_kwh.map(x=>x/1e6):[])],low=Math.floor((Math.min(...all)-.2)*2)/2,high=Math.ceil((Math.max(...all)+.2)*2)/2;
 const x=i=>65+i*800/47,y=v=>295-(v-low)*245/(high-low),path=(a,offset)=>a.map((v,i)=>(i?'L':'M')+x(i+offset).toFixed(2)+','+y(v).toFixed(2)).join(' ');
 const ns='http://www.w3.org/2000/svg',svg=$('plot');svg.replaceChildren();
 function el(name,attrs,text){const n=document.createElementNS(ns,name);for(const [k,v] of Object.entries(attrs))n.setAttribute(k,v);if(text)n.textContent=text;svg.append(n);}
 for(let i=0;i<5;i++){
  const value=low+(high-low)*i/4;el('line',{x1:65,x2:865,y1:y(value),y2:y(value),stroke:'#e5ebed'});el('text',{x:53,y:y(value)+4,'text-anchor':'end','font-size':12,fill:'#52636a'},number(value,1));
 }
 el('text',{x:65,y:23,'font-size':12,fill:'#52636a'},'GWh por intervalo');
 el('line',{x1:x(23.5),x2:x(23.5),y1:40,y2:295,stroke:'#809196','stroke-dasharray':'5 4'});
 if(interval){
  const upper=interval.upper_kwh.map(v=>v/1e6),lower=interval.lower_kwh.map(v=>v/1e6);
  const band=path(upper,24)+' '+lower.slice().reverse().map((v,i)=>'L'+x(47-i).toFixed(2)+','+y(v).toFixed(2)).join(' ')+' Z';
  el('path',{d:band,fill:'#126c70','fill-opacity':.17,stroke:'none'});
 }
 for(const [a,offset,color] of [[history,0,'#596c75'],[base,24,'#bd803b'],[pred,24,'#126c70']].filter(v=>v[0].length))el('path',{d:path(a,offset),fill:'none',stroke:color,'stroke-width':2.4});
 for(let h=0;h<24;h++){
  const dot=document.createElementNS(ns,'circle');dot.setAttribute('cx',x(h+24));dot.setAttribute('cy',y(pred[h]));dot.setAttribute('r',3);dot.setAttribute('fill','#126c70');
  const tip=document.createElementNS(ns,'title');tip.textContent=`${String(h).padStart(2,'0')}:00 · Predicción: ${r.prediccion_kwh[h]} kWh${r.base_kwh?' · M1: '+r.base_kwh[h]+' kWh':''}`;dot.append(tip);svg.append(dot);
 }
 for(const [i,label] of [[0,'Día anterior · 00:00'],[24,r.origen+' · 00:00'],[47,'23:00']])el('text',{x:x(i),y:320,'text-anchor':i===0?'start':i===47?'end':'middle','font-size':12,fill:'#52636a'},label);
}
function show(r){
 last=r;$('output').hidden=false;const values=r.prediccion_kwh.map(v=>v/1e6),max=Math.max(...values),hour=values.indexOf(max);
 $('metrics').replaceChildren();
 for(const [label,value,note] of [['Energía diaria',number(values.reduce((a,b)=>a+b,0),2)+' GWh','Suma de 24 intervalos'],['Pico horario',number(max)+' GWh',String(hour).padStart(2,'0')+':00 · Colombia'],(r.compuerta!==null&&r.compuerta!==undefined?['Compuerta',number(100*r.compuerta,1)+' %','Fracción de corrección residual']:r.texto_documentos!==null&&r.texto_documentos!==undefined?['Documentos en el origen',number(r.texto_documentos,0),'Ventana documental conservada']:['Semilla',String(r.semilla),'Checkpoint prefijado']),['Tiempo de cálculo',number(r.tiempo_ms,0)+' ms','En este navegador']]){
  const box=document.createElement('div');box.className='metric';
  for(const [tag,value2] of [['span',label],['strong',value],['small',note]]){const node=document.createElement(tag);node.textContent=value2;box.append(node);}$('metrics').append(box);
 }
 plot(r);
 const delta=r.base_kwh?r.prediccion_kwh.reduce((sum,v,i)=>sum+v-r.base_kwh[i],0)/24/1e6:null;
 $('interpretation').textContent=`${r.name} sitúa el pico pronosticado a las ${String(hour).padStart(2,'0')}:00.`+(delta!==null?` La memoria modifica el nivel horario medio de M1 en ${delta>=0?'+':''}${number(delta,4)} GWh, con ${r.vecinos.length} episodios disponibles.`:'')+` La trayectoria corresponde a la semilla ${r.semilla}. `+(r.interval?.available?`Banda nominal del ${number(100*r.interval.level,0)} % (${r.interval.scope==='horario'?'por hora':'conjunta de 24 horas'}), calibrada con ${r.interval.n} días hasta ${r.interval.end}. La cobertura es empírica bajo dependencia temporal.`:r.interval?.reason||'Sin calibración disponible.')+(r.texto_documentos!==null&&r.texto_documentos!==undefined?' La ponderación documental se recalcula con la historia numérica y los vectores disponibles del origen.':'');
 $('domain').textContent=r.fuera_rango.length?'Entradas fuera del rango de 2000–2021: '+r.fuera_rango.join(', ')+'.':'Las entradas finitas están dentro de los rangos registrados de 2000–2021.';
 $('provenance').textContent=`${r.motor}. Semilla ${r.semilla}; pesos y escaladores congelados.`+(r.model==='m4_n'?' Episodios de dos a 365 días de edad, disponibles antes del origen. Las entradas modificadas conservan el banco histórico del origen seleccionado.':'')+(r.periodos?' Periodos seleccionados por bloque: '+JSON.stringify(r.periodos)+'.':'');
 $('neighbors').replaceChildren();for(const v of r.vecinos){const li=document.createElement('li');li.textContent=v.fecha+' · '+number(v.peso*100,2)+' %';$('neighbors').append(li);}
 status('Pronóstico calculado en el navegador con los pesos congelados.');
}
function onMessage({data}){
 if(data.type==='progress'){status(data.message.replace('.onnx','')+'…');return;}
 busy(false);
 if(data.type==='error'){status('No se pudo calcular: '+data.message,true);worker.terminate();worker=null;return;}
 show(data.result);
}
async function start(){
 try{
  manifest=await (await fetch('manifest.json')).json();
  const catalogue=await (await fetch('catalogue.json')).json();
  for(const [id,name] of [['m4_n','M4ₙ'],...Object.entries(catalogue.models).map(([id,r])=>[id,r.name])]){const option=document.createElement('option');option.value=id;option.textContent=name;$('model').append(option);}
  manifest.presets.forEach((p,i)=>{const option=document.createElement('option');option.value=i;option.textContent=p.origin;$('origin').append(option);});
  $('origin').value=manifest.presets.length-1;updateTemplate();
 }catch(e){status('No se pudo cargar el contrato: '+e.message,true);$('run').disabled=true;}
}
$('controls').addEventListener('submit',async e=>{
 e.preventDefault();if(!$('controls').reportValidity())return;
 busy(true);$('output').hidden=true;last=null;status('Preparando modelos y estado causal… La primera ejecución descarga el motor WebAssembly.');
 try{
  const index=Number($('origin').value),file=$('history').files[0];
  custom=file?await readHistory(file,index):null;
  worker??=new Worker('./worker.js');worker.onmessage=onMessage;
  worker.onerror=e=>{busy(false);status('No se pudo iniciar el motor del navegador: '+e.message,true);worker.terminate();worker=null;};
  worker.postMessage({model:$('model').value,index,change:Number($('change').value),window:custom,coverage:Number($('coverage').value),scope:$('interval-scope').value});
 }catch(error){busy(false);status(error.message,true);}
});
$('model').addEventListener('change',()=>{if(worker){worker.terminate();worker=null;}invalidate();});
$('origin').addEventListener('change',()=>{$('history').value='';custom=null;updateTemplate();invalidate();});
$('coverage').addEventListener('change',invalidate);$('interval-scope').addEventListener('change',invalidate);
$('history').addEventListener('change',invalidate);$('change').addEventListener('input',invalidate);
$('reset').addEventListener('click',()=>{$('history').value='';$('change').value=0;custom=null;invalidate();});
$('download').addEventListener('click',()=>{
 if(!last)return;const interval=last.interval?.available?last.interval:null;const content='fecha,prediccion_kwh'+(interval?',inferior_kwh,superior_kwh,cobertura_nominal,alcance':'')+(last.base_kwh?',base_m1_kwh':'')+'\n'+last.prediccion_kwh.map((v,h)=>`${last.origen}T${String(h).padStart(2,'0')}:00:00-05:00,${v}${interval?','+interval.lower_kwh[h]+','+interval.upper_kwh[h]+','+interval.level+','+interval.scope:''}${last.base_kwh?','+last.base_kwh[h]:''}`).join('\n');
 const url=URL.createObjectURL(new Blob(['\uFEFF'+content],{type:'text/csv;charset=utf-8'})),a=document.createElement('a');a.href=url;a.download=`${last.name}_${last.origen}.csv`;a.click();URL.revokeObjectURL(url);
});
start();

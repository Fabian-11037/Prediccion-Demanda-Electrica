/* Intervalos causales del checkpoint conservado; sin ajuste de pesos. */
(function(root){
 const day=s=>Date.parse(s+'T00:00:00Z')/86400000;
 function calibrate(meta,key,scores,origin,prediction,level=.9,scope='horario'){
  if(![.8,.9,.95].includes(Number(level)))throw Error('Nivel de cobertura incompatible.');
  if(!['horario','trayectoria24'].includes(scope))throw Error('Alcance incompatible.');
  const model=meta.models[key];if(!model)throw Error('Checkpoint sin calibración.');
  const t=day(origin),eligible=[];
  if(!Number.isFinite(t))throw Error('Origen incompatible.');
  const empty=reason=>({available:false,reason,level:Number(level),scope,n:0});
  if(t<day('2023-01-01'))return empty('No hay reserva de calibración anterior al origen. Los intervalos comienzan en 2023.');
  meta.dias.forEach((d,i)=>{const age=t-day(d);if(age>=2&&age<=366)eligible.push(i);});
  const n=eligible.length,rank=Math.ceil((n+1)*Number(level));
  if(n<60||rank>n)return empty('No hay suficientes errores recientes disponibles para calibrar este origen.');
  let radii;
  if(scope==='horario')radii=Array.from({length:24},(_,h)=>eligible.map(i=>scores[i*24+h]).sort((a,b)=>a-b)[rank-1]);
  else{
   const values=eligible.map(i=>Math.max(...model.scales.map((s,h)=>scores[i*24+h]/s))).sort((a,b)=>a-b),q=values[rank-1];
   radii=model.scales.map(s=>q*s);
  }
  return{available:true,level:Number(level),scope,n,start:meta.dias[eligible[0]],end:meta.dias[eligible.at(-1)],
   lower_kwh:prediction.map((v,h)=>v-radii[h]),upper_kwh:prediction.map((v,h)=>v+radii[h]),seed:model.seed,
   method:'Cuantil móvil causal365; retraso2días; checkpoint individual'};
 }
 async function create(base,key){
  const url=new URL('calibracion/',base),response=await fetch(new URL('manifest.json',url));if(!response.ok)throw Error('No se pudo cargar la calibración.');
  const meta=await response.json(),entry=meta.models[key];if(!entry)throw Error('No hay calibración para el modelo.');
  const r=await fetch(new URL(entry.file,url));if(!r.ok)throw Error('No se pudieron cargar los errores de calibración.');
  const buffer=await r.arrayBuffer(),hash=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',buffer))).map(x=>x.toString(16).padStart(2,'0')).join('');
  if(hash!==entry.sha256||buffer.byteLength!==meta.dias.length*24*4)throw Error('La calibración no coincide con su registro.');
  const scores=new Float32Array(buffer);
  return{infer:(origin,pred,level,scope)=>calibrate(meta,key,scores,origin,pred,level,scope)};
 }
 root.LiveIntervals={create,calibrate};
})(typeof self==='undefined'?globalThis:self);

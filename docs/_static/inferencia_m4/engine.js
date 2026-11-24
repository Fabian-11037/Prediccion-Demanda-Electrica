/* Preparación y ejecución de M4_N. Escalas y pesos congelados. */
(function(scope){
'use strict';
const f32=x=>Math.fround(x), product=d=>d.reduce((a,b)=>a*b,1);
function checkWindow(window,m){
 if(window.length!==168*30)throw Error('Se requieren 168 horas y 30 variables históricas.');
 for(let t=0;t<168;t++)for(let j=0;j<30;j++){
  const v=window[t*30+j];
  if(!Number.isFinite(v)&&!Number.isNaN(v))throw Error('Las entradas no admiten valores infinitos.');
  if(j===0&&(!Number.isFinite(v)||v<=0))throw Error('La demanda debe ser finita y positiva en las 168 horas.');
 }
 for(const group of m.grupos_mascara)for(let t=0;t<168;t++){
  const absent=Number.isNaN(window[t*30+group[0]]);
  if(group.some(j=>Number.isNaN(window[t*30+j])!==absent))throw Error('Las ausencias deben respetar las máscaras comunes de precio, hidrología y clima.');
 }
}
function prepare(state,m,change=0,custom=null){
 if(!Number.isFinite(change)||Math.abs(change)>20)throw Error('El cambio de demanda debe estar entre −20 % y 20 %.');
 const window=new Float64Array(custom||state.window);checkWindow(window,m);
 for(let t=144;t<168;t++)window[t*30]*=1+change/100;
 checkWindow(window,m);
 const context=new Float32Array(state.contexto),counts=new Int32Array(30),counts24=new Int32Array(30),sums=new Float64Array(30),sums24=new Float64Array(30);
 let level=0;
 for(let t=0;t<168;t++){
  level+=window[t*30]/168;
  for(let j=0;j<30;j++)if(Number.isFinite(window[t*30+j])){
   const v=window[t*30+j];counts[j]++;sums[j]+=v;
   if(t>=144){counts24[j]++;sums24[j]+=v;}
  }
 }
 for(let j=0;j<30;j++){
  context[j]=Number.isFinite(window[167*30+j])?window[167*30+j]:0;
  context[30+j]=sums[j]/Math.max(counts[j],1);
  context[60+j]=sums24[j]/Math.max(counts24[j],1);
  context[90+j]=1-counts[j]/168;
 }
 for(const j of [0,30,60])context[j]=context[j]/level;
 context.set(state.cf,120);
 const experts=m.experts.map(sc=>{
  const x=new Float32Array(168*33),cp=new Float32Array(168*23),cf=new Float32Array(24*23);
  const vals=k=>Array.isArray(sc[k])?sc[k]:m.variables.map(c=>sc[k][c]);
  const med=vals('mediana'),mean=vals('media'),std=vals('std');
  const cm=sc.cal_media||sc.media_cal||Object.values(sc.media_calendario||{});
  const cs=sc.cal_std||sc.std_cal||Object.values(sc.std_calendario||{});
  for(let t=0;t<168;t++){
   for(let j=0;j<30;j++){
    const raw=window[t*30+j],v=Number.isFinite(raw)?raw:med[j];
    x[t*33+j]=j===0?(v/level-1)/sc.sigma_demanda:(v-mean[j])/std[j];
   }
   m.grupos_mascara.forEach((group,k)=>x[t*33+30+k]=Number.isNaN(window[t*30+group[0]])?1:0);
   for(let j=0;j<23;j++)cp[t*23+j]=(state.cp[t*23+j]-cm[j])/cs[j];
  }
  for(let t=0;t<24;t++)for(let j=0;j<23;j++)cf[t*23+j]=(state.cf[t*23+j]-cm[j])/cs[j];
  if([...x,...cp,...cf].some(v=>!Number.isFinite(v)))throw Error('El escalado produjo valores no finitos.');
  return {x,cp,cf,dia:state.dia};
 });
 const outside=m.variables.filter((_,j)=>Array.from({length:168},(_,t)=>window[t*30+j]).some(v=>Number.isFinite(v)&&(v<m.bounds[0][j]||v>m.bounds[1][j])));
 return {window,context,level,experts,outside};
}
function unpack(buffer,description){
 const arrays={};
 for(const [key,a] of Object.entries(description.arrays)){
  if(product(a.shape)!==a.length)throw Error('Dimensiones incompatibles en el estado.');
  const T=a.type==='float64'?Float64Array:a.type==='float32'?Float32Array:BigInt64Array;
  arrays[key]=new T(buffer,a.offset,a.length);
 }
 return arrays;
}
async function sha256(buffer){return Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',buffer)),x=>x.toString(16).padStart(2,'0')).join('');}
async function verified(url,hash){
 const r=await fetch(url);if(!r.ok)throw Error('No se pudo cargar '+url);
 const b=await r.arrayBuffer();if(await sha256(b)!==hash)throw Error('El recurso no coincide con su SHA-256: '+url);return b;
}
async function create(ort,base,notify=()=>{}){
 const response=await fetch(new URL('manifest.json',base));if(!response.ok)throw Error('No se encuentra el contrato de inferencia.');
 const m=await response.json(),sessions={},cache=new Map();
 for(const [file,hash] of Object.entries(m.models)){
  notify('Cargando '+file);
  sessions[file]=await ort.InferenceSession.create(await verified(new URL('modelos/'+file,base),hash),{executionProviders:['wasm'],graphOptimizationLevel:'all'});
 }
 const tensor=(type,data,shape)=>new ort.Tensor(type,data,shape);
 async function infer(index,change=0,custom=null){
  const preset=m.presets[index];if(!preset)throw Error('Origen sin estado residual preparado.');
  if(!cache.has(index))cache.set(index,unpack(await verified(new URL(preset.file,base),preset.sha256),preset));
  const state=cache.get(index),start=performance.now(),p=prepare(state,m,change,custom),xp=new Float32Array(72);
  for(let i=0;i<3;i++){
   const s=sessions[`experto_${i}.onnx`],d=p.experts[i];
   const all={x:tensor('float32',d.x,[1,168,33]),cp:tensor('float32',d.cp,[1,168,23]),cf:tensor('float32',d.cf,[1,24,23]),dia:tensor('int64',d.dia,[1])};
   const out=await s.run(Object.fromEntries(s.inputNames.map(k=>[k,all[k]])));
   const pred=out[s.outputNames[0]].data;
   for(let h=0;h<24;h++)xp[h*3+i]=f32(pred[h]*p.level);
  }
  for(let h=0;h<24;h++){
   let mean=0;for(let i=0;i<3;i++){p.context[672+h*3+i]=xp[h*3+i]/p.level;mean+=xp[h*3+i]/3;}
   let variance=0;for(let i=0;i<3;i++)variance+=(xp[h*3+i]-mean)**2/3;
   p.context[744+h]=Math.sqrt(variance)/p.level;
  }
  const normalize=(mean,scale)=>Float32Array.from(p.context,(v,j)=>(v-mean[j])/scale[j]);
  const router=sessions['router.onnx'];
  const routed=await router.run({x:tensor('float32',normalize(m.scales.router_media,m.scales.router_escala),[1,912]),pred:tensor('float32',xp,[1,24,3])});
  const basePrediction=routed[router.outputNames[0]],memory=sessions['memoria.onnx'];
  const out=await memory.run({query:tensor('float32',normalize(m.scales.memory_media,m.scales.memory_escala),[1,912]),keys:tensor('float32',state.keys,[365,912]),days:tensor('int64',state.days,[365]),origin:tensor('int64',state.origin,[1]),residuals:tensor('float32',state.residuals,[365,24]),base:basePrediction});
  const values=Array.from(out.pred.data);if(values.length!==24||values.some(v=>!Number.isFinite(v)))throw Error('El modelo no produjo 24 valores finitos.');
  const attention=out.attention.data,ids=out.neighbors.data,neighbors=[];
  for(let k=0;k<ids.length;k++){
   const j=Number(ids[k]),age=Number(state.origin[0]-state.days[j]);
   if(age>=2&&age<=365&&attention[k]>0)neighbors.push({fecha:new Date(Number(state.days[j])*86400000).toISOString().slice(0,10),peso:attention[k]});
  }
  return {origen:preset.origin,prediccion_kwh:values,base_kwh:Array.from(basePrediction.data),compuerta:out.gate.data[0],vecinos:neighbors,historia_kwh:Array.from({length:168},(_,t)=>p.window[t*30]),fuera_rango:p.outside,tiempo_ms:performance.now()-start,semilla:101,entrenamiento:false,calculado:true,motor:'ONNX Runtime Web · WASM'};
 }
 return {manifest:m,infer};
}
scope.M4Native={create,prepare,unpack,checkWindow};
})(globalThis);

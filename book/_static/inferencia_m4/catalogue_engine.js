/* Catálogo nativo: pesos FP32 conservados y selección dinámica de periodos. */
(function(scope){
'use strict';
async function digest(data){return Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',data)),x=>x.toString(16).padStart(2,'0')).join('');}
async function bytes(url,hash){const r=await fetch(url);if(!r.ok)throw Error('Recurso no disponible: '+url);const b=await r.arrayBuffer();if(await digest(b)!==hash)throw Error('SHA-256 incompatible: '+url);return b;}
function spectralPeriods(e,T,C,k){
 const amplitudes=new Float32Array(Math.floor(T/2)+1);
 for(let f=1;f<amplitudes.length;f++){
  const cos=new Float64Array(T),sin=new Float64Array(T);
  for(let t=0;t<T;t++){cos[t]=Math.cos(2*Math.PI*f*t/T);sin[t]=-Math.sin(2*Math.PI*f*t/T);}
  let mean=0;
  for(let c=0;c<C;c++){let re=0,im=0;for(let t=0;t<T;t++){const v=e[t*C+c];re+=v*cos[t];im+=v*sin[t];}mean+=Math.hypot(re,im)/C;}
  amplitudes[f]=mean;
 }
 const frequencies=Array.from(amplitudes,(_,i)=>i).sort((a,b)=>amplitudes[b]-amplitudes[a]).slice(0,k);
 if(frequencies.includes(0))throw Error('El espectro no permite determinar periodos válidos.');
 const max=Math.max(...frequencies.map(f=>amplitudes[f])),w=frequencies.map(f=>Math.exp(amplitudes[f]-max)),sum=w.reduce((a,b)=>a+b,0);
 return {frequencies,periods:frequencies.map(f=>Math.floor(T/f)),weights:w.map(v=>Math.fround(v/sum))};
}
function inputs(state,m,record,change,custom,index){
 const p=M4Native.prepare(state,m,change,custom),sc=record.scaler,N=record.channels;
 const x=new Float32Array(168*N),cp=new Float32Array(168*23),cf=new Float32Array(24*23);
 for(let t=0;t<168;t++){
  for(let j=0;j<23;j++)cp[t*23+j]=(state.cp[t*23+j]-sc.cal_mean[j])/sc.cal_std[j];
  if(sc.kind==='baseline'){
   x[t*N]=sc.sequential?(p.window[t*30]/p.level-1)/sc.sigma:p.window[t*30]/p.level;
   for(let j=0;j<23;j++)x[t*N+1+j]=cp[t*23+j];
  }else{
   for(let j=0;j<30;j++){
    const raw=p.window[t*30+j],v=Number.isFinite(raw)?raw:sc.median[j];
    x[t*N+j]=j===0?(v/p.level-1)/sc.sigma:(v-sc.mean[j])/sc.std[j];
   }
   if(sc.kind==='repo'){
    for(let j=0;j<23;j++)x[t*N+30+j]=cp[t*23+j];
    for(let j=0;j<30;j++)x[t*N+53+j]=Number.isNaN(p.window[t*30+j])?1:0;
   }else m.grupos_mascara.forEach((group,j)=>x[t*N+30+j]=Number.isNaN(p.window[t*30+group[0]])?1:0);
  }
 }
 for(let t=0;t<24;t++)for(let j=0;j<23;j++)cf[t*23+j]=(state.cf[t*23+j]-sc.cal_mean[j])/sc.cal_std[j];
 if([...x,...cp,...cf].some(v=>!Number.isFinite(v)))throw Error('El escalado produjo entradas no finitas.');
 return {...p,x,cp,cf,dia:record.text_counts?new BigInt64Array([BigInt(index*7+6)]):state.dia};
}
async function create(ort,base,key,notify=()=>{}){
 const [m,catalogue]=await Promise.all(['manifest.json','catalogue.json'].map(async file=>{const r=await fetch(new URL(file,base));if(!r.ok)throw Error('No se pudo cargar el contrato.');return r.json();}));
 const record=catalogue.models[key];if(!record)throw Error('Modelo fuera del catálogo de inferencia.');
 const tensor=(data,shape,type='float32')=>new ort.Tensor(type,data,shape),cache=new Map();
 async function session(description){
  notify('Cargando pesos de '+record.name+' ('+Math.ceil(description.bytes/1048576)+' MB)');
  const data=new Uint8Array(description.bytes);let offset=0;
  for(const part of description.parts){const b=new Uint8Array(await bytes(new URL(part.file,base),part.sha256));data.set(b,offset);offset+=b.length;}
  if(offset!==description.bytes||await digest(data)!==description.sha256)throw Error('El grafo ensamblado no coincide con el original.');
  return ort.InferenceSession.create(data,{executionProviders:['wasm'],graphOptimizationLevel:'all'});
 }
 let direct,prep,norm,convs,finish;
 if(record.mode==='timesnet'){
  prep=await session(record.prepare);norm=await session(record.norm);convs=[];
  for(const conv of record.convs)convs.push(await session(conv));finish=await session(record.finish);
 }else direct=await session(record.graph);
 async function timesnet(d){
  const initial=await prep.run({x:tensor(d.x,[1,168,record.channels]),cp:tensor(d.cp,[1,168,23])});
  let e=initial.e;const T=record.length,C=record.d_model,selected=[];
  for(let layer=0;layer<record.layers;layer++){
   const spectral=spectralPeriods(e.data,T,C,record.top_k),sum=new Float32Array(T*C);selected.push(spectral.periods);
   for(let i=0;i<spectral.periods.length;i++){
    const period=spectral.periods[i],rows=Math.ceil(T/period),length=rows*period,z=new Float32Array(C*length);
    for(let c=0;c<C;c++)for(let t=0;t<T;t++)z[c*length+t]=e.data[t*C+c];
    const conv=convs[layer],out=await conv.run({x:tensor(z,[1,C,rows,period])}),values=out[conv.outputNames[0]].data;
    for(let t=0;t<T;t++)for(let c=0;c<C;c++){const j=t*C+c;sum[j]=Math.fround(sum[j]+Math.fround(values[c*length+t]*spectral.weights[i]));}
   }
   for(let j=0;j<sum.length;j++)sum[j]=Math.fround(sum[j]+e.data[j]);
   const out=await norm.run({e:tensor(sum,[1,T,C])});e=out[norm.outputNames[0]];
  }
  const out=await finish.run({e,mean:initial.mean,std:initial.std,cf:tensor(d.cf,[1,24,23])});
  return {values:out[finish.outputNames[0]].data,periods:selected};
 }
 async function infer(index,change=0,custom=null){
  const preset=m.presets[index];if(!preset)throw Error('Origen sin estado preparado.');
  if(!cache.has(index))cache.set(index,M4Native.unpack(await bytes(new URL(preset.file,base),preset.sha256),preset));
  const state=cache.get(index),start=performance.now(),d=inputs(state,m,record,change,custom,index);
  let values,periods;
  if(record.mode==='timesnet')({values,periods}=await timesnet(d));
  else{
   const all={x:tensor(d.x,[1,168,record.channels]),cp:tensor(d.cp,[1,168,23]),cf:tensor(d.cf,[1,24,23]),dia:tensor(d.dia,[1],'int64')};
   const out=await direct.run(Object.fromEntries(direct.inputNames.map(name=>[name,all[name]])));values=out[direct.outputNames[0]].data;
  }
  const prediction=Array.from(values,v=>v*d.level);
  if(prediction.length!==24||prediction.some(v=>!Number.isFinite(v)))throw Error('No se obtuvieron 24 predicciones finitas.');
  return {origen:preset.origin,model:key,name:record.name,prediccion_kwh:prediction,base_kwh:null,compuerta:null,vecinos:[],historia_kwh:Array.from({length:168},(_,t)=>d.window[t*30]),fuera_rango:record.scaler.kind==='baseline'?d.outside.filter(c=>c===m.variables[0]):d.outside,tiempo_ms:performance.now()-start,semilla:1000,entrenamiento:false,calculado:true,motor:'ONNX Runtime Web · WASM',texto_documentos:record.text_counts?.[index]??null,periodos:periods??null};
 }
 return {infer};
}
scope.CatalogueNative={create,inputs,spectralPeriods};
})(globalThis);

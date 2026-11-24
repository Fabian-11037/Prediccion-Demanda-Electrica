importScripts('./vendor/ort.wasm.min.js','./engine.js','./catalogue_engine.js');
ort.env.wasm.numThreads=1;
ort.env.wasm.proxy=false;
ort.env.wasm.wasmPaths={
 mjs:new URL('./vendor/ort-wasm-simd-threaded.js',self.location.href).href,
 wasm:new URL('./vendor/ort-wasm-simd-threaded.wasm',self.location.href).href
};
let runtime;
self.onmessage=async({data})=>{
 try{
  const key=data.model||'m4_n',base=new URL('./',self.location.href),notify=message=>self.postMessage({type:'progress',message});
  runtime??=key==='m4_n'?M4Native.create(ort,base,notify):CatalogueNative.create(ort,base,key,notify);
  const r=await runtime;
  const result=await r.infer(data.index,data.change,data.window);
  result.model=data.model||'m4_n';result.name??='M4ₙ';
  self.postMessage({type:'result',result});
 }catch(error){runtime=undefined;self.postMessage({type:'error',message:error.message||String(error)});}
};

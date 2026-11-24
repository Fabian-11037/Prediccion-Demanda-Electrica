const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm'),path=require('node:path');
const context={Float32Array,Float64Array,BigInt64Array,performance,URL,crypto:globalThis.crypto};
vm.createContext(context);vm.runInContext(fs.readFileSync(path.resolve(__dirname,'../../book/_static/inferencia_m4/catalogue_engine.js'),'utf8'),context);
test('TimesNet vuelve a seleccionar los periodos según la entrada',()=>{
 const T=192,C=2;
 function signal(f){return Float32Array.from({length:T*C},(_,j)=>Math.cos(2*Math.PI*f*Math.floor(j/C)/T));}
 const a=context.CatalogueNative.spectralPeriods(signal(4),T,C,1);
 const b=context.CatalogueNative.spectralPeriods(signal(8),T,C,1);
 assert.equal(a.frequencies[0],4);assert.equal(b.frequencies[0],8);
 assert.equal(a.periods[0],48);assert.equal(b.periods[0],24);
 assert.equal(a.weights[0],1);assert.equal(b.weights[0],1);
});

import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
const read=path=>JSON.parse(readFileSync(new URL(path,import.meta.url),'utf8'));
const c=read('../site/data/m4_closure.json');

test('El dashboard conserva la verificación ONNX del JBook y sus descargas',()=>{
 const score=read('../../results/auditoria_e3_final_20261124/roundtrip_metricas.json');
 assert.deepEqual(c.export_score,score);
 assert.equal(score.n_intervalos,96);
 assert.equal(score.n_dias,4);
 assert.ok(Math.abs(score.diferencia_MASE)<1e-6);
 const manifest=read('../../results/auditoria_e3_final_20261124/sincronizacion_dashboard.json');
 for(const [path,hash] of Object.entries(manifest.sha256)){
  assert.equal(createHash('sha256').update(readFileSync(new URL('../site/'+path,import.meta.url))).digest('hex'),hash,path);
 }
 assert.deepEqual(c.importance_protocol,{origins:48,repetitions:3,groups:6});
 assert.ok(c.fidelity_at16.atencion>c.fidelity_at16.aleatorio);
});

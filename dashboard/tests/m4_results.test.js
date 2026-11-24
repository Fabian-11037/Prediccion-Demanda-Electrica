import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {windowMetrics,m4AdjustedP} from '../site/core.js';
const read=p=>JSON.parse(readFileSync(new URL('../site/data/'+p,import.meta.url),'utf8'));
const study=read('study.json'),m4=read('m4_results.json'),pred=read('predictions/m4_n.json');

test('M4 reproduce sus métricas individuales guardadas con diez semillas',()=>{
 const score=windowMetrics(pred,0,1461,study.protocol.q168);
 assert.equal(pred.mse_daily.length,10);assert.equal(pred.mean.length,1461);
 for(const [field,key] of [['mae','MAE'],['rmse','RMSE'],['mase','MASE']])assert.ok(Math.abs(score[field]-m4.benchmark[key])<1e-6,field);
 assert.equal(m4.benchmark.semillas,10);
 assert.equal(m4.comparison_benchmark.filter(r=>r.clave==='m4_n').length,1);
 assert.equal(m4.comparison_benchmark.length,15);
 assert.ok(Math.abs(m4.benchmark.MASE-0.5770369087419158)<1e-12);
});

test('Las diez semillas conservan el mínimo interno y sus épocas reales',()=>{
 assert.deepEqual(m4.protocol.seeds,[101,102,103,104,105,106,107,108,109,110]);
 for(const checkpoint of m4.checkpoints){
  const history=m4.curves.filter(r=>r.semilla===checkpoint.semilla);
  assert.equal(history.length,checkpoint.epocas);
  const best=history.reduce((a,b)=>a.val_loss<b.val_loss?a:b);
  assert.equal(best.epoca,checkpoint.mejor_epoca);
  assert.ok(Math.abs(best.val_loss-checkpoint.mejor_perdida)<1e-12);
 }
});

test('Los 60 contrastes pertenecen exactamente a los diez modelos del Top 10',()=>{
 const ids=study.models.filter(m=>m.group==='Top 10').map(m=>m.id).sort();
 assert.deepEqual([...new Set(m4.tests.map(t=>t.comparator_id))].sort(),ids);
 assert.equal(m4.tests.length,60);
 for(const id of ids){
  const rows=m4.tests.filter(t=>t.comparator_id===id);
  assert.equal(rows.length,6);
  assert.equal(new Set(rows.map(t=>t.metodo+':'+t.margen_relativo)).size,6);
  for(const row of rows){assert.equal(row.replicas_bootstrap,49999);assert.ok(row.p_holm_cota_familia1536>=0&&row.p_holm_cota_familia1536<=1);}
 }
 assert.equal(m4.protocol.holm_family,1536);
 assert.ok(m4.tests.filter(t=>t.metodo==='bootstrap_estacionario'&&t.margen_relativo===.03).every(t=>t.p_holm_cota_familia1536<.05));
 assert.equal(m4.comparators.filter(m=>m.grupo!=='Top 10').length,4);
});

test('El resumen usa los tres contrastes ajustados, sin inventar parejas ausentes',()=>{
 const ids=study.models.filter(m=>m.group==='Top 10').map(m=>m.id);
 for(const id of ids){
  const rows=m4.tests.filter(t=>t.comparator_id===id&&t.margen_relativo===0);
  assert.equal(m4AdjustedP(m4.tests,id),Math.max(...rows.map(t=>t.p_holm_cota_familia1536)));
  assert.ok(m4AdjustedP(m4.tests,id)<.05);
 }
 assert.equal(m4AdjustedP(m4.tests,'m4_n'),null);
 assert.equal(m4AdjustedP(m4.tests,'mlp'),null);
 const id=ids[0],incomplete=m4.tests.filter(t=>!(t.comparator_id===id&&t.metodo==='GW_bilateral'&&t.margen_relativo===0));
 assert.equal(m4AdjustedP(incomplete,id),null);
 const invalid=m4.tests.map(t=>t.comparator_id===id&&t.metodo==='GW_bilateral'&&t.margen_relativo===0?{...t,p_holm_cota_familia1536:null}:t);
 assert.equal(m4AdjustedP(invalid,id),null);
});

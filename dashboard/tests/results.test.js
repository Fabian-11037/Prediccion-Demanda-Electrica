import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync,existsSync} from 'node:fs';
import {windowMetrics,spearman,calendarProfile,temporalVerdict} from '../site/core.js';
const json=name=>JSON.parse(readFileSync(new URL('../site/data/'+name,import.meta.url),'utf8'));
const study=json('study.json');
test('La evaluación completa reproduce las métricas guardadas de los 13 procedimientos',()=>{
 assert.equal(study.models.length,13);assert.equal(study.dates.length,1461);
 assert.equal(study.models.filter(m=>m.family==='patchtst').length,1);
 for(const model of study.models){
  const pred=json('predictions/'+model.id+'.json');
  const actual=windowMetrics(pred,0,1461,study.protocol.q168);
  const expected=study.tables.benchmark.find(r=>r.clave===model.id);
  for(const [key,value] of [['mae',expected.MAE],['rmse',expected.RMSE],['mase',expected.MASE]])
   assert.ok(Math.abs(actual[key]-value)<1e-6,model.id+' '+key);
  assert.equal(pred.mean.length,1461);assert.equal(pred.mean[0].length,24);assert.equal(pred.mse_daily.length,5);
  for(const type of ['global','detalle']) assert.ok(existsSync(new URL('../site/architectures/'+model.family+'_'+type+'.png',import.meta.url)));
 }
});
test('Las ventanas respetan el final del periodo y RMSE se calcula antes de agregar semillas',()=>{
 const p={mae_daily:[2,4],mse_daily:[[1,9],[9,25]]};
 const r=windowMetrics(p,0,2,2);assert.equal(r.mae,3);assert.equal(r.rmse,(Math.sqrt(5)+Math.sqrt(17))/2);
 assert.equal(windowMetrics(p,1,30,2).n,1);assert.throws(()=>windowMetrics(p,-1,7,2));
 assert.throws(()=>windowMetrics(p,2,7,2));
});
test('Los perfiles ponderan observaciones; correlaciones ignoran ausencias y tratan empates',()=>{
 const row=(mean,count)=>({mean:Array(24).fill(mean),count:Array(24).fill(count)});
 assert.equal(calendarProfile({2020:{0:row(2,1)},2021:{0:row(6,3)}},2020,2021)[0][0],5);
 assert.equal(spearman([1,1,3,null,4],[2,2,4,9,5]),1);
 assert.equal(spearman([1,1,1],[2,3,4]),null);
 assert.match(temporalVerdict({p_DM_Holm:.8,p_TOST_Holm:.8}),/No se detecta/);
 assert.match(temporalVerdict({p_DM_Holm:.8,p_TOST_Holm:.01}),/Equivalencia/);
});

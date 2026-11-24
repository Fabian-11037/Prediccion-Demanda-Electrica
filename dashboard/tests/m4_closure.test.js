import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {m4AdjustedP} from '../site/core.js';
const read=name=>JSON.parse(readFileSync(new URL('../site/data/'+name,import.meta.url),'utf8'));
const old=read('m4_results.json'),closure=read('m4_closure.json'),study=read('study.json');

test('La ampliación conserva las familias y completa todos los procedimientos del resumen',()=>{
 const combined=[...old.tests,...closure.tests];
 assert.equal(old.tests.length,60);assert.equal(closure.complete_tests.length,102);assert.equal(closure.tests.length,42);
 for(const model of study.models){const p=m4AdjustedP(combined,model.id);assert.ok(Number.isFinite(p)&&p>=0&&p<.05,model.id);}
 assert.equal(m4AdjustedP(combined,'m4_n'),null);
 for(const row of closure.tests)assert.equal(row.familia,'complementaria_102');
 const ids=new Set(closure.complete_tests.map(r=>r.comparator_id));assert.equal(ids.size,17);
 assert.equal(closure.statistics.n_modelos,18);
 assert.deepEqual(closure.mcs.filter(r=>r.retenido_90).map(r=>r.clave),closure.statistics.mcs_retenidos);
});

test('Intervalos, eficiencia y exportación mantienen alcance y unidades',()=>{
 const row=closure.coverage.find(r=>r.metodo==='movil_causal_365'&&r.nivel===.9&&r.segmento==='total');
 assert.ok(row.cobertura>.85&&row.cobertura<.95);assert.ok(row.ancho_GWh>0&&row.ancho_GWh<2);
 assert.equal(closure.calibration.lag_dias,2);assert.equal(closure.calibration.actualiza_pesos,false);
 assert.equal(closure.cost.length,2);assert.ok(closure.cost.every(r=>r.p95_ms>=r.mediana_ms&&r.parametros_almacenados>0));
 assert.equal(Object.keys(closure.property.tests).length,10);assert.ok(Object.values(closure.property.tests).every(Boolean));
 assert.equal(closure.export.modulos,5);assert.ok(closure.export.resultados.every(r=>r.max_error_relativo<closure.export.tolerancia_relativa));
});

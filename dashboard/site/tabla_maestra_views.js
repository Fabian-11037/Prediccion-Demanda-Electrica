import {escapeHTML as escape} from './core.js?v=revision-editorial-20261124';

export function masterRows(data,state={}){
 const group=state.masterGroup??'Todos',field=state.masterOrder??'MASE_prueba',direction=state.masterDirection==='desc'?-1:1;
 return data.rows.filter(r=>group==='Todos'||r.grupo===group).slice().sort((a,b)=>{
  const x=a[field],y=b[field];if(x===null)return 1;if(y===null)return-1;
  return direction*(typeof x==='number'?x-y:String(x).localeCompare(String(y),'es'));
 });
}

export function masterTableHTML(data,api){
 if(!data)return'';
 const{state,num,pvalue,table,field,select,book}=api,e=escape;
 const rows=masterRows(data,state),view=state.masterView??'resumen';
 const small=s=>`<br><span class="fine">${e(s)}</span>`;
 const columns={
  modelo:['Procedimiento',r=>e(r.modelo)+small(r.grupo)],
  entradas:['Entradas',r=>e(r.entradas)],
  desarrollo:['MASE · desarrollo',r=>num(r.MASE_desarrollo,4)+small(r.fuente_desarrollo+' · '+r.periodo_desarrollo)],
  mase:['MASE · prueba e IC95 %',r=>num(r.MASE_prueba,4)+small(`[${num(r.MASE_IC95_inf,4)}; ${num(r.MASE_IC95_sup,4)}]`)],
  mae:['MAE · GWh',r=>num(r.MAE_prueba_kWh/1e6,4)],rmse:['RMSE · GWh',r=>num(r.RMSE_prueba_kWh/1e6,4)],
  seeds:['Semillas · prueba',r=>num(r.semillas_prueba,0)],sd:['DE · semillas MASE',r=>num(r.MASE_DE_semillas,4)],
  p:['p ajustado · M4',r=>r.p_conjunto_Holm===null?'—':pvalue(r.p_conjunto_Holm)],
  pdm:['p Holm · DM–HLN',r=>r.p_DM_HLN_Holm===null?'—':pvalue(r.p_DM_HLN_Holm)],
  parametros:['Parámetros · pipeline',r=>num(r.parametros_pipeline,0)],
  latency:['Latencia p50 / p95 · ms',r=>num(r.latencia_mediana_ms,3)+' / '+num(r.latencia_p95_ms,3)],
  rss:['RSS observado · MiB',r=>num(r.RSS_MiB,1)],weights:['Parámetros · MiB',r=>num(r.parametros_MiB,2)],
  flops:['Millones FLOPs · red',r=>r.FLOPs_forward_medio===undefined?'—':num(r.FLOPs_forward_medio/1e6,3)],
  peak:['Pico global proceso · MiB',r=>r.pico_proceso_MiB===undefined?'—':num(r.pico_proceso_MiB,1)],
  energy:['Energía 15 / 57 W · J',r=>r.energia_15W_J===undefined?'—':num(r.energia_15W_J,3)+' / '+num(r.energia_57W_J,3)],
  gpu:['Medición GPU',r=>e(r.estado_GPU??'No disponible')],
  seedcost:['Semilla · coste',r=>r.semilla_coste===0?'Determinista':num(r.semilla_coste,0)],
  train:['Ajuste final registrado · s',r=>num(r.ajuste_final_registrado_s,2)+small(r.dispositivo_ajuste)]
 };
 const views={resumen:['modelo','desarrollo','mase','mae','rmse','p','parametros','latency','rss'],
  rendimiento:['modelo','entradas','desarrollo','mase','mae','rmse','sd','seeds','p','pdm'],
  coste:['modelo','parametros','weights','flops','latency','rss','peak','energy','gpu','seedcost','train']};
 const keys=view==='completa'?Object.keys(columns):views[view]??views.resumen;
 const groups=['Todos',...new Set(data.rows.map(r=>r.grupo))];
 return `<section class="section master-table" id="model-results-summary" aria-labelledby="model-summary-heading"><h3 id="model-summary-heading">Tabla maestra de rendimiento y coste</h3><p>Comparación de M4ₙ, diez arquitecturas del Top 10, cuatro baselines, un control numérico y dos referencias ingenuas. Prueba: los mismos 1.461 orígenes de 2022–2025. Coste: CPU de un hilo, lote uno, tres fechas prefijadas y 300 mediciones por procedimiento.</p><div class="control-row">${field('Grupo de procedimientos','master-group',select('master-group',groups.map(g=>[g,g]),state.masterGroup??'Todos'))}${field('Vista de la tabla','master-view',select('master-view',[['resumen','Resumen'],['rendimiento','Rendimiento e inferencia'],['coste','Coste y recursos'],['completa','Todos los campos']],view))}${field('Ordenar la tabla por','master-order',select('master-order',[['MASE_prueba','MASE de prueba'],['latencia_mediana_ms','Latencia mediana'],['RSS_MiB','Memoria RSS'],['parametros_pipeline','Parámetros'],['modelo','Nombre']],state.masterOrder??'MASE_prueba'))}${field('Sentido de ordenación','master-direction',select('master-direction',[['asc','Ascendente'],['desc','Descendente']],state.masterDirection??'asc'))}</div><p class="fine">${rows.length} de ${data.rows.length} procedimientos. Desplaza la tabla horizontalmente para consultar todas las columnas.</p>${table(keys.map(k=>columns[k][0]),rows.map(r=>keys.map(k=>columns[k][1](r))))}<p class="caption">Desarrollo identifica la validación de selección y su periodo; M4 utiliza parada interna de 2021. MASE y MAE promedian errores individuales de las semillas; RMSE promedia los RMSE individuales. IC95 % temporales puntuales: bootstrap estacionario de 49.999 réplicas, bloque medio de 14 días, condicionado a las semillas conservadas. La DE describe variación entre inicializaciones.</p><p class="caption">Significancia frente a M4: máximo de los nueve p ajustados de bootstrap, DM–HLN y GW para bloques de 14/28/56 días y HAC de 13/27/55 rezagos, con margen cero y Holm(306). Se exige rechazo en las nueve pruebas; GW es bilateral. Los p conjuntos repetidos reflejan el límite de resolución del bootstrap. — en M4: no se compara consigo mismo.</p><p class="caption">El coste utiliza una realización prefijada: semilla 101 en M4 y 1000 en cada comparador aprendido. Incluye preparación numérica y las operaciones del pipeline; M4 incluye expertos, M1 y memoria. Excluye carga de disco, construcción inicial de índices y generación de embeddings. Parámetros: total almacenado de los módulos, incluidos los inactivos. RSS: proceso con datos y librerías, observado cada 1 ms; no representa memoria exclusiva del modelo. Ajuste final: tiempos guardados de esa semilla; en M4 suma cinco componentes, sin búsqueda ni generación OOF. El hardware de ajuste es heterogéneo.</p><p class="caption">FLOPs: operaciones del forward de la red; pico global: proceso completo. Energía: dos escenarios de potencia constante, no consumo medido. Véase Eficiencia y Pareto para las convenciones.</p><div class="priority-links"><a class="download" href="./downloads/eficiencia_rendimiento_coste_completo.csv" download>Rendimiento y coste ampliado CSV ↓</a><a class="download" href="./downloads/tabla_maestra.csv" download>Tabla completa CSV ↓</a><a class="download" href="./downloads/tabla_maestra_coste_por_origen.csv" download>Coste por fecha CSV ↓</a><a class="download" href="./downloads/tabla_maestra_tiempos_inferencia.csv" download>5.400 mediciones CSV ↓</a><a class="download" href="./downloads/tabla_maestra_protocolo.json" download>Protocolo y alcance ↓</a><a href="${e(book)}10_evaluacion.html#tabla-maestra" target="_blank" rel="noopener">Método, resultados y código en el JBook ↗</a></div></section>`;
}

// Diagramas editables de las implementaciones congeladas; no ejecuta modelos.
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath,pathToFileURL} from 'node:url';
import {createRequire} from 'node:module';
import {createHash} from 'node:crypto';
const ROOT=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const OUT=path.join(ROOT,'results/esquemas_presentacion/ampliados');
const req=createRequire(import.meta.url);
const modulePath=process.env.ARTIFACT_NODE_MODULES
  ? req.resolve('@oai/artifact-tool',{paths:[process.env.ARTIFACT_NODE_MODULES]})
  : req.resolve('@oai/artifact-tool');
const {Presentation,PresentationFile}=await import(pathToFileURL(modulePath).href);
const C={ink:'#24313D',title:'#0B3D4E',line:'#414C59',data:'#A7DCD1',dataEdge:'#229D93',
 layer:'#DFEFD7',layerEdge:'#7AA75E',weight:'#FFE5C4',weightEdge:'#F49C25',op:'#F9CDCD',opEdge:'#E74D50',
 act:'#FFD0A0',actEdge:'#EA7D28',cal:'#FFF2C3',calEdge:'#E9B623',vector:'#E6E7E8',vectorEdge:'#919191',
 union:'#0B3D4E',unionEdge:'#0B3D4E',blue:'#E3EFFB',blueEdge:'#AAC7E9',
 green:'#E8F2DF',greenEdge:'#AED09A',purple:'#EFE6F7',purpleEdge:'#BCA1D4',yellow:'#FFF8DF',yellowEdge:'#E5CA69',
 frozen:'#DBE6F0',frozenEdge:'#7592AF'};
let serial=0;
function text(s,x,y,w,h,t,size=14,{bold=false,font='Calibri',color=C.ink,align='center',italic=false}={}){
 const v=s.shapes.add({geometry:'textbox',name:`texto-${++serial}`,position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 v.text=t;v.text.style={typeface:font,fontSize:size,bold,italic,color,alignment:align,verticalAlignment:'middle',wrap:'square',insets:0,autoFit:'none'};return v;
}
function box(s,x,y,w,h,t,kind='layer',size=14,{bold=false}={}){
 const v=s.shapes.add({geometry:'roundRect',name:`nodo-${++serial}`,position:{left:x,top:y,width:w,height:h},fill:C[kind],line:{fill:C[kind+'Edge']??C.line,width:1.3},borderRadius:6});
 if(t)text(s,x+5,y+3,w-10,h-6,t,size,{bold,color:kind==='union'?'#FFFFFF':C.ink});return v;
}
function circle(s,x,y,d,t,kind='op',size=21){
 const v=s.shapes.add({geometry:'ellipse',name:`circulo-${++serial}`,position:{left:x,top:y,width:d,height:d},fill:C[kind],line:{fill:C[kind+'Edge']??C.line,width:1.4}});
 text(s,x,y,d,d,t,size,{font:'Cambria Math'});return v;
}
function pipe(s,pts,{arrow=true,dashed=false,color=C.line,width=1.7}={}){
 for(let i=1;i<pts.length;i++){
  const [a,b]=[pts[i-1],pts[i]],dx=b[0]-a[0],dy=b[1]-a[1];
  s.shapes.add({geometry:'connector',kind:'straight',name:`conexion-${++serial}`,
   position:{left:Math.min(a[0],b[0]),top:Math.min(a[1],b[1]),width:Math.abs(dx),height:Math.abs(dy),horizontalFlip:dx<0,verticalFlip:dy<0},
   line:{fill:color,width,style:dashed?'dashed':'solid'},tail:{type:arrow&&i===pts.length-1?'triangle':'none'}});
 }
}
function frame(s,x,y,w,h,title){
 s.shapes.add({geometry:'roundRect',name:`marco-${++serial}`,position:{left:x,top:y,width:w,height:h},fill:'#FFFFFF',line:{fill:C.line,width:1.4,style:'dashed'},borderRadius:12});
 text(s,x+10,y+5,w-20,25,title,14,{bold:true});
}
function cards(s,items){
 const kinds=['blue','green','purple','yellow'];
 items.forEach(([title,formula,caption],i)=>{
  const x=36+228*i;box(s,x,397,216,108,'',kinds[i]);
  text(s,x+7,403,202,24,`${i+1}. ${title}`,13.5,{bold:true});
  text(s,x+7,429,202,43,formula,formula.split('\n').some(l=>l.length>29)?14:16.5,{font:'Cambria Math'});
  text(s,x+8,476,200,24,caption,11.2);
 });
}
function legend(s){
 const kinds=['data','layer','weight','op','act','cal','vector','union'];
 const labels=['Datos','Capa','Pesos','Operación','Activación','Calendario','Vector','Unión'];
 const xs=[360,420,480,547,628,707,800,868];
 kinds.forEach((k,i)=>{s.shapes.add({geometry:['data','op'].includes(k)?'ellipse':'rect',name:`leyenda-${++serial}`,position:{left:xs[i],top:95,width:12,height:12},fill:C[k],line:{fill:C[k+'Edge']??C.line,width:.8}});text(s,xs[i]+15,92,68,18,labels[i],10.5,{align:'left',color:'#5D6B74'});});
}
function base(p,title,sub){
 const s=p.slides.add();s.background.fill='#FFFFFF';
 text(s,48,27,864,52,title,36,{bold:true,font:'Cambria',color:C.title,align:'left'});
 text(s,48,86,302,29,sub,14,{align:'left',color:'#5D6B74'});legend(s);return s;
}
const facts=JSON.parse(await fs.readFile(path.join(OUT,'ficha_modelos.json'),'utf8'));
const fmt=n=>n.toLocaleString('es-CO').replaceAll('.',' ');
const subtitle=(k,v='N')=>`${v} · ${fmt(facts[k][v].parametros)} parámetros · MASE ${facts[k][v].MASE.toFixed(3).replace('.',',')}`;
function loss(){return ['Pérdida','L = mean |y/ℓ − r̂|','MAE en unidades normalizadas'];}

function tcn(p){
 let s=base(p,'TCN: arquitectura global',subtitle('tcn'));
 frame(s,36,124,888,261,'Tres bloques residuales causales y lectura con calendario conocido');
 box(s,52,196,104,58,'Historia\n168 h × 56','vector');
 [[180,'Bloque 1\n56 → 64 · d = 1\n2 Conv, k = 5'],[316,'Bloque 2\n64 → 64 · d = 4\nDropout 0,2'],[452,'Bloque 3\n64 → 64 · d = 16\n2 Conv, k = 5']].forEach(([x,t])=>box(s,x,185,112,80,t,'layer',13.2));
 box(s,590,201,80,48,'Último\nestado: 64','vector',13.5);box(s,700,201,70,51,'Unir\n128','union',14,{bold:true});
 box(s,800,193,104,68,'MLP\n128 → 64 → 24','layer',13.5);
 pipe(s,[[156,225],[180,225]]);pipe(s,[[292,225],[316,225]]);pipe(s,[[428,225],[452,225]]);pipe(s,[[564,225],[590,225]]);pipe(s,[[670,225],[700,225]]);pipe(s,[[770,225],[800,225]]);
 box(s,426,311,155,45,'F: 24 h × 23\nAplanar: 552','cal',13.5);box(s,610,311,95,45,'552 → 64\nReLU','layer',13.5);
 pipe(s,[[581,334],[610,334]]);pipe(s,[[705,334],[735,334],[735,252]]);
 circle(s,837,285,31,'1+');box(s,833,337,40,28,'×ℓ','weight',15);circle(s,892,334,32,'ŷ','purple',22);
 pipe(s,[[852,261],[852,285]]);pipe(s,[[852,316],[852,337]]);pipe(s,[[873,351],[892,350]]);
 text(s,56,276,345,22,'30 numéricas + 3 máscaras + 23 de calendario pasado',12,{italic:true});
 cards(s,[['Entrada','U = [X; Calₚ]','56 canales en la historia'],['Campo receptivo','R = 1 + 2(5−1)(1+4+16)\n= 169 h','Dos convoluciones por bloque'],['Calendario','c = ReLU(Wc vec F + bc)','Sólo calendario futuro conocido'],['Salida','ŷ = ℓ[1 + MLP(h₁₆₈;c)]','24 horas en una sola pasada']]);
 s=base(p,'TCN: detalle de celdas',subtitle('tcn'));
 frame(s,36,124,430,261,'Dentro de un bloque residual');frame(s,480,124,444,261,'Núcleo dilatado y alcance temporal');
 text(s,48,158,406,22,'S: proyección 1×1 si cambian los canales; identidad en otro caso',11.5,{italic:true});
 box(s,54,218,44,44,'h','vector',19);box(s,124,189,100,49,'Conv causal\nk = 5, d = dᵦ','layer',13);box(s,250,189,116,49,'Recorte derecho\ny ReLU','act',13);
 box(s,250,267,116,49,'Conv causal\nk = 5, d = dᵦ','layer',13);box(s,124,267,100,49,'Recorte, ReLU\ny dropout pᵦ','act',12.5);circle(s,61,284,32,'+');box(s,54,341,118,28,'ReLU → h siguiente','act',12.8);
 pipe(s,[[98,240],[111,240],[111,213],[124,213]]);pipe(s,[[224,213],[250,213]]);pipe(s,[[308,238],[308,267]]);pipe(s,[[250,292],[224,292]]);pipe(s,[[124,292],[93,300]]);pipe(s,[[76,262],[76,284]],{dashed:true});pipe(s,[[77,316],[77,341]]);
 text(s,197,339,245,32,'pᵦ = 0; 0,2; 0\nLa normalización de pesos se aplica a ambas Conv',12);
 ['t−16','t−12','t−8','t−4','t'].forEach((a,i)=>{const x=507+i*79;box(s,x,194,58,35,a,'vector',15);pipe(s,[[x+29,229],[x+29,252],[705,252],[705,268]],{arrow:i===4});});circle(s,688,268,34,'Σ');text(s,505,165,392,20,'Ejemplo: d = 4, cinco posiciones por filtro',12.5,{bold:true});
 text(s,500,314,406,26,'Dos convoluciones en cada una de las dilataciones 1, 4 y 16',13);text(s,500,345,406,27,'169 horas de alcance ≥ 168 horas de entrada',17,{font:'Cambria Math'});
 cards(s,[['Convolución','zₜ = Σᵢ₌₀⁴ Wᵢ xₜ₋dᵢ + b','Sólo posiciones presentes o pasadas'],['Recorte','p = (k−1)d = 4d','Se elimina el relleno derecho'],['Residuo','h′ = ReLU(S h + F(h))','Atajo y rama convolucional'],loss()]);
}

function itransformer(p){
 let s=base(p,'iTransformer: arquitectura global',subtitle('itransformer'));
 frame(s,36,124,888,261,'Una ficha por variable: atención entre canales');
 box(s,52,187,120,54,'Historia X\n168 h × 33','vector');box(s,52,297,120,54,'Calendario\n192 h × 23','cal');
 box(s,205,182,150,64,'Proyección compartida\n168 → 128\n33 fichas','layer',13.5);box(s,205,292,150,64,'Proyección compartida\n192 → 128\n23 fichas','layer',13.5);
 box(s,418,234,95,63,'Unir 56 fichas\n+ identidad','union',13.5,{bold:true});box(s,557,230,150,73,'Encoder × 2\n8 cabezas\nFF: 128 → 256 → 128','layer',13.5);
 box(s,742,241,86,51,'Token de\ndemanda: 128','vector',13);box(s,851,234,63,65,'Wₒ\n128 → 24','weight',15);circle(s,866,333,34,'ŷ','purple');
 pipe(s,[[172,214],[205,214]]);pipe(s,[[172,324],[205,324]]);pipe(s,[[355,214],[389,214],[389,251],[418,251]]);pipe(s,[[355,324],[389,324],[389,281],[418,281]]);pipe(s,[[513,266],[557,266]]);pipe(s,[[707,266],[742,266]]);pipe(s,[[828,266],[851,266]]);pipe(s,[[882,299],[882,333]]);text(s,793,309,70,20,'1+ y ×ℓ',12.5);
 text(s,52,360,710,19,'33 históricas = 30 variables numéricas + 3 máscaras; calendario pasado y futuro conocido',12,{italic:true});
 cards(s,[['Ficha histórica','eⱼ = W_H Xⱼ + pⱼ','Historia completa de una variable'],['Ficha de calendario','e_c = W_C [Calₚ,c; F_c] + p_c','168 h pasadas y 24 futuras'],['Atención','Z = Encoder(e₁,…,e₅₆)','Relaciona variables, no sólo horas'],['Salida','ŷ = ℓ(1 + Wₒ Z₀ + bₒ)','Sólo la ficha de demanda se lee']]);
 s=base(p,'iTransformer: detalle de celdas',subtitle('itransformer'));
 frame(s,36,124,430,261,'De una trayectoria a una ficha');frame(s,480,124,444,261,'Una capa del encoder: se repite dos veces');
 ['Demanda','Precio','Temperatura','Calendario'].forEach((label,i)=>{
  const y=183+i*49;text(s,49,y,98,32,label,13,{align:'left'});
  [0,1,2].forEach(j=>box(s,152+j*38,y,34,32,j===1?'…':j===0?'t−L':'t−1',i===3?'cal':'vector',11));
  box(s,292,y,51,32,i===3?'W_C':'W_H','weight',14);box(s,380,y,59,32,'e + p','layer',14);
  pipe(s,[[266,y+16],[292,y+16]]);pipe(s,[[343,y+16],[380,y+16]]);
 });text(s,48,370,406,12,'L = 168 para historia; 192 para calendario · 56 fichas de dimensión 128',10.5);
 box(s,518,191,55,40,'Z','vector',20);box(s,605,176,184,47,'Autoatención: 8 cabezas\nQ, K y V desde las 56 fichas','layer',13);circle(s,829,185,29,'+');box(s,788,235,106,31,'LayerNorm','layer',13);
 box(s,605,284,184,43,'Feed-forward con GELU\n128 → 256 → 128','layer',13);circle(s,829,291,29,'+');box(s,788,342,106,31,'LayerNorm','layer',13);
 pipe(s,[[573,211],[589,211],[589,199],[605,199]]);pipe(s,[[789,199],[829,199]]);pipe(s,[[844,214],[844,235]]);
 pipe(s,[[545,191],[545,161],[844,161],[844,185]],{dashed:true});
 pipe(s,[[788,250],[588,250],[588,305],[605,305]]);pipe(s,[[789,305],[829,305]]);pipe(s,[[844,266],[844,291]],{dashed:true});pipe(s,[[844,320],[844,342]]);text(s,491,345,270,28,'Dos capas y LayerNorm final\nDropout: 0,1',12);
 cards(s,[['Pesos de atención','A = softmax(QKᵀ / 4)','16 componentes por cabeza'],['Intercambio','H = Concat(A₁V₁,…,A₈V₈)W','56×56 relaciones por cabeza'],['Normalización','Z′ = LN(Z + MHA(Z))','Residuo y normalización posterior'],loss()]);
}

function relevance(s,{tg=false,tats=false}={}){
 box(s,51,178,130,49,'Media de X\núltimas 24 y 168 h','vector',12.5);
 box(s,215,178,130,49,'MLP de consulta\n66 → 64 → 64','layer',12.5);box(s,375,183,55,39,tg?'q + W_R e₀':'q','vector',tg?11.5:18);
 pipe(s,[[181,202],[215,202]]);pipe(s,[[345,202],[375,202]]);
 box(s,51,287,90,48,'E5: eⱼ\n768','frozen',14);box(s,191,258,70,38,'K\n768 → 64','weight',12);box(s,191,324,70,36,'V\n768 → 64','weight',12);
 box(s,309,245,124,62,'qᵀ K eⱼ / 8\nmáscara disponible\nsoftmax','act',12.5);box(s,309,334,124,33,tats?'Suma ponderada: 64':'z: suma ponderada 64','layer',12);
 pipe(s,[[141,311],[164,311],[164,277],[191,277]]);pipe(s,[[141,311],[164,311],[164,342],[191,342]]);
 pipe(s,[[261,277],[309,277]]);pipe(s,[[402,222],[402,245]]);pipe(s,[[371,307],[371,334]]);pipe(s,[[261,342],[309,350]]);
 text(s,49,370,399,13,tats?'Después: MLP 64 → 64 → 32 → 12; cero exacto si m = 0':'Sin documentos: m = 0 y z = 0; con documentos: m = 1',10.8);
}
function tats(p){
 let s=base(p,'TaTS: arquitectura global',subtitle('tats','N+O'));
 frame(s,36,124,888,261,'El texto se convierte en canales temporales auxiliares');
 box(s,52,183,110,54,'Historia X\n168 h × 33','vector',14);box(s,475,179,86,65,'Concatenar\n46 canales','union',13,{bold:true});box(s,601,174,150,82,'PatchTST\n21 parches por canal\nd = 128 · 3 capas\n16 cabezas','layer',13);
 box(s,791,178,114,47,'Lineal de demanda\n2688 → 24','layer',12.5);box(s,791,279,114,48,'MLP de contexto\n6601 → 128 → 24','layer',12.5);circle(s,833,244,31,'Σ');circle(s,886,242,35,'ŷ','purple');
 pipe(s,[[162,210],[475,210]]);pipe(s,[[561,210],[601,210]]);pipe(s,[[751,210],[771,210],[771,201],[791,201]]);pipe(s,[[751,229],[772,229],[772,303],[791,303]]);
 pipe(s,[[848,225],[848,244]]);pipe(s,[[848,279],[848,275]]);pipe(s,[[864,259],[886,259]]);text(s,868,276,53,18,'1+, ×ℓ',11.5);
 box(s,52,300,112,49,'Documentos\n7 cierres × 72 h','vector',13);box(s,194,300,100,49,'E5 congelado\n768','frozen',13);box(s,323,300,119,49,'Relevancia: 64\nMLP → 12','layer',13);box(s,475,300,100,49,'12 + presencia\nRepetir 24 h','act',12.5);
 pipe(s,[[164,324],[194,324]]);pipe(s,[[294,324],[323,324]]);pipe(s,[[442,324],[475,324]]);pipe(s,[[525,300],[525,244]]);
 pipe(s,[[181,210],[181,270],[382,270],[382,300]],{dashed:true});text(s,190,248,177,19,'Consulta desde X',11.5,{italic:true});
 box(s,609,316,139,39,'F: 552 · Calₚ diario: 161','cal',12);pipe(s,[[748,336],[772,336],[772,316],[791,316]]);
 text(s,51,366,851,17,'Control N: 33 canales, d = 64, 2 capas y 4 cabezas; 466 480 parámetros, sin canales documentales',11.2,{italic:true});
 cards(s,[['Relevancia','α(d,j) = softmaxⱼ\n(qᵀ K e(d,j) / 8)','Hasta 64 documentos por cierre'],['Resumen diario','u_d = MLP(Σⱼ α(d,j)V e(d,j))','Dimensiones: 64 → 64 → 32 → 12'],['Canales de texto','T = repetir₂₄([m_d u_d; m_d])','13 canales adicionales, cero si falta texto'],['Lectura','r̂ = 1 + head(Z₀) + g(c)','Demanda y contexto multicanal']]);
 s=base(p,'TaTS: detalle de celdas',subtitle('tats','N+O'));
 frame(s,36,124,430,261,'Relevancia aprendida en cada cierre diario');frame(s,480,124,444,261,'Dos vías para obtener las 24 salidas');relevance(s,{tats:true});
 box(s,516,180,105,47,'Z: 46 canales\n21 × 128 cada uno','vector',12.5);box(s,658,177,173,47,'Canal de demanda Z₀\nAplanar: 2688 → 24','layer',13);circle(s,870,236,31,'+');circle(s,870,349,35,'ŷ','purple');
 box(s,516,281,112,41,'Media de parches\n46 × 128 = 5888','vector',12);box(s,516,336,112,30,'Calendario: 713','cal',12);box(s,669,292,90,53,'Concatenar\n6601','union',12.5,{bold:true});box(s,791,285,114,53,'MLP de contexto\n6601 → 128 → 24','layer',12.5);
 pipe(s,[[621,203],[658,203]]);pipe(s,[[569,227],[569,281]]);pipe(s,[[628,302],[669,302]]);pipe(s,[[628,351],[650,351],[650,332],[669,332]]);pipe(s,[[759,318],[791,318]]);
 pipe(s,[[831,201],[885,201],[885,236]]);pipe(s,[[848,285],[848,277],[885,277],[885,267]]);pipe(s,[[901,251],[916,251],[916,366],[905,366]]);text(s,767,353,101,20,'1+ suma, ×ℓ',12);
 cards(s,[['Consulta numérica','q = MLP([X̄₂₄; X̄₁₆₈])','Sólo historia anterior al origen'],['Agregación','z_d = V Σⱼ α(d,j)e(d,j)','K y V: 768 → 64'],['Contexto','c = [media(Z); vec F; Calₚ]','5888 + 552 + 161 = 6601'],loss()]);
}

function fusion(p){
 let s=base(p,'Fusión tardía: arquitectura global',subtitle('fusion','N+O'));
 frame(s,36,124,888,261,'La combinación se realiza después de los pronósticos de ambas ramas');
 box(s,52,183,109,54,'Historia X\n168 h × 33','vector',14);box(s,205,174,180,65,'Núcleo PatchTST\nd = 64 · 2 capas\n4 cabezas','layer',13.5);box(s,438,183,123,50,'b(X,F): 24 h\n1 + corrección','vector',13);circle(s,826,220,34,'+');box(s,824,285,40,30,'×ℓ','weight',15);circle(s,826,340,34,'ŷ','purple');
 pipe(s,[[161,210],[205,210]]);pipe(s,[[385,208],[438,208]]);pipe(s,[[561,208],[843,208],[843,220]]);pipe(s,[[843,254],[843,285]]);pipe(s,[[843,315],[843,340]]);
 box(s,205,260,180,28,'Calendario F y Calₚ','cal',13);pipe(s,[[295,260],[295,239]]);
 box(s,52,307,109,45,'Documentos\n72 h · máximo 64','vector',12.5);box(s,205,307,100,45,'E5 congelado\n768','frozen',12.5);box(s,346,307,128,45,'Relevancia\nz: 64','layer',13);box(s,515,307,128,45,'MLP textual\n64 → 128 → 24','layer',13);box(s,689,307,84,45,'× α · m','act',18);box(s,690,250,82,31,'α ∈ ℝ','weight',18);
 pipe(s,[[161,329],[205,329]]);pipe(s,[[305,329],[346,329]]);pipe(s,[[474,329],[515,329]]);pipe(s,[[643,329],[689,329]]);pipe(s,[[731,281],[731,307]]);pipe(s,[[773,329],[798,329],[798,237],[826,237]]);
 pipe(s,[[180,210],[180,297],[411,297],[411,307]],{dashed:true});text(s,460,361,325,19,'Ambas ramas aprenden conjuntamente',12,{italic:true});
 cards(s,[['Rama numérica','b = f_num(X,F)','Núcleo sin acceso a documentos'],['Rama textual','u = MLP₆₄→₁₂₈→₂₄(z)','Relevancia y calendario van por vías distintas'],['Combinación','r̂ = b + α m u','α escalar aprendido; no mezcla convexa'],['Ausencia','m = 0  ⇒  r̂ = b','Aporte documental exactamente cero']]);
 s=base(p,'Fusión tardía: detalle de celdas',subtitle('fusion','N+O'));
 frame(s,36,124,430,261,'Relevancia de documentos del origen');frame(s,480,124,444,261,'Rama textual y combinación de las salidas');relevance(s);
 box(s,519,181,72,40,'z: 64','vector',15);box(s,620,175,100,52,'64 → 128\nGELU','layer',14);box(s,754,181,94,40,'Dropout 0,1','act',13);
 box(s,754,261,94,51,'128 → 24\nW y b: inicio 0','layer',12);box(s,620,269,91,43,'× α · m','act',17);box(s,519,272,73,39,'α m u','vector',15);box(s,519,341,130,28,'Pronóstico numérico b','vector',12);circle(s,703,340,30,'+');box(s,767,341,37,28,'×ℓ','weight',14);circle(s,836,337,35,'ŷ','purple');
 pipe(s,[[591,201],[620,201]]);pipe(s,[[720,201],[754,201]]);pipe(s,[[801,221],[801,261]]);pipe(s,[[754,287],[711,287]]);pipe(s,[[620,290],[592,290]]);pipe(s,[[556,311],[556,326],[718,326],[718,340]]);pipe(s,[[649,355],[703,355]]);pipe(s,[[733,355],[767,355]]);pipe(s,[[804,355],[836,355]]);
 text(s,497,234,228,24,'α inicia en 0,5 y se aprende sin acotación',11.5,{italic:true});
 cards(s,[['Relevancia','αⱼ = softmaxⱼ(qᵀ K eⱼ / 8)','Pesos sólo entre documentos habilitados'],['Vector documental','z = V Σⱼ αⱼ eⱼ','q depende de los promedios de X'],['Rama textual','h = Drop(GELU(W₁z+b₁))\nu = W₂ h + b₂','Se aprende junto con la rama numérica'],loss()]);
}

function tgforecaster(p){
 let s=base(p,'TGForecaster: arquitectura global',subtitle('tgforecaster','N+O'));
 frame(s,36,124,888,267,'Consultas del horizonte guiadas por texto sobre la memoria de la historia');
 box(s,57,187,101,55,'Historia X\n168 h × 33','vector',14);box(s,194,182,119,65,'21 parches\n16 h, paso 8\nSin RevIN','layer',13);box(s,343,182,124,65,'Encoder × 2\nd = 64\n4 cabezas','layer',13);
 box(s,505,179,112,102,'Memoria\n693 parches +\n7 tokens de cal.\n700 × 64','union',13,{bold:true});box(s,671,182,119,79,'Mezcla × 1\nAtención cruzada\nQ del horizonte\nK,V de memoria','layer',12.5);box(s,833,185,83,69,'Lectura\n64 → 8\n5 parches','layer',13);
 pipe(s,[[158,214],[194,214]]);pipe(s,[[313,214],[343,214]]);pipe(s,[[467,214],[505,214]]);pipe(s,[[617,220],[671,220]]);pipe(s,[[790,220],[833,220]]);
 box(s,59,293,351,48,'33 descripciones E5: 768 → 64\nCalendario pasado: 7 días × 23 → 64','frozen',13);pipe(s,[[410,317],[479,317],[479,287],[561,287],[561,281]]);
 box(s,620,309,165,58,'Q: 5 consultas de 64\nDescripción + Fₖ + posición\n+ proyección del texto','cal',12.7);pipe(s,[[702,309],[702,261]]);
 box(s,58,350,140,26,'Documentos: 72 h','vector',12.5);box(s,226,350,114,26,'E5 congelado','frozen',12.5);box(s,375,350,147,26,'Relevancia → z: 64','layer',12.5);
 pipe(s,[[198,363],[226,363]]);pipe(s,[[340,363],[375,363]]);pipe(s,[[522,363],[597,363],[597,338],[620,338]]);
 box(s,833,288,83,47,'Solape\n5 × 8 → 24','layer',13);circle(s,856,346,32,'ŷ','purple');pipe(s,[[874,254],[874,288]]);pipe(s,[[874,335],[874,346]]);text(s,786,350,62,20,'1+, ×ℓ',12);
 text(s,48,379,852,11,'N conserva las descripciones: d = 256, 3 capas temporales, 2 de mezcla, 8 cabezas; 2 899 072 parámetros',10.6,{italic:true});
 cards(s,[['Memoria','M = [Z_c + W_D e_c; Calₚ]','33×21 parches y siete tokens diarios'],['Consulta del horizonte','Qₖ = W_D e₀ + W_F Fₖ\n+ pₖ + m W_T z','Fₖ: ocho horas de calendario conocido'],['Mezcla','H = softmax(Q K_Mᵀ / 4)V_M','5 consultas sobre 700 tokens de memoria'],['Reconstrucción','ŷ_h = ℓ(1 + Σₖ ω(k,h)o(k,h))','Sólo se suman parches que cubren h']]);
 s=base(p,'TGForecaster: detalle de celdas',subtitle('tgforecaster','N+O'));
 frame(s,36,124,430,261,'Identidad de canales y consultas');frame(s,480,124,444,261,'Atención cruzada y solape del horizonte');
 box(s,52,180,130,44,'Z: 33 × 21 × 64','vector',13);box(s,223,180,140,44,'W_D e_c: 33 × 64\nRepetir por parche','frozen',12.2);circle(s,392,186,29,'+');box(s,237,246,184,34,'693 tokens con identidad de canal','vector',12);
 pipe(s,[[182,202],[200,202],[200,170],[407,170],[407,186]]);pipe(s,[[363,201],[392,201]]);pipe(s,[[406,215],[406,246]]);
 box(s,52,281,131,37,'Calₚ: 7 × 23\nProyección a 64','cal',12.2);box(s,299,301,123,40,'Memoria\n700 × 64','union',13,{bold:true});pipe(s,[[330,280],[330,301]]);pipe(s,[[183,299],[247,299],[247,321],[299,321]]);
 box(s,52,340,210,38,'e₀ + Fₖ + posición + texto\nFₖ: ocho horas, paso cuatro','cal',12);box(s,300,352,122,27,'Q: 5 × 64','vector',13.5);pipe(s,[[262,359],[300,365]]);
 box(s,518,178,66,42,'Q\n5 × 64','vector',13);box(s,518,246,66,57,'M\n700 × 64','vector',12);box(s,618,180,49,38,'W_Q','weight',15);box(s,618,252,49,46,'W_K\nW_V','weight',15);box(s,710,178,184,86,'Atención cruzada\n4 cabezas\nQ: 5 · K,V: 700','layer',14);
 pipe(s,[[584,199],[618,199]]);pipe(s,[[667,199],[710,199]]);pipe(s,[[584,274],[618,274]]);pipe(s,[[667,275],[688,275],[688,240],[710,240]]);
 box(s,710,290,184,44,'Residuo + LayerNorm\nFF: 64 → 128 → 64','layer',13);box(s,518,337,125,35,'5 parches de 8','vector',13);box(s,710,351,184,25,'Pesos ω(k,h) aprendidos','weight',12.4);
 pipe(s,[[802,264],[802,290]]);pipe(s,[[710,312],[664,312],[664,354],[643,354]]);pipe(s,[[643,356],[710,363]]);
 cards(s,[['Parches de horizonte','Iₖ = {4k,…,4k+7}\nk = 0,…,4','Cinco parches superpuestos de ocho horas'],['Relevancia documental','q_rel = MLP([X̄₂₄;X̄₁₆₈])\n+ W_R e₀','Proyección distinta de la identidad W_D'],['Peso por hora','ω(k,h) = softmaxₖ a(k,h)\nΣₖ:h∈Iₖ ω(k,h) = 1','El peso se normaliza entre coberturas'],loss()]);
}

function patchtst(p){
 let s=base(p,'PatchTST: arquitectura global',subtitle('patchtst'));
 frame(s,36,124,888,261,'Codificación compartida por canal y dos vías de lectura');
 box(s,52,198,110,65,'Historia X\n168 h × 33\nSin documentos','vector',13.5);
 box(s,205,191,132,79,'21 parches\n16 h, paso 8\nRelleno final: 8','layer',14);
 box(s,386,186,156,91,'Encoder × 2\nd = 64 · 4 cabezas\nFF: 64 → 128 → 64\nDropout: 0,1','layer',13);
 box(s,602,177,172,54,'Canal de demanda\n1344 → 24','layer',14);
 box(s,601,286,173,56,'Media de los 33 canales\n+ calendario: 2825\nMLP: 2825 → 128 → 24','layer',12.8);
 circle(s,839,242,33,'+');circle(s,839,340,34,'ŷ','purple');
 pipe(s,[[162,231],[205,231]]);pipe(s,[[337,231],[386,231]]);
 pipe(s,[[542,209],[570,209],[570,204],[602,204]]);pipe(s,[[542,253],[570,253],[570,314],[601,314]]);
 pipe(s,[[774,204],[856,204],[856,242]]);pipe(s,[[774,314],[856,314],[856,275]]);
 pipe(s,[[872,259],[898,259],[898,357],[873,357]]);text(s,785,318,106,19,'1+ suma, ×ℓ',12);
 box(s,240,311,240,40,'Calendario: F (552) y Calₚ diario (161)','cal',13);pipe(s,[[480,331],[601,331]]);
 text(s,52,357,650,19,'30 variables numéricas y tres máscaras; sin normalización RevIN interna',12,{italic:true});
 cards(s,[['Segmentación','P_c = patch₁₆,₈(X_c)','21 segmentos por variable histórica'],['Atención temporal','Z_c = Encoder(P_c W + p)','Los pesos se comparten entre canales'],['Contexto','c = [media(Z); vec F; Calₚ]','2112 + 552 + 161 = 2825'],['Pronóstico','ŷ = ℓ[1 + head(Z₀) + g(c)]','Demanda y contexto aportan 24 salidas']]);
 s=base(p,'PatchTST: detalle de celdas',subtitle('patchtst'));
 frame(s,36,124,430,261,'Dentro de cada capa del encoder');frame(s,480,124,444,261,'Lectura de demanda y de contexto');
 box(s,52,179,68,42,'E: 21 × 64','vector',12);box(s,155,173,177,54,'Atención temporal\n4 cabezas · 16 por cabeza','layer',13);circle(s,383,183,30,'+');
 box(s,348,241,102,32,'BatchNorm','layer',13);box(s,155,287,177,52,'FF con GELU\n64 → 128 → 64','layer',13);circle(s,383,299,30,'+');box(s,348,348,102,26,'BatchNorm','layer',12);
 pipe(s,[[120,200],[155,200]]);pipe(s,[[332,200],[383,200]]);pipe(s,[[85,179],[85,161],[398,161],[398,183]],{dashed:true});pipe(s,[[398,213],[398,241]]);
 pipe(s,[[348,257],[135,257],[135,313],[155,313]]);pipe(s,[[332,313],[383,313]]);pipe(s,[[398,273],[398,299]],{dashed:true});pipe(s,[[398,329],[398,348]]);
 box(s,516,180,112,48,'Z: 33 canales\n21 × 64 cada uno','vector',12.5);box(s,666,177,168,46,'Demanda Z₀\nAplanar 1344 → 24','layer',13);circle(s,871,232,30,'+');
 box(s,516,285,112,43,'Media de parches\n33 × 64 = 2112','vector',12);box(s,516,343,112,28,'Calendario: 713','cal',12);
 box(s,667,293,88,51,'Concatenar\n2825','union',12.5,{bold:true});box(s,787,287,120,52,'MLP de contexto\n2825 → 128 → 24','layer',12.5);circle(s,869,351,34,'ŷ','purple');
 pipe(s,[[628,204],[666,204]]);pipe(s,[[572,228],[572,285]]);pipe(s,[[628,306],[667,306]]);pipe(s,[[628,357],[649,357],[649,331],[667,331]]);pipe(s,[[755,319],[787,319]]);
 pipe(s,[[834,200],[886,200],[886,232]]);pipe(s,[[847,287],[847,274],[886,274],[886,262]]);pipe(s,[[901,247],[916,247],[916,368],[903,368]]);text(s,769,353,95,19,'1+ suma, ×ℓ',12);
 cards(s,[['Atención','A = softmax(QKᵀ / 4)','21×21 relaciones en cada cabeza'],['Residuo y escala','Z′ = BN(Z + Drop(MHA(Z)))','Luego FF, residuo y BatchNorm'],['Lectura multicanal','g(c) = W₂ Drop(GELU(W₁c+b₁))+b₂','MLP de contexto con ancho fijo 128'],loss()]);
}

async function main(){
 await fs.mkdir(OUT,{recursive:true});
 const p=Presentation.create({slideSize:{width:960,height:540}});
 tcn(p);itransformer(p);tats(p);fusion(p);tgforecaster(p);patchtst(p);
 const models=['tcn','itransformer','tats','fusion','tgforecaster','patchtst'];
 for(let i=0;i<p.slides.items.length;i++){
  const s=p.slides.items[i];
  s.speakerNotes.text=`Fuente: implementación conservada del proyecto y results/evaluacion_e2/benchmark.csv. Familia: ${models[Math.floor(i/2)]}. Dimensiones y variantes en results/esquemas_presentacion/ampliados/ficha_modelos.json. Los diagramas describen los pesos seleccionados sin modificar entrenamiento o evaluación.`;
 }
 await p.inspect({select:'$.slides[*]',include:['id','index','elements.id','elements.kind','elements.text','elements.position'],fileName:path.join(OUT,'contenido.ndjson')});
 const candidate=path.join(OUT,'borrador.pptx');await(await PresentationFile.exportPptx(p)).save(candidate);
 const skill=process.env.PRESENTATIONS_SKILL_DIR;
 if(skill){
  const {finalizePresentation}=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
  const stamp=new Date().toISOString().replaceAll(/[:.]/g,'-');
  const verified=path.join(OUT,'verificados',stamp);await fs.mkdir(verified,{recursive:true});
  const result=await finalizePresentation({workspaceDir:ROOT,candidatePath:candidate,finalPath:path.join(verified,'esquemas_ampliados.pptx'),
   pythonExecutable:process.env.ARTIFACT_PYTHON,
   integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
   layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
   layoutArgs:['--expected-slide-size-emu','9144000,5143500','--validate-heading-fit'],
   explicitTotalSlideCount:12,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],
   fontPolicy:{basis:'reference',families:['Cambria','Calibri','Cambria Math'],referencePath:path.join(ROOT,'results/esquemas_presentacion/fuente_original.pptx'),referenceSha256:createHash('sha256').update(await fs.readFile(path.join(ROOT,'results/esquemas_presentacion/fuente_original.pptx'))).digest('hex')},
   verifyArtifactToolImport:true,receiptPath:path.join(OUT,`validacion_${stamp}.json`)});
  await fs.copyFile(result.finalPath,path.join(OUT,'esquemas_ampliados.pptx'));
  console.log(JSON.stringify({finalPath:result.finalPath,receiptPath:result.receiptPath}));
 }else{await(await PresentationFile.exportPptx(p)).save(path.join(OUT,'esquemas_ampliados.pptx'));}
 console.log('Doce esquemas editables creados; seis familias; sin entrenamiento ni publicación.');
}
main().catch(e=>{console.error(e.message);process.exitCode=1;});

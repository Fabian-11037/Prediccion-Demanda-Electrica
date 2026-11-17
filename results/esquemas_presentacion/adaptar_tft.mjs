import fs from 'node:fs/promises';
import {createRequire} from 'node:module';
import {pathToFileURL} from 'node:url';
const req=createRequire(import.meta.url);
const lib=req.resolve('@oai/artifact-tool',{paths:['C:/Program Files/WindowsApps/OpenAI.CodexPrimaryRuntime.v26-1112-737-0_26.1112.737.0_x64__3k8sg7r9htsxt/dependencies/node/node_modules']});
try {
 const {PresentationFile,FileBlob}=await import(pathToFileURL(lib).href);
 const p=await PresentationFile.importPptx(await FileBlob.load('C:/Users/Work/Downloads/Esquemas_arquitectura_modelos_Entregable2.pptx'));
 const changes=[
 ['sh/547mhg3m','TFT compacto','TFT'],['sh/8z2h8bq1','TFT compacto','TFT'],
 ['sh/k3y5ov21','T02 · 281 443 parámetros · MASE 1,147','T01 · 11 715 596 parámetros · MASE 1,051'],
 ['sh/9kby1g7m','T02 · 281 443 parámetros · MASE 1,147','T01 · 11 715 596 parámetros · MASE 1,051'],
 ['sh/ovmdkjud','83 variables','56 variables'],['sh/d4jado72','83 variables','56 variables'],
 ['sh/qtcjq1gv','Tamaño oculto 16','Tamaño oculto 160'],
 ['sh/ipcjm1gj','pasa a 16','pasa a 160'],['sh/b2xgfydo','Lineal 16 → 1','Lineal 160 → 1']];
 for(const [id,from,to] of changes)p.resolve(id).text.replace(from,to);
 const im=p.resolve('im/0ruls3i1'); const frame=im.frame;
 im.replace({blob:new Uint8Array(await fs.readFile('results/esquemas_presentacion/embedding_tft160.png')),contentType:'image/png'});im.frame=frame;
 await p.inspect({select:'$.slides[14,15]',include:['id','index','elements.id','elements.kind','elements.text','elements.position'],fileName:'results/esquemas_presentacion/tft_adaptado.ndjson'});
 const ppt=await PresentationFile.exportPptx(p);await ppt.save('results/esquemas_presentacion/adaptacion_tft.pptx');
 console.log('TFT adaptado: 56 entradas, ancho160, 11715596 parámetros, MASE1.051.');
} catch(e){console.error(e.message);process.exitCode=1;}

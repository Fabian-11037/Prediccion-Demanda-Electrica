import fs from 'node:fs/promises';
import {createRequire} from 'node:module';
import {pathToFileURL} from 'node:url';
const req=createRequire(import.meta.url);
const lib=req.resolve('@oai/artifact-tool',{paths:['C:/Program Files/WindowsApps/OpenAI.CodexPrimaryRuntime.v26-1112-737-0_26.1112.737.0_x64__3k8sg7r9htsxt/dependencies/node/node_modules']});
try {
const {PresentationFile,FileBlob}=await import(pathToFileURL(lib).href);
const p=await PresentationFile.importPptx(await FileBlob.load('C:/Users/Work/Downloads/Esquemas_arquitectura_modelos_Entregable2.pptx'));
const snap=await p.inspect({fileName:'results/esquemas_presentacion/inspeccion_tft.ndjson',select:'$.slides[14,15]',include:['id','index','elements.id','elements.kind','elements.text','elements.alt','elements.position']});
await fs.writeFile('results/esquemas_presentacion/inspeccion_tft.json',JSON.stringify(snap,null,2));
console.log(JSON.stringify(snap).slice(0,11000));

} catch(e){console.error(e.message);process.exitCode=1;}

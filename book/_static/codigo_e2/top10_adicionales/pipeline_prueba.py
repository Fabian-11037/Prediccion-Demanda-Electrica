"""Inferencia causal de checkpoints: un origen por llamada."""
from pathlib import Path
import json,numpy as np,torch
from entrenamiento.datos import DatosNumericos
from entrenamiento.protocolo_e2 import _escalas
from entrenamiento.redes_e2 import construir

def predecir_checkpoint(raiz_datos,carpeta_checkpoint,familia,origenes,dispositivo="cpu"):
    carpeta=Path(carpeta_checkpoint)
    meta=json.loads((carpeta/"resultado.json").read_text(encoding="utf8"))
    escalas=json.loads((carpeta/"escaladores.json").read_text(encoding="utf8"))
    datos=DatosNumericos(Path(raiz_datos))
    posiciones=datos.indice.get_indexer(origenes)
    assert (posiciones>=168).all()
    entradas=datos.construir(posiciones,_escalas(escalas,datos))
    red=construir(familia,meta["configuracion"],33,23,168,24,None,None).to(dispositivo)
    red.load_state_dict(torch.load(carpeta/"modelo.pt",map_location=dispositivo,weights_only=True),strict=True)
    red.eval();red.requires_grad_(False)
    resultados=[]
    with torch.inference_mode():
        for i in range(len(origenes)):
            args=[torch.from_numpy(a[i:i+1]).to(dispositivo) for a in
                  [entradas.x,entradas.cal_p,entradas.cal_f,entradas.dia]]
            pred=red(*args).cpu().numpy().astype(float)[0]*entradas.nivel[i]
            resultados.append(pred)
    pred=np.stack(resultados)
    error=pred-entradas.y
    metricas={"MASE":float(np.abs(error).mean()/escalas["q168_kwh"]),
              "MAE":float(np.abs(error).mean()),"RMSE":float(np.sqrt((error**2).mean()))}
    return pred,metricas

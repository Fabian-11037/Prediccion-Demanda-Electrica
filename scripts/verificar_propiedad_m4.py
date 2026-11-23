"""Verificación numérica local con memoria fija; no reentrena ni modifica pesos."""
from pathlib import Path
import sys,json
from e3_modelo4 import *
from m4_onnx_congelado import Encoder,Corrector,Q

def ejecutar():
    torch.set_num_threads(4);out=REPO/'results/m4_cierre_gpu_20261123';s=101
    ck=cargar(REPO/f'results/m4_gpu_20261122_194500/final/M4_N/semilla_{s}/modelo.pt')
    red=Memoria(ck['n_num'],ck['n_feat'],ck['n_fuentes'],ck['configuracion'],'M4_N').eval();red.load_state_dict(ck['modelo'])
    enc=Encoder(red).eval();cor=Corrector(red).eval();filas=[];rng=np.random.default_rng(20261123)
    with np.load(out/f'pruebas_congeladas/semilla_{s}/entradas_portables.npz') as z:d={k:z[k] for k in z.files}
    with torch.no_grad():
        qs,ks=enc(torch.from_numpy(d['num']))
        for i in d['indices'][::37]:
            edad=d['dias'][i]-d['dias'];ids=np.flatnonzero((edad>=2)&(edad<=365));k=torch.zeros(365,32);r=torch.zeros(365,24);dd=torch.zeros(365,dtype=torch.long);valid=torch.zeros(365,dtype=torch.bool)
            k[:len(ids)]=ks[ids];r[:len(ids)]=torch.from_numpy(d['residuos'][ids]);dd[:len(ids)]=torch.from_numpy(d['dias'][ids]);valid[:len(ids)]=True
            q=qs[i:i+1];scores=(q@k.T)[0]/np.sqrt(32);v=scores[valid].sort(descending=True).values
            if len(v)<17:continue
            margen=float(v[15]-v[16]);K=float(k.norm(dim=1).max()/np.sqrt(32))
            if margen<=1e-7 or K==0:continue
            norm=min(1e-4,margen/(4*K));dir=torch.from_numpy(rng.normal(size=(1,32)).astype('float32'));delta=dir/dir.norm()*norm
            base=torch.from_numpy(d['base'][i:i+1]);ori=torch.tensor([d['dias'][i]])
            p=cor(q,k,r,dd,valid,ori,base);p2=cor(q+delta,k,r,dd,valid,ori,base)
            assert set(p[2][0].tolist())==set(p2[2][0].tolist())
            R=float(r.abs().max());Lg=.25*1.13*float(torch.linalg.matrix_norm(cor.compuerta[0].weight,ord=2))*float(torch.linalg.matrix_norm(cor.compuerta[-1].weight,ord=2))
            Lz=1+K+2*K*(1+3*R**2/(2*1e-6));cota=(2*R*K+R*Lg*Lz)*norm
            cambio=float((p2[0]-p[0]).abs().max()/Q);assert cambio<=cota+1e-5
            filas.append(dict(indice=int(i),margen=margen,norma_delta=norm,cambio_normalizado=cambio,cota=cota))
    assert filas
    guardar(out/'VERIFICACION_PROPIEDAD_LOCAL.json',dict(semilla=s,casos=len(filas),resultado='verificado numéricamente en los casos registrados',alcance='Banco, base y conjunto top-k fijos. Cota holgada; no garantía global ni de precisión. Tolerancia numérica1e-5 para red FP32.',filas=filas))
if __name__=='__main__':ejecutar()

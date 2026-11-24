"""Componentes tensoriales de M4_N; base M1 externa y pesos históricos intactos."""
import copy
import torch
from torch import nn

Q = 215617.90361589016


class Encoder(nn.Module):
    def __init__(self, original):
        super().__init__()
        n = original.consulta_texto.in_features
        self.q = nn.Sequential(nn.Linear(n, 32), copy.deepcopy(original.proy_q[1]))
        self.k = nn.Sequential(nn.Linear(n, 32), copy.deepcopy(original.proy_k[1]))
        with torch.no_grad():
            for target, source in [(self.q, original.proy_q), (self.k, original.proy_k)]:
                target[0].weight.copy_(source[0].weight[:, :n])
                target[0].bias.copy_(source[0].bias)

    def forward(self, num):
        return self.q(num), self.k(num)


class Corrector(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.compuerta = copy.deepcopy(original.compuerta)
        self.edad_max = original.cfg['edad_maxima_dias']
        self.k = original.cfg['k_vecinos']

    def forward(self, consulta, claves, residuos, fechas, validas, origen, base):
        edad = origen.reshape(1, 1) - fechas.reshape(1, -1)
        valido = validas.reshape(1, -1) & (edad >= 2) & (edad <= self.edad_max)
        score = (consulta @ claves.T) / (32 ** .5)
        score = score.masked_fill(~valido, -1e9)
        valor, ids = score.topk(self.k, -1)
        ok = valido.gather(1, ids)
        hay = ok.any(-1)
        alfa = valor.softmax(-1) * ok
        alfa = alfa / alfa.sum(-1, keepdim=True).clamp_min(1e-8)
        # Descartar valores no admisibles ANTES de reducir, también con NaN/Inf.
        rr = torch.where(ok[:, :, None], residuos[ids], torch.zeros_like(residuos[ids]))
        media = (alfa[:, :, None] * rr).sum(1)
        dispersion = ((rr - media[:, None]).square() * alfa[:, :, None]).sum(1).mean(-1).clamp_min(1e-12).sqrt()
        apoyo = ok.sum(-1).to(consulta.dtype) / self.k
        maxsim = torch.where(hay, valor[:, 0], torch.zeros_like(valor[:, 0]))
        antig = (alfa * edad.gather(1, ids).to(consulta.dtype)).sum(-1) / self.edad_max
        feat = torch.cat([consulta, maxsim[:, None], dispersion[:, None], apoyo[:, None], antig[:, None]], 1)
        gate = torch.sigmoid(self.compuerta(feat))[:, 0] * hay
        pred = torch.where(hay[:, None], base + Q * gate[:, None] * media, base)
        return pred, gate, ids, alfa, ok


class Controlador:
    """Estado externo: solo incorpora episodios revelados en origen + 2 días."""
    def __init__(self, max_dias=365):
        self.max_dias = max_dias
        self.episodios = {}
        self.ultimo_origen = None

    def insertar(self, dia, clave, residuo, disponible_desde):
        if disponible_desde < dia + 2:
            raise ValueError('Un residuo completo requiere retraso de dos días')
        if dia in self.episodios:
            raise ValueError('Episodio duplicado')
        self.episodios[int(dia)] = (clave, residuo, int(disponible_desde))

    def banco(self, origen):
        import numpy as np
        if self.ultimo_origen is not None and origen <= self.ultimo_origen:
            raise ValueError('Los orígenes deben ser estrictamente crecientes')
        self.ultimo_origen = int(origen)
        elegibles = [(d, v) for d, v in sorted(self.episodios.items())
                     if 2 <= origen-d <= self.max_dias and v[2] <= origen]
        claves = np.zeros((365,32), dtype=np.float32)
        residuos = np.zeros((365,24), dtype=np.float32)
        fechas = np.zeros(365,dtype=np.int64)
        validas = np.zeros(365,dtype=bool)
        for i,(d,(k,r,_)) in enumerate(elegibles):
            claves[i], residuos[i], fechas[i], validas[i] = k,r,d,True
        return claves,residuos,fechas,validas

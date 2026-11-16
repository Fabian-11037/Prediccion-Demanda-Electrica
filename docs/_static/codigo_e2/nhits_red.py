import torch

from torch import nn

from .patchtst_numerico import Preparador, entrenar, predecir

from .vendor.nhits.core import NHITSBlock, _IdentityBasis

REVISION = '94b6ce131838be38a1fb5157349f02bbc8c89ef6'

class BaseLinealDeterminista(nn.Module):
    """La interpolación es lineal en sus nodos: se materializa su matriz en CPU."""
    def __init__(self, L, H, nodos):
        super().__init__()
        self.L = L
        matriz = torch.nn.functional.interpolate(torch.eye(nodos)[:, None, :],
                    size=H, mode='linear', align_corners=False)[:, 0, :]
        self.register_buffer('interpolacion', matriz)

    def forward(self, theta):
        return theta[:, :self.L], (theta[:, self.L:] @ self.interpolacion).unsqueeze(-1)

class NHITSNumerico(nn.Module):
    """Tres bloques oficiales, con exógenas históricas y calendario conocido."""
    def __init__(self, p, sigma_demanda, L=168, H=24):
        super().__init__()
        self.sigma_demanda = float(sigma_demanda)
        self.blocks = nn.ModuleList([
            NHITSBlock(input_size=L, h=H, n_theta=L + H // factor,
                mlp_units=[[512, 512]] * 3,
                basis=BaseLinealDeterminista(L, H, H // factor),
                futr_input_size=23, hist_input_size=59, stat_input_size=0,
                n_pool_kernel_size=pool, pooling_mode='MaxPool1d',
                dropout_prob=p['dropout'], activation='ReLU')
            for pool, factor in zip([2, 2, 1], [4, 2, 1])])

    def forward(self, x, cal):
        # Reconstrucción exacta de demanda/nivel, no de etiquetas futuras.
        demanda = 1 + x[:, :, 0] * self.sigma_demanda
        historicas = torch.cat([x[:, :, 1:30], x[:, :, 53:83]], dim=2)
        calendario = torch.cat([x[:, :, 30:53], cal], dim=1)
        residual = demanda.flip(dims=(-1,))
        pronostico = demanda[:, -1:, None]
        for bloque in self.blocks:
            backcast, contribucion = bloque(residual, calendario, historicas, None)
            residual = residual - backcast
            pronostico = pronostico + contribucion
        return pronostico.squeeze(-1)

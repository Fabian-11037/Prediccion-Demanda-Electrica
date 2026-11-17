"""TFT compacto con componentes de referencia Nixtla y contrato común."""
import torch
from torch import nn
from .patchtst_numerico import Preparador, entrenar, predecir
from .vendor.tft.core import TFTEmbedding, TemporalCovariateEncoder, TemporalFusionDecoder, tft_forward

ARQUITECTURA = dict(origen='Nixtla/neuralforecast',
    revision='94b6ce131838be38a1fb5157349f02bbc8c89ef6',
    hidden_size=16, n_head=4, n_rnn_layers=1, rnn_type='lstm',
    activacion_grn='ELU', historia=168, horizonte=24,
    historicas_exogenas=59, calendario=23, estaticas=0,
    salida='lineal univariada de referencia', adaptacion='compacta exploratoria')


class TFTNumerico(nn.Module):
    def __init__(self, p, sigma_demanda, L=168, H=24):
        super().__init__()
        self.sigma_demanda = float(sigma_demanda)
        self.input_size = L
        self.example_length = L + H
        self.n_rnn_layers = 1
        self.rnn_type = 'lstm'
        self.embedding = TFTEmbedding(hidden_size=16, stat_input_size=0,
            futr_input_size=23, hist_input_size=59, tgt_size=1)
        self.temporal_encoder = TemporalCovariateEncoder(hidden_size=16,
            num_historic_vars=83, num_future_vars=23,
            dropout=p['dropout'], grn_activation='ELU', n_rnn_layers=1, rnn_type='lstm')
        self.temporal_fusion_decoder = TemporalFusionDecoder(n_head=4, hidden_size=16,
            example_length=L+H, encoder_length=L, attn_dropout=p['dropout'],
            dropout=p['dropout'], grn_activation='ELU')
        self.output_adapter = nn.Linear(16, 1)
        self.interpretability_params = {}

    def forward(self, x, cal):
        ventanas = dict(insample_y=(1 + x[:, :, :1] * self.sigma_demanda),
            hist_exog=torch.cat([x[:, :, 1:30], x[:, :, 53:83]], dim=2),
            futr_exog=torch.cat([x[:, :, 30:53], cal], dim=1), stat_exog=None)
        return tft_forward(self, ventanas).squeeze(-1)

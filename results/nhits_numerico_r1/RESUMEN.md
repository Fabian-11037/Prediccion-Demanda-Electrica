# NHITS numérico adaptado

{
  "modelo": "NHITS_adaptado",
  "n_final": 5,
  "MASE_validacion": 0.9303906182274658,
  "anios_confirmacion": [
    2019,
    2020,
    2021
  ],
  "MASE_prueba": 1.0236287774259343,
  "MASE_sd": 0.057097578807101854,
  "MAE_prueba": 220712.6910694766,
  "dimensionamiento": {
    "margen_relativo": 0.03,
    "margen_absoluto": 0.03425805078502295,
    "MASE_referencia": 1.1419350261674317,
    "alpha": 0.05,
    "potencia_objetivo": 0.8,
    "replicas": 10000,
    "bloque_dias": 14,
    "semillas_piloto": 5,
    "potencia_limite_temporal": 0.0843,
    "alcance": "Estimación bootstrap exploratoria de precisión bajo alternativa mínima; cinco semillas no certifican potencia ni distribución de nuevas inicializaciones",
    "n_final": null,
    "estado": "precision_temporal_insuficiente",
    "evaluaciones": []
  },
  "nota": "Reevaluación retrospectiva exploratoria. El piloto no certifica potencia futura. Arquitectura oficial adaptada a covariables; sin exención E2 automática.",
  "exploratorio": true,
  "potencia_acreditada": false,
  "semillas": [
    1000,
    1001,
    1002,
    1003,
    1004
  ],
  "mejora_ingenuo_semanal_pct": 24.45022445893018,
  "RMSE_prueba": 281206.85019475647,
  "sMAPE_prueba": 2.4017796910052764,
  "MAPE_prueba": 2.403842360496724
}

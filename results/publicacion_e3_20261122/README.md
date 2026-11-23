# Resultados GPU · 22 de noviembre de 2026

Se publican la evaluación final del corrector, desarrollo y ajuste final de M1, y prueba retrospectiva de M1.

- Corrector y M1: `../corrector_final_y_m1_gpu_20261122_150345/`.
- Prueba de M1: `../prueba_m1_gpu_20261122_185842/INFORME.md`.
- Alcance y archivos reconstruibles conservados localmente: `ALCANCE.json`. Los manifiestos de ejecución incluyen también esos insumos locales; no equivalen a un manifiesto exclusivo de archivos publicados.

M1: MASE 0,634923; NNLS: 0,620904; PatchTST reentrenado: 0,683149. Prueba 2022–2025, 1.461 días y cinco semillas. M1 mejora significativamente frente al experto PatchTST, pero no supera NNLS. Estos experimentos son retrospectivos e informados por el benchmark conocido.

Se preservan pesos y escaladores. No se modifica ni despliega el JBook o el dashboard. M2 no está incluido porque su ejecución es posterior a esta publicación.

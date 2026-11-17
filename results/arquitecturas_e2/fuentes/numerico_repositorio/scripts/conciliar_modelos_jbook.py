"""Consistencia editorial del informe con los artefactos completos."""


def conciliar(books,md):
    for cell in books['09_modelos']:
        if cell.cell_type=='markdown':
            if cell.source.startswith('## 10.11'):
                cell.source='''## 10.11 Evidencia experimental y alcance

| Requisito del E1 | Evidencia incorporada | Alcance |
|---|---|---|
| Tres modelos de Deep Learning | MLP, LSTM y CNN1D; ecuaciones, capacidad y configuraciones | ARX se presenta como contexto. La revisión sistemática de frecuencias sigue pendiente. |
| Pipeline completo | Fuentes de ventanas, escalado, entrenamiento, búsqueda, piloto y prueba | Código íntegro en el apéndice; no se repiten entrenamientos. |
| Piloto y múltiples ejecuciones | Cinco semillas piloto y 195 finales por familia | Dimensionamiento original y cálculo pareado retrospectivo documentados. |
| Presupuesto comparable | Espacios, tiempos y número de ajustes | LSTM evaluó cinco candidatos frente a veinte: limitación de equidad. |
| Métricas, ranking y baseline | MASE, MAE, RMSE, sMAPE, R² y referencias deterministas | Intervalos por inicialización y por días se reportan separadamente. |
| Curvas y estabilidad | Historiales completos, pérdidas y épocas | Ejemplos con semilla 1000 fijada de antemano; todos los historiales se conservan. |
| Errores por horizonte y segmentos | Las 24 horas, años, tipos de día y cuartiles descriptivos | Análisis posterior a congelar configuraciones. |
| Inferencia y residuos | Bootstrap temporal, DM–HLN, Holm, Ljung–Box, ARCH y BDS | Análisis secundario retrospectivo, sin equivalencia con evaluación prospectiva. |
| Complejidad y costo | Parámetros, tiempos de ajuste, hardware, pesos y escaladores | La latencia de inferencia no se registró y no se deduce del tiempo de ajuste. |

El [análisis temporal completo](10_evaluacion.ipynb) verifica las predicciones, incorpora las curvas y desarrolla los diagnósticos. Las fuentes y la prueba son retrospectivas; la versión 2 se diseñó después de consultar la versión 1, conforme a las adendas. Los contrastes secundarios no corrigen esa limitación.'''
            cell.source=cell.source.replace('Estos indicadores permiten describir la parada, pero no reconstruyen las curvas completas por época.',
                'Estos indicadores describen la parada; las curvas completas por época se presentan en el análisis temporal enlazado.')
            cell.source=cell.source.replace('La prueba por rangos también requiere considerar su simetría.',
                'La prueba por rangos también requiere considerar su simetría.')
            cell.source=cell.source.replace('Shapiro–Wilk y Q–Q describen las diferencias; la prueba por rangos también requiere considerar su simetría.',
                'Shapiro–Wilk y Q–Q describen las diferencias. La t se interpreta como aproximación de la media con 195 pares y se contrasta con intervalos bootstrap; Wilcoxon es sensibilidad de rangos y no equivale a un contraste de medias cuando hay asimetría.')
            cell.source=cell.source.replace('Si supera el 3 %, las 195 ejecuciones no garantizan el objetivo de potencia para ese contraste.',
                'En estos contrastes neuronales, el máximo es inferior al margen del 3 % bajo la aproximación utilizada.')
            cell.source=cell.source.replace('Los puntos de la derecha corresponden sólo a h = 1, 12 y 24: no se interpolan los horizontes intermedios ni se reconstruye una curva de 24 errores a partir de tres valores.',
                'El análisis temporal complementario presenta las 24 horas y separa sus errores del cambio global entre periodos.')
            if cell.metadata.get('plantilla_resultados'):
                cell.metadata['plantilla_resultados']=cell.source
        if cell.cell_type=='code':
            if 'cv_lstm=float' in cell.source:
                a=cell.source.index('fig,ax=plt.subplots()')
                b=cell.source.index('cv_lstm=float',a)
                cell.source=cell.source[:a]+"""fig,ax=plt.subplots()
for nombre in modelos:
    d=piloto[piloto.modelo==nombre]
    medias=[d[f'MASE_{a}'].mean() for a in [2018,2019,2020,2021]]
    desv=[d[f'MASE_{a}'].std(ddof=1) for a in [2018,2019,2020,2021]]
    ax.errorbar([2018,2019,2020,2021],medias,yerr=desv,marker='o',capsize=3,label=etiquetas[nombre])
ax.set(xticks=[2018,2019,2020,2021],xlabel='Año externo',ylabel='MASE piloto medio ± DE de cinco semillas')
ax.legend(fontsize=8);fig.tight_layout();display(fig);plt.close(fig)
"""+cell.source[b:]
            if "comparacion=pd.read_csv(R/'comparacion_validacion_prueba.csv')" in cell.source:
                cell.source="""comparacion=pd.read_csv(R/'comparacion_validacion_prueba.csv').set_index('modelo').reindex(modelos)
fig,ax=plt.subplots();pos=np.arange(len(modelos))
ax.bar(pos-.18,comparacion.MASE_validacion_piloto,width=.36,label='Piloto de validación')
ax.bar(pos+.18,comparacion.MASE_prueba,width=.36,label='Prueba')
ax.set(xticks=pos,xticklabels=[etiquetas[n] for n in modelos],ylabel='MASE')
ax.legend();fig.tight_layout();display(fig);plt.close(fig)
brecha_mlp=float(comparacion.loc['mlp','brecha_MASE_pct'])
"""
            if 'display(pd.DataFrame(nemenyi))' in cell.source:
                cell.source=cell.source.replace('display(pd.DataFrame(nemenyi))',
                    "display(pd.DataFrame(nemenyi).style.format({'p_Nemenyi_aprox':lambda p:'<1e-15' if p<1e-15 else f'{p:.3g}'}))")

    for cell in books['08_validacion']:
        if cell.cell_type=='markdown':
            cell.source=cell.source.replace('La inferencia requiere un preregistro específico:',
                'El análisis secundario de los resultados guardados se documenta en el capítulo de modelos y su evaluación temporal. Una evaluación confirmatoria futura requiere un preregistro específico:')
            cell.source=cell.source.replace('La longitud ilustrativa de 14 días no constituye una elección inferencial del estudio.',
                'El análisis secundario temporal utiliza 14 días como bloque principal y 7 y 28 como sensibilidad, conforme al plan registrado antes de cargar las predicciones.')
    for cell in books['11_sintesis']:
        if cell.cell_type=='markdown':
            cell.source=cell.source.replace('Los resultados de entrenamiento se incorporarán una vez completada su evaluación.',
                'Los resultados predictivos y sus diagnósticos se presentan en el capítulo de modelos.')
            cell.source=cell.source.replace('la desigualdad del presupuesto LSTM y la ausencia pública de predicciones y curvas completas limitan la inferencia temporal y la comprobación de todos los requisitos del E1.',
                'la desigualdad del presupuesto LSTM y el diseño retrospectivo limitan la interpretación confirmatoria. La nueva instantánea contiene predicciones y curvas completas, utilizadas en los diagnósticos temporales.')
    books['12_reproducibilidad'].append(md('## Resultados importados y reproducción del benchmark\n\nLa instantánea 02ff447d94fa78e1cde323fa2382da5bce24fb6e se conserva en results/repositorio y su IMPORTACION.json identifica cada archivo. El índice remoto permite verificar pesos, escaladores, curvas, predicciones e identidades de los 1.640 ajustes completados. El código analítico y los planes secundarios están en scripts y results/jbook/modelos.\n\nLa reconstrucción del libro usa esos artefactos sin entrenar ni invocar redes. El apéndice reproduce las fuentes congeladas de entrenamiento y prueba; una nueva ejecución del pipeline exige un directorio distinto y conserva sus opciones explícitas de acceso a prueba. La latencia requiere una medición específica y no se estima desde tiempos de ajuste.'))

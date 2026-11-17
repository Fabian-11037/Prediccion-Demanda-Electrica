from pathlib import Path
from datetime import datetime,timezone
import hashlib,json

root=Path.cwd();sha='02ff447d94fa78e1cde323fa2382da5bce24fb6e'
snapshot=root/'results/repositorio'/sha;r=snapshot/'results/base_v2'
dest=root/'results/jbook/modelos';dest.mkdir(exist_ok=True,parents=True)
plan=dict(fecha=datetime.now(timezone.utc).isoformat(),tipo='Análisis secundario retrospectivo de resultados publicados',
    commit=sha,modelos_neuronales=['mlp','lstm','cnn1d'],referencia_lineal='arx_lineal',
    semilla_analisis=11037,bootstrap_replicas=10000,permutaciones_friedman=19999,
    alfa=.05,delta_rel=.03,potencia_objetivo=.8,
    unidad='Semilla final apareada; inferencia condicional al mismo conjunto de prueba',
    familia_principal='Tres contrastes neuronales; corrección de Holm',
    familia_contexto='Tres contrastes frente a ARX; corrección de Holm separada',
    contrastes=['t pareada de medias','Wilcoxon bilateral como sensibilidad',
                'Friedman con permutación de etiquetas dentro de semilla','Nemenyi aproximado'],
    intervalos='Bootstrap pareado por semillas, percentiles 2.5 y 97.5; t para medias individuales',
    efectos=['Cohen dz pareado','reducción relativa de medias','probabilidad empírica de victoria por semilla'],
    dimensionamiento='Conservar n_final=195 del piloto original; cálculo pareado adicional sobre piloto, sin cambiar ejecuciones',
    limites=['No es preregistro anterior a entrenar ni a consultar resultados publicados',
             'Semillas comparten prueba; no estiman incertidumbre temporal ni sustituyen DM por horizonte',
             'No se ejecuta entrenamiento ni se generan nuevas predicciones',
             'Sin predicciones v2 no se reconstruyen residuos, R2, segmentos ni errores de todos los horizontes',
             'Sin historiales completos v2 no se reconstruyen curvas por época'],
    sha256={n:hashlib.sha256((r/n).read_bytes()).hexdigest() for n in
            ['final_runs.csv','piloto.csv','busqueda.csv','resumen_ajustes.csv','resumen_final.csv']})
path=dest/'plan_analisis_secundario.json'
if not path.exists():path.write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding='utf-8')
else:
    previo=json.loads(path.read_text(encoding='utf-8'))
    assert previo['sha256']==plan['sha256']
print('Plan de análisis secundario guardado, sin ejecutar contrastes')

from datetime import datetime,timezone
from pathlib import Path
import json,hashlib

root=Path.cwd();sha='02ff447d94fa78e1cde323fa2382da5bce24fb6e'
r=root/'results/repositorio'/sha/'results/base_v2'
dest=root/'results/jbook/modelos';dest.mkdir(exist_ok=True,parents=True)
plan=dict(fecha=datetime.now(timezone.utc).isoformat(),commit=sha,
    alcance='Análisis secundario retrospectivo de predicciones y curvas ya guardadas',
    semilla=11038,bootstrap_replicas=10000,bloque_dias=14,bloques_sensibilidad=[7,28],
    remuestreo='Bootstrap circular de días estratificado por año; índices apareados entre modelos',
    unidad='Origen diario con sus 24 horas; pérdida absoluta promediada entre 195 semillas antes del remuestreo',
    estimando='Error esperado sobre inicializaciones observadas; no error de una predicción ensemble',
    intervalo='Percentiles 2.5 y 97.5 del bootstrap por días, condicionado a las 195 realizaciones',
    familias={'global_neuronales':3,'global_contexto_arx':3,'horizontes_neuronales':72,'horizontes_contexto_arx':72},
    multiplicidad='Holm dentro de cada familia declarada',
    contraste_bootstrap='Bilateral centrado bajo igualdad de pérdida media, con corrección +1',
    DM='Diferencia de pérdida absoluta; HAC Bartlett con 14 rezagos diarios; HLN h=1 origen diario no solapado; t n-1',
    residuos='Semilla fija 1000; media diaria del error firmado; Ljung-Box 7/14/28 días, ARCH 7, BDS dimensión 2 últimos 1000 días',
    diagnosticos_multiplicidad='Holm entre los 20 contrastes de residuos',
    segmentos=['año calendario','lunes-viernes no festivo, sábado no festivo, domingo/festivo',
               'cuartiles descriptivos de demanda realizada en prueba; no usados como entradas'],
    curvas='Semilla 1000 elegida antes de examinar curvas; archivos completos conservados para todas las semillas',
    ventana_cualitativa=['2023-07-01','2023-07-07'],
    metrica_primaria='MASE con escala de entrenamiento fija',
    nuevas_predicciones=False,nuevo_entrenamiento=False,
    limites=['El plan es posterior al entrenamiento y a la publicación de métricas agregadas',
             'La variación temporal se reporta separada de la variación por inicialización',
             'No hay diseño confirmatorio independiente; se preservan presupuestos y configuraciones'],
    entradas={n:hashlib.sha256((r/n).read_bytes()).hexdigest() for n in
              ['indice_artefactos.csv','final_runs.csv','mejor_config.json']})
p=dest/'plan_analisis_temporal.json'
if p.exists():assert json.loads(p.read_text(encoding='utf-8'))['entradas']==plan['entradas']
else:p.write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding='utf-8')
print('Plan temporal registrado antes de cargar las predicciones')

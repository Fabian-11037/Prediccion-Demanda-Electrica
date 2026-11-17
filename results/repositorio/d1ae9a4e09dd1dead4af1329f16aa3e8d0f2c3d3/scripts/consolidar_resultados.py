"""Consolida métricas y referencias de artefactos sin reentrenar modelos."""
import csv,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/consolidado'
def leer_csv(path):
    with path.open() as f:return list(csv.DictReader(f))
def guardar_csv(path,rows):
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
def main():
    OUT.mkdir(exist_ok=True);rows=[];correctas=[]
    for version in ('base_v2','base_v3','base_v4'):
        r=ROOT/'results'/version
        assert (r/'ejecucion_completa.json').exists(),version
        for d in leer_csv(r/'resumen_final.csv'):
            if d['modelo'].startswith('ingenuo'):continue
            old=version=='base_v2'
            row=dict(ejecucion=version,modelo=d['modelo'],escalado='Original: incluye ES en folds tempranos' if old else 'Corregido por fold',
                semillas_prueba=int(d['n'] if old else d['n_final']),MASE_validacion=float(d['MASE_validacion_piloto']) if not old else '',
                MASE_prueba=float(d['MASE_media'] if old else d['MASE_prueba']),MAE_prueba=float(d['MAE_media'] if old else d['MAE_prueba']),
                sMAPE_prueba=float(d['sMAPE_media'] if old else d['sMAPE_prueba']))
            rows.append(row)
            if not old:correctas.append(row)
    guardar_csv(OUT/'metricas_ejecuciones.csv',rows)
    candidates={}
    for version in ('base_v3','base_v4'):
        d=json.loads((ROOT/'results'/version/'mejor_config.json').read_text())
        for name,v in d.items():
            if name not in candidates or v['MASE_cv']<candidates[name]['MASE_cv']:
                candidates[name]=dict(ejecucion=version,**v)
    models={};selected=[]
    for name,v in sorted(candidates.items()):
        version=v['ejecucion'];r=ROOT/'results'/version;artifacts=[]
        for p in sorted((r/'ajustes/final').glob('*/identidad.json')):
            i=json.loads(p.read_text())
            if i['modelo']!=name or i['configuracion']!=v['configuracion']:continue
            folder=p.parent
            artifacts.append(dict(semilla=i['semilla'],**{k:str((folder/file).relative_to(ROOT)) for k,file in [('pesos','modelo.pt'),('escaladores','escaladores.json'),('predicciones','predicciones.npz'),('historial','historia.json'),('identidad','identidad.json'),('metricas','resultado.json')]}))
        assert sorted(d['semilla'] for d in artifacts)==list(range(1000,1005)),name
        models[name]=dict(**v,criterio='Menor MASE de confirmación en validación entre v3 y v4; comparación exploratoria posterior',artefactos=artifacts)
        row=next(d for d in correctas if d['modelo']==name and d['ejecucion']==version)
        selected.append(dict(**row,MASE_confirmacion=v['MASE_cv'],configuracion=v['configuracion']['id']))
    guardar_csv(OUT/'modelos_seleccionados.csv',selected)
    (OUT/'modelos.json').write_text(json.dumps(models,ensure_ascii=False,indent=2))
    refs=json.loads((ROOT/'results/base_v4/referencias_test.json').read_text())
    (OUT/'referencias_test.json').write_text(json.dumps(refs,ensure_ascii=False,indent=2))
    inventory=[];nfits=0
    for version in ('base_v3','base_v4'):
        paths=list((ROOT/'results'/version/'ajustes').glob('*/*/resultado.json'))
        expected=json.loads((ROOT/'results'/version/'ejecucion_completa.json').read_text())['ajustes'];assert len(paths)==expected
        nfits+=len(paths)
        for p in sorted(paths):
            i=json.loads((p.parent/'identidad.json').read_text())
            for file in ('modelo.pt','escaladores.json','predicciones.npz','historia.json','identidad.json','resultado.json'):
                q=p.parent/file;assert q.is_file() and q.stat().st_size>0
                with q.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
                inventory.append(dict(ejecucion=version,etapa=i['etapa'],modelo=i['modelo'],semilla=i['semilla'],configuracion=i['configuracion']['id'],fold=i['fold'],archivo=str(q.relative_to(ROOT)),bytes=q.stat().st_size,sha256=h))
    guardar_csv(OUT/'indice_artefactos_v3_v4.csv',inventory)
    md=['# Resultados consolidados','', 'Actualización: 8 de noviembre de 2026. Demanda horaria del SIN; pronóstico diario de 24 horas. Entradas: demanda histórica y calendario. Validación expansiva 2018–2021, prueba fija 2022–2025.','',
        '## Modelos de referencia con preprocesamiento corregido','',
        'Selección por el menor MASE de confirmación de validación entre las configuraciones elegidas en v3 y v4; no por error de prueba ni por la mejor semilla. Se conservan las cinco semillas finales de cada modelo.','',
        '| Familia | Ejecución | MASE validación piloto | MASE prueba | MAE prueba (kWh) | Reducción frente al ingenuo semanal |','|---|---|---:|---:|---:|---:|']
    baseline=refs['ingenuo_168h']['MASE']
    for d in sorted(selected,key=lambda d:d['MASE_prueba']):
        md.append(f"| {d['modelo']} | {d['ejecucion']} | {d['MASE_validacion']:.3f} | {d['MASE_prueba']:.3f} | {d['MAE_prueba']:,.0f} | {100*(1-d['MASE_prueba']/baseline):.1f}% |")
    md+=['',f"Referencias de prueba: ingenuo semanal MASE={baseline:.3f}; ingenuo diario MASE={refs['ingenuo_24h']['MASE']:.3f}.",'',
        '## Trazabilidad y alcance','',
        '- `metricas_ejecuciones.csv`: todas las métricas agregadas de v2, v3 y v4; la ejecución original no se mezcla con las corregidas.',
        '- `modelos.json`: configuración y rutas de pesos, escaladores, predicciones, curvas y métricas de las cinco semillas por familia. Las rutas parten de la raíz del repositorio; no se duplican checkpoints.',
        f'- `indice_artefactos_v3_v4.csv`: SHA-256 de {len(inventory)} archivos, correspondientes a {nfits} ajustes completos. Los artefactos de v2 conservan su índice en `results/base_v2/indice_artefactos.csv`.',
        '- `referencias_test.json`: métricas de ambos ingenuos. Las versiones originales y sus fuentes permanecen disponibles.',
        '- Predicciones: NPZ con `y`, `yhat` y `origen` (nanosegundos UTC); `scripts/exportar_predicciones.py` exporta fechas y semillas a CSV. Los historiales contienen pérdidas de entrenamiento y parada temprana por época.',
        '- Checkpoints: estado PyTorch y metadatos; en v3/v4 el escalador también está disponible como JSON independiente. Para reconstruir v3 se usan `DatosFold` y `Secuencial` de `demanda.reentrenamiento_v3`; para secuenciales v4, `demanda.variantes_v4.Secuencial`. ARX/MLP usan los constructores de `modelos_base`. La clase Entrenador v4 reutiliza el bucle v3 con la fábrica v4.',
        '', '## Limitaciones','',
        'La prueba fue observada repetidamente: todas las comparaciones entre ejecuciones son exploratorias y descriptivas. No se afirma significancia, equivalencia ni confirmación independiente. La selección consolidada también es retrospectiva. V2 usó 195 semillas finales; v3/v4, cinco. Cambian presupuesto, escalado y arquitectura entre versiones, por lo que las diferencias no identifican efectos causales aislados. Las desviaciones entre semillas no representan incertidumbre temporal. El escalado fijo de v2 incluía parte de la parada temprana en los pliegues tempranos; v3/v4 lo corrigen sin modificar las ejecuciones antiguas. El denominador MASE de referencia 2000–2017 permanece fijo para comparabilidad y no participa en la optimización.',
        '', '## Auditoría y diagnósticos','',
        'La auditoría de ventanas y la salvedad del escalado están en `results/auditoria_dataset/`; la sensibilidad a demanda y los controles exploratorios iniciales están en `results/diagnostico_secuenciales/`. Sus hallazgos no deben confundirse con un experimento causal o con resultados de la fase corregida.']
    (OUT/'README.md').write_text('\n'.join(md)+'\n')
    print(f'Consolidados: {len(rows)} resultados, {len(models)} familias, {nfits} ajustes nuevos, {len(inventory)} artefactos.')
if __name__=='__main__':main()

from pathlib import Path
import sys,json,io,hashlib,numpy as np,pandas as pd
R=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(R/'dashboard'))
from app import server
import runtime_inferencia as L
client=server.test_client();rows=[]
for key in L.MODELS:
    response=client.post('/api/inferencia',json={'model':key,'origin':'2026-01-01'})
    r=response.get_json();assert response.status_code==200,(key,r)
    assert len(r['prediction_gwh'])==24 and np.isfinite(r['prediction_gwh']).all()
    assert r['computed'] and r['training'] is False
    rows.append(dict(model=key,status=response.status_code,origin=r['origin'],latency_ms=r['latency_ms']))
    print(key,'nuevo origen correcto',flush=True)
rt=L.get_runtime();base=rt.predict('m4_n','2023-07-01')
mask=rt.bank_dates>=np.datetime64('2023-06-30')
backup={k:rt.bank[k][mask].copy() for k in ['y','pred_base','num','expertos']}
for k in backup:rt.bank[k][mask]=1e12
after=rt.predict('m4_n','2023-07-01')
for k in backup:rt.bank[k][mask]=backup[k]
assert np.array_equal(base['prediction_gwh'],after['prediction_gwh'])
changed=rt.predict('m4_n','2023-07-01',1)
assert not np.allclose(base['prediction_gwh'],changed['prediction_gwh'],rtol=1e-7,atol=1e-7)
assert all(np.datetime64(n['fecha'])<=np.datetime64('2023-06-29') for n in base['detail']['neighbors'])
template=client.get('/api/plantilla.csv');assert template.status_code==200
csv=template.data.decode();frame=pd.read_csv(io.StringIO(csv));assert len(frame)==168
response=client.post('/api/inferencia',json={'model':'m4_n','origin':'2026-01-01','csv':csv,'explain':True})
assert response.status_code==200 and response.get_json()['local_delta_gwh'] is not None,(response.status_code,response.get_json())
assert client.post('/api/inferencia',json={'model':'otro','origin':'2023-07-01'}).status_code==400
assert client.post('/api/inferencia',json={'model':'m4_n','origin':'2023-07-01','demand_pct':99}).status_code==400
assert client.post('/api/inferencia',json={'model':'m4_n','origin':'2026-01-02','csv':csv}).status_code==400
assert client.get('/inferencia/').status_code==200
assert client.get('/dashboard/site/index.html').status_code==200
assert client.get('/healthz').status_code==200
assert client.get('/dashboard/site/../../CONTEXTO_PROYECTO.md').status_code==404
out=dict(models=rows,causal_future_corruption_invariant=True,input_changes_prediction=True,
    csv_inference=True,local_sensitivity=True,invalid_input_rejected=True,no_path_traversal=True,training=False,published=False)
(R/'results/revision_e3_integral_20261123/verificacion_api_inferencia.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print('API, causalidad, CSV, sensibilidad y rutas: correctos.')

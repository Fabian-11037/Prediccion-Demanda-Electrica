"""Restaura N-BEATS/TimesNet originales. Inferencia GPU, sin actualizar pesos."""
from e3_gpu_comun import *
import argparse,traceback
from entrenamiento import redes_e2
from m4_cierre_benchmark import configuraciones

def ejecutar(out):
    car=out/'historicos_nbeats_timesnet';torch.set_num_threads(2);assert torch.cuda.is_available()
    datos=DatosNumericos(PAQUETE);tr,st=datos.ajuste_final();esc=datos.ajustar_escalas(tr)
    cfgs=configuraciones();reg=leer(car/'RECEPCION.json');pesos={r['archivo']:r['sha256'] for r in reg['checkpoints']}
    registro=dict(fecha=pd.Timestamp.now(tz='America/Bogota').isoformat(),operacion='Inferencia de checkpoints originales; cero optimizadores y cero actualización de pesos',
        reconstruccion_escalas='Determinista con DatosNumericos.ajuste_final y ajustar_escalas(tr); ZIP sin escaladores',
        panel_sha256=datos.hash_panel,fuente_sha256=sha(Path(__file__)),semillas=list(range(1000,1005)),configuraciones={k:cfgs[k] for k in ['NBEATS','TimesNet']},
        ajuste=[str(datos.indice[tr[0]]),str(datos.indice[tr[-1]])],parada=[str(datos.indice[st[0]]),str(datos.indice[st[-1]])],q=datos.q168,
        tolerancias=dict(atol_q=1e-5,rtol=1e-5,diferencia_MASE_maxima=1e-5),pesos=pesos,gpu=torch.cuda.get_device_name(0),torch=torch.__version__)
    guardar(car/'REGISTRO_PREINFERENCIA.json',registro);guardar(car/'escaladores_reconstruidos.json',escalas_a_json(esc,datos))
    pos=datos.prueba();c=datos.construir(pos,esc);tt=motor.a_tensores(c,'cuda');y=c.y.astype('float64');dias=c.origen.strftime('%Y-%m-%d').to_numpy().astype('U10');assert y.shape==(1461,24)
    filas=[];indice={};num=0
    def progreso(**kw):
        d=dict(estado='Ejecutando',fase='Inferencia histórica congelada',tareas=num,total=10,gpu=torch.cuda.get_device_name(0),actualizado=pd.Timestamp.now(tz='America/Bogota').isoformat(),**kw)
        guardar(car/'ESTADO.json',d);guardar(out/'progreso/estado_predicciones.json',d)
    for nombre,clave,esperado in [('TimesNet','nuevo_timesnet',.642927818004554),('NBEATS','nuevo_nbeats',.9533426171053975)]:
        arr=[];archivos=[]
        for s in range(1000,1005):
            progreso(etiqueta=f'{nombre} semilla{s}');dest=car/'predicciones'/nombre/f'predicciones_{s}.npz';p=car/'pesos/resultados_e2_nbeats_timesnet'/clave/'final/N'/f'semilla_{s}/modelo.pt'
            assert sha(p)==pesos[str(p.relative_to(car))]
            if dest.exists():
                with np.load(dest) as z:pr=z['pred'];assert np.array_equal(z['dia'],dias)
            else:
                red=redes_e2.construir('timesnet' if nombre=='TimesNet' else 'nbeats',cfgs[nombre],c.x.shape[2],c.cal_p.shape[2],datos.L,datos.H).cuda().eval()
                red.load_state_dict(torch.load(p,map_location='cpu',weights_only=True),strict=True)
                with torch.inference_mode():pr=motor.predecir(red,tt,1 if nombre=='TimesNet' else 64).astype('float64')*c.nivel[:,None]
                assert np.isfinite(pr).all();dest.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(dest,dia=dias,y=y,pred=pr,**({'yhat_lote1_kwh':pr} if nombre=='TimesNet' else {}))
                pd.DataFrame(dict(fecha=np.repeat(dias,24),hora=np.tile(np.arange(24),len(dias)),semilla=s,observado_kwh=y.ravel(),predicho_kwh=pr.ravel())).to_csv(dest.with_suffix('.csv.gz'),index=False,compression='gzip')
                del red;torch.cuda.empty_cache()
            arr.append(pr);archivos.append(str(dest));filas.append(dict(modelo=nombre,semilla=s,**metricas(y,pr,datos.q168)));num+=1;progreso()
        arr=np.asarray(arr);publicado=leer(out/'insumos'/f'{clave}__N.json');media=np.asarray(publicado['mean'])*1e6;recuperado=arr.mean(0)
        mase=float(np.abs(arr-y).mean()/datos.q168);error_MASE=abs(mase-esperado)
        ok=bool(np.all(np.abs((recuperado-media)/datos.q168)<=1e-5+1e-5*np.abs(media/datos.q168)) and error_MASE<=1e-5)
        guardar(car/f'REPRODUCCION_{nombre}.json',dict(MASE=mase,MASE_publicado=esperado,diferencia_MASE=error_MASE,max_error_promedio_kwh=float(np.abs(recuperado-media).max()),reproduccion_aprobada=ok,escalas='reconstruidas únicamente desde entrenamiento',lote=1 if nombre=='TimesNet' else 64))
        assert ok,'La reconstrucción no reproduce el benchmark; no se integra ni se ajustan escalas usando prueba'
        indice[nombre]=dict(archivos=archivos,pred='pred')
    assert all(sha(car/p)==h for p,h in pesos.items())
    pd.DataFrame(filas).to_csv(car/'metricas_por_semilla.csv',index=False)
    # El motor principal y la estadística posterior podrán usar realizaciones
    # originales verificadas en vez de reconstrucciones desde agregados.
    guardar(out/'insumos/top10_adicionales.json',indice)
    guardar(car/'FINALIZADO.json',dict(entrenamiento=False,prueba=True,pesos_congelados=True,semillas=list(range(1000,1005)),origenes=1461,horizonte=24))
    guardar(car/'MANIFIESTO_FINAL.json',[dict(archivo=str(p.relative_to(car)),sha256=sha(p),bytes=p.stat().st_size) for p in sorted(car.rglob('*')) if p.is_file() and p.name!='MANIFIESTO_FINAL.json' and p.suffix not in ['.log','.tmp']])
    progreso(estado_final='Completado');e=leer(car/'ESTADO.json');e['estado']='Finalizado';guardar(car/'ESTADO.json',e);guardar(out/'progreso/estado_predicciones.json',e)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    try:ejecutar(a.out)
    except Exception:guardar(a.out/'historicos_nbeats_timesnet/ERROR.json',dict(error=traceback.format_exc(),pesos_actualizados=False));raise

"""Dashboard y panel Dash de inferencia local; pesos congelados y CPU."""
from pathlib import Path
from urllib.parse import parse_qs
import base64,io,json,os,time
import numpy as np
from flask import Flask,send_from_directory,redirect,request,jsonify,Response
from dash import Dash,html,dcc,Input,Output,State,no_update
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
from runtime_inferencia import ROOT,MODELS,get_runtime

SITE=ROOT/'dashboard/site'
server=Flask(__name__);server.config['MAX_CONTENT_LENGTH']=1000000

@server.get('/')
def index():return redirect('/dashboard/site/index.html')

@server.get('/dashboard/site/<path:name>')
def static_site(name):return send_from_directory(SITE,name)

@server.get('/results/revision_m4_teoria_20261123/_build/html/<path:name>')
def book(name):return send_from_directory(ROOT/'results/revision_m4_teoria_20261123/_build/html',name)

@server.get('/healthz')
def health():return jsonify(status='ok',inference='cpu',training=False,models=len(MODELS))

@server.post('/api/inferencia')
def api():
    try:
        data=request.get_json(force=False,silent=False)
        if not isinstance(data,dict):raise ValueError('Se espera un objeto JSON.')
        started=time.perf_counter()
        result=get_runtime().predict(data.get('model'),data.get('origin'),data.get('demand_pct',0),data.get('csv'),bool(data.get('explain',False)),data.get('coverage',.9),data.get('interval_scope','horario'))
        result['latency_ms']=1000*(time.perf_counter()-started)
        return jsonify(result)
    except (ValueError,KeyError,TypeError) as error:return jsonify(error=str(error)),400


@server.get('/api/plantilla.csv')
def template():
    from runtime_inferencia import LOCK
    try:
        with LOCK:
            I=get_runtime().inputs;I.configure('2026-01-01')
            import pandas as pd
            frame=pd.DataFrame(I.window,columns=I.hist)
            frame.insert(0,'fecha',pd.date_range(I.origin-pd.Timedelta(hours=168),periods=168,freq='h').astype(str))
            return Response(frame.to_csv(index=False),mimetype='text/csv',headers={'Content-Disposition':'attachment; filename=historia_168h.csv'})
    except ValueError as e:return jsonify(error=str(e)),400

dash=Dash(__name__,server=server,url_base_pathname='/inferencia/',title='Inferencia · Demanda SIN',
    external_stylesheets=['/dashboard/site/styles.css'],suppress_callback_exceptions=False)
app=dash
dash.layout=html.Main([
    dcc.Location(id='live-url'),dcc.Store(id='live-result'),
    html.H2('Inferencia en vivo'),
    html.P('Pronóstico de 24 horas calculado en CPU con los pesos y escaladores conservados. Las entradas son anteriores al origen y se alinean con los rezagos del contrato.'),
    html.Div([
        html.Div([html.Label('Modelo',htmlFor='live-model'),dcc.Dropdown(id='live-model',options=[{'label':v,'value':k} for k,v in MODELS.items()],value='m4_n',clearable=False)],className='field',id='live-model-field'),
        html.Div([html.Label('Origen · Colombia'),dcc.DatePickerSingle(id='live-date',date='2026-01-01',min_date_allowed='2022-01-01',max_date_allowed='2035-12-31',display_format='YYYY-MM-DD')],className='field'),
        html.Div([html.Label('Cambio de demanda histórica (%)',htmlFor='live-demand'),dcc.Input(id='live-demand',type='number',min=-20,max=20,step=.5,value=0)],className='field')
    ],className='control-row'),
    html.Div([html.Label('Cobertura nominal'),dcc.Dropdown(id='live-coverage',options=[{'label':str(int(p*100))+' %','value':p}for p in [.8,.9,.95]],value=.9,clearable=False),html.Label('Alcance'),dcc.Dropdown(id='live-interval-scope',options=[{'label':'Por hora','value':'horario'},{'label':'Trayectoria24h','value':'trayectoria24'}],value='horario',clearable=False)],className='control-row'),
    dcc.Upload(id='live-upload',children=html.Div(['Cargar historia propia · CSV de 168 horas']),multiple=False,
        style={'border':'1px dashed #78969b','padding':'15px','borderRadius':'5px','margin':'10px 0'}),
    html.P('El CSV usa fecha y las 30 variables históricas del contrato, ya alineadas por disponibilidad. Sin archivo se utiliza la historia local; el último origen disponible es 2026-01-01.',className='caption'),
    html.A('Descargar plantilla de historia',href='/api/plantilla.csv',className='button'),
    dcc.Checklist(id='live-explain',options=[{'label':' Calcular sensibilidad local a +1 % de demanda en las últimas 24 horas','value':'yes'}],value=['yes']),
    html.Div([html.Button('Generar pronóstico',id='live-run',n_clicks=0,className='primary'),
              html.Button('Descargar pronóstico',id='live-download',n_clicks=0,className='button')],className='actions'),
    dcc.Loading(html.Div(id='live-message',role='status'),type='circle'),
    html.Div(id='live-kpis',className='metrics compact'),
    dcc.Graph(id='live-forecast',figure=go.Figure(),style={'display':'none'},config={'displaylogo':False}),
    html.Div(id='live-interpretation'),html.Div(id='live-neighbors'),
    html.Details([html.Summary('Procedencia de la inferencia'),html.Pre(id='live-provenance')]),
    dcc.Download(id='live-csv')
],className='live-panel',style={'maxWidth':'1160px','padding':'24px','margin':'auto','background':'white'})

@dash.callback(Output('live-model','value'),Output('live-model-field','style'),Input('live-url','search'))
def selected(search):
    key=parse_qs((search or '').lstrip('?')).get('modelo',['m4_n'])[0]
    return (key if key in MODELS else 'm4_n'),({'display':'none'} if parse_qs((search or '').lstrip('?')).get('embed')==['1'] else {})

@dash.callback(Output('live-result','data'),Output('live-message','children'),Input('live-run','n_clicks'),
    State('live-model','value'),State('live-date','date'),State('live-demand','value'),
    State('live-upload','contents'),State('live-explain','value'),State('live-coverage','value'),State('live-interval-scope','value'),prevent_initial_call=True,
    running=[(Output('live-run','disabled'),True,False)])
def compute(clicks,model,date,change,upload,explain,coverage=.9,interval_scope="horario"):
    if not clicks:raise PreventUpdate
    try:
        csv=None
        if upload:
            _,body=upload.split(',',1);csv=base64.b64decode(body,validate=True).decode('utf-8-sig')
        started=time.perf_counter()
        result=get_runtime().predict(model,date,0 if change is None else change,csv,'yes' in (explain or []),coverage,interval_scope)
        result['latency_ms']=1000*(time.perf_counter()-started)
        return result,html.P('Pronóstico calculado con los pesos congelados.',className='outcome')
    except Exception as error:
        # No propagar rutas locales ni trazas a la interfaz.
        if isinstance(error,(ValueError,UnicodeError)):
            message=str(error)
        else:
            server.logger.exception('No se pudo completar la inferencia')
            message='No fue posible calcular el pronóstico. Revisa los archivos de pesos y las entradas del modelo.'
        return None,html.P(message,role='alert',className='error')

@dash.callback(Output('live-forecast','figure'),Output('live-forecast','style'),Output('live-kpis','children'),
    Output('live-interpretation','children'),Output('live-neighbors','children'),Output('live-provenance','children'),Input('live-result','data'))
def display(result):
    if not result:return go.Figure(),{'display':'none'},[],[],[],''
    r=result;fig=go.Figure()
    def dec(value,digits=2):return f'{value:.{digits}f}'.replace('.',',')
    fig.add_trace(go.Scatter(x=r['history_dates'][-48:],y=r['history_gwh'][-48:],name='Demanda de entrada',line={'color':'#596c75'}))
    band=r.get('interval',{})
    if band.get('available'):
        fig.add_trace(go.Scatter(x=r['dates'],y=np.array(band['lower_kwh'])/1e6,mode='lines',line={'width':0},showlegend=False,hoverinfo='skip'))
        fig.add_trace(go.Scatter(x=r['dates'],y=np.array(band['upper_kwh'])/1e6,mode='lines',line={'width':0},fill='tonexty',fillcolor='rgba(18,108,112,0.17)',name=f"Intervalo {100*band['level']:.0f}%"))
    fig.add_trace(go.Scatter(x=r['dates'],y=r['prediction_gwh'],name=r['name'],line={'color':'#126c70','width':3}))
    if r['detail'].get('base_gwh'):
        fig.add_trace(go.Scatter(x=r['dates'],y=r['detail']['base_gwh'],name='Base M1',line={'color':'#bd803b','dash':'dash'}))
    fig.update_layout(template='plotly_white',xaxis_title='Fecha y hora · Colombia',yaxis_title='Demanda (GWh por intervalo)',
        legend={'orientation':'h','y':1.15},margin={'t':45,'b':50,'l':65,'r':20})
    def metric(title,value,note):return html.Div([html.Span(title,className='metric-label'),html.Strong(value),html.Small(note)],className='metric')
    kpis=[metric('Energía diaria',dec(sum(r['prediction_gwh']))+' GWh','Suma de los 24 intervalos'),
        metric('Pico horario',dec(max(r['prediction_gwh']),3)+' GWh','Mayor demanda pronosticada'),
        metric('Tiempo de respuesta',dec(r['latency_ms'],1)+' ms','Preparación e inferencia; sin ajuste'),
        metric('Semilla',str(r['seed']),'Inicialización prefijada')]
    text=[html.P('La trayectoria se calcula al solicitarla. El nivel y la forma del pronóstico dependen de la historia cargada y del calendario del horizonte. Las métricas del benchmark promedian inicializaciones; esta vista utiliza el checkpoint identificado.')]
    if band.get('available'):text.append(html.P(f"Banda nominal del {100*band['level']:.0f} %, alcance {band['scope']}, calibrada con {band['n']} días hasta {band['end']}. Cobertura empírica bajo dependencia temporal."))
    elif band.get('reason'):text.append(html.P(band['reason']))
    if r['outside_domain']:
        text.append(html.P('Fuera de rango de entrenamiento: '+', '.join(r['outside_domain'])+'. La extrapolación puede reducir la precisión.',className='outcome'))
    if min(r['prediction_gwh'])<0:text.append(html.P('El modelo ha producido valores negativos; el resultado requiere revisión.',className='error'))
    if r['local_delta_gwh'] is not None:
        text.append(html.P(f"Al aumentar un 1 % la demanda de las últimas 24 horas de entrada, el pronóstico horario medio cambia {('+' if r['local_delta_gwh']>=0 else '')+dec(r['local_delta_gwh'],4)} GWh. Esta sensibilidad local mantiene las demás entradas y la memoria fijas."))
    neighbors=[]
    if r['detail'].get('neighbors'):
        neighbors=[html.H3('Episodios recuperados'),html.P(f"La compuerta aplica {dec(100*r['detail']['gate'],1)} % de la trayectoria residual recuperada."),
            html.Table([html.Thead(html.Tr([html.Th('Episodio'),html.Th('Peso de atención')])),
                html.Tbody([html.Tr([html.Td(n['fecha']),html.Td(dec(100*n['peso'])+' %')]) for n in r['detail']['neighbors']])])]
    provenance=json.dumps({k:r[k] for k in ['model','origin','seed','last_input','checkpoint_sha256','computed','training']},ensure_ascii=False,indent=2)
    return fig,{},kpis,text,neighbors,provenance

@dash.callback(Output('live-csv','data'),Input('live-download','n_clicks'),State('live-result','data'),prevent_initial_call=True)
def download(clicks,result):
    if not result:raise PreventUpdate
    band=result.get('interval',{});available=band.get('available',False)
    header='fecha,prediccion_gwh'+(',inferior_gwh,superior_gwh,nivel,alcance'if available else'')
    body=header+'\n'+'\n'.join(f'{d},{p:.9f}'+(f",{band['lower_kwh'][h]/1e6:.9f},{band['upper_kwh'][h]/1e6:.9f},{band['level']},{band['scope']}"if available else'')for h,(d,p)in enumerate(zip(result['dates'],result['prediction_gwh'])))
    return dict(content=body,filename=f"pronostico_{result['model']}_{result['origin']}.csv")

if __name__=='__main__':
    dash.run(host='127.0.0.1',port=int(os.environ.get('PORT','8815')),debug=False)

"""Comparación visual desde predicciones conservadas, sin ajuste de modelos."""

INTRO = '''(observado-prediccion)=
## 10.7. Demanda observada frente a predicción

La comparación reúne las 35.064 horas de prueba de 2022–2025 y las cinco inicializaciones de cada familia. Cada panel representa la densidad de pares observado–predicho; la diagonal indica una predicción exacta. Los puntos por encima corresponden a sobreestimación y los situados por debajo, a subestimación. Se utilizan escalas comunes para comparar la dispersión entre modelos. Las anotaciones reproducen el MAE y el RMSE medios entre inicializaciones del benchmark; no se evalúa una predicción promedio entre semillas.'''

DISPERSION = '''# Todas las horas y realizaciones individuales del benchmark.
observado_gwh = y.reshape(-1) / 1e6
observado_repetido = np.broadcast_to(y, pred[modelos[0]].shape).reshape(-1) / 1e6
extremos = [observado_gwh.min(), observado_gwh.max()]
for m in modelos:
    extremos.extend([pred[m].min() / 1e6, pred[m].max() / 1e6])
margen = 0.025 * (max(extremos) - min(extremos))
limites = (min(extremos) - margen, max(extremos) + margen)
resumen_visual = []
fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True,
                         layout='constrained')
for ax, m in zip(axes.ravel(), modelos):
    assert pred[m].shape == (5, 1461, 24)
    errores_vis = pred[m] - y[None, :, :]
    mae_vis = float(np.abs(errores_vis).mean())
    rmse_vis = float(np.sqrt(np.mean(errores_vis**2, axis=(1, 2))).mean())
    esperado = benchmark.loc[benchmark.familia.eq(m)].iloc[0]
    assert np.isclose(mae_vis, esperado.MAE_kWh, rtol=1e-10)
    assert np.isclose(rmse_vis, esperado.RMSE_kWh, rtol=1e-10)
    densidad = ax.hexbin(observado_repetido, pred[m].reshape(-1) / 1e6,
                        gridsize=65, mincnt=1, bins='log', cmap='viridis',
                        extent=(*limites, *limites), vmin=1, vmax=len(observado_repetido))
    ax.plot(limites, limites, '--', color='#c44e52', lw=1.5,
            label='Predicción exacta')
    ax.set(title=etiquetas[m], xlim=limites, ylim=limites,
           xlabel='Observado (GWh por intervalo)',
           ylabel='Predicho (GWh por intervalo)')
    ax.set_aspect('equal', adjustable='box')
    ax.text(.04, .96, f'MAE: {mae_vis / 1e6:.3f} GWh\\nRMSE: {rmse_vis / 1e6:.3f} GWh',
            transform=ax.transAxes, va='top', fontsize=9,
            bbox=dict(facecolor='white', alpha=.9, edgecolor='none'))
    ax.legend(loc='lower right', fontsize=8)
    resumen_visual.append(dict(modelo=m, horas_unicas=y.size, inicializaciones=5,
        pares_hora_semilla=errores_vis.size, MAE_kWh=mae_vis, RMSE_kWh=rmse_vis,
        sesgo_kWh=float(errores_vis.mean()),
        MAE_sobre_demanda_media_pct=100 * mae_vis / y.mean()))
fig.colorbar(densidad, ax=axes.ravel().tolist(), shrink=.8,
             label='Pares hora–semilla por celda (escala logarítmica)')
fig.savefig(OUT/'observado_vs_prediccion_2022_2025.png', dpi=170)
display(fig)
plt.close(fig)
resumen_visual = pd.DataFrame(resumen_visual)
resumen_visual.to_csv(OUT/'resumen_observado_predicho.csv', index=False)
destino_visual = ROOT/'book/_static/resultados/resumen_observado_predicho.csv'
destino_visual.parent.mkdir(parents=True, exist_ok=True)
resumen_visual.to_csv(destino_visual, index=False)
mejor_visual = resumen_visual.loc[resumen_visual.MAE_kWh.idxmin()]
mayor_visual = resumen_visual.loc[resumen_visual.MAE_kWh.idxmax()]
mae_visual_mlp = float(mejor_visual.MAE_kWh / 1e6)
ratio_visual_mlp = float(mejor_visual.MAE_sobre_demanda_media_pct)
mae_visual_lstm = float(mayor_visual.MAE_kWh / 1e6)
'''

LECTURA_DISPERSION = '''La MLP presenta el menor MAE agregado, {mae_visual_mlp:.3f} GWh por intervalo, equivalente al {ratio_visual_mlp:.2f} % de la demanda media de prueba; esta razón contextualiza la escala del error y no es MAPE. La LSTM alcanza {mae_visual_lstm:.3f} GWh. La separación respecto de la diagonal permite localizar sobreestimaciones y subestimaciones que un error medio no distingue. La densidad reúne las cinco inicializaciones sobre las mismas horas; su número de pares no representa observaciones temporales independientes. [Resumen numérico de la figura](_static/resultados/resumen_observado_predicho.csv).'''

SEMANA_INTRO = '''### Perfil horario en una semana de prueba

La ventana del 1 al 7 de julio de 2023 muestra la evolución intradiaria. La línea negra representa la demanda observada, la línea azul la media de los cinco pronósticos y la banda el rango p10–p90 entre inicializaciones. El MAE anotado promedia los errores individuales de esa semana, manteniendo el criterio de agregación del benchmark. La ventana ilustra picos y valles; no sustituye la evaluación del periodo completo.'''

SEMANA = '''import matplotlib.dates as mdates
sel = ((fechas >= pd.Timestamp('2023-07-01', tz='America/Bogota')) &
       (fechas < pd.Timestamp('2023-07-08', tz='America/Bogota')))
assert sel.sum() == 7
tiempo = pd.DatetimeIndex([d + pd.Timedelta(hours=h)
                          for d in fechas[sel] for h in range(24)])
assert len(tiempo) == 168 and tiempo.is_unique and tiempo.is_monotonic_increasing
ventana_mae = {m: float(np.abs(pred[m][:, sel, :] - y[None, sel, :]).mean())
               for m in modelos}
fig, axes = plt.subplots(2, 2, figsize=(13, 7), sharex=True, sharey=True,
                         layout='constrained')
for ax, m in zip(axes.ravel(), modelos):
    v = pred[m][:, sel, :].reshape(5, -1) / 1e6
    ax.fill_between(tiempo, np.quantile(v, .1, axis=0), np.quantile(v, .9, axis=0),
                    color='#247ba0', alpha=.2, label='p10–p90 entre semillas')
    ax.plot(tiempo, y[sel].reshape(-1) / 1e6, color='#202020', lw=1.8,
            label='Demanda observada')
    ax.plot(tiempo, v.mean(axis=0), color='#247ba0', lw=1.4, ls='--',
            label='Predicción media')
    ax.set(title=f'{etiquetas[m]} · MAE semanal {ventana_mae[m] / 1e6:.3f} GWh',
           ylabel='Demanda (GWh por intervalo)', xlabel='Fecha de 2023 · Colombia')
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2, tz=tiempo.tz))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m', tz=tiempo.tz))
    ax.legend(fontsize=7, loc='lower right')
fig.savefig(OUT/'observado_vs_prediccion_semana.png', dpi=170)
display(fig)
plt.close(fig)
mejor_ventana = etiquetas[min(ventana_mae, key=ventana_mae.get)]
mae_mejor_ventana = min(ventana_mae.values()) / 1e6
'''

LECTURA_SEMANA = '''{mejor_ventana} registra el menor MAE semanal, {mae_mejor_ventana:.3f} GWh por intervalo. Las distancias entre las curvas permiten apreciar diferencias en amplitud, picos y valles que las métricas agregadas comprimen. Una banda estrecha indica acuerdo entre inicializaciones, sin garantizar proximidad al observado ni cobertura predictiva calibrada. La media de pronósticos se utiliza para visualizar el perfil; las métricas conservan los errores de cada realización.'''


def integrar_observado(books, md, code, note):
    cells = books['10_modelos']
    start = next(i for i, c in enumerate(cells) if c.source.startswith('## 10.7.') or c.source.startswith('(observado-prediccion)='))
    end = next(i for i in range(start + 1, len(cells)) if cells[i].source.startswith('## 10.8.'))
    cells[start:end] = [md(INTRO), code(DISPERSION), note(LECTURA_DISPERSION),
                        md(SEMANA_INTRO), code(SEMANA), note(LECTURA_SEMANA)]
    for c in cells:
        if c.cell_type == 'markdown' and c.source.startswith('## 10.5.') and '#observado-prediccion' not in c.source:
            c.source += '\n\nLa [comparación entre demanda observada y predicha](#observado-prediccion) permite contextualizar estos errores mediante todas las horas de prueba y el detalle de una semana.'

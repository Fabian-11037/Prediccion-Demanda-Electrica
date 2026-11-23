"""Costes de cómputo (§4.2, §11.4) y registro de experimentos con MLflow (§4.3).

* FLOPs por muestra: ``torch.utils.flop_counter`` sobre un origen en modo evaluación (suma de las
  operaciones de multiplicación y suma que PyTorch cuenta; las operaciones sin regla se omiten).
* Memoria pico: GPU (``max_memory_allocated``) y memoria residente del proceso (muestreada cada segundo).
* Energía de GPU: integral de la potencia leída con NVML cada segundo (solo NVIDIA). Sin NVML se deja vacía.
* MLflow: cada corrida se registra en ``<resultados>/mlflow.db`` (SQLite local; ver con ``mlflow ui --backend-store-uri sqlite:///mlflow.db``).
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import torch


class Medidor:
    def __init__(self, dispositivo):
        self.dev = torch.device(dispositivo)
        self._fin = threading.Event()
        self._hilo = None
        self.rss_pico = 0.0
        self.energia_j = 0.0
        self.potencias = []
        self.t0 = self.t1 = None
        self._nvml = None
        if self.dev.type == "cuda" and torch.version.cuda:
            try:
                import pynvml
                pynvml.nvmlInit()
                self._nvml = (pynvml, pynvml.nvmlDeviceGetHandleByIndex(self.dev.index or 0))
            except Exception:
                self._nvml = None

    def _muestrear(self):
        try:
            import psutil
            proc = psutil.Process()
        except Exception:
            proc = None
        ultimo = time.time()
        while not self._fin.wait(1.0):
            ahora = time.time()
            if proc is not None:
                self.rss_pico = max(self.rss_pico, proc.memory_info().rss / 2**20)
            if self._nvml is not None:
                try:
                    w = self._nvml[0].nvmlDeviceGetPowerUsage(self._nvml[1]) / 1000.0
                    self.potencias.append(w)
                    self.energia_j += w * (ahora - ultimo)
                except Exception:
                    pass
            ultimo = ahora

    def iniciar(self):
        if self.dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.dev)
        self.t0 = time.time()
        self._hilo = threading.Thread(target=self._muestrear, daemon=True)
        self._hilo.start()

    def detener(self):
        self.t1 = time.time()
        self._fin.set()
        if self._hilo is not None:
            self._hilo.join(timeout=5)

    def resumen(self) -> dict:
        r = dict(segundos_entrenamiento=round((self.t1 or time.time()) - (self.t0 or time.time()), 1),
                 memoria_pico_proceso_mb=round(self.rss_pico, 1) or None,
                 memoria_pico_gpu_mb=round(torch.cuda.max_memory_allocated(self.dev) / 2**20, 1) if self.dev.type == "cuda" else None,
                 energia_gpu_wh=round(self.energia_j / 3600.0, 3) if self._nvml is not None else None,
                 potencia_media_gpu_w=round(sum(self.potencias) / len(self.potencias), 1) if self.potencias else None,
                 dispositivo=str(self.dev), gpu=torch.cuda.get_device_name(self.dev) if self.dev.type == "cuda" else None)
        return r


def flops_por_muestra(red, t: dict) -> dict:
    """FLOPs de una pasada hacia adelante con un origen (modo evaluación)."""
    try:
        from torch.utils.flop_counter import FlopCounterMode
        red.eval()
        llamar = lambda: red(t["x"][:1], t["cal_p"][:1], t["cal_f"][:1], t["dia"][:1])   # noqa: E731
        with torch.no_grad(), FlopCounterMode(display=False) as fc:
            llamar()
        return dict(flops_por_origen=int(fc.get_total_flops()) + _flops_lstm(red, llamar))
    except Exception as e:     # una operación sin regla de conteo no detiene el entrenamiento
        return dict(flops_por_origen=None, flops_error=str(e)[:200])


def _flops_lstm(red, llamar) -> int:
    """FlopCounterMode no cuenta la recurrencia de nn.LSTM: se suma 2 x 4 h (i + h) por paso, capa y muestra."""
    import torch
    extra, ganchos = [0], []
    for mod in red.modules():
        if isinstance(mod, torch.nn.LSTM):
            def g(m, entrada, salida):
                x = entrada[0]
                B, T = (x.shape[0], x.shape[1]) if m.batch_first else (x.shape[1], x.shape[0])
                i, h = m.input_size, m.hidden_size
                for capa in range(m.num_layers):
                    extra[0] += 2 * 4 * h * ((i if capa == 0 else h) + h) * T * B
            ganchos.append(mod.register_forward_hook(g))
    with torch.no_grad():
        llamar()
    for x in ganchos:
        x.remove()
    return extra[0]


def _plano(d: dict, prefijo: str = "") -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_plano(v, prefijo + k + "."))
        else:
            out[prefijo + k] = v if isinstance(v, (int, float, str, bool)) else str(v)
    return out


def registrar_mlflow(ctx, carpeta: Path, res: dict) -> None:
    if not getattr(ctx, "mlflow", False):
        return
    try:
        import mlflow
        mlflow.set_tracking_uri("sqlite:///" + (Path(ctx.mlflow_raiz) / "mlflow.db").resolve().as_posix())
        mlflow.set_experiment(ctx.cfg["paquete"]["clave"])
        with mlflow.start_run(run_name=str(Path(carpeta).relative_to(ctx.mlflow_raiz)).replace("\\", "/")):
            mlflow.log_params({k: v for k, v in _plano(res["configuracion"]).items()} | dict(semilla=res["semilla"], variante=res["variante"], evaluacion=res["evaluacion"]))
            m = dict(MASE=res["metricas"]["MASE"], MAE_kwh=res["metricas"]["MAE"], RMSE_kwh=res["metricas"]["RMSE"], MASE_ajuste=res["MASE_ajuste"],
                     MASE_parada=res["MASE_parada"], epocas=res["epocas"], segundos=res["segundos"], n_parametros=res["n_parametros"])
            for k, v in (res.get("costes") or {}).items():
                if isinstance(v, (int, float)) and v is not None:
                    m["coste." + k] = v
            mlflow.log_metrics(m)
            mlflow.set_tags(dict(dispositivo=res["dispositivo"], etiquetas_permutadas=str(res.get("etiquetas_permutadas", False))))
    except Exception as e:
        ctx.mlflow = False
        ctx.log(f"MLflow quedó desactivado en esta ejecución ({type(e).__name__}: {str(e)[:120]}); los resultados se siguen guardando en las carpetas.")

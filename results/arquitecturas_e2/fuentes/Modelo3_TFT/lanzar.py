"""Prepara el entorno de Python (solo la primera vez) y ejecuta el entrenamiento de este paquete.

Solo usa la biblioteca estándar. Crea la carpeta .venv dentro del paquete, instala PyTorch
(con GPU NVIDIA si la detecta) y las demás dependencias, y luego llama a ejecutar.py.
Nada se instala fuera de la carpeta del paquete.
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
VENV = RAIZ / ".venv"
PY = VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
MARCA = VENV / "entorno_listo.json"
MARCAS = ("entorno_listo.json", "modelo6_listo.json")
INDICE = "https://download.pytorch.org/whl/"


def decir(m: str) -> None:
    print(m, flush=True)


def correr(args: list, **kw) -> int:
    return subprocess.call([str(a) for a in args], **kw)


def pip(*args) -> int:
    return correr([PY, "-m", "pip", "--disable-pip-version-check", *args])


def controlador_nvidia() -> float | None:
    """Versión del controlador NVIDIA, o None si no hay GPU NVIDIA utilizable."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        s = subprocess.run([exe, "--query-gpu=driver_version,name,memory.total", "--format=csv,noheader"], capture_output=True, text=True, timeout=30)
        if s.returncode != 0 or not s.stdout.strip():
            return None
        linea = s.stdout.strip().splitlines()[0]
        decir(f"   GPU NVIDIA detectada: {linea}")
        return float(".".join(linea.split(",")[0].strip().split(".")[:2]))
    except Exception:
        return None


def torch_con_gpu() -> bool:
    return correr([PY, "-c", "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0


def torch_instalado() -> bool:
    return correr([PY, "-c", "import torch"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0


def instalar_torch() -> str:
    """Devuelve la variante instalada. Con GPU NVIDIA prueba las compilaciones CUDA compatibles con el controlador."""
    forzado = os.environ.get("ENTRENAMIENTO_TORCH_INDEX") or os.environ.get("MODELO6_TORCH_INDEX")
    if forzado:
        decir(f"   Índice de PyTorch indicado por variable de entorno: {forzado}")
        if pip("install", "torch", "--index-url", forzado) != 0:
            raise SystemExit("No se pudo instalar PyTorch desde el índice indicado.")
        return forzado
    sistema = platform.system()
    if sistema == "Darwin":
        if pip("install", "torch") != 0:
            raise SystemExit("No se pudo instalar PyTorch. Revise la conexión a internet y vuelva a ejecutar.")
        return "pypi (macOS, CPU)"
    ctrl = controlador_nvidia()
    if ctrl is not None:
        if ctrl >= 580:
            candidatos = ["cu132", "cu130", "cu129", "cu128", "cu126"]
        elif ctrl >= 525:
            candidatos = ["cu129", "cu128", "cu126", "cu124", "cu121"]
        else:
            candidatos = ["cu118"]
            decir("   El controlador de NVIDIA es antiguo; conviene actualizarlo desde nvidia.com para usar la GPU.")
        for c in candidatos:
            decir(f"   Instalando PyTorch con GPU ({c}); la descarga pesa entre 2 y 3 GB...")
            if pip("install", "torch", "--index-url", INDICE + c) != 0:
                continue
            if torch_con_gpu():
                return c
            decir(f"   La compilación {c} no pudo usar la GPU; se prueba la siguiente.")
            pip("uninstall", "-y", "torch")
        decir("   No se logró activar la GPU. Se instala la versión para CPU (el entrenamiento será mucho más lento).")
    else:
        decir("   No se detectó GPU NVIDIA: se instala PyTorch para CPU (el entrenamiento completo será lento).")
    if pip("install", "torch", "--index-url", INDICE + "cpu") != 0 and pip("install", "torch") != 0:
        raise SystemExit("No se pudo instalar PyTorch. Revise la conexión a internet y vuelva a ejecutar.")
    return "cpu"


def modulos_requeridos() -> list[str]:
    import re
    lineas = (RAIZ / "requirements.txt").read_text(encoding="utf-8").splitlines()
    return [re.split(r"[<>=!~\[ ]", x.strip(), maxsplit=1)[0] for x in lineas if x.strip() and not x.strip().startswith("#")]


def entorno_vecino() -> Path | None:
    """Entorno completo de otro paquete extraído en la misma carpeta (mismas bibliotecas)."""
    for v in sorted(RAIZ.parent.glob("*/.venv")):
        py = v / PY.relative_to(VENV)
        if v.resolve() == VENV.resolve() or not py.exists() or not any((v / m).exists() for m in MARCAS):
            continue
        if correr([py, "-c", "import torch, " + ", ".join(modulos_requeridos())], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0:
            return v
    return None


def preparar() -> None:
    global PY
    v = sys.version_info
    if not ((3, 10) <= (v.major, v.minor) <= (3, 14)) or sys.maxsize < 2**32:
        raise SystemExit(f"Se necesita Python de 64 bits, versión 3.10 a 3.14 (se encontró {platform.python_version()}). "
                         "Instale Python 3.12 desde https://www.python.org/downloads/ y vuelva a ejecutar.")
    if os.name == "nt" and len(str(RAIZ)) > 90:
        decir(f"AVISO: la ruta de la carpeta es larga ({len(str(RAIZ))} caracteres). Si aparece un error de archivo no encontrado, "
              "mueva la carpeta a una ruta corta, por ejemplo C:\\Entrenamiento, y vuelva a ejecutar.")
    libre = shutil.disk_usage(RAIZ).free / 2**30
    if libre < 10:
        decir(f"AVISO: quedan {libre:.1f} GB libres en el disco; se recomiendan al menos 10 GB.")
    if MARCA.exists() and PY.exists() and torch_instalado():
        return
    vecino = entorno_vecino()
    if vecino is not None:
        decir(f"Se usa el entorno ya instalado por el paquete vecino {vecino.parent.name}: no hay que instalar nada.")
        PY = vecino / PY.relative_to(VENV)
        return
    decir("=" * 78)
    decir("PREPARACIÓN DEL ENTORNO (solo la primera vez; necesita internet y puede tardar 10 a 30 minutos)")
    decir("=" * 78)
    if not PY.exists():
        decir("1/4 Creando el entorno aislado .venv dentro de la carpeta del paquete...")
        if correr([sys.executable, "-m", "venv", VENV]) != 0 or not PY.exists():
            raise SystemExit("No se pudo crear el entorno. En Ubuntu o Debian ejecute primero: sudo apt install python3-venv")
    decir("2/4 Actualizando pip...")
    pip("install", "--upgrade", "pip", "--quiet")
    decir("3/4 Instalando PyTorch...")
    variante = "ya instalado" if torch_instalado() else instalar_torch()
    decir("4/4 Instalando las demás bibliotecas (unos minutos, sin mensajes)...")
    if pip("install", "--quiet", "-r", RAIZ / "requirements.txt") != 0:
        raise SystemExit("No se pudieron instalar las bibliotecas. Revise la conexión a internet y vuelva a ejecutar.")
    MARCA.write_text(json.dumps(dict(torch=variante, python=platform.python_version()), indent=1), encoding="utf-8")
    decir("Entorno listo.\n")


def main() -> int:
    os.chdir(RAIZ)
    preparar()
    entorno = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    return correr([PY, RAIZ / "ejecutar.py", *sys.argv[1:]], env=entorno)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrumpido. Vuelva a ejecutar para continuar.")
        sys.exit(130)

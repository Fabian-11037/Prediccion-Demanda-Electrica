@echo off
setlocal
cd /d "%~dp0"
title Prueba rapida - Modelo 3 TFT
if not exist "lanzar.py" (
  echo.
  echo  No se encuentra lanzar.py junto a este archivo.
  echo  Primero EXTRAIGA el zip completo: clic derecho sobre el zip, "Extraer todo".
  echo  Despues abra la carpeta extraida y haga doble clic de nuevo en este archivo.
  echo.
  pause
  exit /b 1
)
set "PY="
for %%V in (3.12 3.13 3.11 3.14 3.10) do if not defined PY (
  py -%%V -c "import sys" >nul 2>nul && set "PY=py -%%V"
)
if not defined PY (
  python -c "import sys; raise SystemExit(0 if (3,10)<=sys.version_info[:2]<=(3,14) else 1)" >nul 2>nul && set "PY=python"
)
if not defined PY (
  echo.
  echo  No se encontro Python 3.10 a 3.14. Se intenta instalar Python 3.12 con winget...
  echo.
  winget install -e --id Python.Python.3.12 --scope user --accept-package-agreements --accept-source-agreements
  if exist "%LocalAppData%\Programs\Python\Python312\python.exe" set "PY="%LocalAppData%\Programs\Python\Python312\python.exe""
)
if not defined PY (
  echo.
  echo  No fue posible instalar Python automaticamente.
  echo  Instale Python 3.12 de 64 bits desde https://www.python.org/downloads/
  echo  y marque la casilla "Add python.exe to PATH". Luego vuelva a hacer doble clic aqui.
  echo.
  pause
  exit /b 1
)
%PY% lanzar.py --prueba-rapida
echo.
echo  Puede cerrar esta ventana.
pause

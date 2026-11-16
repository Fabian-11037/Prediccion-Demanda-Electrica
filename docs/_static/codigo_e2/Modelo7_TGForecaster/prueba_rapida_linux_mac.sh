#!/usr/bin/env bash
# Uso: bash prueba_rapida_linux_mac.sh
cd "$(dirname "$0")" || exit 1
PY=""
for c in python3.12 python3.13 python3.11 python3.14 python3.10 python3 python; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c "import sys; raise SystemExit(0 if (3,10)<=sys.version_info[:2]<=(3,14) else 1)" >/dev/null 2>&1; then PY="$c"; break; fi
done
if [ -z "$PY" ]; then
  echo "No se encontró Python 3.10 a 3.14. Instale Python 3.12 (https://www.python.org/downloads/) y vuelva a ejecutar."
  exit 1
fi
exec "$PY" lanzar.py --prueba-rapida "$@"

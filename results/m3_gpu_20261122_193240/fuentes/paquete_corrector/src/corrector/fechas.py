"""Extracción de intervalos de efecto anunciados en un documento (reglas de fecha en español, sin clasificar temas).

Para cada mención explícita de fecha ("15 de marzo", "15 de marzo de 2020", "15/03/2020", "del 3 al 9 de abril") se
mira el texto inmediatamente anterior:
  * disparador de inicio ("a partir del", "desde el", "inicia", "comenzará", "entrará en vigencia", "regirá"...):
    la fecha abre un intervalo; el fin es la siguiente fecha con disparador de cierre ("hasta el", "al") en los 200
    caracteres siguientes, o inicio + duración si aparece "durante N días/semanas/meses", o inicio + 6 días;
  * rango "del X al Y de mes" o "entre el X y el Y de mes": intervalo [X, Y].
Sin año explícito se usa el año de publicación, y el siguiente si la fecha quedaría más de 7 días antes de ella.
Se conservan intervalos con inicio en [publicación − 7 d, publicación + 120 d] y duración de hasta 180 días: el
documento anuncia algo próximo, no narra el pasado. Las fechas relativas ("el próximo lunes") no se interpretan.
Cada intervalo guarda la posición del carácter de la mención, para ubicar el fragmento que la contiene.
"""
from __future__ import annotations

import datetime as dt
import re
import unicodedata

MESES = {"enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6, "julio": 7, "agosto": 8,
         "septiembre": 9, "setiembre": 9, "octubre": 10, "noviembre": 11, "diciembre": 12}
_MES = "|".join(MESES)
_NUM = {"un": 1, "una": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5, "seis": 6, "siete": 7, "ocho": 8, "nueve": 9, "diez": 10,
        "quince": 15, "veinte": 20, "treinta": 30}
FECHA = re.compile(rf"\b(\d{{1,2}})\s+de\s+({_MES})(?:\s+(?:de|del)\s+(\d{{4}}))?|\b(\d{{1,2}})/(\d{{1,2}})/(\d{{4}})\b")
RANGO = re.compile(rf"\b(?:del?|entre\s+el)\s+(\d{{1,2}})\s+(?:al?|y\s+el)\s+(\d{{1,2}})\s+de\s+({_MES})(?:\s+(?:de|del)\s+(\d{{4}}))?")
INICIO = re.compile(r"(a\s+partir\s+del?|desde\s+el|desde|inicia(?:ra|n|ran)?|comenzara(?:n)?|empezara(?:n)?|arranca(?:ra)?|"
                    r"entrara\s+en\s+(?:vigencia|vigor)|entra\s+en\s+(?:vigencia|vigor)|regira(?:n)?|rige\s+desde|"
                    r"se\s+aplicara(?:n)?|tendra\s+lugar|se\s+realizara|programad[oa]s?\s+para|el\s+proximo)\s*(?:el\s+)?(?:dia\s+)?(?:lunes|martes|miercoles|jueves|viernes|sabado|domingo)?\s*,?\s*$")
FIN = re.compile(r"(hasta\s+el|hasta|al)\s*(?:proximo\s+)?(?:dia\s+)?(?:lunes|martes|miercoles|jueves|viernes|sabado|domingo)?\s*,?\s*$")
DURACION = re.compile(r"durante\s+(\d{1,3}|" + "|".join(_NUM) + r")\s+(dias|semanas|meses)")


def plano(s: str) -> str:
    """Minúsculas sin tildes, con la misma longitud que el original (las posiciones siguen siendo válidas)."""
    out = []
    for c in (s or "").lower():
        d = unicodedata.normalize("NFKD", c)
        base = "".join(x for x in d if not unicodedata.combining(x))
        out.append(base[:1] if base else " ")
    return "".join(out)


def _fecha(dia: int, mes: int, anio: int | None, pub: dt.date) -> dt.date | None:
    for a in ([anio] if anio else [pub.year, pub.year + 1]):
        try:
            f = dt.date(a, mes, dia)
        except ValueError:
            return None
        if anio or f >= pub - dt.timedelta(days=7):
            return f
    return None


def _duracion(trozo: str) -> int | None:
    m = DURACION.search(trozo)
    if not m:
        return None
    n = int(m.group(1)) if m.group(1).isdigit() else _NUM[m.group(1)]
    return n * {"dias": 1, "semanas": 7, "meses": 30}[m.group(2)]


def intervalos(texto: str, pub: dt.date, max_n: int = 5) -> list[tuple[dt.date, dt.date, int]]:
    """Intervalos (inicio, fin, posición) anunciados en ``texto`` publicado el día ``pub``."""
    t = plano(texto)
    out, usadas = [], set()

    def aceptar(ini, fin, pos):
        if ini is None or fin is None:
            return
        if not (pub - dt.timedelta(days=7) <= ini <= pub + dt.timedelta(days=120)):
            return
        if fin < ini or (fin - ini).days > 180:
            return
        if all(not (a == ini and b == fin) for a, b, _ in out):
            out.append((ini, fin, pos))

    for m in RANGO.finditer(t):
        mes, anio = MESES[m.group(3)], int(m.group(4)) if m.group(4) else None
        ini, fin = _fecha(int(m.group(1)), mes, anio, pub), _fecha(int(m.group(2)), mes, anio, pub)
        if ini and fin and fin < ini and not anio:
            fin = _fecha(int(m.group(2)), mes, ini.year + 1, pub)
        aceptar(ini, fin, m.start())
        usadas.update(range(m.start(), m.end()))
    menciones = []
    for m in FECHA.finditer(t):
        if m.start() in usadas:
            continue
        if m.group(1):
            f = _fecha(int(m.group(1)), MESES[m.group(2)], int(m.group(3)) if m.group(3) else None, pub)
        else:
            try:
                f = dt.date(int(m.group(6)), int(m.group(5)), int(m.group(4)))
            except ValueError:
                f = None
        if f:
            menciones.append((m.start(), m.end(), f))
    cerradas = set()
    for i, (a, b, f) in enumerate(menciones):
        previo = t[max(0, a - 45):a]
        mf = FIN.search(previo)
        if mf and mf.group(1).startswith("hasta") and i not in cerradas:
            aceptar(pub, f, a)                                     # vigente desde la publicación hasta la fecha
            continue
        disp = INICIO.search(previo)
        if not disp:
            continue
        puntual = disp.group(1).startswith(("tendra", "se realizara", "programad", "el proximo"))
        fin = None
        for j2, (a2, _, f2) in enumerate(menciones[i + 1:], start=i + 1):
            if a2 - b > 200:
                break
            if FIN.search(t[max(b, a2 - 20):a2]):
                fin = f2
                cerradas.add(j2)
                break
        if fin is None:
            d = _duracion(t[b:b + 200])
            fin = f + dt.timedelta(days=(d - 1) if d else (0 if puntual else 6))
        aceptar(f, fin, a)
        if len(out) >= max_n:
            break
    return out[:max_n]

"""Cribado asistido por IA: reglas explícitas sobre título, tipo y resumen; las decisiones van en columnas ia_* separadas de los revisores."""  # Propósito del módulo
import re  # Expresiones regulares para las reglas
import sys  # Lectura de argumentos
from pathlib import Path  # Rutas portables

import pandas as pd  # Manejo de la hoja de cribado

AQUI = Path(__file__).resolve().parent  # Carpeta del script
CRIBADO = AQUI / "cribado.csv"  # Hoja de cribado
MANUAL = AQUI / "cribado_ia_manual.csv"  # Decisiones tomadas leyendo títulos (id, ia_decision, ia_motivo, ia_regla)

REGLAS = [  # Reglas en orden de prioridad: (código, nombre de la regla, campo, patrón)
    ("E7", "retractado", "tipo_titulo", r"retract"),  # Publicaciones retractadas
    ("E5", "revision", "tipo_titulo", r"\breview\b|\bsurvey\b|overview|state[- ]of[- ]the[- ]art|bibliometric|systematic literature"),  # Revisiones y estados del arte
    ("E1", "edificio_o_hogar", "titulo", r"building|household|\bhomes?\b|dwelling|apartment|hvac|cooling load|heating load|office|hotel|hospital|campus|school|university|appliance|air[- ]condition"),  # Un edificio, vivienda o aparato aislados
    ("E1", "consumidor_individual", "titulo", r"individual (consumer|customer|household|user)|enterprise|industrial (user|customer|load|consumer)|factory|textile|metallurg|warehouse|airport|firm[- ]level|water utilit"),  # Un consumidor o instalación individual
    ("E1", "vehiculos_o_carga_ev", "titulo", r"electric vehicle|\bevs?\b|charging station|charging load|charging demand|chargepoint|car fleet|electric bus"),  # Recarga de vehículos eléctricos
    ("E1", "otro_dominio", "titulo", r"workload|cloud|server|data cent|virtual machine|base station|\b5g\b|traffic|natural gas|urban gas|gas field|gas load|gas demand|water demand|heat load|thermal load|district heating|solar heating|outdoor air|cchp|steam|hydrogen|aircraft|ship|vessel|crane|elevator|machine tool|reactive (power )?load|impulse power"),  # Cargas que no son demanda eléctrica agregada
    ("E2", "horizonte_largo", "titulo", r"long[- ]term|medium[- ]term|mid[- ]term|annual|yearly|monthly|seasonal load|saturat"),  # Horizonte mediano o largo
]  # Fin de las reglas
CORTO = re.compile(r"short\b|day[- ]ahead|hourly|hour[- ]ahead", re.I)  # Señal de corto plazo que anula la regla de horizonte


def aplicar(fila) -> tuple:  # Devuelve (decisión, motivo, regla) para un registro
    textos = {"titulo": str(fila["titulo"]), "tipo_titulo": f"{fila['tipo']} {fila['titulo']}"}  # Campos a los que miran las reglas
    for codigo, nombre, campo, patron in REGLAS:  # Recorre las reglas por prioridad
        if re.search(patron, textos[campo], re.I):  # La regla coincide
            sin_lstm = re.sub(r"long[- ]short[- ]term memory|long[- ]and[- ]short[- ]term memory", "", textos["titulo"], flags=re.I)  # Quita el nombre de la red LSTM, que no es un horizonte
            if codigo == "E2" and re.search(r"long[- ]term|medium[- ]term|mid[- ]term|annual|yearly|monthly|seasonal|saturat", sin_lstm, re.I) is None:  # Sin LSTM ya no hay señal de horizonte largo
                continue  # No se excluye por horizonte
            if codigo == "E2" and CORTO.search(sin_lstm) and not re.search(r"long[- ]term hourly", sin_lstm, re.I):  # El título también dice corto plazo
                continue  # No se excluye por horizonte
            return "excluir", codigo, nombre  # Excluye con el motivo de la regla
    return "", "", ""  # Sin regla: queda para lectura


def main() -> None:  # Aplica reglas y decisiones manuales
    df = pd.read_csv(CRIBADO, dtype=str, encoding="utf-8-sig").fillna("")  # Lee la hoja
    res = df.apply(aplicar, axis=1, result_type="expand")  # Aplica las reglas a cada fila
    df["ia_decision"], df["ia_motivo"], df["ia_regla"] = res[0], res[1], res[2]  # Guarda en columnas propias
    if MANUAL.exists():  # Hay decisiones leídas a mano por la IA
        man = pd.read_csv(MANUAL, dtype=str, encoding="utf-8-sig").fillna("").set_index("id")  # Las indexa por id
        leidos = df["id"].isin(man.index)  # Filas decididas leyendo el título; esta lectura corrige a las reglas
        for col in ("ia_decision", "ia_motivo", "ia_regla"):  # Columnas a completar
            df.loc[leidos, col] = df.loc[leidos, "id"].map(man[col])  # Copia la decisión leída
    if MANUAL.exists() and "--pendientes" not in sys.argv:  # Con la lectura terminada, lo no excluido pasa
        resto = (df["ia_decision"] == "") & (df["via"] == "bases")  # Registros de bases leídos sin motivo de exclusión
        df.loc[resto, ["ia_decision", "ia_regla"]] = ["incluir", "lectura_titulo"]  # En caso de duda se incluye para texto completo
    if "--pendientes" in sys.argv:  # Modo para listar lo que falta leer
        p = df[(df["ia_decision"] == "") & (df["via"] == "bases")]  # Registros de bases sin decisión
        for _, f in p.iterrows():  # Recorre los pendientes
            print(f"{f['id']}|{f['titulo']}")  # Imprime id y título
        return  # No escribe nada
    df.to_csv(CRIBADO, index=False, encoding="utf-8-sig")  # Guarda la hoja con las columnas ia_*
    print(df[df["via"] == "bases"].groupby(["ia_decision", "ia_motivo", "ia_regla"]).size().to_string())  # Resumen de decisiones


if __name__ == "__main__":  # Punto de entrada
    main()  # Ejecuta el cribado asistido

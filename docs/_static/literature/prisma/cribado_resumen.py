"""Segunda pasada del cribado asistido por IA: lee el resumen y exige comparación entre familias neuronales con métricas (enmienda I4b)."""  # Propósito del módulo
import re  # Expresiones regulares
import sys  # Argumentos de línea de comandos
from pathlib import Path  # Rutas portables

import pandas as pd  # Manejo de la hoja de cribado

AQUI = Path(__file__).resolve().parent  # Carpeta del script
CRIBADO = AQUI / "cribado.csv"  # Hoja de cribado
MANUAL = AQUI / "cribado_resumen_manual.csv"  # Correcciones tras leer resúmenes (id, ia_resumen_decision, ia_resumen_motivo)
MIN_FAMILIAS = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 2  # Familias neuronales distintas exigidas

FAMILIAS = {  # Patrones de cada familia neuronal en el resumen
    "MLP": r"\bmlps?\b|\bffnns?\b|\bfnns?\b|multi[- ]?layer|\banns?\b|\bbpnn\b|\bbp\b|back[- ]?propagation|feed[- ]?forward|multi[- ]?layer perceptron|artificial neural network|\bdnns?\b|deep neural network|\belm\b|extreme learning machine|\brbf",  # Redes densas
    "RNN": r"\brnns?\b|recurrent neural network|\belman\b|echo state|\besn\b",  # Recurrentes simples
    "LSTM": r"lstm|long short[- ]term memory",  # LSTM y BiLSTM
    "GRU": r"\bgrus?\b|bigru|gated recurrent",  # GRU y BiGRU
    "CNN": r"(?<!t)cnns?\b|convolutional neural network|\bconv1d\b",  # Convolucionales (sin TCN)
    "TCN": r"\btcns?\b|temporal convolutional",  # Convolución temporal causal
    "Transformer": r"transformer|informer|autoformer|fedformer|patchtst|\btft\b|temporal fusion",  # Atención
    "NBEATS": r"n-?beats|n-?hits",  # Bases de expansión
    "GNN": r"\bgcns?\b|graph (neural|convolutional|attention)|\bgat\b",  # Grafos
}  # Fin de las familias
METRICAS = re.compile(r"\bmape\b|\brmse\b|\bmae\b|\bmase\b|\bsmape\b|\bnrmse\b|\br2\b|r-squared|\bmse\b|accuracy|error", re.I)  # Señal de resultados numéricos
COMPARA = re.compile(r"compar|benchmark|outperform|\bthan\b|baseline|versus|\bvs\.?\b|superior|competing|against|contrast", re.I)  # Señal de comparación
INDIVIDUAL = re.compile(r"single (building|household|house|home|consumer|factory)|smart home|an office building|a commercial building|a residential building", re.I)  # Caso de un consumidor aislado detectado en el resumen
HIBRIDO = re.compile(r"\b[\w]+(?:[-+/][\w]+)+\b")  # Nombres compuestos como CNN-LSTM o VMD-SSA-LSTM


def familias_comparadas(resumen: str) -> list:  # Familias mencionadas fuera de nombres híbridos
    def quitar(m):  # Decide si un nombre compuesto es un híbrido de varias familias
        n = sum(1 for p in FAMILIAS.values() if re.search(p, m.group(0).replace("-", " ").replace("/", " ").replace("+", " "), re.I))  # Familias dentro del compuesto
        return " " if n >= 2 else m.group(0)  # Solo se retiran los híbridos (CNN-LSTM); Bi-LSTM o LSTM-based se conservan
    sueltos = HIBRIDO.sub(quitar, resumen)  # Retira los nombres híbridos del modelo propuesto
    return sorted(f for f, p in FAMILIAS.items() if re.search(p, sueltos, re.I))  # Familias que aparecen solas


def decidir(fila) -> tuple:  # Decisión de la segunda pasada para un registro
    r = str(fila["resumen"])  # Texto del resumen
    if len(r) < 200 or r.lower().startswith("[no abstract"):  # Sin resumen útil
        return "incluir", "", "sin_resumen", ""  # Se deja pasar para no excluir sin información
    if INDIVIDUAL.search(r):  # El caso es un consumidor aislado
        return "excluir", "E1", "consumidor_individual_resumen", ""  # Otro nivel de agregación
    frases = [s for s in re.split(r"(?<=[.;])\s+", r) if COMPARA.search(s)]  # Solo las oraciones que comparan modelos
    fam = familias_comparadas(" ".join(frases))  # Familias neuronales nombradas en esas oraciones
    if len(fam) < MIN_FAMILIAS:  # No compara suficientes familias
        return "excluir", "E4", f"menos_de_{MIN_FAMILIAS}_familias", ";".join(fam)  # Criterio I4b no cumplido
    if not (METRICAS.search(r) and COMPARA.search(r)):  # Sin lenguaje de comparación o métricas
        return "excluir", "E4", "sin_comparacion_o_metricas", ";".join(fam)  # Criterio I4 no cumplido
    return "incluir", "", "cumple_I4b", ";".join(fam)  # Pasa a texto completo


def main() -> None:  # Aplica la segunda pasada a los incluidos por título
    df = pd.read_csv(CRIBADO, dtype=str, encoding="utf-8-sig").fillna("")  # Lee la hoja
    objetivo = (df["via"] == "bases") & (df["ia_decision"] == "incluir")  # Solo los que pasaron el título
    for col in ("ia_resumen_decision", "ia_resumen_motivo", "ia_resumen_regla", "ia_familias"):  # Columnas nuevas
        df[col] = ""  # Vacías al inicio
    res = df[objetivo].apply(decidir, axis=1, result_type="expand")  # Decide cada registro
    df.loc[objetivo, ["ia_resumen_decision", "ia_resumen_motivo", "ia_resumen_regla", "ia_familias"]] = res.values  # Guarda
    if MANUAL.exists():  # Correcciones tras leer resúmenes
        man = pd.read_csv(MANUAL, dtype=str, encoding="utf-8-sig").fillna("").set_index("id")  # Las indexa
        leidos = df["id"].isin(man.index)  # Filas corregidas
        df.loc[leidos, "ia_resumen_decision"] = df.loc[leidos, "id"].map(man["ia_resumen_decision"])  # Decisión leída
        df.loc[leidos, "ia_resumen_motivo"] = df.loc[leidos, "id"].map(man["ia_resumen_motivo"])  # Motivo leído
        df.loc[leidos, "ia_resumen_regla"] = "lectura_resumen"  # Marca la lectura
    df.to_csv(CRIBADO, index=False, encoding="utf-8-sig")  # Guarda la hoja
    print(df[objetivo].groupby(["ia_resumen_decision", "ia_resumen_regla"]).size().to_string())  # Resumen


if __name__ == "__main__":  # Punto de entrada
    main()  # Ejecuta

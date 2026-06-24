"""
Utilidades opcionales.

Incluye una función de AUC en Python puro para que puedas validar tu modelo
de forma honesta sin depender de scikit-learn. Úsala (o no) como prefieras.
"""


def auc(puntuaciones, etiquetas):
    """
    Área bajo la curva ROC (AUC).

    - puntuaciones: lista de probabilidades predichas (floats).
    - etiquetas: lista de 0/1 (1 = ganador).

    Devuelve un valor entre 0 y 1 (0.5 = azar).
    """
    pares = sorted(zip(puntuaciones, etiquetas))
    positivos = sum(etiquetas)
    negativos = len(etiquetas) - positivos
    if positivos == 0 or negativos == 0:
        return float("nan")
    suma_rangos = 0.0
    i = 0
    while i < len(pares):
        j = i
        while j < len(pares) and pares[j][0] == pares[i][0]:
            j += 1
        rango_promedio = (i + 1 + j) / 2.0  # promedio de rangos en caso de empates
        for k in range(i, j):
            if pares[k][1] == 1:
                suma_rangos += rango_promedio
        i = j
    return (suma_rangos - positivos * (positivos + 1) / 2.0) / (positivos * negativos)

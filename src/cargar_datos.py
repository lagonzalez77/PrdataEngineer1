"""
Punto de partida para la prueba técnica.

Este script NO está completo a propósito: carga parte de los datos y deja
marcadas con TODO las decisiones que debes tomar tú. Úsalo como base, o
reemplázalo por tu propio enfoque (pandas, polars, SQL, etc.) si lo prefieres.

Usa solo la librería estándar de Python (no requiere instalar nada).
"""
import csv
import json
import os

# Carpeta raíz del lago de datos
RUTA_LAGO = os.path.join(os.path.dirname(__file__), "..", "datos", "lago")
RUTA_ENTREGA = os.path.join(os.path.dirname(__file__), "..", "entrega")


def cargar_csv(nombre):
    """Carga un CSV del lago como lista de diccionarios (una por fila)."""
    ruta = os.path.join(RUTA_LAGO, nombre)
    with open(ruta, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def cargar_jsonl(nombre):
    """Carga un archivo JSON-lines (un objeto JSON por línea) del lago."""
    ruta = os.path.join(RUTA_LAGO, nombre)
    filas = []
    with open(ruta, encoding="utf-8") as f:
        for linea in f:
            linea = linea.strip()
            if linea:
                filas.append(json.loads(linea))
    return filas


def cargar_lago():
    """
    Devuelve los diseños de ENTRENAMIENTO para empezar a trabajar.

    Es un punto de partida: carga la tabla principal. Amplíalo según necesites.
    """
    disenos = cargar_csv("pipeline_designs.csv")

    # TODO: prepara aquí los datos que tu modelo necesite.

    return disenos


def construir_features(disenos):
    """
    A partir de los diseños, construye la matriz de features para el modelo.

    TODO: elige tus variables de entrada y justifícalas en entrega/NOTES.md.
    """
    raise NotImplementedError("Completa construir_features()")


def main():
    disenos = cargar_lago()
    print(f"Diseños de entrenamiento cargados: {len(disenos)}")
    if disenos:
        print("Columnas en pipeline_designs:", ", ".join(disenos[0].keys()))

    nuevos = cargar_csv(os.path.join("holdout", "new_designs", "pipeline_designs.csv"))
    print(f"Diseños NUEVOS a predecir: {len(nuevos)}")

    # Ejemplo de entrega MÍNIMA con una predicción constante.
    # SUSTITÚYELA por la salida de tu modelo: una probabilidad real por diseño.
    os.makedirs(RUTA_ENTREGA, exist_ok=True)
    salida = os.path.join(RUTA_ENTREGA, "predictions.csv")
    with open(salida, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["design_id", "win_probability"])
        for d in nuevos:
            w.writerow([d["id"], 0.5])  # TODO: reemplaza 0.5 por la probabilidad de tu modelo
    print(f"Escrito ejemplo en {salida} (predicción constante: reemplázala por tu modelo).")


if __name__ == "__main__":
    main()

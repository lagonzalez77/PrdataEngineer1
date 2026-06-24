# Prueba técnica — Ingeniería de Datos

Bienvenido/a. Con esta prueba queremos conocer **cómo trabajas con datos y cómo razonas** al tomar decisiones técnicas. No buscamos una respuesta perfecta, sino entender tu forma de abordar un problema.

Puedes usar las herramientas y la IA que usarías en tu día a día.

---

## Contexto del proyecto

Somos una marca de comercio electrónico que vende camisetas con diseños de temática musical (guitarras, vinilos, instrumentos, baterías, etc.). Cada semana generamos cientos de diseños y solo algunos se convierten en **éxitos de venta**, a los que llamamos **"ganadores"**.

Tenemos un volcado de los datos de nuestra operación en `datos/lago/` y queremos empezar a **predecir qué diseños nuevos serán ganadores**, para decidir cuáles llevar a producción y a publicidad antes de invertir en ellos.

---

## Objetivo de la prueba

A partir de los datos en `datos/lago/`, construye un proceso que:

1. Prepare los datos para poder trabajar con ellos.
2. Entrene un modelo que prediga si un diseño será **ganador**.
3. Genere predicciones para los **diseños nuevos** que están en `datos/lago/holdout/new_designs/`.

**Definición de "ganador" (etiqueta de entrenamiento):** un diseño es ganador si en la tabla `design_market_labels` su columna `label` vale `'winner'`.

---

## Cómo empezar (setup)

Requisito: **Python 3.10+**. No necesitas nada más para arrancar.

```bash
# (opcional) entorno virtual
python3 -m venv .venv && source .venv/bin/activate

# (opcional) librerías de datos -- también puedes resolverlo solo con la librería estándar
pip install -r requirements.txt

# punto de partida
python3 src/cargar_datos.py
```

`src/cargar_datos.py` es un punto de partida e incluye `TODO`s. Puedes construir sobre él o reemplazarlo por tu propio enfoque.

---

## Qué entregar

Coloca tus resultados en la carpeta `entrega/`:

- **`entrega/predictions.csv`** — columnas exactas `design_id,win_probability`. Una fila por cada diseño en `datos/lago/holdout/new_designs/`. `win_probability` entre 0 y 1.
- **`entrega/feature_manifest.json`** — `{"features": ["...las columnas que usaste como entrada del modelo..."]}`.
- **`entrega/NOTES.md`** — en español: qué decisiones tomaste y **por qué**, y una línea con el formato exacto:
  `self_reported_validation_auc: <el AUC que esperas honestamente en datos no vistos>`.
- *(opcional)* **`entrega/silver_designs.csv`** — tu tabla preparada, si construiste una.
- Tu **código**. Commits claros y ordenados son bienvenidos: nos dicen cómo piensas.

> Los nombres de archivo de la entrega son fijos (los lee nuestro sistema de evaluación). El **contenido** va en español.

---

## Estructura del repositorio

| Ruta | Qué es |
|------|--------|
| `datos/lago/` | Los datos del proyecto. |
| `datos/lago/holdout/new_designs/` | Los **diseños nuevos** que debes predecir. |
| `src/` | Código base para arrancar. |
| `entrega/` | Donde dejas tus resultados. |
| `TAREAS.md` | El detalle de las tareas a resolver. |
| `CRITERIOS_DE_EVALUACION.md` | Qué miramos al evaluar. |

---

## Notas

- **Está bien no terminar todo.** Preferimos un trabajo bien razonado e incompleto a uno completo pero descuidado.
- **Documenta tus supuestos y decisiones** en `entrega/NOTES.md`.
- No evaluamos solo el resultado, sino **cómo llegaste a él**.

¡Mucho éxito!

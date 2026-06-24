# Tareas

No es necesario completarlas todas. **Documenta tus decisiones** en `entrega/NOTES.md` a medida que avanzas. Cada tarea también está abierta como un *issue* en este repositorio.

---

## 1. Familiarízate con los datos

Tómate un tiempo para entender los datos de `datos/lago/`: qué hay y cómo se relaciona la información de cada diseño.

---

## 2. Prepara los datos

Construye una tabla, **una fila por diseño**, con la que puedas entrenar un modelo.

---

## 3. Construye y valida el modelo

Entrena un modelo que prediga si un diseño será **ganador** (`design_market_labels.label == 'winner'`).

- Elige con criterio **qué columnas usas como entrada (features)** y justifícalo.
- **Valida tu modelo de forma honesta.** ¿Qué rendimiento esperas en diseños que nunca ha visto? Repórtalo en `NOTES.md` como `self_reported_validation_auc`.

---

## 4. Predice y entrega

- Genera `entrega/predictions.csv` para los diseños de `datos/lago/holdout/new_designs/`.
- Completa `entrega/feature_manifest.json` con las columnas que usaste.
- Termina tu `entrega/NOTES.md`: decisiones tomadas y por qué.

---

### Recomendación

Dedica tiempo a entender los datos antes de modelar. Preferimos ver buen criterio en las tareas 1–3 que un `predictions.csv` apresurado.

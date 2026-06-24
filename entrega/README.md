# Entrega

Coloca aquí tus resultados. Estos nombres de archivo son **fijos** (los lee nuestro sistema de evaluación); el contenido va en español.

- **`predictions.csv`** — columnas `design_id,win_probability`. Una fila por diseño de `datos/lago/holdout/new_designs/`, probabilidad entre 0 y 1.
- **`feature_manifest.json`** — `{"features": ["..."]}` con las columnas que usaste como entrada del modelo.
- **`NOTES.md`** — tus notas en español. Incluye obligatoriamente la línea:
  `self_reported_validation_auc: <número>`
  con el AUC que esperas honestamente en diseños no vistos.
- *(opcional)* **`silver_designs.csv`** — tu tabla limpia, una fila por diseño.

No borres este archivo.

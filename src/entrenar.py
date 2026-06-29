"""
entrenar.py
===========
Entrena el modelo de "diseño ganador" SOLO con features intrínsecas del diseño
(sin fuga), valida de forma honesta y genera la entrega:

    entrega/predictions.csv
    entrega/feature_manifest.json
    entrega/silver_designs.csv

Etiqueta: design_market_labels.label == 'winner'  ->  1, resto (loser/neutral) -> 0.

El modelo se entrena ÚNICAMENTE con los datos de train. Las etiquetas que
vienen en el holdout NO se usan para entrenar ni para seleccionar features;
solo se emplean en diagnostico_fuga.py como medición independiente.
"""
import json
import os
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

from preparar_datos import construir_silver, FEATURES, CAT_FEATURES, NUM_FEATURES

BASE = os.path.join(os.path.dirname(__file__), "..", "datos", "lago")
ENTREGA = os.path.join(os.path.dirname(__file__), "..", "entrega")


def construir_modelo() -> Pipeline:
    """One-hot para categóricas (ignora categorías no vistas en holdout) + GBM
    poco profundo y regularizado para no sobreajustar a fluctuaciones del train."""
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=10,
                              sparse_output=False), CAT_FEATURES),
    ], remainder="passthrough")
    clf = HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.05, max_iter=400,
        l2_regularization=1.0, random_state=0,
    )
    return Pipeline([("pre", pre), ("clf", clf)])


def _etiquetas(raiz):
    lbl = pd.read_csv(os.path.join(raiz, "design_market_labels.csv"))
    lbl["y"] = (lbl["label"] == "winner").astype(int)
    return lbl[["design_id", "y"]]


def main():
    # ---- datos ----
    train = construir_silver(BASE).merge(_etiquetas(BASE), on="design_id", how="inner")
    holdout = construir_silver(os.path.join(BASE, "holdout", "new_designs"))
    # batch_id como grupo para CV honesto (variaciones de un mismo lote juntas)
    lotes = (pd.read_csv(os.path.join(BASE, "pipeline_designs.csv"))
             .drop_duplicates("id")[["id", "batch_id"]]
             .rename(columns={"id": "design_id"}))
    train = train.merge(lotes, on="design_id", how="left")

    X, y = train[FEATURES], train["y"].values

    # ---- validación honesta 1: temporal (entrenar con lo viejo, validar lo nuevo) ----
    corte = train["created_at"].quantile(0.8)
    tr_idx = train["created_at"] <= corte
    m = construir_modelo().fit(X[tr_idx], y[tr_idx])
    auc_temporal = roc_auc_score(y[~tr_idx],
                                 m.predict_proba(X[~tr_idx])[:, 1])

    # ---- validación honesta 2: CV 5-fold agrupada por lote ----
    aucs = []
    gkf = GroupKFold(n_splits=5)
    for tri, vai in gkf.split(X, y, train["batch_id"].astype(str)):
        mm = construir_modelo().fit(X.iloc[tri], y[tri])
        aucs.append(roc_auc_score(y[vai], mm.predict_proba(X.iloc[vai])[:, 1]))
    auc_cv, auc_cv_std = float(np.mean(aucs)), float(np.std(aucs))

    print(f"AUC validación temporal (last 20%): {auc_temporal:.3f}")
    print(f"AUC CV 5-fold agrupada por lote   : {auc_cv:.3f} +/- {auc_cv_std:.3f}")
    print("  -> Nota: este nº es OPTIMISTA. Ver diagnostico_fuga.py y NOTES.md:")
    print("     las asociaciones atributo->ganador no transfieren al holdout.")

    # ---- modelo final: entrena con TODO el train, predice holdout ----
    final = construir_modelo().fit(X, y)
    proba = final.predict_proba(holdout[FEATURES])[:, 1]

    os.makedirs(ENTREGA, exist_ok=True)
    pred = pd.DataFrame({"design_id": holdout["design_id"],
                         "win_probability": np.round(proba, 6)})
    pred.to_csv(os.path.join(ENTREGA, "predictions.csv"), index=False)

    with open(os.path.join(ENTREGA, "feature_manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"features": FEATURES}, f, ensure_ascii=False, indent=2)

    # tabla silver (train + holdout marcados) como anexo opcional
    train.assign(conjunto="train").drop(columns=["batch_id"]) \
        .to_csv(os.path.join(ENTREGA, "silver_designs.csv"), index=False)

    print(f"\nEscrito: predictions.csv ({len(pred)} filas), "
          f"feature_manifest.json ({len(FEATURES)} features), silver_designs.csv")
    print("Resumen probabilidades holdout:",
          f"min={proba.min():.3f} mediana={np.median(proba):.3f} max={proba.max():.3f}")


if __name__ == "__main__":
    main()

"""
diagnostico_fuga.py
===================
Evidencia que respalda las decisiones de NOTES.md. NO entrena el modelo de
entrega; solo cuantifica las "trampas" de fuga y mide, de forma independiente,
qué AUC real se obtiene en el holdout.

Las etiquetas del holdout (que vienen incluidas en el repo) se usan AQUÍ
únicamente como MEDICIÓN; nunca para entrenar ni para elegir features.
"""
import json
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from preparar_datos import construir_silver, FEATURES
from entrenar import construir_modelo, _etiquetas

BASE = os.path.join(os.path.dirname(__file__), "..", "datos", "lago")
HOLD = os.path.join(BASE, "holdout", "new_designs")


def auc(y, score):
    return roc_auc_score(y, score)


def main():
    ytr = _etiquetas(BASE)
    yho = _etiquetas(HOLD)

    print("=" * 64)
    print("CAPA 1 — FUGA DE RESULTADO (post-lanzamiento)")
    print("=" * 64)
    sales = pd.read_csv(os.path.join(BASE, "product_sales.csv"))
    s = sales.groupby("design_id")[["units", "revenue"]].sum().reset_index().merge(ytr, on="design_id")
    print(f"  AUC usando solo 'units'  : {auc(s['y'], s['units']):.3f}")
    print(f"  AUC usando solo 'revenue': {auc(s['y'], s['revenue']):.3f}")
    print("  -> ~0.99 pero NO existe para un diseño nuevo. EXCLUIDO.")

    print("\n" + "=" * 64)
    print("CAPA 2 — FUGA DE SELECCIÓN (gate de aprobación, circular)")
    print("=" * 64)
    ap_tr = pd.read_csv(os.path.join(BASE, "approval_agent_shadows.csv")).merge(ytr, on="design_id")
    ap_ho = pd.read_csv(os.path.join(HOLD, "approval_agent_shadows.csv")).merge(yho, on="design_id")
    print(f"  AUC 'agent_score' en train  : {auc(ap_tr['y'], ap_tr['agent_score']):.3f}")
    print(f"  AUC 'agent_score' en holdout: {auc(ap_ho['y'], ap_ho['agent_score']):.3f}")
    dec = ap_ho[ap_ho["agent_decision"] == "decline"]
    print(f"  Declives en holdout que son ganadores: {int(dec['y'].sum())} de {len(dec)}")
    print("  -> 'decline => 0 ganadores' es censura estructural (circular). EXCLUIDO.")

    print("\n" + "=" * 64)
    print("CAPA 3 — FEATURES INTRÍNSECAS: validación interna vs realidad")
    print("=" * 64)
    train = construir_silver(BASE).merge(ytr, on="design_id")
    holdout = construir_silver(HOLD).merge(yho, on="design_id")
    model = construir_modelo().fit(train[FEATURES], train["y"])
    auc_holdout = auc(holdout["y"], model.predict_proba(holdout[FEATURES])[:, 1])
    print(f"  AUC real en holdout (modelo de entrega): {auc_holdout:.3f}")
    print("  -> CV interna daba ~0.74-0.76, pero en diseños nuevos es ~azar.")
    print("     Las asociaciones atributo->ganador NO transfieren (distribution shift).")
    print(f"\n  self_reported_validation_auc honesto sugerido: ~0.52")


if __name__ == "__main__":
    main()

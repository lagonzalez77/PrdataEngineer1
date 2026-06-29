"""
preparar_datos.py
=================
Construye una tabla SILVER con UNA FILA POR DISEÑO a partir del lago.

La función `construir_silver(raiz)` es reutilizable: se le pasa la ruta del
lago de entrenamiento o la del holdout y devuelve exactamente las mismas
columnas, de modo que el modelo ve features idénticas en train y en predicción.

Decisión central (ver entrega/NOTES.md): SOLO se incluyen features
*intrínsecas del diseño*, conocibles ANTES de decidir invertir en él. Las
columnas de resultado (ventas), de selección (gate de aprobación) y de etapa
del pipeline se excluyen a propósito porque son fuga de información.
"""
import json
import os
import re
import pandas as pd

# ----------------------------------------------------------------------------
# Definición de features (la "fuente de la verdad" del modelo)
# ----------------------------------------------------------------------------
# Categóricas intrínsecas del diseño
CAT_FEATURES = [
    "concept_origin",      # new / variation
    "generation_model",    # modelo generativo usado
    "product_type",        # tee, ...
    "provenance_class",    # trazabilidad del origen del diseño
    "best_color",          # color principal del arte
    "subject",             # tema (atributos_a)
    "instrument",          # instrumento (atributos_a)
    "era",                 # época (atributos_a)
    "emotion",             # emoción (atributos_b)
    "buyer_identity",      # identidad del comprador objetivo (atributos_b)
    "design_style",        # estilo (atributos_b)
    "composition",         # composición (atributos_b)
    "source",              # cómo entró el diseño (atributos_b)
    "brief_type",          # tipo de brief (design_briefs)
    "exploration_bucket",  # exploración/explotación (design_briefs)
    "g_sampler",           # sampler de generación (generation_runs)
    "gen_status",          # estado de la corrida de generación
]
# Numéricas intrínsecas del diseño
NUM_FEATURES = [
    "from_bestseller",     # 1 si deriva de un bestseller (source_bestseller != nulo)
    "color_count",         # nº de colores del arte
    "g_style_strength",    # parámetro de generación
    "g_steps",             # parámetro de generación
    "g_reference_used",    # 1 si usó imagen de referencia
    "cost_usd",            # coste de generación
    "output_image_px",     # ancho de la imagen generada
]
FEATURES = CAT_FEATURES + NUM_FEATURES

# Columnas EXCLUIDAS a propósito (documentadas para auditoría):
#   product_sales.units/revenue, products.*, ad_comments.*, design_versions.*  -> resultado (post-lanzamiento)
#   pipeline_designs.stage                                                     -> resultado (estado posterior)
#   approval_agent_shadows.*, design_reviews.*                                 -> selección (gate circular)
#   design_market_labels.evidence_status / contamination_level                 -> metadatos de la etiqueta
#   title / generation_prompt / hypothesis / seed                              -> texto libre o ruido memorizable
#   created_at / updated_at                                                    -> solo para split temporal, no como feature


def _norm(serie: pd.Series) -> pd.Series:
    """Normaliza texto categórico: minúsculas, sin espacios extra y unificando
    'electric_guitar' == 'electric guitar', 'bold_graphic' == 'bold graphic', etc."""
    s = serie.astype(str).str.strip().str.lower()
    s = s.str.replace("_", " ", regex=False)
    s = s.apply(lambda x: re.sub(r"\s+", " ", x) if isinstance(x, str) else x)
    s = s.replace({"nan": pd.NA, "none": pd.NA, "": pd.NA})
    return s


def _leer(raiz, nombre):
    return os.path.join(raiz, nombre)


def construir_silver(raiz: str) -> pd.DataFrame:
    """Devuelve un DataFrame con una fila por design_id y las columnas FEATURES
    (más design_id y created_at, este último solo para validación temporal)."""
    # --- tabla maestra: pipeline_designs (deduplicada por id) ---
    pdz = pd.read_csv(_leer(raiz, "pipeline_designs.csv"))
    pdz = pdz.sort_values("updated_at").drop_duplicates("id", keep="last")
    pdz["created_at"] = pd.to_datetime(
        pdz["created_at"].astype(str).str.replace("/", "-", regex=False),
        errors="coerce", utc=True,
    )
    df = pdz[["id", "concept_origin", "generation_model", "product_type",
              "provenance_class", "best_color", "source_bestseller",
              "created_at"]].rename(columns={"id": "design_id"})
    df["from_bestseller"] = df["source_bestseller"].notna().astype(int)
    df = df.drop(columns=["source_bestseller"])

    # --- atributos del diseño ---
    a = pd.read_csv(_leer(raiz, "design_attributes_a.csv"))
    b = pd.read_csv(_leer(raiz, "design_attributes_b.csv"))
    df = df.merge(a, on="design_id", how="left").merge(b, on="design_id", how="left")

    # --- brief ---
    briefs = pd.DataFrame(json.load(open(_leer(raiz, "design_briefs.json"))))
    df = df.merge(briefs[["design_id", "brief_type", "exploration_bucket"]],
                  on="design_id", how="left")

    # --- parámetros de generación (pre-lanzamiento) ---
    gen = pd.read_json(_leer(raiz, "design_generation_runs.jsonl"), lines=True)
    params = pd.json_normalize(gen["generation_params"])
    params["design_id"] = gen["design_id"].values
    gen_feats = pd.DataFrame({
        "design_id": gen["design_id"].values,
        "g_sampler": params.get("sampler"),
        "g_style_strength": params.get("style_strength"),
        "g_steps": params.get("steps"),
        "g_reference_used": params.get("reference.used"),
        "gen_status": gen["status"].values,
        "cost_usd": gen["cost_usd"].values,
        "output_image_px": gen["output_image_width"].values,
    })
    # un diseño puede tener varias corridas; nos quedamos con la última registrada
    gen_feats = gen_feats.drop_duplicates("design_id", keep="last")
    df = df.merge(gen_feats, on="design_id", how="left")
    df["g_reference_used"] = df["g_reference_used"].fillna(False).astype(int)

    # --- normalización de categóricas de texto ---
    for c in ["subject", "instrument", "era", "emotion", "buyer_identity",
              "design_style", "composition", "source", "concept_origin",
              "generation_model", "product_type", "provenance_class",
              "best_color", "brief_type", "exploration_bucket", "g_sampler",
              "gen_status"]:
        if c in df.columns:
            df[c] = _norm(df[c])

    # text_present llega como booleano/cadena en atributos_b; no es feature por
    # defecto, pero color_count sí: aseguramos tipo numérico
    for c in ["color_count", "g_style_strength", "g_steps", "cost_usd", "output_image_px"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # garantizamos que existan todas las columnas declaradas
    for c in FEATURES:
        if c not in df.columns:
            df[c] = pd.NA

    return df[["design_id", "created_at"] + FEATURES]


if __name__ == "__main__":
    base = os.path.join(os.path.dirname(__file__), "..", "datos", "lago")
    tr = construir_silver(base)
    ho = construir_silver(os.path.join(base, "holdout", "new_designs"))
    print(f"Silver TRAIN: {tr.shape[0]} diseños x {len(FEATURES)} features")
    print(f"Silver HOLDOUT: {ho.shape[0]} diseños x {len(FEATURES)} features")
    print("Features:", ", ".join(FEATURES))

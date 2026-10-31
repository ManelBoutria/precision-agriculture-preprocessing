"""preprocessing.py - final preprocessing pipeline (assembly).

Pieces:
  - Student 1: fixed leakage columns + categorical branch (impute + one-hot)
  - Student 2: outliers_scaling.py (OutlierClipper, get_scaler)   [imported if present]
  - Student 3: folds from make_splits.py (dev_index.csv, splits_validation.csv)
  - Student 4: feature selection / rebalancing slots, final assembly

Everything that LEARNS from data (imputer, clipper bounds, scaler, encoder,
selector, sampler) lives inside the Pipeline, so it is fitted on the training
fold only.  Fixed rules (dropping leakage columns) are applied once, before.

Usage (from the repo root, or add ../src to sys.path in a notebook):
    from preprocessing import load_dev_and_folds, build_pipeline
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler

try:                                   # Student 2's file
    from outliers_scaling import OutlierClipper, get_scaler, ZERO_INFLATED
    HAS_S2 = True
except ImportError:
    HAS_S2 = False
    ZERO_INFLATED = []

SEED = 42
TARGET = "environmental_condition"

# --- Fixed rules (decided by the group, NOT learned from data) -------------
LEAKAGE_COLS = ["irrigation_status", "irrigation_amount_l", "water_flow_rate_l_min"]
# soil_moisture_pct: pending the teacher's answer. If removed, add it above.

# --- Decision pending (Student 1 / Student 3): kept out of the features ----
NON_FEATURE_COLS = ["timestamp", "sensor_node_id", "field_zone"]

CAT_COLS = ["crop_type", "growth_stage", "sensor_status"]   # < 10 modalities: one-hot


# ---------------------------------------------------------------------------
# Data and folds
# ---------------------------------------------------------------------------
def load_dev_and_folds(csv_path, splits_dir="results/splits"):
    """Return X_dev, y_dev and the 30 (train_pos, val_pos) folds.

    The test set is NOT returned: use it only once, for the final evaluation.
    The row order of the CSV must not be changed (splits use original row numbers).
    """
    splits_dir = Path(splits_dir)
    df = pd.read_csv(csv_path)
    dev_idx = pd.read_csv(splits_dir / "dev_index.csv")["index"]
    dev = df.loc[dev_idx]
    y_dev = dev[TARGET]
    X_dev = dev.drop(columns=[TARGET] + LEAKAGE_COLS + NON_FEATURE_COLS)

    sv = pd.read_csv(splits_dir / "splits_validation.csv")
    folds = []
    for _, g in sv.groupby(["repetition", "fold"]):
        val = X_dev.index.get_indexer(g["index"])
        train = np.setdiff1d(np.arange(len(X_dev)), val)
        folds.append((train, val))
    return X_dev, y_dev, folds


# ---------------------------------------------------------------------------
# Preprocessor
# ---------------------------------------------------------------------------
def build_preprocessor(X, scale=True, clip=True, scaler="robust"):
    """ColumnTransformer: numeric (impute -> clip -> scale) + categorical (impute -> one-hot).

    scale=True  : SVM, logistic regression, KNN, MLP
    scale=False : tree models (Random Forest, XGBoost, LightGBM, ExtraTrees)
    """
    cat_cols = [c for c in CAT_COLS if c in X.columns]
    num_cols = [c for c in X.columns if c not in cat_cols]

    # keep a DataFrame after the imputer, because OutlierClipper(columns=[...]) needs column names
    num_steps = [("imp", SimpleImputer(strategy="median").set_output(transform="pandas"))]
    if clip:
        if HAS_S2:
            # zero-inflated columns (rainfall...) are excluded: their IQR is 0, so clipping
            # would flatten every real rain event to 0 (decision from Student 2's week 1)
            clip_cols = [c for c in num_cols if c not in ZERO_INFLATED]
            num_steps.append(("clip", OutlierClipper(columns=clip_cols, method="iqr", factor=1.5)))
        else:
            warnings.warn("outliers_scaling.py not found: outlier clipping skipped")
    if scale:
        if HAS_S2:
            num_steps.append(("scale", get_scaler(scaler)))
        else:
            num_steps.append(("scale", RobustScaler() if scaler == "robust" else StandardScaler()))

    cat_branch = SkPipeline([("imp", SimpleImputer(strategy="most_frequent")),
                             ("ohe", OneHotEncoder(handle_unknown="ignore"))])

    return ColumnTransformer([("num", SkPipeline(num_steps), num_cols),
                              ("cat", cat_branch, cat_cols)])


def build_pipeline(model, X, scale=True, clip=True, scaler="robust",
                   selector=None, sampler=None):
    """Full pipeline: preprocess -> [selector] -> [sampler] -> model.

    The sampler (SMOTE, ADASYN, RandomUnderSampler) only acts during fit,
    so validation/test rows are never resampled.
    """
    steps = [("prep", build_preprocessor(X, scale, clip, scaler))]
    if selector is not None:
        steps.append(("select", selector))
    if sampler is not None:
        steps.append(("sampler", sampler))
    steps.append(("model", model))
    return Pipeline(steps)

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, RobustScaler, StandardScaler

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
def load_dev_and_folds(csv_path, splits_dir="results/splits", temporal_features=None):
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

    # Optional: temporal features from Student 3 (src/temporal_features.py), joined on the row index.
    # The first rows have missing lags: the median imputer of the pipeline handles them.
    if temporal_features is not None:
        feats = pd.read_csv(temporal_features, index_col="index")
        X_dev = X_dev.join(feats, how="left")

    sv = pd.read_csv(splits_dir / "splits_validation.csv")
    folds = []
    for _, g in sv.groupby(["repetition", "fold"]):
        val = X_dev.index.get_indexer(g["index"])
        train = np.setdiff1d(np.arange(len(X_dev)), val)
        folds.append((train, val))
    return X_dev, y_dev, folds


def load_final_dev_and_folds(final_dev_csv="data/final_data_dev.csv", splits_dir="results/splits"):
    """Like load_dev_and_folds, but reads the dataset already cleaned, encoded and scaled by
    scale_outliers.py (data/final_data_dev.csv, development rows only, in original order).

    Use build_pipeline(..., prepared=True) with it: the preprocessing is skipped (already done).
    CAUTION: the scaler and clipping bounds of this file were learned on ALL development rows,
    including the validation rows of every fold, so cross-validation scores are slightly
    optimistic. For unbiased comparisons use load_dev_and_folds + build_pipeline(prepared=False).
    """
    splits_dir = Path(splits_dir)
    dev = pd.read_csv(final_dev_csv)
    dev_idx = pd.read_csv(splits_dir / "dev_index.csv")["index"]
    assert len(dev) == len(dev_idx), "final_data_dev.csv must have the same rows as dev_index.csv"
    dev.index = dev_idx.to_numpy()
    y_dev = dev[TARGET]
    X_dev = dev.drop(columns=[TARGET, "split"] + NON_FEATURE_COLS, errors="ignore")
    sv = pd.read_csv(splits_dir / "splits_validation.csv")
    folds = []
    for _, g in sv.groupby(["repetition", "fold"]):
        val = X_dev.index.get_indexer(g["index"])
        train = np.setdiff1d(np.arange(len(X_dev)), val)
        folds.append((train, val))
    return X_dev, y_dev, folds


def load_temporal_folds(X_dev, splits_dir="results/splits"):
    """Return the 5 (train_pos, val_pos) folds of Student 3's TimeSeriesSplit (complementary scheme).

    Train and validation are both stored in splits_temporel.csv (a 24 h gap separates them),
    so the training set is NOT "dev minus validation" here. Not stratified.
    """
    ts = pd.read_csv(Path(splits_dir) / "splits_temporel.csv")
    folds = []
    for _, g in ts.groupby("fold"):
        train = X_dev.index.get_indexer(g.loc[g["ensemble"] == "train", "index"])
        val = X_dev.index.get_indexer(g.loc[g["ensemble"] == "val", "index"])
        assert (train >= 0).all() and (val >= 0).all(), "index not found in the development set"
        folds.append((train, val))
    return folds


def get_cleaned_data(csv_path, splits_dir="results/splits", preprocessed=False,
                     scale=True, scaler="robust"):
    """Return (dev, test): the cleaned dataset, split as in make_splits.py (dev = first rows,
    test = last 20%, 48 rows = 24 h between them are in neither).

    preprocessed=False : only the fixed rules are applied (leakage columns and
        timestamp / sensor_node_id / field_zone removed). Values are untouched, nothing is
        learned from data, so it is leakage-free by construction.
    preprocessed=True  : the full preprocessing (impute, clip, scale, one-hot) FITTED ON THE
        DEVELOPMENT SET ONLY, then applied to dev and test. Use it for the final fit and
        the final test. Do NOT cross-validate on it (use build_pipeline for that).

    Both DataFrames keep the original row number in `row_index` and the target column.
    """
    splits_dir = Path(splits_dir)
    df = pd.read_csv(csv_path)
    dev_idx = pd.read_csv(splits_dir / "dev_index.csv")["index"]
    test_idx = pd.read_csv(splits_dir / "test_index.csv")["index"]
    drop = LEAKAGE_COLS + NON_FEATURE_COLS
    dev, test = df.loc[dev_idx].drop(columns=drop), df.loc[test_idx].drop(columns=drop)
    if not preprocessed:
        dev, test = dev.copy(), test.copy()
        dev.insert(0, "row_index", dev_idx.to_numpy())
        test.insert(0, "row_index", test_idx.to_numpy())
        return dev, test

    X_dev, X_test = dev.drop(columns=[TARGET]), test.drop(columns=[TARGET])
    prep = build_preprocessor(X_dev, scale=scale, scaler=scaler).fit(X_dev, dev[TARGET])
    names = [n.split("__", 1)[1] for n in prep.get_feature_names_out()]
    out = []
    for X, part, idx in ((X_dev, dev, dev_idx), (X_test, test, test_idx)):
        arr = prep.transform(X)
        arr = arr.toarray() if hasattr(arr, "toarray") else arr
        res = pd.DataFrame(arr, columns=names)
        res.insert(0, "row_index", idx.to_numpy())
        res[TARGET] = part[TARGET].to_numpy()
        out.append(res)
    return out[0], out[1]


# ---------------------------------------------------------------------------
# Preprocessor
# ---------------------------------------------------------------------------
def build_preprocessor(X, scale=True, clip=True, scaler="robust", prepared=False):
    """ColumnTransformer: numeric (impute -> clip -> scale) + categorical (impute -> one-hot).

    scale=True  : SVM, logistic regression, KNN, MLP
    scale=False : tree models (Random Forest, XGBoost, LightGBM, ExtraTrees)
    prepared=True : X is already cleaned, encoded and scaled (final_data): do nothing.
    """
    if prepared:
        return FunctionTransformer(feature_names_out="one-to-one")
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
                   selector=None, sampler=None, prepared=False):
    """Full pipeline: preprocess -> [selector] -> [sampler] -> model.

    The sampler (SMOTE, ADASYN, RandomUnderSampler) only acts during fit,
    so validation/test rows are never resampled.
    """
    steps = [("prep", build_preprocessor(X, scale, clip, scaler, prepared))]
    if selector is not None:
        steps.append(("select", selector))
    if sampler is not None:
        steps.append(("sampler", sampler))
    steps.append(("model", model))
    return Pipeline(steps)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Write the cleaned / preprocessed datasets")
    ap.add_argument("--csv", default="data/precision_agriculture_dataset.csv")
    ap.add_argument("--splits", default="results/splits")
    ap.add_argument("--out", default="data/cleaned")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for kind, flag in (("cleaned", False), ("preprocessed", True)):
        dev, test = get_cleaned_data(a.csv, a.splits, preprocessed=flag)
        dev.to_csv(out / f"precision_agriculture_{kind}_dev.csv", index=False)
        test.to_csv(out / f"precision_agriculture_{kind}_test.csv", index=False)
        print(f"{kind}: dev {dev.shape}, test {test.shape}")
    print("written to", out)

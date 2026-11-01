"""Leakage tests for the preprocessing pipeline (Student 4, week 2).

Run from the repo root:   python -m pytest tests -v
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from preprocessing import (LEAKAGE_COLS, SEED, TARGET, build_pipeline,  # noqa: E402
                           build_preprocessor, load_dev_and_folds)

CSV = ROOT / "data" / "precision_agriculture_dataset.csv"
SPLITS = ROOT / "results" / "splits"


@pytest.fixture(scope="module")
def data():
    return load_dev_and_folds(CSV, SPLITS)


# 1. The target and its known derived columns are not in the features ---------
def test_target_and_leakage_columns_not_in_features(data):
    X, y, _ = data
    assert TARGET not in X.columns
    for c in LEAKAGE_COLS:
        assert c not in X.columns, f"{c} must be removed"


def test_no_feature_is_a_copy_of_the_target(data):
    X, y, _ = data
    codes = y.astype("category").cat.codes
    for c in X.select_dtypes("number"):
        assert not np.array_equal(X[c].fillna(-1).to_numpy(), codes.to_numpy()), c


# 2. Folds and splits are clean -------------------------------------------------
def test_folds_are_disjoint_and_cover_dev(data):
    X, y, folds = data
    assert len(folds) == 30
    for tr, va in folds:
        assert len(np.intersect1d(tr, va)) == 0
        assert len(tr) + len(va) == len(X)


def test_each_repetition_validates_every_row_once(data):
    X, y, folds = data
    for r in range(3):
        seen = np.concatenate([va for _, va in folds[r * 10:(r + 1) * 10]])
        assert len(seen) == len(X) and len(np.unique(seen)) == len(X)


def test_test_set_is_disjoint_and_later_in_time():
    df = pd.read_csv(CSV)
    dev = pd.read_csv(SPLITS / "dev_index.csv")["index"]
    test = pd.read_csv(SPLITS / "test_index.csv")["index"]
    assert len(np.intersect1d(dev, test)) == 0
    t = pd.to_datetime(df["timestamp"])
    assert t.loc[dev].max() <= t.loc[test].min(), "test set must be the chronological end"


# 3. Everything learned uses the training fold only -----------------------------
def _num_pipe(prep):
    return prep.named_transformers_["num"]


def test_scaler_uses_training_statistics_only(data):
    X, y, folds = data
    tr, va = folds[0]
    prep = build_preprocessor(X, scale=True, scaler="robust").fit(X.iloc[tr], y.iloc[tr])
    num = _num_pipe(prep)
    # statistics recomputed by hand on the training fold (after impute + clip)
    Xtr_clipped = num[:-1].transform(X.iloc[tr][num.feature_names_in_]
                                     if hasattr(num, "feature_names_in_") else X.iloc[tr])
    med_train = np.median(np.asarray(Xtr_clipped, float), axis=0)
    assert np.allclose(num.named_steps["scale"].center_, med_train)
    # ... and they differ from statistics that would include the validation rows
    prep_all = build_preprocessor(X, scale=True, scaler="robust").fit(X, y)
    assert not np.allclose(_num_pipe(prep_all).named_steps["scale"].center_,
                           num.named_steps["scale"].center_, rtol=0, atol=0)


def test_validation_values_do_not_change_what_is_learned(data):
    X, y, folds = data
    tr, va = folds[0]
    Xmod = X.copy()
    num_cols = X.select_dtypes("number").columns
    Xmod.iloc[va, [X.columns.get_loc(c) for c in num_cols]] *= 1000      # corrupt validation rows only
    a = build_preprocessor(X).fit(X.iloc[tr], y.iloc[tr]).transform(X.iloc[tr])
    b = build_preprocessor(Xmod).fit(Xmod.iloc[tr], y.iloc[tr]).transform(Xmod.iloc[tr])
    assert np.allclose(np.asarray(a, float), np.asarray(b, float))


# 4. Shuffled labels must give chance-level scores ------------------------------
@pytest.mark.parametrize("model,scale", [
    (LogisticRegression(max_iter=1000, random_state=SEED), True),
    (RandomForestClassifier(100, random_state=SEED, n_jobs=1), False)])
def test_shuffled_labels_give_chance_level(data, model, scale):
    X, y, folds = data
    y_perm = pd.Series(np.random.default_rng(SEED).permutation(y.to_numpy()), index=y.index)
    scores = []
    for tr, va in folds[:3]:
        pipe = build_pipeline(model, X, scale=scale)
        pipe.fit(X.iloc[tr], y_perm.iloc[tr])
        scores.append(balanced_accuracy_score(y_perm.iloc[va], pipe.predict(X.iloc[va])))
    chance = 1 / y.nunique()
    assert abs(np.mean(scores) - chance) < 0.04, f"{np.mean(scores):.3f} vs chance {chance:.3f}"


# 5. Reproducibility -------------------------------------------------------------
def test_rerun_gives_identical_predictions(data):
    X, y, folds = data
    tr, va = folds[0]

    def run():
        pipe = build_pipeline(RandomForestClassifier(100, random_state=SEED, n_jobs=1), X, scale=False)
        pipe.fit(X.iloc[tr], y.iloc[tr])
        return pipe.predict_proba(X.iloc[va])

    assert np.array_equal(run(), run())

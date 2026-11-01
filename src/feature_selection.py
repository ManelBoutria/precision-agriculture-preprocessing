"""feature_selection.py - Student 4, week 2.

Compares three families of selectors, all fitted on the TRAINING part of each
of the 30 shared folds (3x10 repeated stratified CV from make_splits.py):

  anova : filter   - SelectKBest(f_classif)
  rfe   : wrapper  - RFE with logistic regression
  l1    : embedded - SelectFromModel(LogisticRegression(penalty="l1"))

Selection happens AFTER the preprocessor (impute/clip/scale/one-hot), on the
one-hot expanded columns, and never sees the validation fold.

Two ways to use this file:
  * build_selector(method, k_or_C)  -> a selector to pass to
        preprocessing.build_pipeline(model, X, selector=...)
  * python src/feature_selection.py -> runs the comparison, writes
        results/feature_selection_k_curve.csv
        results/feature_selection_table.csv
        figures/feature_selection_k_curve.png

k rule: smallest k whose mean macro-F1 is within TOLERANCE of the best k
(parsimony: fewer features at the same performance).
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.feature_selection import RFE, SelectFromModel, SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier

sys.path.insert(0, str(Path(__file__).parent))
from preprocessing import SEED, build_preprocessor, load_dev_and_folds  # noqa: E402

CSV = "data/precision_agriculture_dataset.csv"
K_GRID = [3, 5, 8, 10, 12, 15, 20, "all"]
C_GRID = [0.003, 0.01, 0.03, 0.1, 0.3]       # L1: smaller C -> fewer features
TOLERANCE = 0.005
FREQ_KEEP = 0.5                               # kept if selected in >= 50% of folds


def _lr(**kw):
    return LogisticRegression(max_iter=2000, random_state=SEED, **kw)


def build_selector(method, value):
    """Selector for build_pipeline(..., selector=...). value = k (anova, rfe) or C (l1)."""
    if method == "anova":
        return SelectKBest(f_classif, k=value)
    if method == "rfe":
        return RFE(_lr(), n_features_to_select=value, step=1)
    if method == "l1":
        return SelectFromModel(_l1(value), threshold=1e-8, importance_getter=_l1_importance)
    raise ValueError(method)


def _l1(C):
    # one-vs-rest liblinear: same L1 idea as multinomial saga, ~50x faster here
    return OneVsRestClassifier(LogisticRegression(penalty="l1", solver="liblinear", C=C,
                                                  max_iter=2000, random_state=SEED))


def _l1_importance(est):
    """A feature is kept if ANY class's L1 model gives it a non-zero weight."""
    return np.abs(np.vstack([e.coef_ for e in est.estimators_])).max(axis=0)


def _fit_fold(X, y, tr, va):
    """One fold: returns {(method, setting): (macro_f1, selected_mask)} and feature names."""
    prep = build_preprocessor(X, scale=True)
    Xt = prep.fit_transform(X.iloc[tr], y.iloc[tr])
    Xv = prep.transform(X.iloc[va])
    Xt, Xv = np.asarray(Xt, float), np.asarray(Xv, float)
    names = prep.get_feature_names_out()
    ytr, yva = y.iloc[tr], y.iloc[va]
    p = Xt.shape[1]

    def score(mask):
        m = _lr().fit(Xt[:, mask], ytr)
        return f1_score(yva, m.predict(Xv[:, mask]), average="macro")

    out = {}
    # ANOVA: rank by F score (train only), then take top-k for every k
    F, _ = f_classif(Xt, ytr)
    order = np.argsort(-np.nan_to_num(F))
    # RFE: one run down to 1 feature gives the full elimination ranking
    rank = RFE(_lr(), n_features_to_select=1, step=1).fit(Xt, ytr).ranking_
    rfe_order = np.argsort(rank)
    for k in K_GRID:
        kk = p if k == "all" else min(k, p)
        for method, o in (("anova", order), ("rfe", rfe_order)):
            mask = np.zeros(p, bool)
            mask[o[:kk]] = True
            out[(method, k)] = (score(mask), mask)
    for C in C_GRID:
        mask = _l1_importance(_l1(C).fit(Xt, ytr)) > 1e-8
        if mask.sum() == 0:
            continue
        out[("l1", C)] = (score(mask), mask)
    return out, names


def run(csv=CSV, n_jobs=2):
    X, y, folds = load_dev_and_folds(csv)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = Parallel(n_jobs=n_jobs)(delayed(_fit_fold)(X, y, tr, va) for tr, va in folds)
    names = res[0][1]

    rows = []
    for key in res[0][0]:
        f1s = [r[0][key][0] for r in res if key in r[0]]
        nsel = [r[0][key][1].sum() for r in res if key in r[0]]
        rows.append(dict(method=key[0], setting=key[1], n_folds=len(f1s),
                         mean_n_features=np.mean(nsel), macro_f1=np.mean(f1s), std=np.std(f1s)))
    curve = pd.DataFrame(rows)

    best = {}
    for method, g in curve.groupby("method"):
        g = g.copy()
        g["n"] = g["mean_n_features"]
        ok = g[g["macro_f1"] >= g["macro_f1"].max() - TOLERANCE].sort_values("n")
        best[method] = ok.iloc[0]["setting"]
    curve["chosen"] = [best[m] == s for m, s in zip(curve["method"], curve["setting"])]

    # selection frequency of every feature at the chosen setting, over the 30 folds
    table = pd.DataFrame(index=names)
    for method, s in best.items():
        masks = np.array([r[0][(method, s)][1] for r in res if (method, s) in r[0]])
        table[method] = masks.mean(axis=0)
    table["n_methods_keep"] = (table[list(best)] >= FREQ_KEEP).sum(axis=1)
    table = table.sort_values(["n_methods_keep", *best], ascending=False)
    return curve, table, best


if __name__ == "__main__":
    curve, table, best = run()
    Path("results").mkdir(exist_ok=True)
    Path("figures").mkdir(exist_ok=True)
    curve.to_csv("results/feature_selection_k_curve.csv", index=False)
    table.round(3).to_csv("results/feature_selection_table.csv")
    print("Chosen setting per method:", best)
    print(curve.round(4).to_string(index=False))
    print(table.round(2).to_string())

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    for m, g in curve.groupby("method"):
        g = g.sort_values("mean_n_features")
        ax.errorbar(g["mean_n_features"], g["macro_f1"], yerr=g["std"] / np.sqrt(g["n_folds"]),
                    marker="o", capsize=2, label=m)
    ax.set_xlabel("number of features kept (mean over 30 folds)")
    ax.set_ylabel("macro F1 (logistic regression)")
    ax.legend()
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig("figures/feature_selection_k_curve.png", dpi=150)

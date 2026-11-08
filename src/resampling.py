"""resampling.py - Student 4, week 2.

Compares ways of handling class imbalance (Heat_Stress 4%, High_Humidity 7%)
on the 30 shared folds. Every strategy lives INSIDE the imblearn pipeline, so
it only touches the training part of each fold; validation rows are never
resampled.

Strategies : none | class_weight | smote | adasyn | undersample
Models     : logistic regression (scaled) and random forest (unscaled)
Metrics    : macro F1, balanced accuracy, recall of each class
             (accuracy is NOT used: 43% of rows are "Optimal")

Run:  python src/resampling.py
Out:  results/resampling_summary.csv      (mean/std over 30 folds + paired diff vs none)
      results/resampling_recall_by_class.csv
      figures/resampling_comparison.png

Caveat for the report: SMOTE/ADASYN run after one-hot encoding, so the synthetic
rows can have fractional values in one-hot columns. Fine for a comparison; mention it.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import ADASYN, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score

sys.path.insert(0, str(Path(__file__).parent))
from preprocessing import (SEED, build_pipeline, load_dev_and_folds,  # noqa: E402
                           load_final_dev_and_folds)

CSV = "data/precision_agriculture_dataset.csv"
STRATEGIES = ["none", "class_weight", "smote", "adasyn", "undersample"]


def make_model(name, class_weight=None):
    if name == "logreg":
        return LogisticRegression(max_iter=2000, random_state=SEED, class_weight=class_weight)
    return RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=1,
                                  class_weight=class_weight)


def make_sampler(strategy):
    return {"smote": SMOTE(random_state=SEED),
            "adasyn": ADASYN(random_state=SEED),
            "undersample": RandomUnderSampler(random_state=SEED)}.get(strategy)


def build(model_name, strategy, X, prepared=False):
    cw = "balanced" if strategy == "class_weight" else None
    return build_pipeline(make_model(model_name, cw), X,
                          scale=(model_name == "logreg"), sampler=make_sampler(strategy),
                          prepared=prepared)


def _one(X, y, fold_id, tr, va, model_name, strategy, prepared=False):
    pipe = build(model_name, strategy, X, prepared)
    pipe.fit(X.iloc[tr], y.iloc[tr])
    pred = pipe.predict(X.iloc[va])
    yv = y.iloc[va]
    classes = sorted(y.unique())
    rec = recall_score(yv, pred, labels=classes, average=None, zero_division=0)
    return dict(fold=fold_id, model=model_name, strategy=strategy,
                macro_f1=f1_score(yv, pred, average="macro"),
                balanced_acc=balanced_accuracy_score(yv, pred),
                **{f"recall_{c}": r for c, r in zip(classes, rec)})


def run(csv=CSV, n_jobs=2, prepared=False):
    """prepared=True: read data/final_data_dev.csv (already cleaned, encoded, scaled)."""
    X, y, folds = load_final_dev_and_folds() if prepared else load_dev_and_folds(csv)
    jobs = [(i, tr, va, m, s) for i, (tr, va) in enumerate(folds)
            for m in ("logreg", "rf") for s in STRATEGIES]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rows = Parallel(n_jobs=n_jobs)(delayed(_one)(X, y, i, tr, va, m, s, prepared)
                                       for i, tr, va, m, s in jobs)
    return pd.DataFrame(rows)


def summarize(df):
    metrics = ["macro_f1", "balanced_acc"]
    base = df[df.strategy == "none"].set_index(["model", "fold"])[metrics]
    out = []
    for (m, s), g in df.groupby(["model", "strategy"]):
        g = g.set_index(["model", "fold"])
        row = dict(model=m, strategy=s, n_folds=len(g))
        for k in metrics:
            row[k] = g[k].mean()
            row[k + "_std"] = g[k].std()
            d = g[k] - base[k]
            row[k + "_diff_vs_none"] = d.mean()
            row[k + "_diff_se"] = d.std() / np.sqrt(len(d))
            row[k + "_folds_better"] = int((d > 0).sum())
        out.append(row)
    return pd.DataFrame(out)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=["raw", "final"], default="raw",
                    help="raw: original CSV + preprocessing inside each fold (default, no leakage); "
                         "final: data/final_data_dev.csv, already scaled (outputs get a _final suffix)")
    args = ap.parse_args()
    sfx = "_final" if args.data == "final" else ""
    df = run(prepared=args.data == "final")
    Path("results").mkdir(exist_ok=True)
    Path("figures").mkdir(exist_ok=True)
    df.to_csv(f"results/resampling_per_fold{sfx}.csv", index=False)
    summ = summarize(df)
    summ.to_csv(f"results/resampling_summary{sfx}.csv", index=False)
    rc = [c for c in df.columns if c.startswith("recall_")]
    rec = df.groupby(["model", "strategy"])[rc].mean()
    rec.to_csv(f"results/resampling_recall_by_class{sfx}.csv")
    pd.set_option("display.width", 200)
    print(summ[["model", "strategy", "macro_f1", "macro_f1_diff_vs_none", "macro_f1_diff_se",
                "balanced_acc", "balanced_acc_diff_vs_none", "balanced_acc_diff_se",
                "balanced_acc_folds_better"]].round(4).to_string(index=False))
    print(rec.round(3).to_string())

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
    for ax, k in zip(axes, ["macro_f1", "balanced_acc"]):
        for m, g in summ.groupby("model"):
            ax.errorbar(g["strategy"], g[k], yerr=g[k + "_std"] / np.sqrt(g["n_folds"]),
                        marker="o", capsize=2, label=m)
        ax.set_title(k)
        ax.grid(alpha=.3)
        ax.tick_params(axis="x", rotation=30)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(f"figures/resampling_comparison{sfx}.png", dpi=150)

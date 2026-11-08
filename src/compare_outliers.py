"""compare_outliers.py - Student 2, Week 2 (standalone & configurable).

Compares models WITH vs WITHOUT outlier treatment, and with different scalers, on the
SAME folds as everybody (Student 3's splits_validation.csv: 3 repetitions x 10 folds).
For each fold: train = development set minus validation rows (same logic as make_splits.py);
all preprocessing is a scikit-learn Pipeline fitted on the training rows only.

HOW TO RUN (from anywhere):   python compare_outliers.py
HOW TO ADAPT: if a teammate renames a file or a column, change ONLY the CONFIG block below.
The script looks for the files by name in its own folder, the parent folders and the
sub-folders of the project, so it normally needs no change at all.

Outputs: results_outlier_folds.csv, results_outlier_summary.csv, results_outlier_deltas.csv
"""
import os
import sys
import time
import warnings
from pathlib import Path

# ============================== CONFIG (edit here only) ==============================
SEED = 42
TARGET = "environmental_condition"
DATA_FILES = ["data_clean.csv", "precision_agriculture_dataset.csv"]   # first one found is used
DEV_INDEX_FILES = ["dev_index.csv"]
FOLDS_FILES = ["splits_validation.csv"]
FEATURE_LIST_FILES = ["Exploratory_Data_S2_feature_list.csv"]           # optional (auto-detected if absent)
FOLDS_COLUMNS = {"repetition": "repetition", "fold": "fold", "index": "index"}
# Never used as features (target, date, location, identifiers, leakage found by Student 1)
NOT_FEATURES = [TARGET, "timestamp", "latitude", "longitude", "row_id",
                "irrigation_status", "irrigation_amount_l", "water_flow_rate_l_min"]
EXTRA_DROP = []          # add "soil_moisture_pct" here if the instructor decides to remove it
EXPECTED_ROWS = 15000    # sanity check: row order of the data file must be the original one
MAX_FOLDS = int(os.environ.get("S2_MAX_FOLDS", 0)) or None   # None = all 30 folds (small number = quick test only)
# ====================================================================================

os.environ["PYTHONWARNINGS"] = "ignore"        # also silences worker processes
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.pipeline import Pipeline

HERE = Path(__file__).resolve().parent
os.chdir(HERE)
sys.path.insert(0, str(HERE))
from outliers_scaling import ZERO_INFLATED, make_outlier_scaler_steps  # noqa: E402


def find_file(names, what):
    """Look for the first existing file among `names`: script folder, 3 parent folders,
    then anywhere under the project root (parent of this folder)."""
    roots = [HERE, HERE.parent, HERE.parent.parent, HERE.parent.parent.parent]
    for name in names:
        for r in roots:
            if (r / name).is_file():
                return r / name
    project = HERE.parent
    for name in names:
        for p in project.rglob(name):
            if p.is_file() and ".git" not in p.parts and "site-packages" not in p.parts:
                return p
    raise FileNotFoundError(f"Cannot find {what}: looked for {names} in {HERE} and the project folders. "
                            f"Copy the file next to this script or change the CONFIG block.")


def first_col(df, wanted):
    """Column called `wanted`, else the first column of the file."""
    return df[wanted] if wanted in df.columns else df.iloc[:, 0]


# ---- data (ORIGINAL row numbers kept: read WITHOUT sorting/filtering, then .loc) ----
data_path = find_file(DATA_FILES, "the dataset")
df = pd.read_csv(data_path)
if EXPECTED_ROWS and len(df) != EXPECTED_ROWS:
    raise ValueError(f"{data_path.name}: expected {EXPECTED_ROWS} rows, found {len(df)}. "
                     f"The row order/number must stay the original one (indices come from Student 3).")
dev_idx = first_col(pd.read_csv(find_file(DEV_INDEX_FILES, "dev_index.csv")), "index")
folds = pd.read_csv(find_file(FOLDS_FILES, "splits_validation.csv"))
c_rep, c_fold, c_idx = (FOLDS_COLUMNS[k] for k in ("repetition", "fold", "index"))
missing = [c for c in (c_rep, c_fold, c_idx) if c not in folds.columns]
if missing:
    raise KeyError(f"Columns {missing} not found in the folds file (found: {list(folds.columns)}). "
                   f"Fix FOLDS_COLUMNS in the CONFIG block.")

# ---- features: from Week 1's list if available, else detected automatically ----
try:
    features = pd.read_csv(find_file(FEATURE_LIST_FILES, "feature list")).iloc[:, 0].tolist()
    source = "feature list file"
except FileNotFoundError:
    features = [c for c in df.select_dtypes("number").columns if c not in NOT_FEATURES]
    source = "auto-detected numeric columns"
wanted = [c for c in features if c not in EXTRA_DROP and c not in NOT_FEATURES]
absent = [c for c in wanted if c not in df.columns]
if absent:
    raise KeyError(f"{data_path.name} has no column(s) {absent}. The experiment needs the ORIGINAL numeric sensor "
                   f"columns (raw units, original names). Use the cleaned file that only drops columns.")
features = wanted

dev = df.loc[dev_idx]
_st = dev[features].agg(["mean", "std"])
if (_st.loc["mean"].abs() < 0.2).all() and _st.loc["std"].between(0.8, 1.2).all():
    raise ValueError(f"{data_path.name} looks already SCALED (all means ~0, std ~1). Clipping/scaling must be tested "
                     f"on RAW values: use the unscaled cleaned file (scaling belongs inside the Pipeline).")
print(f"Data file: {data_path.name} | Dev set: {dev[features].shape} | features: {len(features)} ({source}) | "
      f"folds: {folds[c_rep].nunique()} x {folds[c_fold].nunique()}")

# ---- experiments: (model, outlier treatment, scaler) ----
ALL_BUT_ZERO = [c for c in features if c not in ZERO_INFLATED]
TREATMENTS = {
    "no_clipping": None,
    "clip_wind_only": [c for c in ["wind_speed_m_s"] if c in features],
    "clip_all_IQR": ALL_BUT_ZERO,          # aggressive on purpose (stress test)
}
EXPERIMENTS = []
for t in TREATMENTS:                         # logistic regression (scale-sensitive) + standard scaler
    EXPERIMENTS.append(("LogReg", t, "standard"))
for s in ["none", "robust"]:                 # scaling ablation (no clipping)
    EXPERIMENTS.append(("LogReg", "no_clipping", s))
for t in TREATMENTS:                         # random forest (tree) -> unscaled
    EXPERIMENTS.append(("RandomForest", t, "none"))


def build(model_name, treatment, scaler):
    steps = make_outlier_scaler_steps(scaler != "none", scaler, TREATMENTS[treatment])
    if model_name == "LogReg":
        model = LogisticRegression(max_iter=1000, random_state=SEED)
    else:
        model = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=1)
    return Pipeline(steps + [("model", model)])


def run_fold(rep, fold, val_rows):
    warnings.filterwarnings("ignore")
    train_rows = dev_idx[~dev_idx.isin(val_rows)]
    Xtr, ytr = df.loc[train_rows, features], df.loc[train_rows, TARGET]
    Xva, yva = df.loc[val_rows, features], df.loc[val_rows, TARGET]
    out = []
    for model_name, treatment, scaler in EXPERIMENTS:
        pipe = build(model_name, treatment, scaler)
        t0 = time.time()
        pipe.fit(Xtr, ytr)                   # everything is learned on the training rows only
        pred = pipe.predict(Xva)
        out.append(dict(model=model_name, treatment=treatment, scaler=scaler,
                        repetition=rep, fold=fold,
                        macro_f1=f1_score(yva, pred, average="macro"),
                        balanced_accuracy=balanced_accuracy_score(yva, pred),
                        fit_time_s=time.time() - t0))
    return out


jobs = [(r, f, g[c_idx].to_numpy()) for (r, f), g in folds.groupby([c_rep, c_fold])]
if MAX_FOLDS:
    jobs = jobs[:MAX_FOLDS]
    print(f"[QUICK TEST] only {len(jobs)} folds")
t0 = time.time()
res = Parallel(n_jobs=-1)(delayed(run_fold)(r, f, v) for r, f, v in jobs)
res = pd.DataFrame([row for fold_rows in res for row in fold_rows])
print(f"Done in {time.time() - t0:.0f}s ({len(jobs)} folds x {len(EXPERIMENTS)} experiments)")
res.to_csv("results_outlier_folds.csv", index=False)

# ---- statistics at repetition level (PDF section 8: never t-test the 30 folds) ----
keys = ["model", "treatment", "scaler"]
per_rep = res.groupby(keys + ["repetition"])[["macro_f1", "balanced_accuracy"]].mean().reset_index()
rows = []
for k, g in per_rep.groupby(keys):
    row = dict(zip(keys, k))
    for m in ["macro_f1", "balanced_accuracy"]:
        row[f"{m}_mean"] = g[m].mean()
        row[f"{m}_std_3rep"] = g[m].std(ddof=1)
    f30 = res[(res.model == k[0]) & (res.treatment == k[1]) & (res.scaler == k[2])]
    row["macro_f1_std_30folds"] = f30["macro_f1"].std(ddof=1)
    row["fit_time_s_mean"] = f30["fit_time_s"].mean()
    rows.append(row)
summary = pd.DataFrame(rows)
summary.to_csv("results_outlier_summary.csv", index=False)

# paired deltas vs reference, repetition by repetition
refs = {"LogReg": ("no_clipping", "standard"), "RandomForest": ("no_clipping", "none")}
d_rows = []
for model, (t_ref, s_ref) in refs.items():
    ref = per_rep[(per_rep.model == model) & (per_rep.treatment == t_ref) & (per_rep.scaler == s_ref)] \
        .set_index("repetition")
    for k, g in per_rep[per_rep.model == model].groupby(["treatment", "scaler"]):
        if k == (t_ref, s_ref):
            continue
        g = g.set_index("repetition")
        d = (g["macro_f1"] - ref["macro_f1"]).dropna()
        row = dict(model=model, treatment=k[0], scaler=k[1], reference=f"{t_ref}+{s_ref}",
                   delta_macro_f1_mean=d.mean())
        for i, v in enumerate(d.tolist(), 1):
            row[f"delta_rep{i}"] = v
        row["delta_bal_acc_mean"] = (g["balanced_accuracy"] - ref["balanced_accuracy"]).mean()
        row["practically_relevant"] = abs(d.mean()) >= 0.02
        d_rows.append(row)
deltas = pd.DataFrame(d_rows)
deltas.to_csv("results_outlier_deltas.csv", index=False)

pd.set_option("display.width", 200)
print("\n=== Mean over repetitions (each = mean of 10 folds) ===")
print(summary[keys + ["macro_f1_mean", "macro_f1_std_3rep", "balanced_accuracy_mean",
                      "fit_time_s_mean"]].round(4).to_string(index=False))
print("\n=== Paired delta vs reference (macro F1), practical threshold 0.02 ===")
print(deltas.round(4).to_string(index=False))
"""Week 1 - Student 2: univariate stats, plots, outliers (standalone, no teammates needed).
Run from this folder:  python week1_student2.py
Needs next to it: precision_agriculture_dataset.csv, dev_index.csv, test_index.csv (Student 3).
All exploration is done on the 80% development set only (test 20% locked)."""
import os, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from scipy.stats import skew, kurtosis
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
# Always work in the folder of this script (fixes FileNotFoundError with Code Runner / VS Code)
os.chdir(os.path.dirname(os.path.abspath(__file__)))
SEED, TARGET = 42, "environmental_condition"
os.makedirs("plots", exist_ok=True)

# ---- columns that are NOT sensor features for outlier analysis ----
EXCLUDE = ["latitude", "longitude", TARGET]          # location + target
# Leakage found by Student 1 (irrigation_status == "On" <=> Dry_Stress, Cramer's V = 1.00).
# Kept in the univariate tables (for the report) but REMOVED from multivariate outlier detection.
LEAK = ["irrigation_amount_l", "water_flow_rate_l_min"]
# ---- physical limits (domain knowledge). None = no limit ----
LIMITS = {
    "air_temperature_c": (-50, 60), "soil_temperature_c": (-20, 70),
    "relative_humidity_pct": (0, 100), "soil_moisture_pct": (0, 100),
    "leaf_wetness_pct": (0, 100), "battery_level_pct": (0, 100),
    "soil_ph": (0, 14), "soil_ec_ds_m": (0, None),
    "nitrogen_mg_kg": (0, None), "phosphorus_mg_kg": (0, None), "potassium_mg_kg": (0, None),
    "atmospheric_pressure_hpa": (300, 1100), "solar_radiation_w_m2": (0, 1500),
    "light_intensity_lux": (0, 150000), "wind_speed_m_s": (0, 75),
    "wind_direction_deg": (0, 360), "rainfall_mm": (0, None),
    "co2_ppm": (250, 5000), "evapotranspiration_mm": (0, None),
    "irrigation_amount_l": (0, None), "water_flow_rate_l_min": (0, None),
    "signal_strength_dbm": (-120, 0),
}

# ---- 1. load + OFFICIAL split from Student 3 (dev_index.csv / test_index.csv) ----
# CONFIG: if a teammate renames a file, change ONLY these names.
from pathlib import Path
HERE = Path(__file__).resolve().parent
DATA_FILES = ["precision_agriculture_dataset.csv", "data_clean.csv"]   # original first (the audit shows the leakage columns)
DEV_INDEX_FILES, TEST_INDEX_FILES = ["dev_index.csv"], ["test_index.csv"]


def find_file(names, what):
    """First existing file: script folder, 3 parent folders, then anywhere in the project."""
    roots = [HERE, HERE.parent, HERE.parent.parent, HERE.parent.parent.parent]
    for name in names:
        for r in roots:
            if (r / name).is_file():
                return r / name
    for name in names:
        for p in HERE.parent.rglob(name):
            if p.is_file() and ".git" not in p.parts and "site-packages" not in p.parts:
                return p
    raise FileNotFoundError(f"Cannot find {what} ({names}). Copy it next to this script or edit the CONFIG names.")


def first_col(frame, wanted="index"):
    return frame[wanted] if wanted in frame.columns else frame.iloc[:, 0]


# Original row numbers: read the CSV WITHOUT sorting or filtering, then use .loc
data_path = find_file(DATA_FILES, "the dataset")
df = pd.read_csv(data_path, parse_dates=["timestamp"])
assert len(df) == 15000, f"{data_path.name}: expected 15000 rows, found {len(df)} (row order must stay the original)"
dev_ids = first_col(pd.read_csv(find_file(DEV_INDEX_FILES, "dev_index.csv")))
test_ids = first_col(pd.read_csv(find_file(TEST_INDEX_FILES, "test_index.csv")))
assert len(set(dev_ids) & set(test_ids)) == 0, "dev/test overlap!"
dev = df.loc[dev_ids].copy()          # <- ONLY this is explored (first 80% chronologically)
print(f"Data file: {data_path.name}")
print(f"Development: {dev.shape} | locked test: {len(test_ids)} rows")
print("Dev period:", dev.timestamp.min(), "->", dev.timestamp.max())


# ---- 1b. quick audit numbers (dev set) -> saved, needed in the report ----
audit = pd.DataFrame([
    dict(item="n_rows_dev", value=len(dev)), dict(item="n_cols", value=df.shape[1] - 0),
    dict(item="n_missing_cells", value=int(dev.isna().sum().sum())),
    dict(item="n_full_duplicate_rows", value=int(dev.drop(columns=[]).duplicated().sum())),
    dict(item="n_duplicate_keys(timestamp,node,zone)", value=int(dev.duplicated(["timestamp", "sensor_node_id", "field_zone"]).sum())),
    dict(item="dev_start", value=dev.timestamp.min()), dict(item="dev_end", value=dev.timestamp.max()),
])
audit.to_csv("Exploratory_Data_S2_audit.csv", index=False)

num_cols = [c for c in dev.select_dtypes("number").columns if c not in EXCLUDE + ["row_id"]]
cat_cols = [c for c in dev.select_dtypes(exclude="number").columns
            if c not in ["timestamp", TARGET]]

# ---- 2. univariate stats + IQR + z-score + physical check + clipped-at-bound ----
rows = []
for c in num_cols:
    x = dev[c].dropna()
    q1, med, q3 = x.quantile([.25, .5, .75]); iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    iqr_n = int(((x < lo) | (x > hi)).sum())
    z_n = int((np.abs((x - x.mean()) / x.std()) > 3).sum()) if x.std() > 0 else 0
    phys = np.nan
    if c in LIMITS:
        a, b = LIMITS[c]; bad = pd.Series(False, index=x.index)
        if a is not None: bad |= x < a
        if b is not None: bad |= x > b
        phys = int(bad.sum())
    rows.append(dict(
        feature=c, missing_pct=dev[c].isna().mean() * 100, mean=x.mean(), std=x.std(),
        min=x.min(), q1=q1, median=med, q3=q3, max=x.max(),
        skewness=skew(x), kurtosis_excess=kurtosis(x),
        iqr_lower=lo, iqr_upper=hi, iqr_outliers_n=iqr_n, iqr_outliers_pct=iqr_n / len(x) * 100,
        z3_outliers_n=z_n, z3_outliers_pct=z_n / len(x) * 100,
        physically_impossible_n=phys,
        n_at_min=int((x == x.min()).sum()), n_at_max=int((x == x.max()).sum()),
        n_zeros=int((x == 0).sum()), zero_pct=(x == 0).mean() * 100,
        at_bounds_pct=((x == x.min()) | (x == x.max())).mean() * 100))
stats = pd.DataFrame(rows)
def decide(r):
    if r.feature in LEAK: return "REMOVE (target leakage)"
    if r.physically_impossible_n and r.physically_impossible_n > 0: return "fix impossible values"
    if r.zero_pct > 50: return "no clipping (zero-inflated: IQR meaningless)"
    if r.iqr_outliers_pct < 1: return "no treatment needed"
    return "keep or light clipping"
stats["suggested_action"] = stats.apply(decide, axis=1)
stats.to_csv("Exploratory_Data_S2_univariate.csv", index=False)

cat = pd.DataFrame([dict(feature=c, n_modalities=dev[c].nunique(),
                         missing_pct=dev[c].isna().mean() * 100,
                         most_frequent=dev[c].mode().iloc[0]) for c in cat_cols + [TARGET]])
cat.to_csv("Exploratory_Data_S2_categorical.csv", index=False)
dev[TARGET].value_counts().rename_axis("class").reset_index(name="count") \
   .assign(pct=lambda d: d["count"] / d["count"].sum() * 100) \
   .to_csv("Exploratory_Data_S2_target_distribution.csv", index=False)

# ---- 3. multivariate outliers: Isolation Forest + LOF (exploration only) ----
multi_cols = [c for c in num_cols if c not in LEAK]
Xs = StandardScaler().fit_transform(dev[multi_cols].fillna(dev[multi_cols].median()))
dev["out_iforest"] = IsolationForest(contamination="auto", random_state=SEED).fit_predict(Xs) == -1
dev["out_lof"] = LocalOutlierFactor(n_neighbors=20, contamination=0.01).fit_predict(Xs) == -1
pd.DataFrame([
    dict(method="IsolationForest (contamination=auto)", n=dev.out_iforest.sum(), pct=dev.out_iforest.mean() * 100),
    dict(method="LOF (k=20, contamination=0.01 -> fixed 1%)", n=dev.out_lof.sum(), pct=dev.out_lof.mean() * 100),
]).to_csv("Exploratory_Data_S2_multivariate_outliers.csv", index=False)
(dev.groupby(TARGET)[["out_iforest", "out_lof"]].mean() * 100).to_csv(
    "Exploratory_Data_S2_outliers_by_class.csv")

# ---- 4. plots ----
def grid(n, ncols=4):
    nr = int(np.ceil(n / ncols))
    fig, ax = plt.subplots(nr, ncols, figsize=(4.2 * ncols, 3.2 * nr))
    return fig, np.atleast_1d(ax).ravel()
for kind in ["hist", "box"]:
    fig, axes = grid(len(num_cols))
    for ax, c in zip(axes, num_cols):
        (sns.histplot(dev[c], kde=True, ax=ax, color="teal") if kind == "hist"
         else sns.boxplot(x=dev[c], ax=ax, color="orange"))
        ax.set_title(c, fontsize=9); ax.set_xlabel("")
    for ax in axes[len(num_cols):]: ax.axis("off")
    fig.tight_layout(); fig.savefig(f"plots/numerical_{kind}.png", dpi=150); plt.close(fig)
bars = cat_cols + [TARGET]
fig, axes = grid(len(bars), 3)
for ax, c in zip(axes, bars):
    dev[c].value_counts().plot(kind="bar", ax=ax, color="purple")
    ax.set_title(f"{c} ({dev[c].nunique()} modalities)", fontsize=9)
for ax in axes[len(bars):]: ax.axis("off")
fig.tight_layout(); fig.savefig("plots/categorical_barplots.png", dpi=150); plt.close(fig)

pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
print(stats[["feature", "min", "max", "skewness", "iqr_outliers_pct", "z3_outliers_pct",
             "physically_impossible_n", "n_at_min", "n_at_max"]].round(2).to_string(index=False))
print(stats[["feature", "zero_pct", "suggested_action"]].round(1).to_string(index=False))
print(cat.to_string(index=False))
print(pd.read_csv("Exploratory_Data_S2_multivariate_outliers.csv").to_string(index=False))
print("\nOutliers % by class:\n", (dev.groupby(TARGET)[["out_iforest", "out_lof"]].mean() * 100).round(2))

# ---- 6. feature list handed to Week 2 (numeric features kept for the clipper/scaler) ----
kept = [c for c in num_cols if c not in LEAK]
pd.DataFrame({"numeric_features_for_week2": kept}).to_csv("Exploratory_Data_S2_feature_list.csv", index=False)
print("\nNumeric features for Week 2:", len(kept), "| removed as leakage:", LEAK)
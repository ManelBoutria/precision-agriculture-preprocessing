"""scale_outliers.py - takes the dataset and writes a new one with the leakage columns removed,
the outliers capped and the numeric columns scaled.

    python src/scale_outliers.py   # reads data/precision_agriculture_dataset.csv, writes data/final_data*.csv
    python src/scale_outliers.py --input data/coded_data.csv --output data/final_data.csv
    python src/scale_outliers.py --scaler standard        # instead of robust

What it does (uses Student 2's outliers_scaling.py):
  1. Outliers: values outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR] are replaced by the bound
     (rows are never removed). rainfall_mm, solar_radiation_w_m2 and light_intensity_lux are
     NOT clipped: they are zero-inflated (IQR = 0), their large values are real events.
  2. Scaling: robust (median / IQR) by default, or standard (mean / std), on the numeric columns.
  3. The 3 irrigation columns (irrigation_status, irrigation_amount_l, water_flow_rate_l_min)
     reveal the target: they are REMOVED, whatever the input file.
  4. crop_type, growth_stage and sensor_status are one-hot encoded (0/1 columns) if still text.
     timestamp, ids, zone, one-hot 0/1 columns and the target are left unchanged.
     The input can be the original dataset, data_clean_1.csv (categories as text) or coded_data.csv (one-hot).
     A saved index column ("Unnamed: 0") is dropped.

NO LEAKAGE: the clipping bounds and the scaler statistics are learned on the DEVELOPMENT rows
only (rows listed in results/splits/dev_index.csv, made by make_splits.py) and then applied to the
test rows (test_index.csv). The 48 rows between dev and test (24 h gap) are in neither and are
not written. Outputs: <output>_dev.csv
(training data), <output>_test.csv (test rows, only APPLIED with the dev statistics, never used
to learn anything) and <output> (both, with a `split` column). `--dev-only` writes the dev file
only and leaves the test rows untouched. Learning them on all 15,000 rows
would let the test set influence the training data.

For cross-validation, still use build_pipeline() from preprocessing.py with the 30 folds:
this file's dev statistics have already seen the validation rows of every fold.
It is meant for exploration and for the final fit / final test.

Also writes <output stem>_report.csv: bounds, number of clipped values and scaler statistics
per column (for the report).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from outliers_scaling import ZERO_INFLATED, OutlierClipper, get_scaler  # noqa: E402

TARGET = "environmental_condition"
CAT_COLS = ["crop_type", "growth_stage", "sensor_status"]
LEAKAGE_COLS = ["irrigation_status", "irrigation_amount_l", "water_flow_rate_l_min"]
TIME_COL = "timestamp"


def run(df, scaler_kind="robust", factor=1.5, splits_dir="results/splits"):
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")].copy()      # drop a saved index column
    # the 3 irrigation columns reveal the target (fixed group rule): always removed, also their
    # one-hot versions (irrigation_status_On ...) if the input is already encoded
    df = df.drop(columns=[c for c in df.columns if c in LEAKAGE_COLS or c.startswith("irrigation_status")])
    # categories still as text -> one-hot 0/1 columns (crop_type_Wheat ...), so the file is model-ready
    text_cats = [c for c in CAT_COLS if c in df.columns]
    df = pd.get_dummies(df, columns=text_cats, dtype=int)
    t = pd.to_datetime(df[TIME_COL])
    assert t.is_monotonic_increasing, "rows must be sorted by timestamp (do not reorder the file)"
    # dev / test rows come from Student 3's make_splits.py (row numbers of the original CSV).
    # The rows in neither (the 48-row gap = 24 h between dev and test) are left out of the output.
    dev_idx = pd.read_csv(Path(splits_dir) / "dev_index.csv")["index"]
    test_idx = pd.read_csv(Path(splits_dir) / "test_index.csv")["index"]
    df = df.reset_index(drop=True)
    is_dev = df.index.isin(dev_idx).astype(bool)
    is_test = df.index.isin(test_idx).astype(bool)

    # one-hot columns (only 0/1) are left as they are: clipping or scaling them makes no sense
    is_binary = lambda c: set(df[c].dropna().unique()) <= {0, 1}
    num_cols = [c for c in df.select_dtypes("number").columns if c != TARGET and not is_binary(c)]
    clip_cols = [c for c in num_cols if c not in ZERO_INFLATED]
    dev = df.loc[is_dev, num_cols]

    # learn on dev only
    clipper = OutlierClipper(columns=clip_cols, method="iqr", factor=factor).fit(dev)
    scaler = get_scaler(scaler_kind)
    clipped = clipper.transform(df[num_cols])
    n_clipped = (clipped.loc[is_dev | is_test, clip_cols] != df.loc[is_dev | is_test, clip_cols]).sum()
    if scaler is not None:
        scaler.fit(clipped.loc[is_dev])
        out_num = pd.DataFrame(scaler.transform(clipped), columns=num_cols, index=df.index)
    else:
        out_num = clipped

    out = df.copy()
    out[num_cols] = out_num
    out["split"] = np.where(is_dev, "dev", np.where(is_test, "test", "gap"))
    out = out[out.split != "gap"]

    rep = pd.DataFrame({"column": num_cols})
    rep["clipped"] = rep["column"].isin(clip_cols)
    rep["lower_bound"] = rep["column"].map(clipper.lower_)
    rep["upper_bound"] = rep["column"].map(clipper.upper_)
    rep["n_values_clipped"] = rep["column"].map(n_clipped).fillna(0).astype(int)
    rep["pct_clipped"] = (100 * rep["n_values_clipped"] / len(out)).round(2)
    if scaler is not None:
        center = getattr(scaler, "center_", getattr(scaler, "mean_", None))
        scale = scaler.scale_
        rep["scaler_center"] = center
        rep["scaler_scale"] = scale
    return out, rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/precision_agriculture_dataset.csv")
    ap.add_argument("--output", default="data/final_data.csv")
    ap.add_argument("--scaler", default="robust", choices=["robust", "standard", "none"])
    ap.add_argument("--splits", default="results/splits", help="folder with dev_index.csv / test_index.csv")
    ap.add_argument("--factor", type=float, default=1.5, help="IQR multiplier")
    ap.add_argument("--dev-only", action="store_true",
                    help="write only the development rows; the test rows are not transformed or written")
    a = ap.parse_args()

    df = pd.read_csv(a.input)
    out, rep = run(df, a.scaler, a.factor, a.splits)
    stem = Path(a.output).with_suffix("")
    out[out.split == "dev"].to_csv(f"{stem}_dev.csv", index=False)
    if not a.dev_only:
        out.to_csv(a.output, index=False)
        out[out.split == "test"].to_csv(f"{stem}_test.csv", index=False)
    rep_path = Path(a.output).with_name(Path(a.output).stem + "_report.csv")
    rep.to_csv(rep_path, index=False)
    print(f"rows: {len(out)} (dev {(out.split == 'dev').sum()}, test {(out.split == 'test').sum()})")
    print(f"written: {stem}_dev.csv" + ("" if a.dev_only else f", {stem}_test.csv, {a.output}") + f" and {rep_path}")
    print(rep[["column", "clipped", "n_values_clipped", "pct_clipped"]].to_string(index=False))


if __name__ == "__main__":
    main()
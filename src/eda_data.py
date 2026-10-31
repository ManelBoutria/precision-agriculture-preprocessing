"""
eda_data.py
===========
ينشئ ملف Exploratory_Data.csv: سطر واحد لكل متغير (variable) وبنفس الأعمدة
المطلوبة في مخطط الوثيقة.

ملاحظة منهجية: كل الأرقام هنا وصفية (EDA) للتقرير فقط، ولا تُستعمل لتحويل
البيانات قبل النمذجة (لتفادي data leakage).

التشغيل:
    python src/eda_data.py --data data/precision_agriculture_dataset.csv \
                           --out  results/Exploratory_Data.csv
"""

import argparse
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------
# 0) الإعدادات (غيّروها هنا فقط)
# ----------------------------------------------------------------------
SEED = 42                       # seed موحّد لكل العمليات العشوائية
TARGET = "environmental_condition"
TIME_COL = "timestamp"
ID_COLS = ["sensor_node_id"]    # معرّفات (لا تُعامل كمتغيرات عددية)
KEY_COLS = ["timestamp", "sensor_node_id", "field_zone"]   # المفتاح لكشف المكررات

# عتبات الأعلام في عمود notes
LEAK_F1 = 0.80        # شجرة بمتغير واحد تتجاوز هذا macro-F1 => مشتبه بالتسرب
CORR_THR = 0.95       # ارتباط مرتفع
VIF_THR = 10.0        # VIF مرتفع
OUT_THR = 5.0         # % شواذ IQR مرتفعة
SKEW_THR = 1.0        # التواء مرتفع
ASSOC_THR = 0.90      # Cramer's V >= هذا => ارتباط شبه كامل بالـ target
ETA2_THR = 0.50       # eta^2 (ANOVA) >= هذا => المتغير يفسّر نصف تباين الـ target
CONTAM = 0.01         # نسبة الشواذ المتوقعة لـ Isolation Forest و LOF (1%)

# تصنيف المتغيرات إلى عائلات (family) كما في الوثيقة
FAMILY = {
    "timestamp": "Temporal",
    "sensor_node_id": "IoT", "field_zone": "Spatial",
    "latitude": "Spatial", "longitude": "Spatial",
    "crop_type": "Crop", "growth_stage": "Crop",
    "air_temperature_c": "Environmental", "relative_humidity_pct": "Environmental",
    "wind_speed_m_s": "Environmental", "wind_direction_deg": "Environmental",
    "rainfall_mm": "Environmental", "leaf_wetness_pct": "Environmental",
    "evapotranspiration_mm": "Environmental",
    "soil_temperature_c": "Soil", "soil_moisture_pct": "Soil", "soil_ph": "Soil",
    "soil_ec_ds_m": "Soil", "nitrogen_mg_kg": "Soil",
    "phosphorus_mg_kg": "Soil", "potassium_mg_kg": "Soil",
    "atmospheric_pressure_hpa": "Atmospheric", "solar_radiation_w_m2": "Atmospheric",
    "light_intensity_lux": "Atmospheric", "co2_ppm": "Atmospheric",
    "irrigation_status": "Irrigation", "irrigation_amount_l": "Irrigation",
    "water_flow_rate_l_min": "Irrigation",
    "battery_level_pct": "IoT", "signal_strength_dbm": "IoT", "sensor_status": "IoT",
    TARGET: "Target",
}


# ----------------------------------------------------------------------
# 1) تحديد نوع كل عمود: numeric / categorical / datetime / id / target
# ----------------------------------------------------------------------
def infer_type(col: str, s: pd.Series) -> str:
    if col == TARGET:
        return "Target"
    if col == TIME_COL:
        return "datetime"
    if col in ID_COLS:
        return "id"
    if pd.api.types.is_numeric_dtype(s):
        return "numeric"
    return "categorical"


# ----------------------------------------------------------------------
# 2) إحصاءات وصفية للمتغيرات العددية
# ----------------------------------------------------------------------
def numeric_stats(s: pd.Series) -> dict:
    x = s.dropna()
    q1, med, q3 = x.quantile([0.25, 0.5, 0.75])
    return {
        "min": x.min(), "q1": q1, "median": med, "q3": q3, "max": x.max(),
        "mean": x.mean(), "std": x.std(),
        "skewness": x.skew(),          # الالتواء
        "kurtosis": x.kurt(),          # التفرطح الزائد (excess): 0 = توزيع طبيعي
        "iqr": q3 - q1,
    }


# ----------------------------------------------------------------------
# 3) كشف الشواذ بأربع طرق (لكل متغير عددي على حدة، أحادي البعد)
# ----------------------------------------------------------------------
def outlier_stats(s: pd.Series, n_total: int) -> dict:
    x = s.dropna().to_numpy().reshape(-1, 1)
    flat = x.ravel()

    # (أ) IQR: خارج [Q1 - 1.5*IQR , Q3 + 1.5*IQR]
    q1, q3 = np.percentile(flat, [25, 75])
    iqr = q3 - q1
    n_iqr = int(((flat < q1 - 1.5 * iqr) | (flat > q3 + 1.5 * iqr)).sum())

    # (ب) z-score: |z| > 3
    sd = flat.std()
    n_z = int((np.abs((flat - flat.mean()) / sd) > 3).sum()) if sd > 0 else 0

    # (ج) Isolation Forest (contamination=CONTAM)
    if np.unique(flat).size > 1:
        iso = IsolationForest(random_state=SEED, contamination=CONTAM, n_jobs=-1)
        n_if = int((iso.fit_predict(x) == -1).sum())
        # (د) LOF (contamination=CONTAM)
        lof = LocalOutlierFactor(n_neighbors=20, contamination=CONTAM, n_jobs=-1)
        n_lof = int((lof.fit_predict(x) == -1).sum())
    else:                                    # متغير ثابت: لا شواذ
        n_if = n_lof = 0

    pct = lambda n: round(100.0 * n / n_total, 2)
    return {"n_out_iqr": n_iqr, "pct_out_iqr": pct(n_iqr),
            "n_out_z": n_z, "pct_out_z": pct(n_z),
            "n_out_if": n_if, "pct_out_if": pct(n_if),
            "n_out_lof": n_lof, "pct_out_lof": pct(n_lof)}


# ----------------------------------------------------------------------
# 4) الارتباط الأعظمي (Pearson) و VIF لكل متغير عددي
# ----------------------------------------------------------------------
def corr_and_vif(df: pd.DataFrame, num_cols: list):
    """يُرجع قاموسين: (max_abs_corr, corr_partner) و vif لكل متغير."""
    X = df[num_cols].dropna()
    corr_arr = X.corr().abs().to_numpy().copy()
    np.fill_diagonal(corr_arr, 0.0)          # نستبعد ارتباط المتغير بنفسه
    corr = pd.DataFrame(corr_arr, index=num_cols, columns=num_cols)

    corr_info = {c: (corr[c].max(), corr[c].idxmax()) for c in num_cols}

    # VIF_i = العنصر i في القطر من معكوس مصفوفة الارتباط
    # (يعادل 1/(1-R^2) من انحدار المتغير على بقية المتغيرات)
    R = X.corr().to_numpy()
    vif_vals = np.diag(np.linalg.pinv(R))
    vif = dict(zip(num_cols, vif_vals))
    return corr_info, vif


# ----------------------------------------------------------------------
# 5) الارتباط بالـ target: ANOVA للعددي، chi2 للفئوي
# ----------------------------------------------------------------------
def target_association(s: pd.Series, y: pd.Series, kind: str):
    mask = s.notna()
    s, y = s[mask], y[mask]
    if kind == "numeric":
        groups = [s[y == c].to_numpy() for c in y.unique()]
        if s.nunique() < 2:
            return "", np.nan
        F, p = stats.f_oneway(*groups)
        k, N = len(groups), len(s)
        eta2 = F * (k - 1) / (F * (k - 1) + (N - k))   # حجم الأثر: نسبة التباين المفسَّر
        return f"anova_F={F:.1f}; eta2={eta2:.2f}", p
    # فئوي / id: اختبار chi2 على جدول التقاطع
    table = pd.crosstab(s, y)
    chi2, p, _, _ = stats.chi2_contingency(table)
    n = table.to_numpy().sum()
    cramers_v = np.sqrt(chi2 / (n * (min(table.shape) - 1)))
    return f"chi2={chi2:.1f}; cramers_v={cramers_v:.2f}", p


# ----------------------------------------------------------------------
# 6) اختبار التسرب: شجرة عمقها 2 على متغير واحد فقط (macro-F1 بـ CV)
#    إن كان متغير وحده يفسّر الـ target تقريباً => مشتبه بالتسرب
# ----------------------------------------------------------------------
def single_feature_f1(s: pd.Series, y: pd.Series, kind: str) -> float:
    if kind == "datetime":
        return np.nan
    if kind == "numeric":
        X = s.fillna(s.median()).to_numpy().reshape(-1, 1)
    else:                                    # فئوي: ترميز one-hot
        X = pd.get_dummies(s.astype("string").fillna("missing")).to_numpy()
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    tree = DecisionTreeClassifier(max_depth=2, random_state=SEED)
    return cross_val_score(tree, X, y, cv=cv, scoring="f1_macro").mean()


# ----------------------------------------------------------------------
# 7) بناء الملاحظات (notes) تلقائياً من الأرقام السابقة
# ----------------------------------------------------------------------
def build_notes(row: dict, leak_f1: float) -> str:
    notes = []
    # قاعدة تسرب إضافية: ارتباط شبه كامل بالـ target (يلتقط المتغيرات الثنائية
    # مثل irrigation_status التي لا تستطيع شجرة بمتغير واحد فصل 5 فئات بها)
    assoc = str(row.get("target_assoc", ""))
    strong = False
    if "cramers_v=" in assoc:
        strong = float(assoc.split("cramers_v=")[1]) >= ASSOC_THR
    if "eta2=" in assoc:
        strong = float(assoc.split("eta2=")[1]) >= ETA2_THR
    if strong:
        notes.append("LEAKAGE SUSPECT (very strong association with target)")
    if not np.isnan(leak_f1):
        notes.append(f"single-var tree macroF1={leak_f1:.2f}")
        if leak_f1 >= LEAK_F1 and not strong:
            notes.insert(0, "LEAKAGE SUSPECT")
    if row["type"] == "numeric":
        if row["n_unique"] <= 1:
            notes.append("CONSTANT")
        if row["max_abs_corr"] > CORR_THR:
            notes.append(f"collinear with {row['corr_partner']} (|r|={row['max_abs_corr']:.2f})")
        if row["vif"] > VIF_THR:
            notes.append(f"HIGH VIF ({row['vif']:.1f})")
        if row["pct_out_iqr"] > OUT_THR:
            notes.append(f"many IQR outliers ({row['pct_out_iqr']}%)")
        if abs(row["skewness"]) > SKEW_THR:
            notes.append(f"skewed ({row['skewness']:.2f})")
        if row["pct_missing"] > 0:
            notes.append("has missing values")
    return "; ".join(notes)


# ----------------------------------------------------------------------
# 8) الدالة الرئيسية
# ----------------------------------------------------------------------
def main(data_path: str, out_path: str):
    df = pd.read_csv(data_path)
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
    n_total = len(df)
    y = df[TARGET]

    # أنواع الأعمدة
    types = {c: infer_type(c, df[c]) for c in df.columns}
    num_cols = [c for c in df.columns if types[c] == "numeric"]

    # مكررات المفتاح (timestamp + node + zone) تُسجَّل في كل الأسطر
    n_dup_key = int(df.duplicated(subset=KEY_COLS).sum())
    n_dup_rows = int(df.duplicated().sum())

    corr_info, vif = corr_and_vif(df, num_cols)

    rows = []
    for col in df.columns:
        s, kind = df[col], types[col]
        row = {
            "variable": col,
            "family": FAMILY.get(col, "Other"),
            "type": kind,
            "n_total": n_total,
            "n_missing": int(s.isna().sum()),
            "pct_missing": round(100.0 * s.isna().mean(), 2),
            "n_unique": int(s.nunique()),
            "n_duplicates_key": n_dup_key,
        }

        # --- أعمدة خاصة بالعددي فقط (تبقى فارغة لغيره) ---
        num_keys = ["min", "q1", "median", "q3", "max", "mean", "std", "skewness",
                    "kurtosis", "iqr", "n_out_iqr", "pct_out_iqr", "n_out_z",
                    "pct_out_z", "n_out_if", "pct_out_if", "n_out_lof",
                    "pct_out_lof", "max_abs_corr", "corr_partner", "vif"]
        row.update({k: np.nan for k in num_keys})
        if kind == "numeric":
            row.update(numeric_stats(s))
            row.update(outlier_stats(s, n_total))
            row["max_abs_corr"], row["corr_partner"] = corr_info[col]
            row["vif"] = vif[col]

        # --- عدد الأنماط (modalities) للفئوي/المعرّف/الـ target ---
        row["modalities"] = int(s.nunique()) if kind in ("categorical", "id", "Target") else np.nan

        # --- الارتباط بالـ target (لا معنى له للـ target نفسه ولا للتاريخ) ---
        if kind in ("numeric", "categorical", "id"):
            row["target_assoc"], row["target_pval"] = target_association(s, y, kind)
            f1 = single_feature_f1(s, y, kind)
        else:
            row["target_assoc"], row["target_pval"], f1 = "", np.nan, np.nan

        row["notes"] = build_notes(row, f1)
        rows.append(row)

    # ترتيب الأعمدة حسب مخطط الوثيقة
    order = ["variable", "family", "type", "n_total", "n_missing", "pct_missing",
             "n_unique", "n_duplicates_key", "min", "q1", "median", "q3", "max",
             "mean", "std", "skewness", "kurtosis", "iqr", "n_out_iqr",
             "pct_out_iqr", "n_out_z", "pct_out_z", "n_out_if", "pct_out_if",
             "n_out_lof", "pct_out_lof", "max_abs_corr", "corr_partner", "vif",
             "modalities", "target_assoc", "target_pval", "notes"]
    out = pd.DataFrame(rows)[order]

    # الأعمدة العدّادية تبقى أعداداً صحيحة رغم وجود خانات فارغة (Int64 يقبل NA)
    for c in ["n_out_iqr", "n_out_z", "n_out_if", "n_out_lof", "modalities"]:
        out[c] = out[c].astype("Int64")

    # تدوير الأرقام العشرية (الصيغة: UTF-8، فاصلة، نقطة عشرية)
    float_cols = out.select_dtypes("float").columns.drop("target_pval")
    out[float_cols] = out[float_cols].round(4)
    out.to_csv(out_path, index=False, encoding="utf-8")

    # ملخص سريع على الشاشة
    print(f"rows={n_total}, columns={df.shape[1]}, full-row duplicates={n_dup_rows}, "
          f"key duplicates={n_dup_key}")
    print(f"saved -> {out_path}")
    flagged = out[out["notes"].str.contains("LEAKAGE", na=False)]
    print("LEAKAGE SUSPECTS:", flagged["variable"].tolist())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="C:/Users/User Pc/Desktop/student 1/data.csv")
    ap.add_argument("--out", default="C:/Users/User Pc/Desktop/student 1/Exploratory_Data.csv")
    a = ap.parse_args()
    main(a.data, a.out)

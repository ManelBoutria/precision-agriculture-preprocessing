"""outliers_scaling.py - Student 2, Week 2.

Scikit-learn compatible building blocks for outlier handling and scaling.
Everything that LEARNS from data does it in fit() (training fold only) and
transform() only APPLIES what was learned -> no data leakage inside a Pipeline.

Main pieces
    OutlierClipper                 learns clipping bounds on the training fold
    get_scaler(kind)               "standard" | "robust" | "none"
    make_outlier_scaler_steps()    [(name, transformer), ...] to plug in a Pipeline
    ZERO_INFLATED, LEAKAGE_COLS    decisions taken from the Week 1 analysis
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler, StandardScaler

# ---- decisions from Week 1 (documented in the report) ----
# Zero-inflated: IQR is meaningless (Q1 = Q3 = 0) and non-zero values are real events.
ZERO_INFLATED = ["rainfall_mm", "solar_radiation_w_m2", "light_intensity_lux"]
# Target leakage (irrigation is On <=> Dry_Stress): never used as features.
LEAKAGE_COLS = ["irrigation_amount_l", "water_flow_rate_l_min", "irrigation_status"]


class OutlierClipper(BaseEstimator, TransformerMixin):
    """Clip (winsorize) numeric columns to bounds learned on the training data.

    Parameters
    ----------
    columns : list[str] | None
        Columns to clip. None = all columns. Other columns pass through untouched.
        (Needs a DataFrame input when columns is given.)
    method : "iqr" or "percentile"
        "iqr"        -> [Q1 - factor*IQR, Q3 + factor*IQR]
        "percentile" -> [lower_q, upper_q] quantiles
    factor : float      IQR multiplier (default 1.5)
    lower_q, upper_q : float   quantiles used when method="percentile"

    Rows are NEVER removed: values beyond the bounds are replaced by the bound.
    fit() learns the bounds (training fold only); transform() applies them unchanged.
    """

    def __init__(self, columns=None, method="iqr", factor=1.5, lower_q=0.01, upper_q=0.99):
        self.columns = columns
        self.method = method
        self.factor = factor
        self.lower_q = lower_q
        self.upper_q = upper_q

    def fit(self, X, y=None):
        X = self._as_frame(X)
        cols = list(X.columns) if self.columns is None else list(self.columns)
        missing = [c for c in cols if c not in X.columns]
        if missing:
            raise ValueError(f"Columns not found in X: {missing}")
        if self.method == "iqr":
            q1, q3 = X[cols].quantile(0.25), X[cols].quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - self.factor * iqr, q3 + self.factor * iqr
        elif self.method == "percentile":
            lower = X[cols].quantile(self.lower_q)
            upper = X[cols].quantile(self.upper_q)
        else:
            raise ValueError("method must be 'iqr' or 'percentile'")
        self.columns_ = cols
        self.lower_ = lower
        self.upper_ = upper
        self.feature_names_in_ = np.array(X.columns)
        return self

    def transform(self, X):
        was_frame = isinstance(X, pd.DataFrame)
        Xf = self._as_frame(X).copy()
        for c in self.columns_:
            Xf[c] = Xf[c].clip(self.lower_[c], self.upper_[c])
        return Xf if was_frame else Xf.to_numpy()

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_

    def _as_frame(self, X):
        if isinstance(X, pd.DataFrame):
            return X
        if self.columns is not None:
            raise ValueError("Give a DataFrame when 'columns' is specified.")
        return pd.DataFrame(X)


def get_scaler(kind="standard"):
    """Return a scaler. kind: 'standard', 'robust' or 'none' (None = no scaling)."""
    if kind == "standard":
        return StandardScaler()
    if kind == "robust":
        return RobustScaler()
    if kind in ("none", None):
        return None
    raise ValueError("kind must be 'standard', 'robust' or 'none'")


def make_outlier_scaler_steps(scaled, scaler_kind="standard", clip_columns=None,
                              method="iqr", factor=1.5):
    """Steps to put in a Pipeline BEFORE the model.

    scaled=True  -> for scale-sensitive models: SVM, Logistic Regression, KNN, MLP, PCA
    scaled=False -> for tree models: Random Forest, Extra Trees, XGBoost, LightGBM
    clip_columns -> columns to clip (None = no outlier treatment at all)

    Order: OutlierClipper -> Scaler (clipping before scaling, as bounds are in raw units).
    """
    steps = []
    if clip_columns:
        steps.append(("outlier_clipper",
                      OutlierClipper(columns=clip_columns, method=method, factor=factor)))
    if scaled:
        sc = get_scaler(scaler_kind)
        if sc is not None:
            steps.append(("scaler", sc))
    return steps

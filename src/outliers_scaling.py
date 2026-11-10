"""outliers_scaling.py - helpers used by preprocessing.py, scale_outliers.py and compare_outliers.py.

Provides:
  - ZERO_INFLATED          : columns that must NOT be clipped (IQR = 0, large values are real events)
  - OutlierClipper         : IQR-based capping, scikit-learn compatible (works inside Pipeline / ColumnTransformer)
  - get_scaler             : returns a scaler ("robust", "standard") or None
  - make_outlier_scaler_steps : list of Pipeline steps (clip + scale) used by compare_outliers.py
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Zero-inflated columns: their IQR is 0, so IQR clipping would destroy real large events.
ZERO_INFLATED = ["rainfall_mm", "solar_radiation_w_m2", "light_intensity_lux"]


class OutlierClipper(BaseEstimator, TransformerMixin):
    """IQR-based capping. fit() learns bounds, transform() clips.

    Parameters
    ----------
    columns : list of str
        Numeric columns to clip.
    method : {"iqr"}
        Only "iqr" is implemented.
    factor : float
        IQR multiplier (default 1.5 -> Tukey's fences).

    NOTE: parameters are stored AS IS in __init__ (no list(), float()...):
    scikit-learn's clone() requires it, otherwise Pipeline/ColumnTransformer raise errors.
    """

    def __init__(self, columns=None, method="iqr", factor=1.5):
        self.columns = columns
        self.method = method
        self.factor = factor

    def _as_frame(self, X):
        if isinstance(X, pd.DataFrame):
            return X
        names = getattr(self, "feature_names_in_", None)
        return pd.DataFrame(np.asarray(X), columns=names)

    def fit(self, X, y=None):
        if self.method != "iqr":
            raise ValueError(f"unknown method: {self.method!r} (only 'iqr' is supported)")
        X = self._as_frame(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        cols = list(self.columns) if self.columns is not None else list(X.columns)
        self.lower_, self.upper_ = {}, {}
        for c in cols:
            s = pd.to_numeric(X[c], errors="coerce").dropna()
            if s.empty:
                self.lower_[c], self.upper_[c] = -np.inf, np.inf
                continue
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            self.lower_[c] = q1 - self.factor * iqr
            self.upper_[c] = q3 + self.factor * iqr
        return self

    def transform(self, X):
        X = self._as_frame(X)
        out = X.copy()
        for c, lo in self.lower_.items():
            if c in out.columns:
                out[c] = out[c].clip(lower=lo, upper=self.upper_[c])
        return out

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = self.feature_names_in_
        return np.asarray(input_features, dtype=object)


def get_scaler(kind="robust"):
    """Return a scaler instance, or None for kind='none'.

    - "robust"   : median / IQR  (sklearn.preprocessing.RobustScaler)
    - "standard" : mean / std    (sklearn.preprocessing.StandardScaler)
    - "none"     : no scaling
    """
    kind = (kind or "none").lower()
    if kind == "none":
        return None
    from sklearn.preprocessing import RobustScaler, StandardScaler

    if kind == "robust":
        return RobustScaler()
    if kind == "standard":
        return StandardScaler()
    raise ValueError(f"unknown scaler: {kind!r} (use 'robust', 'standard' or 'none')")


def make_outlier_scaler_steps(do_scale, scaler_kind, clip_columns):
    """Pipeline steps [("clip", ...), ("scale", ...)] used by compare_outliers.py.

    do_scale     : bool, add a scaler step
    scaler_kind  : "standard" / "robust"
    clip_columns : list of columns to clip, or None / [] for no clipping
    """
    steps = []
    if clip_columns:
        steps.append(("clip", OutlierClipper(columns=list(clip_columns), method="iqr", factor=1.5)))
    if do_scale:
        sc = get_scaler(scaler_kind)
        if sc is not None:
            steps.append(("scale", sc))
    return steps

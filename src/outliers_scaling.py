"""outliers_scaling.py - helpers used by scale_outliers.py.

Provides:
  - ZERO_INFLATED : columns that must NOT be clipped (IQR = 0, large values are real events)
  - OutlierClipper: IQR-based capping (values outside [Q1 - f*IQR, Q3 + f*IQR] -> bound)
  - get_scaler   : returns a fitted-ready scaler ("robust", "standard" or None)
"""
import numpy as np
import pandas as pd

# Zero-inflated columns: their IQR is 0, so IQR clipping would destroy real large events.
ZERO_INFLATED = ["rainfall_mm", "solar_radiation_w_m2", "light_intensity_lux"]


class OutlierClipper:
    """IQR-based capping. fit() learns bounds, transform() clips.

    Parameters
    ----------
    columns : list of str
        Numeric columns to clip.
    method : {"iqr"}
        Only "iqr" is implemented.
    factor : float
        IQR multiplier (default 1.5 -> Tukey's fences).
    """

    def __init__(self, columns, method="iqr", factor=1.5):
        if method != "iqr":
            raise ValueError(f"unknown method: {method!r} (only 'iqr' is supported)")
        self.columns = list(columns)
        self.method = method
        self.factor = float(factor)
        self.lower_ = {}
        self.upper_ = {}

    def fit(self, X):
        """Learn per-column bounds from X (a DataFrame). Returns self."""
        for c in self.columns:
            s = pd.to_numeric(X[c], errors="coerce").dropna()
            if s.empty:
                self.lower_[c] = -np.inf
                self.upper_[c] = np.inf
                continue
            q1 = s.quantile(0.25)
            q3 = s.quantile(0.75)
            iqr = q3 - q1
            self.lower_[c] = q1 - self.factor * iqr
            self.upper_[c] = q3 + self.factor * iqr
        return self

    def transform(self, X):
        """Return a copy of X with self.columns clipped to the learned bounds."""
        out = X.copy()
        for c in self.columns:
            lo = self.lower_.get(c, -np.inf)
            hi = self.upper_.get(c, np.inf)
            out[c] = out[c].clip(lower=lo, upper=hi)
        return out

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def get_scaler(kind="robust"):
    """Return a scaler instance, or None for kind='none'.

    - "robust"   : median / IQR  (sklearn.preprocessing.RobustScaler)
    - "standard" : mean / std    (sklearn.preprocessing.StandardScaler)
    - "none"     : no scaling
    """
    kind = (kind or "none").lower()
    if kind == "none":
        return None
    try:
        from sklearn.preprocessing import RobustScaler, StandardScaler
    except ImportError as e:
        raise ImportError(
            "scikit-learn is required for scaling. Install it with: pip install scikit-learn"
        ) from e

    if kind == "robust":
        return RobustScaler()
    if kind == "standard":
        return StandardScaler()
    raise ValueError(f"unknown scaler: {kind!r} (use 'robust', 'standard' or 'none')")
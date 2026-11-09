# Precision agriculture monitoring - data preparation and preprocessing


## What the project is about

Farm sensors (IoT nodes) record air and soil measurements over time. The goal is to predict, from these measurements, the **environmental condition** of the field. This is a 5-class classification problem:

| Class | Share of rows |
| --- | --- |
| Optimal | 42.8% |
| Dry_Stress | 23.5% |
| Adverse | 22.3% |
| High_Humidity | 7.3% |
| Heat_Stress | 4.0% |

The classes are imbalanced, so we report **macro F1, balanced accuracy and per-class recall**, never plain accuracy.

**The data:** `data/precision_agriculture_dataset.csv`, 15,000 rows (one measurement of one sensor at one time) and 32 columns: the target `environmental_condition`; time and place (`timestamp`, `sensor_node_id`, `field_zone`, `latitude`, `longitude`); crop (`crop_type`, `growth_stage`); air, soil and weather measurements; three irrigation columns; and sensor health (`battery_level_pct`, `signal_strength_dbm`, `sensor_status`).

## What this repository gives you

1. **One preprocessing pipeline** (`src/preprocessing.py`) that you plug any model into.
2. **Saved splits** (`results/splits/`): a locked chronological test set and 3 x 10 validation folds, so all groups compare on the same folds.
3. **Analysis and decisions** with their results: correlations, feature selection, class rebalancing.
4. **Leakage tests** (`tests/test_leakage.py`) that check nothing from validation or test data reaches training.

## How the data is split

- The **last 20% of rows in time order is the test set** (`results/splits/test_index.csv`). Use it once, for the final evaluation. Never tune anything on it.
- The **first 80% is the development set** (12,000 rows). `results/splits/splits_validation.csv` holds 3 repetitions x 10 stratified folds on it (seed 42), 30 folds in total.
- `results/splits/splits_temporel.csv` holds time-series folds (5 folds with a gap of 48 rows) for temporal validation.
- Do not reorder the CSV: the split files use the original row numbers.

## What the pipeline does

1. Removes the three irrigation columns that reveal the target (a fixed rule, applied once). `timestamp`, `sensor_node_id` and `field_zone` are kept out of the features.
2. Inside each fold, **using the training rows only**: fills missing values (median for numbers, most frequent for categories), caps outliers (IQR rule, except zero-inflated columns such as rainfall), scales the numbers, and one-hot encodes `crop_type`, `growth_stage` and `sensor_status`. The result has 35 columns.
3. Optionally selects features and rebalances classes, still on the training rows only.
4. Trains your model.

Everything that learns from data lives inside the pipeline, so it is fitted on the training part of each fold and never sees validation rows.

## Use it

```python
import sys; sys.path.append("src")
from preprocessing import load_dev_and_folds, build_pipeline
from sklearn.ensemble import RandomForestClassifier

X, y, folds = load_dev_and_folds("data/precision_agriculture_dataset.csv")   # X: (12000, 25), 30 folds
for train, val in folds:
    pipe = build_pipeline(RandomForestClassifier(random_state=42), X, scale=False)
    pipe.fit(X.iloc[train], y.iloc[train])
    pred = pipe.predict(X.iloc[val])
```

- `scale=True` for logistic regression, SVM, KNN and MLP; `scale=False` for tree models (random forest, XGBoost, LightGBM).
- Optional `selector=` (see `feature_selection.build_selector`) and `sampler=` (SMOTE, ADASYN, RandomUnderSampler).

## Main findings

- **Leakage.** `irrigation_status`, `irrigation_amount_l` and `water_flow_rate_l_min` encode the target (irrigation is On exactly when the class is Dry_Stress), so they are removed.
- **The target is a rule.** Dry_Stress corresponds exactly to `soil_moisture_pct` <= 35, and the other classes depend on `relative_humidity_pct` and `air_temperature_c`. A decision tree of depth 6 reaches macro F1 0.995. This is why high scores are expected with tree models. `soil_moisture_pct` is kept as a feature for now; to remove it, add it to `LEAKAGE_COLS` in `src/preprocessing.py`.
- **Redundant features.** Radiation, light, evapotranspiration and temperatures are strongly correlated (up to r = 0.99). An iterative VIF flags `evapotranspiration_mm`, `solar_radiation_w_m2` and `soil_temperature_c`.
- **Few features matter.** Soil moisture, humidity and air temperature carry almost all the signal. With 5 features, a logistic regression reaches the same macro F1 (0.81) as with all 35.
- **Resampling.** For logistic regression, SMOTE and ADASYN raise balanced accuracy from 0.816 to 0.847 (better in all 30 folds), at the cost of a slightly lower macro F1. For random forest, no resampling strategy helps.
- **Classes overlap** in feature space (silhouette -0.04); the boundary is a threshold rule, so tree models fit it better than linear ones.

| Figure | Shows |
| --- | --- |
| `figures/correlation_heatmaps.png` | Pearson and Spearman correlations |
| `figures/pca.png`, `figures/tsne_umap.png` | PCA, t-SNE and UMAP |
| `figures/feature_scores.png`, `figures/class_centroid_distance.png` | class separability |
| `figures/feature_selection_k_curve.png` | macro F1 against number of features kept |
| `figures/resampling_comparison.png` | resampling strategies compared |

## Repository layout

```
data/        precision_agriculture_dataset.csv
src/         preprocessing.py          the pipeline everybody imports
             outliers_scaling.py       outlier clipping and scalers
             eda_data.py               exploration
             make_splits.py            test set and folds
             feature_analysis.py         correlations, VIF, PCA, t-SNE/UMAP, separability
             feature_selection.py      ANOVA, RFE, L1 inside the 30 folds
             resampling.py             class_weight, SMOTE, ADASYN, undersampling
tests/       test_leakage.py           10 leakage and reproducibility tests
results/     splits/ and the result tables (CSV)
figures/     plots
```


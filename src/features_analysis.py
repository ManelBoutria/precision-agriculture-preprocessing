#!/usr/bin/env python
# coding: utf-8

# # 04 - Correlations, dimensionality reduction and class separability
# **Student 4 - Week 1.** Precision agriculture monitoring, target: `environmental_condition`.
# 
# All analyses use the **official development set** from Student 3's `make_splits.py`: the first 80% of the rows in chronological order (`results/splits/dev_index.csv`). The last 20% (the locked test set, `test_index.csv`) is never used for any decision here.

# In[1]:


def display(x):
    print(x.to_string() if hasattr(x, 'to_string') else x)


import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.tree import DecisionTreeClassifier
from statsmodels.stats.outliers_influence import variance_inflation_factor
import umap

RS = 42
ROOT = Path(__file__).resolve().parents[1]   # repo root, works from any folder
FIG, RES = ROOT/"figures", ROOT/"results"
FIG.mkdir(exist_ok=True); RES.mkdir(exist_ok=True)
sns.set_theme(style="whitegrid")

df = pd.read_csv(ROOT/"data/precision_agriculture_dataset.csv")
TARGET = "environmental_condition"
dev_idx  = pd.read_csv(ROOT/"results/splits/dev_index.csv")["index"]
test_idx = pd.read_csv(ROOT/"results/splits/test_index.csv")["index"]
df_dev, df_test = df.loc[dev_idx], df.loc[test_idx]
print("full:", df.shape, "| dev:", df_dev.shape, "| locked test:", df_test.shape)
ts = pd.to_datetime(df["timestamp"])
print("dev period :", ts.loc[dev_idx].min(), "->", ts.loc[dev_idx].max())
print("test period:", ts.loc[test_idx].min(), "->", ts.loc[test_idx].max())
mix = pd.DataFrame({"dev_%": df_dev[TARGET].value_counts(normalize=True).mul(100).round(1),
                    "test_%": df_test[TARGET].value_counts(normalize=True).mul(100).round(1)})
display(mix)   # class mix should be similar: no strong drift between the two periods



# ## 1. Leakage check on the target
# The number of `irrigation_status == On` rows equals the number of `Dry_Stress` rows, so we test whether irrigation is derived from the target.

# In[2]:


ct = pd.crosstab(df_dev["irrigation_status"], df_dev[TARGET])
display(ct)
off = df_dev[df_dev.irrigation_status == "Off"][["irrigation_amount_l", "water_flow_rate_l_min"]]
on  = df_dev[df_dev.irrigation_status == "On"][["irrigation_amount_l", "water_flow_rate_l_min"]]
print("amount/flow when Off (sum):", off.abs().sum().to_dict())
print("rows On with amount == 0 or flow == 0:", int(((on == 0).any(axis=1)).sum()), "of", len(on))
LEAK = ["irrigation_status", "irrigation_amount_l", "water_flow_rate_l_min"]
print("\nConclusion: irrigation On <=> Dry_Stress, and amount/flow are > 0 only when On. All of", LEAK, "encode the target -> drop.")


# In[3]:


# After removing the leaky columns: is the target still a rule on the inputs?
from sklearn.model_selection import cross_val_score
num_cols = [c for c in df.select_dtypes("number").columns if c not in LEAK]
X, y = df_dev[num_cols], df_dev[TARGET]
print(len(num_cols), "numeric features kept for the analysis")
rows = []
for d in (2, 3, 4, 5, 6, 8):
    s = cross_val_score(DecisionTreeClassifier(max_depth=d, random_state=RS), X, y, cv=5, scoring="f1_macro").mean()
    rows.append((d, round(s, 3)))
display(pd.DataFrame(rows, columns=["tree_depth", "macro_F1_cv"]))
tree = DecisionTreeClassifier(max_depth=6, random_state=RS).fit(X, y)
imp = pd.Series(tree.feature_importances_, num_cols).sort_values(ascending=False)
display(imp.head(6).round(3).to_frame("tree_importance"))
SIGNAL = imp[imp > 0.01].index.tolist()
print("Features the rule actually uses:", SIGNAL)


# ## 2. Correlations (Pearson and Spearman)

# In[4]:


pear, spear = X.corr("pearson"), X.corr("spearman")
fig, ax = plt.subplots(1, 2, figsize=(20, 8))
for a, m, t in zip(ax, [pear, spear], ["Pearson", "Spearman"]):
    sns.heatmap(m, cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=a, cbar_kws={"shrink":.7}, xticklabels=True, yticklabels=True)
    a.set_title(t); a.tick_params(labelsize=7)
plt.tight_layout(); plt.savefig(FIG/"correlation_heatmaps.png", dpi=130); plt.close()

def pairs(m, thr):
    u = m.where(np.triu(np.ones(m.shape), 1).astype(bool)).stack().rename("r").reset_index()
    u.columns = ["feature_a","feature_b","r"]
    return u[u.r.abs() >= thr].sort_values("r", key=abs, ascending=False)
print("Pairs with |Pearson| >= 0.5:"); display(pairs(pear, .5))
print("Pairs with |Spearman| >= 0.5:"); display(pairs(spear, .5))
high = pairs(pear, .9); print("Pairs above 0.9 (redundant):", len(high))


# ## 3. VIF (multicollinearity)
# Plain VIF first, then an iterative elimination that **never removes a feature the target rule depends on** (`SIGNAL`).

# In[5]:


Xs = StandardScaler().fit_transform(X)   # standardized dev features, reused by PCA / t-SNE / UMAP / silhouette
def vif_table(cols):
    Z = StandardScaler().fit_transform(X[cols])
    return pd.Series([variance_inflation_factor(Z, i) for i in range(len(cols))], cols).sort_values(ascending=False)
vif = vif_table(num_cols).rename("VIF").reset_index().rename(columns={"index": "feature"})
display(vif.head(8))

feats, vif_drops = list(num_cols), []
while True:
    v = vif_table(feats)
    cand = v[~v.index.isin(SIGNAL)]
    if v.max() < 10 or cand.iloc[0] < 10: break
    d = cand.index[0]; vif_drops.append((d, round(cand.iloc[0], 1))); feats.remove(d)
print("Iterative VIF drops (signal features protected):", vif_drops)
print("Max VIF after:", vif_table(feats).head(4).round(2).to_dict())


# ## 4. Feature vs target: ANOVA F, mutual information, Fisher ratio

# In[6]:


F, p = f_classif(X, y)
mi = mutual_info_classif(X, y, random_state=RS)
# Fisher ratio = between-class variance / within-class variance (per feature)
mu = X.mean(); g = X.groupby(y)
between = sum(len(v)*(v.mean()-mu)**2 for _, v in g)
within  = sum(((v - v.mean())**2).sum() for _, v in g)
fisher = between / within
scores = pd.DataFrame({"feature": num_cols, "anova_F": F, "anova_p": p, "mutual_info": mi, "fisher_ratio": fisher.values}) \
           .sort_values("mutual_info", ascending=False).reset_index(drop=True)
display(scores.round(4))
fig, ax = plt.subplots(1, 2, figsize=(14, 7))
scores.sort_values("mutual_info").plot.barh(x="feature", y="mutual_info", ax=ax[0], legend=False, color="#2a7f62")
ax[0].set_title("Mutual information with target")
scores.sort_values("fisher_ratio").plot.barh(x="feature", y="fisher_ratio", ax=ax[1], legend=False, color="#e08a1e")
ax[1].set_title("Fisher ratio")
plt.tight_layout(); plt.savefig(FIG/"feature_scores.png", dpi=130); plt.close()


# ### 4b. Categorical features vs target (chi-square, Cramér's V)

# In[7]:


from scipy.stats import chi2_contingency
cat_cols = ["sensor_node_id", "field_zone", "crop_type", "growth_stage", "sensor_status"]
rows = []
for c in cat_cols:
    t = pd.crosstab(df_dev[c], df_dev[TARGET]); chi2, p, dof, _ = chi2_contingency(t)
    v = np.sqrt(chi2 / (t.values.sum() * (min(t.shape) - 1)))
    rows.append((c, df_dev[c].nunique(), round(chi2, 1), p, round(v, 4)))
cat_scores = pd.DataFrame(rows, columns=["feature", "n_categories", "chi2", "p_value", "cramers_v"]).sort_values("cramers_v", ascending=False)
display(cat_scores)
cat_scores.to_csv(RES/"student4_categorical_scores.csv", index=False)


# ## 5. PCA (on standardized data)

# In[8]:


pca = PCA(random_state=RS).fit(Xs)
cum = np.cumsum(pca.explained_variance_ratio_)
n95 = int(np.argmax(cum >= .95) + 1); n90 = int(np.argmax(cum >= .90) + 1)
print(f"components for 90% variance: {n90} | for 95%: {n95} (of {len(num_cols)})")
Z = pca.transform(Xs)
fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))
ax[0].plot(range(1, len(cum)+1), cum, marker="o"); ax[0].axhline(.95, ls="--", c="grey")
ax[0].set(xlabel="components", ylabel="cumulative explained variance", title="PCA explained variance")
sns.scatterplot(x=Z[:,0], y=Z[:,1], hue=y.values, s=8, alpha=.5, ax=ax[1]); ax[1].set(xlabel="PC1", ylabel="PC2", title="PCA, first two components")
plt.tight_layout(); plt.savefig(FIG/"pca.png", dpi=130); plt.close()
load = pd.DataFrame(pca.components_[:3].T, index=num_cols, columns=["PC1","PC2","PC3"])
display(load.reindex(load.PC1.abs().sort_values(ascending=False).index).head(8).round(3))


# ## 6. t-SNE and UMAP (visualization only, 4,000-row sample)

# In[9]:


idx = np.random.RandomState(RS).choice(len(Xs), 4000, replace=False)
Xsub, ysub = Xs[idx], y.values[idx]
tsne = TSNE(n_components=2, perplexity=30, random_state=RS, init="pca").fit_transform(Xsub)
um = umap.UMAP(n_components=2, random_state=RS).fit_transform(Xsub)
fig, ax = plt.subplots(1, 2, figsize=(15, 6))
for a, E, t in zip(ax, [tsne, um], ["t-SNE", "UMAP"]):
    sns.scatterplot(x=E[:,0], y=E[:,1], hue=ysub, s=8, alpha=.6, ax=a); a.set_title(t)
plt.tight_layout(); plt.savefig(FIG/"tsne_umap.png", dpi=130); plt.close()


# ## 7. Class separability (silhouette)

# In[10]:


sil_all = silhouette_score(Xsub, ysub)
ss = silhouette_samples(Xsub, ysub)
sil_cls = pd.Series(ss).groupby(ysub).mean().sort_values()
print("overall silhouette (true labels):", round(sil_all, 3))
display(sil_cls.rename("mean_silhouette").round(3).to_frame())
# same score using only the features the target rule depends on
sig_idx = [num_cols.index(c) for c in SIGNAL]
print("silhouette on SIGNAL features only", SIGNAL, "=", round(silhouette_score(Xsub[:, sig_idx], ysub), 3))
# class overlap: mean centroid distance between classes (standardized space)
cent = pd.DataFrame(Xsub).groupby(ysub).mean()
from scipy.spatial.distance import pdist, squareform
D = pd.DataFrame(squareform(pdist(cent)), index=cent.index, columns=cent.index)
plt.figure(figsize=(6,5)); sns.heatmap(D, annot=True, fmt=".2f", cmap="viridis"); plt.title("Distance between class centroids")
plt.tight_layout(); plt.savefig(FIG/"class_centroid_distance.png", dpi=130); plt.close()


# ## 8. Imbalance (for the rebalancing decision)

# In[11]:


cnt = y.value_counts(); print(cnt); print("majority : minority =", round(cnt.max()/cnt.min(), 1), ": 1  (project limit 3 : 1)")


# ## 9. Drop recommendations and export

# In[12]:


rows = [(f, "leakage", "encodes the target (irrigation On <=> Dry_Stress)") for f in LEAK]
for f, v in vif_drops:
    rows.append((f, "redundant", f"VIF {v} (collinear with kept features; signal features protected)"))
weak = scores[(scores.anova_p > .05) & (scores.mutual_info < .005) & ~scores.feature.isin(SIGNAL)].feature.tolist()
for f in weak:
    rows.append((f, "no_signal", "ANOVA p > 0.05 and MI < 0.005; target rule ignores it (test in week 2 with feature selection)"))
for _, r in cat_scores[cat_scores.cramers_v < .05].iterrows():
    rows.append((r.feature, "no_signal", f"categorical, Cramer's V = {r.cramers_v} (< 0.05, negligible association); sensor_node_id also needs the group's decision on unseen-sensor generalization"))
drop_df = pd.DataFrame(rows, columns=["feature", "category", "reason"])
display(drop_df)
kept = [c for c in df.columns if c not in set(drop_df.feature) | {TARGET, "timestamp"}]
print("\nKept (excluding timestamp, handled by Student 3):", kept)

out = scores.merge(vif, on="feature").merge(drop_df[["feature","category","reason"]], on="feature", how="left")
out.to_csv(RES/"student4_feature_analysis.csv", index=False)
drop_df.to_csv(RES/"student4_drop_list.csv", index=False)
pairs(pear, .5).to_csv(RES/"student4_correlated_pairs.csv", index=False)
print("saved results in", RES)


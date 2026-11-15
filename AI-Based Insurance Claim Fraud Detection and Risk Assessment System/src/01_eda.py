"""Sub-module 1.1 + 1.2: load dataset, EDA (imbalance ratio, missingness, distributions)."""
import json, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from config import *

sns.set_theme(style="whitegrid")
FIGS.mkdir(parents=True, exist_ok=True); REPORTS.mkdir(parents=True, exist_ok=True)

if not RAW_CSV.exists():
    sys.exit(f"Dataset not found: {RAW_CSV}\nDownload fraud_oracle.csv from Kaggle (shivamb/vehicle-claim-fraud-detection) into data/raw/")

df = pd.read_csv(RAW_CSV)
print("Shape:", df.shape)
y = df[TARGET]

# ---------- 1.2a Imbalance ratio ----------
n_fraud, n_legit = int((y == 1).sum()), int((y == 0).sum())
imb = {"n_rows": len(df), "n_fraud": n_fraud, "n_legit": n_legit,
       "fraud_rate_pct": round(100 * n_fraud / len(df), 3),
       "imbalance_ratio_IR": round(n_legit / n_fraud, 2)}   # IR = N_majority / N_minority
print("Imbalance:", imb)

fig, ax = plt.subplots(figsize=(5, 4))
sns.countplot(x=TARGET, data=df, ax=ax, palette=["#4c72b0", "#c44e52"], hue=TARGET, legend=False)
for p in ax.patches: ax.annotate(int(p.get_height()), (p.get_x()+p.get_width()/2, p.get_height()), ha="center", va="bottom")
ax.set_title(f"Class distribution (IR = {imb['imbalance_ratio_IR']}:1)"); fig.tight_layout()
fig.savefig(FIGS / "01_class_balance.png", dpi=150); plt.close(fig)

# ---------- 1.2b Missingness (explicit NaN + disguised placeholders) ----------
miss = pd.DataFrame({"explicit_NaN": df.isna().sum()})
dis = {c: int(df[c].isin(v).sum()) for c, v in DISGUISED_MISSING.items() if c in df}
miss["disguised_placeholder"] = pd.Series(dis).reindex(miss.index).fillna(0).astype(int)
miss["total_missing"] = miss.sum(axis=1)
miss["missing_pct"] = (100 * miss.total_missing / len(df)).round(3)
miss.sort_values("total_missing", ascending=False).to_csv(REPORTS / "missingness_table.csv")
print("\nColumns with any missingness:\n", miss[miss.total_missing > 0])

fig, ax = plt.subplots(figsize=(8, 5))
m = miss[miss.total_missing > 0].sort_values("missing_pct")
if len(m): m.missing_pct.plot.barh(ax=ax, color="#dd8452")
ax.set_xlabel("% missing (NaN + placeholders)"); ax.set_title("Missingness per column"); fig.tight_layout()
fig.savefig(FIGS / "02_missingness.png", dpi=150); plt.close(fig)

# ---------- 1.2c Distributions ----------
num_cols = df.select_dtypes("number").columns.drop([TARGET, "PolicyNumber"], errors="ignore")
cat_cols = df.select_dtypes(exclude="number").columns
df[num_cols].hist(figsize=(12, 7), bins=25, color="#55a868"); plt.suptitle("Numeric features"); plt.tight_layout()
plt.savefig(FIGS / "03_numeric_hist.png", dpi=150); plt.close()

rows = []
for c in cat_cols:
    g = df.groupby(c)[TARGET].agg(["mean", "size"]); g["feature"] = c
    rows.append(g.reset_index().rename(columns={c: "value"}))
pd.concat(rows).to_csv(REPORTS / "fraud_rate_by_category.csv", index=False)

n = len(cat_cols); ncol = 4; nrow = int(np.ceil(n / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(20, 3.6 * nrow))
for ax, c in zip(axes.ravel(), cat_cols):
    g = df.groupby(c)[TARGET].mean().sort_values(ascending=False).head(8) * 100
    g.plot.bar(ax=ax, color="#c44e52"); ax.set_title(c); ax.set_ylabel("fraud %"); ax.set_xlabel("")
    ax.axhline(100 * y.mean(), ls="--", c="k", lw=.8); ax.tick_params(axis="x", rotation=45, labelsize=7)
for ax in axes.ravel()[n:]: ax.axis("off")
fig.suptitle("Fraud rate by category (dashed = overall rate)"); fig.tight_layout()
fig.savefig(FIGS / "04_fraud_rate_by_category.png", dpi=120); plt.close(fig)

corr = df[list(num_cols) + [TARGET]].corr()
fig, ax = plt.subplots(figsize=(6, 5)); sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
ax.set_title("Numeric correlation"); fig.tight_layout(); fig.savefig(FIGS / "05_correlation.png", dpi=150); plt.close(fig)

# ---------- summary + optional ydata-profiling ----------
df.describe(include="all").T.to_csv(REPORTS / "summary_statistics.csv")
json.dump({"imbalance": imb, "n_columns": df.shape[1],
           "duplicate_rows": int(df.duplicated().sum()),
           "constant_columns": [c for c in df if df[c].nunique() == 1]},
          open(REPORTS / "eda_summary.json", "w"), indent=2)
try:
    from ydata_profiling import ProfileReport
    ProfileReport(df, minimal=True, title="Vehicle Insurance Fraud EDA").to_file(REPORTS / "profiling_report.html")
    print("ydata-profiling report written.")
except ImportError:
    print("(optional) pip install ydata-profiling for an auto EDA report")
print("EDA done -> reports/")

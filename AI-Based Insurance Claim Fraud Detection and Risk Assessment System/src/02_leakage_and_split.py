"""Sub-modules 1.3 + 1.4: leakage screening, clean dataset, leakage-safe stratified train/val/test split,
and automatic data card."""
import json, sys, datetime
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from config import *

if not RAW_CSV.exists(): sys.exit("Run after placing the dataset in data/raw/")
PROCESSED.mkdir(parents=True, exist_ok=True); DOCS.mkdir(parents=True, exist_ok=True); REPORTS.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(RAW_CSV)

# ---------- 1.3 leakage screening ----------
id_like = [c for c in df if c != TARGET and df[c].nunique() > 0.5 * len(df)]
y = df[TARGET]; auc_rows = []
for c in df.columns.drop(TARGET):
    x = df[c]
    enc = x.map(df.groupby(c)[TARGET].mean()) if (x.dtype == object or x.nunique() < 30) else x
    try: auc = roc_auc_score(y, enc.fillna(enc.median()))
    except Exception: auc = np.nan
    auc_rows.append((c, max(auc, 1 - auc)))
auc_df = pd.DataFrame(auc_rows, columns=["feature", "single_feature_AUC"]).sort_values("single_feature_AUC", ascending=False)
auc_df["flag"] = np.where(auc_df.single_feature_AUC > LEAK_AUC_THRESHOLD, "SUSPICIOUS", "")
auc_df.to_csv(REPORTS / "leakage_screen.csv", index=False)
print("Identifier-like columns:", id_like)
print(auc_df.head(8).to_string(index=False))

# ---------- clean dataset ----------
dups = int(df.duplicated().sum())
clean = df.drop(columns=[c for c in DROP_COLUMNS if c in df]).drop_duplicates().reset_index(drop=True)
for c, vals in DISGUISED_MISSING.items():
    if c in clean: clean[c] = clean[c].replace(vals, np.nan)
clean.to_csv(PROCESSED / "clean_full.csv", index=False)

# ---------- 1.4 leakage-safe split ----------
# Split FIRST (stratified, fixed seed). Imputation / scaling / encoding / SMOTE must be fitted on TRAIN only.
X, yy = clean.drop(columns=TARGET), clean[TARGET]
X_tmp, X_test, y_tmp, y_test = train_test_split(X, yy, test_size=TEST_SIZE, stratify=yy, random_state=RANDOM_STATE)
X_tr, X_val, y_tr, y_val = train_test_split(X_tmp, y_tmp, test_size=VAL_SIZE / (1 - TEST_SIZE), stratify=y_tmp, random_state=RANDOM_STATE)
for name, Xs, ys in [("train", X_tr, y_tr), ("val", X_val, y_val), ("test", X_test, y_test)]:
    pd.concat([Xs, ys], axis=1).to_csv(PROCESSED / f"{name}.csv", index=False)
split_info = {n: {"rows": len(a), "fraud": int(b.sum()), "fraud_pct": round(float(100 * b.mean()), 3)}
              for n, a, b in [("train", X_tr, y_tr), ("val", X_val, y_val), ("test", X_test, y_test)]}
print("Split:", split_info)

# ---------- data card ----------
imb = json.load(open(REPORTS / "eda_summary.json"))["imbalance"] if (REPORTS / "eda_summary.json").exists() else {}
miss = pd.read_csv(REPORTS / "missingness_table.csv", index_col=0) if (REPORTS / "missingness_table.csv").exists() else None
miss_txt = "\n".join(f"- `{c}`: {int(r.total_missing)} rows ({r.missing_pct}%)" for c, r in miss[miss.total_missing > 0].iterrows()) if miss is not None else "(run 01_eda.py first)"
drop_txt = "\n".join(f"- `{c}`: {r}" for c, r in DROP_COLUMNS.items())
watch_txt = "\n".join(f"- `{c}`: {r}" for c, r in WATCH_COLUMNS.items())
sus = ", ".join(auc_df[auc_df.flag != ""].feature) or "none above threshold"
card = f"""# Data Card: Vehicle Insurance Claim Fraud (Team CSM-A10)
_Generated {datetime.date.today()} by `src/02_leakage_and_split.py`. Format follows Datasheets for Datasets (Gebru et al., 2018)._

## 1. Motivation
Binary classification of insurance claims as fraudulent (`{TARGET}=1`) or genuine, for the project
*AI-Based Insurance Claim Fraud Detection and Risk Assessment System*.

## 2. Source
Kaggle, "Vehicle Insurance Fraud Detection" (shivamb), file `fraud_oracle.csv`
https://www.kaggle.com/datasets/shivamb/vehicle-claim-fraud-detection  (check the licence on the Kaggle page before redistributing).

## 3. Composition
- Rows: {len(df)} raw, {len(clean)} after removing {dups} exact duplicate rows
- Columns: {df.shape[1]} raw ({df.shape[1]-1} features + target)
- Fraud: {imb.get('n_fraud','?')} ({imb.get('fraud_rate_pct','?')}%), Genuine: {imb.get('n_legit','?')}
- **Imbalance ratio IR = N_majority / N_minority = {imb.get('imbalance_ratio_IR','?')} : 1**
- Mostly categorical features (policy, vehicle, claim timing) plus a few numeric ones (Age, Deductible, DriverRating, ...).

## 4. Missingness
Real NaN plus "disguised" placeholders (Age=0, '0' month/day), converted to NaN in `clean_full.csv`:
{miss_txt}

## 5. Leakage risks and handling
Dropped:
{drop_txt}

Flagged for discussion with the guide (kept for now):
{watch_txt}

Single-feature AUC screen (threshold {LEAK_AUC_THRESHOLD}): {sus}. Full table: `reports/leakage_screen.csv`.
The screen uses in-sample target encoding, so it is only a rough red-flag detector.

## 6. Split protocol (leakage-safe)
Stratified train/val/test = {1-TEST_SIZE-VAL_SIZE:.0%}/{VAL_SIZE:.0%}/{TEST_SIZE:.0%}, `random_state={RANDOM_STATE}`. The split is done BEFORE any
imputation, scaling, encoding or resampling; those steps are fitted on train only. SMOTE only inside CV folds on train
(imbalanced-learn "common pitfalls").
```
{json.dumps(split_info, indent=2)}
```

## 7. Known limitations
- Public dataset covering 1994-1996; fraud patterns may not match current claims.
- Labels may be noisy; only investigated claims carry reliable fraud labels.
- Very rare fraud class: accuracy is misleading, use PR-AUC, recall, F1.
- Sensitive attributes (Sex, MaritalStatus, Age) raise fairness concerns; revisit in the XAI/fairness module.
"""
(DOCS / "data_card.md").write_text(card)
json.dump(split_info, open(REPORTS / "split_summary.json", "w"), indent=2)
print("Wrote docs/data_card.md and data/processed/{train,val,test}.csv")

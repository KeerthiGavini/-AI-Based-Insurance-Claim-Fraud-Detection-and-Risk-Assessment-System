# Data Card: Vehicle Insurance Claim Fraud (Team CSM-A10)
_Generated 2026-11-09 by `src/02_leakage_and_split.py`. Format follows Datasheets for Datasets (Gebru et al., 2018)._

## 1. Motivation
Binary classification of insurance claims as fraudulent (`FraudFound_P=1`) or genuine, for the project
*AI-Based Insurance Claim Fraud Detection and Risk Assessment System*.

## 2. Source
Kaggle, "Vehicle Insurance Fraud Detection" (shivamb), file `fraud_oracle.csv`
https://www.kaggle.com/datasets/shivamb/vehicle-claim-fraud-detection  (check the licence on the Kaggle page before redistributing).

## 3. Composition
- Rows: 15420 raw, 15420 after removing 0 exact duplicate rows
- Columns: 33 raw (32 features + target)
- Fraud: 923 (5.986%), Genuine: 14497
- **Imbalance ratio IR = N_majority / N_minority = 15.71 : 1**
- Mostly categorical features (policy, vehicle, claim timing) plus a few numeric ones (Age, Deductible, DriverRating, ...).

## 4. Missingness
Real NaN plus "disguised" placeholders (Age=0, '0' month/day), converted to NaN in `clean_full.csv`:
- `Age`: 320 rows (2.075%)
- `DayOfWeekClaimed`: 1 rows (0.006%)
- `MonthClaimed`: 1 rows (0.006%)

## 5. Leakage risks and handling
Dropped:
- `PolicyNumber`: Row identifier (unique per claim). No predictive meaning; lets a model memorise rows.
- `RepNumber`: Claims-representative ID (1-16). Reflects who handled the claim, i.e. possibly the investigation workflow, not the claimant. Leakage / bias risk, drop by default.

Flagged for discussion with the guide (kept for now):
- `Year`: Only 1994-1996. A model may learn year-specific fraud rate; not generalisable. Consider time-based split.
- `Age`: Contains Age = 0 placeholders (missing in disguise); duplicates information in AgeOfPolicyHolder.
- `AgeOfPolicyHolder`: Binned version of Age; redundant with Age.
- `MonthClaimed`: Contains a '0' placeholder (missing in disguise).
- `DayOfWeekClaimed`: Contains a '0' placeholder (missing in disguise).
- `PoliceReportFiled`: Filed after the incident; confirm with guide it is known at claim-decision time.
- `WitnessPresent`: Recorded during claim handling; confirm it is available at decision time.

Single-feature AUC screen (threshold 0.85): none above threshold. Full table: `reports/leakage_screen.csv`.
The screen uses in-sample target encoding, so it is only a rough red-flag detector.

## 6. Split protocol (leakage-safe)
Stratified train/val/test = 70%/15%/15%, `random_state=42`. The split is done BEFORE any
imputation, scaling, encoding or resampling; those steps are fitted on train only. SMOTE only inside CV folds on train
(imbalanced-learn "common pitfalls").
```
{
  "train": {
    "rows": 10794,
    "fraud": 646,
    "fraud_pct": 5.985
  },
  "val": {
    "rows": 2313,
    "fraud": 139,
    "fraud_pct": 6.01
  },
  "test": {
    "rows": 2313,
    "fraud": 138,
    "fraud_pct": 5.966
  }
}
```

## 7. Known limitations
- Public dataset covering 1994-1996; fraud patterns may not match current claims.
- Labels may be noisy; only investigated claims carry reliable fraud labels.
- Very rare fraud class: accuracy is misleading, use PR-AUC, recall, F1.
- Sensitive attributes (Sex, MaritalStatus, Age) raise fairness concerns; revisit in the XAI/fairness module.

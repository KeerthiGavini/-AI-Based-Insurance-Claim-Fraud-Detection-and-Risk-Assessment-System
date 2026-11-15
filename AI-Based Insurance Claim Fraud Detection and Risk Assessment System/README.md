# CSM-A10 | Week 1: M1 Data & EDA
Project: AI-Based Insurance Claim Fraud Detection and Risk Assessment System
Dataset chosen: **Kaggle Vehicle Insurance Fraud Detection** (`fraud_oracle.csv`, shivamb)

## Run (5 minutes)
```bash
pip install -r requirements.txt
# 1. download fraud_oracle.csv into data/raw/
cd src
python 01_eda.py                # sub-modules 1.1, 1.2: EDA, imbalance ratio, missingness, figures
python 02_leakage_and_split.py  # sub-modules 1.3, 1.4: leakage screen, clean data, split, data card
```

## What you get
| Sub-module | Output |
|---|---|
| 1.1 dataset | `data/raw/fraud_oracle.csv` (you download) |
| 1.2 EDA | `reports/figures/*.png`, `reports/missingness_table.csv`, `reports/eda_summary.json` |
| 1.3 leakage | `reports/leakage_screen.csv`, drop/watch lists in `src/config.py` |
| 1.4 split + card | `data/processed/{train,val,test}.csv`, `docs/data_card.md` |
| Review II math | `docs/review2_math.md` |

## Your to-do (things the code cannot do for you)
1. Download the dataset and run both scripts.
2. Open the figures and **write your own observations** (what stands out, what is odd).
3. Read the base paper (Khalil et al., IEEE Access 2024) first.
4. Discuss `DROP_COLUMNS` / `WATCH_COLUMNS` in `src/config.py` with your guide and adjust.
5. Read the data card, correct anything that is wrong, and add your reasoning.
6. Push to GitHub by pull request, hold the guide meeting, submit the report by 07 Nov 2026.

Note: the leakage lists reflect our reading of the column names and are not verified against the real file.
Confirm them against your EDA output. `make_synthetic_demo.py` creates FAKE data for testing only; never report numbers from it.

## One command
```bash
python src/run_all.py
```

## Download option (Kaggle CLI, needs your kaggle.json API token)
```bash
pip install kaggle
kaggle datasets download -d shivamb/vehicle-claim-fraud-detection -p data/raw --unzip
```

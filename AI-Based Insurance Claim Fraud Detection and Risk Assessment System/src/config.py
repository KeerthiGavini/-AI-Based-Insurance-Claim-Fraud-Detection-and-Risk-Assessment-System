"""Central settings for Week 1 (M1: Data & EDA)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "raw" / "fraud_oracle.csv"   # Kaggle: shivamb/vehicle-claim-fraud-detection
PROCESSED = ROOT / "data" / "processed"
FIGS = ROOT / "reports" / "figures"
REPORTS = ROOT / "reports"
DOCS = ROOT / "docs"

TARGET = "FraudFound_P"
RANDOM_STATE = 42

# --- M1.3 Leakage / non-informative fields -------------------------------
# Review these with your guide. Reasons are written into the data card.
DROP_COLUMNS = {
    "PolicyNumber": "Row identifier (unique per claim). No predictive meaning; lets a model memorise rows.",
    "RepNumber": "Claims-representative ID (1-16). Reflects who handled the claim, i.e. possibly the "
                 "investigation workflow, not the claimant. Leakage / bias risk, drop by default.",
}
# Fields to DISCUSS with guide (kept by default, but flagged):
WATCH_COLUMNS = {
    "Year": "Only 1994-1996. A model may learn year-specific fraud rate; not generalisable. Consider time-based split.",
    "Age": "Contains Age = 0 placeholders (missing in disguise); duplicates information in AgeOfPolicyHolder.",
    "AgeOfPolicyHolder": "Binned version of Age; redundant with Age.",
    "MonthClaimed": "Contains a '0' placeholder (missing in disguise).",
    "DayOfWeekClaimed": "Contains a '0' placeholder (missing in disguise).",
    "PoliceReportFiled": "Filed after the incident; confirm with guide it is known at claim-decision time.",
    "WitnessPresent": "Recorded during claim handling; confirm it is available at decision time.",
}
# Values that actually mean "missing" in this dataset
DISGUISED_MISSING = {"Age": [0], "MonthClaimed": ["0"], "DayOfWeekClaimed": ["0"]}

# Single-feature AUC above this is flagged as suspicious leakage
LEAK_AUC_THRESHOLD = 0.85
TEST_SIZE = 0.15
VAL_SIZE = 0.15

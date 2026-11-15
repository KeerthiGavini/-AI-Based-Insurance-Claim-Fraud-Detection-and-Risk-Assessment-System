"""Runs the whole Week 1 pipeline: EDA -> leakage screen -> split -> data card."""
import runpy, sys
from pathlib import Path
here = Path(__file__).parent
sys.path.insert(0, str(here))
for script in ["01_eda.py", "02_leakage_and_split.py"]:
    print(f"\n===== {script} =====")
    runpy.run_path(str(here / script), run_name="__main__")
print("\nAll done. Check reports/, data/processed/ and docs/data_card.md")

#!/usr/bin/env python3
"""Ruleaza analiza inginereasca suplimentara: comparare algoritmi + performanta."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
DB = BASE / "baza_date_licenta.db"
OUTPUTS_AI = BASE / "outputs_ai"
OUTPUTS_DESC = BASE / "outputs_descriptive_analysis"
OUTPUTS_ENG = BASE / "outputs_engineering"


def run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, cwd=BASE, check=True)


def main() -> None:
    if not DB.exists():
        raise FileNotFoundError("Nu gasesc baza_date_licenta.db. Ruleaza mai intai build_features_v4_final_audit.py.")
    OUTPUTS_ENG.mkdir(exist_ok=True)
    run([sys.executable, "compare_clustering_methods.py", "--db", str(DB), "--output-dir", str(OUTPUTS_ENG)])
    run([
        sys.executable,
        "measure_engineering_performance.py",
        "--db", str(DB),
        "--outputs-ai", str(OUTPUTS_AI),
        "--outputs-desc", str(OUTPUTS_DESC),
        "--output-dir", str(OUTPUTS_ENG),
    ])
    print("\nAnaliza inginereasca a fost finalizata. Output: outputs_engineering")


if __name__ == "__main__":
    main()

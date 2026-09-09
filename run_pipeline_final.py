#!/usr/bin/env python3
"""Ruleaza pipeline-ul complet local."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
DB = BASE / "baza_date_licenta.db"


def run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, cwd=BASE, check=True)


def main() -> None:
    run([sys.executable, "build_features_v4_final_audit.py"])
    run([sys.executable, "train_ai_final.py", "--db", str(DB), "--force-k", "3", "--output-dir", "outputs_ai"])
    run([sys.executable, "generate_descriptive_analysis.py", "--db", str(DB), "--output-dir", "outputs_descriptive_analysis"])
    run([sys.executable, "run_engineering_analysis.py"])
    run([sys.executable, "run_validation_final.py"])
    print("\nPipeline complet finalizat si validat.")


if __name__ == "__main__":
    main()

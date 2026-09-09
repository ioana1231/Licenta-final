#!/usr/bin/env python3
"""Ruleaza testele unitare si validarea output-urilor finale."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DB = ROOT / "baza_date_licenta.db"
OUTPUTS_AI = ROOT / "outputs_ai"
OUTPUTS_DESC = ROOT / "outputs_descriptive_analysis"
OUTPUTS_ENG = ROOT / "outputs_engineering"
REPORT = ROOT / "reports" / "validation_report.md"


def run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    run([sys.executable, "-m", "pytest", "tests/test_pipeline_unit.py", "-q"])
    run([
        sys.executable,
        "validate_pipeline_outputs.py",
        "--db", str(DB),
        "--outputs-ai", str(OUTPUTS_AI),
        "--outputs-desc", str(OUTPUTS_DESC),
        "--outputs-eng", str(OUTPUTS_ENG),
        "--report", str(REPORT),
    ])
    print(f"\nRaport final: {REPORT}")


if __name__ == "__main__":
    main()

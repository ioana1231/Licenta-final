#!/usr/bin/env python3
"""Ruleaza cap-coada pipeline-ul final, validarea si porneste dashboard-ul local.

Ordine:
1. build_features_v4_final_audit.py -> genereaza baza SQLite si feature-urile
2. train_ai_final.py -> ruleaza KMeans final cu k=3
3. generate_descriptive_analysis.py -> statistici descriptive + figuri
4. run_engineering_analysis.py -> comparare algoritmi + performanta
5. run_validation_final.py -> teste unitare + validare output-uri
6. streamlit run dashboard_app_engineering.py -> porneste dashboard-ul local

Observatie: ultimul pas ramane activ pana cand opresti terminalul cu Ctrl+C.
"""
from __future__ import annotations

import argparse
import socket
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
DB = BASE / "baza_date_licenta.db"


def find_available_port(preferred_port: int = 8501, port_range: int = 20) -> int:
    """Return the first free localhost port near the requested one."""
    for offset in range(port_range + 1):
        port = preferred_port + offset
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"Nu am gasit un port liber in intervalul {preferred_port}-{preferred_port + port_range}.")


def resolve_dashboard_script() -> str:
    candidates = [
        "dashboard_app_engineering.py",
        "dashboard_app_recording.py",
        "dashboard_app_polished.py",
        "dashboard_app.py",
    ]
    for candidate in candidates:
        if (BASE / candidate).exists():
            return candidate
    raise FileNotFoundError("Nu exista niciun dashboard valid in folderul proiectului.")


def run(cmd: list[str], *, blocking: bool = True) -> None:
    print("\n$ " + " ".join(cmd))
    if blocking:
        subprocess.run(cmd, cwd=BASE, check=True)
    else:
        subprocess.Popen(cmd, cwd=BASE)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ruleaza pipeline-ul final si porneste dashboard-ul.")
    parser.add_argument("--skip-validation", action="store_true", help="Nu ruleaza testele si validarea output-urilor.")
    parser.add_argument("--skip-engineering", action="store_true", help="Nu ruleaza analiza inginereasca suplimentara.")
    parser.add_argument("--no-dashboard", action="store_true", help="Ruleaza doar pipeline-ul, fara sa porneasca dashboard-ul.")
    parser.add_argument("--port", type=int, default=8501, help="Portul Streamlit pentru dashboard.")
    args = parser.parse_args()

    run([sys.executable, "build_features_v4_final_audit.py"])
    run([sys.executable, "train_ai_final.py", "--db", str(DB), "--force-k", "3", "--output-dir", "outputs_ai"])
    run([sys.executable, "generate_descriptive_analysis.py", "--db", str(DB), "--output-dir", "outputs_descriptive_analysis"])

    if not args.skip_engineering:
        run([sys.executable, "run_engineering_analysis.py"])

    if not args.skip_validation:
        run([sys.executable, "run_validation_final.py"])

    if args.no_dashboard:
        print("\nPipeline finalizat. Dashboard-ul nu a fost pornit (--no-dashboard).")
        return

    dashboard_script = resolve_dashboard_script()
    dashboard_port = find_available_port(args.port)

    print("\nPipeline finalizat. Pornesc dashboard-ul local.")
    print(f"Deschide in browser: http://127.0.0.1:{dashboard_port}")
    print("Daca rulezi din WSL/VS Code, deschide URL-ul in browser-ul Windows sau foloseste port-forwarding din VS Code.")
    print("Opresti dashboard-ul cu Ctrl+C in terminal.\n")
    run([
        sys.executable, "-m", "streamlit", "run", dashboard_script,
        "--server.address", "127.0.0.1",
        "--server.port", str(dashboard_port),
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false",
    ])


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
run.py — single entry point for all MemoViT benchmarks.

Dispatches to the per-benchmark pipeline under src/. Each pipeline keeps its own
configuration block (dataset paths, output dirs) at the top of its file; set
those before running. Extra arguments after the benchmark name are forwarded to
the underlying script.

    python run.py drone          # Drone-Anomaly        (Tables 1-2, MemoViT)
    python run.py uitadrone      # UIT-ADrone           (Tables 1-2, MemoViT)
    python run.py baselines --method patchcore   # aerial baselines (Tables 1-2)
    python run.py mvtec          # MVTec-AD / VisA      (Tables 3-4, 7-8)
    python run.py agri           # Agriculture-Vision   (Table 5)
    python run.py ablation       # ablations            (Table 6)
"""

import argparse
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"

# Benchmark name -> pipeline script. `drone` and `uitadrone` share the aerial
# pipeline (the dataset is chosen in that script's configuration block).
PIPELINES = {
    "drone":     "aerial_memovit.py",
    "uitadrone": "aerial_memovit.py",
    "baselines": "baselines.py",
    "mvtec":     "mvtec_visa_memovit.py",
    "visa":      "mvtec_visa_memovit.py",
    "agri":      "agriculture_memovit.py",
    "ablation":  "ablation.py",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a MemoViT benchmark pipeline.")
    parser.add_argument("benchmark", choices=sorted(PIPELINES), help="which pipeline to run")
    parser.add_argument("args", nargs=argparse.REMAINDER, help="arguments forwarded to the script")
    ns = parser.parse_args()

    script = SRC / PIPELINES[ns.benchmark]
    if not script.exists():
        sys.exit(f"pipeline script not found: {script}")

    cmd = [sys.executable, str(script), *ns.args]
    print(f"[run] {ns.benchmark} -> {' '.join(cmd)}")
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())

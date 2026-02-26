from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_phase6_pipeline_smoke_generates_artifacts(tmp_path: Path) -> None:
    output_root = tmp_path / "phase6_runs"
    output_root.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "scripts/run_phase6.py",
        "--seed",
        "7",
        "--output-dir",
        str(output_root),
        "--num-sequences",
        "32",
        "--seq-len",
        "8",
        "--phase1-steps",
        "10",
        "--phase2-steps",
        "6",
        "--phase3-steps",
        "6",
        "--phase4-steps",
        "6",
        "--phase5-steps",
        "6",
        "--sigma-stage-steps",
        "4",
        "--phase6-batch-size",
        "8",
        "--ei-audit-interval",
        "2",
        "--ei-audit-samples",
        "32",
        "--ei-interventions-per-dim",
        "4",
        "--tmaze-eval-seeds",
        "7",
        "--tmaze-eval-episodes",
        "2",
    ]
    subprocess.run(cmd, check=True, cwd="/Users/alankarchmer/agi")

    run_dirs = sorted(p for p in output_root.iterdir() if p.is_dir())
    assert run_dirs, "expected a timestamped phase6 run directory"
    run_dir = run_dirs[-1]

    summary_path = run_dir / "summary.json"
    stage_metrics_path = run_dir / "stage_metrics.csv"
    gates_path = run_dir / "gates.json"

    assert summary_path.exists()
    assert stage_metrics_path.exists()
    assert gates_path.exists()

    summary = json.loads(summary_path.read_text())
    assert "phase6" in summary
    assert "artifacts" in summary

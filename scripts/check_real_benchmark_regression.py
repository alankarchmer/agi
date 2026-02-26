from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.regression import RegressionTolerances, compare_real_benchmark_against_baseline


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description="Check real-data benchmark regression versus a locked baseline JSON.")
    parser.add_argument("--candidate", type=str, required=True)
    parser.add_argument("--baseline", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--no-require-pass-all", action="store_true")

    parser.add_argument("--tol-nll-ci95-high", type=float, default=0.0)
    parser.add_argument("--tol-kl-ci95-low", type=float, default=0.0)
    parser.add_argument("--tol-kl-ci95-high", type=float, default=0.0)
    parser.add_argument("--tol-temporal-drift-ci95-high", type=float, default=0.0)
    parser.add_argument("--tol-ood-gap-ci95-low", type=float, default=0.0)
    parser.add_argument("--tol-gate-pass-rate-mean", type=float, default=0.0)
    args = parser.parse_args()

    candidate_path = Path(args.candidate)
    baseline_path = Path(args.baseline)
    candidate = _load_json(candidate_path)
    baseline = _load_json(baseline_path)

    tolerances = RegressionTolerances(
        nll_ci95_high=args.tol_nll_ci95_high,
        kl_ci95_low=args.tol_kl_ci95_low,
        kl_ci95_high=args.tol_kl_ci95_high,
        temporal_drift_ci95_high=args.tol_temporal_drift_ci95_high,
        ood_gap_ci95_low=args.tol_ood_gap_ci95_low,
        gate_pass_rate_mean=args.tol_gate_pass_rate_mean,
    )
    overall, checks, details = compare_real_benchmark_against_baseline(
        candidate=candidate,
        baseline=baseline,
        tolerances=tolerances,
        require_pass_all=(not args.no_require_pass_all),
    )

    print("[Real Benchmark Regression Check]")
    print(f"  candidate: {candidate_path}")
    print(f"  baseline: {baseline_path}")
    print("  checks:")
    for name, passed in checks.items():
        print(f"    {name}: {'PASS' if passed else 'FAIL'}")
    print(f"\nOverall: {'PASS' if overall else 'FAIL'}")

    report = {
        "overall_pass": overall,
        "checks": checks,
        "details": details,
        "candidate": str(candidate_path),
        "baseline": str(baseline_path),
    }
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2))
        print(f"Report JSON: {out}")

    if not overall:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

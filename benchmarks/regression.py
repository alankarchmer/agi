from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class RegressionTolerances:
    nll_ci95_high: float = 0.0
    kl_ci95_low: float = 0.0
    kl_ci95_high: float = 0.0
    temporal_drift_ci95_high: float = 0.0
    ood_gap_ci95_low: float = 0.0
    gate_pass_rate_mean: float = 0.0


def _metric(summary: dict[str, Any], name: str, key: str) -> float:
    try:
        return float(summary["metrics"][name][key])
    except KeyError as exc:
        raise KeyError(f"Missing metric field metrics.{name}.{key}") from exc


def compare_real_benchmark_against_baseline(
    candidate: dict[str, Any],
    baseline: dict[str, Any],
    tolerances: RegressionTolerances,
    require_pass_all: bool = True,
) -> tuple[bool, dict[str, bool], dict[str, float | bool]]:
    checks: dict[str, bool] = {}
    details: dict[str, float | bool] = {}

    cand_nll_hi = _metric(candidate, "nll_mean", "ci95_high")
    base_nll_hi = _metric(baseline, "nll_mean", "ci95_high")
    checks["nll_ci95_high_nonreg"] = cand_nll_hi <= (base_nll_hi + tolerances.nll_ci95_high)
    details["candidate_nll_ci95_high"] = cand_nll_hi
    details["baseline_nll_ci95_high"] = base_nll_hi

    cand_kl_lo = _metric(candidate, "kl_mean", "ci95_low")
    base_kl_lo = _metric(baseline, "kl_mean", "ci95_low")
    checks["kl_ci95_low_nonreg"] = cand_kl_lo >= (base_kl_lo - tolerances.kl_ci95_low)
    details["candidate_kl_ci95_low"] = cand_kl_lo
    details["baseline_kl_ci95_low"] = base_kl_lo

    cand_kl_hi = _metric(candidate, "kl_mean", "ci95_high")
    base_kl_hi = _metric(baseline, "kl_mean", "ci95_high")
    checks["kl_ci95_high_nonreg"] = cand_kl_hi <= (base_kl_hi + tolerances.kl_ci95_high)
    details["candidate_kl_ci95_high"] = cand_kl_hi
    details["baseline_kl_ci95_high"] = base_kl_hi

    cand_drift_hi = _metric(candidate, "temporal_drift", "ci95_high")
    base_drift_hi = _metric(baseline, "temporal_drift", "ci95_high")
    checks["temporal_drift_ci95_high_nonreg"] = cand_drift_hi <= (
        base_drift_hi + tolerances.temporal_drift_ci95_high
    )
    details["candidate_temporal_drift_ci95_high"] = cand_drift_hi
    details["baseline_temporal_drift_ci95_high"] = base_drift_hi

    cand_ood_lo = _metric(candidate, "ood_gap", "ci95_low")
    base_ood_lo = _metric(baseline, "ood_gap", "ci95_low")
    checks["ood_gap_ci95_low_nonreg"] = cand_ood_lo >= (base_ood_lo - tolerances.ood_gap_ci95_low)
    details["candidate_ood_gap_ci95_low"] = cand_ood_lo
    details["baseline_ood_gap_ci95_low"] = base_ood_lo

    cand_gate_mean = _metric(candidate, "gate_pass_rate", "mean")
    base_gate_mean = _metric(baseline, "gate_pass_rate", "mean")
    checks["gate_pass_rate_mean_nonreg"] = cand_gate_mean >= (base_gate_mean - tolerances.gate_pass_rate_mean)
    details["candidate_gate_pass_rate_mean"] = cand_gate_mean
    details["baseline_gate_pass_rate_mean"] = base_gate_mean

    if require_pass_all:
        cand_pass_all = bool(candidate.get("pass_all", False))
        checks["candidate_pass_all"] = cand_pass_all
        details["candidate_pass_all"] = cand_pass_all

    overall = all(checks.values())
    details["overall_pass"] = overall
    details["require_pass_all"] = require_pass_all
    details["tolerances"] = asdict(tolerances)
    return overall, checks, details

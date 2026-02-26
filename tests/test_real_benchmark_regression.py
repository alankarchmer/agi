from __future__ import annotations

from benchmarks.regression import RegressionTolerances, compare_real_benchmark_against_baseline


def _summary(
    nll_hi: float = 10.0,
    kl_lo: float = 1.0,
    kl_hi: float = 2.0,
    drift_hi: float = 0.05,
    ood_lo: float = -0.01,
    gate_mean: float = 1.0,
    pass_all: bool = True,
) -> dict:
    return {
        "pass_all": pass_all,
        "metrics": {
            "nll_mean": {"ci95_high": nll_hi},
            "kl_mean": {"ci95_low": kl_lo, "ci95_high": kl_hi},
            "temporal_drift": {"ci95_high": drift_hi},
            "ood_gap": {"ci95_low": ood_lo},
            "gate_pass_rate": {"mean": gate_mean},
        },
    }


def test_regression_identical_passes() -> None:
    baseline = _summary()
    candidate = _summary()
    overall, checks, _ = compare_real_benchmark_against_baseline(
        candidate=candidate,
        baseline=baseline,
        tolerances=RegressionTolerances(),
        require_pass_all=True,
    )
    assert overall is True
    assert all(checks.values())


def test_regression_detects_worse_nll() -> None:
    baseline = _summary(nll_hi=10.0)
    candidate = _summary(nll_hi=10.1)
    overall, checks, _ = compare_real_benchmark_against_baseline(
        candidate=candidate,
        baseline=baseline,
        tolerances=RegressionTolerances(),
        require_pass_all=True,
    )
    assert overall is False
    assert checks["nll_ci95_high_nonreg"] is False


def test_regression_tolerance_allows_small_degradation() -> None:
    baseline = _summary(nll_hi=10.0)
    candidate = _summary(nll_hi=10.1)
    overall, checks, _ = compare_real_benchmark_against_baseline(
        candidate=candidate,
        baseline=baseline,
        tolerances=RegressionTolerances(nll_ci95_high=0.2),
        require_pass_all=True,
    )
    assert overall is True
    assert checks["nll_ci95_high_nonreg"] is True

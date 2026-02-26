from __future__ import annotations

from benchmarks.strict_harness import BenchmarkSummary, SeedStats, _ci95, summary_to_json


def test_ci95_computation_has_expected_ordering() -> None:
    stats = _ci95([0.1, 0.2, 0.3, 0.4, 0.5])
    assert stats.ci95_low <= stats.mean <= stats.ci95_high
    assert stats.std > 0.0


def test_summary_to_json_contains_expected_fields() -> None:
    summary = BenchmarkSummary(
        pass_all=True,
        threshold_results={"a": True},
        metrics={"m": SeedStats(values=[1.0], mean=1.0, std=0.0, ci95_low=1.0, ci95_high=1.0)},
        per_seed={"m": [1.0]},
        profile="phase6",
        mode="smoke",
    )
    out = summary_to_json(summary)

    assert out["profile"] == "phase6"
    assert out["mode"] == "smoke"
    assert out["pass_all"] is True
    assert out["threshold_results"]["a"] is True
    assert out["metrics"]["m"]["mean"] == 1.0
    assert out["per_seed"]["m"] == [1.0]

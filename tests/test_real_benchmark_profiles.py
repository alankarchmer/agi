from __future__ import annotations

from benchmarks.real_data_harness import (
    RealBenchmarkSummary,
    SeedStats,
    eval_cfg_for_profile,
    summary_to_json,
    thresholds_for_profile,
)


def test_pusht_profile_is_less_strict_than_legacy_for_nll_and_kl_max() -> None:
    legacy = thresholds_for_profile("legacy")
    pusht = thresholds_for_profile("pusht")

    assert pusht.nll_ci95_high_max > legacy.nll_ci95_high_max
    assert pusht.kl_ci95_high_max > legacy.kl_ci95_high_max


def test_eval_cfg_profile_alignment() -> None:
    legacy = eval_cfg_for_profile("legacy")
    pusht = eval_cfg_for_profile("pusht")

    assert legacy.nll_max < pusht.nll_max
    assert legacy.kl_max < pusht.kl_max


def test_real_benchmark_summary_json_includes_profile() -> None:
    summary = RealBenchmarkSummary(
        pass_all=True,
        threshold_results={"ok": True},
        metrics={"nll": SeedStats(values=[1.0], mean=1.0, std=0.0, ci95_low=1.0, ci95_high=1.0)},
        per_seed={"nll": [1.0]},
        profile="pusht",
        reference_checkpoint="/tmp/ref.pt",
    )
    out = summary_to_json(summary)
    assert out["profile"] == "pusht"
    assert out["reference_checkpoint"] == "/tmp/ref.pt"

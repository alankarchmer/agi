from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.real_data_harness import RealBenchmarkConfig, run_real_data_benchmarks, write_summary
from utils.runtime_hygiene import apply_runtime_warning_filters, runtime_health_message

apply_runtime_warning_filters()


def _parse_seeds(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _print_metric(name: str, mean: float, low: float, high: float, std: float) -> None:
    print(f"  {name}: mean={mean:.4f}, std={std:.4f}, ci95=[{low:.4f}, {high:.4f}]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run real-data ASL benchmark harness")
    parser.add_argument("--profile", choices=["pusht", "legacy"], default="pusht")
    parser.add_argument("--data-source", type=str, required=True)
    parser.add_argument("--action-space-type", choices=["discrete", "continuous"], default="discrete")
    parser.add_argument("--action-dim", type=int, default=3)
    parser.add_argument("--obs-likelihood", choices=["mse", "gaussian", "bernoulli"], default="gaussian")
    parser.add_argument("--seeds", type=str, default="7,11,19,23,29")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--reference-checkpoint", type=str, default=None)
    parser.add_argument("--checkpoint-mode", choices=["strict", "partial"], default="strict")
    parser.add_argument("--output", type=str, default="runs/benchmarks/real_data_summary.json")
    parser.add_argument("--no-fail-on-threshold", action="store_true")
    parser.add_argument("--show-runtime-health", action="store_true")
    args = parser.parse_args()

    if args.show_runtime_health:
        print(runtime_health_message())
        print()

    cfg = RealBenchmarkConfig(
        data_source=args.data_source,
        profile=args.profile,
        action_space_type=args.action_space_type,
        action_dim=args.action_dim,
        obs_likelihood=args.obs_likelihood,
        seeds=_parse_seeds(args.seeds),
        device=args.device,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        train_steps=args.steps,
        reference_checkpoint=args.reference_checkpoint,
        checkpoint_mode=args.checkpoint_mode,
    )

    try:
        summary = run_real_data_benchmarks(cfg)
    except ValueError as exc:
        msg = str(exc)
        if args.action_space_type == "discrete" and "Discrete actions must be shape" in msg:
            raise SystemExit(
                "Dataset actions appear continuous. Re-run with:\n"
                "  --action-space-type continuous --action-dim <dim>\n"
                "For lerobot/pusht, use: --action-space-type continuous --action-dim 2"
            ) from exc
        raise
    out_path = Path(args.output)
    write_summary(summary, out_path)

    print("[ASL Real-Data Benchmark Summary]")
    print(f"  profile: {summary.profile}")
    if summary.reference_checkpoint:
        print(f"  reference_checkpoint: {summary.reference_checkpoint}")
    for metric_name, stats in summary.metrics.items():
        _print_metric(metric_name, stats.mean, stats.ci95_low, stats.ci95_high, stats.std)

    print("\n[Threshold Checks]")
    for name, passed in summary.threshold_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")

    print(f"\nOverall: {'PASS' if summary.pass_all else 'FAIL'}")
    print(f"Summary JSON: {out_path}")

    if (not args.no_fail_on_threshold) and (not summary.pass_all):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

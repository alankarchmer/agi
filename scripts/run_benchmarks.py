from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.runtime_hygiene import apply_runtime_warning_filters, runtime_health_message

apply_runtime_warning_filters()

from benchmarks.strict_harness import BenchmarkConfig, run_strict_benchmarks, write_summary


def _parse_seeds(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _mode_defaults(mode: str) -> dict[str, int]:
    if mode == "smoke":
        return {
            "tmaze_train_steps": 100,
            "tmaze_num_sequences": 96,
            "tmaze_seq_len": 16,
            "tmaze_eval_episodes": 24,
            "grid_train_steps_l1": 80,
            "grid_train_steps_l2_pre": 50,
            "grid_train_steps_l2_post": 40,
            "grid_num_sequences": 96,
            "grid_seq_len": 16,
            "trap_episodes": 80,
        }
    return {
        "tmaze_train_steps": 180,
        "tmaze_num_sequences": 192,
        "tmaze_seq_len": 20,
        "tmaze_eval_episodes": 40,
        "grid_train_steps_l1": 140,
        "grid_train_steps_l2_pre": 90,
        "grid_train_steps_l2_post": 70,
        "grid_num_sequences": 160,
        "grid_seq_len": 18,
        "trap_episodes": 140,
    }


def _print_metric(name: str, mean: float, low: float, high: float, std: float) -> None:
    print(f"  {name}: mean={mean:.4f}, std={std:.4f}, ci95=[{low:.4f}, {high:.4f}]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run strict ASL benchmark harness")
    parser.add_argument("--seeds", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output", type=str, default="runs/benchmarks/strict_summary.json")
    parser.add_argument("--profile", choices=["legacy", "phase6"], default="legacy")
    parser.add_argument("--mode", choices=["smoke", "full"], default="full")
    parser.add_argument(
        "--no-fail-on-threshold",
        action="store_true",
        help="Do not return non-zero exit code when any strict threshold fails.",
    )

    parser.add_argument("--tmaze-train-steps", type=int, default=None)
    parser.add_argument("--grid-l1-steps", type=int, default=None)
    parser.add_argument("--grid-l2-pre-steps", type=int, default=None)
    parser.add_argument("--grid-l2-post-steps", type=int, default=None)
    parser.add_argument("--trap-episodes", type=int, default=None)
    parser.add_argument(
        "--show-runtime-health",
        action="store_true",
        help="Print package hygiene diagnostics for the active Python environment.",
    )

    args = parser.parse_args()

    if args.show_runtime_health:
        print(runtime_health_message())
        print()

    defaults = _mode_defaults(args.mode)
    default_seed_str = "7,11" if args.mode == "smoke" else "7,11,19,23,29"
    seed_str = args.seeds if args.seeds is not None else default_seed_str

    cfg = BenchmarkConfig(
        seeds=_parse_seeds(seed_str),
        device=args.device,
        profile=args.profile,
        mode=args.mode,
        tmaze_train_steps=(
            int(args.tmaze_train_steps) if args.tmaze_train_steps is not None else defaults["tmaze_train_steps"]
        ),
        tmaze_num_sequences=defaults["tmaze_num_sequences"],
        tmaze_seq_len=defaults["tmaze_seq_len"],
        tmaze_eval_episodes=defaults["tmaze_eval_episodes"],
        grid_train_steps_l1=int(args.grid_l1_steps) if args.grid_l1_steps is not None else defaults["grid_train_steps_l1"],
        grid_train_steps_l2_pre=(
            int(args.grid_l2_pre_steps) if args.grid_l2_pre_steps is not None else defaults["grid_train_steps_l2_pre"]
        ),
        grid_train_steps_l2_post=(
            int(args.grid_l2_post_steps) if args.grid_l2_post_steps is not None else defaults["grid_train_steps_l2_post"]
        ),
        grid_num_sequences=defaults["grid_num_sequences"],
        grid_seq_len=defaults["grid_seq_len"],
        trap_episodes=int(args.trap_episodes) if args.trap_episodes is not None else defaults["trap_episodes"],
    )

    summary = run_strict_benchmarks(cfg)
    out_path = Path(args.output)
    write_summary(summary, out_path)

    print("[ASL Strict Benchmark Summary]")
    print(f"  profile={summary.profile}, mode={summary.mode}, seeds={cfg.seeds}")
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

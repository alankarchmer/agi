#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/.venv/bin/python"

if [[ ! -x "$PY" ]]; then
  echo "Missing venv python at: $PY"
  echo "Recreate with: python3 -m venv $ROOT/.venv && $ROOT/.venv/bin/pip install -e $ROOT"
  exit 1
fi

DATA_SOURCE="${DATA_SOURCE:-$ROOT/data/lerobot_pusht_npz}"
BASELINE_JSON="${BASELINE_JSON:-$ROOT/configs/baselines/phase6_real_pusht_v1.benchmark.json}"
DEVICE="${DEVICE:-cpu}"
SEEDS="${SEEDS:-7,11,19,23,29}"
OUTPUT_JSON="${OUTPUT_JSON:-$ROOT/runs/benchmarks/finals_rssm_large_s7_kl_drift_tuned.json}"
REGRESSION_JSON="${REGRESSION_JSON:-$ROOT/runs/benchmarks/finals_rssm_large_s7_kl_drift_tuned_vs_baseline.json}"

if [[ $# -ge 1 ]]; then
  CKPT="$1"
else
  CKPT="$(find "$ROOT/runs/matrix/rssm_large_s7_kl_drift_tuned" -type f -name final_world_model.pt 2>/dev/null | sort | tail -1 || true)"
fi

if [[ ! -f "$CKPT" ]]; then
  echo "Checkpoint not found: $CKPT"
  echo "Expected default under: $ROOT/runs/matrix/rssm_large_s7_kl_drift_tuned/*/final_world_model.pt"
  echo "Pass it explicitly:"
  echo "  $0 /absolute/path/to/final_world_model.pt"
  exit 1
fi

mkdir -p "$(dirname "$OUTPUT_JSON")"

echo "[1/2] Running benchmark..."
echo "  checkpoint: $CKPT"
echo "  output: $OUTPUT_JSON"
"$PY" "$ROOT/scripts/run_real_benchmarks.py" \
  --profile pusht \
  --data-source "$DATA_SOURCE" \
  --action-space-type continuous \
  --action-dim 2 \
  --obs-likelihood gaussian \
  --seeds "$SEEDS" \
  --device "$DEVICE" \
  --reference-checkpoint "$CKPT" \
  --checkpoint-mode strict \
  --output "$OUTPUT_JSON"

echo "[2/2] Running regression gate..."
echo "  baseline: $BASELINE_JSON"
echo "  output: $REGRESSION_JSON"
"$PY" "$ROOT/scripts/check_real_benchmark_regression.py" \
  --candidate "$OUTPUT_JSON" \
  --baseline "$BASELINE_JSON" \
  --output "$REGRESSION_JSON"

echo "[Complete]"
echo "  benchmark: $OUTPUT_JSON"
echo "  regression: $REGRESSION_JSON"

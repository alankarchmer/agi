# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ASL v1 is the reference implementation of the **Active Strange Loop** architecture — a multi-phase reinforcement learning system that builds hierarchical world models through active exploration, causal discovery, and loop closure. It targets POMDPs with partial observability and supports both synthetic benchmarks and real-world robotic data (LeRobot PushT).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -e '.[dev]'        # ruff, mypy, pytest-cov
pip install -e '.[hf]'         # HuggingFace datasets (LeRobot)
```

## Common Commands

```bash
# Tests
python -m pytest -q                                   # all tests
python -m pytest tests/test_rssm.py -q                # single test file
python -m pytest -k "test_phase1" -q                  # filter by name

# Level-1 convergence (mandatory gate)
python scripts/test_level1.py
python scripts/test_level1.py --steps 220 --num-sequences 192 --seq-len 20 --batch-size 32

# Linting / formatting
ruff check .
ruff format .
mypy <module>

# Runtime hygiene check
python scripts/check_runtime.py --strict
```

## Training Pipelines

```bash
# Synthetic environments (phases 1-5, quick smoke mode)
python scripts/run_phases.py --quick

# Phase 6 reliability pipeline (sigma-prior ramping + EI audits)
python scripts/run_phase6.py --output-dir runs/phase6

# Real-data training (offline-first, supports NPZ/JSONL/CSV)
python scripts/train_real_data.py \
  --data-source /path/to/episodes.npz \
  --action-space-type discrete \
  --action-dim 3 \
  --obs-likelihood gaussian \
  --steps 2000 --use-amp
```

Device flags available on all scripts: `--mps`, `--cuda`, `--cpu` (defaults to MPS on Apple Silicon).

## Benchmarks

```bash
# Strict multi-seed synthetic (T-Maze + gridworld + ambiguity-trap)
python scripts/run_benchmarks.py --profile phase6 --mode full \
  --seeds 7,11,19,23,29 --output runs/benchmarks/strict_summary.json

# Real-data CI benchmark
python scripts/run_real_benchmarks.py \
  --data-source data/lerobot_pusht_npz --profile pusht \
  --seeds 7,11,19,23,29 --output runs/benchmarks/real_data_summary.json

# Regression gate against locked baseline
python scripts/check_real_benchmark_regression.py \
  --candidate runs/benchmarks/real_data_summary.json \
  --baseline configs/baselines/phase6_real_pusht_v1.benchmark.json
```

`phase6` profile enforces CI-bound gates (lower/upper bounds); `legacy` uses mean-threshold gates.

## Architecture

### Training Phases

Six sequential training phases in `training/`:

| Phase | File | Purpose |
|-------|------|---------|
| 1 | `phase1_world_model.py` | Foundation RSSM with variational free energy + beta-annealing |
| 2 | `phase2_attractor.py` | Non-stationary attractor dynamics via spectral loss |
| 3 | `phase3_causal.py` | Causal structure discovery via direct-effect interventions (NIS) |
| 4 | `phase4_joint.py` | Joint training: RSSM + attractor + macro-state flow |
| 5 | `phase5_loop_closure.py` | Feedback from attractor back to micro-state |
| 6 | `phase6_reliability.py` / `phase6_real_data.py` | Sigma-prior ramping + EI audits + real-data support |

**Important**: Cross-level gradients are intentionally detached in phases 1–3.

### Core Models (`models/`)

- **`rssm.py`**: Central model — Gaussian encoder/decoder with hierarchical latents (`h`: hidden state, `z`: stochastic latent, `sigma`: uncertainty, `y_macro`: macro-level state)
- **`attractor.py`**: AttractorDynamics — settling-based non-stationary dynamics with Jacobian
- **`joots.py`**: JOOTS Controller — detects learning plateaus and triggers SAM + SGLD recovery
- **`nis.py`**: Neural Intervention Scoring — causal intervention mechanism

### Key Supporting Modules

- **`optim/adaptive_optimizer.py`**: Adam wrapper with SAM (Sharpness Aware Minimization) and SGLD noise injection
- **`training/contracts.py`**: All phase configuration dataclasses — the canonical source of hyperparameter defaults
- **`training/datasets.py`**: Unified loader for NPZ (episode dirs or batched), JSONL, CSV, and HuggingFace datasets
- **`training/offline_online.py`**: Offline pretrain → online fine-tuning framework
- **`utils/device.py`**: Unified device selection
- **`configs/default.yaml`**: Full hyperparameter manifest

### Environments (`envs/`)

- `random_walk_1d.py`: Simple exploration baseline
- `tmaze.py`: T-Maze POMDP with hidden goal + informative cue
- `non_stationary_gridworld.py`: Dynamic gridworld (environment changes during training)
- `ambiguity_trap.py`: Adversarial disambiguation environment

### Output Structure

All runs write artifacts to `runs/`:
- `runs/real_data/<timestamp>/tb/` — TensorBoard logs
- `runs/phase6/<timestamp>/` — `summary.json`, `stage_metrics.csv`, `gates.json`, `checkpoints/`

TensorBoard: `tensorboard --logdir runs/real_data --port 6006`
If `pkg_resources` error: `python scripts/tensorboard_compat.py --logdir runs/real_data --port 6006`

## Notes

- WandB is optional and auto-disabled when not installed.
- Replace deprecated `gym`/`shimmy`/`pynvml` with `gymnasium`/`nvidia-ml-py` if detected.
- `configs/baselines/phase6_real_pusht_v1.benchmark.json` is a locked regression baseline — do not modify without updating nightly CI.

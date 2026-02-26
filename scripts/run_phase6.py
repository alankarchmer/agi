from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.tmaze import TMazeEnv
from models.attractor import AttractorDynamics
from models.joots import JOOTSController
from models.nis import MacroTransition, NISMacroState
from models.rssm import GaussianRSSM
from training.contracts import (
    LoggingConfig,
    OptimizationConfig,
    Phase1Config,
    Phase2Config,
    Phase3Config,
    Phase4Config,
    Phase5Config,
    Phase6Config,
)
from training.data_collection import collect_random_trajectories
from training.phase1_world_model import train_phase1_world_model
from training.phase2_attractor import train_phase2_attractor
from training.phase3_causal import train_phase3_causal
from training.phase4_joint import train_phase4_joint
from training.phase5_loop_closure import train_phase5_loop_closure
from training.phase6_reliability import train_phase6_reliability
from utils.device import get_device
from utils.runtime_hygiene import apply_runtime_warning_filters, runtime_health_message
from utils.seed import set_seed

apply_runtime_warning_filters()


def _parse_float_list(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _latest(values: list[float]) -> float:
    return float(values[-1]) if values else float("nan")


def _infer_ckpt_dims(state: dict[str, torch.Tensor]) -> dict[str, int | None]:
    obs_dim = int(state["obs_mean"].numel()) if "obs_mean" in state else None
    action_dim = int(state["action_mean"].numel()) if "action_mean" in state else None
    hidden_dim = None
    latent_dim = None
    if "gru_cell.bias_ih" in state:
        hidden_dim = int(state["gru_cell.bias_ih"].numel() // 3)
    if "prior_head.2.bias" in state:
        latent_dim = int(state["prior_head.2.bias"].numel() // 2)
    return {
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "hidden_dim": hidden_dim,
        "latent_dim": latent_dim,
    }


def _load_module_state(
    module: torch.nn.Module,
    state: dict[str, torch.Tensor],
    name: str,
    mode: str,
) -> tuple[int, int]:
    if mode == "strict":
        module.load_state_dict(state)
        return len(state), len(state)

    if mode != "partial":
        raise ValueError(f"Unknown checkpoint mode: {mode}")

    target = module.state_dict()
    matched: dict[str, torch.Tensor] = {}
    for k, v in state.items():
        if k in target and target[k].shape == v.shape:
            matched[k] = v

    if not matched:
        return 0, len(state)

    target.update(matched)
    module.load_state_dict(target, strict=False)
    return len(matched), len(state)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ASL Phase 6 reliability pipeline")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--mps", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--show-runtime-health", action="store_true")

    parser.add_argument("--seq-len", type=int, default=24)
    parser.add_argument("--num-sequences", type=int, default=256)

    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--latent-dim", type=int, default=24)
    parser.add_argument("--attractor-dim", type=int, default=64)
    parser.add_argument("--macro-dim", type=int, default=16)

    parser.add_argument("--phase1-steps", type=int, default=220)
    parser.add_argument("--phase2-steps", type=int, default=140)
    parser.add_argument("--phase3-steps", type=int, default=120)
    parser.add_argument("--phase4-steps", type=int, default=120)
    parser.add_argument("--phase5-steps", type=int, default=120)

    parser.add_argument("--sigma-ramp-weights", type=str, default="0.0,0.25,0.5,0.75,1.0")
    parser.add_argument("--sigma-stage-steps", type=int, default=120)
    parser.add_argument("--phase6-batch-size", type=int, default=32)
    parser.add_argument("--ei-audit-interval", type=int, default=50)
    parser.add_argument("--ei-audit-samples", type=int, default=256)
    parser.add_argument("--ei-interventions-per-dim", type=int, default=16)
    parser.add_argument("--jooots-cooldown-steps", type=int, default=25)
    parser.add_argument("--jooots-max-triggers-per-window", type=int, default=4)
    parser.add_argument("--tmaze-eval-seeds", type=str, default="7,11")
    parser.add_argument("--tmaze-eval-episodes", type=int, default=24)
    parser.add_argument("--resume-checkpoint", type=str, default=None)
    parser.add_argument("--checkpoint-mode", choices=["strict", "partial"], default="strict")
    parser.add_argument("--skip-warmup", action="store_true")

    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output-dir", type=str, default="runs/phase6")

    args = parser.parse_args()

    if args.show_runtime_health:
        print(runtime_health_message())
        print()

    if args.quick:
        args.phase1_steps = 80
        args.phase2_steps = 40
        args.phase3_steps = 30
        args.phase4_steps = 30
        args.phase5_steps = 30
        args.sigma_stage_steps = 24
        args.phase6_batch_size = 24
        args.num_sequences = 128
        args.seq_len = 16
        args.ei_audit_interval = 8
        args.ei_audit_samples = 96
        args.ei_interventions_per_dim = 6
        args.tmaze_eval_episodes = 12

    if args.cpu and (args.cuda or args.mps):
        raise ValueError("--cpu cannot be combined with --cuda or --mps")

    set_seed(args.seed)
    if args.cpu:
        device = torch.device("cpu")
    else:
        device = get_device(prefer_cuda=args.cuda, prefer_mps=(args.mps or not args.cuda))

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    stage_ckpt_dir = run_dir / "checkpoints"

    print(f"Running on device: {device}")
    print(f"Output directory: {run_dir}")

    dataset = collect_random_trajectories(
        env_fn=lambda: TMazeEnv(seed=args.seed),
        num_sequences=args.num_sequences,
        seq_len=args.seq_len,
        seed=args.seed,
    )
    obs_seq = dataset["obs_seq"]
    act_seq = dataset["act_seq"]

    world_model = GaussianRSSM(
        obs_dim=obs_seq.shape[-1],
        action_dim=act_seq.shape[-1],
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        sigma_dim=args.attractor_dim,
        macro_dim=args.macro_dim,
    )
    micro_dim = world_model.hidden_dim + world_model.latent_dim

    attractor = AttractorDynamics(micro_dim=micro_dim, attractor_dim=args.attractor_dim)
    nis = NISMacroState(micro_dim=args.attractor_dim, macro_dim=args.macro_dim)
    macro_transition = MacroTransition(macro_dim=args.macro_dim)

    if args.resume_checkpoint:
        payload = torch.load(args.resume_checkpoint, map_location="cpu")
        world_state = None
        if "world_model" in payload:
            world_state = payload["world_model"]
        elif "model" in payload:
            world_state = payload["model"]

        if world_state is not None:
            try:
                loaded, total = _load_module_state(world_model, world_state, "world_model", args.checkpoint_mode)
            except RuntimeError as exc:
                ckpt_dims = _infer_ckpt_dims(world_state)
                raise SystemExit(
                    "Checkpoint is incompatible with current Phase6 TMaze world-model architecture.\n"
                    f"Current model dims: obs_dim={world_model.obs_dim}, action_dim={world_model.action_dim}, "
                    f"hidden_dim={world_model.hidden_dim}, latent_dim={world_model.latent_dim}\n"
                    f"Checkpoint dims: obs_dim={ckpt_dims['obs_dim']}, action_dim={ckpt_dims['action_dim']}, "
                    f"hidden_dim={ckpt_dims['hidden_dim']}, latent_dim={ckpt_dims['latent_dim']}\n"
                    "Use a compatible TMaze checkpoint produced by a prior run_phase6 run, "
                    "or run without --resume-checkpoint, or use --checkpoint-mode partial."
                ) from exc
            print(f"[Resume] world_model tensors loaded: {loaded}/{total} (mode={args.checkpoint_mode})")
            if args.checkpoint_mode == "partial" and loaded == 0:
                ckpt_dims = _infer_ckpt_dims(world_state)
                raise SystemExit(
                    "No compatible world_model tensors found for partial checkpoint load.\n"
                    f"Current model dims: obs_dim={world_model.obs_dim}, action_dim={world_model.action_dim}, "
                    f"hidden_dim={world_model.hidden_dim}, latent_dim={world_model.latent_dim}\n"
                    f"Checkpoint dims: obs_dim={ckpt_dims['obs_dim']}, action_dim={ckpt_dims['action_dim']}, "
                    f"hidden_dim={ckpt_dims['hidden_dim']}, latent_dim={ckpt_dims['latent_dim']}\n"
                    "Use a compatible TMaze phase checkpoint or rerun without --resume-checkpoint."
                )

        if "attractor" in payload:
            loaded, total = _load_module_state(attractor, payload["attractor"], "attractor", args.checkpoint_mode)
            print(f"[Resume] attractor tensors loaded: {loaded}/{total} (mode={args.checkpoint_mode})")
        if "nis" in payload:
            loaded, total = _load_module_state(nis, payload["nis"], "nis", args.checkpoint_mode)
            print(f"[Resume] nis tensors loaded: {loaded}/{total} (mode={args.checkpoint_mode})")
        if "macro_transition" in payload:
            loaded, total = _load_module_state(
                macro_transition, payload["macro_transition"], "macro_transition", args.checkpoint_mode
            )
            print(f"[Resume] macro_transition tensors loaded: {loaded}/{total} (mode={args.checkpoint_mode})")

    warmup_controller = JOOTSController(patience=60)
    phase6_controller = JOOTSController(
        patience=60,
        tolerance=1e-4,
        severe_var_threshold=1e-8,
        loss_floor=1e-4,
        gsnr_floor=0.25,
        residual_floor=1e-3,
        cooldown_steps=args.jooots_cooldown_steps,
        max_triggers_per_window=args.jooots_max_triggers_per_window,
        recovery_window=20,
        sam_steps=25,
        sgld_variance=1e-2,
    )

    optim_cfg = OptimizationConfig(lr_l1=3e-4, lr_l2=1e-3, lr_l3=1e-4, grad_clip_norm=1.0)

    # Warm-up through phase 1 -> 5.
    if args.skip_warmup:
        phase1 = phase2 = phase3 = phase4 = phase5 = None
    else:
        phase1 = train_phase1_world_model(
            model=world_model,
            obs_seq=obs_seq,
            act_seq=act_seq,
            phase_cfg=Phase1Config(steps=args.phase1_steps, batch_size=32),
            optim_cfg=optim_cfg,
            device=device,
            logger_cfg=LoggingConfig(log_dir=str(run_dir / "phase1"), use_wandb=False),
        )
        phase2 = train_phase2_attractor(
            world_model=world_model,
            attractor=attractor,
            obs_seq=obs_seq,
            act_seq=act_seq,
            phase_cfg=Phase2Config(steps=args.phase2_steps, batch_size=32, lambda_spec=1.0),
            optim_cfg=optim_cfg,
            device=device,
            logger_cfg=LoggingConfig(log_dir=str(run_dir / "phase2"), use_wandb=False),
        )
        phase3 = train_phase3_causal(
            world_model=world_model,
            attractor=attractor,
            nis=nis,
            macro_transition=macro_transition,
            obs_seq=obs_seq,
            act_seq=act_seq,
            phase_cfg=Phase3Config(steps=args.phase3_steps, batch_size=32, alpha_dei=0.1),
            optim_cfg=optim_cfg,
            device=device,
            logger_cfg=LoggingConfig(log_dir=str(run_dir / "phase3"), use_wandb=False),
        )
        phase4 = train_phase4_joint(
            world_model=world_model,
            attractor=attractor,
            nis=nis,
            macro_transition=macro_transition,
            controller=warmup_controller,
            obs_seq=obs_seq,
            act_seq=act_seq,
            phase_cfg=Phase4Config(steps=args.phase4_steps, batch_size=32, alpha_dei=0.05),
            optim_cfg=optim_cfg,
            device=device,
            logger_cfg=LoggingConfig(log_dir=str(run_dir / "phase4"), use_wandb=False),
        )
        phase5 = train_phase5_loop_closure(
            world_model=world_model,
            attractor=attractor,
            nis=nis,
            macro_transition=macro_transition,
            obs_seq=obs_seq,
            act_seq=act_seq,
            phase_cfg=Phase5Config(steps=args.phase5_steps, batch_size=32, beta=1.0, macro_align_weight=0.25),
            optim_cfg=optim_cfg,
            device=device,
            logger_cfg=LoggingConfig(log_dir=str(run_dir / "phase5"), use_wandb=False),
        )

    phase6_cfg = Phase6Config(
        sigma_ramp_weights=_parse_float_list(args.sigma_ramp_weights),
        sigma_stage_steps=args.sigma_stage_steps,
        batch_size=args.phase6_batch_size,
        ei_audit_interval=args.ei_audit_interval,
        ei_audit_samples=args.ei_audit_samples,
        ei_interventions_per_dim=args.ei_interventions_per_dim,
        jooots_cooldown_steps=args.jooots_cooldown_steps,
        jooots_max_triggers_per_window=args.jooots_max_triggers_per_window,
        tmaze_eval_seeds=_parse_int_list(args.tmaze_eval_seeds),
        tmaze_eval_episodes=args.tmaze_eval_episodes,
    )

    phase6 = train_phase6_reliability(
        world_model=world_model,
        attractor=attractor,
        nis=nis,
        macro_transition=macro_transition,
        controller=phase6_controller,
        obs_seq=obs_seq,
        act_seq=act_seq,
        phase_cfg=phase6_cfg,
        optim_cfg=optim_cfg,
        device=device,
        checkpoint_dir=stage_ckpt_dir,
        logger_cfg=LoggingConfig(log_dir=str(run_dir / "phase6"), use_wandb=False),
    )

    stage_metrics_path = run_dir / "stage_metrics.csv"
    with stage_metrics_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "stage_index",
                "sigma_weight",
                "attempts",
                "passed",
                "rolled_back",
                "median_vfe",
                "tmaze_ci_low",
                "tmaze_ci_high",
                "tmaze_ci_low_no_sigma",
                "tmaze_ci_high_no_sigma",
                "tmaze_adv_delta_vs_no_sigma",
                "dei_audit_last",
                "proxy_audit_spearman",
                "checkpoint_path",
            ]
        )
        for stage in phase6.stage_results:
            writer.writerow(
                [
                    stage.stage_index,
                    stage.sigma_weight,
                    stage.attempts,
                    int(stage.passed),
                    int(stage.rolled_back),
                    stage.median_vfe,
                    stage.tmaze_ci_low,
                    stage.tmaze_ci_high,
                    stage.tmaze_ci_low_no_sigma,
                    stage.tmaze_ci_high_no_sigma,
                    stage.tmaze_adv_delta_vs_no_sigma,
                    stage.dei_audit_last,
                    stage.proxy_audit_spearman,
                    stage.checkpoint_path or "",
                ]
            )

    gates = {
        "all_passed": phase6.all_passed,
        "stages": [
            {
                "stage_index": s.stage_index,
                "sigma_weight": s.sigma_weight,
                "passed": s.passed,
                "gates": s.gates,
            }
            for s in phase6.stage_results
        ],
    }
    gates_path = run_dir / "gates.json"
    gates_path.write_text(json.dumps(gates, indent=2))

    summary = {
        "config_version": "phase6-realdata-v1",
        "seed": args.seed,
        "device": str(device),
        "quick": bool(args.quick),
        "runtime": {
            "seq_len": args.seq_len,
            "num_sequences": args.num_sequences,
        },
        "config": vars(args),
        "warmup": {
            "phase1_loss": None if phase1 is None else _latest(phase1.loss_history),
            "phase2_loss": None if phase2 is None else _latest(phase2.loss_history),
            "phase3_loss": None if phase3 is None else _latest(phase3.loss_history),
            "phase4_total": None if phase4 is None else _latest(phase4.total_loss_history),
            "phase5_total": None if phase5 is None else _latest(phase5.total_loss_history),
        },
        "phase6": {
            "all_passed": phase6.all_passed,
            "total_jooots_triggers": phase6.total_jooots_triggers,
            "stage_count": len(phase6.stage_results),
            "proxy_history_len": len(phase6.proxy_history),
            "audit_history_len": len(phase6.audit_history),
        },
        "artifacts": {
            "stage_metrics_csv": str(stage_metrics_path),
            "gates_json": str(gates_path),
            "stage_checkpoints_dir": str(stage_ckpt_dir),
        },
    }
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print("[ASL Phase 6 Pipeline Complete]")
    print(f"  all_passed: {phase6.all_passed}")
    print(f"  total_jooots_triggers: {phase6.total_jooots_triggers}")
    print(f"  stage_count: {len(phase6.stage_results)}")
    print(f"  summary: {summary_path}")
    print(f"  stage_metrics: {stage_metrics_path}")
    print(f"  gates: {gates_path}")
    print(f"  stage checkpoints: {stage_ckpt_dir}")


if __name__ == "__main__":
    main()

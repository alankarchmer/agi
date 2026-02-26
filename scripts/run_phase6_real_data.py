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

from models.attractor import AttractorDynamics
from models.joots import JOOTSController
from models.nis import MacroTransition, NISMacroState
from models.rssm import GaussianRSSM
from training.contracts import EvalConfig, LoggingConfig, OptimizationConfig, Phase2Config, Phase3Config, Phase6Config
from training.datasets import EpisodeRecord, load_real_episodes, make_sequence_dataloader, split_episodes
from training.phase2_attractor import train_phase2_attractor
from training.phase3_causal import train_phase3_causal
from training.phase6_real_data import train_phase6_real_data
from utils.device import get_device
from utils.runtime_hygiene import apply_runtime_warning_filters, runtime_health_message
from utils.seed import set_seed

apply_runtime_warning_filters()


def _parse_float_list(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _obs_stats(episodes: list[EpisodeRecord]) -> tuple[torch.Tensor, torch.Tensor]:
    obs = torch.cat([ep.obs for ep in episodes], dim=0)
    return obs.mean(dim=0), obs.std(dim=0).clamp_min(1e-3)


def _action_stats(episodes: list[EpisodeRecord]) -> tuple[torch.Tensor, torch.Tensor]:
    act = torch.cat([ep.action for ep in episodes], dim=0)
    return act.mean(dim=0), act.std(dim=0).clamp_min(1e-3)


def _collect_sequence_bank(
    episodes: list[EpisodeRecord],
    seq_len: int,
    batch_size: int,
    stride: int,
    max_sequences: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    loader = make_sequence_dataloader(
        episodes=episodes,
        seq_len=seq_len,
        batch_size=batch_size,
        shuffle=True,
        stride=stride,
        include_partial=True,
        drop_last=False,
    )
    obs_chunks: list[torch.Tensor] = []
    act_chunks: list[torch.Tensor] = []
    mask_chunks: list[torch.Tensor] = []
    total = 0
    for batch in loader:
        obs_chunks.append(batch["obs_seq"])
        act_chunks.append(batch["act_seq"])
        mask_chunks.append(batch["valid_mask"])
        total += int(batch["obs_seq"].shape[0])
        if total >= max_sequences:
            break

    if not obs_chunks:
        raise ValueError("Unable to build training sequence bank from episodes.")

    obs_seq = torch.cat(obs_chunks, dim=0)[:max_sequences]
    act_seq = torch.cat(act_chunks, dim=0)[:max_sequences]
    valid_mask = torch.cat(mask_chunks, dim=0)[:max_sequences]
    return obs_seq, act_seq, valid_mask


def _infer_checkpoint_model_config(state: dict[str, torch.Tensor]) -> dict[str, int | str | None]:
    obs_dim = int(state["obs_mean"].numel()) if "obs_mean" in state else None
    action_dim = int(state["action_mean"].numel()) if "action_mean" in state else None

    hidden_dim = None
    latent_dim = None
    if "gru_cell.bias_ih" in state:
        hidden_dim = int(state["gru_cell.bias_ih"].numel() // 3)
    if "prior_head.2.bias" in state:
        latent_dim = int(state["prior_head.2.bias"].numel() // 2)

    sigma_dim = None
    if hidden_dim is not None and "prior_head.0.weight" in state:
        sigma_dim = int(state["prior_head.0.weight"].shape[1] - hidden_dim)

    macro_dim = None
    if obs_dim is not None and "obs_encoder.0.weight" in state:
        in_dim = int(state["obs_encoder.0.weight"].shape[1])
        macro_dim = max(0, in_dim - obs_dim)

    obs_likelihood = "gaussian" if "decoder_log_std" in state else None
    action_space_type = None
    if action_dim is not None and "policy_head.2.bias" in state:
        out_dim = int(state["policy_head.2.bias"].numel())
        if out_dim == action_dim:
            action_space_type = "discrete"
        elif out_dim == 2 * action_dim:
            action_space_type = "continuous"

    return {
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "hidden_dim": hidden_dim,
        "latent_dim": latent_dim,
        "sigma_dim": sigma_dim,
        "macro_dim": macro_dim,
        "obs_likelihood": obs_likelihood,
        "action_space_type": action_space_type,
    }


def _load_module_state(
    module: torch.nn.Module,
    state: dict[str, torch.Tensor],
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
    parser = argparse.ArgumentParser(description="Run Phase 6 reliability on real datasets/checkpoints")
    parser.add_argument("--data-source", type=str, required=True)
    parser.add_argument("--action-space-type", choices=["discrete", "continuous"], default="continuous")
    parser.add_argument("--action-dim", type=int, default=None)
    parser.add_argument("--obs-likelihood", choices=["mse", "gaussian", "bernoulli"], default=None)
    parser.add_argument("--train-split", type=float, default=0.8)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--test-split", type=float, default=0.1)

    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-sequences", type=int, default=512)

    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--latent-dim", type=int, default=None)
    parser.add_argument("--attractor-dim", type=int, default=None)
    parser.add_argument("--macro-dim", type=int, default=None)

    parser.add_argument("--resume-checkpoint", type=str, default=None)
    parser.add_argument("--checkpoint-mode", choices=["strict", "partial"], default="strict")
    parser.add_argument("--skip-prep-phases", action="store_true")
    parser.add_argument("--phase2-steps", type=int, default=120)
    parser.add_argument("--phase3-steps", type=int, default=120)

    parser.add_argument("--sigma-ramp-weights", type=str, default="0.0,0.25,0.5,0.75,1.0")
    parser.add_argument("--sigma-stage-steps", type=int, default=120)
    parser.add_argument("--phase6-batch-size", type=int, default=32)
    parser.add_argument("--ei-audit-interval", type=int, default=50)
    parser.add_argument("--ei-audit-samples", type=int, default=256)
    parser.add_argument("--ei-interventions-per-dim", type=int, default=16)
    parser.add_argument("--jooots-cooldown-steps", type=int, default=25)
    parser.add_argument("--jooots-max-triggers-per-window", type=int, default=4)

    parser.add_argument("--eval-nll-max", type=float, default=8000.0)
    parser.add_argument("--eval-kl-min", type=float, default=1.0)
    parser.add_argument("--eval-kl-max", type=float, default=120.0)
    parser.add_argument("--eval-ood-drift-max", type=float, default=0.25)
    parser.add_argument("--eval-intervention-spearman-min", type=float, default=0.30)

    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--mps", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--show-runtime-health", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output-dir", type=str, default="runs/phase6_real_data")
    args = parser.parse_args()

    if args.show_runtime_health:
        print(runtime_health_message())
        print()

    if args.quick:
        args.max_sequences = min(args.max_sequences, 192)
        args.phase2_steps = min(args.phase2_steps, 30)
        args.phase3_steps = min(args.phase3_steps, 30)
        args.sigma_stage_steps = min(args.sigma_stage_steps, 24)
        args.phase6_batch_size = min(args.phase6_batch_size, 24)
        args.ei_audit_interval = min(args.ei_audit_interval, 8)
        args.ei_audit_samples = min(args.ei_audit_samples, 96)
        args.ei_interventions_per_dim = min(args.ei_interventions_per_dim, 6)

    if args.cpu and (args.cuda or args.mps):
        raise ValueError("--cpu cannot be combined with --cuda or --mps")
    if args.action_space_type == "discrete" and args.action_dim is None:
        raise ValueError("--action-dim is required for discrete action datasets.")

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

    action_dim_hint = int(args.action_dim) if args.action_dim is not None else 1
    episodes = load_real_episodes(
        source=args.data_source,
        action_space_type=args.action_space_type,
        action_dim=action_dim_hint,
    )
    train_eps, val_eps, test_eps = split_episodes(
        episodes=episodes,
        train_frac=args.train_split,
        val_frac=args.val_split,
        test_frac=args.test_split,
        seed=args.seed,
    )
    eval_eps = test_eps if test_eps else val_eps
    if not train_eps or not eval_eps:
        raise ValueError("Need non-empty train and eval episode splits.")

    obs_dim_data = int(train_eps[0].obs.shape[-1])
    act_dim_data = int(train_eps[0].action.shape[-1])
    if args.action_dim is not None and int(args.action_dim) != act_dim_data:
        raise ValueError(
            f"--action-dim={args.action_dim} does not match dataset action dim {act_dim_data} "
            "(after preprocessing)."
        )

    world_state = None
    payload = None
    inferred: dict[str, int | str | None] = {}
    if args.resume_checkpoint:
        payload = torch.load(args.resume_checkpoint, map_location="cpu")
        world_state = payload.get("world_model", payload.get("model"))
        if world_state is not None:
            inferred = _infer_checkpoint_model_config(world_state)

    hidden_dim = int(args.hidden_dim or inferred.get("hidden_dim") or 128)
    latent_dim = int(args.latent_dim or inferred.get("latent_dim") or 32)
    sigma_dim = int(args.attractor_dim or inferred.get("sigma_dim") or 64)
    macro_dim = int(args.macro_dim or inferred.get("macro_dim") or 16)
    obs_likelihood = str(args.obs_likelihood or inferred.get("obs_likelihood") or "gaussian")

    if sigma_dim <= 0:
        raise ValueError("Real-data Phase 6 requires sigma_dim > 0.")
    if macro_dim <= 0:
        raise ValueError("Real-data Phase 6 requires macro_dim > 0.")

    world_model = GaussianRSSM(
        obs_dim=obs_dim_data,
        action_dim=act_dim_data,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        sigma_dim=sigma_dim,
        macro_dim=macro_dim,
        obs_encoder_type="mlp",
        action_space_type=args.action_space_type,
        obs_likelihood=obs_likelihood,
        normalize_obs=True,
        normalize_action=(args.action_space_type == "continuous"),
    )
    micro_dim = world_model.hidden_dim + world_model.latent_dim

    attractor = AttractorDynamics(micro_dim=micro_dim, attractor_dim=world_model.sigma_dim)
    nis = NISMacroState(micro_dim=world_model.sigma_dim, macro_dim=world_model.macro_dim)
    macro_transition = MacroTransition(macro_dim=world_model.macro_dim)

    if world_state is not None:
        try:
            loaded, total = _load_module_state(world_model, world_state, args.checkpoint_mode)
        except RuntimeError as exc:
            raise SystemExit(
                "Failed to load world_model checkpoint into real-data Phase6 model. "
                "Use compatible dims or --checkpoint-mode partial."
            ) from exc
        print(f"[Resume] world_model tensors loaded: {loaded}/{total} (mode={args.checkpoint_mode})")

    if payload is not None and "attractor" in payload:
        loaded, total = _load_module_state(attractor, payload["attractor"], args.checkpoint_mode)
        print(f"[Resume] attractor tensors loaded: {loaded}/{total} (mode={args.checkpoint_mode})")
    if payload is not None and "nis" in payload:
        loaded, total = _load_module_state(nis, payload["nis"], args.checkpoint_mode)
        print(f"[Resume] nis tensors loaded: {loaded}/{total} (mode={args.checkpoint_mode})")
    if payload is not None and "macro_transition" in payload:
        loaded, total = _load_module_state(macro_transition, payload["macro_transition"], args.checkpoint_mode)
        print(f"[Resume] macro_transition tensors loaded: {loaded}/{total} (mode={args.checkpoint_mode})")

    obs_mean, obs_std = _obs_stats(train_eps)
    act_mean, act_std = _action_stats(train_eps)
    world_model.set_normalization_stats(obs_mean=obs_mean, obs_std=obs_std, action_mean=act_mean, action_std=act_std)

    obs_seq, act_seq, valid_mask = _collect_sequence_bank(
        episodes=train_eps,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        stride=args.stride,
        max_sequences=args.max_sequences,
    )

    eval_loader = make_sequence_dataloader(
        episodes=eval_eps,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        shuffle=False,
        stride=args.stride,
        include_partial=True,
        drop_last=False,
    )

    optim_cfg = OptimizationConfig(lr_l1=3e-4, lr_l2=1e-3, lr_l3=1e-4, grad_clip_norm=1.0)

    phase2 = phase3 = None
    if not args.skip_prep_phases:
        phase2 = train_phase2_attractor(
            world_model=world_model,
            attractor=attractor,
            obs_seq=obs_seq,
            act_seq=act_seq,
            valid_mask_seq=valid_mask,
            phase_cfg=Phase2Config(steps=args.phase2_steps, batch_size=args.batch_size, lambda_spec=1.0),
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
            valid_mask_seq=valid_mask,
            phase_cfg=Phase3Config(steps=args.phase3_steps, batch_size=args.batch_size, alpha_dei=0.1),
            optim_cfg=optim_cfg,
            device=device,
            logger_cfg=LoggingConfig(log_dir=str(run_dir / "phase3"), use_wandb=False),
        )

    controller = JOOTSController(
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

    phase6_cfg = Phase6Config(
        sigma_ramp_weights=_parse_float_list(args.sigma_ramp_weights),
        sigma_stage_steps=args.sigma_stage_steps,
        batch_size=args.phase6_batch_size,
        ei_audit_interval=args.ei_audit_interval,
        ei_audit_samples=args.ei_audit_samples,
        ei_interventions_per_dim=args.ei_interventions_per_dim,
        jooots_cooldown_steps=args.jooots_cooldown_steps,
        jooots_max_triggers_per_window=args.jooots_max_triggers_per_window,
        tmaze_eval_seeds=_parse_int_list("7,11"),
        tmaze_eval_episodes=1,
    )
    eval_cfg = EvalConfig(
        nll_max=args.eval_nll_max,
        kl_min=args.eval_kl_min,
        kl_max=args.eval_kl_max,
        ood_drift_max=args.eval_ood_drift_max,
        intervention_trend_min_spearman=args.eval_intervention_spearman_min,
    )

    phase6 = train_phase6_real_data(
        world_model=world_model,
        attractor=attractor,
        nis=nis,
        macro_transition=macro_transition,
        controller=controller,
        obs_seq=obs_seq,
        act_seq=act_seq,
        valid_mask_seq=valid_mask,
        eval_loader=eval_loader,
        phase_cfg=phase6_cfg,
        optim_cfg=optim_cfg,
        eval_cfg=eval_cfg,
        device=device,
        checkpoint_dir=stage_ckpt_dir,
        logger_cfg=LoggingConfig(log_dir=str(run_dir / "phase6_real"), use_wandb=False),
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
                "eval_nll_mean",
                "eval_kl_mean",
                "eval_temporal_drift",
                "eval_ood_gap",
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
                    stage.eval_nll_mean,
                    stage.eval_kl_mean,
                    stage.eval_temporal_drift,
                    stage.eval_ood_gap,
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

    final_ckpt = run_dir / "final_phase6_real.pt"
    torch.save(
        {
            "world_model": world_model.state_dict(),
            "attractor": attractor.state_dict(),
            "nis": nis.state_dict(),
            "macro_transition": macro_transition.state_dict(),
            "config": vars(args),
        },
        final_ckpt,
    )

    summary = {
        "seed": args.seed,
        "device": str(device),
        "quick": bool(args.quick),
        "data_source": args.data_source,
        "episode_counts": {"train": len(train_eps), "val": len(val_eps), "test": len(test_eps)},
        "prep": {
            "phase2_loss_last": None if phase2 is None else float(phase2.loss_history[-1]),
            "phase3_loss_last": None if phase3 is None else float(phase3.loss_history[-1]),
        },
        "phase6": {
            "all_passed": phase6.all_passed,
            "total_jooots_triggers": phase6.total_jooots_triggers,
            "stage_count": len(phase6.stage_results),
            "proxy_history_len": len(phase6.proxy_history),
            "audit_history_len": len(phase6.audit_history),
        },
        "artifacts": {
            "summary_json": str(run_dir / "summary.json"),
            "stage_metrics_csv": str(stage_metrics_path),
            "gates_json": str(gates_path),
            "final_checkpoint": str(final_ckpt),
            "stage_checkpoints_dir": str(stage_ckpt_dir),
        },
    }
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print("[ASL Phase 6 Real-Data Pipeline Complete]")
    print(f"  all_passed: {phase6.all_passed}")
    print(f"  total_jooots_triggers: {phase6.total_jooots_triggers}")
    print(f"  stage_count: {len(phase6.stage_results)}")
    failed = [s for s in phase6.stage_results if not s.passed]
    if failed:
        print("  failed_stages:")
        for stage in failed:
            failed_gates = [name for name, ok in stage.gates.items() if not ok]
            print(
                f"    - stage={stage.stage_index} sigma={stage.sigma_weight:.2f} "
                f"failed_gates={','.join(failed_gates)}"
            )
    print(f"  summary: {summary_path}")
    print(f"  stage_metrics: {stage_metrics_path}")
    print(f"  gates: {gates_path}")
    print(f"  final checkpoint: {final_ckpt}")


if __name__ == "__main__":
    main()

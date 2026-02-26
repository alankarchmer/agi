from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.random_walk_1d import RandomWalk1DEnv
from envs.tmaze import TMazeEnv
from models.rssm import GaussianRSSM
from training.contracts import EvalConfig, InfraConfig, OfflineOnlineConfig, OptimizationConfig, Phase1Config
from training.datasets import EpisodeRecord, load_real_episodes, make_sequence_dataloader, split_episodes
from training.infra import dataclass_to_dict, save_run_metadata
from training.offline_online import (
    collect_online_episodes,
    train_world_model_offline,
    train_world_model_offline_online,
)
from training.real_eval import evaluate_real_data_gates
from training.replay import EpisodeReplayBuffer
from utils.device import get_device
from utils.logging import LoggerConfig
from utils.runtime_hygiene import apply_runtime_warning_filters, runtime_health_message
from utils.seed import set_seed

apply_runtime_warning_filters()


def _infer_action_dim(episodes: list[EpisodeRecord], fallback: int | None) -> int:
    if fallback is not None:
        return int(fallback)
    if not episodes:
        raise ValueError("Cannot infer action_dim without episodes.")
    return int(episodes[0].action.shape[-1])


def _obs_stats(episodes: list[EpisodeRecord]) -> tuple[torch.Tensor, torch.Tensor]:
    obs = torch.cat([ep.obs for ep in episodes], dim=0)
    return obs.mean(dim=0), obs.std(dim=0).clamp_min(1e-3)


def _action_stats(episodes: list[EpisodeRecord]) -> tuple[torch.Tensor, torch.Tensor]:
    act = torch.cat([ep.action for ep in episodes], dim=0)
    return act.mean(dim=0), act.std(dim=0).clamp_min(1e-3)


def _online_env_factory(name: str, seed: int):
    if name == "tmaze":
        return lambda: TMazeEnv(seed=seed)
    if name == "random_walk":
        return lambda: RandomWalk1DEnv(max_steps=64, seed=seed)
    raise ValueError(f"Unknown online environment: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ASL world model on real trajectory data.")
    parser.add_argument("--data-source", type=str, required=True, help="Path to dataset file or directory.")
    parser.add_argument("--action-space-type", choices=["discrete", "continuous"], default="discrete")
    parser.add_argument("--action-dim", type=int, default=None)
    parser.add_argument("--obs-likelihood", choices=["mse", "gaussian", "bernoulli"], default="gaussian")

    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--train-split", type=float, default=0.8)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--test-split", type=float, default=0.1)

    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--sigma-dim", type=int, default=64)
    parser.add_argument("--macro-dim", type=int, default=16)
    parser.add_argument("--obs-encoder-type", choices=["mlp", "conv"], default="mlp")

    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--beta-start", type=float, default=0.1)
    parser.add_argument("--beta-end", type=float, default=1.0)
    parser.add_argument("--beta-anneal-frac", type=float, default=0.2)
    parser.add_argument("--kl-balance", type=float, default=1.0)
    parser.add_argument("--kl-free-nats", type=float, default=0.0)
    parser.add_argument("--overshooting-horizon", type=int, default=1)
    parser.add_argument("--overshooting-weight", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)

    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--early-stop-patience", type=int, default=30)
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-4)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--resume-checkpoint", type=str, default=None)

    parser.add_argument("--online-env", choices=["none", "tmaze", "random_walk"], default="none")
    parser.add_argument("--online-episodes", type=int, default=20)
    parser.add_argument("--online-steps", type=int, default=200)
    parser.add_argument("--online-fraction", type=float, default=0.2)
    parser.add_argument("--online-beta", type=float, default=None)
    parser.add_argument("--replay-capacity-steps", type=int, default=200000)

    parser.add_argument("--eval-nll-max", type=float, default=2.0)
    parser.add_argument("--eval-kl-min", type=float, default=0.01)
    parser.add_argument("--eval-kl-max", type=float, default=5.0)
    parser.add_argument("--eval-ood-drift-max", type=float, default=0.25)
    parser.add_argument("--eval-intervention-spearman-min", type=float, default=0.30)

    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--mps", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--disable-tensorboard", action="store_true")
    parser.add_argument("--show-runtime-health", action="store_true")
    parser.add_argument("--output-dir", type=str, default="runs/real_data")
    args = parser.parse_args()

    if args.show_runtime_health:
        print(runtime_health_message())
        print()

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
    tb_log_dir = run_dir / "tb"
    logger_cfg = None
    if not args.disable_tensorboard:
        logger_cfg = LoggerConfig(log_dir=str(tb_log_dir), use_wandb=False)

    print(f"[Run] device={device} run_dir={run_dir}")
    if logger_cfg is not None:
        print(f"[Run] tensorboard_logdir={tb_log_dir}")

    episodes = load_real_episodes(
        source=args.data_source,
        action_space_type=args.action_space_type,
        action_dim=int(args.action_dim) if args.action_dim is not None else 3,
    )
    train_eps, val_eps, test_eps = split_episodes(
        episodes=episodes,
        train_frac=args.train_split,
        val_frac=args.val_split,
        test_frac=args.test_split,
        seed=args.seed,
    )

    if not train_eps:
        raise ValueError("No training episodes found after split.")

    action_dim = _infer_action_dim(train_eps, args.action_dim)
    obs_dim = int(train_eps[0].obs.shape[-1])

    train_loader = make_sequence_dataloader(
        episodes=train_eps,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        shuffle=True,
        stride=args.stride,
        include_partial=True,
        drop_last=True,
    )
    val_loader = (
        make_sequence_dataloader(
            episodes=val_eps,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            shuffle=False,
            stride=args.stride,
            include_partial=True,
            drop_last=False,
        )
        if val_eps
        else None
    )
    test_loader = make_sequence_dataloader(
        episodes=test_eps if test_eps else val_eps,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        shuffle=False,
        stride=args.stride,
        include_partial=True,
        drop_last=False,
    )

    model = GaussianRSSM(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        sigma_dim=args.sigma_dim,
        macro_dim=args.macro_dim,
        obs_encoder_type=args.obs_encoder_type,
        action_space_type=args.action_space_type,
        obs_likelihood=args.obs_likelihood,
        normalize_obs=True,
        normalize_action=(args.action_space_type == "continuous"),
    )

    obs_mean, obs_std = _obs_stats(train_eps)
    action_mean, action_std = _action_stats(train_eps)
    model.set_normalization_stats(obs_mean=obs_mean, obs_std=obs_std, action_mean=action_mean, action_std=action_std)

    phase1_cfg = Phase1Config(
        steps=args.steps,
        batch_size=args.batch_size,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        beta_anneal_frac=args.beta_anneal_frac,
        kl_balance=args.kl_balance,
        kl_free_nats=args.kl_free_nats,
        overshooting_horizon=args.overshooting_horizon,
        overshooting_weight=args.overshooting_weight,
    )
    optim_cfg = OptimizationConfig(lr_l1=args.lr, weight_decay=args.weight_decay, grad_clip_norm=args.grad_clip_norm)
    infra_cfg = InfraConfig(
        use_amp=args.use_amp,
        grad_accum_steps=args.grad_accum_steps,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
        checkpoint_every=args.checkpoint_every,
    )

    offline_result = train_world_model_offline(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        phase_cfg=phase1_cfg,
        optim_cfg=optim_cfg,
        infra_cfg=infra_cfg,
        device=device,
        run_dir=run_dir,
        logger_cfg=logger_cfg,
        resume_checkpoint=Path(args.resume_checkpoint) if args.resume_checkpoint else None,
    )

    online_result = None
    if args.online_env != "none" and args.online_steps > 0:
        offline_replay = EpisodeReplayBuffer(capacity_steps=args.replay_capacity_steps, seed=args.seed)
        offline_replay.extend(train_eps)

        online_replay = EpisodeReplayBuffer(capacity_steps=args.replay_capacity_steps // 4, seed=args.seed + 1)
        online_episodes = collect_online_episodes(
            env_fn=_online_env_factory(args.online_env, args.seed + 1000),
            model=model.to(device),
            num_episodes=args.online_episodes,
            max_steps=args.seq_len,
            seed=args.seed + 1000,
        )
        online_replay.extend(online_episodes)

        online_cfg = OfflineOnlineConfig(
            online_steps=args.online_steps,
            online_fraction=args.online_fraction,
            replay_capacity_steps=args.replay_capacity_steps,
        )
        online_result = train_world_model_offline_online(
            model=model,
            offline_replay=offline_replay,
            online_replay=online_replay,
            cfg=online_cfg,
            optim_cfg=optim_cfg,
            device=device,
            beta=(args.beta_end if args.online_beta is None else args.online_beta),
            kl_balance=args.kl_balance,
            kl_free_nats=args.kl_free_nats,
            overshooting_horizon=args.overshooting_horizon,
            overshooting_weight=args.overshooting_weight,
            seq_len=args.seq_len,
        )

    eval_cfg = EvalConfig(
        nll_max=args.eval_nll_max,
        kl_min=args.eval_kl_min,
        kl_max=args.eval_kl_max,
        ood_drift_max=args.eval_ood_drift_max,
        intervention_trend_min_spearman=args.eval_intervention_spearman_min,
    )
    eval_result = evaluate_real_data_gates(
        model=model,
        dataloader=test_loader,
        cfg=eval_cfg,
        device=device,
        intervention_audit_history=None,
    )

    final_ckpt = run_dir / "final_world_model.pt"
    torch.save({"model": model.state_dict()}, final_ckpt)

    summary = {
        "seed": args.seed,
        "device": str(device),
        "data_source": args.data_source,
        "episode_counts": {"train": len(train_eps), "val": len(val_eps), "test": len(test_eps)},
        "offline_training": {
            "steps": args.steps,
            "train_loss_last": offline_result.train_loss_history[-1] if offline_result.train_loss_history else None,
            "best_val_loss": offline_result.best_val_loss,
            "best_checkpoint": offline_result.best_checkpoint,
        },
        "online_training": (
            None
            if online_result is None
            else {
                "online_steps": args.online_steps,
                "online_episodes_collected": online_result.online_episodes_collected,
                "loss_last": online_result.loss_history[-1] if online_result.loss_history else None,
            }
        ),
        "real_data_eval": eval_result.to_dict(),
        "artifacts": {"run_dir": str(run_dir), "final_checkpoint": str(final_ckpt)},
    }

    save_run_metadata(run_dir / "summary.json", summary)
    save_run_metadata(
        run_dir / "config.json",
        {
            "phase1": dataclass_to_dict(phase1_cfg),
            "optim": dataclass_to_dict(optim_cfg),
            "infra": dataclass_to_dict(infra_cfg),
            "eval": dataclass_to_dict(eval_cfg),
            "cli_args": vars(args),
        },
    )

    print("[Real Data Training Complete]")
    print(f"  run_dir: {run_dir}")
    print(f"  best_val_loss: {offline_result.best_val_loss:.6f}")
    if online_result is not None and online_result.loss_history:
        print(f"  online_loss_last: {online_result.loss_history[-1]:.6f}")
    print(f"  eval_pass_all: {eval_result.pass_all}")
    print(f"  summary: {run_dir / 'summary.json'}")


if __name__ == "__main__":
    main()

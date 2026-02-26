from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
)
from training.data_collection import collect_random_trajectories
from training.phase1_world_model import train_phase1_world_model
from training.phase2_attractor import train_phase2_attractor
from training.phase3_causal import train_phase3_causal
from training.phase4_joint import train_phase4_joint
from training.phase5_loop_closure import train_phase5_loop_closure
from utils.device import get_device
from utils.runtime_hygiene import apply_runtime_warning_filters, runtime_health_message
from utils.seed import set_seed
from envs.random_walk_1d import RandomWalk1DEnv

apply_runtime_warning_filters()


def _latest(values: list[float]) -> float:
    return float(values[-1]) if values else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ASL phases 1->5 sequentially")
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
    parser.add_argument("--batch-size", type=int, default=32)

    parser.add_argument("--quick", action="store_true", help="Run a short smoke pipeline")
    parser.add_argument("--output-dir", type=str, default="runs/phases")

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
        args.num_sequences = 128
        args.seq_len = 16
        args.batch_size = 24

    if args.cpu and (args.cuda or args.mps):
        raise ValueError("--cpu cannot be combined with --cuda or --mps")

    set_seed(args.seed)
    if args.cpu:
        device = torch.device("cpu")
    else:
        device = get_device(prefer_cuda=args.cuda, prefer_mps=(args.mps or not args.cuda))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = collect_random_trajectories(
        env_fn=lambda: RandomWalk1DEnv(max_steps=args.seq_len + 2, seed=args.seed),
        num_sequences=args.num_sequences,
        seq_len=args.seq_len,
        seed=args.seed,
    )
    obs_seq = data["obs_seq"]
    act_seq = data["act_seq"]

    world_model = GaussianRSSM(
        obs_dim=obs_seq.shape[-1],
        action_dim=act_seq.shape[-1],
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        sigma_dim=args.attractor_dim,
        macro_dim=args.macro_dim,
    )
    micro_dim = world_model.hidden_dim + world_model.latent_dim

    attractor = AttractorDynamics(
        micro_dim=micro_dim,
        attractor_dim=args.attractor_dim,
        settling_steps=10,
        tau=0.1,
    )
    nis = NISMacroState(micro_dim=args.attractor_dim, macro_dim=args.macro_dim)
    macro_transition = MacroTransition(macro_dim=args.macro_dim)
    controller = JOOTSController(patience=60)

    optim_cfg = OptimizationConfig(lr_l1=3e-4, lr_l2=1e-3, lr_l3=1e-4, grad_clip_norm=1.0)

    print(f"Running on device: {device}")

    phase1 = train_phase1_world_model(
        model=world_model,
        obs_seq=obs_seq,
        act_seq=act_seq,
        phase_cfg=Phase1Config(steps=args.phase1_steps, batch_size=args.batch_size),
        optim_cfg=optim_cfg,
        device=device,
        logger_cfg=LoggingConfig(log_dir=str(output_dir / "phase1"), use_wandb=False),
    )

    phase2 = train_phase2_attractor(
        world_model=world_model,
        attractor=attractor,
        obs_seq=obs_seq,
        act_seq=act_seq,
        phase_cfg=Phase2Config(steps=args.phase2_steps, batch_size=args.batch_size, lambda_spec=1.0),
        optim_cfg=optim_cfg,
        device=device,
        logger_cfg=LoggingConfig(log_dir=str(output_dir / "phase2"), use_wandb=False),
    )

    phase3 = train_phase3_causal(
        world_model=world_model,
        attractor=attractor,
        nis=nis,
        macro_transition=macro_transition,
        obs_seq=obs_seq,
        act_seq=act_seq,
        phase_cfg=Phase3Config(steps=args.phase3_steps, batch_size=args.batch_size, alpha_dei=0.1),
        optim_cfg=optim_cfg,
        device=device,
        logger_cfg=LoggingConfig(log_dir=str(output_dir / "phase3"), use_wandb=False),
    )

    phase4 = train_phase4_joint(
        world_model=world_model,
        attractor=attractor,
        nis=nis,
        macro_transition=macro_transition,
        controller=controller,
        obs_seq=obs_seq,
        act_seq=act_seq,
        phase_cfg=Phase4Config(
            steps=args.phase4_steps,
            batch_size=args.batch_size,
            alpha_dei=0.05,
            attr_loss_weight=0.5,
            macro_loss_weight=0.5,
        ),
        optim_cfg=optim_cfg,
        device=device,
        logger_cfg=LoggingConfig(log_dir=str(output_dir / "phase4"), use_wandb=False),
    )

    phase5 = train_phase5_loop_closure(
        world_model=world_model,
        attractor=attractor,
        nis=nis,
        macro_transition=macro_transition,
        obs_seq=obs_seq,
        act_seq=act_seq,
        phase_cfg=Phase5Config(
            steps=args.phase5_steps,
            batch_size=args.batch_size,
            beta=1.0,
            macro_align_weight=0.25,
        ),
        optim_cfg=optim_cfg,
        device=device,
        logger_cfg=LoggingConfig(log_dir=str(output_dir / "phase5"), use_wandb=False),
    )

    metrics = {
        "phase1_loss": _latest(phase1.loss_history),
        "phase1_kl": _latest(phase1.kl_history),
        "phase2_loss": _latest(phase2.loss_history),
        "phase2_rho": _latest(phase2.radius_history),
        "phase3_loss": _latest(phase3.loss_history),
        "phase3_dei": _latest(phase3.dei_history),
        "phase4_total": _latest(phase4.total_loss_history),
        "phase4_vfe": _latest(phase4.vfe_history),
        "phase5_total": _latest(phase5.total_loss_history),
        "phase5_vfe": _latest(phase5.vfe_history),
        "phase5_macro_align": _latest(phase5.macro_align_history),
        "macro_feedback_enabled": bool(world_model.enable_macro_feedback),
    }

    ckpt = {
        "world_model": world_model.state_dict(),
        "attractor": attractor.state_dict(),
        "nis": nis.state_dict(),
        "macro_transition": macro_transition.state_dict(),
        "metrics": metrics,
        "config": vars(args),
    }

    ckpt_path = output_dir / "asl_phase5_checkpoint.pt"
    torch.save(ckpt, ckpt_path)

    metrics_path = output_dir / "summary.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    print("[ASL Phase Pipeline Complete]")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Summary: {metrics_path}")


if __name__ == "__main__":
    main()

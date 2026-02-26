from __future__ import annotations

import math

from envs.random_walk_1d import RandomWalk1DEnv
from models.rssm import GaussianRSSM
from training.contracts import OptimizationConfig, Phase1Config
from training.data_collection import collect_random_trajectories
from training.phase1_world_model import train_phase1_world_model
from utils.device import get_device
from utils.seed import set_seed


def test_level1_convergence_smoke() -> None:
    set_seed(13)
    device = get_device(prefer_cuda=False)

    data = collect_random_trajectories(
        env_fn=lambda: RandomWalk1DEnv(max_steps=20, seed=13),
        num_sequences=128,
        seq_len=16,
        seed=13,
    )

    model = GaussianRSSM(
        obs_dim=data["obs_seq"].shape[-1],
        action_dim=data["act_seq"].shape[-1],
        hidden_dim=64,
        latent_dim=16,
    )

    result = train_phase1_world_model(
        model=model,
        obs_seq=data["obs_seq"],
        act_seq=data["act_seq"],
        phase_cfg=Phase1Config(steps=140, batch_size=32),
        optim_cfg=OptimizationConfig(lr_l1=3e-4, grad_clip_norm=1.0),
        device=device,
        logger_cfg=None,
    )

    assert all(math.isfinite(v) for v in result.loss_history)
    split = max(8, len(result.loss_history) // 5)
    early = sum(result.loss_history[:split]) / split
    late = sum(result.loss_history[-split:]) / split
    improvement = (early - late) / max(abs(early), 1e-6)

    assert improvement > 0.10

from __future__ import annotations

import torch

from envs.random_walk_1d import RandomWalk1DEnv
from models.attractor import AttractorDynamics
from models.rssm import GaussianRSSM
from training.contracts import OptimizationConfig, Phase1Config, Phase2Config
from training.data_collection import collect_random_trajectories
from training.phase1_world_model import train_phase1_world_model
from training.phase2_attractor import train_phase2_attractor
from utils.device import get_device
from utils.seed import set_seed


def test_phase2_updates_attractor_and_not_world_model() -> None:
    set_seed(21)
    device = get_device(prefer_cuda=False)

    data = collect_random_trajectories(
        env_fn=lambda: RandomWalk1DEnv(max_steps=18, seed=21),
        num_sequences=96,
        seq_len=14,
        seed=21,
    )

    world = GaussianRSSM(
        obs_dim=data["obs_seq"].shape[-1],
        action_dim=data["act_seq"].shape[-1],
        hidden_dim=48,
        latent_dim=12,
    )

    train_phase1_world_model(
        model=world,
        obs_seq=data["obs_seq"],
        act_seq=data["act_seq"],
        phase_cfg=Phase1Config(steps=60, batch_size=24),
        optim_cfg=OptimizationConfig(lr_l1=3e-4, grad_clip_norm=1.0),
        device=device,
        logger_cfg=None,
    )

    micro_dim = world.hidden_dim + world.latent_dim
    attractor = AttractorDynamics(micro_dim=micro_dim, attractor_dim=32)

    world_before = [p.detach().clone() for p in world.parameters()]
    attr_before = [p.detach().clone() for p in attractor.parameters()]

    result = train_phase2_attractor(
        world_model=world,
        attractor=attractor,
        obs_seq=data["obs_seq"],
        act_seq=data["act_seq"],
        phase_cfg=Phase2Config(steps=8, batch_size=24, lambda_spec=0.5),
        optim_cfg=OptimizationConfig(lr_l2=1e-3, grad_clip_norm=1.0),
        device=device,
        logger_cfg=None,
    )

    assert len(result.loss_history) == 8

    world_after = [p.detach() for p in world.parameters()]
    attr_after = [p.detach() for p in attractor.parameters()]

    assert all(torch.allclose(a, b) for a, b in zip(world_before, world_after))
    assert any(not torch.allclose(a, b) for a, b in zip(attr_before, attr_after))

from __future__ import annotations

import math

import torch

from envs.random_walk_1d import RandomWalk1DEnv
from models.attractor import AttractorDynamics
from models.nis import MacroTransition, NISMacroState
from models.rssm import GaussianRSSM
from training.contracts import OptimizationConfig, Phase1Config, Phase2Config, Phase3Config, Phase5Config
from training.data_collection import collect_random_trajectories
from training.phase1_world_model import train_phase1_world_model
from training.phase2_attractor import train_phase2_attractor
from training.phase3_causal import train_phase3_causal
from training.phase5_loop_closure import train_phase5_loop_closure
from utils.device import get_device
from utils.seed import set_seed


def test_phase5_updates_world_model_under_macro_feedback() -> None:
    set_seed(31)
    device = get_device(prefer_cuda=False)

    data = collect_random_trajectories(
        env_fn=lambda: RandomWalk1DEnv(max_steps=16, seed=31),
        num_sequences=96,
        seq_len=14,
        seed=31,
    )

    obs_seq = data["obs_seq"]
    act_seq = data["act_seq"]

    world = GaussianRSSM(
        obs_dim=obs_seq.shape[-1],
        action_dim=act_seq.shape[-1],
        hidden_dim=48,
        latent_dim=12,
        sigma_dim=24,
        macro_dim=8,
    )
    attractor = AttractorDynamics(micro_dim=world.hidden_dim + world.latent_dim, attractor_dim=24)
    nis = NISMacroState(micro_dim=24, macro_dim=8)
    macro_transition = MacroTransition(macro_dim=8)

    optim_cfg = OptimizationConfig(lr_l1=3e-4, lr_l2=1e-3, lr_l3=1e-4, grad_clip_norm=1.0)

    train_phase1_world_model(
        model=world,
        obs_seq=obs_seq,
        act_seq=act_seq,
        phase_cfg=Phase1Config(steps=40, batch_size=24),
        optim_cfg=optim_cfg,
        device=device,
        logger_cfg=None,
    )

    train_phase2_attractor(
        world_model=world,
        attractor=attractor,
        obs_seq=obs_seq,
        act_seq=act_seq,
        phase_cfg=Phase2Config(steps=10, batch_size=24, lambda_spec=0.5),
        optim_cfg=optim_cfg,
        device=device,
        logger_cfg=None,
    )

    train_phase3_causal(
        world_model=world,
        attractor=attractor,
        nis=nis,
        macro_transition=macro_transition,
        obs_seq=obs_seq,
        act_seq=act_seq,
        phase_cfg=Phase3Config(steps=10, batch_size=24, alpha_dei=0.1),
        optim_cfg=optim_cfg,
        device=device,
        logger_cfg=None,
    )

    world_before = [p.detach().clone() for p in world.parameters()]
    attractor_before = [p.detach().clone() for p in attractor.parameters()]

    result = train_phase5_loop_closure(
        world_model=world,
        attractor=attractor,
        nis=nis,
        macro_transition=macro_transition,
        obs_seq=obs_seq,
        act_seq=act_seq,
        phase_cfg=Phase5Config(steps=8, batch_size=24, beta=1.0, macro_align_weight=0.25),
        optim_cfg=optim_cfg,
        device=device,
        logger_cfg=None,
    )

    assert len(result.total_loss_history) == 8
    assert len(result.vfe_history) == 8
    assert len(result.macro_align_history) == 8

    assert all(math.isfinite(v) for v in result.total_loss_history)
    assert all(math.isfinite(v) for v in result.vfe_history)
    assert all(math.isfinite(v) for v in result.macro_align_history)

    assert world.enable_macro_feedback is True

    world_after = [p.detach() for p in world.parameters()]
    attractor_after = [p.detach() for p in attractor.parameters()]

    assert any(not torch.allclose(a, b) for a, b in zip(world_before, world_after))
    assert all(torch.allclose(a, b) for a, b in zip(attractor_before, attractor_after))

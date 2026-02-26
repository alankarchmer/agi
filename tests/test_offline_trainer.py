from __future__ import annotations

import numpy as np
import torch

from models.rssm import GaussianRSSM
from training.contracts import InfraConfig, OptimizationConfig, Phase1Config
from training.datasets import load_real_episodes, make_sequence_dataloader
from training.offline_online import train_world_model_offline


def test_offline_trainer_smoke(tmp_path) -> None:
    obs_seq = np.random.randn(12, 10, 4).astype(np.float32)
    act_seq = np.eye(3, dtype=np.float32)[np.random.randint(0, 3, size=(12, 10))]
    rew_seq = np.random.randn(12, 10).astype(np.float32)
    done_seq = np.zeros((12, 10), dtype=np.float32)
    done_seq[:, -1] = 1.0
    ts_seq = np.tile(np.arange(10, dtype=np.float32), (12, 1))
    np.savez(tmp_path / "batched.npz", obs_seq=obs_seq, act_seq=act_seq, reward_seq=rew_seq, done_seq=done_seq, timestamp_seq=ts_seq)

    episodes = load_real_episodes(tmp_path / "batched.npz", action_space_type="discrete", action_dim=3)
    train_loader = make_sequence_dataloader(episodes[:8], seq_len=10, batch_size=4, shuffle=True, drop_last=True)
    val_loader = make_sequence_dataloader(episodes[8:], seq_len=10, batch_size=2, shuffle=False, drop_last=False)

    model = GaussianRSSM(obs_dim=4, action_dim=3, hidden_dim=32, latent_dim=8)
    result = train_world_model_offline(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        phase_cfg=Phase1Config(steps=20, batch_size=4),
        optim_cfg=OptimizationConfig(lr_l1=3e-4, grad_clip_norm=1.0),
        infra_cfg=InfraConfig(use_amp=False, grad_accum_steps=1, early_stop_patience=100),
        device=torch.device("cpu"),
        run_dir=tmp_path / "run",
        logger_cfg=None,
    )

    assert len(result.train_loss_history) > 0
    assert torch.isfinite(torch.tensor(result.train_loss_history)).all()


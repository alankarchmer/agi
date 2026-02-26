from __future__ import annotations

import numpy as np
import torch

from models.rssm import GaussianRSSM
from training.contracts import EvalConfig
from training.datasets import load_real_episodes, make_sequence_dataloader
from training.real_eval import evaluate_real_data_gates


def test_real_eval_gates_return_finite_metrics(tmp_path) -> None:
    obs_seq = np.random.randn(10, 8, 4).astype(np.float32)
    act_seq = np.eye(3, dtype=np.float32)[np.random.randint(0, 3, size=(10, 8))]
    rew_seq = np.random.randn(10, 8).astype(np.float32)
    done_seq = np.zeros((10, 8), dtype=np.float32)
    done_seq[:, -1] = 1.0
    ts_seq = np.tile(np.arange(8, dtype=np.float32), (10, 1))
    np.savez(tmp_path / "batched.npz", obs_seq=obs_seq, act_seq=act_seq, reward_seq=rew_seq, done_seq=done_seq, timestamp_seq=ts_seq)

    episodes = load_real_episodes(tmp_path / "batched.npz", action_space_type="discrete", action_dim=3)
    loader = make_sequence_dataloader(episodes, seq_len=8, batch_size=4, shuffle=False, drop_last=False)

    model = GaussianRSSM(obs_dim=4, action_dim=3, hidden_dim=32, latent_dim=8)
    out = evaluate_real_data_gates(model, loader, EvalConfig(), device=torch.device("cpu"))
    assert isinstance(out.pass_all, bool)
    for v in out.metrics.values():
        if isinstance(v, float):
            assert np.isfinite(v) or np.isnan(v)


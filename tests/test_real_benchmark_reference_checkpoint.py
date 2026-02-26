from __future__ import annotations

import numpy as np
import torch

from benchmarks.real_data_harness import RealBenchmarkConfig, run_real_data_benchmarks
from models.rssm import GaussianRSSM


def _write_episode(path, obs_dim: int, action_dim: int, steps: int = 10) -> None:
    rng = np.random.default_rng(123)
    obs = rng.normal(size=(steps, obs_dim)).astype(np.float32)
    action = np.tanh(rng.normal(size=(steps, action_dim))).astype(np.float32)
    reward = np.zeros((steps,), dtype=np.float32)
    done = np.zeros((steps,), dtype=np.float32)
    done[-1] = 1.0
    timestamp = np.arange(steps, dtype=np.float32)
    np.savez_compressed(path, obs=obs, action=action, reward=reward, done=done, timestamp=timestamp)


def test_real_benchmark_uses_reference_checkpoint_without_retraining(tmp_path, monkeypatch) -> None:
    obs_dim = 5
    action_dim = 2

    data_dir = tmp_path / "episodes"
    data_dir.mkdir(parents=True, exist_ok=True)
    for i in range(8):
        _write_episode(data_dir / f"episode_{i:03d}.npz", obs_dim=obs_dim, action_dim=action_dim)

    model = GaussianRSSM(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=128,
        latent_dim=32,
        action_space_type="continuous",
        obs_likelihood="gaussian",
        normalize_obs=True,
        normalize_action=True,
    )
    ckpt_path = tmp_path / "reference.pt"
    torch.save({"model": model.state_dict()}, ckpt_path)

    def _should_not_train(*args, **kwargs):
        raise AssertionError("train_world_model_offline should not be called with a reference checkpoint")

    monkeypatch.setattr("benchmarks.real_data_harness.train_world_model_offline", _should_not_train)

    summary = run_real_data_benchmarks(
        RealBenchmarkConfig(
            data_source=str(data_dir),
            profile="pusht",
            action_space_type="continuous",
            action_dim=action_dim,
            obs_likelihood="gaussian",
            seeds=[7],
            device="cpu",
            seq_len=8,
            batch_size=4,
            train_steps=5,
            reference_checkpoint=str(ckpt_path),
            checkpoint_mode="strict",
        )
    )

    assert summary.reference_checkpoint == str(ckpt_path)
    assert "nll_mean" in summary.metrics

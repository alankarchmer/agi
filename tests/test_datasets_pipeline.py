from __future__ import annotations

import numpy as np

from training.datasets import load_real_episodes, make_sequence_dataloader


def test_load_real_episodes_from_npz_dir_and_masked_loader(tmp_path) -> None:
    ep0 = {
        "obs": np.random.randn(6, 4).astype(np.float32),
        "action": np.eye(3, dtype=np.float32)[np.array([0, 1, 2, 1, 0, 2])],
        "reward": np.random.randn(6).astype(np.float32),
        "done": np.array([0, 0, 0, 0, 0, 1], dtype=np.float32),
        "timestamp": np.arange(6, dtype=np.float32),
    }
    ep1 = {
        "obs": np.random.randn(3, 4).astype(np.float32),
        "action": np.eye(3, dtype=np.float32)[np.array([1, 2, 0])],
        "reward": np.random.randn(3).astype(np.float32),
        "done": np.array([0, 0, 1], dtype=np.float32),
        "timestamp": np.arange(3, dtype=np.float32),
    }

    np.savez(tmp_path / "ep0.npz", **ep0)
    np.savez(tmp_path / "ep1.npz", **ep1)

    episodes = load_real_episodes(tmp_path, action_space_type="discrete", action_dim=3)
    assert len(episodes) == 2
    assert episodes[0].obs.shape[-1] == 4
    assert episodes[0].action.shape[-1] == 3

    loader = make_sequence_dataloader(
        episodes=episodes,
        seq_len=5,
        batch_size=2,
        stride=1,
        include_partial=True,
        drop_last=False,
        shuffle=False,
    )
    batch = next(iter(loader))
    assert batch["obs_seq"].shape[-1] == 4
    assert batch["act_seq"].shape[-1] == 3
    assert batch["valid_mask"].shape[1] == 5
    assert (batch["valid_mask"].sum(dim=-1) <= 5).all()


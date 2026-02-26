from __future__ import annotations

import torch

from training.datasets import EpisodeRecord
from training.replay import EpisodeReplayBuffer, OfflineOnlineReplay


def _episode(length: int, obs_dim: int = 4, act_dim: int = 3, eid: str = "ep") -> EpisodeRecord:
    return EpisodeRecord(
        obs=torch.randn(length, obs_dim),
        action=torch.randn(length, act_dim),
        reward=torch.randn(length),
        done=torch.zeros(length),
        timestamp=torch.arange(length, dtype=torch.float32),
        episode_id=eid,
    )


def test_replay_buffer_sampling_shapes() -> None:
    rb = EpisodeReplayBuffer(capacity_steps=100, seed=0)
    rb.add_episode(_episode(8, eid="a"))
    rb.add_episode(_episode(5, eid="b"))

    batch = rb.sample_sequences(batch_size=4, seq_len=6)
    assert batch.obs_seq.shape == (4, 6, 4)
    assert batch.act_seq.shape == (4, 6, 3)
    assert batch.valid_mask.shape == (4, 6)


def test_offline_online_mixer_samples_both_sources() -> None:
    off = EpisodeReplayBuffer(capacity_steps=100, seed=1)
    on = EpisodeReplayBuffer(capacity_steps=100, seed=2)
    off.add_episode(_episode(10, eid="off"))
    on.add_episode(_episode(10, eid="on"))

    mix = OfflineOnlineReplay(off, on, online_fraction=0.5)
    batch = mix.sample(batch_size=6, seq_len=4)
    assert batch.obs_seq.shape == (6, 4, 4)
    assert batch.valid_mask.sum().item() > 0


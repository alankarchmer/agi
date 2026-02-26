from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import torch

from training.datasets import EpisodeRecord


@dataclass
class ReplayBatch:
    obs_seq: torch.Tensor
    act_seq: torch.Tensor
    reward_seq: torch.Tensor
    done_seq: torch.Tensor
    valid_mask: torch.Tensor
    timestamp_seq: torch.Tensor


class EpisodeReplayBuffer:
    def __init__(self, capacity_steps: int = 200_000, seed: int = 0) -> None:
        self.capacity_steps = int(capacity_steps)
        self._episodes: deque[EpisodeRecord] = deque()
        self._total_steps = 0
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self._episodes)

    @property
    def total_steps(self) -> int:
        return self._total_steps

    def add_episode(self, episode: EpisodeRecord) -> None:
        self._episodes.append(episode)
        self._total_steps += episode.length
        self._evict_if_needed()

    def extend(self, episodes: list[EpisodeRecord]) -> None:
        for ep in episodes:
            self.add_episode(ep)

    def _evict_if_needed(self) -> None:
        while self._total_steps > self.capacity_steps and self._episodes:
            ep = self._episodes.popleft()
            self._total_steps -= ep.length

    def _sample_window(self, seq_len: int) -> tuple[EpisodeRecord, int, int]:
        if not self._episodes:
            raise ValueError("Replay buffer is empty.")
        ep_idx = int(self._rng.integers(0, len(self._episodes)))
        ep = self._episodes[ep_idx]
        if ep.length <= seq_len:
            return ep, 0, ep.length
        start = int(self._rng.integers(0, ep.length - seq_len + 1))
        return ep, start, seq_len

    def sample_sequences(self, batch_size: int, seq_len: int) -> ReplayBatch:
        if batch_size <= 0 or seq_len <= 0:
            raise ValueError("batch_size and seq_len must be > 0")

        sample_ep = self._episodes[0]
        obs_dim = int(sample_ep.obs.shape[-1])
        act_dim = int(sample_ep.action.shape[-1])

        obs = torch.zeros(batch_size, seq_len, obs_dim, dtype=torch.float32)
        act = torch.zeros(batch_size, seq_len, act_dim, dtype=torch.float32)
        rew = torch.zeros(batch_size, seq_len, dtype=torch.float32)
        done = torch.ones(batch_size, seq_len, dtype=torch.float32)
        ts = torch.zeros(batch_size, seq_len, dtype=torch.float32)
        mask = torch.zeros(batch_size, seq_len, dtype=torch.float32)

        for i in range(batch_size):
            ep, start, length = self._sample_window(seq_len)
            end = start + length
            obs[i, :length] = ep.obs[start:end]
            act[i, :length] = ep.action[start:end]
            rew[i, :length] = ep.reward[start:end]
            done[i, :length] = ep.done[start:end]
            ts[i, :length] = ep.timestamp[start:end]
            mask[i, :length] = 1.0

        return ReplayBatch(
            obs_seq=obs,
            act_seq=act,
            reward_seq=rew,
            done_seq=done,
            valid_mask=mask,
            timestamp_seq=ts,
        )


class OfflineOnlineReplay:
    def __init__(
        self,
        offline: EpisodeReplayBuffer,
        online: EpisodeReplayBuffer,
        online_fraction: float = 0.2,
    ) -> None:
        self.offline = offline
        self.online = online
        self.online_fraction = float(np.clip(online_fraction, 0.0, 1.0))

    def sample(self, batch_size: int, seq_len: int) -> ReplayBatch:
        n_online = int(round(batch_size * self.online_fraction))
        n_offline = batch_size - n_online

        if len(self.online) == 0:
            n_online = 0
            n_offline = batch_size
        if len(self.offline) == 0:
            n_offline = 0
            n_online = batch_size
        if n_offline == 0 and n_online == 0:
            raise ValueError("Both offline and online replay buffers are empty.")

        chunks: list[ReplayBatch] = []
        if n_offline > 0:
            chunks.append(self.offline.sample_sequences(batch_size=n_offline, seq_len=seq_len))
        if n_online > 0:
            chunks.append(self.online.sample_sequences(batch_size=n_online, seq_len=seq_len))

        obs = torch.cat([c.obs_seq for c in chunks], dim=0)
        act = torch.cat([c.act_seq for c in chunks], dim=0)
        rew = torch.cat([c.reward_seq for c in chunks], dim=0)
        done = torch.cat([c.done_seq for c in chunks], dim=0)
        mask = torch.cat([c.valid_mask for c in chunks], dim=0)
        ts = torch.cat([c.timestamp_seq for c in chunks], dim=0)

        return ReplayBatch(
            obs_seq=obs,
            act_seq=act,
            reward_seq=rew,
            done_seq=done,
            valid_mask=mask,
            timestamp_seq=ts,
        )


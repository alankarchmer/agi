from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


@dataclass
class EpisodeRecord:
    obs: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    done: torch.Tensor
    timestamp: torch.Tensor
    episode_id: str

    @property
    def length(self) -> int:
        return int(self.obs.shape[0])


def _to_float_tensor(x: np.ndarray | torch.Tensor) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.float()
    return torch.from_numpy(np.asarray(x, dtype=np.float32))


def _to_long_tensor(x: np.ndarray | torch.Tensor) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.long()
    return torch.from_numpy(np.asarray(x, dtype=np.int64))


def _as_2d_obs(obs: torch.Tensor) -> torch.Tensor:
    if obs.dim() == 1:
        return obs.unsqueeze(-1)
    return obs


def ensure_action_representation(
    action: torch.Tensor,
    action_space_type: str,
    action_dim: int,
) -> torch.Tensor:
    if action_space_type not in {"discrete", "continuous"}:
        raise ValueError(f"Unknown action_space_type: {action_space_type}")

    if action_space_type == "discrete":
        if action.dim() == 1:
            idx = action.long().clamp_min(0)
            return torch.nn.functional.one_hot(idx, num_classes=action_dim).float()
        if action.dim() == 2 and action.shape[-1] == action_dim:
            return action.float()
        raise ValueError("Discrete actions must be shape (T,) indices or (T, action_dim) one-hot.")

    # continuous
    if action.dim() == 1:
        return action.float().unsqueeze(-1)
    if action.dim() == 2:
        return action.float()
    raise ValueError("Continuous actions must be shape (T,) or (T, action_dim).")


def _episode_from_arrays(
    obs: np.ndarray | torch.Tensor,
    action: np.ndarray | torch.Tensor,
    reward: np.ndarray | torch.Tensor | None,
    done: np.ndarray | torch.Tensor | None,
    timestamp: np.ndarray | torch.Tensor | None,
    episode_id: str,
    action_space_type: str,
    action_dim: int,
) -> EpisodeRecord:
    obs_t = _as_2d_obs(_to_float_tensor(obs))
    t = int(obs_t.shape[0])

    action_t = ensure_action_representation(_to_float_tensor(action), action_space_type=action_space_type, action_dim=action_dim)
    if action_t.shape[0] != t:
        raise ValueError(f"Episode {episode_id}: action length {action_t.shape[0]} does not match obs length {t}")

    if reward is None:
        reward_t = torch.zeros(t, dtype=torch.float32)
    else:
        reward_t = _to_float_tensor(reward).reshape(t)

    if done is None:
        done_t = torch.zeros(t, dtype=torch.float32)
        done_t[-1] = 1.0
    else:
        done_t = _to_float_tensor(done).reshape(t)

    if timestamp is None:
        ts_t = torch.arange(t, dtype=torch.float32)
    else:
        ts_t = _to_float_tensor(timestamp).reshape(t)

    return EpisodeRecord(
        obs=obs_t,
        action=action_t,
        reward=reward_t,
        done=done_t,
        timestamp=ts_t,
        episode_id=str(episode_id),
    )


def _load_npz_episode(path: Path, action_space_type: str, action_dim: int, episode_id: str | None = None) -> EpisodeRecord:
    data = np.load(path, allow_pickle=True)
    keys = set(data.files)
    obs_key = "obs" if "obs" in keys else "observation"
    action_key = "action" if "action" in keys else "act"
    reward_key = "reward" if "reward" in keys else ("rew" if "rew" in keys else None)
    done_key = "done" if "done" in keys else None
    time_key = "timestamp" if "timestamp" in keys else ("time" if "time" in keys else None)

    if obs_key not in keys or action_key not in keys:
        raise ValueError(f"{path}: expected keys including obs/action, found {sorted(keys)}")

    return _episode_from_arrays(
        obs=data[obs_key],
        action=data[action_key],
        reward=None if reward_key is None else data[reward_key],
        done=None if done_key is None else data[done_key],
        timestamp=None if time_key is None else data[time_key],
        episode_id=episode_id or path.stem,
        action_space_type=action_space_type,
        action_dim=action_dim,
    )


def _load_npz_batched(path: Path, action_space_type: str, action_dim: int) -> list[EpisodeRecord]:
    data = np.load(path, allow_pickle=True)
    keys = set(data.files)
    obs_key = "obs_seq" if "obs_seq" in keys else "obs"
    act_key = "act_seq" if "act_seq" in keys else "action"
    rew_key = "reward_seq" if "reward_seq" in keys else None
    done_key = "done_seq" if "done_seq" in keys else None
    time_key = "timestamp_seq" if "timestamp_seq" in keys else None

    obs = np.asarray(data[obs_key])
    act = np.asarray(data[act_key])
    if obs.ndim < 2 or act.ndim < 2:
        raise ValueError(f"{path}: batched arrays must include batch dimension.")
    batch = obs.shape[0]

    episodes: list[EpisodeRecord] = []
    for i in range(batch):
        episodes.append(
            _episode_from_arrays(
                obs=obs[i],
                action=act[i],
                reward=None if rew_key is None else np.asarray(data[rew_key])[i],
                done=None if done_key is None else np.asarray(data[done_key])[i],
                timestamp=None if time_key is None else np.asarray(data[time_key])[i],
                episode_id=f"{path.stem}_{i}",
                action_space_type=action_space_type,
                action_dim=action_dim,
            )
        )
    return episodes


def _load_jsonl(path: Path, action_space_type: str, action_dim: int) -> list[EpisodeRecord]:
    episodes: list[EpisodeRecord] = []
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            episodes.append(
                _episode_from_arrays(
                    obs=np.asarray(record["obs"], dtype=np.float32),
                    action=np.asarray(record["action"], dtype=np.float32),
                    reward=np.asarray(record.get("reward", []), dtype=np.float32) if "reward" in record else None,
                    done=np.asarray(record.get("done", []), dtype=np.float32) if "done" in record else None,
                    timestamp=np.asarray(record.get("timestamp", []), dtype=np.float32) if "timestamp" in record else None,
                    episode_id=str(record.get("episode_id", f"{path.stem}_{idx}")),
                    action_space_type=action_space_type,
                    action_dim=action_dim,
                )
            )
    return episodes


def _load_csv_rows(path: Path, action_space_type: str, action_dim: int) -> list[EpisodeRecord]:
    rows_by_ep: dict[str, list[dict[str, str]]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ep = row.get("episode_id", "0")
            rows_by_ep.setdefault(ep, []).append(row)

    episodes: list[EpisodeRecord] = []
    for ep, rows in rows_by_ep.items():
        rows_sorted = sorted(rows, key=lambda r: float(r.get("timestamp", r.get("t", "0"))))
        obs_cols = sorted([c for c in rows_sorted[0].keys() if c.startswith("obs_")], key=lambda c: int(c.split("_")[1]))
        act_cols = sorted([c for c in rows_sorted[0].keys() if c.startswith("act_")], key=lambda c: int(c.split("_")[1]))

        if not obs_cols:
            raise ValueError(f"{path}: expected obs_* columns in CSV format.")
        if not act_cols and action_space_type == "continuous":
            raise ValueError(f"{path}: expected act_* columns for continuous actions.")

        obs = np.asarray([[float(r[c]) for c in obs_cols] for r in rows_sorted], dtype=np.float32)
        if act_cols:
            action = np.asarray([[float(r[c]) for c in act_cols] for r in rows_sorted], dtype=np.float32)
        else:
            action = np.asarray([int(float(r["action"])) for r in rows_sorted], dtype=np.int64)
        reward = np.asarray([float(r.get("reward", "0")) for r in rows_sorted], dtype=np.float32)
        done = np.asarray([float(r.get("done", "0")) for r in rows_sorted], dtype=np.float32)
        ts = np.asarray([float(r.get("timestamp", r.get("t", str(i)))) for i, r in enumerate(rows_sorted)], dtype=np.float32)

        episodes.append(
            _episode_from_arrays(
                obs=obs,
                action=action,
                reward=reward,
                done=done,
                timestamp=ts,
                episode_id=str(ep),
                action_space_type=action_space_type,
                action_dim=action_dim,
            )
        )
    return episodes


def load_real_episodes(
    source: str | Path,
    action_space_type: str = "discrete",
    action_dim: int = 3,
) -> list[EpisodeRecord]:
    src = Path(source)
    if not src.exists():
        raise FileNotFoundError(src)

    episodes: list[EpisodeRecord] = []
    if src.is_dir():
        for file in sorted(src.glob("*.npz")):
            episodes.append(_load_npz_episode(file, action_space_type=action_space_type, action_dim=action_dim))
        return episodes

    suffix = src.suffix.lower()
    if suffix == ".npz":
        data = np.load(src, allow_pickle=True)
        if "obs_seq" in data.files or ("obs" in data.files and np.asarray(data["obs"]).ndim >= 3):
            episodes.extend(_load_npz_batched(src, action_space_type=action_space_type, action_dim=action_dim))
        else:
            episodes.append(_load_npz_episode(src, action_space_type=action_space_type, action_dim=action_dim))
    elif suffix == ".jsonl":
        episodes.extend(_load_jsonl(src, action_space_type=action_space_type, action_dim=action_dim))
    elif suffix == ".csv":
        episodes.extend(_load_csv_rows(src, action_space_type=action_space_type, action_dim=action_dim))
    elif suffix == ".pt":
        payload = torch.load(src, map_location="cpu")
        if isinstance(payload, list):
            for i, item in enumerate(payload):
                episodes.append(
                    _episode_from_arrays(
                        obs=item["obs"],
                        action=item["action"],
                        reward=item.get("reward"),
                        done=item.get("done"),
                        timestamp=item.get("timestamp"),
                        episode_id=str(item.get("episode_id", i)),
                        action_space_type=action_space_type,
                        action_dim=action_dim,
                    )
                )
        elif isinstance(payload, dict):
            obs = payload["obs_seq"] if "obs_seq" in payload else payload["obs"]
            act = payload["act_seq"] if "act_seq" in payload else payload["action"]
            rew = payload.get("reward_seq", payload.get("reward"))
            done = payload.get("done_seq", payload.get("done"))
            ts = payload.get("timestamp_seq", payload.get("timestamp"))

            obs_t = _to_float_tensor(obs)
            act_t = _to_float_tensor(act)
            if obs_t.dim() < 3 or act_t.dim() < 2:
                raise ValueError(f"{src}: dict payload must provide batched sequences.")

            for i in range(obs_t.shape[0]):
                episodes.append(
                    _episode_from_arrays(
                        obs=obs_t[i],
                        action=act_t[i],
                        reward=None if rew is None else _to_float_tensor(rew)[i],
                        done=None if done is None else _to_float_tensor(done)[i],
                        timestamp=None if ts is None else _to_float_tensor(ts)[i],
                        episode_id=f"{src.stem}_{i}",
                        action_space_type=action_space_type,
                        action_dim=action_dim,
                    )
                )
        else:
            raise ValueError(f"Unsupported .pt payload type: {type(payload)!r}")
    else:
        raise ValueError(f"Unsupported dataset source: {src}")

    return episodes


def split_episodes(
    episodes: list[EpisodeRecord],
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 0,
) -> tuple[list[EpisodeRecord], list[EpisodeRecord], list[EpisodeRecord]]:
    if not episodes:
        return [], [], []
    if abs((train_frac + val_frac + test_frac) - 1.0) > 1e-6:
        raise ValueError("train_frac + val_frac + test_frac must sum to 1.0")

    rng = np.random.default_rng(seed)
    indices = np.arange(len(episodes))
    rng.shuffle(indices)

    n = len(episodes)
    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))
    n_train = min(max(1, n_train), n)
    n_val = min(max(0, n_val), n - n_train)
    n_test = n - n_train - n_val

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val : n_train + n_val + n_test]

    train = [episodes[int(i)] for i in train_idx]
    val = [episodes[int(i)] for i in val_idx]
    test = [episodes[int(i)] for i in test_idx]
    return train, val, test


class MaskedSequenceDataset(Dataset):
    def __init__(
        self,
        episodes: list[EpisodeRecord],
        seq_len: int,
        stride: int = 1,
        include_partial: bool = True,
    ) -> None:
        if seq_len <= 0:
            raise ValueError("seq_len must be > 0")
        if stride <= 0:
            raise ValueError("stride must be > 0")

        self.episodes = episodes
        self.seq_len = int(seq_len)
        self.stride = int(stride)
        self.include_partial = bool(include_partial)

        self.index: list[tuple[int, int, int]] = []
        for ep_i, ep in enumerate(self.episodes):
            t = ep.length
            if t <= 0:
                continue
            if t <= self.seq_len:
                self.index.append((ep_i, 0, t))
                continue

            starts = list(range(0, max(1, t - self.seq_len + 1), self.stride))
            for s in starts:
                self.index.append((ep_i, s, min(self.seq_len, t - s)))
            if self.include_partial:
                tail_start = max(0, t - self.seq_len // 2)
                tail_len = t - tail_start
                if tail_len > 0 and (ep_i, tail_start, tail_len) not in self.index:
                    self.index.append((ep_i, tail_start, tail_len))

        if not self.index:
            raise ValueError("No windows available for the provided episodes.")

        sample = self.episodes[self.index[0][0]]
        self.obs_dim = int(sample.obs.shape[-1])
        self.action_dim = int(sample.action.shape[-1])

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        ep_idx, start, length = self.index[idx]
        ep = self.episodes[ep_idx]
        end = start + length

        obs = torch.zeros(self.seq_len, ep.obs.shape[-1], dtype=torch.float32)
        act = torch.zeros(self.seq_len, ep.action.shape[-1], dtype=torch.float32)
        rew = torch.zeros(self.seq_len, dtype=torch.float32)
        done = torch.ones(self.seq_len, dtype=torch.float32)
        ts = torch.zeros(self.seq_len, dtype=torch.float32)
        mask = torch.zeros(self.seq_len, dtype=torch.float32)

        obs[:length] = ep.obs[start:end]
        act[:length] = ep.action[start:end]
        rew[:length] = ep.reward[start:end]
        done[:length] = ep.done[start:end]
        ts[:length] = ep.timestamp[start:end]
        mask[:length] = 1.0

        return {
            "obs_seq": obs,
            "act_seq": act,
            "reward_seq": rew,
            "done_seq": done,
            "valid_mask": mask,
            "timestamp_seq": ts,
            "episode_id": ep.episode_id,
        }


def masked_sequence_collate(batch: list[dict[str, torch.Tensor | str]]) -> dict[str, torch.Tensor | list[str]]:
    return {
        "obs_seq": torch.stack([b["obs_seq"] for b in batch], dim=0),
        "act_seq": torch.stack([b["act_seq"] for b in batch], dim=0),
        "reward_seq": torch.stack([b["reward_seq"] for b in batch], dim=0),
        "done_seq": torch.stack([b["done_seq"] for b in batch], dim=0),
        "valid_mask": torch.stack([b["valid_mask"] for b in batch], dim=0),
        "timestamp_seq": torch.stack([b["timestamp_seq"] for b in batch], dim=0),
        "episode_id": [str(b["episode_id"]) for b in batch],
    }


def make_sequence_dataloader(
    episodes: list[EpisodeRecord],
    seq_len: int,
    batch_size: int,
    shuffle: bool = True,
    stride: int = 1,
    include_partial: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    dataset = MaskedSequenceDataset(
        episodes=episodes,
        seq_len=seq_len,
        stride=stride,
        include_partial=include_partial,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        collate_fn=masked_sequence_collate,
    )


def stack_full_sequences(episodes: Iterable[EpisodeRecord]) -> dict[str, torch.Tensor]:
    eps = list(episodes)
    if not eps:
        raise ValueError("No episodes provided.")
    lengths = [e.length for e in eps]
    if len(set(lengths)) != 1:
        raise ValueError("stack_full_sequences requires equal episode lengths.")

    return {
        "obs_seq": torch.stack([e.obs for e in eps], dim=0),
        "act_seq": torch.stack([e.action for e in eps], dim=0),
        "reward_seq": torch.stack([e.reward for e in eps], dim=0),
        "done_seq": torch.stack([e.done for e in eps], dim=0),
        "timestamp_seq": torch.stack([e.timestamp for e in eps], dim=0),
        "valid_mask": torch.ones(len(eps), lengths[0], dtype=torch.float32),
    }

from __future__ import annotations

from typing import Callable

import numpy as np
import torch


def one_hot_action(action_idx: int, action_dim: int) -> np.ndarray:
    out = np.zeros(action_dim, dtype=np.float32)
    out[action_idx] = 1.0
    return out


def _is_discrete_action_space(action_space: object) -> bool:
    return hasattr(action_space, "n")


def _sample_random_action(action_space: object, rng: np.random.Generator) -> tuple[np.ndarray, object]:
    if _is_discrete_action_space(action_space):
        action_idx = int(rng.integers(0, int(action_space.n)))
        return one_hot_action(action_idx, int(action_space.n)), action_idx

    if hasattr(action_space, "shape"):
        shape = tuple(int(s) for s in action_space.shape)
        low = np.asarray(action_space.low, dtype=np.float32).reshape(shape)
        high = np.asarray(action_space.high, dtype=np.float32).reshape(shape)
        action = rng.uniform(low=low, high=high).astype(np.float32)
        return action.reshape(-1), action

    raise ValueError("Unsupported action_space type for random sampling.")


def collect_random_trajectories(
    env_fn: Callable[[], object],
    num_sequences: int,
    seq_len: int,
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    rng = np.random.default_rng(seed)

    env = env_fn()
    obs0, _ = env.reset(seed=seed)
    obs_dim = int(np.asarray(obs0).reshape(-1).shape[0])
    if _is_discrete_action_space(env.action_space):
        action_dim = int(env.action_space.n)
    else:
        action_dim = int(np.prod(np.asarray(env.action_space.shape, dtype=np.int64)))

    obs_arr = np.zeros((num_sequences, seq_len, obs_dim), dtype=np.float32)
    act_arr = np.zeros((num_sequences, seq_len, action_dim), dtype=np.float32)
    rew_arr = np.zeros((num_sequences, seq_len), dtype=np.float32)
    done_arr = np.zeros((num_sequences, seq_len), dtype=np.float32)

    for i in range(num_sequences):
        obs, _ = env.reset(seed=seed + i)
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)

        for t in range(seq_len):
            obs_arr[i, t] = obs
            action, action_env = _sample_random_action(env.action_space, rng)
            act_arr[i, t] = action

            next_obs, reward, terminated, truncated, _ = env.step(action_env)
            rew_arr[i, t] = float(reward)
            done_arr[i, t] = float(terminated or truncated)

            if terminated or truncated:
                next_obs, _ = env.reset()
            obs = np.asarray(next_obs, dtype=np.float32).reshape(-1)

    return {
        "obs_seq": torch.from_numpy(obs_arr),
        "act_seq": torch.from_numpy(act_arr),
        "reward_seq": torch.from_numpy(rew_arr),
        "done_seq": torch.from_numpy(done_arr),
    }

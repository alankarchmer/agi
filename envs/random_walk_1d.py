from __future__ import annotations

import numpy as np
from utils.runtime_hygiene import import_gymnasium_clean

gym, spaces = import_gymnasium_clean()


class RandomWalk1DEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, bound: int = 10, max_steps: int = 64, noise_std: float = 0.05, seed: int | None = None):
        super().__init__()
        self.bound = bound
        self.max_steps = max_steps
        self.noise_std = noise_std
        self.rng = np.random.default_rng(seed)

        self.action_space = spaces.Discrete(3)  # left, stay, right
        self.observation_space = spaces.Box(low=-1.5, high=1.5, shape=(1,), dtype=np.float32)

        self.position = 0
        self.goal = 0
        self.step_count = 0

    def _obs(self) -> np.ndarray:
        noisy = self.position / max(1, self.bound) + self.rng.normal(0.0, self.noise_std)
        return np.array([noisy], dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.position = int(self.rng.integers(-self.bound // 2, self.bound // 2 + 1))
        self.goal = int(self.rng.integers(-self.bound, self.bound + 1))
        self.step_count = 0
        return self._obs(), {"goal": self.goal, "position": self.position}

    def step(self, action: int):
        self.step_count += 1
        delta = int(action) - 1  # 0->-1, 1->0, 2->+1
        self.position = int(np.clip(self.position + delta, -self.bound, self.bound))

        distance = abs(self.goal - self.position)
        reward = 1.0 if distance == 0 else -distance / max(1, self.bound)
        terminated = distance == 0
        truncated = self.step_count >= self.max_steps

        obs = self._obs()
        info = {"goal": self.goal, "position": self.position, "distance": distance}
        return obs, float(reward), terminated, truncated, info

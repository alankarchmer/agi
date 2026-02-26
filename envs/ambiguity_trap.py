from __future__ import annotations

import numpy as np
from utils.runtime_hygiene import import_gymnasium_clean

gym, spaces = import_gymnasium_clean()


class AmbiguityTrapEnv(gym.Env):
    """Local-minimum environment with a sticky low-reward loop and a hidden escape action."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        max_steps: int = 20,
        loop_reward: float = 0.05,
        explore_penalty: float = -0.15,
        escape_reward: float = 1.0,
        escape_probability: float = 0.6,
        seed: int | None = None,
    ):
        super().__init__()
        self.max_steps = max_steps
        self.loop_reward = loop_reward
        self.explore_penalty = explore_penalty
        self.escape_reward = escape_reward
        self.escape_probability = escape_probability
        self.rng = np.random.default_rng(seed)

        self.action_space = spaces.Discrete(2)  # 0=loop exploit, 1=explore escape
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32)

        self.step_count = 0
        self.discovered_escape = False
        self.explores = 0

    def _obs(self) -> np.ndarray:
        step_frac = min(1.0, self.step_count / max(1, self.max_steps))
        explore_frac = min(1.0, self.explores / max(1, self.max_steps))
        discovered = 1.0 if self.discovered_escape else 0.0
        return np.array([step_frac, explore_frac, discovered], dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.step_count = 0
        self.discovered_escape = False
        self.explores = 0
        return self._obs(), {"escaped": False, "discovered_escape": False}

    def step(self, action: int):
        self.step_count += 1
        action = int(action)

        escaped = False
        if action == 0:
            reward = self.loop_reward
        else:
            self.explores += 1
            if self.rng.uniform() < self.escape_probability:
                self.discovered_escape = True
                escaped = True
                reward = self.escape_reward
            else:
                reward = self.explore_penalty

        terminated = escaped
        truncated = self.step_count >= self.max_steps

        obs = self._obs()
        info = {
            "escaped": escaped,
            "discovered_escape": self.discovered_escape,
            "explores": self.explores,
        }
        return obs, float(reward), terminated, truncated, info

from __future__ import annotations

import numpy as np
from utils.runtime_hygiene import import_gymnasium_clean

gym, spaces = import_gymnasium_clean()


class TMazeEnv(gym.Env):
    """A small T-maze POMDP with an informative cue on the corridor."""

    metadata = {"render_modes": []}

    def __init__(self, corridor_len: int = 2, max_steps: int = 20, seed: int | None = None):
        super().__init__()
        self.corridor_len = corridor_len
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)

        self.action_space = spaces.Discrete(3)  # forward, left, right
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)

        self.x = 0
        self.y = 0
        self.hidden_goal = 1
        self.cue_y = 1
        self.visited_cue = False
        self.step_count = 0

    def _at_junction(self) -> bool:
        return self.y == self.corridor_len and self.x == 0

    def _at_cue(self) -> bool:
        return self.y == self.cue_y and self.x == 0

    def _obs(self) -> np.ndarray:
        x_norm = float(self.x)
        y_norm = float(self.y / max(1, self.corridor_len))
        cue_on = 1.0 if self._at_cue() else 0.0
        cue_val = float(self.hidden_goal) if self._at_cue() else 0.0
        at_junction = 1.0 if self._at_junction() else 0.0
        return np.array([x_norm, y_norm, cue_on, cue_val, at_junction], dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.x = 0
        self.y = 0
        self.hidden_goal = int(self.rng.choice([-1, 1]))
        self.visited_cue = False
        self.step_count = 0
        return self._obs(), {"hidden_goal": self.hidden_goal, "visited_cue": self.visited_cue}

    def step(self, action: int):
        self.step_count += 1
        terminated = False
        reward = -0.01

        if self._at_junction():
            if action == 1:
                self.x = -1
                terminated = True
                reward = 1.0 if self.hidden_goal == -1 else 0.0
            elif action == 2:
                self.x = 1
                terminated = True
                reward = 1.0 if self.hidden_goal == 1 else 0.0
        else:
            if action == 0:
                self.y = min(self.corridor_len, self.y + 1)

        if self._at_cue():
            self.visited_cue = True

        truncated = self.step_count >= self.max_steps
        obs = self._obs()
        info = {
            "hidden_goal": self.hidden_goal,
            "visited_cue": self.visited_cue,
            "position": (self.x, self.y),
        }
        return obs, float(reward), terminated, truncated, info

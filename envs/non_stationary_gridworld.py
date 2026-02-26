from __future__ import annotations

import numpy as np
from utils.runtime_hygiene import import_gymnasium_clean

gym, spaces = import_gymnasium_clean()


class NonStationaryGridworldEnv(gym.Env):
    """Gridworld where the hidden goal shifts periodically."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        grid_size: int = 7,
        max_steps: int = 40,
        shift_interval: int = 25,
        seed: int | None = None,
        fixed_goal_index: int | None = None,
    ):
        super().__init__()
        if grid_size < 4:
            raise ValueError("grid_size must be >= 4")

        self.grid_size = grid_size
        self.max_steps = max_steps
        self.shift_interval = shift_interval
        self.fixed_goal_index = fixed_goal_index
        self.rng = np.random.default_rng(seed)

        self.action_space = spaces.Discrete(4)  # up, down, left, right
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        g = grid_size - 1
        self.goals = [(0, 0), (0, g), (g, 0), (g, g)]

        self.position = (grid_size // 2, grid_size // 2)
        self.goal_index = 0
        self.step_count = 0
        self.total_steps = 0
        self.shift_events = 0

    def _clip_pos(self, x: int, y: int) -> tuple[int, int]:
        x = int(np.clip(x, 0, self.grid_size - 1))
        y = int(np.clip(y, 0, self.grid_size - 1))
        return x, y

    def _obs(self) -> np.ndarray:
        x, y = self.position
        gx, gy = self.goals[self.goal_index]

        # Partial observability: only directional hint sign, not exact goal coordinates.
        hint_x = float(np.sign(gx - x))
        hint_y = float(np.sign(gy - y))

        x_norm = 2.0 * (x / max(1, self.grid_size - 1)) - 1.0
        y_norm = 2.0 * (y / max(1, self.grid_size - 1)) - 1.0
        return np.array([x_norm, y_norm, hint_x, hint_y], dtype=np.float32)

    def _goal(self) -> tuple[int, int]:
        return self.goals[self.goal_index]

    def _maybe_shift_goal(self) -> None:
        if self.fixed_goal_index is not None:
            return
        if self.shift_interval <= 0:
            return
        if self.total_steps > 0 and self.total_steps % self.shift_interval == 0:
            self.goal_index = (self.goal_index + 1) % len(self.goals)
            self.shift_events += 1

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.position = (self.grid_size // 2, self.grid_size // 2)
        self.step_count = 0

        if self.fixed_goal_index is not None:
            self.goal_index = int(self.fixed_goal_index) % len(self.goals)
        else:
            self.goal_index = int(self.rng.integers(0, len(self.goals)))

        return self._obs(), {
            "goal_index": self.goal_index,
            "goal": self._goal(),
            "shift_events": self.shift_events,
        }

    def step(self, action: int):
        self.step_count += 1
        self.total_steps += 1

        x, y = self.position
        if int(action) == 0:
            y -= 1
        elif int(action) == 1:
            y += 1
        elif int(action) == 2:
            x -= 1
        elif int(action) == 3:
            x += 1

        self.position = self._clip_pos(x, y)
        self._maybe_shift_goal()

        goal = self._goal()
        reached_goal = self.position == goal

        reward = -0.02
        if reached_goal:
            reward = 1.0

        terminated = reached_goal
        truncated = self.step_count >= self.max_steps

        obs = self._obs()
        info = {
            "goal_index": self.goal_index,
            "goal": goal,
            "position": self.position,
            "shift_events": self.shift_events,
        }
        return obs, float(reward), terminated, truncated, info

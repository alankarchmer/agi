from __future__ import annotations

from envs.ambiguity_trap import AmbiguityTrapEnv
from envs.non_stationary_gridworld import NonStationaryGridworldEnv


def test_non_stationary_gridworld_shifts_goal() -> None:
    env = NonStationaryGridworldEnv(grid_size=6, max_steps=50, shift_interval=3, seed=1)
    _, info = env.reset(seed=1)
    goal0 = info["goal_index"]

    for _ in range(4):
        _, _, terminated, truncated, info = env.step(0)
        if terminated or truncated:
            env.reset()

    goal1 = info["goal_index"]
    assert goal1 != goal0


def test_non_stationary_gridworld_fixed_goal() -> None:
    env = NonStationaryGridworldEnv(grid_size=6, max_steps=50, shift_interval=2, fixed_goal_index=2, seed=2)
    _, info = env.reset(seed=2)
    assert info["goal_index"] == 2

    for _ in range(10):
        _, _, terminated, truncated, info = env.step(1)
        if terminated or truncated:
            env.reset()

    assert info["goal_index"] == 2


def test_ambiguity_trap_escape_flag() -> None:
    env = AmbiguityTrapEnv(max_steps=5, escape_probability=1.0, seed=3)
    env.reset(seed=3)
    _, reward, terminated, truncated, info = env.step(1)

    assert reward > 0.0
    assert terminated is True
    assert truncated is False
    assert info["escaped"] is True

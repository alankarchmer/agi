from __future__ import annotations

from models.joots import JOOTSController


class DummyPolicy:
    def __init__(self) -> None:
        self.temperature_scale = 1.0
        self.epistemic_foraging = False

    def increase_temperature(self, factor: float = 2.0) -> None:
        self.temperature_scale *= factor

    def force_epistemic_foraging(self, enabled: bool = True) -> None:
        self.epistemic_foraging = enabled


class DummyOptimizer:
    def __init__(self) -> None:
        self.sgld_variance = None
        self.sam_steps = 0

    def inject_sgld_noise(self, variance: float = 0.01) -> None:
        self.sgld_variance = variance

    def enable_sam_mode(self, steps: int) -> None:
        self.sam_steps = steps


def test_jooots_trigger_escape_calls_expected_hooks() -> None:
    ctrl = JOOTSController(patience=4, sam_steps=17, sgld_variance=0.05)
    policy = DummyPolicy()
    optim = DummyOptimizer()

    ctrl.trigger_escape(1, policy, optim)
    assert policy.temperature_scale == 2.0
    assert policy.epistemic_foraging is True
    assert optim.sgld_variance is None

    ctrl.trigger_escape(2, policy, optim)
    assert optim.sgld_variance == 0.05
    assert optim.sam_steps == 17

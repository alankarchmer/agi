from __future__ import annotations

from models.joots import JOOTSController


def test_jooots_severe_requires_full_conjunction() -> None:
    ctrl = JOOTSController(
        patience=5,
        tolerance=1e-6,
        severe_var_threshold=1e-9,
        loss_floor=0.1,
        gsnr_floor=0.2,
        residual_floor=0.2,
    )

    for _ in range(5):
        ctrl.update_metrics(current_vfe=0.5, grad_norm=0.05, residual_error=0.05)

    # Stagnant + low GSNR but residual floor not met => mild, not severe.
    assert ctrl.detect_stagnation() == 1


def test_jooots_cooldown_and_budget_are_enforced() -> None:
    ctrl = JOOTSController(
        patience=5,
        tolerance=1e-6,
        severe_var_threshold=1e-9,
        loss_floor=0.1,
        gsnr_floor=0.2,
        residual_floor=0.1,
        cooldown_steps=2,
        max_triggers_per_window=1,
    )

    for _ in range(5):
        ctrl.update_metrics(current_vfe=0.6, grad_norm=0.05, residual_error=0.3)
    assert ctrl.detect_stagnation() == 2

    ctrl.update_metrics(current_vfe=0.6, grad_norm=0.05, residual_error=0.3)
    assert ctrl.detect_stagnation() == 0
    ctrl.update_metrics(current_vfe=0.6, grad_norm=0.05, residual_error=0.3)
    assert ctrl.detect_stagnation() == 0

    # Cooldown expired, but trigger budget still saturated in window.
    ctrl.update_metrics(current_vfe=0.6, grad_norm=0.05, residual_error=0.3)
    assert ctrl.detect_stagnation() == 1

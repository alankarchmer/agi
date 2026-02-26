from __future__ import annotations

from models.joots import JOOTSController


def test_jooots_detects_normal_vs_mild_vs_severe() -> None:
    normal = JOOTSController(patience=6, tolerance=1e-4, loss_floor=0.1)
    for i in range(6):
        normal.update_metrics(current_vfe=1.0 - 0.1 * i, grad_norm=1.0, residual_error=0.0)
    assert normal.detect_stagnation() == 0

    mild = JOOTSController(patience=6, tolerance=1e-3, loss_floor=0.1)
    for val in [0.8, 0.802, 0.798, 0.8015, 0.7985, 0.8]:
        mild.update_metrics(current_vfe=val, grad_norm=1.0, residual_error=0.0)
    assert mild.detect_stagnation() == 1

    severe = JOOTSController(
        patience=6,
        tolerance=1e-4,
        severe_var_threshold=1e-8,
        loss_floor=0.1,
        gsnr_floor=0.6,
        residual_floor=0.01,
    )
    for val in [0.5] * 6:
        severe.update_metrics(current_vfe=val, grad_norm=0.1, residual_error=0.2)
    assert severe.detect_stagnation() == 2

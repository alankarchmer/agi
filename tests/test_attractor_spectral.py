from __future__ import annotations

import torch

from models.attractor import AttractorDynamics


def test_spectral_penalty_behaves_as_expected() -> None:
    attr = AttractorDynamics(micro_dim=16, attractor_dim=12)

    with torch.no_grad():
        attr.J.copy_(torch.eye(12) * 1.2)
    loss_high = attr.get_spectral_loss(target_radius=0.95, weight=1.0)
    assert loss_high.item() > 0.0

    with torch.no_grad():
        attr.J.copy_(torch.eye(12) * 0.7)
    loss_low = attr.get_spectral_loss(target_radius=0.95, weight=1.0)
    assert loss_low.item() == 0.0

    jac = attr.jacobian(torch.randn(3, 16))
    assert jac.shape == (12, 12)
    assert torch.isfinite(jac).all()

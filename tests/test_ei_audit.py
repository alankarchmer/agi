from __future__ import annotations

import torch

from training.phase6_reliability import compute_dei_audit


def test_ei_audit_returns_finite_scalar() -> None:
    torch.manual_seed(2)
    y_seq = torch.randn(12, 9, 4)
    out = compute_dei_audit(y_seq, samples=64, interventions_per_dim=8)

    assert out.dim() == 0
    assert torch.isfinite(out)


def test_ei_audit_positive_on_synthetic_causal_signal() -> None:
    torch.manual_seed(3)
    bsz, steps, dim = 20, 10, 4
    a = torch.tensor(
        [
            [0.7, 0.1, 0.0, 0.0],
            [0.0, 0.8, 0.1, 0.0],
            [0.0, 0.0, 0.75, 0.1],
            [0.05, 0.0, 0.0, 0.7],
        ],
        dtype=torch.float32,
    )

    y = torch.zeros(bsz, steps, dim)
    y[:, 0, :] = torch.randn(bsz, dim)
    for t in range(steps - 1):
        noise = 0.01 * torch.randn(bsz, dim)
        y[:, t + 1, :] = y[:, t, :] @ a + noise

    out = compute_dei_audit(y, samples=128, interventions_per_dim=10)
    assert torch.isfinite(out)
    assert out.item() > 0.0

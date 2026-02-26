from __future__ import annotations

import torch

from optim.adaptive_optimizer import AdaptiveOptimizer


def test_sgld_injection_scoped_to_l1_group_only() -> None:
    torch.manual_seed(4)

    p_l1 = torch.nn.Parameter(torch.zeros(6))
    p_l2 = torch.nn.Parameter(torch.zeros(6))

    optimizer = AdaptiveOptimizer(
        [
            {"params": [p_l1], "lr": 0.1, "name": "l1"},
            {"params": [p_l2], "lr": 0.1, "name": "l2"},
        ],
        lr=0.1,
    )

    p_l1.grad = torch.zeros_like(p_l1)
    p_l2.grad = torch.zeros_like(p_l2)

    optimizer.inject_sgld_noise(variance=0.05, target_groups=("l1",))
    optimizer.step()

    assert not torch.allclose(p_l1.detach(), torch.zeros_like(p_l1))
    assert torch.allclose(p_l2.detach(), torch.zeros_like(p_l2))


def test_enable_sam_mode_records_target_groups() -> None:
    p_l1 = torch.nn.Parameter(torch.zeros(2))
    p_l2 = torch.nn.Parameter(torch.zeros(2))
    optimizer = AdaptiveOptimizer(
        [
            {"params": [p_l1], "lr": 1e-3, "name": "l1"},
            {"params": [p_l2], "lr": 1e-3, "name": "l2"},
        ],
        lr=1e-3,
    )

    optimizer.enable_sam_mode(steps=5, target_groups=("l1",))
    assert optimizer.sam_steps_remaining == 5
    assert optimizer.sam_target_groups == ("l1",)

from __future__ import annotations

import torch

from models.rssm import GaussianRSSM


def test_vfe_loss_is_finite_and_logstd_clamped() -> None:
    model = GaussianRSSM(obs_dim=4, action_dim=3, hidden_dim=24, latent_dim=10)

    # Force extreme gaussian head outputs to validate internal clamping.
    with torch.no_grad():
        model.prior_head[-1].weight.zero_()
        model.prior_head[-1].bias.fill_(50.0)
        model.post_head[-1].weight.zero_()
        model.post_head[-1].bias.fill_(50.0)

    obs = torch.randn(8, 6, 4)
    act = torch.randn(8, 6, 3)

    out = model.compute_vfe_loss(obs, act, beta=1.0)
    assert torch.isfinite(out["total"])
    assert torch.isfinite(out["kl"])
    assert out["log_std_min"].item() >= -10.0 - 1e-5
    assert out["log_std_max"].item() <= 2.0 + 1e-5

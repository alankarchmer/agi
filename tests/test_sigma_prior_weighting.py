from __future__ import annotations

import torch

from models.rssm import GaussianRSSM


def test_sigma_prior_weight_zero_matches_no_sigma_path() -> None:
    torch.manual_seed(0)
    model = GaussianRSSM(obs_dim=5, action_dim=3, hidden_dim=32, latent_dim=8, sigma_dim=6)
    state = model.init_state(batch_size=4)

    obs = torch.randn(4, 5)
    prev_action = torch.zeros(4, 3)
    sigma = torch.randn(4, 6)

    model.set_sigma_prior_weight(0.0)
    out_with_sigma = model.forward_step(obs, prev_action, state, sigma_prior=sigma)
    out_without_sigma = model.forward_step(obs, prev_action, state, sigma_prior=None)

    assert torch.allclose(out_with_sigma.prior.mu, out_without_sigma.prior.mu, atol=1e-6, rtol=1e-6)
    assert torch.allclose(out_with_sigma.prior.log_std, out_without_sigma.prior.log_std, atol=1e-6, rtol=1e-6)


def test_sigma_prior_weight_nonzero_changes_prior() -> None:
    torch.manual_seed(1)
    model = GaussianRSSM(obs_dim=5, action_dim=3, hidden_dim=32, latent_dim=8, sigma_dim=6)
    state = model.init_state(batch_size=4)

    obs = torch.randn(4, 5)
    prev_action = torch.zeros(4, 3)
    sigma = torch.randn(4, 6)

    model.set_sigma_prior_weight(0.0)
    out_zero = model.forward_step(obs, prev_action, state, sigma_prior=sigma)

    model.set_sigma_prior_weight(1.0)
    out_one = model.forward_step(obs, prev_action, state, sigma_prior=sigma)

    assert not torch.allclose(out_zero.prior.mu, out_one.prior.mu)

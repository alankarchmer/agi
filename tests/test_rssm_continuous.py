from __future__ import annotations

import torch

from models.rssm import GaussianRSSM


def test_rssm_continuous_action_path_and_masked_loss() -> None:
    model = GaussianRSSM(
        obs_dim=5,
        action_dim=2,
        hidden_dim=32,
        latent_dim=8,
        action_space_type="continuous",
        obs_likelihood="gaussian",
        normalize_action=True,
    )
    model.set_normalization_stats(action_mean=torch.zeros(2), action_std=torch.ones(2))

    obs = torch.randn(6, 7, 5)
    act = torch.randn(6, 7, 2)
    mask = torch.ones(6, 7)
    mask[:, -2:] = 0.0

    out = model.compute_vfe_loss(obs, act, beta=1.0, valid_mask=mask, kl_free_nats=0.01, kl_balance=0.8)
    assert torch.isfinite(out["total"])

    state = model.init_state(batch_size=3)
    action, _ = model.sample_action(state, deterministic=False)
    assert action.shape == (3, 2)
    assert (action <= 1.0 + 1e-6).all()
    assert (action >= -1.0 - 1e-6).all()


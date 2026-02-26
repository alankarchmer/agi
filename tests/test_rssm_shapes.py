from __future__ import annotations

import torch

from models.rssm import GaussianRSSM


def test_rssm_forward_and_rollout_shapes() -> None:
    model = GaussianRSSM(obs_dim=6, action_dim=3, hidden_dim=32, latent_dim=12, sigma_dim=5, macro_dim=2)
    model.set_macro_feedback(True)

    bsz, seq_len = 4, 7
    obs = torch.randn(bsz, seq_len, 6)
    act = torch.randn(bsz, seq_len, 3)
    sigma_seq = torch.randn(bsz, seq_len, 5)
    y_seq = torch.randn(bsz, seq_len, 2)

    init_state = model.init_state(batch_size=bsz)
    step = model.forward_step(
        obs_t=obs[:, 0],
        prev_action=act[:, 0],
        prev_state=init_state,
        sigma_prior=sigma_seq[:, 0],
        y_macro=y_seq[:, 0],
    )

    assert step.recon.shape == (bsz, 6)
    assert step.action_logits.shape == (bsz, 3)
    assert step.state.h.shape == (bsz, 32)
    assert step.state.z.shape == (bsz, 12)

    rollout = model.rollout(obs, act, sigma_seq=sigma_seq, y_seq=y_seq)
    assert rollout.prior_mu.shape == (bsz, seq_len, 12)
    assert rollout.prior_log_std.shape == (bsz, seq_len, 12)
    assert rollout.post_mu.shape == (bsz, seq_len, 12)
    assert rollout.post_log_std.shape == (bsz, seq_len, 12)
    assert rollout.h_seq.shape == (bsz, seq_len, 32)
    assert rollout.z_seq.shape == (bsz, seq_len, 12)
    assert rollout.recon_seq.shape == (bsz, seq_len, 6)

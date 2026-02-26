from __future__ import annotations

import torch

from models.rssm import GaussianRSSM


def test_phase5_macro_feedback_path_on_off() -> None:
    model = GaussianRSSM(obs_dim=4, action_dim=3, hidden_dim=32, latent_dim=10, macro_dim=2)

    obs = torch.randn(10, 6, 4)
    act = torch.randn(10, 6, 3)
    y_seq = torch.randn(10, 6, 2)

    model.set_macro_feedback(True)
    out_on = model.compute_vfe_loss(obs, act, beta=1.0, y_seq=y_seq)
    out_on["total"].backward()
    assert torch.isfinite(out_on["total"])

    model.zero_grad(set_to_none=True)
    model.set_macro_feedback(False)
    out_off = model.compute_vfe_loss(obs, act, beta=1.0, y_seq=y_seq)
    out_off["total"].backward()
    assert torch.isfinite(out_off["total"])

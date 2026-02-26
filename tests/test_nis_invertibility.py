from __future__ import annotations

import torch

from models.nis import NISMacroState, RealNVP


def test_realnvp_invertibility() -> None:
    flow = RealNVP(dim=8, num_coupling_layers=4, hidden_dim=32)
    x = torch.randn(32, 8)

    z, _ = flow(x)
    x_rec, _ = flow.inverse(z)

    max_err = (x - x_rec).abs().max().item()
    assert max_err < 1e-4


def test_nis_macro_split_and_dei_proxy() -> None:
    nis = NISMacroState(micro_dim=10, macro_dim=4, num_coupling_layers=4, hidden_dim=32)
    sigma = torch.randn(6, 12, 10)

    y, z_noise, _ = nis(sigma)
    assert y.shape == (6, 12, 4)
    assert z_noise.shape == (6, 12, 6)

    sigma_rec = nis.inverse(y, z_noise)
    max_err = (sigma - sigma_rec).abs().max().item()
    assert max_err < 1e-4

    dei = nis.compute_dei_proxy(y)
    assert torch.isfinite(dei)

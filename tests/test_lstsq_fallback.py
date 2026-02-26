from __future__ import annotations

import torch

from models.nis import NISMacroState
from training.phase6_reliability import compute_dei_audit


def _patch_lstsq_raise_once(monkeypatch):
    original = torch.linalg.lstsq
    state = {"calls": 0}

    def wrapped(*args, **kwargs):
        if state["calls"] == 0:
            state["calls"] += 1
            raise NotImplementedError(
                "aten::linalg_lstsq.out is not currently implemented for the MPS device"
            )
        return original(*args, **kwargs)

    monkeypatch.setattr(torch.linalg, "lstsq", wrapped)
    return state


def test_nis_dei_proxy_lstsq_fallback(monkeypatch) -> None:
    state = _patch_lstsq_raise_once(monkeypatch)

    nis = NISMacroState(micro_dim=10, macro_dim=4, num_coupling_layers=2, hidden_dim=16)
    sigma = torch.randn(4, 7, 10)
    y, _, _ = nis(sigma)

    dei = nis.compute_dei_proxy(y)
    assert torch.isfinite(dei)
    assert state["calls"] == 1


def test_phase6_dei_audit_lstsq_fallback(monkeypatch) -> None:
    state = _patch_lstsq_raise_once(monkeypatch)

    y_seq = torch.randn(6, 8, 4)
    out = compute_dei_audit(y_seq, samples=32, interventions_per_dim=4)

    assert out.dim() == 0
    assert torch.isfinite(out)
    assert state["calls"] == 1

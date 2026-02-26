from __future__ import annotations

import torch


def mps_available() -> bool:
    return bool(
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
        and torch.backends.mps.is_built()
    )


def get_device(
    prefer_cuda: bool = True,
    prefer_mps: bool = False,
) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    if prefer_mps and mps_available():
        return torch.device("mps")
    return torch.device("cpu")

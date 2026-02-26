from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass
class EarlyStopState:
    patience: int
    min_delta: float = 1e-4
    best: float = float("inf")
    bad_steps: int = 0
    should_stop: bool = False

    def update(self, value: float) -> bool:
        if value < (self.best - self.min_delta):
            self.best = float(value)
            self.bad_steps = 0
            self.should_stop = False
            return True
        self.bad_steps += 1
        if self.bad_steps >= self.patience:
            self.should_stop = True
        return False


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    scaler: torch.cuda.amp.GradScaler | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "step": int(step),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    if extra:
        payload["extra"] = extra
    torch.save(payload, path)


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    model.load_state_dict(payload["model"])
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    if scaler is not None and "scaler" in payload:
        scaler.load_state_dict(payload["scaler"])
    return payload


def save_run_metadata(path: Path, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2))


def dataclass_to_dict(x: Any) -> dict[str, Any]:
    if hasattr(x, "__dataclass_fields__"):
        return asdict(x)
    raise TypeError(f"Expected dataclass instance, got {type(x)!r}")


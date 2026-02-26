from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from torch.utils.tensorboard import SummaryWriter


@dataclass
class LoggerConfig:
    log_dir: str = "runs/asl"
    use_wandb: bool = False
    wandb_project: str = "asl-v1"
    wandb_run_name: str | None = None


class MetricLogger:
    def __init__(self, config: LoggerConfig):
        self.config = config
        Path(config.log_dir).mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=config.log_dir)
        self._wandb = None

        if config.use_wandb:
            try:
                import wandb  # type: ignore

                wandb.init(project=config.wandb_project, name=config.wandb_run_name)
                self._wandb = wandb
            except Exception:
                self._wandb = None

    def log_scalar(self, name: str, value: float, step: int) -> None:
        self.writer.add_scalar(name, value, step)
        if self._wandb is not None:
            self._wandb.log({name: value}, step=step)

    def log_dict(self, metrics: dict[str, float], step: int) -> None:
        for k, v in metrics.items():
            self.log_scalar(k, float(v), step)

    def close(self) -> None:
        self.writer.flush()
        self.writer.close()
        if self._wandb is not None:
            self._wandb.finish()

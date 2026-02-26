from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn.utils import clip_grad_norm_

from models.rssm import GaussianRSSM
from training.contracts import OptimizationConfig, Phase1Config
from utils.logging import LoggerConfig, MetricLogger


@dataclass
class Phase1Result:
    loss_history: list[float]
    kl_history: list[float]
    recon_history: list[float]
    grad_norm_history: list[float]


def beta_schedule(step: int, total_steps: int, start: float, end: float, anneal_frac: float) -> float:
    anneal_steps = max(1, int(total_steps * anneal_frac))
    if step >= anneal_steps:
        return end
    alpha = step / anneal_steps
    return start + (end - start) * alpha


def _sample_batch(
    obs_seq: torch.Tensor,
    act_seq: torch.Tensor,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    idx = torch.randint(0, obs_seq.shape[0], (batch_size,), device=obs_seq.device)
    return obs_seq[idx], act_seq[idx], idx


def train_phase1_world_model(
    model: GaussianRSSM,
    obs_seq: torch.Tensor,
    act_seq: torch.Tensor,
    phase_cfg: Phase1Config,
    optim_cfg: OptimizationConfig,
    device: torch.device,
    logger_cfg: LoggerConfig | None = None,
    valid_mask_seq: torch.Tensor | None = None,
) -> Phase1Result:
    model = model.to(device)
    model.train()

    obs_seq = obs_seq.to(device)
    act_seq = act_seq.to(device)
    if valid_mask_seq is not None:
        valid_mask_seq = valid_mask_seq.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=optim_cfg.lr_l1, weight_decay=optim_cfg.weight_decay)
    logger = MetricLogger(logger_cfg) if logger_cfg is not None else None

    loss_hist: list[float] = []
    kl_hist: list[float] = []
    recon_hist: list[float] = []
    grad_hist: list[float] = []

    for step in range(phase_cfg.steps):
        beta = beta_schedule(
            step=step,
            total_steps=phase_cfg.steps,
            start=phase_cfg.beta_start,
            end=phase_cfg.beta_end,
            anneal_frac=phase_cfg.beta_anneal_frac,
        )

        batch_obs, batch_act, idx = _sample_batch(obs_seq, act_seq, phase_cfg.batch_size)
        batch_mask = None
        if valid_mask_seq is not None:
            batch_mask = valid_mask_seq[idx]

        loss_dict = model.compute_vfe_loss(
            batch_obs,
            batch_act,
            beta=beta,
            valid_mask=batch_mask,
            kl_balance=phase_cfg.kl_balance,
            kl_free_nats=phase_cfg.kl_free_nats,
            overshooting_horizon=phase_cfg.overshooting_horizon,
            overshooting_weight=phase_cfg.overshooting_weight,
        )

        optimizer.zero_grad(set_to_none=True)
        loss_dict["total"].backward()
        grad_norm = float(clip_grad_norm_(model.parameters(), optim_cfg.grad_clip_norm).item())
        optimizer.step()

        loss_val = float(loss_dict["total"].item())
        kl_val = float(loss_dict["kl"].item())
        recon_val = float(loss_dict["recon"].item())

        if not torch.isfinite(loss_dict["total"]):
            raise FloatingPointError("Phase 1 produced non-finite loss")
        if loss_dict["log_std_min"].item() < -10.0001 or loss_dict["log_std_max"].item() > 2.0001:
            raise FloatingPointError("log_std left clamped range")

        loss_hist.append(loss_val)
        kl_hist.append(kl_val)
        recon_hist.append(recon_val)
        grad_hist.append(grad_norm)

        if logger is not None:
            logger.log_dict(
                {
                    "Loss_VFE": loss_val,
                    "Loss_Recon": recon_val,
                    "Loss_KL": kl_val,
                    "Grad_Norm": grad_norm,
                    "KL_Beta": beta,
                },
                step,
            )

    if logger is not None:
        logger.close()

    return Phase1Result(
        loss_history=loss_hist,
        kl_history=kl_hist,
        recon_history=recon_hist,
        grad_norm_history=grad_hist,
    )

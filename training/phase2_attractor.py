from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from models.attractor import AttractorDynamics
from models.rssm import GaussianRSSM
from training.contracts import OptimizationConfig, Phase2Config
from utils.logging import LoggerConfig, MetricLogger


@dataclass
class Phase2Result:
    loss_history: list[float]
    mse_history: list[float]
    spectral_history: list[float]
    radius_history: list[float]
    residual_history: list[float]


def _sample_batch(
    obs_seq: torch.Tensor,
    act_seq: torch.Tensor,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    idx = torch.randint(0, obs_seq.shape[0], (batch_size,), device=obs_seq.device)
    return obs_seq[idx], act_seq[idx], idx


def train_phase2_attractor(
    world_model: GaussianRSSM,
    attractor: AttractorDynamics,
    obs_seq: torch.Tensor,
    act_seq: torch.Tensor,
    phase_cfg: Phase2Config,
    optim_cfg: OptimizationConfig,
    device: torch.device,
    logger_cfg: LoggerConfig | None = None,
    valid_mask_seq: torch.Tensor | None = None,
) -> Phase2Result:
    world_model = world_model.to(device)
    attractor = attractor.to(device)

    world_model.eval()
    for p in world_model.parameters():
        p.requires_grad_(False)

    obs_seq = obs_seq.to(device)
    act_seq = act_seq.to(device)
    if valid_mask_seq is not None:
        valid_mask_seq = valid_mask_seq.to(device)

    optimizer = torch.optim.Adam(attractor.parameters(), lr=optim_cfg.lr_l2, weight_decay=optim_cfg.weight_decay)
    logger = MetricLogger(logger_cfg) if logger_cfg is not None else None

    loss_hist: list[float] = []
    mse_hist: list[float] = []
    spec_hist: list[float] = []
    rho_hist: list[float] = []
    residual_hist: list[float] = []

    for step in range(phase_cfg.steps):
        batch_obs, batch_act, idx = _sample_batch(obs_seq, act_seq, phase_cfg.batch_size)
        batch_mask = valid_mask_seq[idx] if valid_mask_seq is not None else None

        with torch.no_grad():
            rollout = world_model.rollout(batch_obs, batch_act)
            micro = torch.cat([rollout.h_seq, rollout.z_seq], dim=-1).detach()

        micro_flat = micro.reshape(-1, micro.shape[-1])
        out = attractor(micro_flat, return_trajectory=True)
        sigma = out.sigma
        x_hat = attractor.reconstruct_micro(sigma)

        if batch_mask is None:
            mse = F.mse_loss(x_hat, micro_flat)
        else:
            mask_flat = batch_mask.reshape(-1)
            mse_vec = ((x_hat - micro_flat).pow(2)).mean(dim=-1)
            denom = mask_flat.sum().clamp_min(1.0)
            mse = (mse_vec * mask_flat).sum() / denom
        spec = attractor.get_spectral_loss(target_radius=phase_cfg.target_radius, weight=phase_cfg.lambda_spec)
        loss = mse + spec

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        clip_grad_norm_(attractor.parameters(), optim_cfg.grad_clip_norm)
        optimizer.step()

        rho = float(attractor.spectral_radius().item())
        if out.trajectory is not None and out.trajectory.shape[1] > 1:
            residual_vec = (out.trajectory[:, -1] - out.trajectory[:, -2]).norm(dim=-1)
            if batch_mask is None:
                residual = residual_vec.mean()
            else:
                mask_flat = batch_mask.reshape(-1)
                denom = mask_flat.sum().clamp_min(1.0)
                residual = (residual_vec * mask_flat).sum() / denom
        else:
            residual = torch.tensor(0.0, device=device)

        loss_hist.append(float(loss.item()))
        mse_hist.append(float(mse.item()))
        spec_hist.append(float(spec.item()))
        rho_hist.append(rho)
        residual_hist.append(float(residual.item()))

        if logger is not None:
            logger.log_dict(
                {
                    "Attractor_Loss": loss_hist[-1],
                    "Attractor_MSE": mse_hist[-1],
                    "Attractor_SpectralLoss": spec_hist[-1],
                    "Attractor_Max_Eigenvalue": rho_hist[-1],
                    "Attractor_SettlingResidual": residual_hist[-1],
                },
                step,
            )

    if logger is not None:
        logger.close()

    return Phase2Result(
        loss_history=loss_hist,
        mse_history=mse_hist,
        spectral_history=spec_hist,
        radius_history=rho_hist,
        residual_history=residual_hist,
    )

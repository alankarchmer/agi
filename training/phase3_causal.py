from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from models.attractor import AttractorDynamics
from models.nis import MacroTransition, NISMacroState
from models.rssm import GaussianRSSM
from training.contracts import OptimizationConfig, Phase3Config
from utils.logging import LoggerConfig, MetricLogger


@dataclass
class Phase3Result:
    loss_history: list[float]
    macro_mse_history: list[float]
    dei_history: list[float]


def _sample_batch(
    obs_seq: torch.Tensor,
    act_seq: torch.Tensor,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    idx = torch.randint(0, obs_seq.shape[0], (batch_size,), device=obs_seq.device)
    return obs_seq[idx], act_seq[idx], idx


def train_phase3_causal(
    world_model: GaussianRSSM,
    attractor: AttractorDynamics,
    nis: NISMacroState,
    macro_transition: MacroTransition,
    obs_seq: torch.Tensor,
    act_seq: torch.Tensor,
    phase_cfg: Phase3Config,
    optim_cfg: OptimizationConfig,
    device: torch.device,
    logger_cfg: LoggerConfig | None = None,
    valid_mask_seq: torch.Tensor | None = None,
) -> Phase3Result:
    world_model = world_model.to(device)
    attractor = attractor.to(device)
    nis = nis.to(device)
    macro_transition = macro_transition.to(device)

    world_model.eval()
    attractor.eval()
    for p in world_model.parameters():
        p.requires_grad_(False)
    for p in attractor.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.Adam(
        list(nis.parameters()) + list(macro_transition.parameters()),
        lr=optim_cfg.lr_l3,
        weight_decay=optim_cfg.weight_decay,
    )

    obs_seq = obs_seq.to(device)
    act_seq = act_seq.to(device)
    if valid_mask_seq is not None:
        valid_mask_seq = valid_mask_seq.to(device)
    logger = MetricLogger(logger_cfg) if logger_cfg is not None else None

    loss_hist: list[float] = []
    macro_mse_hist: list[float] = []
    dei_hist: list[float] = []

    for step in range(phase_cfg.steps):
        batch_obs, batch_act, idx = _sample_batch(obs_seq, act_seq, phase_cfg.batch_size)
        batch_mask = valid_mask_seq[idx] if valid_mask_seq is not None else None

        with torch.no_grad():
            rollout = world_model.rollout(batch_obs, batch_act)
            micro = torch.cat([rollout.h_seq, rollout.z_seq], dim=-1).detach()
            micro_flat = micro.reshape(-1, micro.shape[-1])
            sigma_flat = attractor(micro_flat)
            sigma_seq = sigma_flat.reshape(micro.shape[0], micro.shape[1], -1)

        y_seq, _, _ = nis(sigma_seq)
        y_pred = macro_transition(y_seq[:, :-1, :])
        y_next = y_seq[:, 1:, :]

        if batch_mask is None:
            macro_mse = F.mse_loss(y_pred, y_next)
        else:
            trans_mask = (batch_mask[:, :-1] * batch_mask[:, 1:]).float()
            mse_bt = (y_pred - y_next).pow(2).mean(dim=-1)
            denom = trans_mask.sum().clamp_min(1.0)
            macro_mse = (mse_bt * trans_mask).sum() / denom
        dei = nis.compute_dei_proxy(y_seq)
        loss = macro_mse - phase_cfg.alpha_dei * dei

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        clip_grad_norm_(list(nis.parameters()) + list(macro_transition.parameters()), optim_cfg.grad_clip_norm)
        optimizer.step()

        loss_hist.append(float(loss.item()))
        macro_mse_hist.append(float(macro_mse.item()))
        dei_hist.append(float(dei.item()))

        if logger is not None:
            logger.log_dict(
                {
                    "Phase3_Loss": loss_hist[-1],
                    "Macro_Pred_MSE": macro_mse_hist[-1],
                    "dEI_Proxy": dei_hist[-1],
                },
                step,
            )

    if logger is not None:
        logger.close()

    return Phase3Result(loss_history=loss_hist, macro_mse_history=macro_mse_hist, dei_history=dei_hist)

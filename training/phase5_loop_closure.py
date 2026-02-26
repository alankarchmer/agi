from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from models.attractor import AttractorDynamics
from models.nis import MacroTransition, NISMacroState
from models.rssm import GaussianRSSM
from training.contracts import OptimizationConfig, Phase5Config
from utils.logging import LoggerConfig, MetricLogger


@dataclass
class Phase5Result:
    total_loss_history: list[float]
    vfe_history: list[float]
    macro_align_history: list[float]


def _sample_batch(
    obs_seq: torch.Tensor,
    act_seq: torch.Tensor,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    idx = torch.randint(0, obs_seq.shape[0], (batch_size,), device=obs_seq.device)
    return obs_seq[idx], act_seq[idx], idx


def train_phase5_loop_closure(
    world_model: GaussianRSSM,
    attractor: AttractorDynamics,
    nis: NISMacroState,
    macro_transition: MacroTransition,
    obs_seq: torch.Tensor,
    act_seq: torch.Tensor,
    phase_cfg: Phase5Config,
    optim_cfg: OptimizationConfig,
    device: torch.device,
    logger_cfg: LoggerConfig | None = None,
    valid_mask_seq: torch.Tensor | None = None,
) -> Phase5Result:
    """
    Close the strange loop by feeding macro-state y_t back as sensory context.

    This phase keeps attractor and macro modules frozen and adapts the world model
    to operate with macro feedback enabled.
    """
    world_model = world_model.to(device)
    attractor = attractor.to(device)
    nis = nis.to(device)
    macro_transition = macro_transition.to(device)

    # Phase 5 trains the world-model under macro feedback while keeping higher levels fixed.
    world_model.train()
    attractor.eval()
    nis.eval()
    macro_transition.eval()

    for p in world_model.parameters():
        p.requires_grad_(True)
    for p in attractor.parameters():
        p.requires_grad_(False)
    for p in nis.parameters():
        p.requires_grad_(False)
    for p in macro_transition.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.Adam(
        world_model.parameters(),
        lr=optim_cfg.lr_l1,
        weight_decay=optim_cfg.weight_decay,
    )

    obs_seq = obs_seq.to(device)
    act_seq = act_seq.to(device)
    if valid_mask_seq is not None:
        valid_mask_seq = valid_mask_seq.to(device)

    logger = MetricLogger(logger_cfg) if logger_cfg is not None else None

    total_hist: list[float] = []
    vfe_hist: list[float] = []
    macro_align_hist: list[float] = []

    world_model.set_macro_feedback(True)

    for step in range(phase_cfg.steps):
        batch_obs, batch_act, idx = _sample_batch(obs_seq, act_seq, phase_cfg.batch_size)
        batch_mask = valid_mask_seq[idx] if valid_mask_seq is not None else None

        # Build a detached macro sequence from current frozen higher-level state extractor.
        with torch.no_grad():
            rollout_base = world_model.rollout(batch_obs, batch_act, y_seq=None)
            micro_base = torch.cat([rollout_base.h_seq, rollout_base.z_seq], dim=-1)
            micro_flat = micro_base.reshape(-1, micro_base.shape[-1])
            sigma_flat = attractor(micro_flat)
            sigma_seq = sigma_flat.reshape(micro_base.shape[0], micro_base.shape[1], -1)
            y_seq, _, _ = nis(sigma_seq)
            y_seq = y_seq.detach()

        loss_vfe = world_model.compute_vfe_loss(
            batch_obs,
            batch_act,
            beta=phase_cfg.beta,
            y_seq=y_seq,
            valid_mask=batch_mask,
        )

        # Encourage feedback-consistency: the feedback-conditioned trajectory should
        # regenerate compatible macro-states under frozen L2/L3 transforms.
        rollout_feedback = world_model.rollout(batch_obs, batch_act, y_seq=y_seq)
        micro_feedback = torch.cat([rollout_feedback.h_seq, rollout_feedback.z_seq], dim=-1)
        micro_feedback_flat = micro_feedback.reshape(-1, micro_feedback.shape[-1])

        sigma_feedback_flat = attractor(micro_feedback_flat)
        sigma_feedback_seq = sigma_feedback_flat.reshape(micro_feedback.shape[0], micro_feedback.shape[1], -1)
        y_feedback, _, _ = nis(sigma_feedback_seq)

        if batch_mask is None:
            macro_align = F.mse_loss(y_feedback, y_seq)
        else:
            mask_bt = batch_mask.float()
            mse_bt = (y_feedback - y_seq).pow(2).mean(dim=-1)
            denom = mask_bt.sum().clamp_min(1.0)
            macro_align = (mse_bt * mask_bt).sum() / denom
        total = loss_vfe["total"] + phase_cfg.macro_align_weight * macro_align

        optimizer.zero_grad(set_to_none=True)
        total.backward()
        clip_grad_norm_(world_model.parameters(), optim_cfg.grad_clip_norm)
        optimizer.step()

        total_hist.append(float(total.item()))
        vfe_hist.append(float(loss_vfe["total"].item()))
        macro_align_hist.append(float(macro_align.item()))

        if logger is not None:
            logger.log_dict(
                {
                    "Phase5_Total": total_hist[-1],
                    "Loss_VFE": vfe_hist[-1],
                    "Phase5_MacroAlign": macro_align_hist[-1],
                },
                step,
            )

    if logger is not None:
        logger.close()

    return Phase5Result(
        total_loss_history=total_hist,
        vfe_history=vfe_hist,
        macro_align_history=macro_align_hist,
    )

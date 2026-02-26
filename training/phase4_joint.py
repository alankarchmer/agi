from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from models.attractor import AttractorDynamics
from models.joots import JOOTSController
from models.nis import MacroTransition, NISMacroState
from models.rssm import GaussianRSSM
from optim.adaptive_optimizer import AdaptiveOptimizer
from training.contracts import OptimizationConfig, Phase4Config
from utils.logging import LoggerConfig, MetricLogger


@dataclass
class Phase4Result:
    total_loss_history: list[float]
    vfe_history: list[float]
    attr_loss_history: list[float]
    macro_loss_history: list[float]
    jooots_severity_history: list[int]


def _sample_batch(
    obs_seq: torch.Tensor,
    act_seq: torch.Tensor,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    idx = torch.randint(0, obs_seq.shape[0], (batch_size,), device=obs_seq.device)
    return obs_seq[idx], act_seq[idx], idx


def _global_grad_norm(parameters) -> float:
    norms = []
    for p in parameters:
        if p.grad is not None:
            norms.append(p.grad.norm(2))
    if not norms:
        return 0.0
    return float(torch.norm(torch.stack(norms), 2).item())


def train_phase4_joint(
    world_model: GaussianRSSM,
    attractor: AttractorDynamics,
    nis: NISMacroState,
    macro_transition: MacroTransition,
    controller: JOOTSController,
    obs_seq: torch.Tensor,
    act_seq: torch.Tensor,
    phase_cfg: Phase4Config,
    optim_cfg: OptimizationConfig,
    device: torch.device,
    logger_cfg: LoggerConfig | None = None,
    valid_mask_seq: torch.Tensor | None = None,
) -> Phase4Result:
    world_model = world_model.to(device)
    attractor = attractor.to(device)
    nis = nis.to(device)
    macro_transition = macro_transition.to(device)

    world_model.train()
    attractor.train()
    nis.train()
    macro_transition.train()

    for p in list(world_model.parameters()) + list(attractor.parameters()) + list(nis.parameters()) + list(
        macro_transition.parameters()
    ):
        p.requires_grad_(True)

    optimizer = AdaptiveOptimizer(
        [
            {"params": world_model.parameters(), "lr": optim_cfg.lr_l1, "name": "l1"},
            {"params": attractor.parameters(), "lr": optim_cfg.lr_l2, "name": "l2"},
            {"params": nis.parameters(), "lr": optim_cfg.lr_l3, "name": "l3"},
            {"params": macro_transition.parameters(), "lr": optim_cfg.lr_l3, "name": "l3"},
        ],
        lr=optim_cfg.lr_l1,
        weight_decay=optim_cfg.weight_decay,
    )

    obs_seq = obs_seq.to(device)
    act_seq = act_seq.to(device)
    if valid_mask_seq is not None:
        valid_mask_seq = valid_mask_seq.to(device)

    logger = MetricLogger(logger_cfg) if logger_cfg is not None else None

    all_params = list(world_model.parameters()) + list(attractor.parameters()) + list(nis.parameters()) + list(
        macro_transition.parameters()
    )

    total_hist: list[float] = []
    vfe_hist: list[float] = []
    attr_hist: list[float] = []
    macro_hist: list[float] = []
    severity_hist: list[int] = []

    trigger_count = 0

    for step in range(phase_cfg.steps):
        batch_obs, batch_act, idx = _sample_batch(obs_seq, act_seq, phase_cfg.batch_size)
        batch_mask = valid_mask_seq[idx] if valid_mask_seq is not None else None
        metrics: dict[str, float] = {}

        def closure():
            optimizer.zero_grad(set_to_none=True)

            vfe = world_model.compute_vfe_loss(batch_obs, batch_act, beta=1.0, valid_mask=batch_mask)
            rollout = world_model.rollout(batch_obs, batch_act)
            micro = torch.cat([rollout.h_seq, rollout.z_seq], dim=-1)
            micro_flat = micro.reshape(-1, micro.shape[-1])

            attractor_out = attractor(micro_flat, return_trajectory=True)
            sigma_flat = attractor_out.sigma
            x_hat = attractor.reconstruct_micro(sigma_flat)
            if batch_mask is None:
                attr_mse = F.mse_loss(x_hat, micro_flat)
            else:
                mask_flat = batch_mask.reshape(-1)
                mse_vec = (x_hat - micro_flat).pow(2).mean(dim=-1)
                denom = mask_flat.sum().clamp_min(1.0)
                attr_mse = (mse_vec * mask_flat).sum() / denom
            spec = attractor.get_spectral_loss(target_radius=0.95, weight=1.0)
            attr_loss = attr_mse + spec
            if attractor_out.trajectory is not None and attractor_out.trajectory.shape[1] > 1:
                residual_error = float(
                    (attractor_out.trajectory[:, -1] - attractor_out.trajectory[:, -2]).norm(dim=-1).mean().item()
                )
            else:
                residual_error = 0.0

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
            macro_loss = macro_mse - phase_cfg.alpha_dei * dei

            total = vfe["total"] + phase_cfg.attr_loss_weight * attr_loss + phase_cfg.macro_loss_weight * macro_loss
            total.backward()
            clip_grad_norm_(all_params, optim_cfg.grad_clip_norm)

            metrics["total"] = float(total.item())
            metrics["vfe"] = float(vfe["total"].item())
            metrics["attr_loss"] = float(attr_loss.item())
            metrics["macro_loss"] = float(macro_loss.item())
            metrics["rho"] = float(attractor.spectral_radius().item())
            metrics["dei"] = float(dei.item())
            metrics["macro_mse"] = float(macro_mse.item())
            metrics["grad_norm"] = _global_grad_norm(all_params)
            metrics["residual_error"] = residual_error
            return total

        if optimizer.sam_steps_remaining > 0:
            optimizer.step(closure)
        else:
            closure()
            optimizer.step()

        controller.update_metrics(metrics["vfe"], metrics["grad_norm"], metrics["residual_error"])
        severity = controller.detect_stagnation()
        if severity > 0:
            controller.trigger_escape(severity, world_model, optimizer)
            trigger_count += 1
        controller.apply_recovery(world_model, consecutive_steps=20)

        total_hist.append(metrics["total"])
        vfe_hist.append(metrics["vfe"])
        attr_hist.append(metrics["attr_loss"])
        macro_hist.append(metrics["macro_loss"])
        severity_hist.append(severity)

        if logger is not None:
            logger.log_dict(
                {
                    "Loss_VFE": metrics["vfe"],
                    "Phase4_Total": metrics["total"],
                    "Attractor_MSE": metrics["attr_loss"],
                    "Attractor_Max_Eigenvalue": metrics["rho"],
                    "dEI_Proxy": metrics["dei"],
                    "Macro_Pred_MSE": metrics["macro_mse"],
                    "JOOTS_Severity": float(severity),
                    "JOOTS_Trigger_Count": float(trigger_count),
                    "Policy_Temperature": float(world_model.temperature),
                    "Grad_Norm": metrics["grad_norm"],
                },
                step,
            )

    if logger is not None:
        logger.close()

    return Phase4Result(
        total_loss_history=total_hist,
        vfe_history=vfe_hist,
        attr_loss_history=attr_hist,
        macro_loss_history=macro_hist,
        jooots_severity_history=severity_hist,
    )

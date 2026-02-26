from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from models.attractor import AttractorDynamics
from models.joots import JOOTSController
from models.nis import MacroTransition, NISMacroState
from models.rssm import GaussianRSSM
from optim.adaptive_optimizer import AdaptiveOptimizer
from training.contracts import EvalConfig, OptimizationConfig, Phase6Config
from training.phase6_reliability import compute_dei_audit, spearman_corr
from training.real_eval import evaluate_real_data_gates
from utils.logging import LoggerConfig, MetricLogger


@dataclass
class Phase6RealStageResult:
    stage_index: int
    sigma_weight: float
    attempts: int
    passed: bool
    rolled_back: bool
    median_vfe: float
    eval_nll_mean: float
    eval_kl_mean: float
    eval_temporal_drift: float
    eval_ood_gap: float
    dei_audit_last: float
    proxy_audit_spearman: float
    gates: dict[str, bool]
    checkpoint_path: str | None = None


@dataclass
class Phase6RealResult:
    stage_results: list[Phase6RealStageResult]
    all_passed: bool
    total_jooots_triggers: int
    proxy_history: list[float]
    audit_history: list[float]


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


def _build_sigma_macro_sequences(
    world_model: GaussianRSSM,
    attractor: AttractorDynamics,
    nis: NISMacroState,
    obs_batch: torch.Tensor,
    act_batch: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    with torch.no_grad():
        rollout = world_model.rollout(obs_batch, act_batch)
        micro = torch.cat([rollout.h_seq, rollout.z_seq], dim=-1)
        micro_flat = micro.reshape(-1, micro.shape[-1])

        attractor_out = attractor(micro_flat, return_trajectory=True)
        sigma_flat = attractor_out.sigma
        sigma_seq = sigma_flat.reshape(micro.shape[0], micro.shape[1], -1)

        y_seq, _, _ = nis(sigma_seq)

        if attractor_out.trajectory is not None and attractor_out.trajectory.shape[1] > 1:
            residual_error = float(
                (attractor_out.trajectory[:, -1] - attractor_out.trajectory[:, -2]).norm(dim=-1).mean().item()
            )
        else:
            residual_error = 0.0

    return sigma_seq.detach(), y_seq.detach(), residual_error


def train_phase6_real_data(
    world_model: GaussianRSSM,
    attractor: AttractorDynamics,
    nis: NISMacroState,
    macro_transition: MacroTransition,
    controller: JOOTSController,
    obs_seq: torch.Tensor,
    act_seq: torch.Tensor,
    eval_loader: DataLoader,
    phase_cfg: Phase6Config,
    optim_cfg: OptimizationConfig,
    eval_cfg: EvalConfig,
    device: torch.device,
    checkpoint_dir: Path | None = None,
    logger_cfg: LoggerConfig | None = None,
    valid_mask_seq: torch.Tensor | None = None,
) -> Phase6RealResult:
    world_model = world_model.to(device)
    attractor = attractor.to(device)
    nis = nis.to(device)
    macro_transition = macro_transition.to(device)

    obs_seq = obs_seq.to(device)
    act_seq = act_seq.to(device)
    if valid_mask_seq is not None:
        valid_mask_seq = valid_mask_seq.to(device)

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

    world_model.set_macro_feedback(True)

    logger = MetricLogger(logger_cfg) if logger_cfg is not None else None

    stage_results: list[Phase6RealStageResult] = []
    proxy_history: list[float] = []
    audit_history: list[float] = []
    total_jooots_triggers = 0

    prev_stage_median_vfe: float | None = None
    prev_stage_nll: float | None = None
    min_corr_points = 6

    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for stage_index, sigma_weight in enumerate(phase_cfg.sigma_ramp_weights):
        world_model.set_sigma_prior_weight(sigma_weight)
        stage_start_state = {k: v.detach().clone() for k, v in world_model.state_dict().items()}

        stage_lr = optim_cfg.lr_l1
        attempts = 0
        final_stage: Phase6RealStageResult | None = None

        for attempt in range(2):
            attempts = attempt + 1
            if attempt > 0:
                world_model.load_state_dict(stage_start_state)
                stage_lr *= 0.5

            stage_controller = JOOTSController(
                patience=controller.patience,
                tolerance=controller.tolerance,
                severe_var_threshold=controller.severe_var_threshold,
                loss_floor=controller.loss_floor,
                gsnr_floor=controller.gsnr_floor,
                residual_floor=max(controller.residual_floor, 1e-6),
                cooldown_steps=phase_cfg.jooots_cooldown_steps,
                max_triggers_per_window=phase_cfg.jooots_max_triggers_per_window,
                recovery_window=controller.recovery_window,
                sam_steps=controller.sam_steps,
                sgld_variance=controller.sgld_variance,
            )

            optimizer = AdaptiveOptimizer(
                [{"params": world_model.parameters(), "lr": stage_lr, "name": "l1"}],
                lr=stage_lr,
                weight_decay=optim_cfg.weight_decay,
            )

            stage_vfe: list[float] = []
            stage_proxy: list[float] = []
            stage_audit: list[float] = []

            for step in range(phase_cfg.sigma_stage_steps):
                batch_obs, batch_act, idx = _sample_batch(obs_seq, act_seq, phase_cfg.batch_size)
                batch_mask = valid_mask_seq[idx] if valid_mask_seq is not None else None
                sigma_seq, y_seq, residual_error = _build_sigma_macro_sequences(
                    world_model,
                    attractor,
                    nis,
                    batch_obs,
                    batch_act,
                )

                metrics: dict[str, float] = {}

                def closure():
                    optimizer.zero_grad(set_to_none=True)
                    loss_vfe = world_model.compute_vfe_loss(
                        batch_obs,
                        batch_act,
                        beta=1.0,
                        sigma_seq=sigma_seq,
                        y_seq=y_seq,
                        valid_mask=batch_mask,
                    )
                    total = loss_vfe["total"]
                    total.backward()
                    clip_grad_norm_(world_model.parameters(), optim_cfg.grad_clip_norm)

                    metrics["vfe"] = float(loss_vfe["total"].item())
                    metrics["grad_norm"] = _global_grad_norm(world_model.parameters())
                    return total

                if optimizer.sam_steps_remaining > 0:
                    optimizer.step(closure)
                else:
                    closure()
                    optimizer.step()

                stage_controller.update_metrics(metrics["vfe"], metrics["grad_norm"], residual_error)
                severity = stage_controller.detect_stagnation()
                if severity > 0:
                    stage_controller.trigger_escape(severity, world_model, optimizer)
                    total_jooots_triggers += 1
                stage_controller.apply_recovery(world_model, consecutive_steps=20)

                stage_vfe.append(metrics["vfe"])

                if step % max(1, phase_cfg.ei_audit_interval) == 0:
                    with torch.no_grad():
                        rollout_feedback = world_model.rollout(batch_obs, batch_act, sigma_seq=sigma_seq, y_seq=y_seq)
                        micro_feedback = torch.cat([rollout_feedback.h_seq, rollout_feedback.z_seq], dim=-1)
                        micro_feedback_flat = micro_feedback.reshape(-1, micro_feedback.shape[-1])
                        sigma_feedback = attractor(micro_feedback_flat).reshape(
                            micro_feedback.shape[0], micro_feedback.shape[1], -1
                        )
                        y_feedback, _, _ = nis(sigma_feedback)

                        proxy_val = float(nis.compute_dei_proxy(y_feedback).item())
                        audit_val = float(
                            compute_dei_audit(
                                y_feedback,
                                samples=phase_cfg.ei_audit_samples,
                                interventions_per_dim=phase_cfg.ei_interventions_per_dim,
                            ).item()
                        )

                    stage_proxy.append(proxy_val)
                    stage_audit.append(audit_val)
                    proxy_history.append(proxy_val)
                    audit_history.append(audit_val)

                if logger is not None:
                    logger.log_dict(
                        {
                            f"Phase6Real/Stage{stage_index}/Loss_VFE": stage_vfe[-1],
                            f"Phase6Real/Stage{stage_index}/SigmaWeight": float(sigma_weight),
                            f"Phase6Real/Stage{stage_index}/Residual": float(residual_error),
                            f"Phase6Real/Stage{stage_index}/JOOTSSeverity": float(severity),
                        },
                        step,
                    )

            median_vfe = float(torch.tensor(stage_vfe, dtype=torch.float32).median().item())

            eval_result = evaluate_real_data_gates(
                model=world_model,
                dataloader=eval_loader,
                cfg=eval_cfg,
                device=device,
                intervention_audit_history=None,
            )
            world_model.train()

            eval_nll = float(eval_result.metrics["nll_mean"])
            eval_kl = float(eval_result.metrics["kl_mean"])
            eval_drift = float(eval_result.metrics["temporal_drift"])
            eval_ood_gap = float(eval_result.metrics["ood_gap"])

            last_audit = stage_audit[-1] if stage_audit else float("nan")
            audit_tail = stage_audit[-min(3, len(stage_audit)) :] if stage_audit else []
            audit_gate_value = (
                float(torch.tensor(audit_tail, dtype=torch.float32).mean().item()) if audit_tail else float("nan")
            )
            corr = spearman_corr(stage_proxy, stage_audit)

            gates = {
                "vfe_non_regression": (
                    True
                    if prev_stage_median_vfe is None
                    else median_vfe <= prev_stage_median_vfe * (1.0 + phase_cfg.max_vfe_regression)
                ),
                "eval_nll_bound": (torch.isfinite(torch.tensor(eval_nll)).item() and eval_nll <= eval_cfg.nll_max),
                "eval_kl_bound": (eval_cfg.kl_min <= eval_kl <= eval_cfg.kl_max),
                "eval_temporal_drift_bound": (eval_drift <= eval_cfg.ood_drift_max),
                "eval_nll_non_regression": (
                    True
                    if prev_stage_nll is None
                    else eval_nll <= prev_stage_nll * (1.0 + phase_cfg.max_vfe_regression)
                ),
                "dei_audit_positive": (True if not stage_audit else audit_gate_value > 0.0),
                "proxy_audit_correlation": (
                    True if len(stage_proxy) < min_corr_points else (corr >= phase_cfg.min_proxy_audit_spearman)
                ),
            }
            passed = all(gates.values())

            rolled_back = False
            if not passed and attempt == 0:
                continue
            if not passed and attempt == 1:
                world_model.load_state_dict(stage_start_state)
                rolled_back = True

            checkpoint_path = None
            if checkpoint_dir is not None:
                ckpt_path = checkpoint_dir / f"stage_{stage_index}_sigma_{sigma_weight:.2f}.pt"
                torch.save(
                    {
                        "stage_index": stage_index,
                        "sigma_weight": sigma_weight,
                        "world_model": world_model.state_dict(),
                        "attractor": attractor.state_dict(),
                        "nis": nis.state_dict(),
                        "macro_transition": macro_transition.state_dict(),
                        "metrics": {
                            "median_vfe": median_vfe,
                            "eval_nll_mean": eval_nll,
                            "eval_kl_mean": eval_kl,
                            "eval_temporal_drift": eval_drift,
                            "eval_ood_gap": eval_ood_gap,
                            "dei_audit_last": last_audit,
                            "dei_audit_gate_value": audit_gate_value,
                            "proxy_audit_spearman": corr,
                        },
                        "gates": gates,
                        "passed": passed,
                        "rolled_back": rolled_back,
                    },
                    ckpt_path,
                )
                checkpoint_path = str(ckpt_path)

            final_stage = Phase6RealStageResult(
                stage_index=stage_index,
                sigma_weight=float(sigma_weight),
                attempts=attempts,
                passed=passed,
                rolled_back=rolled_back,
                median_vfe=median_vfe,
                eval_nll_mean=eval_nll,
                eval_kl_mean=eval_kl,
                eval_temporal_drift=eval_drift,
                eval_ood_gap=eval_ood_gap,
                dei_audit_last=float(last_audit),
                proxy_audit_spearman=float(corr),
                gates=gates,
                checkpoint_path=checkpoint_path,
            )
            break

        if final_stage is None:
            raise RuntimeError(f"Real-data Stage {stage_index} failed to finalize")

        stage_results.append(final_stage)
        if final_stage.passed:
            prev_stage_median_vfe = final_stage.median_vfe
            prev_stage_nll = final_stage.eval_nll_mean

        if logger is not None:
            logger.log_dict(
                {
                    "Phase6Real/StagePassed": float(final_stage.passed),
                    "Phase6Real/StageMedianVFE": float(final_stage.median_vfe),
                    "Phase6Real/EvalNLL": float(final_stage.eval_nll_mean),
                    "Phase6Real/EvalKL": float(final_stage.eval_kl_mean),
                },
                stage_index,
            )

    if logger is not None:
        logger.close()

    return Phase6RealResult(
        stage_results=stage_results,
        all_passed=all(s.passed for s in stage_results),
        total_jooots_triggers=total_jooots_triggers,
        proxy_history=proxy_history,
        audit_history=audit_history,
    )

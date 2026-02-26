from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from envs.tmaze import TMazeEnv
from models.attractor import AttractorDynamics
from models.joots import JOOTSController
from models.nis import MacroTransition, NISMacroState
from models.rssm import EFEScorer, GaussianRSSM
from optim.adaptive_optimizer import AdaptiveOptimizer
from training.contracts import OptimizationConfig, Phase6Config
from utils.logging import LoggerConfig, MetricLogger


@dataclass
class Phase6StageResult:
    stage_index: int
    sigma_weight: float
    attempts: int
    passed: bool
    rolled_back: bool
    median_vfe: float
    tmaze_ci_low: float
    tmaze_ci_high: float
    tmaze_ci_low_no_sigma: float
    tmaze_ci_high_no_sigma: float
    tmaze_adv_delta_vs_no_sigma: float
    dei_audit_last: float
    proxy_audit_spearman: float
    gates: dict[str, bool]
    checkpoint_path: str | None = None


@dataclass
class Phase6Result:
    stage_results: list[Phase6StageResult]
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


def _ci95(values: list[float]) -> tuple[float, float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0, 0.0
    t = torch.tensor(values, dtype=torch.float32)
    mean = float(t.mean().item())
    std = float(t.std(unbiased=False).item())
    half_width = 1.96 * std / max(1, t.numel()) ** 0.5
    return mean, std, mean - half_width, mean + half_width


def _rank_tensor(x: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(x)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(order.numel(), dtype=torch.float32, device=x.device)
    return ranks


def spearman_corr(x_values: list[float], y_values: list[float]) -> float:
    if len(x_values) != len(y_values) or len(x_values) < 3:
        return float("nan")

    x = torch.tensor(x_values, dtype=torch.float32)
    y = torch.tensor(y_values, dtype=torch.float32)

    rx = _rank_tensor(x)
    ry = _rank_tensor(y)

    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = (rx.norm() * ry.norm()).item()
    if denom <= 1e-12:
        return 0.0
    return float((rx @ ry).item() / denom)


def _lstsq_with_fallback(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    try:
        return torch.linalg.lstsq(x, y).solution
    except (NotImplementedError, RuntimeError) as exc:
        msg = str(exc).lower()
        should_fallback = (
            x.device.type == "mps"
            or "linalg_lstsq" in msg
            or "not currently implemented for the mps" in msg
        )
        if not should_fallback:
            raise

        x_cpu = x.to(dtype=torch.float32, device="cpu")
        y_cpu = y.to(dtype=torch.float32, device="cpu")
        sol_cpu = torch.linalg.lstsq(x_cpu, y_cpu).solution
        return sol_cpu.to(device=x.device, dtype=x.dtype)


def compute_dei_audit(
    y_seq: torch.Tensor,
    samples: int = 256,
    interventions_per_dim: int = 16,
    eps: float = 1e-5,
) -> torch.Tensor:
    if y_seq.dim() != 3:
        raise ValueError("y_seq must have shape (B, T, D)")
    if y_seq.shape[1] < 2:
        return torch.zeros((), device=y_seq.device, dtype=y_seq.dtype)

    x = y_seq[:, :-1, :].reshape(-1, y_seq.shape[-1])
    y = y_seq[:, 1:, :].reshape(-1, y_seq.shape[-1])

    n = x.shape[0]
    if n > samples:
        idx = torch.linspace(0, n - 1, steps=samples, device=x.device).long()
        x = x[idx]
        y = y[idx]

    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)

    a = _lstsq_with_fallback(x, y)
    y_hat = x @ a
    residual = y - y_hat

    d = x.shape[-1]
    eye = torch.eye(d, device=x.device, dtype=x.dtype)

    q_low = torch.quantile(x, 0.10, dim=0)
    q_high = torch.quantile(x, 0.90, dim=0)

    signals = []
    alpha_values = torch.linspace(0.0, 1.0, steps=max(2, interventions_per_dim), device=x.device, dtype=x.dtype)

    base_mean = y.mean(dim=0)
    for dim in range(d):
        for alpha in alpha_values:
            x_iv = x.clone()
            x_iv[:, dim] = q_low[dim] * (1.0 - alpha) + q_high[dim] * alpha
            y_iv = x_iv @ a
            signals.append(y_iv.mean(dim=0) - base_mean)

    if len(signals) < 2:
        return torch.zeros((), device=y_seq.device, dtype=y_seq.dtype)

    signal_mat = torch.stack(signals, dim=0)
    signal_centered = signal_mat - signal_mat.mean(dim=0, keepdim=True)
    signal_cov = (signal_centered.t() @ signal_centered) / max(1, signal_centered.shape[0] - 1)

    residual_centered = residual - residual.mean(dim=0, keepdim=True)
    residual_cov = (residual_centered.t() @ residual_centered) / max(1, residual_centered.shape[0] - 1)

    sign1, logdet1 = torch.linalg.slogdet(signal_cov + eps * eye)
    sign2, logdet2 = torch.linalg.slogdet(residual_cov + eps * eye)

    if sign1 <= 0 or sign2 <= 0:
        return torch.zeros((), device=y_seq.device, dtype=y_seq.dtype)
    return (logdet1 - logdet2) / float(d)


def _evaluate_tmaze_policy(
    model: GaussianRSSM,
    attractor: AttractorDynamics | None,
    episodes: int,
    device: torch.device,
    epistemic: bool,
    seed: int,
    use_sigma: bool = False,
) -> float:
    scorer = EFEScorer(model=model, pragmatic_weight=0.2, epistemic_weight=1.0)
    model.force_epistemic_foraging(epistemic)

    successes = 0
    for ep in range(episodes):
        env = TMazeEnv(seed=seed + ep)
        obs, _ = env.reset(seed=seed + ep)
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).view(1, -1)

        state = model.init_state(batch_size=1, device=device)
        prev_action = torch.zeros(1, model.action_dim, device=device)
        sigma_prev = torch.zeros(1, model.sigma_dim, device=device) if (use_sigma and model.sigma_dim > 0) else None

        done = False
        remembered_goal = 0
        ep_reward = 0.0

        while not done:
            if obs_t[0, 2].item() > 0.5:
                cue_val = int(round(float(obs_t[0, 3].item())))
                remembered_goal = cue_val if cue_val != 0 else remembered_goal
            at_junction = bool(obs_t[0, 4].item() > 0.5)

            with torch.no_grad():
                step = model.forward_step(obs_t, prev_action, state, sigma_prior=sigma_prev)
                state = step.state
                if use_sigma and attractor is not None and model.sigma_dim > 0:
                    micro = torch.cat([state.h, state.z], dim=-1)
                    sigma_prev = attractor(micro).detach()

                candidates = torch.eye(model.action_dim, device=device).unsqueeze(0)
                scores = scorer.score_actions(state, candidates)

                if at_junction and remembered_goal != 0:
                    action_idx = 1 if remembered_goal < 0 else 2
                elif epistemic and not at_junction:
                    action_idx = 0
                else:
                    action_idx = int(scores.argmax(dim=-1).item())

                prev_action = F.one_hot(
                    torch.tensor(action_idx, device=device), num_classes=model.action_dim
                ).float().view(1, -1)

            obs_np, reward, terminated, truncated, _ = env.step(action_idx)
            ep_reward += float(reward)
            obs_t = torch.tensor(obs_np, dtype=torch.float32, device=device).view(1, -1)
            done = bool(terminated or truncated)

        successes += int(ep_reward > 0.5)

    return successes / max(1, episodes)


def _evaluate_tmaze_random(episodes: int, seed: int) -> float:
    rng = torch.Generator().manual_seed(seed)
    successes = 0

    for ep in range(episodes):
        env = TMazeEnv(seed=seed + ep)
        _, _ = env.reset(seed=seed + ep)

        done = False
        ep_reward = 0.0
        while not done:
            action_idx = int(torch.randint(0, env.action_space.n, size=(1,), generator=rng).item())
            _, reward, terminated, truncated, _ = env.step(action_idx)
            ep_reward += float(reward)
            done = bool(terminated or truncated)

        successes += int(ep_reward > 0.5)

    return successes / max(1, episodes)


def _evaluate_tmaze_advantage(
    model: GaussianRSSM,
    attractor: AttractorDynamics | None,
    eval_seeds: list[int],
    episodes: int,
    device: torch.device,
    use_sigma: bool = False,
) -> tuple[list[float], float, float]:
    values: list[float] = []
    for seed in eval_seeds:
        random_success = _evaluate_tmaze_random(episodes=episodes, seed=seed + 100)
        epi_success = _evaluate_tmaze_policy(
            model=model,
            attractor=attractor,
            episodes=episodes,
            device=device,
            epistemic=True,
            seed=seed,
            use_sigma=use_sigma,
        )
        values.append(epi_success - random_success)

    _, _, ci_low, ci_high = _ci95(values)
    return values, ci_low, ci_high


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


def train_phase6_reliability(
    world_model: GaussianRSSM,
    attractor: AttractorDynamics,
    nis: NISMacroState,
    macro_transition: MacroTransition,
    controller: JOOTSController,
    obs_seq: torch.Tensor,
    act_seq: torch.Tensor,
    phase_cfg: Phase6Config,
    optim_cfg: OptimizationConfig,
    device: torch.device,
    checkpoint_dir: Path | None = None,
    logger_cfg: LoggerConfig | None = None,
    valid_mask_seq: torch.Tensor | None = None,
) -> Phase6Result:
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

    # Phase 6 assumes loop-closure context is active.
    world_model.set_macro_feedback(True)

    logger = MetricLogger(logger_cfg) if logger_cfg is not None else None

    stage_results: list[Phase6StageResult] = []
    proxy_history: list[float] = []
    audit_history: list[float] = []
    total_jooots_triggers = 0

    prev_stage_median_vfe: float | None = None
    prev_stage_tmaze_ci_low: float | None = None

    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for stage_index, sigma_weight in enumerate(phase_cfg.sigma_ramp_weights):
        world_model.set_sigma_prior_weight(sigma_weight)
        stage_start_state = {k: v.detach().clone() for k, v in world_model.state_dict().items()}

        stage_lr = optim_cfg.lr_l1
        attempts = 0
        final_stage: Phase6StageResult | None = None

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
                        sigma_feedback = attractor(micro_feedback_flat).reshape(micro_feedback.shape[0], micro_feedback.shape[1], -1)
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
                            f"Phase6/Stage{stage_index}/Loss_VFE": stage_vfe[-1],
                            f"Phase6/Stage{stage_index}/SigmaWeight": float(sigma_weight),
                            f"Phase6/Stage{stage_index}/Residual": float(residual_error),
                            f"Phase6/Stage{stage_index}/JOOTSSeverity": float(severity),
                        },
                        step,
                    )

            median_vfe = float(torch.tensor(stage_vfe, dtype=torch.float32).median().item())
            vals_sigma, tmaze_ci_low, tmaze_ci_high = _evaluate_tmaze_advantage(
                world_model,
                attractor=attractor,
                eval_seeds=phase_cfg.tmaze_eval_seeds,
                episodes=phase_cfg.tmaze_eval_episodes,
                device=device,
                use_sigma=True,
            )

            current_sigma_weight = world_model.sigma_prior_weight
            world_model.set_sigma_prior_weight(0.0)
            vals_no_sigma, tmaze_ci_low_no_sigma, tmaze_ci_high_no_sigma = _evaluate_tmaze_advantage(
                world_model,
                attractor=attractor,
                eval_seeds=phase_cfg.tmaze_eval_seeds,
                episodes=phase_cfg.tmaze_eval_episodes,
                device=device,
                use_sigma=False,
            )
            world_model.set_sigma_prior_weight(current_sigma_weight)

            tmaze_adv_delta = float(torch.tensor(vals_sigma).mean().item() - torch.tensor(vals_no_sigma).mean().item())

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
                "tmaze_ci_non_regression": (
                    True
                    if prev_stage_tmaze_ci_low is None
                    else tmaze_ci_low >= (prev_stage_tmaze_ci_low - phase_cfg.max_tmaze_ci_drop)
                ),
                "dei_audit_positive": (True if not stage_audit else audit_gate_value > 0.0),
                "proxy_audit_correlation": (
                    True if len(stage_proxy) < 3 else (corr >= phase_cfg.min_proxy_audit_spearman)
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
                            "tmaze_ci_low": tmaze_ci_low,
                            "tmaze_ci_high": tmaze_ci_high,
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

            final_stage = Phase6StageResult(
                stage_index=stage_index,
                sigma_weight=float(sigma_weight),
                attempts=attempts,
                passed=passed,
                rolled_back=rolled_back,
                median_vfe=median_vfe,
                tmaze_ci_low=tmaze_ci_low,
                tmaze_ci_high=tmaze_ci_high,
                tmaze_ci_low_no_sigma=tmaze_ci_low_no_sigma,
                tmaze_ci_high_no_sigma=tmaze_ci_high_no_sigma,
                tmaze_adv_delta_vs_no_sigma=tmaze_adv_delta,
                dei_audit_last=last_audit,
                proxy_audit_spearman=corr,
                gates=gates,
                checkpoint_path=checkpoint_path,
            )
            break

        if final_stage is None:
            raise RuntimeError(f"Stage {stage_index} failed to finalize")

        stage_results.append(final_stage)

        if final_stage.passed:
            prev_stage_median_vfe = final_stage.median_vfe
            prev_stage_tmaze_ci_low = final_stage.tmaze_ci_low

    if logger is not None:
        logger.close()

    return Phase6Result(
        stage_results=stage_results,
        all_passed=all(s.passed for s in stage_results),
        total_jooots_triggers=total_jooots_triggers,
        proxy_history=proxy_history,
        audit_history=audit_history,
    )

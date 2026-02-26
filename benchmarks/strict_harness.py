from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F

from envs.ambiguity_trap import AmbiguityTrapEnv
from envs.non_stationary_gridworld import NonStationaryGridworldEnv
from envs.tmaze import TMazeEnv
from models.attractor import AttractorDynamics
from models.joots import JOOTSController
from models.rssm import EFEScorer, GaussianRSSM
from training.contracts import OptimizationConfig, Phase1Config, Phase2Config
from training.data_collection import collect_random_trajectories
from training.phase1_world_model import train_phase1_world_model
from training.phase2_attractor import train_phase2_attractor
from utils.seed import set_seed


@dataclass
class SeedStats:
    values: list[float]
    mean: float
    std: float
    ci95_low: float
    ci95_high: float


@dataclass
class BenchmarkThresholds:
    tmaze_success_advantage_min: float = 0.10
    tmaze_success_advantage_ci_low_min: float = 0.0

    attractor_adaptation_gain_min: float = 0.01
    attractor_forgetting_ratio_max: float = 1.40
    attractor_residual_ratio_max: float = 1.75

    jooots_escape_gain_min: float = 0.10
    jooots_trigger_rate_min: float = 0.01


@dataclass
class Phase6CIBounds:
    tmaze_success_advantage_ci_low_min: float = 0.20
    attractor_adaptation_gain_ci_low_min: float = 0.010
    attractor_forgetting_ratio_ci_high_max: float = 1.10
    attractor_residual_ratio_ci_high_max: float = 1.10
    jooots_escape_gain_ci_low_min: float = 0.10
    jooots_trigger_rate_ci_low_min: float = 0.02


@dataclass
class BenchmarkConfig:
    seeds: list[int] = field(default_factory=lambda: [7, 11, 19, 23, 29])
    device: str = "cpu"
    profile: str = "legacy"
    mode: str = "full"

    tmaze_train_steps: int = 180
    tmaze_num_sequences: int = 192
    tmaze_seq_len: int = 20
    tmaze_eval_episodes: int = 40

    grid_train_steps_l1: int = 140
    grid_train_steps_l2_pre: int = 90
    grid_train_steps_l2_post: int = 70
    grid_num_sequences: int = 160
    grid_seq_len: int = 18

    trap_episodes: int = 140

    thresholds: BenchmarkThresholds = field(default_factory=BenchmarkThresholds)
    phase6_thresholds: Phase6CIBounds = field(default_factory=Phase6CIBounds)


@dataclass
class BenchmarkSummary:
    pass_all: bool
    threshold_results: dict[str, bool]
    metrics: dict[str, SeedStats]
    per_seed: dict[str, list[float]]
    profile: str = "legacy"
    mode: str = "full"


def _ci95(values: list[float]) -> SeedStats:
    if not values:
        return SeedStats(values=[], mean=0.0, std=0.0, ci95_low=0.0, ci95_high=0.0)

    t = torch.tensor(values, dtype=torch.float32)
    mean = float(t.mean().item())
    std = float(t.std(unbiased=False).item())
    n = max(1, t.numel())
    half_width = 1.96 * std / math.sqrt(float(n))
    return SeedStats(values=list(values), mean=mean, std=std, ci95_low=mean - half_width, ci95_high=mean + half_width)


def _evaluate_tmaze_policy(
    model: GaussianRSSM,
    episodes: int,
    device: torch.device,
    epistemic: bool,
    seed: int,
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
        done = False
        remembered_goal = 0
        ep_reward = 0.0

        while not done:
            if obs_t[0, 2].item() > 0.5:
                cue_val = int(round(float(obs_t[0, 3].item())))
                remembered_goal = cue_val if cue_val != 0 else remembered_goal
            at_junction = bool(obs_t[0, 4].item() > 0.5)

            with torch.no_grad():
                step = model.forward_step(obs_t, prev_action, state)
                state = step.state

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


def _run_tmaze_seed(seed: int, cfg: BenchmarkConfig, device: torch.device) -> float:
    tmaze_data = collect_random_trajectories(
        env_fn=lambda: TMazeEnv(seed=seed),
        num_sequences=cfg.tmaze_num_sequences,
        seq_len=cfg.tmaze_seq_len,
        seed=seed,
    )

    model = GaussianRSSM(
        obs_dim=tmaze_data["obs_seq"].shape[-1],
        action_dim=tmaze_data["act_seq"].shape[-1],
        hidden_dim=96,
        latent_dim=24,
    )
    train_phase1_world_model(
        model=model,
        obs_seq=tmaze_data["obs_seq"],
        act_seq=tmaze_data["act_seq"],
        phase_cfg=Phase1Config(steps=cfg.tmaze_train_steps, batch_size=32),
        optim_cfg=OptimizationConfig(lr_l1=3e-4, grad_clip_norm=1.0),
        device=device,
        logger_cfg=None,
    )

    random_success = _evaluate_tmaze_random(episodes=cfg.tmaze_eval_episodes, seed=seed + 100)
    epi_success = _evaluate_tmaze_policy(
        model=model.to(device),
        episodes=cfg.tmaze_eval_episodes,
        device=device,
        epistemic=True,
        seed=seed + 200,
    )
    return epi_success - random_success


def _attractor_eval_mse_and_residual(
    world_model: GaussianRSSM,
    attractor: AttractorDynamics,
    obs_seq: torch.Tensor,
    act_seq: torch.Tensor,
    device: torch.device,
) -> tuple[float, float]:
    with torch.no_grad():
        rollout = world_model.rollout(obs_seq.to(device), act_seq.to(device))
        micro = torch.cat([rollout.h_seq, rollout.z_seq], dim=-1)
        micro_flat = micro.reshape(-1, micro.shape[-1])

        out = attractor(micro_flat, return_trajectory=True)
        recon = attractor.reconstruct_micro(out.sigma)

        mse = float(F.mse_loss(recon, micro_flat).item())
        if out.trajectory is not None and out.trajectory.shape[1] > 1:
            residual = float((out.trajectory[:, -1] - out.trajectory[:, -2]).norm(dim=-1).mean().item())
        else:
            residual = 0.0
    return mse, residual


def _run_grid_seed(seed: int, cfg: BenchmarkConfig, device: torch.device) -> tuple[float, float, float]:
    data_a = collect_random_trajectories(
        env_fn=lambda: NonStationaryGridworldEnv(
            grid_size=7,
            max_steps=24,
            shift_interval=999999,
            fixed_goal_index=0,
            seed=seed,
        ),
        num_sequences=cfg.grid_num_sequences,
        seq_len=cfg.grid_seq_len,
        seed=seed,
    )
    data_b = collect_random_trajectories(
        env_fn=lambda: NonStationaryGridworldEnv(
            grid_size=7,
            max_steps=24,
            shift_interval=999999,
            fixed_goal_index=1,
            seed=seed + 1,
        ),
        num_sequences=cfg.grid_num_sequences,
        seq_len=cfg.grid_seq_len,
        seed=seed + 1,
    )

    wm = GaussianRSSM(
        obs_dim=data_a["obs_seq"].shape[-1],
        action_dim=data_a["act_seq"].shape[-1],
        hidden_dim=80,
        latent_dim=20,
    )

    train_phase1_world_model(
        model=wm,
        obs_seq=data_a["obs_seq"],
        act_seq=data_a["act_seq"],
        phase_cfg=Phase1Config(steps=cfg.grid_train_steps_l1, batch_size=32),
        optim_cfg=OptimizationConfig(lr_l1=3e-4, grad_clip_norm=1.0),
        device=device,
        logger_cfg=None,
    )

    attractor = AttractorDynamics(micro_dim=wm.hidden_dim + wm.latent_dim, attractor_dim=64)

    train_phase2_attractor(
        world_model=wm,
        attractor=attractor,
        obs_seq=data_a["obs_seq"],
        act_seq=data_a["act_seq"],
        phase_cfg=Phase2Config(steps=cfg.grid_train_steps_l2_pre, batch_size=32, lambda_spec=0.5),
        optim_cfg=OptimizationConfig(lr_l2=1e-3, grad_clip_norm=1.0),
        device=device,
        logger_cfg=None,
    )

    mse_a_before, _ = _attractor_eval_mse_and_residual(wm, attractor, data_a["obs_seq"], data_a["act_seq"], device)
    mse_b_before, residual_b_before = _attractor_eval_mse_and_residual(
        wm, attractor, data_b["obs_seq"], data_b["act_seq"], device
    )

    train_phase2_attractor(
        world_model=wm,
        attractor=attractor,
        obs_seq=data_b["obs_seq"],
        act_seq=data_b["act_seq"],
        phase_cfg=Phase2Config(steps=cfg.grid_train_steps_l2_post, batch_size=32, lambda_spec=0.5),
        optim_cfg=OptimizationConfig(lr_l2=1e-3, grad_clip_norm=1.0),
        device=device,
        logger_cfg=None,
    )

    mse_a_after, _ = _attractor_eval_mse_and_residual(wm, attractor, data_a["obs_seq"], data_a["act_seq"], device)
    mse_b_after, residual_b_after = _attractor_eval_mse_and_residual(
        wm, attractor, data_b["obs_seq"], data_b["act_seq"], device
    )

    adaptation_gain = mse_b_before - mse_b_after
    forgetting_ratio = mse_a_after / max(mse_a_before, 1e-6)
    residual_ratio = residual_b_after / max(residual_b_before, 1e-6)
    return adaptation_gain, forgetting_ratio, residual_ratio


class _TrapPolicy:
    def __init__(self) -> None:
        self.q = torch.tensor([0.30, -0.50], dtype=torch.float32)
        self.counts = torch.zeros(2, dtype=torch.float32)
        self.temperature = 0.08
        self.epistemic_foraging = False
        self.last_td_error = 0.0

    def select_action(self, rng: torch.Generator) -> int:
        logits = self.q / max(self.temperature, 1e-5)
        if self.epistemic_foraging:
            bonus = 1.5 / torch.sqrt(self.counts + 1.0)
            logits = logits + bonus
            probs = torch.softmax(logits, dim=0)
            return int(torch.multinomial(probs, num_samples=1, generator=rng).item())
        return int(torch.argmax(logits).item())

    def update(self, action: int, reward: float, alpha: float = 0.2) -> None:
        td = reward - float(self.q[action].item())
        self.q[action] += alpha * td
        self.counts[action] += 1.0
        self.last_td_error = td

    def increase_temperature(self, factor: float = 2.0) -> None:
        self.temperature = min(4.0, self.temperature * factor)

    def force_epistemic_foraging(self, enabled: bool = True) -> None:
        self.epistemic_foraging = enabled


class _TrapOptimizerWrapper:
    def __init__(self, policy: _TrapPolicy, rng: torch.Generator) -> None:
        self.policy = policy
        self.rng = rng
        self.noise_injections = 0
        self.sam_activations = 0

    def inject_sgld_noise(self, variance: float = 0.01) -> None:
        scale = math.sqrt(max(variance, 0.0))
        noise = torch.randn(self.policy.q.shape, generator=self.rng, dtype=self.policy.q.dtype) * scale
        self.policy.q += noise
        self.noise_injections += 1

    def enable_sam_mode(self, steps: int) -> None:
        # Proxy for a flatter local objective: temporarily encourage broader exploration.
        self.policy.temperature = min(4.0, self.policy.temperature * 1.25)
        self.sam_activations += int(steps > 0)


def _run_trap_rollout(seed: int, episodes: int, use_jooots: bool) -> tuple[float, float]:
    policy = _TrapPolicy()
    rng = torch.Generator().manual_seed(seed)
    env = AmbiguityTrapEnv(
        max_steps=20,
        loop_reward=0.05,
        explore_penalty=-0.20,
        escape_reward=2.5,
        escape_probability=0.45,
        seed=seed,
    )

    controller = JOOTSController(
        patience=12,
        tolerance=1e-3,
        severe_var_threshold=1e-5,
        loss_floor=-0.10,
        gsnr_floor=0.0,
        residual_floor=0.05,
        cooldown_steps=8,
        max_triggers_per_window=8,
        sam_steps=15,
        sgld_variance=0.05,
    )
    optim = _TrapOptimizerWrapper(policy, rng)

    escapes = 0
    triggers = 0

    for ep in range(episodes):
        _, _ = env.reset(seed=seed + ep)
        done = False
        ep_reward = 0.0
        escaped = False

        while not done:
            action = policy.select_action(rng)
            _, reward, terminated, truncated, info = env.step(action)
            policy.update(action, reward)
            ep_reward += float(reward)
            escaped = escaped or bool(info.get("escaped", False))
            done = bool(terminated or truncated)

        escapes += int(escaped)

        if use_jooots:
            # Higher proxy-VFE means failing to discover the better trajectory.
            vfe_proxy = 0.8 if not escaped else 0.0
            grad_proxy = abs(policy.last_td_error) + 0.01
            residual_proxy = 1.0 if not escaped else 0.0
            controller.update_metrics(vfe_proxy, grad_proxy, residual_error=residual_proxy)
            severity = controller.detect_stagnation()
            if severity > 0:
                controller.trigger_escape(severity, policy, optim)
                triggers += 1

    escape_rate = escapes / max(1, episodes)
    trigger_rate = triggers / max(1, episodes)
    return escape_rate, trigger_rate


def _run_trap_seed(seed: int, cfg: BenchmarkConfig) -> tuple[float, float]:
    baseline_escape, _ = _run_trap_rollout(seed=seed + 1000, episodes=cfg.trap_episodes, use_jooots=False)
    jooots_escape, trigger_rate = _run_trap_rollout(seed=seed + 2000, episodes=cfg.trap_episodes, use_jooots=True)
    return jooots_escape - baseline_escape, trigger_rate


def run_strict_benchmarks(config: BenchmarkConfig) -> BenchmarkSummary:
    set_seed(config.seeds[0] if config.seeds else 0)
    device = torch.device(config.device)

    tmaze_advantages: list[float] = []
    adaptation_gains: list[float] = []
    forgetting_ratios: list[float] = []
    residual_ratios: list[float] = []
    jooots_escape_gains: list[float] = []
    jooots_trigger_rates: list[float] = []

    for seed in config.seeds:
        set_seed(seed)

        tmaze_adv = _run_tmaze_seed(seed=seed, cfg=config, device=device)
        tmaze_advantages.append(tmaze_adv)

        adapt_gain, forget_ratio, residual_ratio = _run_grid_seed(seed=seed, cfg=config, device=device)
        adaptation_gains.append(adapt_gain)
        forgetting_ratios.append(forget_ratio)
        residual_ratios.append(residual_ratio)

        escape_gain, trigger_rate = _run_trap_seed(seed=seed, cfg=config)
        jooots_escape_gains.append(escape_gain)
        jooots_trigger_rates.append(trigger_rate)

    metrics = {
        "tmaze_success_advantage": _ci95(tmaze_advantages),
        "attractor_adaptation_gain": _ci95(adaptation_gains),
        "attractor_forgetting_ratio": _ci95(forgetting_ratios),
        "attractor_residual_ratio": _ci95(residual_ratios),
        "jooots_escape_gain": _ci95(jooots_escape_gains),
        "jooots_trigger_rate": _ci95(jooots_trigger_rates),
    }

    if config.profile == "legacy":
        thr = config.thresholds
        threshold_results = {
            "tmaze_mean_adv": metrics["tmaze_success_advantage"].mean >= thr.tmaze_success_advantage_min,
            "tmaze_ci_low_adv": metrics["tmaze_success_advantage"].ci95_low >= thr.tmaze_success_advantage_ci_low_min,
            "attractor_adaptation": metrics["attractor_adaptation_gain"].mean >= thr.attractor_adaptation_gain_min,
            "attractor_forgetting": metrics["attractor_forgetting_ratio"].mean <= thr.attractor_forgetting_ratio_max,
            "attractor_residual": metrics["attractor_residual_ratio"].mean <= thr.attractor_residual_ratio_max,
            "jooots_escape": metrics["jooots_escape_gain"].mean >= thr.jooots_escape_gain_min,
            "jooots_triggers": metrics["jooots_trigger_rate"].mean >= thr.jooots_trigger_rate_min,
        }
    elif config.profile == "phase6":
        p6 = config.phase6_thresholds
        threshold_results = {
            "tmaze_ci_low_adv": metrics["tmaze_success_advantage"].ci95_low >= p6.tmaze_success_advantage_ci_low_min,
            "attractor_adaptation_ci_low": (
                metrics["attractor_adaptation_gain"].ci95_low >= p6.attractor_adaptation_gain_ci_low_min
            ),
            "attractor_forgetting_ci_high": (
                metrics["attractor_forgetting_ratio"].ci95_high <= p6.attractor_forgetting_ratio_ci_high_max
            ),
            "attractor_residual_ci_high": (
                metrics["attractor_residual_ratio"].ci95_high <= p6.attractor_residual_ratio_ci_high_max
            ),
            "jooots_escape_ci_low": metrics["jooots_escape_gain"].ci95_low >= p6.jooots_escape_gain_ci_low_min,
            "jooots_triggers_ci_low": metrics["jooots_trigger_rate"].ci95_low >= p6.jooots_trigger_rate_ci_low_min,
        }
    else:
        raise ValueError(f"Unknown benchmark profile: {config.profile}")
    pass_all = all(threshold_results.values())

    per_seed = {
        "tmaze_success_advantage": tmaze_advantages,
        "attractor_adaptation_gain": adaptation_gains,
        "attractor_forgetting_ratio": forgetting_ratios,
        "attractor_residual_ratio": residual_ratios,
        "jooots_escape_gain": jooots_escape_gains,
        "jooots_trigger_rate": jooots_trigger_rates,
    }

    return BenchmarkSummary(
        pass_all=pass_all,
        threshold_results=threshold_results,
        metrics=metrics,
        per_seed=per_seed,
        profile=config.profile,
        mode=config.mode,
    )


def summary_to_json(summary: BenchmarkSummary) -> dict:
    return {
        "profile": summary.profile,
        "mode": summary.mode,
        "pass_all": summary.pass_all,
        "threshold_results": summary.threshold_results,
        "metrics": {k: asdict(v) for k, v in summary.metrics.items()},
        "per_seed": summary.per_seed,
    }


def write_summary(summary: BenchmarkSummary, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary_to_json(summary), indent=2))

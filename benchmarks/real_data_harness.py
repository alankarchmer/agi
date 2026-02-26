from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from models.rssm import GaussianRSSM
from training.contracts import EvalConfig, InfraConfig, OptimizationConfig, Phase1Config
from training.datasets import EpisodeRecord, load_real_episodes, make_sequence_dataloader, split_episodes
from training.offline_online import train_world_model_offline
from training.real_eval import evaluate_real_data_gates
from utils.seed import set_seed


def _supported_profiles() -> tuple[str, ...]:
    return ("pusht", "legacy")


@dataclass
class SeedStats:
    values: list[float]
    mean: float
    std: float
    ci95_low: float
    ci95_high: float


@dataclass
class RealBenchmarkThresholds:
    nll_ci95_high_max: float = 2.0
    kl_ci95_low_min: float = 0.01
    kl_ci95_high_max: float = 5.0
    temporal_drift_ci95_high_max: float = 0.25
    ood_gap_ci95_low_min: float = -0.05
    gate_pass_rate_min: float = 0.8


@dataclass
class RealBenchmarkConfig:
    data_source: str = "data/episodes.npz"
    profile: str = "pusht"
    action_space_type: str = "discrete"
    action_dim: int = 3
    obs_likelihood: str = "gaussian"
    seeds: list[int] = field(default_factory=lambda: [7, 11, 19, 23, 29])
    device: str = "cpu"
    seq_len: int = 32
    batch_size: int = 32
    stride: int = 1
    train_steps: int = 600
    reference_checkpoint: str | None = None
    checkpoint_mode: str = "strict"  # strict|partial
    thresholds: RealBenchmarkThresholds | None = None
    eval_cfg: EvalConfig | None = None


@dataclass
class RealBenchmarkSummary:
    pass_all: bool
    threshold_results: dict[str, bool]
    metrics: dict[str, SeedStats]
    per_seed: dict[str, list[float]]
    profile: str
    reference_checkpoint: str | None = None


def thresholds_for_profile(profile: str) -> RealBenchmarkThresholds:
    if profile == "legacy":
        return RealBenchmarkThresholds(
            nll_ci95_high_max=2.0,
            kl_ci95_low_min=0.01,
            kl_ci95_high_max=5.0,
            temporal_drift_ci95_high_max=0.25,
            ood_gap_ci95_low_min=-0.05,
            gate_pass_rate_min=0.8,
        )
    if profile == "pusht":
        # Calibrated for vector continuous-control trajectories like lerobot/pusht.
        return RealBenchmarkThresholds(
            nll_ci95_high_max=8000.0,
            kl_ci95_low_min=1.0,
            kl_ci95_high_max=120.0,
            temporal_drift_ci95_high_max=0.25,
            ood_gap_ci95_low_min=-0.20,
            gate_pass_rate_min=0.8,
        )
    raise ValueError(f"Unknown real benchmark profile: {profile!r}. Supported: {_supported_profiles()}")


def eval_cfg_for_profile(profile: str) -> EvalConfig:
    if profile == "legacy":
        return EvalConfig()
    if profile == "pusht":
        return EvalConfig(
            nll_max=8000.0,
            kl_min=1.0,
            kl_max=120.0,
            ood_drift_max=0.25,
            intervention_trend_min_spearman=0.30,
        )
    raise ValueError(f"Unknown real benchmark profile: {profile!r}. Supported: {_supported_profiles()}")


def _ci95(values: list[float]) -> SeedStats:
    if not values:
        return SeedStats(values=[], mean=0.0, std=0.0, ci95_low=0.0, ci95_high=0.0)
    t = torch.tensor(values, dtype=torch.float32)
    mean = float(t.mean().item())
    std = float(t.std(unbiased=False).item())
    n = max(1, t.numel())
    half_width = 1.96 * std / math.sqrt(float(n))
    return SeedStats(values=list(values), mean=mean, std=std, ci95_low=mean - half_width, ci95_high=mean + half_width)


def _obs_stats(episodes: list[EpisodeRecord]) -> tuple[torch.Tensor, torch.Tensor]:
    obs = torch.cat([ep.obs for ep in episodes], dim=0)
    return obs.mean(dim=0), obs.std(dim=0).clamp_min(1e-3)


def _action_stats(episodes: list[EpisodeRecord]) -> tuple[torch.Tensor, torch.Tensor]:
    act = torch.cat([ep.action for ep in episodes], dim=0)
    return act.mean(dim=0), act.std(dim=0).clamp_min(1e-3)


def _extract_checkpoint_state(payload: object) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict):
        if "world_model" in payload and isinstance(payload["world_model"], dict):
            return payload["world_model"]
        if "model" in payload and isinstance(payload["model"], dict):
            return payload["model"]
        if payload and all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in payload.items()):
            return payload  # raw state dict
    raise ValueError("Unsupported checkpoint format; expected dict with world_model/model state dict.")


def _infer_checkpoint_model_config(state: dict[str, torch.Tensor]) -> dict[str, int | str | None]:
    obs_dim = int(state["obs_mean"].numel()) if "obs_mean" in state else None
    action_dim = int(state["action_mean"].numel()) if "action_mean" in state else None

    hidden_dim = None
    latent_dim = None
    if "gru_cell.bias_ih" in state:
        hidden_dim = int(state["gru_cell.bias_ih"].numel() // 3)
    if "prior_head.2.bias" in state:
        latent_dim = int(state["prior_head.2.bias"].numel() // 2)

    sigma_dim = None
    if hidden_dim is not None and "prior_head.0.weight" in state:
        sigma_dim = int(state["prior_head.0.weight"].shape[1] - hidden_dim)

    macro_dim = None
    if obs_dim is not None and "obs_encoder.0.weight" in state:
        in_dim = int(state["obs_encoder.0.weight"].shape[1])
        macro_dim = max(0, in_dim - obs_dim)

    obs_likelihood = "gaussian" if "decoder_log_std" in state else None
    action_space_type = None
    if action_dim is not None and "policy_head.2.bias" in state:
        out_dim = int(state["policy_head.2.bias"].numel())
        if out_dim == action_dim:
            action_space_type = "discrete"
        elif out_dim == 2 * action_dim:
            action_space_type = "continuous"

    return {
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "hidden_dim": hidden_dim,
        "latent_dim": latent_dim,
        "sigma_dim": sigma_dim,
        "macro_dim": macro_dim,
        "obs_likelihood": obs_likelihood,
        "action_space_type": action_space_type,
    }


def _load_model_state(
    model: GaussianRSSM,
    state: dict[str, torch.Tensor],
    mode: str,
) -> tuple[int, int]:
    if mode == "strict":
        model.load_state_dict(state)
        return len(state), len(state)
    if mode != "partial":
        raise ValueError(f"Unknown checkpoint_mode: {mode}")

    target = model.state_dict()
    matched = {k: v for k, v in state.items() if k in target and target[k].shape == v.shape}
    if matched:
        target.update(matched)
        model.load_state_dict(target, strict=False)
    return len(matched), len(state)


def run_real_data_benchmarks(config: RealBenchmarkConfig) -> RealBenchmarkSummary:
    if config.profile not in _supported_profiles():
        raise ValueError(f"Unknown real benchmark profile: {config.profile!r}. Supported: {_supported_profiles()}")

    episodes = load_real_episodes(
        config.data_source,
        action_space_type=config.action_space_type,
        action_dim=config.action_dim,
    )
    thresholds = config.thresholds if config.thresholds is not None else thresholds_for_profile(config.profile)
    eval_cfg = config.eval_cfg if config.eval_cfg is not None else eval_cfg_for_profile(config.profile)
    device = torch.device(config.device)

    checkpoint_state: dict[str, torch.Tensor] | None = None
    inferred_from_checkpoint: dict[str, int | str | None] = {}
    if config.reference_checkpoint:
        payload = torch.load(config.reference_checkpoint, map_location="cpu")
        checkpoint_state = _extract_checkpoint_state(payload)
        inferred_from_checkpoint = _infer_checkpoint_model_config(checkpoint_state)

    nll_vals: list[float] = []
    kl_vals: list[float] = []
    drift_vals: list[float] = []
    ood_gap_vals: list[float] = []
    gate_pass_vals: list[float] = []

    for seed in config.seeds:
        set_seed(seed)
        train_eps, val_eps, test_eps = split_episodes(episodes, seed=seed)
        eval_eps = test_eps if test_eps else val_eps
        if not train_eps or not eval_eps:
            raise ValueError("Need non-empty train and eval splits for real benchmark harness.")

        obs_dim = int(train_eps[0].obs.shape[-1])
        act_dim = int(train_eps[0].action.shape[-1])

        train_loader = make_sequence_dataloader(
            train_eps,
            seq_len=config.seq_len,
            batch_size=config.batch_size,
            stride=config.stride,
            shuffle=True,
            include_partial=True,
            drop_last=True,
        )
        val_loader = make_sequence_dataloader(
            eval_eps,
            seq_len=config.seq_len,
            batch_size=config.batch_size,
            stride=config.stride,
            shuffle=False,
            include_partial=True,
            drop_last=False,
        )

        hidden_dim = int(inferred_from_checkpoint.get("hidden_dim") or 96)
        latent_dim = int(inferred_from_checkpoint.get("latent_dim") or 24)
        sigma_dim = int(inferred_from_checkpoint.get("sigma_dim") or 0)
        macro_dim = int(inferred_from_checkpoint.get("macro_dim") or 0)
        obs_likelihood = str(inferred_from_checkpoint.get("obs_likelihood") or config.obs_likelihood)
        action_space_type = str(inferred_from_checkpoint.get("action_space_type") or config.action_space_type)

        ckpt_obs_dim = inferred_from_checkpoint.get("obs_dim")
        if ckpt_obs_dim is not None and int(ckpt_obs_dim) != obs_dim:
            raise ValueError(
                f"Checkpoint obs_dim={ckpt_obs_dim} does not match dataset obs_dim={obs_dim} for seed {seed}."
            )
        ckpt_action_dim = inferred_from_checkpoint.get("action_dim")
        if ckpt_action_dim is not None and int(ckpt_action_dim) != act_dim:
            raise ValueError(
                f"Checkpoint action_dim={ckpt_action_dim} does not match dataset action_dim={act_dim} for seed {seed}."
            )

        model = GaussianRSSM(
            obs_dim=obs_dim,
            action_dim=act_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            sigma_dim=sigma_dim,
            macro_dim=macro_dim,
            action_space_type=action_space_type,
            obs_likelihood=obs_likelihood,
            normalize_obs=True,
            normalize_action=(action_space_type == "continuous"),
        )

        if checkpoint_state is None:
            obs_mean, obs_std = _obs_stats(train_eps)
            act_mean, act_std = _action_stats(train_eps)
            model.set_normalization_stats(obs_mean=obs_mean, obs_std=obs_std, action_mean=act_mean, action_std=act_std)

            _ = train_world_model_offline(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                phase_cfg=Phase1Config(steps=config.train_steps, batch_size=config.batch_size),
                optim_cfg=OptimizationConfig(lr_l1=3e-4, grad_clip_norm=1.0),
                infra_cfg=InfraConfig(use_amp=False, grad_accum_steps=1, early_stop_patience=1000, checkpoint_every=1000),
                device=device,
                run_dir=None,
                logger_cfg=None,
            )
        else:
            _load_model_state(model, checkpoint_state, config.checkpoint_mode)

        eval_result = evaluate_real_data_gates(
            model=model.to(device),
            dataloader=val_loader,
            cfg=eval_cfg,
            device=device,
        )

        nll_vals.append(float(eval_result.metrics["nll_mean"]))
        kl_vals.append(float(eval_result.metrics["kl_mean"]))
        drift_vals.append(float(eval_result.metrics["temporal_drift"]))
        ood_gap_vals.append(float(eval_result.metrics["ood_gap"]))
        gate_pass_vals.append(1.0 if eval_result.pass_all else 0.0)

    metrics = {
        "nll_mean": _ci95(nll_vals),
        "kl_mean": _ci95(kl_vals),
        "temporal_drift": _ci95(drift_vals),
        "ood_gap": _ci95(ood_gap_vals),
        "gate_pass_rate": _ci95(gate_pass_vals),
    }

    thr = thresholds
    threshold_results = {
        "nll_ci95_high": metrics["nll_mean"].ci95_high <= thr.nll_ci95_high_max,
        "kl_ci95_low": metrics["kl_mean"].ci95_low >= thr.kl_ci95_low_min,
        "kl_ci95_high": metrics["kl_mean"].ci95_high <= thr.kl_ci95_high_max,
        "temporal_drift_ci95_high": metrics["temporal_drift"].ci95_high <= thr.temporal_drift_ci95_high_max,
        "ood_gap_ci95_low": metrics["ood_gap"].ci95_low >= thr.ood_gap_ci95_low_min,
        "gate_pass_rate": metrics["gate_pass_rate"].mean >= thr.gate_pass_rate_min,
    }
    pass_all = all(threshold_results.values())

    per_seed = {
        "nll_mean": nll_vals,
        "kl_mean": kl_vals,
        "temporal_drift": drift_vals,
        "ood_gap": ood_gap_vals,
        "gate_pass_rate": gate_pass_vals,
    }
    return RealBenchmarkSummary(
        pass_all=pass_all,
        threshold_results=threshold_results,
        metrics=metrics,
        per_seed=per_seed,
        profile=config.profile,
        reference_checkpoint=config.reference_checkpoint,
    )


def summary_to_json(summary: RealBenchmarkSummary) -> dict:
    return {
        "pass_all": summary.pass_all,
        "profile": summary.profile,
        "reference_checkpoint": summary.reference_checkpoint,
        "threshold_results": summary.threshold_results,
        "metrics": {k: asdict(v) for k, v in summary.metrics.items()},
        "per_seed": summary.per_seed,
    }


def write_summary(summary: RealBenchmarkSummary, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary_to_json(summary), indent=2))

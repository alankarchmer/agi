from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    obs_dim: int
    action_dim: int
    hidden_dim: int = 128
    latent_dim: int = 32
    attractor_dim: int = 64
    macro_dim: int = 16


@dataclass
class OptimizationConfig:
    lr_l1: float = 3e-4
    lr_l2: float = 1e-3
    lr_l3: float = 1e-4
    weight_decay: float = 0.0
    grad_clip_norm: float = 1.0


@dataclass
class Phase1Config:
    steps: int = 500
    batch_size: int = 32
    beta_start: float = 0.1
    beta_end: float = 1.0
    beta_anneal_frac: float = 0.2
    kl_balance: float = 1.0
    kl_free_nats: float = 0.0
    overshooting_horizon: int = 1
    overshooting_weight: float = 0.0


@dataclass
class Phase2Config:
    steps: int = 300
    batch_size: int = 32
    lambda_spec: float = 1.0
    target_radius: float = 0.95


@dataclass
class Phase3Config:
    steps: int = 300
    batch_size: int = 32
    alpha_dei: float = 0.1


@dataclass
class Phase4Config:
    steps: int = 300
    batch_size: int = 32
    alpha_dei: float = 0.05
    attr_loss_weight: float = 0.5
    macro_loss_weight: float = 0.5


@dataclass
class Phase5Config:
    steps: int = 300
    batch_size: int = 32
    beta: float = 1.0
    macro_align_weight: float = 0.25


@dataclass
class Phase6Config:
    sigma_ramp_weights: list[float] = field(default_factory=lambda: [0.0, 0.25, 0.5, 0.75, 1.0])
    sigma_stage_steps: int = 120
    batch_size: int = 32

    ei_audit_interval: int = 50
    ei_audit_samples: int = 256
    ei_interventions_per_dim: int = 16

    jooots_cooldown_steps: int = 25
    jooots_max_triggers_per_window: int = 4

    max_vfe_regression: float = 0.05
    max_tmaze_ci_drop: float = 0.05
    min_proxy_audit_spearman: float = 0.30

    tmaze_eval_seeds: list[int] = field(default_factory=lambda: [7, 11])
    tmaze_eval_episodes: int = 24


@dataclass
class EnvConfig:
    seq_len: int = 32
    num_sequences: int = 256
    max_steps: int = 64
    seed: int = 0


@dataclass
class DataConfig:
    source: str | None = None
    format: str = "auto"  # auto|npz|npz_dir|jsonl|csv|pt
    action_space_type: str = "discrete"  # discrete|continuous
    action_dim: int | None = None
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    seq_len: int = 32
    stride: int = 1
    batch_size: int = 32
    include_partial_windows: bool = True
    normalize_obs: bool = True
    normalize_action: bool = False


@dataclass
class OfflineOnlineConfig:
    offline_steps: int = 1000
    online_steps: int = 200
    online_fraction: float = 0.2
    replay_capacity_steps: int = 200000
    collect_steps_per_iter: int = 1000


@dataclass
class InfraConfig:
    use_amp: bool = False
    grad_accum_steps: int = 1
    early_stop_patience: int = 20
    early_stop_min_delta: float = 1e-4
    checkpoint_every: int = 100
    max_checkpoints: int = 5
    config_version: str = "phase6-realdata-v1"


@dataclass
class EvalConfig:
    nll_max: float = 2.0
    kl_min: float = 0.01
    kl_max: float = 5.0
    ood_drift_max: float = 0.25
    intervention_trend_min_spearman: float = 0.30


@dataclass
class LoggingConfig:
    log_dir: str = "runs/asl"
    use_wandb: bool = False
    wandb_project: str = "asl-v1"
    wandb_run_name: str | None = None


@dataclass
class ASLConfig:
    model: ModelConfig
    optim: OptimizationConfig = field(default_factory=OptimizationConfig)
    data: DataConfig = field(default_factory=DataConfig)
    offline_online: OfflineOnlineConfig = field(default_factory=OfflineOnlineConfig)
    infra: InfraConfig = field(default_factory=InfraConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    phase1: Phase1Config = field(default_factory=Phase1Config)
    phase2: Phase2Config = field(default_factory=Phase2Config)
    phase3: Phase3Config = field(default_factory=Phase3Config)
    phase4: Phase4Config = field(default_factory=Phase4Config)
    phase5: Phase5Config = field(default_factory=Phase5Config)
    phase6: Phase6Config = field(default_factory=Phase6Config)
    env: EnvConfig = field(default_factory=EnvConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    enable_macro_feedback: bool = False

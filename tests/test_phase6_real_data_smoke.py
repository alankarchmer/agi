from __future__ import annotations

import torch

from models.attractor import AttractorDynamics
from models.joots import JOOTSController
from models.nis import MacroTransition, NISMacroState
from models.rssm import GaussianRSSM
from training.contracts import EvalConfig, OptimizationConfig, Phase6Config
from training.datasets import EpisodeRecord, make_sequence_dataloader, stack_full_sequences
from training.phase6_real_data import train_phase6_real_data


def _make_episode(ep_id: int, steps: int = 12, obs_dim: int = 3, act_dim: int = 2) -> EpisodeRecord:
    obs = torch.randn(steps, obs_dim)
    action = torch.randn(steps, act_dim).clamp(-1.0, 1.0)
    reward = torch.randn(steps) * 0.01
    done = torch.zeros(steps)
    done[-1] = 1.0
    ts = torch.arange(steps, dtype=torch.float32)
    return EpisodeRecord(
        obs=obs,
        action=action,
        reward=reward,
        done=done,
        timestamp=ts,
        episode_id=f"ep_{ep_id}",
    )


def test_phase6_real_data_smoke(tmp_path) -> None:
    torch.manual_seed(5)
    train_eps = [_make_episode(i) for i in range(20)]
    eval_eps = [_make_episode(100 + i) for i in range(6)]

    stacked = stack_full_sequences(train_eps)
    obs_seq = stacked["obs_seq"]
    act_seq = stacked["act_seq"]
    valid_mask = stacked["valid_mask"]

    eval_loader = make_sequence_dataloader(
        episodes=eval_eps,
        seq_len=12,
        batch_size=4,
        shuffle=False,
        stride=1,
        include_partial=True,
        drop_last=False,
    )

    world_model = GaussianRSSM(
        obs_dim=3,
        action_dim=2,
        hidden_dim=32,
        latent_dim=8,
        sigma_dim=12,
        macro_dim=4,
        action_space_type="continuous",
        obs_likelihood="mse",
        normalize_obs=True,
        normalize_action=True,
    )

    obs_all = torch.cat([e.obs for e in train_eps], dim=0)
    act_all = torch.cat([e.action for e in train_eps], dim=0)
    world_model.set_normalization_stats(
        obs_mean=obs_all.mean(dim=0),
        obs_std=obs_all.std(dim=0).clamp_min(1e-3),
        action_mean=act_all.mean(dim=0),
        action_std=act_all.std(dim=0).clamp_min(1e-3),
    )

    attractor = AttractorDynamics(micro_dim=40, attractor_dim=12, settling_steps=6, tau=0.1)
    nis = NISMacroState(micro_dim=12, macro_dim=4, num_coupling_layers=2, hidden_dim=32)
    macro_transition = MacroTransition(macro_dim=4, hidden_dim=16)
    controller = JOOTSController(
        patience=8,
        cooldown_steps=4,
        max_triggers_per_window=2,
        severe_var_threshold=1e-8,
        loss_floor=1e-6,
        gsnr_floor=0.2,
        residual_floor=1e-4,
    )

    phase_cfg = Phase6Config(
        sigma_ramp_weights=[0.0, 0.5],
        sigma_stage_steps=4,
        batch_size=6,
        ei_audit_interval=1,
        ei_audit_samples=32,
        ei_interventions_per_dim=4,
        jooots_cooldown_steps=4,
        jooots_max_triggers_per_window=2,
        min_proxy_audit_spearman=-1.0,
    )
    optim_cfg = OptimizationConfig(lr_l1=5e-4, grad_clip_norm=1.0)
    eval_cfg = EvalConfig(
        nll_max=1e6,
        kl_min=0.0,
        kl_max=1e6,
        ood_drift_max=10.0,
        intervention_trend_min_spearman=-1.0,
    )

    result = train_phase6_real_data(
        world_model=world_model,
        attractor=attractor,
        nis=nis,
        macro_transition=macro_transition,
        controller=controller,
        obs_seq=obs_seq,
        act_seq=act_seq,
        valid_mask_seq=valid_mask,
        eval_loader=eval_loader,
        phase_cfg=phase_cfg,
        optim_cfg=optim_cfg,
        eval_cfg=eval_cfg,
        device=torch.device("cpu"),
        checkpoint_dir=tmp_path / "stage_ckpts",
        logger_cfg=None,
    )

    assert len(result.stage_results) == 2
    assert isinstance(result.all_passed, bool)
    assert result.total_jooots_triggers >= 0
    assert (tmp_path / "stage_ckpts").exists()

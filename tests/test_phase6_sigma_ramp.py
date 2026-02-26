from __future__ import annotations

import torch

from models.attractor import AttractorDynamics
from models.joots import JOOTSController
from models.nis import MacroTransition, NISMacroState
from models.rssm import GaussianRSSM
from training.contracts import OptimizationConfig, Phase6Config
from training.phase6_reliability import train_phase6_reliability


def test_phase6_sigma_ramp_retry_and_rollback(monkeypatch) -> None:
    torch.manual_seed(5)
    device = torch.device("cpu")

    world_model = GaussianRSSM(obs_dim=5, action_dim=3, hidden_dim=24, latent_dim=8, sigma_dim=16, macro_dim=6)
    attractor = AttractorDynamics(micro_dim=32, attractor_dim=16, settling_steps=3, tau=0.2)
    nis = NISMacroState(micro_dim=16, macro_dim=6, num_coupling_layers=2, hidden_dim=32)
    macro_transition = MacroTransition(macro_dim=6, hidden_dim=16)
    controller = JOOTSController(patience=4, tolerance=1e-4)

    obs_seq = torch.randn(24, 6, 5)
    act_seq = torch.randn(24, 6, 3)

    call_idx = {"i": 0}

    def fake_eval(*args, **kwargs):
        call_idx["i"] += 1
        if call_idx["i"] == 1:
            return [0.40], 0.40, 0.45
        return [0.10], 0.10, 0.15

    monkeypatch.setattr("training.phase6_reliability._evaluate_tmaze_advantage", fake_eval)
    monkeypatch.setattr(
        "training.phase6_reliability.compute_dei_audit",
        lambda *args, **kwargs: torch.tensor(0.25, dtype=torch.float32),
    )

    phase_cfg = Phase6Config(
        sigma_ramp_weights=[0.0, 1.0],
        sigma_stage_steps=3,
        batch_size=8,
        ei_audit_interval=1,
        ei_audit_samples=32,
        ei_interventions_per_dim=4,
        tmaze_eval_seeds=[7],
        tmaze_eval_episodes=2,
        max_vfe_regression=0.05,
        max_tmaze_ci_drop=0.05,
        min_proxy_audit_spearman=-1.0,
    )

    result = train_phase6_reliability(
        world_model=world_model,
        attractor=attractor,
        nis=nis,
        macro_transition=macro_transition,
        controller=controller,
        obs_seq=obs_seq,
        act_seq=act_seq,
        phase_cfg=phase_cfg,
        optim_cfg=OptimizationConfig(lr_l1=3e-4, grad_clip_norm=1.0),
        device=device,
        checkpoint_dir=None,
        logger_cfg=None,
    )

    assert len(result.stage_results) == 2
    stage0, stage1 = result.stage_results
    assert stage0.passed is True
    assert stage1.attempts == 2
    assert stage1.passed is False
    assert stage1.rolled_back is True

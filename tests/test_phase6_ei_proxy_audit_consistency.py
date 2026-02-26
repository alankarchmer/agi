from __future__ import annotations

import types

import torch

from models.attractor import AttractorDynamics
from models.joots import JOOTSController
from models.nis import MacroTransition, NISMacroState
from models.rssm import GaussianRSSM
from training.contracts import OptimizationConfig, Phase6Config
from training.phase6_reliability import train_phase6_reliability


def test_phase6_proxy_audit_correlation_gate(monkeypatch) -> None:
    torch.manual_seed(6)
    device = torch.device("cpu")

    world_model = GaussianRSSM(obs_dim=5, action_dim=3, hidden_dim=24, latent_dim=8, sigma_dim=12, macro_dim=4)
    attractor = AttractorDynamics(micro_dim=32, attractor_dim=12, settling_steps=3, tau=0.2)
    nis = NISMacroState(micro_dim=12, macro_dim=4, num_coupling_layers=2, hidden_dim=24)
    macro_transition = MacroTransition(macro_dim=4, hidden_dim=16)
    controller = JOOTSController(patience=4, tolerance=1e-4)

    obs_seq = torch.randn(20, 6, 5)
    act_seq = torch.randn(20, 6, 3)

    monkeypatch.setattr(
        "training.phase6_reliability._evaluate_tmaze_advantage",
        lambda *args, **kwargs: ([0.3], 0.3, 0.35),
    )

    counter = {"i": 0}

    def fake_proxy(self, y_seq):
        counter["i"] += 1
        return torch.tensor(float(counter["i"]), dtype=y_seq.dtype, device=y_seq.device)

    def fake_audit(y_seq, *args, **kwargs):
        return torch.tensor(float(counter["i"]) + 0.5, dtype=y_seq.dtype, device=y_seq.device)

    monkeypatch.setattr(nis, "compute_dei_proxy", types.MethodType(fake_proxy, nis))
    monkeypatch.setattr("training.phase6_reliability.compute_dei_audit", fake_audit)

    phase_cfg = Phase6Config(
        sigma_ramp_weights=[0.0],
        sigma_stage_steps=6,
        batch_size=8,
        ei_audit_interval=1,
        ei_audit_samples=32,
        ei_interventions_per_dim=4,
        tmaze_eval_seeds=[7],
        tmaze_eval_episodes=2,
        min_proxy_audit_spearman=0.3,
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

    assert len(result.proxy_history) >= 3
    assert len(result.proxy_history) == len(result.audit_history)
    stage = result.stage_results[0]
    assert stage.gates["proxy_audit_correlation"] is True
    assert stage.proxy_audit_spearman >= 0.9

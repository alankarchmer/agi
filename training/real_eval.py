from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch.utils.data import DataLoader

from models.rssm import GaussianRSSM
from training.contracts import EvalConfig
from training.phase6_reliability import spearman_corr


@dataclass
class RealEvalResult:
    pass_all: bool
    gates: dict[str, bool]
    metrics: dict[str, float]

    def to_dict(self) -> dict:
        return asdict(self)


def _batch_to_device(batch: dict, device: torch.device) -> dict:
    out: dict = {}
    for k, v in batch.items():
        out[k] = v.to(device) if isinstance(v, torch.Tensor) else v
    return out


@torch.no_grad()
def evaluate_real_data_gates(
    model: GaussianRSSM,
    dataloader: DataLoader,
    cfg: EvalConfig,
    device: torch.device,
    intervention_audit_history: list[float] | None = None,
) -> RealEvalResult:
    model = model.to(device)
    model.eval()

    nll_values: list[float] = []
    kl_values: list[float] = []
    nll_ood_values: list[float] = []
    time_slices: list[tuple[float, float]] = []

    for batch in dataloader:
        b = _batch_to_device(batch, device)
        out = model.compute_vfe_loss(b["obs_seq"], b["act_seq"], beta=1.0, valid_mask=b.get("valid_mask"))
        nll_values.append(float(out["nll"].item()))
        kl_values.append(float(out["kl"].item()))

        obs = b["obs_seq"]
        noise = 0.05 * torch.randn_like(obs)
        obs_ood = obs + noise
        out_ood = model.compute_vfe_loss(obs_ood, b["act_seq"], beta=1.0, valid_mask=b.get("valid_mask"))
        nll_ood_values.append(float(out_ood["nll"].item()))

        if "valid_mask" in b:
            m = b["valid_mask"]
            if m.shape[1] >= 4:
                q = m.shape[1] // 4
                first_mask = m[:, :q]
                last_mask = m[:, -q:]
                first = model.compute_vfe_loss(b["obs_seq"][:, :q], b["act_seq"][:, :q], beta=1.0, valid_mask=first_mask)["nll"]
                last = model.compute_vfe_loss(b["obs_seq"][:, -q:], b["act_seq"][:, -q:], beta=1.0, valid_mask=last_mask)["nll"]
                time_slices.append((float(first.item()), float(last.item())))

    nll_mean = float(torch.tensor(nll_values, dtype=torch.float32).mean().item()) if nll_values else float("inf")
    kl_mean = float(torch.tensor(kl_values, dtype=torch.float32).mean().item()) if kl_values else float("inf")
    ood_nll_mean = float(torch.tensor(nll_ood_values, dtype=torch.float32).mean().item()) if nll_ood_values else float("inf")
    ood_gap = ood_nll_mean - nll_mean

    if time_slices:
        first_mean = float(torch.tensor([x for x, _ in time_slices], dtype=torch.float32).mean().item())
        last_mean = float(torch.tensor([y for _, y in time_slices], dtype=torch.float32).mean().item())
        temporal_drift = abs(last_mean - first_mean) / max(abs(first_mean), 1e-6)
    else:
        temporal_drift = 0.0

    trend_spearman = float("nan")
    if intervention_audit_history is not None and len(intervention_audit_history) >= 3:
        trend_spearman = spearman_corr(list(range(len(intervention_audit_history))), intervention_audit_history)

    gates = {
        "nll_finite": bool(torch.isfinite(torch.tensor(nll_mean))) and nll_mean <= cfg.nll_max,
        "kl_range": cfg.kl_min <= kl_mean <= cfg.kl_max,
        "temporal_drift": temporal_drift <= cfg.ood_drift_max,
        "intervention_trend": (
            True if intervention_audit_history is None else trend_spearman >= cfg.intervention_trend_min_spearman
        ),
    }
    pass_all = all(gates.values())

    metrics = {
        "nll_mean": nll_mean,
        "kl_mean": kl_mean,
        "nll_ood_mean": ood_nll_mean,
        "ood_gap": ood_gap,
        "temporal_drift": temporal_drift,
        "intervention_trend_spearman": trend_spearman,
    }
    return RealEvalResult(pass_all=pass_all, gates=gates, metrics=metrics)


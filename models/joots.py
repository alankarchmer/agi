from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class JOOTSController:
    patience: int = 100
    tolerance: float = 1e-4
    severe_var_threshold: float = 1e-8
    loss_floor: float = 1e-4

    # Rigor gates
    gsnr_floor: float = 1.0
    residual_floor: float = 0.0

    # Trigger management
    cooldown_steps: int = 0
    max_triggers_per_window: int = 999
    recovery_window: int = 20

    # Escape settings
    sam_steps: int = 25
    sgld_variance: float = 1e-2

    vfe_history: list[float] = field(default_factory=list)
    gsnr_history: list[float] = field(default_factory=list)
    residual_history: list[float] = field(default_factory=list)

    _step_count: int = 0
    _cooldown_remaining: int = 0
    _trigger_steps: list[int] = field(default_factory=list)
    _improve_streak: int = 0
    _last_vfe: float | None = None

    def update_metrics(self, current_vfe: float, grad_norm: float, residual_error: float = 0.0) -> None:
        vfe = float(current_vfe)
        gsnr = float(grad_norm)
        residual = float(residual_error)

        self._step_count += 1

        self.vfe_history.append(vfe)
        self.gsnr_history.append(gsnr)
        self.residual_history.append(residual)

        if len(self.vfe_history) > self.patience:
            self.vfe_history.pop(0)
            self.gsnr_history.pop(0)
            self.residual_history.pop(0)

        if self._last_vfe is not None and vfe < (self._last_vfe - self.tolerance):
            self._improve_streak += 1
        else:
            self._improve_streak = 0
        self._last_vfe = vfe

    def _slope(self) -> float:
        y = torch.tensor(self.vfe_history, dtype=torch.float32)
        n = y.shape[0]
        if n < 2:
            return 0.0
        x = torch.arange(n, dtype=torch.float32)
        x_centered = x - x.mean()
        y_centered = y - y.mean()
        denom = x_centered.pow(2).sum() + 1e-12
        slope = (x_centered * y_centered).sum() / denom
        return float(slope.item())

    def _trigger_budget_available(self) -> bool:
        window = max(1, self.patience)
        threshold_step = self._step_count - window
        self._trigger_steps = [s for s in self._trigger_steps if s >= threshold_step]
        return len(self._trigger_steps) < self.max_triggers_per_window

    def detect_stagnation(self) -> int:
        if len(self.vfe_history) < self.patience:
            if self._cooldown_remaining > 0:
                self._cooldown_remaining -= 1
            return 0

        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            return 0

        y = torch.tensor(self.vfe_history, dtype=torch.float32)
        g = torch.tensor(self.gsnr_history, dtype=torch.float32)
        r = torch.tensor(self.residual_history, dtype=torch.float32)

        slope = self._slope()
        loss_var = float(y.var(unbiased=False).item())
        mean_loss = float(y.mean().item())
        mean_gsnr = float(g.mean().item())
        mean_residual = float(r.mean().item())

        mild = abs(slope) < self.tolerance and mean_loss > self.loss_floor
        if not mild:
            return 0

        severe = (
            loss_var < self.severe_var_threshold
            and slope > -self.tolerance
            and mean_gsnr < self.gsnr_floor
            and mean_residual > self.residual_floor
            and self._trigger_budget_available()
        )

        if severe:
            self._trigger_steps.append(self._step_count)
            self._cooldown_remaining = max(0, self.cooldown_steps)
            return 2
        return 1

    def apply_recovery(self, policy_module, consecutive_steps: int | None = None) -> bool:
        target = self.recovery_window if consecutive_steps is None else int(consecutive_steps)
        if self._improve_streak < max(1, target):
            return False

        policy_module.force_epistemic_foraging(False)
        self._cooldown_remaining = 0
        self._trigger_steps.clear()
        self._improve_streak = 0
        return True

    def trigger_escape(self, severity: int, policy_module, optimizer_wrapper) -> None:
        if severity == 1:
            policy_module.increase_temperature(factor=2.0)
            policy_module.force_epistemic_foraging(True)
        elif severity == 2:
            try:
                optimizer_wrapper.inject_sgld_noise(variance=self.sgld_variance, target_groups=("l1",))
                optimizer_wrapper.enable_sam_mode(steps=self.sam_steps, target_groups=("l1",))
            except TypeError:
                # Backward compatibility for wrappers that do not support group scoping.
                optimizer_wrapper.inject_sgld_noise(variance=self.sgld_variance)
                optimizer_wrapper.enable_sam_mode(steps=self.sam_steps)

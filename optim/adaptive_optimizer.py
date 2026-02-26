from __future__ import annotations

import math
from typing import Iterable

import torch


class AdaptiveOptimizer:
    """Adam wrapper with optional SAM mode and SGLD noise injection."""

    def __init__(
        self,
        params: Iterable,
        lr: float = 3e-4,
        betas: tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 0.0,
        sam_rho: float = 0.05,
    ) -> None:
        self.base_optimizer = torch.optim.Adam(params, lr=lr, betas=betas, weight_decay=weight_decay)
        self.sam_rho = sam_rho

        self.sam_steps_remaining = 0
        self.sam_target_groups: tuple[str, ...] | None = None

        self.sgld_variance = 0.0
        self.sgld_target_groups: tuple[str, ...] | None = None

    @property
    def param_groups(self):
        return self.base_optimizer.param_groups

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict:
        return {
            "base": self.base_optimizer.state_dict(),
            "sam_steps_remaining": self.sam_steps_remaining,
            "sam_target_groups": self.sam_target_groups,
            "sgld_variance": self.sgld_variance,
            "sgld_target_groups": self.sgld_target_groups,
            "sam_rho": self.sam_rho,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.base_optimizer.load_state_dict(state_dict["base"])
        self.sam_steps_remaining = int(state_dict.get("sam_steps_remaining", 0))

        stg = state_dict.get("sam_target_groups")
        self.sam_target_groups = tuple(stg) if stg is not None else None

        self.sgld_variance = float(state_dict.get("sgld_variance", 0.0))
        sng = state_dict.get("sgld_target_groups")
        self.sgld_target_groups = tuple(sng) if sng is not None else None

        self.sam_rho = float(state_dict.get("sam_rho", self.sam_rho))

    @staticmethod
    def _normalize_target_groups(target_groups: tuple[str, ...] | list[str] | None) -> tuple[str, ...] | None:
        if target_groups is None:
            return None
        groups = tuple(str(g) for g in target_groups)
        if not groups or "all" in groups:
            return None
        return groups

    @staticmethod
    def _matches_group(group: dict, target_groups: tuple[str, ...] | None) -> bool:
        if target_groups is None:
            return True
        name = str(group.get("name", ""))
        return name in target_groups

    def enable_sam_mode(self, steps: int, target_groups: tuple[str, ...] = ("all",)) -> None:
        self.sam_steps_remaining = max(self.sam_steps_remaining, int(steps))
        self.sam_target_groups = self._normalize_target_groups(target_groups)

    def inject_sgld_noise(self, variance: float = 0.01, target_groups: tuple[str, ...] = ("all",)) -> None:
        self.sgld_variance = max(0.0, float(variance))
        self.sgld_target_groups = self._normalize_target_groups(target_groups)

    def _iter_params(self, target_groups: tuple[str, ...] | None = None):
        for group in self.base_optimizer.param_groups:
            if not self._matches_group(group, target_groups):
                continue
            for p in group["params"]:
                yield p

    def _grad_norm(self, target_groups: tuple[str, ...] | None = None) -> torch.Tensor:
        norms = []
        for p in self._iter_params(target_groups):
            if p.grad is not None:
                norms.append(p.grad.norm(2))
        if not norms:
            return torch.tensor(0.0)
        return torch.norm(torch.stack(norms), 2)

    def _inject_sgld_grad_noise(self) -> None:
        if self.sgld_variance <= 0.0:
            return

        scale = math.sqrt(self.sgld_variance)
        for p in self._iter_params(self.sgld_target_groups):
            if p.grad is not None:
                p.grad.add_(torch.randn_like(p.grad) * scale)

    def step(self, loss_closure=None):
        if self.sam_steps_remaining > 0:
            if loss_closure is None:
                raise ValueError("SAM mode requires a loss closure")

            loss_closure()

            first_param = next(self._iter_params(self.sam_target_groups), None)
            if first_param is None:
                self._inject_sgld_grad_noise()
                self.base_optimizer.step()
                self.sam_steps_remaining -= 1
                return None

            grad_norm = self._grad_norm(self.sam_target_groups).to(first_param.device)
            if grad_norm.item() <= 0.0:
                self._inject_sgld_grad_noise()
                self.base_optimizer.step()
                self.sam_steps_remaining -= 1
                return None

            scale = self.sam_rho / (grad_norm + 1e-12)

            e_ws: list[tuple[torch.nn.Parameter, torch.Tensor]] = []
            with torch.no_grad():
                for p in self._iter_params(self.sam_target_groups):
                    if p.grad is None:
                        continue
                    e_w = p.grad * scale
                    p.add_(e_w)
                    e_ws.append((p, e_w))

            self.zero_grad(set_to_none=True)
            loss = loss_closure()
            self._inject_sgld_grad_noise()

            with torch.no_grad():
                for p, e_w in e_ws:
                    p.sub_(e_w)

            self.base_optimizer.step()
            self.sam_steps_remaining -= 1
            return loss

        loss = None
        if loss_closure is not None:
            loss = loss_closure()
        self._inject_sgld_grad_noise()
        self.base_optimizer.step()
        return loss

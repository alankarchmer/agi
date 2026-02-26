from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AttractorForwardOutput:
    sigma: torch.Tensor
    trajectory: torch.Tensor | None = None


class AttractorDynamics(nn.Module):
    def __init__(
        self,
        micro_dim: int,
        attractor_dim: int,
        settling_steps: int = 10,
        tau: float = 0.1,
    ) -> None:
        super().__init__()
        self.micro_dim = micro_dim
        self.attractor_dim = attractor_dim
        self.settling_steps = settling_steps
        self.tau = tau

        self.input_proj = nn.Linear(micro_dim, attractor_dim)
        self.J = nn.Parameter(torch.empty(attractor_dim, attractor_dim))
        self.readout = nn.Linear(attractor_dim, micro_dim)

        nn.init.orthogonal_(self.J)
        with torch.no_grad():
            self.J.mul_(0.90)

    def forward(self, x: torch.Tensor, return_trajectory: bool = False) -> torch.Tensor | AttractorForwardOutput:
        sigma = torch.zeros(x.shape[0], self.attractor_dim, device=x.device, dtype=x.dtype)
        traj = []

        drive = self.input_proj(x)
        for _ in range(self.settling_steps):
            recurrent = F.linear(sigma, self.J) / max(self.tau, 1e-6)
            sigma = torch.tanh(drive + recurrent)
            if return_trajectory:
                traj.append(sigma)

        if return_trajectory:
            return AttractorForwardOutput(sigma=sigma, trajectory=torch.stack(traj, dim=1))
        return sigma

    def reconstruct_micro(self, sigma_star: torch.Tensor) -> torch.Tensor:
        return self.readout(sigma_star)

    def jacobian(self, x: torch.Tensor) -> torch.Tensor:
        """Compute one-step Jacobian d sigma_{k+1} / d sigma_k at sigma=0 for a reference input."""
        if x.dim() != 2:
            raise ValueError("x must have shape (B, micro_dim)")

        x_ref = x[:1].detach()
        drive = self.input_proj(x_ref)
        sigma0 = torch.zeros(1, self.attractor_dim, device=x.device, dtype=x.dtype, requires_grad=True)

        def one_step(sigma: torch.Tensor) -> torch.Tensor:
            recurrent = F.linear(sigma, self.J) / max(self.tau, 1e-6)
            return torch.tanh(drive + recurrent)

        jac = torch.autograd.functional.jacobian(one_step, sigma0, vectorize=True)
        return jac.squeeze(0).squeeze(1)

    def spectral_radius(self) -> torch.Tensor:
        return self.spectral_radius_with_method(method="exact")

    def spectral_radius_with_method(self, method: str = "exact", power_iters: int = 25) -> torch.Tensor:
        if method == "exact":
            try:
                eigvals = torch.linalg.eigvals(self.J)
                return eigvals.abs().max().real
            except RuntimeError:
                # Some backends (notably MPS) may not support eigvals for all shapes/dtypes.
                eigvals = torch.linalg.eigvals(self.J.float().cpu())
                return eigvals.abs().max().real.to(self.J.device)

        if method == "power":
            v = torch.randn(self.attractor_dim, device=self.J.device, dtype=self.J.dtype)
            v = v / (v.norm() + 1e-12)
            for _ in range(max(2, int(power_iters))):
                v = self.J @ v
                v = v / (v.norm() + 1e-12)
            rayleigh = torch.dot(v, self.J @ v)
            return rayleigh.abs().real

        raise ValueError(f"Unknown spectral method: {method}")

    def get_spectral_loss(
        self,
        target_radius: float = 0.95,
        weight: float = 1.0,
        method: str = "auto",
        power_iters: int = 25,
    ) -> torch.Tensor:
        if method == "auto":
            method = "power" if self.attractor_dim > 256 else "exact"
        rho = self.spectral_radius_with_method(method=method, power_iters=power_iters)
        return weight * F.relu(rho - target_radius)

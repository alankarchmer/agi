from __future__ import annotations

import torch
import torch.nn as nn


class RealNVPCoupling(nn.Module):
    def __init__(self, dim: int, mask: torch.Tensor, hidden_dim: int = 128) -> None:
        super().__init__()
        self.dim = dim
        self.register_buffer("mask", mask.float())

        self.s_net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim),
        )
        self.t_net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        m = self.mask
        x_masked = x * m
        s = 0.8 * torch.tanh(self.s_net(x_masked)) * (1.0 - m)
        t = self.t_net(x_masked) * (1.0 - m)
        y = x_masked + (1.0 - m) * (x * torch.exp(s) + t)
        log_det = s.sum(dim=-1)
        return y, log_det

    def inverse(self, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        m = self.mask
        y_masked = y * m
        s = 0.8 * torch.tanh(self.s_net(y_masked)) * (1.0 - m)
        t = self.t_net(y_masked) * (1.0 - m)
        x = y_masked + (1.0 - m) * (y - t) * torch.exp(-s)
        log_det = -s.sum(dim=-1)
        return x, log_det


class RealNVP(nn.Module):
    def __init__(self, dim: int, num_coupling_layers: int = 4, hidden_dim: int = 128) -> None:
        super().__init__()
        self.dim = dim
        layers = []
        base_mask = torch.zeros(dim)
        base_mask[::2] = 1.0

        for i in range(num_coupling_layers):
            mask = base_mask if i % 2 == 0 else 1.0 - base_mask
            layers.append(RealNVPCoupling(dim=dim, mask=mask, hidden_dim=hidden_dim))

        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = x
        log_det_sum = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        for layer in self.layers:
            z, log_det = layer(z)
            log_det_sum = log_det_sum + log_det
        return z, log_det_sum

    def inverse(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = z
        log_det_sum = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)
        for layer in reversed(self.layers):
            x, log_det = layer.inverse(x)
            log_det_sum = log_det_sum + log_det
        return x, log_det_sum


class NISMacroState(nn.Module):
    def __init__(self, micro_dim: int, macro_dim: int, num_coupling_layers: int = 4, hidden_dim: int = 128) -> None:
        super().__init__()
        if macro_dim <= 0 or macro_dim > micro_dim:
            raise ValueError("macro_dim must be in [1, micro_dim]")
        self.micro_dim = micro_dim
        self.macro_dim = macro_dim
        self.noise_dim = micro_dim - macro_dim
        self.normalizing_flow = RealNVP(dim=micro_dim, num_coupling_layers=num_coupling_layers, hidden_dim=hidden_dim)

    def forward(self, attractor_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat = attractor_state.reshape(-1, attractor_state.shape[-1])
        transformed, log_det = self.normalizing_flow(flat)

        y = transformed[:, : self.macro_dim]
        z_noise = transformed[:, self.macro_dim :]

        out_shape = attractor_state.shape[:-1]
        y = y.view(*out_shape, self.macro_dim)
        if self.noise_dim > 0:
            z_noise = z_noise.view(*out_shape, self.noise_dim)
        else:
            z_noise = torch.zeros(*out_shape, 0, device=attractor_state.device, dtype=attractor_state.dtype)
        log_det = log_det.view(*out_shape)
        return y, z_noise, log_det

    def inverse(self, y: torch.Tensor, z_noise: torch.Tensor | None = None) -> torch.Tensor:
        out_shape = y.shape[:-1]
        flat_y = y.reshape(-1, y.shape[-1])

        if self.noise_dim > 0:
            if z_noise is None:
                z_noise = torch.zeros(*out_shape, self.noise_dim, device=y.device, dtype=y.dtype)
            flat_noise = z_noise.reshape(-1, z_noise.shape[-1])
            flat = torch.cat([flat_y, flat_noise], dim=-1)
        else:
            flat = flat_y

        x, _ = self.normalizing_flow.inverse(flat)
        return x.view(*out_shape, self.micro_dim)

    @staticmethod
    def _hutchinson_trace(mat: torch.Tensor, num_samples: int = 16) -> torch.Tensor:
        dim = mat.shape[0]
        device = mat.device
        dtype = mat.dtype
        trace_est = torch.zeros((), device=device, dtype=dtype)
        for _ in range(max(1, int(num_samples))):
            v = torch.randint(0, 2, (dim,), device=device, dtype=torch.int64).float()
            v = v * 2.0 - 1.0
            trace_est = trace_est + (v @ (mat @ v))
        return trace_est / float(max(1, int(num_samples)))

    @staticmethod
    def _lstsq_with_fallback(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        try:
            return torch.linalg.lstsq(x, y).solution
        except (NotImplementedError, RuntimeError) as exc:
            msg = str(exc).lower()
            should_fallback = (
                x.device.type == "mps"
                or "linalg_lstsq" in msg
                or "not currently implemented for the mps" in msg
            )
            if not should_fallback:
                raise

            x_cpu = x.to(dtype=torch.float32, device="cpu")
            y_cpu = y.to(dtype=torch.float32, device="cpu")
            sol_cpu = torch.linalg.lstsq(x_cpu, y_cpu).solution
            return sol_cpu.to(device=x.device, dtype=x.dtype)

    def compute_dei_proxy(
        self,
        y_seq: torch.Tensor,
        eps: float = 1e-5,
        estimator: str = "logdet",
        hutchinson_samples: int = 16,
    ) -> torch.Tensor:
        if y_seq.dim() != 3:
            raise ValueError("y_seq must have shape (B, T, D)")
        if y_seq.shape[1] < 2:
            return torch.zeros((), device=y_seq.device, dtype=y_seq.dtype)

        x = y_seq[:, :-1, :].reshape(-1, y_seq.shape[-1])
        y = y_seq[:, 1:, :].reshape(-1, y_seq.shape[-1])

        x = x - x.mean(dim=0, keepdim=True)
        y = y - y.mean(dim=0, keepdim=True)

        a = self._lstsq_with_fallback(x, y)
        y_hat = x @ a
        residual = y - y_hat

        n = max(residual.shape[0] - 1, 1)
        d = y_seq.shape[-1]
        eye = torch.eye(d, device=y_seq.device, dtype=y_seq.dtype)

        cov_eps = (residual.t() @ residual) / float(n) + eps * eye
        mat = a.t() @ a + eps * eye

        if estimator == "hutchinson":
            signal_trace = self._hutchinson_trace(mat, num_samples=hutchinson_samples)
            noise_trace = self._hutchinson_trace(cov_eps, num_samples=hutchinson_samples)
            return (signal_trace - noise_trace) / float(d)

        sign1, logdet1 = torch.linalg.slogdet(mat)
        sign2, logdet2 = torch.linalg.slogdet(cov_eps)
        if sign1 <= 0 or sign2 <= 0:
            return torch.zeros((), device=y_seq.device, dtype=y_seq.dtype)
        return (logdet1 - logdet2) / float(d)


class MacroTransition(nn.Module):
    def __init__(self, macro_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(macro_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, macro_dim),
        )

    def forward(self, y_t: torch.Tensor) -> torch.Tensor:
        return self.net(y_t)

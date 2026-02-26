from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


LOG_STD_MIN = -10.0
LOG_STD_MAX = 2.0


@dataclass
class DiagGaussianParams:
    mu: torch.Tensor
    log_std: torch.Tensor

    def clamped(self) -> "DiagGaussianParams":
        return DiagGaussianParams(self.mu, self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX))

    def std(self) -> torch.Tensor:
        return self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX).exp()

    def rsample(self) -> torch.Tensor:
        eps = torch.randn_like(self.mu)
        return self.mu + self.std() * eps

    def entropy(self) -> torch.Tensor:
        log_std = self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)
        return (log_std + 0.5 * math.log(2.0 * math.pi * math.e)).sum(dim=-1)


@dataclass
class RSSMState:
    h: torch.Tensor
    z: torch.Tensor
    sigma: torch.Tensor | None = None
    y_macro: torch.Tensor | None = None


@dataclass
class StepOutput:
    prior: DiagGaussianParams
    posterior: DiagGaussianParams
    state: RSSMState
    recon: torch.Tensor
    action_logits: torch.Tensor


@dataclass
class RolloutOutput:
    prior_mu: torch.Tensor
    prior_log_std: torch.Tensor
    post_mu: torch.Tensor
    post_log_std: torch.Tensor
    h_seq: torch.Tensor
    z_seq: torch.Tensor
    recon_seq: torch.Tensor
    recon_raw_seq: torch.Tensor
    action_logits_seq: torch.Tensor


class ConvObsEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int, image_hw: tuple[int, int]) -> None:
        super().__init__()
        h, w = image_hw
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
        )
        h2 = max(1, h // 4)
        w2 = max(1, w // 4)
        self.proj = nn.Linear(64 * h2 * w2, hidden_dim)
        self.h2 = h2
        self.w2 = w2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv(x)
        flat = feat.reshape(feat.shape[0], -1)
        return self.proj(flat)


class GaussianRSSM(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        latent_dim: int = 32,
        sigma_dim: int = 0,
        macro_dim: int = 0,
        obs_shape: tuple[int, ...] | None = None,
        obs_encoder_type: str = "mlp",
        action_space_type: str = "discrete",
        obs_likelihood: str = "mse",
        normalize_obs: bool = False,
        normalize_action: bool = False,
    ) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.latent_dim = int(latent_dim)
        self.sigma_dim = int(sigma_dim)
        self.macro_dim = int(macro_dim)
        self.obs_shape = obs_shape
        self.obs_encoder_type = obs_encoder_type
        self.action_space_type = action_space_type
        self.obs_likelihood = obs_likelihood
        self.normalize_obs = bool(normalize_obs)
        self.normalize_action = bool(normalize_action)

        self.enable_macro_feedback = False
        self.sigma_prior_weight = 1.0
        self.temperature = 1.0
        self.is_epistemic_foraging = False

        self.register_buffer("obs_mean", torch.zeros(self.obs_dim))
        self.register_buffer("obs_std", torch.ones(self.obs_dim))
        self.register_buffer("action_mean", torch.zeros(self.action_dim))
        self.register_buffer("action_std", torch.ones(self.action_dim))

        if self.obs_encoder_type not in {"mlp", "conv"}:
            raise ValueError(f"Unsupported obs_encoder_type: {self.obs_encoder_type}")
        if self.action_space_type not in {"discrete", "continuous"}:
            raise ValueError(f"Unsupported action_space_type: {self.action_space_type}")
        if self.obs_likelihood not in {"mse", "gaussian", "bernoulli"}:
            raise ValueError(f"Unsupported obs_likelihood: {self.obs_likelihood}")

        if self.obs_encoder_type == "conv":
            if self.obs_shape is None or len(self.obs_shape) != 3:
                raise ValueError("obs_shape=(C,H,W) is required for conv encoder.")
            c, h, w = self.obs_shape
            self.obs_encoder = ConvObsEncoder(in_channels=int(c), hidden_dim=self.hidden_dim, image_hw=(int(h), int(w)))
            self.macro_obs_proj = nn.Linear(self.macro_dim, self.hidden_dim) if self.macro_dim > 0 else None
        else:
            obs_input_dim = self.obs_dim + self.macro_dim
            self.obs_encoder = nn.Sequential(
                nn.Linear(obs_input_dim, self.hidden_dim),
                nn.ELU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ELU(),
            )
            self.macro_obs_proj = None

        self.gru_cell = nn.GRUCell(self.latent_dim + self.action_dim, self.hidden_dim)
        self.prior_head = nn.Sequential(
            nn.Linear(self.hidden_dim + self.sigma_dim, self.hidden_dim),
            nn.ELU(),
            nn.Linear(self.hidden_dim, self.latent_dim * 2),
        )
        self.post_head = nn.Sequential(
            nn.Linear(self.hidden_dim + self.hidden_dim, self.hidden_dim),
            nn.ELU(),
            nn.Linear(self.hidden_dim, self.latent_dim * 2),
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_dim + self.latent_dim, self.hidden_dim),
            nn.ELU(),
            nn.Linear(self.hidden_dim, self.obs_dim),
        )
        if self.obs_likelihood == "gaussian":
            self.decoder_log_std = nn.Parameter(torch.full((self.obs_dim,), -0.5))
        else:
            self.decoder_log_std = None

        if self.action_space_type == "discrete":
            self.policy_head = nn.Sequential(
                nn.Linear(self.hidden_dim + self.latent_dim, self.hidden_dim),
                nn.ELU(),
                nn.Linear(self.hidden_dim, self.action_dim),
            )
        else:
            self.policy_head = nn.Sequential(
                nn.Linear(self.hidden_dim + self.latent_dim, self.hidden_dim),
                nn.ELU(),
                nn.Linear(self.hidden_dim, self.action_dim * 2),
            )

        self.reward_head = nn.Sequential(
            nn.Linear(self.hidden_dim + self.latent_dim, self.hidden_dim),
            nn.ELU(),
            nn.Linear(self.hidden_dim, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(self.hidden_dim + self.latent_dim, self.hidden_dim),
            nn.ELU(),
            nn.Linear(self.hidden_dim, 1),
        )

    def set_macro_feedback(self, enabled: bool) -> None:
        self.enable_macro_feedback = enabled

    def set_sigma_prior_weight(self, weight: float) -> None:
        self.sigma_prior_weight = float(max(0.0, weight))

    def set_normalization_stats(
        self,
        obs_mean: torch.Tensor | None = None,
        obs_std: torch.Tensor | None = None,
        action_mean: torch.Tensor | None = None,
        action_std: torch.Tensor | None = None,
    ) -> None:
        if obs_mean is not None:
            self.obs_mean.copy_(obs_mean.reshape(-1)[: self.obs_dim].to(self.obs_mean.device, dtype=self.obs_mean.dtype))
        if obs_std is not None:
            self.obs_std.copy_(obs_std.reshape(-1)[: self.obs_dim].to(self.obs_std.device, dtype=self.obs_std.dtype).clamp_min(1e-6))
        if action_mean is not None:
            self.action_mean.copy_(
                action_mean.reshape(-1)[: self.action_dim].to(self.action_mean.device, dtype=self.action_mean.dtype)
            )
        if action_std is not None:
            self.action_std.copy_(
                action_std.reshape(-1)[: self.action_dim].to(self.action_std.device, dtype=self.action_std.dtype).clamp_min(1e-6)
            )

    def increase_temperature(self, factor: float = 2.0) -> None:
        self.temperature = max(1e-3, self.temperature * factor)

    def force_epistemic_foraging(self, enabled: bool = True) -> None:
        self.is_epistemic_foraging = enabled

    def init_state(self, batch_size: int, device: torch.device | None = None) -> RSSMState:
        dev = device if device is not None else next(self.parameters()).device
        h = torch.zeros(batch_size, self.hidden_dim, device=dev)
        z = torch.zeros(batch_size, self.latent_dim, device=dev)
        sigma = torch.zeros(batch_size, self.sigma_dim, device=dev) if self.sigma_dim > 0 else None
        y_macro = torch.zeros(batch_size, self.macro_dim, device=dev) if self.macro_dim > 0 else None
        return RSSMState(h=h, z=z, sigma=sigma, y_macro=y_macro)

    def _normalize_obs(self, obs: torch.Tensor) -> torch.Tensor:
        if not self.normalize_obs:
            return obs
        obs_flat = obs.reshape(obs.shape[0], -1)
        return (obs_flat - self.obs_mean.view(1, -1)) / self.obs_std.view(1, -1).clamp_min(1e-6)

    def _normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        if not (self.normalize_action and self.action_space_type == "continuous"):
            return action
        return (action - self.action_mean.view(1, -1)) / self.action_std.view(1, -1).clamp_min(1e-6)

    def _encode_obs(self, obs_t: torch.Tensor, y_macro: torch.Tensor | None) -> torch.Tensor:
        if self.obs_encoder_type == "conv":
            if obs_t.dim() == 2:
                if self.obs_shape is None:
                    raise ValueError("obs_shape is required for conv observation processing.")
                obs_t = obs_t.view(obs_t.shape[0], *self.obs_shape)
            if obs_t.dim() != 4:
                raise ValueError("conv observation encoder expects (B,C,H,W) input.")

            obs_embed = self.obs_encoder(obs_t.float())
            if self.macro_dim > 0 and self.enable_macro_feedback:
                if y_macro is None:
                    y_macro = torch.zeros(obs_embed.shape[0], self.macro_dim, device=obs_embed.device, dtype=obs_embed.dtype)
                obs_embed = obs_embed + self.macro_obs_proj(y_macro.float())
            return obs_embed

        obs = obs_t.float()
        if obs.dim() > 2:
            obs = obs.view(obs.shape[0], -1)
        obs = self._normalize_obs(obs)

        if self.macro_dim > 0:
            if self.enable_macro_feedback:
                if y_macro is None:
                    y_macro = torch.zeros(obs.shape[0], self.macro_dim, device=obs.device, dtype=obs.dtype)
                obs = torch.cat([obs, y_macro.float()], dim=-1)
            else:
                obs = torch.cat([obs, torch.zeros(obs.shape[0], self.macro_dim, device=obs.device, dtype=obs.dtype)], dim=-1)
        return self.obs_encoder(obs)

    def _decode_obs(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.decoder(features)
        if self.obs_likelihood == "bernoulli":
            recon = torch.sigmoid(raw)
        else:
            recon = raw
        return recon, raw

    def _split_gaussian(self, raw: torch.Tensor) -> DiagGaussianParams:
        mu, log_std = torch.chunk(raw, 2, dim=-1)
        return DiagGaussianParams(mu, log_std.clamp(LOG_STD_MIN, LOG_STD_MAX))

    def _compute_prior(self, h_t: torch.Tensor, sigma_prior: torch.Tensor | None) -> DiagGaussianParams:
        if self.sigma_dim > 0:
            if sigma_prior is None:
                sigma_prior = torch.zeros(h_t.shape[0], self.sigma_dim, device=h_t.device, dtype=h_t.dtype)
            sigma_prior = sigma_prior.detach() * self.sigma_prior_weight
            prior_in = torch.cat([h_t, sigma_prior], dim=-1)
        else:
            prior_in = h_t
        return self._split_gaussian(self.prior_head(prior_in))

    def _policy_params(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        raw = self.policy_head(features)
        if self.action_space_type == "discrete":
            return raw / max(self.temperature, 1e-6), None
        mean, log_std = torch.chunk(raw, 2, dim=-1)
        log_std = log_std.clamp(LOG_STD_MIN, LOG_STD_MAX) + math.log(max(self.temperature, 1e-6))
        return mean, log_std

    def forward_step(
        self,
        obs_t: torch.Tensor,
        prev_action: torch.Tensor,
        prev_state: RSSMState,
        sigma_prior: torch.Tensor | None = None,
        y_macro: torch.Tensor | None = None,
    ) -> StepOutput:
        if prev_action.dim() == 1:
            prev_action = prev_action.unsqueeze(-1)
        prev_action = prev_action.float()
        prev_action = self._normalize_action(prev_action)

        rnn_in = torch.cat([prev_state.z, prev_action], dim=-1)
        h_t = self.gru_cell(rnn_in, prev_state.h)

        prior = self._compute_prior(h_t, sigma_prior)

        obs_embed = self._encode_obs(obs_t, y_macro)
        post_in = torch.cat([h_t, obs_embed], dim=-1)
        posterior = self._split_gaussian(self.post_head(post_in))
        z_t = posterior.rsample()

        features = torch.cat([h_t, z_t], dim=-1)
        recon, _ = self._decode_obs(features)
        policy_primary, _ = self._policy_params(features)

        state = RSSMState(
            h=h_t,
            z=z_t,
            sigma=sigma_prior.detach() if sigma_prior is not None else prev_state.sigma,
            y_macro=y_macro.detach() if y_macro is not None else prev_state.y_macro,
        )
        return StepOutput(prior=prior, posterior=posterior, state=state, recon=recon, action_logits=policy_primary)

    def rollout(
        self,
        obs_seq: torch.Tensor,
        act_seq: torch.Tensor,
        init_state: RSSMState | None = None,
        use_posterior: bool = True,
        sigma_seq: torch.Tensor | None = None,
        y_seq: torch.Tensor | None = None,
    ) -> RolloutOutput:
        if obs_seq.dim() not in {3, 5}:
            raise ValueError("obs_seq must have shape (B,T,D) for vectors or (B,T,C,H,W) for images.")
        if act_seq.dim() != 3:
            raise ValueError("act_seq must have shape (B,T,A)")

        bsz, seq_len = obs_seq.shape[0], obs_seq.shape[1]
        device = obs_seq.device
        state = init_state if init_state is not None else self.init_state(bsz, device=device)

        prior_mu, prior_log_std = [], []
        post_mu, post_log_std = [], []
        h_seq, z_seq = [], []
        recon_seq, recon_raw_seq, action_logits_seq = [], [], []

        prev_action = torch.zeros(bsz, self.action_dim, device=device)
        for t in range(seq_len):
            sigma_t = sigma_seq[:, t] if sigma_seq is not None else None
            y_t = y_seq[:, t] if y_seq is not None else None
            step = self.forward_step(obs_seq[:, t], prev_action, state, sigma_prior=sigma_t, y_macro=y_t)

            if use_posterior:
                state = step.state
                z_use = step.state.z
            else:
                z_use = step.prior.rsample()
                state = RSSMState(h=step.state.h, z=z_use, sigma=step.state.sigma, y_macro=step.state.y_macro)

            features = torch.cat([state.h, z_use], dim=-1)
            recon, recon_raw = self._decode_obs(features)
            policy_primary, _ = self._policy_params(features)

            prior_mu.append(step.prior.mu)
            prior_log_std.append(step.prior.log_std)
            post_mu.append(step.posterior.mu)
            post_log_std.append(step.posterior.log_std)
            h_seq.append(state.h)
            z_seq.append(z_use)
            recon_seq.append(recon)
            recon_raw_seq.append(recon_raw)
            action_logits_seq.append(policy_primary)

            prev_action = act_seq[:, t]

        return RolloutOutput(
            prior_mu=torch.stack(prior_mu, dim=1),
            prior_log_std=torch.stack(prior_log_std, dim=1),
            post_mu=torch.stack(post_mu, dim=1),
            post_log_std=torch.stack(post_log_std, dim=1),
            h_seq=torch.stack(h_seq, dim=1),
            z_seq=torch.stack(z_seq, dim=1),
            recon_seq=torch.stack(recon_seq, dim=1),
            recon_raw_seq=torch.stack(recon_raw_seq, dim=1),
            action_logits_seq=torch.stack(action_logits_seq, dim=1),
        )

    @staticmethod
    def _kl_diag_gaussian(
        mu_a: torch.Tensor,
        log_std_a: torch.Tensor,
        mu_b: torch.Tensor,
        log_std_b: torch.Tensor,
    ) -> torch.Tensor:
        log_std_a = log_std_a.clamp(LOG_STD_MIN, LOG_STD_MAX)
        log_std_b = log_std_b.clamp(LOG_STD_MIN, LOG_STD_MAX)
        var_a = (2.0 * log_std_a).exp()
        var_b = (2.0 * log_std_b).exp()

        kl = log_std_b - log_std_a
        kl = kl + (var_a + (mu_a - mu_b).pow(2)) / (2.0 * var_b)
        kl = kl - 0.5
        return kl.sum(dim=-1)

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        if mask is None:
            return values.mean()
        m = mask.float()
        denom = m.sum().clamp_min(1.0)
        return (values * m).sum() / denom

    def _recon_nll_per_step(
        self,
        recon_seq: torch.Tensor,
        recon_raw_seq: torch.Tensor,
        obs_seq: torch.Tensor,
    ) -> torch.Tensor:
        target = obs_seq.float()
        pred = recon_seq.float()
        pred_raw = recon_raw_seq.float()

        if target.dim() == 5:
            target = target.view(target.shape[0], target.shape[1], -1)
        if pred.dim() == 5:
            pred = pred.view(pred.shape[0], pred.shape[1], -1)
        if pred_raw.dim() == 5:
            pred_raw = pred_raw.view(pred_raw.shape[0], pred_raw.shape[1], -1)

        if self.obs_likelihood == "mse":
            nll = (pred - target).pow(2).mean(dim=-1)
        elif self.obs_likelihood == "gaussian":
            if self.decoder_log_std is None:
                raise RuntimeError("decoder_log_std must be initialized for gaussian likelihood.")
            log_std = self.decoder_log_std.clamp(LOG_STD_MIN, LOG_STD_MAX).view(1, 1, -1)
            var = (2.0 * log_std).exp()
            nll = 0.5 * (((target - pred_raw).pow(2) / var) + 2.0 * log_std + math.log(2.0 * math.pi)).mean(dim=-1)
        else:
            nll = F.binary_cross_entropy_with_logits(pred_raw, target.clamp(0.0, 1.0), reduction="none").mean(dim=-1)
        return nll

    def compute_vfe_loss(
        self,
        obs_seq: torch.Tensor,
        act_seq: torch.Tensor,
        beta: float = 1.0,
        sigma_seq: torch.Tensor | None = None,
        y_seq: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
        kl_balance: float = 1.0,
        kl_free_nats: float = 0.0,
        overshooting_horizon: int = 1,
        overshooting_weight: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        rollout = self.rollout(obs_seq, act_seq, sigma_seq=sigma_seq, y_seq=y_seq)

        nll_bt = self._recon_nll_per_step(rollout.recon_seq, rollout.recon_raw_seq, obs_seq)
        nll = self._masked_mean(nll_bt, valid_mask)
        recon = nll

        kl_forward_bt = self._kl_diag_gaussian(rollout.post_mu, rollout.post_log_std, rollout.prior_mu, rollout.prior_log_std)
        if kl_balance >= 0.999:
            kl_bt = kl_forward_bt
        else:
            kl_reverse_bt = self._kl_diag_gaussian(
                rollout.prior_mu.detach(),
                rollout.prior_log_std.detach(),
                rollout.post_mu.detach(),
                rollout.post_log_std.detach(),
            )
            alpha = float(max(0.0, min(1.0, kl_balance)))
            kl_bt = alpha * kl_forward_bt + (1.0 - alpha) * kl_reverse_bt
        if kl_free_nats > 0.0:
            kl_bt = torch.clamp(kl_bt, min=float(kl_free_nats))
        kl = self._masked_mean(kl_bt, valid_mask)

        overshoot = torch.zeros((), device=kl.device, dtype=kl.dtype)
        if overshooting_weight > 0.0 and overshooting_horizon > 1 and kl_bt.shape[1] > 1:
            horizon = int(min(overshooting_horizon, kl_bt.shape[1]))
            terms = []
            for k in range(2, horizon + 1):
                k_slice = kl_bt[:, k - 1 :]
                mask_k = None if valid_mask is None else valid_mask[:, k - 1 :]
                terms.append(self._masked_mean(k_slice, mask_k) / float(k))
            if terms:
                overshoot = torch.stack(terms).mean() * float(overshooting_weight)

        total = nll + beta * kl + overshoot
        if not torch.isfinite(total):
            raise FloatingPointError("Non-finite VFE loss detected")

        return {
            "total": total,
            "nll": nll,
            "kl": kl,
            "recon": recon,
            "overshoot_kl": overshoot,
            "log_std_min": torch.minimum(rollout.post_log_std.min(), rollout.prior_log_std.min()),
            "log_std_max": torch.maximum(rollout.post_log_std.max(), rollout.prior_log_std.max()),
        }

    def policy_logits(self, state: RSSMState) -> torch.Tensor:
        features = torch.cat([state.h, state.z], dim=-1)
        primary, _ = self._policy_params(features)
        return primary

    def policy_distribution(self, state: RSSMState) -> tuple[torch.Tensor, torch.Tensor | None]:
        features = torch.cat([state.h, state.z], dim=-1)
        return self._policy_params(features)

    def predict_reward(self, state: RSSMState) -> torch.Tensor:
        features = torch.cat([state.h, state.z], dim=-1)
        return self.reward_head(features).squeeze(-1)

    def predict_value(self, state: RSSMState) -> torch.Tensor:
        features = torch.cat([state.h, state.z], dim=-1)
        return self.value_head(features).squeeze(-1)

    def sample_action(self, state: RSSMState, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        if self.action_space_type == "discrete":
            logits = self.policy_logits(state)
            if deterministic:
                action_idx = logits.argmax(dim=-1)
            else:
                action_idx = torch.distributions.Categorical(logits=logits).sample()
            action = F.one_hot(action_idx, num_classes=self.action_dim).float()
            return action, action_idx

        mean, log_std = self.policy_distribution(state)
        assert log_std is not None
        std = log_std.exp()
        if deterministic:
            action = torch.tanh(mean)
        else:
            dist = torch.distributions.Normal(mean, std)
            action = torch.tanh(dist.rsample())
        return action, action


class EFEScorer:
    def __init__(
        self,
        model: GaussianRSSM,
        pragmatic_weight: float = 1.0,
        epistemic_weight: float = 1.0,
        value_weight: float = 0.2,
        discount: float = 0.99,
        planning_horizon: int = 1,
    ) -> None:
        self.model = model
        self.pragmatic_weight = float(pragmatic_weight)
        self.epistemic_weight = float(epistemic_weight)
        self.value_weight = float(value_weight)
        self.discount = float(discount)
        self.planning_horizon = int(max(1, planning_horizon))

    def _step_imagine(self, state: RSSMState, action: torch.Tensor) -> tuple[RSSMState, torch.Tensor, torch.Tensor]:
        h_next = self.model.gru_cell(torch.cat([state.z, action], dim=-1), state.h)
        prior = self.model._compute_prior(h_next, state.sigma)
        z_next = prior.mu
        next_state = RSSMState(h=h_next, z=z_next, sigma=state.sigma, y_macro=state.y_macro)
        reward = self.model.predict_reward(next_state)
        value = self.model.predict_value(next_state)
        return next_state, reward, value + prior.entropy()

    def score_actions(
        self,
        state: RSSMState,
        candidate_actions: torch.Tensor,
        target_obs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.planning_horizon > 1:
            return self.score_actions_horizon(state, candidate_actions, horizon=self.planning_horizon)

        if candidate_actions.dim() == 2:
            candidate_actions = candidate_actions.unsqueeze(0).expand(state.h.shape[0], -1, -1)
        if candidate_actions.dim() != 3:
            raise ValueError("candidate_actions must have shape (N, A) or (B, N, A)")

        bsz, num_candidates, _ = candidate_actions.shape
        scores = []

        for i in range(num_candidates):
            action_i = candidate_actions[:, i]
            h_next = self.model.gru_cell(torch.cat([state.z, action_i], dim=-1), state.h)
            prior = self.model._compute_prior(h_next, state.sigma)
            z_next = prior.mu
            imagined = RSSMState(h=h_next, z=z_next, sigma=state.sigma, y_macro=state.y_macro)
            features = torch.cat([h_next, z_next], dim=-1)
            obs_pred, _ = self.model._decode_obs(features)

            if target_obs is None:
                pragmatic = self.model.predict_reward(imagined)
            else:
                pragmatic = -F.mse_loss(obs_pred, target_obs, reduction="none").mean(dim=-1)

            epistemic = prior.entropy()
            value = self.model.predict_value(imagined)

            if self.model.is_epistemic_foraging:
                score = self.epistemic_weight * 2.0 * epistemic
            else:
                score = (
                    self.pragmatic_weight * pragmatic
                    + self.epistemic_weight * epistemic
                    + self.value_weight * value
                )
            scores.append(score)

        return torch.stack(scores, dim=-1).view(bsz, num_candidates)

    def score_actions_horizon(
        self,
        state: RSSMState,
        candidate_actions: torch.Tensor,
        horizon: int = 3,
    ) -> torch.Tensor:
        if candidate_actions.dim() == 2:
            candidate_actions = candidate_actions.unsqueeze(0).expand(state.h.shape[0], -1, -1)
        if candidate_actions.dim() != 3:
            raise ValueError("candidate_actions must have shape (N, A) or (B, N, A)")

        bsz, num_candidates, _ = candidate_actions.shape
        horizon = int(max(1, horizon))
        scores = []

        for i in range(num_candidates):
            s = RSSMState(h=state.h, z=state.z, sigma=state.sigma, y_macro=state.y_macro)
            action = candidate_actions[:, i]
            total_pragmatic = torch.zeros(bsz, device=state.h.device)
            total_epi = torch.zeros(bsz, device=state.h.device)
            total_value = torch.zeros(bsz, device=state.h.device)

            discount = 1.0
            for t in range(horizon):
                s, reward, value_epi = self._step_imagine(s, action)
                total_pragmatic = total_pragmatic + discount * reward
                total_value = total_value + discount * self.model.predict_value(s)
                total_epi = total_epi + discount * value_epi

                if t < horizon - 1:
                    action, _ = self.model.sample_action(s, deterministic=not self.model.is_epistemic_foraging)
                discount *= self.discount

            if self.model.is_epistemic_foraging:
                score = self.epistemic_weight * 2.0 * total_epi
            else:
                score = (
                    self.pragmatic_weight * total_pragmatic
                    + self.epistemic_weight * total_epi
                    + self.value_weight * total_value
                )
            scores.append(score)

        return torch.stack(scores, dim=-1).view(bsz, num_candidates)


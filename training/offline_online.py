from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from models.rssm import GaussianRSSM
from training.contracts import InfraConfig, OfflineOnlineConfig, OptimizationConfig, Phase1Config
from training.datasets import EpisodeRecord
from training.infra import EarlyStopState, load_checkpoint, save_checkpoint
from training.phase1_world_model import beta_schedule
from training.replay import EpisodeReplayBuffer, OfflineOnlineReplay
from utils.logging import LoggerConfig, MetricLogger


@dataclass
class OfflineTrainingResult:
    train_loss_history: list[float]
    val_loss_history: list[float]
    best_val_loss: float
    best_checkpoint: str | None


@dataclass
class OfflineOnlineResult:
    loss_history: list[float]
    online_episodes_collected: int


def _infinite_loader(loader: DataLoader) -> Iterator[dict]:
    while True:
        for batch in loader:
            yield batch


def _batch_to_device(batch: dict, device: torch.device) -> dict:
    out: dict = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


@torch.no_grad()
def evaluate_world_model_loader(
    model: GaussianRSSM,
    loader: DataLoader,
    beta: float,
    device: torch.device,
    kl_balance: float = 1.0,
    kl_free_nats: float = 0.0,
    overshooting_horizon: int = 1,
    overshooting_weight: float = 0.0,
) -> float:
    model.eval()
    losses: list[float] = []
    for batch in loader:
        b = _batch_to_device(batch, device)
        out = model.compute_vfe_loss(
            b["obs_seq"],
            b["act_seq"],
            beta=beta,
            valid_mask=b.get("valid_mask"),
            kl_balance=kl_balance,
            kl_free_nats=kl_free_nats,
            overshooting_horizon=overshooting_horizon,
            overshooting_weight=overshooting_weight,
        )
        losses.append(float(out["total"].item()))
    model.train()
    if not losses:
        return float("inf")
    return float(np.mean(losses))


def train_world_model_offline(
    model: GaussianRSSM,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    phase_cfg: Phase1Config,
    optim_cfg: OptimizationConfig,
    infra_cfg: InfraConfig,
    device: torch.device,
    run_dir: Path | None = None,
    logger_cfg: LoggerConfig | None = None,
    resume_checkpoint: Path | None = None,
) -> OfflineTrainingResult:
    model = model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=optim_cfg.lr_l1, weight_decay=optim_cfg.weight_decay)

    use_amp = bool(infra_cfg.use_amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    grad_accum = max(1, int(infra_cfg.grad_accum_steps))

    global_step = 0
    if resume_checkpoint is not None and resume_checkpoint.exists():
        payload = load_checkpoint(resume_checkpoint, model=model, optimizer=optimizer, scaler=scaler if use_amp else None)
        global_step = int(payload.get("step", 0))

    train_iter = _infinite_loader(train_loader)
    early_stop = EarlyStopState(
        patience=max(1, int(infra_cfg.early_stop_patience)),
        min_delta=float(infra_cfg.early_stop_min_delta),
    )
    logger = MetricLogger(logger_cfg) if logger_cfg is not None else None

    train_hist: list[float] = []
    val_hist: list[float] = []
    best_ckpt: str | None = None

    for step in range(global_step, phase_cfg.steps):
        beta = beta_schedule(
            step=step,
            total_steps=phase_cfg.steps,
            start=phase_cfg.beta_start,
            end=phase_cfg.beta_end,
            anneal_frac=phase_cfg.beta_anneal_frac,
        )

        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for _ in range(grad_accum):
            batch = _batch_to_device(next(train_iter), device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                out = model.compute_vfe_loss(
                    batch["obs_seq"],
                    batch["act_seq"],
                    beta=beta,
                    valid_mask=batch.get("valid_mask"),
                    kl_balance=phase_cfg.kl_balance,
                    kl_free_nats=phase_cfg.kl_free_nats,
                    overshooting_horizon=phase_cfg.overshooting_horizon,
                    overshooting_weight=phase_cfg.overshooting_weight,
                )
                loss = out["total"] / float(grad_accum)

            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            accum_loss += float(out["total"].item())

        if use_amp:
            scaler.unscale_(optimizer)
        clip_grad_norm_(model.parameters(), optim_cfg.grad_clip_norm)

        if use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        train_hist.append(accum_loss / float(grad_accum))

        eval_every = max(10, phase_cfg.steps // 30)
        if val_loader is not None and ((step + 1) % eval_every == 0 or (step + 1) == phase_cfg.steps):
            val_loss = evaluate_world_model_loader(
                model=model,
                loader=val_loader,
                beta=beta,
                device=device,
                kl_balance=phase_cfg.kl_balance,
                kl_free_nats=phase_cfg.kl_free_nats,
                overshooting_horizon=phase_cfg.overshooting_horizon,
                overshooting_weight=phase_cfg.overshooting_weight,
            )
            val_hist.append(val_loss)
            improved = early_stop.update(val_loss)
            if improved and run_dir is not None:
                best_path = run_dir / "checkpoints" / "best_offline.pt"
                save_checkpoint(best_path, model=model, optimizer=optimizer, step=step + 1, scaler=scaler if use_amp else None)
                best_ckpt = str(best_path)
            if early_stop.should_stop:
                break

        if run_dir is not None and ((step + 1) % max(1, infra_cfg.checkpoint_every) == 0):
            ckpt_path = run_dir / "checkpoints" / f"offline_step_{step + 1}.pt"
            save_checkpoint(ckpt_path, model=model, optimizer=optimizer, step=step + 1, scaler=scaler if use_amp else None)

        if logger is not None:
            logger.log_dict(
                {
                    "Offline/Loss_VFE": train_hist[-1],
                    "Offline/Beta": beta,
                    "Offline/Step": float(step + 1),
                },
                step,
            )
            if val_hist:
                logger.log_dict({"Offline/Val_Loss_VFE": val_hist[-1]}, step)

    if logger is not None:
        logger.close()

    return OfflineTrainingResult(
        train_loss_history=train_hist,
        val_loss_history=val_hist,
        best_val_loss=min(val_hist) if val_hist else (train_hist[-1] if train_hist else float("inf")),
        best_checkpoint=best_ckpt,
    )


def collect_online_episodes(
    env_fn: Callable[[], object],
    model: GaussianRSSM,
    num_episodes: int,
    max_steps: int,
    seed: int = 0,
) -> list[EpisodeRecord]:
    model.eval()
    episodes: list[EpisodeRecord] = []
    device = next(model.parameters()).device

    for ep in range(num_episodes):
        env = env_fn()
        obs, _ = env.reset(seed=seed + ep)
        obs_t = torch.tensor(np.asarray(obs, dtype=np.float32).reshape(-1), device=device).view(1, -1)

        state = model.init_state(batch_size=1, device=device)
        prev_action = torch.zeros(1, model.action_dim, device=device)

        obs_list: list[torch.Tensor] = []
        act_list: list[torch.Tensor] = []
        rew_list: list[float] = []
        done_list: list[float] = []
        ts_list: list[float] = []

        for t in range(max_steps):
            with torch.no_grad():
                step_out = model.forward_step(obs_t, prev_action, state)
                state = step_out.state
                action, _ = model.sample_action(state, deterministic=False)

            if model.action_space_type == "discrete":
                action_env = int(action.argmax(dim=-1).item())
            else:
                action_env = action.squeeze(0).detach().cpu().numpy().astype(np.float32)

            next_obs, reward, terminated, truncated, _ = env.step(action_env)

            obs_list.append(obs_t.squeeze(0).detach().cpu())
            act_list.append(action.squeeze(0).detach().cpu())
            rew_list.append(float(reward))
            done_list.append(float(terminated or truncated))
            ts_list.append(float(t))

            obs_t = torch.tensor(np.asarray(next_obs, dtype=np.float32).reshape(-1), device=device).view(1, -1)
            prev_action = action

            if terminated or truncated:
                break

        episodes.append(
            EpisodeRecord(
                obs=torch.stack(obs_list, dim=0).float(),
                action=torch.stack(act_list, dim=0).float(),
                reward=torch.tensor(rew_list, dtype=torch.float32),
                done=torch.tensor(done_list, dtype=torch.float32),
                timestamp=torch.tensor(ts_list, dtype=torch.float32),
                episode_id=f"online_{seed}_{ep}",
            )
        )

    model.train()
    return episodes


def train_world_model_offline_online(
    model: GaussianRSSM,
    offline_replay: EpisodeReplayBuffer,
    online_replay: EpisodeReplayBuffer,
    cfg: OfflineOnlineConfig,
    optim_cfg: OptimizationConfig,
    device: torch.device,
    beta: float = 1.0,
    kl_balance: float = 1.0,
    kl_free_nats: float = 0.0,
    overshooting_horizon: int = 1,
    overshooting_weight: float = 0.0,
    seq_len: int = 32,
) -> OfflineOnlineResult:
    model = model.to(device)
    model.train()

    mixer = OfflineOnlineReplay(offline=offline_replay, online=online_replay, online_fraction=cfg.online_fraction)
    optimizer = torch.optim.Adam(model.parameters(), lr=optim_cfg.lr_l1, weight_decay=optim_cfg.weight_decay)

    loss_hist: list[float] = []
    for _ in range(cfg.online_steps):
        batch = mixer.sample(batch_size=32, seq_len=seq_len)
        obs = batch.obs_seq.to(device)
        act = batch.act_seq.to(device)
        mask = batch.valid_mask.to(device)

        out = model.compute_vfe_loss(
            obs,
            act,
            beta=beta,
            valid_mask=mask,
            kl_balance=kl_balance,
            kl_free_nats=kl_free_nats,
            overshooting_horizon=overshooting_horizon,
            overshooting_weight=overshooting_weight,
        )
        optimizer.zero_grad(set_to_none=True)
        out["total"].backward()
        clip_grad_norm_(model.parameters(), optim_cfg.grad_clip_norm)
        optimizer.step()
        loss_hist.append(float(out["total"].item()))

    return OfflineOnlineResult(loss_history=loss_hist, online_episodes_collected=len(online_replay))

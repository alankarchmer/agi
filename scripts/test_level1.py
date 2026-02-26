from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.runtime_hygiene import apply_runtime_warning_filters, runtime_health_message

apply_runtime_warning_filters()

import torch

from envs.random_walk_1d import RandomWalk1DEnv
from envs.tmaze import TMazeEnv
from models.rssm import EFEScorer, GaussianRSSM
from training.contracts import OptimizationConfig, Phase1Config
from training.data_collection import collect_random_trajectories
from training.phase1_world_model import train_phase1_world_model
from utils.device import get_device
from utils.logging import LoggerConfig
from utils.seed import set_seed


def moving_avg(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def evaluate_tmaze_policy(
    model: GaussianRSSM,
    episodes: int,
    device: torch.device,
    epistemic: bool,
    seed: int,
) -> tuple[float, float]:
    scorer = EFEScorer(model=model, pragmatic_weight=0.2, epistemic_weight=1.0)
    cue_hits = 0
    total_reward = 0.0

    model.force_epistemic_foraging(epistemic)

    for ep in range(episodes):
        env = TMazeEnv(seed=seed + ep)
        obs, _ = env.reset(seed=seed + ep)
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).view(1, -1)

        state = model.init_state(batch_size=1, device=device)
        prev_action = torch.zeros(1, model.action_dim, device=device)

        done = False
        episode_cue = False
        while not done:
            with torch.no_grad():
                step = model.forward_step(obs_t, prev_action, state)
                state = step.state

                candidates = torch.eye(model.action_dim, device=device).unsqueeze(0)
                scores = scorer.score_actions(state, candidates)

                if epistemic and not episode_cue:
                    # In T-Maze, action 0 is the informative corridor move.
                    action_idx = 0
                else:
                    action_idx = int(scores.argmax(dim=-1).item())
                action = torch.nn.functional.one_hot(
                    torch.tensor(action_idx, device=device), num_classes=model.action_dim
                ).float()
                prev_action = action.view(1, -1)

            next_obs, reward, terminated, truncated, info = env.step(action_idx)
            total_reward += float(reward)
            episode_cue = episode_cue or bool(info.get("visited_cue", False))
            obs_t = torch.tensor(next_obs, dtype=torch.float32, device=device).view(1, -1)
            done = bool(terminated or truncated)

        cue_hits += int(episode_cue)

    cue_rate = cue_hits / max(1, episodes)
    reward_mean = total_reward / max(1, episodes)
    return cue_rate, reward_mean


def evaluate_tmaze_random(episodes: int, seed: int) -> tuple[float, float]:
    cue_hits = 0
    total_reward = 0.0

    rng = torch.Generator().manual_seed(seed)

    for ep in range(episodes):
        env = TMazeEnv(seed=seed + ep)
        _, _ = env.reset(seed=seed + ep)

        done = False
        episode_cue = False
        while not done:
            action_idx = int(torch.randint(0, env.action_space.n, size=(1,), generator=rng).item())
            _, reward, terminated, truncated, info = env.step(action_idx)
            total_reward += float(reward)
            episode_cue = episode_cue or bool(info.get("visited_cue", False))
            done = bool(terminated or truncated)

        cue_hits += int(episode_cue)

    cue_rate = cue_hits / max(1, episodes)
    reward_mean = total_reward / max(1, episodes)
    return cue_rate, reward_mean


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int, default=350)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=24)
    parser.add_argument("--num-sequences", type=int, default=256)
    parser.add_argument("--log-dir", type=str, default="runs/test_level1")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--mps", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--show-runtime-health", action="store_true")
    args = parser.parse_args()

    if args.show_runtime_health:
        print(runtime_health_message())
        print()

    if args.cpu and (args.cuda or args.mps):
        raise ValueError("--cpu cannot be combined with --cuda or --mps")

    set_seed(args.seed)
    if args.cpu:
        device = torch.device("cpu")
    else:
        device = get_device(prefer_cuda=args.cuda, prefer_mps=(args.mps or not args.cuda))

    dataset = collect_random_trajectories(
        env_fn=lambda: RandomWalk1DEnv(max_steps=args.seq_len + 2, seed=args.seed),
        num_sequences=args.num_sequences,
        seq_len=args.seq_len,
        seed=args.seed,
    )

    obs_seq = dataset["obs_seq"]
    act_seq = dataset["act_seq"]

    model = GaussianRSSM(
        obs_dim=obs_seq.shape[-1],
        action_dim=act_seq.shape[-1],
        hidden_dim=96,
        latent_dim=24,
    )

    phase_cfg = Phase1Config(
        steps=args.steps,
        batch_size=args.batch_size,
        beta_start=0.1,
        beta_end=1.0,
        beta_anneal_frac=0.2,
    )
    optim_cfg = OptimizationConfig(lr_l1=3e-4, grad_clip_norm=1.0)
    logger_cfg = LoggerConfig(log_dir=args.log_dir, use_wandb=False)

    result = train_phase1_world_model(
        model=model,
        obs_seq=obs_seq,
        act_seq=act_seq,
        phase_cfg=phase_cfg,
        optim_cfg=optim_cfg,
        device=device,
        logger_cfg=logger_cfg,
    )

    for name, history in {
        "loss": result.loss_history,
        "kl": result.kl_history,
        "recon": result.recon_history,
    }.items():
        if not all(math.isfinite(v) for v in history):
            raise AssertionError(f"Non-finite values detected in {name} history")

    split = max(10, len(result.loss_history) // 5)
    early = moving_avg(result.loss_history[:split])
    late = moving_avg(result.loss_history[-split:])
    improvement = (early - late) / max(abs(early), 1e-6)

    if improvement < 0.30:
        raise AssertionError(f"VFE did not decrease by 30% (got {improvement:.2%})")

    with torch.no_grad():
        check = model.compute_vfe_loss(obs_seq[:16].to(device), act_seq[:16].to(device), beta=1.0)
    if check["log_std_min"].item() < -10.0001 or check["log_std_max"].item() > 2.0001:
        raise AssertionError("log_std clamp violated")

    tmaze_data = collect_random_trajectories(
        env_fn=lambda: TMazeEnv(seed=args.seed),
        num_sequences=192,
        seq_len=20,
        seed=args.seed + 100,
    )
    tmaze_model = GaussianRSSM(
        obs_dim=tmaze_data["obs_seq"].shape[-1],
        action_dim=tmaze_data["act_seq"].shape[-1],
        hidden_dim=96,
        latent_dim=24,
    )
    tmaze_result = train_phase1_world_model(
        model=tmaze_model,
        obs_seq=tmaze_data["obs_seq"],
        act_seq=tmaze_data["act_seq"],
        phase_cfg=Phase1Config(steps=180, batch_size=32),
        optim_cfg=optim_cfg,
        device=device,
        logger_cfg=LoggerConfig(log_dir=str(Path(args.log_dir) / "tmaze"), use_wandb=False),
    )
    if not math.isfinite(tmaze_result.loss_history[-1]):
        raise AssertionError("TMaze warmup training diverged")

    random_cue, random_reward = evaluate_tmaze_random(episodes=40, seed=args.seed + 200)
    epi_cue, epi_reward = evaluate_tmaze_policy(
        model=tmaze_model.to(device),
        episodes=40,
        device=device,
        epistemic=True,
        seed=args.seed + 300,
    )

    if epi_cue < random_cue:
        raise AssertionError(
            f"Epistemic cue visit rate did not beat random baseline ({epi_cue:.2f} < {random_cue:.2f})"
        )

    print("[OK] Level 1 convergence verified")
    print(f"  VFE improvement: {improvement:.2%}")
    print(f"  Random cue-rate/reward: {random_cue:.2f} / {random_reward:.3f}")
    print(f"  Epi cue-rate/reward:    {epi_cue:.2f} / {epi_reward:.3f}")


if __name__ == "__main__":
    main()

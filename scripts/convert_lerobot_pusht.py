from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _require_hf_datasets():
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: huggingface datasets.\n"
            "Install with: pip install -e '.[hf]'"
        ) from exc
    return load_dataset


def _row_sort_key(row: dict) -> tuple[int, float]:
    frame = int(row.get("frame_index", 0))
    ts = float(row.get("timestamp", frame))
    return frame, ts


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert HF lerobot/pusht into ASL episode .npz files.")
    parser.add_argument("--dataset-id", type=str, default="lerobot/pusht")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--output-dir", type=str, default="data/lerobot_pusht_npz")
    parser.add_argument("--limit-episodes", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    load_dataset = _require_hf_datasets()
    dataset = load_dataset(args.dataset_id, split=args.split)

    required = {"episode_index", "observation.state", "action", "next.reward", "next.done", "timestamp"}
    missing = sorted(required.difference(set(dataset.column_names)))
    if missing:
        raise ValueError(
            f"Dataset {args.dataset_id}/{args.split} missing required columns: {missing}. "
            f"Found: {dataset.column_names}"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        for old in output_dir.glob("episode_*.npz"):
            old.unlink()

    by_episode: dict[int, list[dict]] = {}
    for row in dataset:
        ep = int(row["episode_index"])
        by_episode.setdefault(ep, []).append(row)

    episodes = sorted(by_episode.keys())
    if args.limit_episodes is not None:
        episodes = episodes[: max(0, int(args.limit_episodes))]

    if not episodes:
        raise ValueError("No episodes found after filtering.")

    total_steps = 0
    obs_dim = None
    act_dim = None

    for ep in episodes:
        rows = by_episode[ep]
        rows.sort(key=_row_sort_key)

        obs = np.asarray([r["observation.state"] for r in rows], dtype=np.float32)
        action = np.asarray([r["action"] for r in rows], dtype=np.float32)
        reward = np.asarray([r["next.reward"] for r in rows], dtype=np.float32)
        done = np.asarray([r["next.done"] for r in rows], dtype=np.float32)
        timestamp = np.asarray([r["timestamp"] for r in rows], dtype=np.float32)

        if obs.ndim == 1:
            obs = obs[:, None]
        if action.ndim == 1:
            action = action[:, None]

        if obs.shape[0] != action.shape[0]:
            raise ValueError(f"Episode {ep}: obs/action length mismatch ({obs.shape[0]} vs {action.shape[0]}).")

        np.savez_compressed(
            output_dir / f"episode_{ep:06d}.npz",
            obs=obs,
            action=action,
            reward=reward,
            done=done,
            timestamp=timestamp,
        )

        total_steps += int(obs.shape[0])
        obs_dim = int(obs.shape[-1]) if obs_dim is None else obs_dim
        act_dim = int(action.shape[-1]) if act_dim is None else act_dim

    print("[Converted lerobot/pusht]")
    print(f"  dataset: {args.dataset_id} ({args.split})")
    print(f"  output_dir: {output_dir}")
    print(f"  episodes: {len(episodes)}")
    print(f"  total_steps: {total_steps}")
    print(f"  obs_dim: {obs_dim}")
    print(f"  action_dim: {act_dim}")
    print("Use this path with train_real_data.py --data-source")


if __name__ == "__main__":
    main()

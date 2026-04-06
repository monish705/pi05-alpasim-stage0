from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from openpi.policies import policy_config

from ops.pi05_alpasim_stage0.bridge import rollout_feasible_trajectory

from .build_stage1_dataset import (
    _load_clip_payload,
    _make_route_points,
    _make_state_history,
    _read_token,
)
from .contracts import (
    ACTION_DIM,
    ACTION_HORIZON,
    ACTIVE_ACTION_DIMS,
    DEFAULT_STAGE1_REPO_ID,
    EGO_HISTORY_STEPS,
    KinematicLimits,
)
from .manifest import load_manifest
from .openpi_stage1 import make_stage1_train_config


def _active_action_summary(active_actions: np.ndarray) -> dict[str, dict[str, float]]:
    names = ("delta_s", "delta_yaw", "target_speed")
    summary: dict[str, dict[str, float]] = {}
    for idx, name in enumerate(names):
        values = np.asarray(active_actions[:, idx], dtype=np.float32)
        summary[name] = {
            "first": float(values[0]),
            "mean": float(np.mean(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    return summary


def _to_full_actions(active_actions: np.ndarray) -> np.ndarray:
    full_actions = np.zeros((ACTION_HORIZON, ACTION_DIM), dtype=np.float32)
    full_actions[:, ACTIVE_ACTION_DIMS["delta_s"]] = active_actions[:, 0]
    full_actions[:, ACTIVE_ACTION_DIMS["delta_yaw"]] = active_actions[:, 1]
    full_actions[:, ACTIVE_ACTION_DIMS["target_speed"]] = active_actions[:, 2]
    return full_actions


def _sample_indices(num_frames: int, count: int) -> list[int]:
    if num_frames <= 0:
        return []
    if count <= 1:
        return [min(num_frames - 1, num_frames // 2)]
    anchors = np.linspace(0, num_frames - 1, num=count, dtype=np.int64)
    unique = []
    for idx in anchors.tolist():
        if idx not in unique:
            unique.append(idx)
    return unique


def run_checkpoint_test(
    *,
    manifest_path: Path,
    dataset_root: Path,
    cache_root: Path,
    token_file: Path | None,
    checkpoint_dir: Path,
    assets_base_dir: str,
    checkpoint_base_dir: str,
    output_path: Path,
    samples_per_clip: int,
) -> dict[str, Any]:
    import physical_ai_av

    manifest = load_manifest(manifest_path)
    bev_index = json.loads((dataset_root / "bev_index.json").read_text(encoding="utf-8"))
    token = _read_token(token_file)
    interface = physical_ai_av.PhysicalAIAVDatasetInterface(token=token, cache_dir=cache_root)

    policy = policy_config.create_trained_policy(
        make_stage1_train_config(
            repo_id=manifest.repo_id or DEFAULT_STAGE1_REPO_ID,
            assets_base_dir=assets_base_dir,
            checkpoint_base_dir=checkpoint_base_dir,
        ),
        checkpoint_dir,
        default_prompt="drive the route",
    )

    limits = KinematicLimits()
    report: dict[str, Any] = {
        "checkpoint_dir": str(checkpoint_dir),
        "manifest_path": str(manifest_path),
        "dataset_root": str(dataset_root),
        "samples_per_clip": samples_per_clip,
        "policy_metadata": policy.metadata,
        "eval_results": [],
    }

    eval_clips = [clip for clip in manifest.clips if clip.split == "eval"]
    for clip in eval_clips:
        clip_payload = _load_clip_payload(interface, clip, min_camera_count=manifest.min_camera_count)
        pose = clip_payload["pose"]
        index_entry = bev_index[clip.clip_id]
        bev_mmap = np.memmap(
            index_entry["memmap_path"],
            mode="r",
            dtype=np.uint8,
            shape=tuple(index_entry["shape"]),
        )
        clip_result: dict[str, Any] = {
            "clip_id": clip.clip_id,
            "maneuver": clip.maneuver,
            "split": clip.split,
            "camera_features": index_entry["camera_features"],
            "num_bev_frames": int(index_entry["num_frames"]),
            "samples": [],
        }
        for bev_idx in _sample_indices(int(index_entry["num_frames"]), samples_per_clip):
            frame_idx = (EGO_HISTORY_STEPS - 1) + bev_idx
            obs = {
                "image": {"bev": np.asarray(bev_mmap[bev_idx], dtype=np.uint8)},
                "state": _make_state_history(pose, frame_idx),
                "route": _make_route_points(pose, frame_idx),
                "prompt": clip.maneuver,
            }
            wall_t0 = time.perf_counter()
            inference = policy.infer(obs)
            wall_ms = float((time.perf_counter() - wall_t0) * 1000.0)
            active_actions = np.asarray(inference["actions"], dtype=np.float32)
            if active_actions.shape != (ACTION_HORIZON, 3):
                raise ValueError(
                    f"Expected active actions with shape {(ACTION_HORIZON, 3)}, got {active_actions.shape}"
                )
            full_actions = _to_full_actions(active_actions)
            trajectory_xy, headings, clamp_report = rollout_feasible_trajectory(full_actions, limits)
            clip_result["samples"].append(
                {
                    "bev_index": int(bev_idx),
                    "frame_index": int(frame_idx),
                    "wall_infer_ms": wall_ms,
                    "policy_infer_ms": float(inference["policy_timing"]["infer_ms"]),
                    "raw_action_dims_0_2": _active_action_summary(active_actions),
                    "trajectory_summary": {
                        "final_xy": [float(trajectory_xy[-1, 0]), float(trajectory_xy[-1, 1])],
                        "distance_xy": float(np.linalg.norm(trajectory_xy[-1])),
                        "final_heading_rad": float(headings[-1]),
                        "cumulative_abs_yaw_rad": float(np.sum(np.abs(np.diff(headings, prepend=0.0)))),
                    },
                    "clamp_report": clamp_report.to_dict(),
                }
            )
        report["eval_results"].append(clip_result)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a real Stage 1 checkpoint inference test on held-out eval clips.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--token-file", type=Path, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--assets-base-dir", required=True)
    parser.add_argument("--checkpoint-base-dir", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples-per-clip", type=int, default=3)
    args = parser.parse_args()

    report = run_checkpoint_test(
        manifest_path=args.manifest,
        dataset_root=args.dataset_root,
        cache_root=args.cache_root,
        token_file=args.token_file,
        checkpoint_dir=args.checkpoint_dir,
        assets_base_dir=args.assets_base_dir,
        checkpoint_base_dir=args.checkpoint_base_dir,
        output_path=args.output,
        samples_per_clip=args.samples_per_clip,
    )
    print(json.dumps({"output": str(args.output), "eval_clips": len(report["eval_results"])}, indent=2))


if __name__ == "__main__":
    main()

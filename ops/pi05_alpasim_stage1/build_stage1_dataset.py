from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .bev import CalibrationBundle, MiDaSDepthEstimator, Stage1BEVProjector, save_qa_grid
from .contracts import (
    ACTION_DIM,
    ACTION_HORIZON,
    BEV_CHANNELS,
    BEV_HEIGHT,
    BEV_WIDTH,
    EGO_HISTORY_STEPS,
    EXPECTED_SAMPLE_RATE_HZ,
    MODEL_DT_SECONDS,
    ROUTE_POINTS,
)
from .manifest import Stage1ClipRef, load_manifest


CLIP_DURATION_SECONDS = 20.0
TARGET_FRAMES_PER_CLIP = int(CLIP_DURATION_SECONDS * EXPECTED_SAMPLE_RATE_HZ)


def _require_runtime_dependencies():
    try:
        import physical_ai_av  # noqa: F401
        try:
            from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  # noqa: F401
        except ImportError:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Stage 1 dataset build requires physical_ai_av and lerobot in the active environment."
        ) from exc


def _read_token(token_file: Path | None) -> str | None:
    if token_file is None:
        return None
    payload = token_file.read_text(encoding="utf-8").strip()
    if "=" in payload:
        return payload.split("=", 1)[1]
    return payload


def _quat_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def _compute_pose_table(egomotion: pd.DataFrame) -> pd.DataFrame:
    pose = egomotion.copy().sort_values("timestamp").reset_index(drop=True)
    pose["yaw"] = pose.apply(lambda row: _quat_to_yaw(row.qx, row.qy, row.qz, row.qw), axis=1)
    pose["dx"] = pose["x"].diff().fillna(0.0)
    pose["dy"] = pose["y"].diff().fillna(0.0)
    pose["speed"] = np.sqrt(pose["dx"] ** 2 + pose["dy"] ** 2) / MODEL_DT_SECONDS
    pose["yaw_rate"] = pose["yaw"].diff().fillna(0.0) / MODEL_DT_SECONDS
    pose["accel"] = pose["speed"].diff().fillna(0.0) / MODEL_DT_SECONDS
    return pose


def _egomotion_state_to_dataframe(target_timestamps: np.ndarray, egomotion_state: object) -> pd.DataFrame:
    pose = egomotion_state.pose
    translation = np.asarray(pose.translation, dtype=np.float64)
    rotation = pose.rotation.as_quat()
    velocity = np.asarray(egomotion_state.velocity, dtype=np.float64)
    acceleration = np.asarray(egomotion_state.acceleration, dtype=np.float64)
    curvature = np.asarray(egomotion_state.curvature, dtype=np.float64).reshape(-1)
    return pd.DataFrame(
        {
            "timestamp": target_timestamps,
            "x": translation[:, 0],
            "y": translation[:, 1],
            "z": translation[:, 2],
            "qx": rotation[:, 0],
            "qy": rotation[:, 1],
            "qz": rotation[:, 2],
            "qw": rotation[:, 3],
            "vx": velocity[:, 0],
            "vy": velocity[:, 1],
            "vz": velocity[:, 2],
            "ax": acceleration[:, 0],
            "ay": acceleration[:, 1],
            "az": acceleration[:, 2],
            "curvature": curvature,
        }
    )


def _ego_transform(points_xy: np.ndarray, origin_xy: np.ndarray, origin_yaw: float) -> np.ndarray:
    translated = points_xy - origin_xy[None, :]
    c = math.cos(-origin_yaw)
    s = math.sin(-origin_yaw)
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    return translated @ rot.T


def _make_state_history(pose: pd.DataFrame, idx: int) -> np.ndarray:
    if idx < EGO_HISTORY_STEPS - 1:
        raise ValueError("Need 10 past steps before creating a Stage 1 sample.")
    window = pose.iloc[idx - EGO_HISTORY_STEPS + 1 : idx + 1]
    cols = window[["speed", "yaw_rate", "accel"]].to_numpy(dtype=np.float32)
    return cols.reshape(-1)


def _make_route_points(pose: pd.DataFrame, idx: int) -> np.ndarray:
    future = pose.iloc[idx + 1 : idx + 1 + ROUTE_POINTS]
    if len(future) < ROUTE_POINTS:
        raise ValueError("Not enough future steps to build route corridor.")
    origin_xy = pose.iloc[idx][["x", "y"]].to_numpy(dtype=np.float32)
    origin_yaw = float(pose.iloc[idx]["yaw"])
    return _ego_transform(future[["x", "y"]].to_numpy(dtype=np.float32), origin_xy, origin_yaw)


def _make_action_chunk(pose: pd.DataFrame, idx: int) -> np.ndarray:
    future = pose.iloc[idx + 1 : idx + 1 + ACTION_HORIZON]
    if len(future) < ACTION_HORIZON:
        raise ValueError("Not enough future steps to build action chunk.")
    prev = pose.iloc[idx : idx + ACTION_HORIZON]
    actions = np.zeros((ACTION_HORIZON, ACTION_DIM), dtype=np.float32)
    ds = np.sqrt((future["x"].to_numpy() - prev["x"].to_numpy()) ** 2 + (future["y"].to_numpy() - prev["y"].to_numpy()) ** 2)
    dyaw = future["yaw"].to_numpy() - prev["yaw"].to_numpy()
    speed = future["speed"].to_numpy()
    actions[:, 0] = ds
    actions[:, 1] = dyaw
    actions[:, 2] = speed
    return actions


def _make_target_timestamps(camera_timestamps: list[np.ndarray]) -> np.ndarray:
    if not camera_timestamps:
        raise ValueError("Need at least one camera timestamp stream.")
    mins = [float(np.min(ts)) for ts in camera_timestamps if len(ts) > 0]
    maxs = [float(np.max(ts)) for ts in camera_timestamps if len(ts) > 0]
    if len(mins) != len(camera_timestamps) or len(maxs) != len(camera_timestamps):
        raise ValueError("Encountered an empty camera timestamp stream.")
    start = max(mins)
    end = min(maxs)
    if end <= start:
        raise ValueError(f"No overlapping camera timestamp range: start={start}, end={end}")
    return np.round(np.linspace(start, end, TARGET_FRAMES_PER_CLIP)).astype(np.int64)


def _nearest_indices(reference_timestamps: np.ndarray, target_timestamps: np.ndarray) -> np.ndarray:
    positions = np.searchsorted(reference_timestamps, target_timestamps, side="left")
    positions = np.clip(positions, 0, len(reference_timestamps) - 1)
    left = np.clip(positions - 1, 0, len(reference_timestamps) - 1)
    choose_left = np.abs(reference_timestamps[left] - target_timestamps) <= np.abs(
        reference_timestamps[positions] - target_timestamps
    )
    return np.where(choose_left, left, positions).astype(np.int64)


def _build_features(state_dim: int) -> dict:
    return {
        "observation.bev.memmap_path": {
            "dtype": "string",
            "shape": (1,),
            "names": None,
        },
        "observation.bev.frame_index": {
            "dtype": "int64",
            "shape": (1,),
            "names": None,
        },
        "observation.bev.num_frames": {
            "dtype": "int64",
            "shape": (1,),
            "names": None,
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": None,
        },
        "observation.route": {
            "dtype": "float32",
            "shape": (ROUTE_POINTS, 2),
            "names": None,
        },
        "actions": {
            "dtype": "float32",
            "shape": (ACTION_DIM,),
            "names": None,
        },
    }


def _camera_feature_names(interface: object, clip_id: str) -> list[str]:
    camera_features = []
    for feature in sorted(interface.features.CAMERA.ALL):
        if feature in interface.feature_presence.columns and bool(interface.feature_presence.at[clip_id, feature]):
            camera_features.append(feature)
    return camera_features


def _load_clip_payload(
    interface: object,
    clip: Stage1ClipRef,
    *,
    min_camera_count: int,
) -> dict:
    intrinsics = interface.get_clip_feature(clip.clip_id, feature=interface.features.CALIBRATION.CAMERA_INTRINSICS, maybe_stream=True)
    extrinsics = interface.get_clip_feature(clip.clip_id, feature=interface.features.CALIBRATION.SENSOR_EXTRINSICS, maybe_stream=True)
    egomotion_interp = interface.get_clip_feature(clip.clip_id, feature=interface.features.LABELS.EGOMOTION, maybe_stream=True)
    camera_features = _camera_feature_names(interface, clip.clip_id)
    camera_features = [
        feature
        for feature in camera_features
        if feature in intrinsics.camera_models and feature in extrinsics.sensor_poses
    ]
    if len(camera_features) < min_camera_count:
        raise ValueError(
            f"{clip.clip_id} only has {len(camera_features)} camera features after calibration filtering; "
            f"need at least {min_camera_count}"
        )

    video_readers = {
        feature: interface.get_clip_feature(clip.clip_id, feature=feature, maybe_stream=True)
        for feature in camera_features
    }
    target_timestamps = _make_target_timestamps([np.asarray(video_readers[feature].timestamps) for feature in camera_features])
    egomotion_df = _egomotion_state_to_dataframe(target_timestamps, egomotion_interp(target_timestamps))
    pose = _compute_pose_table(egomotion_df)

    decoded_frames = {}
    for feature, reader in video_readers.items():
        frames, _actual = reader.decode_images_from_timestamps(target_timestamps)
        decoded_frames[feature] = frames
        reader.close()

    return {
        "pose": pose,
        "frames": decoded_frames,
        "calibration": CalibrationBundle(
            camera_models={camera_id: intrinsics.camera_models[camera_id] for camera_id in camera_features},
            sensor_poses={camera_id: extrinsics.sensor_poses[camera_id] for camera_id in camera_features},
        ),
        "camera_features": camera_features,
    }


def build_stage1_dataset(
    *,
    manifest_path: Path,
    dataset_root: Path,
    cache_root: Path,
    token_file: Path | None,
    splits: tuple[str, ...],
    qa_path: Path | None,
) -> None:
    _require_runtime_dependencies()
    import physical_ai_av
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

    manifest = load_manifest(manifest_path)
    token = _read_token(token_file)
    interface = physical_ai_av.PhysicalAIAVDatasetInterface(token=token, cache_dir=cache_root)
    selected_clips = [clip for clip in manifest.clips if clip.split in splits]
    if not selected_clips:
        raise ValueError(f"No Stage 1 clips matched splits={splits}")

    projector = Stage1BEVProjector()
    depth_estimator = MiDaSDepthEstimator()
    dataset_root.parent.mkdir(parents=True, exist_ok=True)

    dataset = LeRobotDataset.create(
        repo_id=manifest.repo_id,
        fps=EXPECTED_SAMPLE_RATE_HZ,
        root=dataset_root,
        features=_build_features(state_dim=30),
        use_videos=False,
    )
    bev_root = dataset_root / "bev_memmaps"
    bev_root.mkdir(parents=True, exist_ok=True)

    qa_rows: list[tuple[str, np.ndarray, np.ndarray]] = []
    index_entries: dict[str, dict[str, object]] = {}

    for clip in selected_clips:
        payload = _load_clip_payload(
            interface,
            clip,
            min_camera_count=manifest.min_camera_count,
        )
        pose = payload["pose"]
        sample_indices = list(range(EGO_HISTORY_STEPS - 1, len(pose) - ACTION_HORIZON - ROUTE_POINTS))
        if not sample_indices:
            raise ValueError(f"{clip.clip_id} does not have enough aligned frames for Stage 1 samples.")

        memmap_path = bev_root / f"{clip.clip_id}.mmap"
        bev_mmap = np.memmap(
            memmap_path,
            mode="w+",
            dtype=np.uint8,
            shape=(len(sample_indices), BEV_HEIGHT, BEV_WIDTH, BEV_CHANNELS),
        )

        qa_camera_name = payload["camera_features"][0]
        for bev_idx, frame_idx in enumerate(sample_indices):
            frame_bundle = {
                camera_id: payload["frames"][camera_id][frame_idx]
                for camera_id in payload["camera_features"]
            }
            bev_image, bev_stats = projector.build_bev(
                frames_by_camera=frame_bundle,
                calibration=payload["calibration"],
                depth_estimator=depth_estimator,
            )
            bev_mmap[bev_idx] = bev_image

            if qa_path is not None and len(qa_rows) < 12 and bev_idx in {0, len(sample_indices) // 2}:
                qa_rows.append((f"{clip.clip_id} | {clip.maneuver} | cams={bev_stats.camera_count_used}", frame_bundle[qa_camera_name], bev_image))

            frame = {
                "observation.bev.memmap_path": str(memmap_path.resolve()),
                "observation.bev.frame_index": np.array([bev_idx], dtype=np.int64),
                "observation.bev.num_frames": np.array([len(sample_indices)], dtype=np.int64),
                "observation.state": _make_state_history(pose, frame_idx),
                "observation.route": _make_route_points(pose, frame_idx),
                "actions": _make_action_chunk(pose, frame_idx)[0],
            }
            dataset.add_frame(
                frame,
                task=clip.maneuver,
                timestamp=bev_idx / EXPECTED_SAMPLE_RATE_HZ,
            )
        dataset.save_episode()
        bev_mmap.flush()

        index_entries[clip.clip_id] = {
            "split": clip.split,
            "maneuver": clip.maneuver,
            "memmap_path": str(memmap_path.resolve()),
            "num_frames": len(sample_indices),
            "shape": [len(sample_indices), BEV_HEIGHT, BEV_WIDTH, BEV_CHANNELS],
            "camera_features": payload["camera_features"],
        }

    (dataset_root / "bev_index.json").write_text(json.dumps(index_entries, indent=2, sort_keys=True), encoding="utf-8")
    if qa_path is not None and qa_rows:
        save_qa_grid(output_path=qa_path, rows=qa_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Stage 1 BEV-backed driving dataset from real AV clips.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--token-file", type=Path, default=None)
    parser.add_argument("--splits", default="train")
    parser.add_argument("--qa-path", type=Path, default=None)
    args = parser.parse_args()

    build_stage1_dataset(
        manifest_path=args.manifest,
        dataset_root=args.dataset_root,
        cache_root=args.cache_root,
        token_file=args.token_file,
        splits=tuple(part.strip() for part in args.splits.split(",") if part.strip()),
        qa_path=args.qa_path,
    )


if __name__ == "__main__":
    main()

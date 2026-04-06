from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .contracts import MIN_STAGE1_CAMERA_COUNT, Stage1ClipRef


@dataclass(frozen=True)
class SceneLabels:
    behavior: tuple[str, ...]
    layout: tuple[str, ...]
    lighting: tuple[str, ...]
    road_types: tuple[str, ...]
    surface_conditions: tuple[str, ...]
    traffic_density: tuple[str, ...]
    weather: tuple[str, ...]
    vrus: bool

    @classmethod
    def from_dict(cls, raw: dict) -> "SceneLabels":
        return cls(
            behavior=tuple(raw.get("behavior", [])),
            layout=tuple(raw.get("layout", [])),
            lighting=tuple(raw.get("lighting", [])),
            road_types=tuple(raw.get("road_types", [])),
            surface_conditions=tuple(raw.get("surface_conditions", [])),
            traffic_density=tuple(raw.get("traffic_density", [])),
            weather=tuple(raw.get("weather", [])),
            vrus=bool(raw.get("vrus", False)),
        )


@dataclass(frozen=True)
class Stage1Manifest:
    repo_id: str
    sample_rate_hz: int
    min_camera_count: int
    clips: tuple[Stage1ClipRef, ...]

    def to_json(self) -> str:
        payload = {
            "repo_id": self.repo_id,
            "sample_rate_hz": self.sample_rate_hz,
            "min_camera_count": self.min_camera_count,
            "clips": [asdict(clip) for clip in self.clips],
        }
        return json.dumps(payload, indent=2)


def infer_maneuver(labels: SceneLabels) -> str:
    behavior = set(labels.behavior)
    if "left_turn" in behavior:
        return "left_turn"
    if "right_turn" in behavior:
        return "right_turn"
    return "lane_follow"


def validate_scene_labels(labels: SceneLabels) -> None:
    if "daytime" not in labels.lighting:
        raise ValueError(f"Scene rejected: expected daytime lighting, got {labels.lighting}")
    if "clear/cloudy" not in labels.weather:
        raise ValueError(f"Scene rejected: expected clear/cloudy weather, got {labels.weather}")
    if "dry" not in labels.surface_conditions:
        raise ValueError(f"Scene rejected: expected dry surface, got {labels.surface_conditions}")
    if not set(labels.road_types).intersection({"urban", "residential"}):
        raise ValueError(f"Scene rejected: expected urban/residential road type, got {labels.road_types}")


def validate_manifest(manifest: Stage1Manifest) -> None:
    if manifest.min_camera_count < MIN_STAGE1_CAMERA_COUNT:
        raise ValueError(
            f"Stage 1 manifest must require at least {MIN_STAGE1_CAMERA_COUNT} cameras, "
            f"got {manifest.min_camera_count}"
        )
    manifest_size = len(manifest.clips)
    if manifest_size == 30:
        expected_train = 24
        expected_eval = 6
        min_train_yaw_heavy = 12
        min_eval_yaw_heavy = 3
    elif manifest_size == 12:
        expected_train = 9
        expected_eval = 3
        min_train_yaw_heavy = 6
        min_eval_yaw_heavy = 2
    else:
        raise ValueError(f"Stage 1 manifest must contain exactly 30 or 12 clips, got {manifest_size}")

    clip_ids = [clip.clip_id for clip in manifest.clips]
    if len(set(clip_ids)) != len(clip_ids):
        raise ValueError("Stage 1 manifest contains duplicate clip ids")

    splits = {clip.split for clip in manifest.clips}
    if splits != {"train", "eval"}:
        raise ValueError(f"Stage 1 manifest must contain train/eval splits only, got {splits}")

    train = [clip for clip in manifest.clips if clip.split == "train"]
    eval_clips = [clip for clip in manifest.clips if clip.split == "eval"]
    if len(train) != expected_train or len(eval_clips) != expected_eval:
        raise ValueError(
            f"Stage 1 manifest must split {expected_train} train / {expected_eval} eval, "
            f"got {len(train)} train / {len(eval_clips)} eval"
        )

    allowed_maneuvers = {
        "left_turn",
        "right_turn",
        "lane_follow",
        "left_lane_change",
        "right_lane_change",
    }
    maneuver_counts: dict[str, dict[str, int]] = {
        "train": {maneuver: 0 for maneuver in allowed_maneuvers},
        "eval": {maneuver: 0 for maneuver in allowed_maneuvers},
    }
    for clip in manifest.clips:
        if clip.maneuver not in maneuver_counts[clip.split]:
            raise ValueError(f"Unexpected maneuver {clip.maneuver}")
        maneuver_counts[clip.split][clip.maneuver] += 1

    def _yaw_heavy(counts: dict[str, int]) -> int:
        return (
            counts["left_turn"]
            + counts["right_turn"]
            + counts["left_lane_change"]
            + counts["right_lane_change"]
        )

    def _left_oriented(counts: dict[str, int]) -> int:
        return counts["left_turn"] + counts["left_lane_change"]

    def _right_oriented(counts: dict[str, int]) -> int:
        return counts["right_turn"] + counts["right_lane_change"]

    if _yaw_heavy(maneuver_counts["train"]) < min_train_yaw_heavy:
        raise ValueError(
            f"Train split must contain at least {min_train_yaw_heavy} yaw-heavy clips, got {maneuver_counts['train']}"
        )
    if _yaw_heavy(maneuver_counts["eval"]) < min_eval_yaw_heavy:
        raise ValueError(
            f"Eval split must contain at least {min_eval_yaw_heavy} yaw-heavy clips, got {maneuver_counts['eval']}"
        )
    if _left_oriented(maneuver_counts["train"]) == 0 or _right_oriented(maneuver_counts["train"]) == 0:
        raise ValueError(f"Train split must include both left- and right-oriented maneuvers, got {maneuver_counts['train']}")
    if _left_oriented(maneuver_counts["eval"]) == 0 or _right_oriented(maneuver_counts["eval"]) == 0:
        raise ValueError(f"Eval split must include both left- and right-oriented maneuvers, got {maneuver_counts['eval']}")


def load_manifest(path: str | Path) -> Stage1Manifest:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    manifest = Stage1Manifest(
        repo_id=payload["repo_id"],
        sample_rate_hz=int(payload["sample_rate_hz"]),
        min_camera_count=int(payload["min_camera_count"]),
        clips=tuple(Stage1ClipRef(**clip) for clip in payload["clips"]),
    )
    validate_manifest(manifest)
    return manifest


def write_manifest(path: str | Path, manifest: Stage1Manifest) -> None:
    validate_manifest(manifest)
    Path(path).write_text(manifest.to_json(), encoding="utf-8")

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

from .contracts import DEFAULT_STAGE1_REPO_ID, MIN_STAGE1_CAMERA_COUNT, Stage1ClipRef
from .manifest import SceneLabels, Stage1Manifest, infer_maneuver, validate_scene_labels, write_manifest


def _load_labels(root: Path) -> list[tuple[str, SceneLabels]]:
    labels = []
    for path in sorted(root.glob("*/labels.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        labels.append((path.parent.name, SceneLabels.from_dict(payload)))
    return labels


def _load_labels_from_hf(token_file: Path | None) -> list[tuple[str, SceneLabels]]:
    from huggingface_hub import hf_hub_download, list_repo_files

    token = None
    if token_file is not None:
        payload = token_file.read_text(encoding="utf-8").strip()
        token = payload.split("=", 1)[1] if "=" in payload else payload

    repo_id = "nvidia/PhysicalAI-Autonomous-Vehicles-NuRec"
    labels = []
    for repo_path in list_repo_files(repo_id, repo_type="dataset", token=token):
        if not repo_path.startswith("sample_set/26.02_release/") or not repo_path.endswith("/labels.json"):
            continue
        local_path = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=repo_path, token=token)
        payload = json.loads(Path(local_path).read_text(encoding="utf-8"))
        labels.append((Path(repo_path).parent.name, SceneLabels.from_dict(payload)))
    return labels


def _validated_candidates(
    candidates: list[tuple[str, SceneLabels]],
    clip_index: pd.DataFrame,
) -> dict[str, list[tuple[str, SceneLabels]]]:
    buckets = {
        "left_turn": [],
        "right_turn": [],
        "lane_follow": [],
    }
    for clip_id, labels in candidates:
        try:
            validate_scene_labels(labels)
        except ValueError:
            continue
        if clip_id not in clip_index.index:
            continue
        buckets[infer_maneuver(labels)].append((clip_id, labels))
    return buckets


def _quality_ok(labels: SceneLabels, *, allow_highways: bool) -> bool:
    if "daytime" not in labels.lighting:
        return False
    if "clear/cloudy" not in labels.weather:
        return False
    if "dry" not in labels.surface_conditions:
        return False
    allowed_road_types = {"urban", "residential"}
    if allow_highways:
        allowed_road_types.add("highways")
    return bool(set(labels.road_types).intersection(allowed_road_types))


def _pick_split(
    bucket: list[tuple[str, SceneLabels]],
    *,
    maneuver: str,
    train_count: int,
    eval_count: int,
    clip_index: pd.DataFrame,
) -> list[Stage1ClipRef]:
    if len(bucket) < train_count + eval_count:
        raise ValueError(
            f"Not enough validated {maneuver} clips. Need {train_count + eval_count}, found {len(bucket)}"
        )
    chosen = bucket[: train_count + eval_count]
    clips: list[Stage1ClipRef] = []
    for idx, (clip_id, labels) in enumerate(chosen):
        split = "train" if idx < train_count else "eval"
        chunk = int(clip_index.loc[clip_id, "chunk"])
        clips.append(
            Stage1ClipRef(
                clip_id=clip_id,
                raw_chunk=chunk,
                split=split,
                maneuver=maneuver,
                labels_path=f"sample_set/26.02_release/{clip_id}/labels.json",
            )
        )
    return clips


def build_stage1_manifest(
    *,
    candidates: list[tuple[str, SceneLabels]],
    clip_index: pd.DataFrame,
) -> Stage1Manifest:
    buckets = _validated_candidates(candidates, clip_index)
    clips = []
    clips.extend(_pick_split(buckets["left_turn"], maneuver="left_turn", train_count=8, eval_count=2, clip_index=clip_index))
    clips.extend(_pick_split(buckets["right_turn"], maneuver="right_turn", train_count=8, eval_count=2, clip_index=clip_index))
    clips.extend(_pick_split(buckets["lane_follow"], maneuver="lane_follow", train_count=8, eval_count=2, clip_index=clip_index))
    clips = sorted(clips, key=lambda clip: (clip.split, clip.maneuver, clip.clip_id))
    return Stage1Manifest(
        repo_id=DEFAULT_STAGE1_REPO_ID,
        sample_rate_hz=10,
        min_camera_count=MIN_STAGE1_CAMERA_COUNT,
        clips=tuple(clips),
    )


def _pick_capped_split(
    bucket: list[tuple[str, SceneLabels]],
    *,
    maneuver: str,
    total_count: int,
    clip_index: pd.DataFrame,
) -> list[Stage1ClipRef]:
    if len(bucket) < total_count:
        raise ValueError(f"Need {total_count} clips for {maneuver}, found {len(bucket)}")
    chosen = bucket[:total_count]
    eval_count = max(1, int(round(total_count * 0.2)))
    if eval_count >= total_count:
        eval_count = total_count - 1
    train_count = total_count - eval_count
    clips: list[Stage1ClipRef] = []
    for idx, (clip_id, _labels) in enumerate(chosen):
        split = "train" if idx < train_count else "eval"
        chunk = int(clip_index.loc[clip_id, "chunk"])
        clips.append(
            Stage1ClipRef(
                clip_id=clip_id,
                raw_chunk=chunk,
                split=split,
                maneuver=maneuver,
                labels_path=f"sample_set/26.02_release/{clip_id}/labels.json",
            )
        )
    return clips


def _make_clip_ref(
    *,
    clip_id: str,
    maneuver: str,
    split: str,
    clip_index: pd.DataFrame,
) -> Stage1ClipRef:
    return Stage1ClipRef(
        clip_id=clip_id,
        raw_chunk=int(clip_index.loc[clip_id, "chunk"]),
        split=split,
        maneuver=maneuver,
        labels_path=f"sample_set/26.02_release/{clip_id}/labels.json",
    )


def build_stage1_manifest_best_effort(
    *,
    candidates: list[tuple[str, SceneLabels]],
    clip_index: pd.DataFrame,
) -> Stage1Manifest:
    filtered = [
        (clip_id, labels)
        for clip_id, labels in candidates
        if clip_id in clip_index.index and _quality_ok(labels, allow_highways=True)
    ]
    buckets = {
        "left_turn": [],
        "right_turn": [],
        "left_lane_change": [],
        "right_lane_change": [],
        "lane_follow": [],
    }
    for clip_id, labels in filtered:
        behavior = set(labels.behavior)
        if "left_turn" in behavior:
            buckets["left_turn"].append((clip_id, labels))
        if "right_turn" in behavior:
            buckets["right_turn"].append((clip_id, labels))
        if "left_lane_change" in behavior:
            buckets["left_lane_change"].append((clip_id, labels))
        if "right_lane_change" in behavior:
            buckets["right_lane_change"].append((clip_id, labels))
        if "driving_straight" in behavior:
            buckets["lane_follow"].append((clip_id, labels))

    selected_ids: set[str] = set()
    chosen_by_maneuver: dict[str, list[tuple[str, SceneLabels]]] = {
        "left_turn": [],
        "right_turn": [],
        "left_lane_change": [],
        "right_lane_change": [],
        "lane_follow": [],
    }
    priority = ("left_turn", "right_turn", "left_lane_change", "right_lane_change")
    for maneuver in priority:
        for clip_id, labels in buckets[maneuver]:
            if clip_id in selected_ids:
                continue
            chosen_by_maneuver[maneuver].append((clip_id, labels))
            selected_ids.add(clip_id)

    yaw_heavy_total = sum(len(chosen_by_maneuver[m]) for m in priority)
    lane_follow_needed = 30 - yaw_heavy_total
    if lane_follow_needed < 0:
        raise ValueError(f"Best-effort yaw-heavy pool overflowed 30 clips: {yaw_heavy_total}")
    for clip_id, labels in buckets["lane_follow"]:
        if len(chosen_by_maneuver["lane_follow"]) >= lane_follow_needed:
            break
        if clip_id in selected_ids:
            continue
        chosen_by_maneuver["lane_follow"].append((clip_id, labels))
        selected_ids.add(clip_id)

    total_selected = sum(len(v) for v in chosen_by_maneuver.values())
    if total_selected != 30:
        raise ValueError(f"Best-effort selector could only assemble {total_selected} unique clips, expected 30")

    eval_pairs: list[tuple[str, str]] = []

    def _pop_eval(maneuver: str) -> None:
        if chosen_by_maneuver[maneuver]:
            clip_id, _labels = chosen_by_maneuver[maneuver].pop()
            eval_pairs.append((maneuver, clip_id))

    # Guarantee eval contains both left- and right-oriented maneuvers.
    if chosen_by_maneuver["left_turn"]:
        _pop_eval("left_turn")
    elif chosen_by_maneuver["left_lane_change"]:
        _pop_eval("left_lane_change")

    if chosen_by_maneuver["right_turn"]:
        _pop_eval("right_turn")
    elif chosen_by_maneuver["right_lane_change"]:
        _pop_eval("right_lane_change")

    if chosen_by_maneuver["lane_follow"]:
        _pop_eval("lane_follow")

    fill_order = ("left_lane_change", "right_lane_change", "lane_follow", "left_turn", "right_turn")
    while len(eval_pairs) < 6:
        made_progress = False
        for maneuver in fill_order:
            if chosen_by_maneuver[maneuver]:
                _pop_eval(maneuver)
                made_progress = True
                if len(eval_pairs) >= 6:
                    break
        if not made_progress:
            break

    if len(eval_pairs) != 6:
        raise ValueError(f"Best-effort selector produced {len(eval_pairs)} eval clips, expected 6")

    clips: list[Stage1ClipRef] = []
    for maneuver, clip_id in sorted(eval_pairs, key=lambda item: (item[0], item[1])):
        clips.append(_make_clip_ref(clip_id=clip_id, maneuver=maneuver, split="eval", clip_index=clip_index))
    for maneuver, bucket in chosen_by_maneuver.items():
        for clip_id, _labels in bucket:
            clips.append(_make_clip_ref(clip_id=clip_id, maneuver=maneuver, split="train", clip_index=clip_index))
    clips = sorted(clips, key=lambda clip: (clip.split, clip.maneuver, clip.clip_id))
    return Stage1Manifest(
        repo_id=DEFAULT_STAGE1_REPO_ID,
        sample_rate_hz=10,
        min_camera_count=MIN_STAGE1_CAMERA_COUNT,
        clips=tuple(clips),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Select a 30-clip Stage 1 manifest with a 24/6 split.")
    parser.add_argument("--nurec-sample-root", type=Path, default=None)
    parser.add_argument("--clip-index-parquet", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--token-file", type=Path, default=None)
    args = parser.parse_args()

    if args.nurec_sample_root is not None:
        candidates = _load_labels(args.nurec_sample_root)
    else:
        candidates = _load_labels_from_hf(args.token_file)
    clip_index = pd.read_parquet(args.clip_index_parquet)
    try:
        manifest = build_stage1_manifest(candidates=candidates, clip_index=clip_index)
    except ValueError as exc:
        print(f"Exact 10/10/10 manifest unavailable from official labeled overlap: {exc}")
        manifest = build_stage1_manifest_best_effort(candidates=candidates, clip_index=clip_index)
    write_manifest(args.output_manifest, manifest)
    print(f"Wrote Stage 1 manifest to {args.output_manifest}")


if __name__ == "__main__":
    main()

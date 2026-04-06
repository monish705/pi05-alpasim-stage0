from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from alpasim_driver.models.base import (
    BaseTrajectoryModel,
    DriveCommand,
    ModelPrediction,
    PredictionInput,
)
from alpasim_driver.schema import ModelConfig
from openpi.policies import policy_config

from pi05_alpasim_stage0.bridge import rollout_feasible_trajectory
from pi05_alpasim_stage1.bev import (
    CalibrationBundle,
    BEVProjectStats,
    MiDaSDepthEstimator,
    Stage1BEVProjector,
)
from pi05_alpasim_stage1.contracts import (
    ACTION_DIM,
    ACTION_HORIZON,
    ACTIVE_ACTION_DIMS,
    KinematicLimits,
    MIN_STAGE1_CAMERA_COUNT,
    MODEL_DT_SECONDS,
    ROUTE_POINTS,
)
from pi05_alpasim_stage1.openpi_stage1 import make_stage1_train_config

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CameraDiagnostic:
    available: bool
    nonzero: bool
    mean_intensity: float
    shape: tuple[int, int, int]
    source: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "nonzero": self.nonzero,
            "mean_intensity": self.mean_intensity,
            "shape": list(self.shape),
            "source": self.source,
        }


@dataclass(frozen=True)
class Stage1RuntimeConfig:
    trace_log_path: Path | None
    dump_dir: Path | None
    dump_images: bool

    @classmethod
    def from_env(cls) -> "Stage1RuntimeConfig":
        trace_log_raw = os.getenv("PI05_STAGE1_TRACE_LOG")
        dump_dir_raw = os.getenv("PI05_STAGE1_DUMP_DIR")
        dump_images_raw = os.getenv("PI05_STAGE1_DUMP_IMAGES", "0").strip().lower()
        return cls(
            trace_log_path=Path(trace_log_raw) if trace_log_raw else None,
            dump_dir=Path(dump_dir_raw) if dump_dir_raw else None,
            dump_images=dump_images_raw in {"1", "true", "yes", "on"},
        )


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _quat_to_yaw(quaternion: Any) -> float:
    return math.atan2(
        2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y),
        1.0 - 2.0 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z),
    )


def _quat_to_rotmat(quaternion: Any) -> np.ndarray:
    x = float(quaternion.x)
    y = float(quaternion.y)
    z = float(quaternion.z)
    w = float(quaternion.w)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _command_to_prompt(command: DriveCommand) -> str:
    if command == DriveCommand.LEFT:
        return "left_turn"
    if command == DriveCommand.RIGHT:
        return "right_turn"
    return "lane_follow"


def _frame_from_payload(frame: Any) -> np.ndarray:
    if hasattr(frame, "image"):
        image = frame.image
    elif isinstance(frame, tuple) and len(frame) >= 2:
        image = frame[1]
    else:
        raise TypeError(f"Unsupported camera frame type: {type(frame)!r}")
    return np.asarray(image, dtype=np.uint8)


def _frame_shape(frame: np.ndarray) -> tuple[int, int, int]:
    return (int(frame.shape[0]), int(frame.shape[1]), int(frame.shape[2]))


def _camera_status(frame: np.ndarray, *, available: bool, source: str) -> CameraDiagnostic:
    return CameraDiagnostic(
        available=available,
        nonzero=bool(np.any(frame)),
        mean_intensity=float(np.mean(frame)),
        shape=_frame_shape(frame),
        source=source,
    )


def _latest_live_frame(camera_images: dict[str, list[Any]], camera_id: str) -> tuple[np.ndarray | None, CameraDiagnostic]:
    frames = camera_images.get(camera_id, [])
    if not frames:
        return None, CameraDiagnostic(
            available=False,
            nonzero=False,
            mean_intensity=0.0,
            shape=(0, 0, 0),
            source="missing",
        )
    frame = _frame_from_payload(frames[-1])
    return frame, _camera_status(frame, available=True, source="live")


def _build_route_array(route_waypoints: list[Any] | None) -> np.ndarray:
    if not route_waypoints:
        return np.zeros((ROUTE_POINTS, 2), dtype=np.float32)

    route_xy = np.zeros((ROUTE_POINTS, 2), dtype=np.float32)
    usable = min(len(route_waypoints), ROUTE_POINTS)
    for idx in range(usable):
        waypoint = route_waypoints[idx]
        route_xy[idx, 0] = float(waypoint.x)
        route_xy[idx, 1] = float(waypoint.y)
    if usable > 0 and usable < ROUTE_POINTS:
        route_xy[usable:] = route_xy[usable - 1]
    return route_xy


def _build_state_history(
    ego_pose_history: list[Any],
    current_speed_mps: float,
    current_accel_mps2: float,
    *,
    history_steps: int = 10,
) -> np.ndarray:
    if not ego_pose_history:
        return np.zeros((history_steps * 3,), dtype=np.float32)

    padded = list(ego_pose_history[-history_steps:])
    while len(padded) < history_steps:
        padded.insert(0, padded[0])

    speeds = np.zeros((history_steps,), dtype=np.float32)
    yaw_rates = np.zeros((history_steps,), dtype=np.float32)
    accels = np.zeros((history_steps,), dtype=np.float32)

    for idx in range(1, history_steps):
        prev = padded[idx - 1]
        curr = padded[idx]
        dt = max((curr.timestamp_us - prev.timestamp_us) / 1_000_000.0, MODEL_DT_SECONDS)
        dx = float(curr.pose.vec.x - prev.pose.vec.x)
        dy = float(curr.pose.vec.y - prev.pose.vec.y)
        speeds[idx] = float(math.hypot(dx, dy) / dt)
        prev_yaw = _quat_to_yaw(prev.pose.quat)
        curr_yaw = _quat_to_yaw(curr.pose.quat)
        yaw_rates[idx] = float(_wrap_to_pi(curr_yaw - prev_yaw) / dt)
        accels[idx] = float((speeds[idx] - speeds[idx - 1]) / dt)

    speeds[-1] = float(current_speed_mps)
    accels[-1] = float(current_accel_mps2)
    return np.stack([speeds, yaw_rates, accels], axis=-1).reshape(-1)


class _PoseAdapter:
    def __init__(self, pose_proto: Any) -> None:
        self._rot = _quat_to_rotmat(pose_proto.quat)
        self._trans = np.array(
            [float(pose_proto.vec.x), float(pose_proto.vec.y), float(pose_proto.vec.z)],
            dtype=np.float64,
        )

    def apply(self, points_camera: np.ndarray) -> np.ndarray:
        return (self._rot @ points_camera.T).T + self._trans


class _PinholeLikeCamera:
    def __init__(self, intrinsics: Any, actual_resolution_hw: tuple[int, int]) -> None:
        if intrinsics.WhichOneof("camera_param") == "opencv_fisheye_param":
            param = intrinsics.opencv_fisheye_param
        else:
            param = intrinsics.opencv_pinhole_param
        calib_h = max(int(intrinsics.resolution_h), 1)
        calib_w = max(int(intrinsics.resolution_w), 1)
        actual_h, actual_w = actual_resolution_hw
        sx = actual_w / calib_w
        sy = actual_h / calib_h
        self._fx = float(param.focal_length_x) * sx
        self._fy = float(param.focal_length_y) * sy
        self._cx = float(param.principal_point_x) * sx
        self._cy = float(param.principal_point_y) * sy

    def pixel2ray(self, pixels: np.ndarray) -> np.ndarray:
        pixels = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
        x = (pixels[:, 0] - self._cx) / max(self._fx, 1.0e-9)
        y = (pixels[:, 1] - self._cy) / max(self._fy, 1.0e-9)
        rays = np.stack([x, y, np.ones_like(x)], axis=-1)
        norms = np.linalg.norm(rays, axis=1, keepdims=True)
        norms = np.clip(norms, a_min=1.0e-9, a_max=None)
        return rays / norms


class _FthetaCamera:
    def __init__(self, intrinsics: Any, actual_resolution_hw: tuple[int, int]) -> None:
        param = intrinsics.ftheta_param
        calib_h = max(int(intrinsics.resolution_h), 1)
        calib_w = max(int(intrinsics.resolution_w), 1)
        actual_h, actual_w = actual_resolution_hw
        sx = actual_w / calib_w
        sy = actual_h / calib_h
        scale = 0.5 * (sx + sy)
        self._principal_point = np.array(
            [float(param.principal_point_x) * sx, float(param.principal_point_y) * sy],
            dtype=np.float64,
        )
        self._pix_to_angle = np.asarray(param.pixeldist_to_angle_poly, dtype=np.float64) * np.array(
            [scale ** (-idx) for idx in range(len(param.pixeldist_to_angle_poly))],
            dtype=np.float64,
        )
        if param.HasField("linear_cde"):
            linear_c = float(param.linear_cde.linear_c)
            linear_d = float(param.linear_cde.linear_d)
            linear_e = float(param.linear_cde.linear_e)
        else:
            linear_c, linear_d, linear_e = 1.0, 0.0, 0.0
        self._linear_matrix = np.array(
            [[linear_c, linear_d], [linear_e, 1.0]],
            dtype=np.float64,
        )
        self._linear_inv = np.linalg.inv(self._linear_matrix)

    def pixel2ray(self, pixels: np.ndarray) -> np.ndarray:
        pixels = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
        pixel_offsets = pixels - self._principal_point
        offsets = pixel_offsets @ self._linear_inv.T
        radii = np.linalg.norm(offsets, axis=1)
        theta = np.polynomial.polynomial.polyval(radii, self._pix_to_angle)
        unit_xy = np.divide(
            offsets,
            radii[:, None],
            out=np.zeros_like(offsets),
            where=radii[:, None] > 1.0e-9,
        )
        sin_theta = np.sin(theta)
        z = np.cos(theta)
        xy = unit_xy * sin_theta[:, None]
        rays = np.concatenate([xy, z[:, None]], axis=1)
        norms = np.linalg.norm(rays, axis=1, keepdims=True)
        norms = np.clip(norms, a_min=1.0e-9, a_max=None)
        return rays / norms


def _build_calibration_bundle(camera_specs: dict[str, Any], camera_frames: dict[str, np.ndarray]) -> CalibrationBundle:
    camera_models: dict[str, Any] = {}
    sensor_poses: dict[str, Any] = {}
    for camera_id, frame in camera_frames.items():
        camera_spec = camera_specs.get(camera_id)
        if camera_spec is None:
            continue
        intrinsics = camera_spec.intrinsics
        actual_resolution_hw = (int(frame.shape[0]), int(frame.shape[1]))
        camera_param_kind = intrinsics.WhichOneof("camera_param")
        if camera_param_kind == "ftheta_param":
            camera_models[camera_id] = _FthetaCamera(intrinsics, actual_resolution_hw)
        elif camera_param_kind in {"opencv_pinhole_param", "opencv_fisheye_param"}:
            camera_models[camera_id] = _PinholeLikeCamera(intrinsics, actual_resolution_hw)
        else:
            logger.warning("Skipping unsupported camera model %s for %s", camera_param_kind, camera_id)
            continue
        sensor_poses[camera_id] = _PoseAdapter(camera_spec.rig_to_camera)
    return CalibrationBundle(camera_models=camera_models, sensor_poses=sensor_poses)


class Pi05Stage1Model(BaseTrajectoryModel):
    @classmethod
    def from_config(
        cls,
        model_cfg: ModelConfig,
        device: torch.device,
        camera_ids: list[str],
        context_length: int | None,
        output_frequency_hz: int,
    ) -> "Pi05Stage1Model":
        del device
        # Read sim timestep from env var (Hydra rejects unknown config keys).
        # Set STAGE1_SIM_DT to match the actual AlpaSim sim step interval.
        sim_dt = float(os.environ.get("STAGE1_SIM_DT", MODEL_DT_SECONDS))
        return cls(
            checkpoint_dir=model_cfg.checkpoint_path,
            camera_ids=camera_ids,
            context_length=context_length or 1,
            output_frequency_hz=output_frequency_hz,
            sim_dt=sim_dt,
        )

    def __init__(
        self,
        *,
        checkpoint_dir: str,
        camera_ids: list[str],
        context_length: int,
        output_frequency_hz: int,
        sim_dt: float = MODEL_DT_SECONDS,
    ) -> None:
        if len(camera_ids) < 3:
            raise ValueError(
                f"Stage 1 PI driver expects at least 3 cameras, got {camera_ids}"
            )
        if context_length != 1:
            raise ValueError(f"Stage 1 PI driver expects context_length=1, got {context_length}")

        self._camera_ids = list(camera_ids)
        self._required_camera_count = len(self._camera_ids)
        self._context_length_value = context_length
        self._output_frequency_hz_value = output_frequency_hz
        self._sim_dt = sim_dt
        self._limits = KinematicLimits()
        self._runtime_cfg = Stage1RuntimeConfig.from_env()
        self._call_index = 0
        self._projector = Stage1BEVProjector()
        self._depth_estimator = MiDaSDepthEstimator()
        logger.info("Stage 1 sim_dt=%.3fs (training dt=%.3fs)", sim_dt, MODEL_DT_SECONDS)
        self._policy = policy_config.create_trained_policy(
            make_stage1_train_config(
                repo_id="local/stage1_av_driving",
                assets_base_dir="/mnt/data/assets",
                checkpoint_base_dir="/mnt/data/checkpoints",
            ),
            checkpoint_dir,
            default_prompt="drive the route",
        )
        logger.info("Loaded PI 0.5 Stage 1 policy from %s", checkpoint_dir)
        # Pre-warm: absorb the ~23s JIT/CUDA compilation cost during driver startup
        # rather than on the first real policy call.
        self.warm_up()

    @property
    def camera_ids(self) -> list[str]:
        return self._camera_ids

    @property
    def context_length(self) -> int:
        return self._context_length_value

    @property
    def output_frequency_hz(self) -> int:
        return self._output_frequency_hz_value

    def warm_up(self) -> None:
        """Run a dummy inference to trigger JIT compilation and CUDA warmup.

        This absorbs the ~23s first-call cost before the first real session starts,
        so that the first real policy call is already at warm latency (~658ms).
        """
        logger.info("Warming up Stage 1 policy (dummy inference)...")
        warm_t0 = time.perf_counter()
        dummy_bev = np.zeros((200, 200, 3), dtype=np.uint8)
        dummy_state = np.zeros((30,), dtype=np.float32)  # 10 steps * 3 (speed, yaw_rate, accel)
        dummy_route = np.zeros((32, 2), dtype=np.float32)
        obs = {
            "image": {"bev": dummy_bev},
            "state": dummy_state,
            "route": dummy_route,
            "prompt": "lane_follow",
        }
        inference = self._policy.infer(obs)
        warm_ms = (time.perf_counter() - warm_t0) * 1000.0
        logger.info(
            "Stage 1 warm-up complete in %.1fms (policy_infer=%.1fms). "
            "JIT/CUDA caches are now primed.",
            warm_ms,
            float(inference.get("policy_timing", {}).get("infer_ms", 0)),
        )

    def _encode_command(self, command: DriveCommand) -> str:
        return _command_to_prompt(command)

    def _raw_action_summary(self, active_actions: np.ndarray) -> dict[str, Any]:
        names = ("delta_s", "delta_yaw", "target_speed")
        return {
            name: {
                "first": float(active_actions[0, idx]),
                "mean": float(np.mean(active_actions[:, idx])),
                "min": float(np.min(active_actions[:, idx])),
                "max": float(np.max(active_actions[:, idx])),
            }
            for idx, name in enumerate(names)
        }

    def _emit_trace(self, payload: dict[str, Any]) -> None:
        trace_line = json.dumps(payload, separators=(",", ":"), sort_keys=True)
        logger.info("stage1_trace %s", trace_line)
        if self._runtime_cfg.trace_log_path is None:
            return
        self._runtime_cfg.trace_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self._runtime_cfg.trace_log_path.open("a", encoding="utf-8") as handle:
            handle.write(trace_line + "\n")

    def _resolve_live_frames(
        self,
        prediction_input: PredictionInput,
    ) -> tuple[dict[str, np.ndarray], dict[str, CameraDiagnostic]]:
        camera_frames: dict[str, np.ndarray] = {}
        diagnostics: dict[str, CameraDiagnostic] = {}
        for camera_id in self._camera_ids:
            frame, diagnostic = _latest_live_frame(prediction_input.camera_images, camera_id)
            diagnostics[camera_id] = diagnostic
            if frame is not None:
                camera_frames[camera_id] = frame
        if len(camera_frames) < self._required_camera_count:
            raise ValueError(
                "Stage 1 BEV needs all configured live calibrated cameras; "
                f"expected {self._required_camera_count}, got {sorted(camera_frames.keys())}"
            )
        return camera_frames, diagnostics

    def _dump_call_artifacts(
        self,
        *,
        call_index: int,
        time_in: str,
        time_out: str,
        bev_image: np.ndarray,
        bev_stats: BEVProjectStats,
        obs: dict[str, Any],
        camera_diagnostics: dict[str, CameraDiagnostic],
        active_actions: np.ndarray,
        full_actions: np.ndarray,
        trajectory_xy: np.ndarray,
        headings: np.ndarray,
        clamp_report: Any,
        prediction_input: PredictionInput,
        inference: dict[str, Any],
    ) -> None:
        if self._runtime_cfg.dump_dir is None:
            return

        dump_dir = self._runtime_cfg.dump_dir
        dump_dir.mkdir(parents=True, exist_ok=True)
        stem = f"call_{call_index:04d}"
        meta_payload = {
            "call_index": call_index,
            "timestamp_in_utc": time_in,
            "timestamp_out_utc": time_out,
            "camera_status": {
                camera_id: diagnostic.to_dict() for camera_id, diagnostic in camera_diagnostics.items()
            },
            "prompt": obs["prompt"],
            "speed_mps": float(prediction_input.speed),
            "acceleration_mps2": float(prediction_input.acceleration),
            "bev_stats": bev_stats.to_dict(),
            "policy_timing": {
                key: float(value) if isinstance(value, (int, float, np.floating)) else value
                for key, value in inference.get("policy_timing", {}).items()
            },
            "clamp_report": clamp_report.to_dict(),
        }
        (dump_dir / f"{stem}.json").write_text(json.dumps(meta_payload, indent=2, sort_keys=True), encoding="utf-8")

        arrays: dict[str, np.ndarray] = {
            "bev": np.asarray(bev_image, dtype=np.uint8),
            "state": np.asarray(obs["state"], dtype=np.float32),
            "route": np.asarray(obs["route"], dtype=np.float32),
            "active_actions": np.asarray(active_actions, dtype=np.float32),
            "full_actions": np.asarray(full_actions, dtype=np.float32),
            "trajectory_xy": np.asarray(trajectory_xy, dtype=np.float32),
            "headings": np.asarray(headings, dtype=np.float32),
        }
        if self._runtime_cfg.dump_images:
            for camera_id, frame in prediction_input.camera_images.items():
                if frame:
                    arrays[f"camera_{camera_id}"] = _frame_from_payload(frame[-1])
        np.savez_compressed(dump_dir / f"{stem}.npz", **arrays)

    def predict(self, prediction_input: PredictionInput) -> ModelPrediction:
        self._validate_cameras(prediction_input.camera_images)
        self._call_index += 1
        wall_t0 = time.perf_counter()
        time_in = _utc_now_iso()

        camera_frames, camera_diagnostics = self._resolve_live_frames(prediction_input)
        camera_specs = getattr(prediction_input, "camera_specs", None)
        if not camera_specs:
            raise ValueError("Stage 1 BEV requires camera_specs on PredictionInput.")
        calibration = _build_calibration_bundle(camera_specs, camera_frames)
        bev_image, bev_stats = self._projector.build_bev(
            frames_by_camera=camera_frames,
            calibration=calibration,
            depth_estimator=self._depth_estimator,
        )
        obs = {
            "image": {"bev": bev_image},
            "state": _build_state_history(
                prediction_input.ego_pose_history,
                prediction_input.speed,
                prediction_input.acceleration,
            ),
            "route": _build_route_array(getattr(prediction_input, "route_waypoints", None)),
            "prompt": self._encode_command(prediction_input.command),
        }

        inference = self._policy.infer(obs)
        active_actions = np.asarray(inference["actions"], dtype=np.float32)
        if active_actions.shape != (ACTION_HORIZON, 3):
            raise ValueError(f"Expected Stage 1 active actions {(ACTION_HORIZON, 3)}, got {active_actions.shape}")

        full_actions = np.zeros((ACTION_HORIZON, ACTION_DIM), dtype=np.float32)
        full_actions[:, ACTIVE_ACTION_DIMS["delta_s"]] = active_actions[:, 0]
        full_actions[:, ACTIVE_ACTION_DIMS["delta_yaw"]] = active_actions[:, 1]
        full_actions[:, ACTIVE_ACTION_DIMS["target_speed"]] = active_actions[:, 2]
        trajectory_xy, headings, clamp_report = rollout_feasible_trajectory(
            full_actions,
            self._limits,
            initial_speed_mps=prediction_input.speed,
            dt=self._sim_dt,
        )

        wall_t1 = time.perf_counter()
        time_out = _utc_now_iso()
        trace_payload = {
            "call_index": self._call_index,
            "timestamp_in_utc": time_in,
            "timestamp_out_utc": time_out,
            "latency_ms": float((wall_t1 - wall_t0) * 1000.0),
            "policy_infer_ms": float(inference["policy_timing"]["infer_ms"]),
            "prompt": obs["prompt"],
            "bev_stats": bev_stats.to_dict(),
            "camera_status": {
                camera_id: diagnostic.to_dict() for camera_id, diagnostic in camera_diagnostics.items()
            },
            "raw_action_dims_0_2": self._raw_action_summary(active_actions),
            "clamp_report": clamp_report.to_dict(),
            "speed_mps": float(prediction_input.speed),
            "acceleration_mps2": float(prediction_input.acceleration),
        }
        self._emit_trace(trace_payload)
        self._dump_call_artifacts(
            call_index=self._call_index,
            time_in=time_in,
            time_out=time_out,
            bev_image=bev_image,
            bev_stats=bev_stats,
            obs=obs,
            camera_diagnostics=camera_diagnostics,
            active_actions=active_actions,
            full_actions=full_actions,
            trajectory_xy=trajectory_xy,
            headings=headings,
            clamp_report=clamp_report,
            prediction_input=prediction_input,
            inference=inference,
        )

        reasoning = (
            f"prompt={obs['prompt']} infer_ms={inference['policy_timing']['infer_ms']:.1f} "
            f"latency_ms={(wall_t1 - wall_t0) * 1000.0:.1f} "
            f"bev_cameras={bev_stats.camera_count_used} bev_points={bev_stats.points_projected} "
            f"any_clamp={clamp_report.any_clamp} speed_clamps={clamp_report.speed_clamps} "
            f"accel_clamps={clamp_report.accel_clamps}"
        )
        return ModelPrediction(
            trajectory_xy=trajectory_xy,
            headings=headings,
            reasoning_text=reasoning,
        )

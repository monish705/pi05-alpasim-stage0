from __future__ import annotations

import dataclasses
import math
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import (
    BEV_CELL_SIZE_M,
    BEV_CHANNELS,
    BEV_HEIGHT,
    BEV_WIDTH,
    BEV_X_MAX_M,
    BEV_X_MIN_M,
    BEV_Y_MAX_M,
    BEV_Y_MIN_M,
)

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None
    F = None


def _require_torch() -> None:
    if torch is None or F is None:
        raise ImportError("Stage 1 BEV generation requires torch in the active environment.")


@dataclasses.dataclass(frozen=True)
class CalibrationBundle:
    camera_models: dict[str, Any]
    sensor_poses: dict[str, Any]

    def available_camera_ids(self) -> tuple[str, ...]:
        return tuple(
            camera_id
            for camera_id in self.camera_models.keys()
            if camera_id in self.sensor_poses
        )


@dataclasses.dataclass(frozen=True)
class BEVProjectStats:
    camera_count_used: int
    points_projected: int
    occupied_cells: int
    min_depth_m: float
    max_depth_m: float

    def to_dict(self) -> dict[str, int | float]:
        return {
            "camera_count_used": self.camera_count_used,
            "points_projected": self.points_projected,
            "occupied_cells": self.occupied_cells,
            "min_depth_m": self.min_depth_m,
            "max_depth_m": self.max_depth_m,
        }


class MiDaSDepthEstimator:
    """Real monocular depth estimator used for offline Stage 1 lift-splat preprocessing."""

    def __init__(
        self,
        *,
        model_type: str = "DPT_Hybrid",
        device: str | None = None,
    ) -> None:
        _require_torch()
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
        self._model.to(self._device)
        self._model.eval()

        transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        if model_type in {"DPT_Large", "DPT_Hybrid"}:
            self._transform = transforms.dpt_transform
        else:
            self._transform = transforms.small_transform

    @property
    def device(self) -> str:
        return self._device

    def predict_depth(self, image_rgb: np.ndarray) -> np.ndarray:
        _require_torch()
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError(f"Expected HxWx3 RGB image, got {image_rgb.shape}")
        with torch.no_grad():
            batch = self._transform(image_rgb).to(self._device)
            prediction = self._model(batch)
            if prediction.ndim == 3:
                prediction = prediction.unsqueeze(1)
            depth = F.interpolate(
                prediction,
                size=image_rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            )
        depth_np = depth.squeeze().detach().cpu().numpy().astype(np.float32)
        depth_np = np.nan_to_num(depth_np, nan=0.0, posinf=0.0, neginf=0.0)
        finite = depth_np[depth_np > 0]
        if finite.size == 0:
            return np.zeros_like(depth_np, dtype=np.float32)
        q01 = float(np.quantile(finite, 0.01))
        q99 = float(np.quantile(finite, 0.99))
        clipped = np.clip(depth_np, q01, q99)
        # Normalize to a stable pseudo-metric range for projection.
        normalized = (clipped - q01) / max(q99 - q01, 1.0e-6)
        return 1.0 + 49.0 * normalized.astype(np.float32)


@dataclasses.dataclass(frozen=True)
class Stage1BEVProjector:
    pixel_stride: int = 8
    z_min_m: float = -2.5
    z_max_m: float = 3.0
    min_depth_m: float = 1.0
    max_depth_m: float = 50.0
    bev_x_min_m: float = BEV_X_MIN_M
    bev_x_max_m: float = BEV_X_MAX_M
    bev_y_min_m: float = BEV_Y_MIN_M
    bev_y_max_m: float = BEV_Y_MAX_M
    bev_cell_size_m: float = BEV_CELL_SIZE_M
    bev_height: int = BEV_HEIGHT
    bev_width: int = BEV_WIDTH

    def _pixel_grid(self, height: int, width: int) -> np.ndarray:
        ys = np.arange(0, height, self.pixel_stride, dtype=np.float64)
        xs = np.arange(0, width, self.pixel_stride, dtype=np.float64)
        grid_x, grid_y = np.meshgrid(xs, ys)
        return np.stack([grid_x, grid_y], axis=-1).reshape(-1, 2)

    def build_bev(
        self,
        *,
        frames_by_camera: dict[str, np.ndarray],
        calibration: CalibrationBundle,
        depth_estimator: MiDaSDepthEstimator,
    ) -> tuple[np.ndarray, BEVProjectStats]:
        accum_rgb = np.zeros((self.bev_height, self.bev_width, BEV_CHANNELS), dtype=np.float64)
        accum_count = np.zeros((self.bev_height, self.bev_width), dtype=np.float64)
        all_depths: list[np.ndarray] = []
        total_points = 0
        camera_count = 0

        for camera_id in calibration.available_camera_ids():
            frame = frames_by_camera.get(camera_id)
            if frame is None:
                continue
            if frame.ndim != 3 or frame.shape[2] != 3:
                continue

            depth_map = depth_estimator.predict_depth(frame)
            pixels = self._pixel_grid(frame.shape[0], frame.shape[1])
            px = pixels[:, 0].astype(np.int64)
            py = pixels[:, 1].astype(np.int64)
            depth = depth_map[py, px].astype(np.float64)
            valid_depth = np.isfinite(depth) & (depth >= self.min_depth_m) & (depth <= self.max_depth_m)
            if not np.any(valid_depth):
                continue

            pixels = pixels[valid_depth]
            depth = depth[valid_depth]
            colors = frame[py[valid_depth], px[valid_depth]].astype(np.float64) / 255.0

            rays = calibration.camera_models[camera_id].pixel2ray(pixels)
            points_camera = rays * depth[:, None]
            points_ego = calibration.sensor_poses[camera_id].apply(points_camera)

            valid_xyz = (
                np.isfinite(points_ego).all(axis=1)
                & (points_ego[:, 2] >= self.z_min_m)
                & (points_ego[:, 2] <= self.z_max_m)
                & (points_ego[:, 0] >= self.bev_x_min_m)
                & (points_ego[:, 0] < self.bev_x_max_m)
                & (points_ego[:, 1] >= self.bev_y_min_m)
                & (points_ego[:, 1] < self.bev_y_max_m)
            )
            if not np.any(valid_xyz):
                continue

            points_ego = points_ego[valid_xyz]
            colors = colors[valid_xyz]
            depth = depth[valid_xyz]
            all_depths.append(depth.astype(np.float32))
            camera_count += 1
            total_points += int(len(points_ego))

            rows = self.bev_height - 1 - np.floor(
                (points_ego[:, 0] - self.bev_x_min_m) / self.bev_cell_size_m
            ).astype(np.int64)
            cols = np.floor(
                (points_ego[:, 1] - self.bev_y_min_m) / self.bev_cell_size_m
            ).astype(np.int64)
            inside = (
                (rows >= 0)
                & (rows < self.bev_height)
                & (cols >= 0)
                & (cols < self.bev_width)
            )
            rows = rows[inside]
            cols = cols[inside]
            colors = colors[inside]

            np.add.at(accum_rgb[..., 0], (rows, cols), colors[:, 0])
            np.add.at(accum_rgb[..., 1], (rows, cols), colors[:, 1])
            np.add.at(accum_rgb[..., 2], (rows, cols), colors[:, 2])
            np.add.at(accum_count, (rows, cols), 1.0)

        bev = np.zeros((self.bev_height, self.bev_width, BEV_CHANNELS), dtype=np.uint8)
        occupied = accum_count > 0
        if np.any(occupied):
            mean_rgb = np.zeros_like(accum_rgb, dtype=np.float64)
            mean_rgb[occupied] = accum_rgb[occupied] / accum_count[occupied, None]
            occ_norm = np.clip(accum_count / max(float(accum_count.max()), 1.0), 0.0, 1.0)
            bev_float = np.clip(mean_rgb * (0.35 + 0.65 * occ_norm[..., None]), 0.0, 1.0)
            bev = (255.0 * bev_float).astype(np.uint8)

        if all_depths:
            all_depth = np.concatenate(all_depths, axis=0)
            min_depth = float(np.min(all_depth))
            max_depth = float(np.max(all_depth))
        else:
            min_depth = 0.0
            max_depth = 0.0

        stats = BEVProjectStats(
            camera_count_used=camera_count,
            points_projected=total_points,
            occupied_cells=int(np.count_nonzero(occupied)),
            min_depth_m=min_depth,
            max_depth_m=max_depth,
        )
        return bev, stats


def save_qa_grid(
    *,
    output_path: Path,
    rows: list[tuple[str, np.ndarray, np.ndarray]],
) -> None:
    from PIL import Image, ImageDraw

    if not rows:
        raise ValueError("No rows provided for QA grid.")

    tile_w = 512
    tile_h = 256
    canvas = Image.new("RGB", (tile_w * 2, tile_h * len(rows)), color=(8, 12, 20))
    draw = ImageDraw.Draw(canvas)

    for idx, (label, camera_frame, bev_image) in enumerate(rows):
        y0 = idx * tile_h
        camera = Image.fromarray(camera_frame).resize((tile_w, tile_h))
        bev = Image.fromarray(bev_image).resize((tile_w, tile_h))
        canvas.paste(camera, (0, y0))
        canvas.paste(bev, (tile_w, y0))
        draw.rectangle((0, y0, 260, y0 + 26), fill=(0, 0, 0))
        draw.text((8, y0 + 6), label, fill=(255, 255, 255))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)

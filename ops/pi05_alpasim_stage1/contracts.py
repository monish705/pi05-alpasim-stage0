from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


ACTION_HORIZON = 50
ACTION_DIM = 32
MODEL_DT_SECONDS = 0.1
ROUTE_POINTS = 32
# Runtime-configurable sim timestep. Defaults to training dt (0.1s) but can be
# overridden to match the actual AlpaSim simulation step interval (e.g., 0.2s).
SIM_DT_SECONDS: float = MODEL_DT_SECONDS
ACTIVE_ACTION_DIMS = {
    "delta_s": 0,
    "delta_yaw": 1,
    "target_speed": 2,
}
ACTIVE_ACTION_DIM_NAMES = tuple(ACTIVE_ACTION_DIMS.keys())
EXPECTED_SAMPLE_RATE_HZ = 10
BEV_HEIGHT = 200
BEV_WIDTH = 200
BEV_CHANNELS = 3
BEV_CELL_SIZE_M = 0.5
BEV_X_MIN_M = -20.0
BEV_X_MAX_M = 80.0
BEV_Y_MIN_M = -50.0
BEV_Y_MAX_M = 50.0
EGO_HISTORY_STEPS = 10
MIN_STAGE1_CAMERA_COUNT = 6
DEFAULT_STAGE1_REPO_ID = "local/stage1_av_driving"


@dataclass(frozen=True)
class Stage1ClipRef:
    clip_id: str
    raw_chunk: int
    split: str
    maneuver: str
    nurec_release: str = "26.02_release"
    labels_path: str | None = None


@dataclass(frozen=True)
class KinematicLimits:
    max_speed_mps: float = 20.0
    min_speed_mps: float = 0.0
    max_longitudinal_accel_mps2: float = 6.0
    min_longitudinal_accel_mps2: float = -6.0
    max_yaw_rate_radps: float = 0.7
    max_lateral_accel_mps2: float = 4.5
    wheelbase_m: float = 2.9


@dataclass(frozen=True)
class Stage1Paths:
    workspace_root: Path
    dataset_root: Path
    cache_root: Path
    manifest_path: Path
    checkpoint_root: Path
    assets_root: Path
    clip_cache_root: Path = field(init=False)
    nurec_cache_root: Path = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "clip_cache_root", self.cache_root / "raw_av")
        object.__setattr__(self, "nurec_cache_root", self.cache_root / "nurec")


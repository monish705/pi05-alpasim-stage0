from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import ACTION_DIM, ACTION_HORIZON, BEV_CHANNELS, BEV_HEIGHT, BEV_WIDTH

try:
    import openpi.models.pi0_config as pi0_config
    import openpi.training.config as training_config
    import openpi.training.weight_loaders as weight_loaders
    import openpi.transforms as transforms
except ImportError:  # pragma: no cover
    pi0_config = None
    training_config = None
    weight_loaders = None
    transforms = None


def _require_openpi() -> None:
    if pi0_config is None:
        raise ImportError("openpi is required for Stage 1 training utilities. Run this on the remote openpi env.")


def _parse_image(image: Any) -> np.ndarray:
    array = np.asarray(image)
    if np.issubdtype(array.dtype, np.floating):
        array = (255.0 * array).clip(0, 255).astype(np.uint8)
    if array.ndim == 3 and array.shape[0] == 3:
        array = np.transpose(array, (1, 2, 0))
    return array


@dataclasses.dataclass
class LoadBEVFromMemmap:
    _cache: dict[tuple[str, int], np.memmap] = dataclasses.field(default_factory=dict)

    def __call__(self, data: dict) -> dict:
        path = str(data["bev_memmap_path"])
        frame_index = int(np.asarray(data["bev_frame_index"]).reshape(-1)[0])
        num_frames = int(np.asarray(data["bev_num_frames"]).reshape(-1)[0])
        cache_key = (path, num_frames)
        if cache_key not in self._cache:
            self._cache[cache_key] = np.memmap(
                path,
                mode="r",
                dtype=np.uint8,
                shape=(num_frames, BEV_HEIGHT, BEV_WIDTH, BEV_CHANNELS),
            )
        bev = np.asarray(self._cache[cache_key][frame_index], dtype=np.uint8)
        return {
            "image": {"bev": bev},
            "state": data["state"],
            "route": data["route"],
            "actions": data["actions"],
            "prompt": data["prompt"],
        }


@dataclasses.dataclass(frozen=True)
class Stage1DrivingInputs:
    def __call__(self, data: dict) -> dict:
        bev = _parse_image(data["image"]["bev"])
        placeholder = np.zeros_like(bev)
        ego_state = np.asarray(data["state"], dtype=np.float32).reshape(-1)
        route = np.asarray(data["route"], dtype=np.float32).reshape(-1)
        inputs = {
            "state": np.concatenate([ego_state, route], axis=0),
            "image": {
                "base_0_rgb": bev,
                "left_wrist_0_rgb": placeholder,
                "right_wrist_0_rgb": placeholder,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.False_,
                "right_wrist_0_rgb": np.False_,
            },
        }
        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class Stage1DrivingOutputs:
    active_action_dim: int = 3

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"], dtype=np.float32)
        return {"actions": actions[:, : self.active_action_dim]}


@dataclasses.dataclass(frozen=True)
class Stage1DrivingDataConfig(training_config.DataConfigFactory if training_config else object):
    repo_id: str
    assets: Any = dataclasses.field(default_factory=lambda: training_config.AssetsConfig() if training_config else None)
    base_config: Any = None

    def create(self, assets_dirs: Path, model_config: Any) -> Any:
        _require_openpi()
        repack_transform = transforms.Group(
            inputs=[
                transforms.RepackTransform(
                    {
                        "bev_memmap_path": "observation.bev.memmap_path",
                        "bev_frame_index": "observation.bev.frame_index",
                        "bev_num_frames": "observation.bev.num_frames",
                        "state": "observation.state",
                        "route": "observation.route",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                ),
                LoadBEVFromMemmap(),
            ]
        )
        data_transforms = transforms.Group(
            inputs=[Stage1DrivingInputs()],
            outputs=[Stage1DrivingOutputs()],
        )
        model_transforms = training_config.ModelTransformFactory(default_prompt="drive the route")(model_config)
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


def make_stage1_train_config(
    *,
    repo_id: str,
    assets_base_dir: str,
    checkpoint_base_dir: str,
    exp_name: str = "pi05_stage1_av",
    num_train_steps: int = 5000,
    batch_size: int = 8,
) -> Any:
    _require_openpi()
    model = pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
        action_dim=ACTION_DIM,
        action_horizon=ACTION_HORIZON,
        max_token_len=1024,
    )
    return training_config.TrainConfig(
        name="pi05_stage1_av",
        exp_name=exp_name,
        model=model,
        data=Stage1DrivingDataConfig(
            repo_id=repo_id,
            base_config=training_config.DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            action_dim=ACTION_DIM,
            action_horizon=ACTION_HORIZON,
        ).get_freeze_filter(),
        ema_decay=None,
        assets_base_dir=assets_base_dir,
        checkpoint_base_dir=checkpoint_base_dir,
        num_train_steps=num_train_steps,
        batch_size=batch_size,
        save_interval=500,
        log_interval=25,
        keep_period=1000,
        overwrite=True,
        wandb_enabled=True,
        num_workers=8,
        policy_metadata={
            "stage": "stage1",
            "representation": "offline_bev",
            "action_semantics": ["delta_s", "delta_yaw", "target_speed"],
        },
    )

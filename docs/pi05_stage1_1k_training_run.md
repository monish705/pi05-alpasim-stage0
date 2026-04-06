# PI0.5 AlpaSim Stage 1

## 1k Training Run Summary

This document records the first real Stage 1 training run that used:

- real NVIDIA PhysicalAI AV / NuRec data
- offline BEV preprocessing from the surround-camera rig
- OpenPI JAX `pi0.5` LoRA fine-tuning
- a saved checkpoint at step `1000`

This is a real run summary only. It does not claim Stage 1 AlpaSim closed-loop success yet.

## Run Identity

- server: `ubuntu@185.216.21.7`
- GPU: `NVIDIA RTX A6000`
- active env: `/mnt/data/venvs/stage1`
- repo root on server: `/mnt/data/repos/new-project`
- OpenPI root on server: `/mnt/data/repos/openpi`
- train log:
  - `/mnt/data/logs/stage1/train_stage1.log`
- offline WandB run:
  - `/mnt/data/repos/new-project/wandb/offline-run-20260402_080446-yj7li3p0`

## Dataset Used

- repo id: `local/stage1_av_driving`
- sample rate: `10 Hz`
- minimum camera count: `6`
- dataset root:
  - `/mnt/data/datasets/stage1_fast_real_v1`
- dataset size on disk: about `151 MB`
- manifest:
  - `/mnt/data/assets/stage1_manifest_12clips.json`
- QA image:
  - `/mnt/data/assets/stage1_qa/full_bev_qa.png`

### Split

- train clips: `9`
- eval clips: `3`

### Eval clips

- `lane_follow`: `0fd2c051-f5e1-4416-9bb5-9b93d92f55fb`
- `left_turn`: `f0674f19-6030-4e5c-9031-cd2528c35e06`
- `right_turn`: `4bad2f63-1bab-4dbf-b55d-4a6606969103`

### Train clips

- `lane_follow`: `01d503d4-449b-46fc-8d78-9085e70d3554`
- `lane_follow`: `048b974e-1546-488a-b8f9-d32bff77f5aa`
- `left_lane_change`: `1c841180-ba2a-41f0-836b-650e535d8741`
- `left_lane_change`: `4ce9f18b-d251-4bfc-a72d-58bac48fd525`
- `right_lane_change`: `0ce6f2a3-a6cf-42c3-9133-111139be1dd1`
- `left_turn`: `2854bbf0-3dac-431a-9e48-4c26ebcd1c77`
- `left_turn`: `adb899bd-45f6-420f-b04a-79eb32a02e92`
- `right_turn`: `07981e6a-22dd-4796-ad2f-1252037ecd28`
- `right_turn`: `2c9a4206-432a-45f6-8507-98a8648621ca`

## Model / Train Config

From the real run config in the offline WandB debug log:

- run name: `pi05_stage1_av`
- project name: `openpi`
- model family: `pi0.5`
- `action_dim = 32`
- `action_horizon = 50`
- `max_token_len = 1024`
- dtype: `bfloat16`
- `paligemma_variant = gemma_2b_lora`
- `action_expert_variant = gemma_300m_lora`
- `pi05 = True`
- `discrete_state_input = True`
- base weights:
  - `gs://openpi-assets/checkpoints/pi05_base/params`
- repo id:
  - `local/stage1_av_driving`
- assets base dir:
  - `/mnt/data/assets/stage1`
- checkpoint base dir:
  - `/mnt/data/checkpoints/stage1`
- seed: `42`
- batch size: `8`
- num workers: `8`
- requested train steps: `5000`
- log interval: `25`
- save interval: `500`
- keep period: `1000`
- WandB mode: `offline`

### Optimizer / LR schedule

- warmup steps: `1000`
- peak lr: `2.5e-05`
- decay steps: `30000`
- decay lr: `2.5e-06`
- Adam-style params:
  - `b1 = 0.9`
  - `b2 = 0.95`
  - `eps = 1e-08`
- weight decay: `1e-10`
- gradient clip norm: `1.0`

### Policy metadata

- stage: `stage1`
- representation: `offline_bev`
- action semantics:
  - `delta_s`
  - `delta_yaw`
  - `target_speed`

## Stage 1 Input Contract Used

The real training batch used:

- images:
  - `base_0_rgb`: BEV image
  - `left_wrist_0_rgb`: zero placeholder, mask `False`
  - `right_wrist_0_rgb`: zero placeholder, mask `False`
- image masks for all three image keys
- state vector: `(8, 94)`
- tokenized prompt: `(8, 1024)`
- actions: `(8, 50, 32)`

Important:

- Stage 1 still uses BEV as the real visual input
- the extra two image slots were only added because OpenPI requires a full image-key dict
- this fixed a schema issue only; it did not change the model head or training target

## Norm Stats

Computed from the real Stage 1 dataset and stored under:

- `/mnt/data/assets/stage1/pi05_stage1_av/local/stage1_av_driving`

Active-action summary:

- `delta_s`
  - mean: `0.6571820974349976`
  - std: `0.4613502323627472`
  - q01: `0.0`
  - q99: `1.6812640031337738`
- `delta_yaw`
  - mean: `0.004950278904289007`
  - std: `0.015823908150196075`
  - q01: `-0.021419895529747008`
  - q99: `0.060551427958905696`
- `target_speed`
  - mean: `6.571818828582764`
  - std: `4.613505840301514`
  - q01: `0.0`
  - q99: `16.812639571762087`

## Runtime / Memory

- GPU memory during training: about `36.8 GB`
- first compile was slow, then steady-state training settled near:
  - `~3.0 s / step`

## Training Timeline

- run start: approximately `2026-04-02 08:04:46`
- first real training progress after compile: around `08:07`
- `500` checkpoint finalized: around `08:32:31`
- `1000` checkpoint finalized: around `08:57:33`
- run stopped immediately after the `1000` checkpoint was preserved

Approximate wall time from start to `1000`:

- about `52 minutes`

## Checkpoints

Saved checkpoint for evaluation:

- `/mnt/data/checkpoints/stage1/pi05_stage1_av/pi05_stage1_av/1000`

Checkpoint size:

- about `9.0 GB`

Checkpoint contents:

- `params`
- `train_state`
- embedded assets metadata including `norm_stats.json`

The earlier `500` checkpoint was deleted after `1000` completed because of the checkpoint retention policy.

## Scalar Training Metrics

The plain training log did not print the scalar lines directly, but the real offline WandB run artifact contains them. The following values were extracted from:

- `/mnt/data/repos/new-project/wandb/offline-run-20260402_080446-yj7li3p0/run-yj7li3p0.wandb`

Recorded scalar snapshots:

- step `100`: `grad_norm=3.1207`, `loss=0.6971`, `param_norm=1803.7703`
- step `125`: `grad_norm=3.2542`, `loss=0.6141`, `param_norm=1803.7706`
- step `150`: `grad_norm=2.6234`, `loss=0.4921`, `param_norm=1803.7706`
- step `175`: `grad_norm=2.2735`, `loss=0.3763`, `param_norm=1803.7708`
- step `200`: `grad_norm=2.0312`, `loss=0.2971`, `param_norm=1803.7709`
- step `225`: `grad_norm=1.7094`, `loss=0.2231`, `param_norm=1803.7720`
- step `250`: `grad_norm=1.4752`, `loss=0.1596`, `param_norm=1803.7720`
- step `275`: `grad_norm=1.1230`, `loss=0.1084`, `param_norm=1803.7721`
- step `300`: `grad_norm=0.9705`, `loss=0.0777`, `param_norm=1803.7728`
- step `325`: `grad_norm=0.7979`, `loss=0.0565`, `param_norm=1803.7731`
- step `350`: `grad_norm=0.6274`, `loss=0.0456`, `param_norm=1803.7732`
- step `375`: `grad_norm=0.5738`, `loss=0.0367`, `param_norm=1803.7734`
- step `400`: `grad_norm=0.4868`, `loss=0.0324`, `param_norm=1803.7736`
- step `425`: `grad_norm=0.5134`, `loss=0.0310`, `param_norm=1803.7743`
- step `450`: `grad_norm=0.4621`, `loss=0.0297`, `param_norm=1803.7747`
- step `475`: `grad_norm=0.3848`, `loss=0.0229`, `param_norm=1803.7759`
- step `500`: `grad_norm=0.3615`, `loss=0.0200`, `param_norm=1803.7759`
- step `525`: `grad_norm=0.3383`, `loss=0.0191`, `param_norm=1803.7769`
- step `550`: `grad_norm=0.3522`, `loss=0.0173`, `param_norm=1803.7771`
- step `575`: `grad_norm=0.3312`, `loss=0.0166`, `param_norm=1803.7775`
- step `600`: `grad_norm=0.3090`, `loss=0.0148`, `param_norm=1803.7778`
- step `625`: `grad_norm=0.3226`, `loss=0.0163`, `param_norm=1803.7797`
- step `650`: `grad_norm=0.2791`, `loss=0.0142`, `param_norm=1803.7798`
- step `675`: `grad_norm=0.2619`, `loss=0.0119`, `param_norm=1803.7808`
- step `700`: `grad_norm=0.2710`, `loss=0.0115`, `param_norm=1803.7812`
- step `725`: `grad_norm=0.2749`, `loss=0.0122`, `param_norm=1803.7821`
- step `750`: `grad_norm=0.2696`, `loss=0.0109`, `param_norm=1803.7836`
- step `775`: `grad_norm=0.2675`, `loss=0.0115`, `param_norm=1803.7845`
- step `800`: `grad_norm=0.2394`, `loss=0.0092`, `param_norm=1803.7852`
- step `825`: `grad_norm=0.2319`, `loss=0.0094`, `param_norm=1803.7863`
- step `850`: `grad_norm=0.2115`, `loss=0.0082`, `param_norm=1803.7876`
- step `875`: `grad_norm=0.2615`, `loss=0.0100`, `param_norm=1803.7886`
- step `900`: `grad_norm=0.1976`, `loss=0.0069`, `param_norm=1803.7891`
- step `925`: `grad_norm=0.2172`, `loss=0.0081`, `param_norm=1803.7905`

### Training read

This run shows a strong monotonic reduction in total loss from:

- `0.6971` at step `100`

to:

- `0.0069` at step `900`

with small local noise but an overall healthy downward trend.

Important limitation:

- these are total training losses only
- this run did not yet export separate scalar histories for:
  - `delta_s`
  - `delta_yaw`
  - `target_speed`

So the `1k` run proves optimization is working, but it does not yet isolate the Stage 1 turning signal by head.

## What This Run Proves

- the real Stage 1 BEV dataset was built successfully
- OpenPI `pi0.5` LoRA training ran successfully on the A6000
- the Stage 1 input schema is valid after adding masked placeholder image keys
- a real `1000`-step checkpoint exists and is ready for evaluation
- the run is not a mockup or placeholder path

## What Is Still Missing

- Stage 1 AlpaSim runtime evaluation from checkpoint `1000`
- per-head validation metrics, especially `delta_yaw`
- closed-loop turning metrics on held-out eval scenes


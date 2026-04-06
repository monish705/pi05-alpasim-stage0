# PI0.5 Stage 1 Handoff

Date: 2026-04-03

## Goal Right Now

Run a **real Stage 1 closed-loop AlpaSim test** with:

- real Stage 1 checkpoint `1000`
- real live BEV inside the external driver
- **6 or 7 runtime cameras**, not 4
- saved MP4, trace, metrics, logs, and BEV dumps

## What Is Already Proven

### 1. Stage 1 training is real

- Model family: OpenPI `pi0.5` LoRA
- Checkpoint saved: `/mnt/backup50/pi05_stage1_fast_20260402/checkpoints/stage1/pi05_stage1_av/pi05_stage1_av/1000`
- Real BEV dataset saved: `/mnt/backup50/pi05_stage1_fast_20260402/datasets/stage1_fast_real_v1`
- Real Stage 1 env saved: `/mnt/backup50/pi05_stage1_fast_20260402/venvs/stage1`

Training truth:

- Stage 1 was trained on **offline BEV**, not raw camera tensors directly
- The BEV was built from **real NuRec/AV multi-camera clips**
- Stage 1 dataset build enforced `MIN_STAGE1_CAMERA_COUNT = 6`

Code proof:

- [contracts.py](C:\Users\brind\Documents\New project\ops\pi05_alpasim_stage1\contracts.py)
- [build_stage1_dataset.py](C:\Users\brind\Documents\New project\ops\pi05_alpasim_stage1\build_stage1_dataset.py)

Data proof:

- [stage1_checkpoint_1000_eval.json](C:\Users\brind\Documents\New project\artifacts\stage1_checkpoint_1000_eval.json)

That eval JSON shows 7 real camera features on held-out samples:

- `camera_cross_left_120fov`
- `camera_cross_right_120fov`
- `camera_front_tele_30fov`
- `camera_front_wide_120fov`
- `camera_rear_left_70fov`
- `camera_rear_right_70fov`
- `camera_rear_tele_30fov`

### 2. A real Stage 1 closed-loop run already completed

Completed run:

- Scene: `clipgt-048b974e-1546-488a-b8f9-d32bff77f5aa`
- Runtime mode: official `local_external_driver`
- Result: **session completed**

Important boundary:

- this completed run used **4 live scene cameras**
- not because AlpaSim only allows 4
- because this specific scene only exposed 4 valid runtime cameras

Artifacts on PC:

- [Run Doc](C:\Users\brind\Documents\New project\docs\pi05_stage1_e2e_run_20260403.md)
- [Run Dashboard](C:\Users\brind\Documents\New project\docs\pi05_stage1_e2e_run_20260403.html)
- [MP4](C:\Users\brind\Documents\New project\artifacts\stage1_e2e_20260403\stage1_scene4cam_run.mp4)
- [Trace](C:\Users\brind\Documents\New project\artifacts\stage1_e2e_20260403\trace.jsonl)
- [Driver Log](C:\Users\brind\Documents\New project\artifacts\stage1_e2e_20260403\driver.log)
- [Wizard Log](C:\Users\brind\Documents\New project\artifacts\stage1_e2e_20260403\wizard.log)
- [Runtime Log](C:\Users\brind\Documents\New project\artifacts\stage1_e2e_20260403\runtime_worker_0.log)

Key 4-camera run numbers:

- `n_policy_calls = 97`
- `progress_last = 0.5392535088220874`
- `dist_traveled_m_last = 60.08402543040564`
- `offroad_last = 0.0`
- `wrong_lane_last = 1.0`
- warm policy inference about `78.6 ms`
- warm end-to-end latency about `658.4 ms`

## Current 7-Camera Work

### Official 7-camera runtime rig

The official wizard already has:

- `src/wizard/configs/cameras/stage1_7cam.yaml`

That rig is:

- `camera_cross_left_120fov`
- `camera_front_wide_120fov`
- `camera_cross_right_120fov`
- `camera_rear_left_70fov`
- `camera_rear_tele_30fov`
- `camera_rear_right_70fov`
- `camera_front_tele_30fov`

This matches the Stage 1 training/eval rig much better than the 4-camera scene run.

### What was changed to support this

The launcher was parameterized so the same official run path can be reused for 4-camera or 7-camera configs without editing it again:

- [tmp_stage1_e2e_run.sh](C:\Users\brind\Documents\New project\tmp_stage1_e2e_run.sh)

New helper scripts added:

- [tmp_stage1_7cam_probe_remote.sh](C:\Users\brind\Documents\New project\tmp_stage1_7cam_probe_remote.sh)
- [tmp_stage1_find_7cam_scene_remote.sh](C:\Users\brind\Documents\New project\tmp_stage1_find_7cam_scene_remote.sh)

These were used to:

- start the real Stage 1 driver with `stage1_pi05`
- run the official wizard with `+cameras=stage1_7cam`
- stop only when a scene actually supports that full rig

### Current 7-camera blocker

The **remaining blocker is not the model**.

It is:

- finding a `clipgt-*` AlpaSim scene whose runtime camera catalog actually defines the full Stage 1 7-camera rig

The official scene scan started and produced real failures of the form:

- `camera_rear_tele_30fov` not defined
- `camera_rear_left_70fov` not defined

This means:

- the probe method is correct
- the scene choice is wrong
- we need to keep scanning until a scene truly defines the full 7-camera rig

### 7-camera probe runs reached so far

These runs are already saved on attached storage under:

- `/mnt/backup50/pi05_stage1_fast_20260402/stage1_e2e/runs`

Confirmed scene failures:

1. `clipgt-026d6a39-bd8f-4175-bc61-fe50ed0403a3`
   - failed because `camera_rear_tele_30fov` is not defined
   - run dir: `/mnt/data/stage1_e2e/runs/stage1_1000_20260403_113348`

2. `clipgt-04749bb9-9b37-495b-bed0-77f0e33ac7da`
   - failed because `camera_rear_left_70fov` is not defined
   - run dir: `/mnt/data/stage1_e2e/runs/stage1_1000_20260403_115126`

3. `clipgt-048b974e-1546-488a-b8f9-d32bff77f5aa`
   - failed because `camera_rear_left_70fov` is not defined
   - run dir: `/mnt/data/stage1_e2e/runs/stage1_1000_20260403_115302`

4. `clipgt-0499fb41-122d-4180-83af-f954a9974d3b`
   - failed because `camera_rear_left_70fov` is not defined
   - run dir: `/mnt/data/stage1_e2e/runs/stage1_1000_20260403_115508`

At wrap-up time, all active probes were stopped cleanly.

## BEV Status

### What is real

- Stage 1 driver builds **live BEV** during closed-loop runs
- dumps are enabled with:
  - `PI05_STAGE1_DUMP_DIR`
  - `PI05_STAGE1_DUMP_IMAGES=1`

The completed 4-camera run already wrote BEV dumps under its run directory.

### What is weak

The current BEV representation is still likely underpowered:

- MiDaS monocular depth
- sparse projection
- RGB splatting

So the current plan remains:

1. finish a **real 6/7-camera closed-loop test first**
2. then upgrade BEV quality

## Runtime / Infrastructure Status

### Working

- official AlpaSim repo and Docker image build
- Docker storage moved to large attached volume
- real Stage 1 env restored
- real Stage 1 checkpoint restored
- official external-driver path working
- real 4-camera Stage 1 closed-loop completion proven

### Not the blocker anymore

- not HF token
- not storage
- not checkpoint restore
- not driver startup
- not official runtime setup

### Actual blocker right now

- **scene-to-camera-catalog mismatch for the 7-camera rig**

## Important Diagnosis

### Training-side truth

- Stage 1 training really used 6-7 camera source data
- via BEV, not raw camera tensors directly

### Runtime-side truth

- runtime scenes in AlpaSim do not automatically expose the same richer rig
- you have to pick a scene whose camera catalog actually defines the cameras you request

### What is proven vs not proven

Proven:

- real Stage 1 checkpoint training
- real held-out Stage 1 offline eval
- real Stage 1 closed-loop 4-camera run
- real Stage 1 live BEV in driver

Not proven yet:

- real Stage 1 closed-loop **6/7-camera** run
- whether a matching scene exists in the current scanned subset

## Next Exact Step

Resume the scan from the next unseen official scene using:

- `/home/ubuntu/tmp_stage1_find_7cam_scene_remote.sh`

If a scene succeeds with all 7 cameras:

1. rerun that scene with larger `SIM_STEPS`
2. keep BEV dumps on
3. collect MP4, metrics, trace, driver log, runtime log
4. export BEV frames/video for inspection

## Saved Locations

Main attached storage root:

- `/mnt/backup50/pi05_stage1_fast_20260402`

Most important saved items:

- checkpoint: `/mnt/backup50/pi05_stage1_fast_20260402/checkpoints/stage1/pi05_stage1_av/pi05_stage1_av/1000`
- dataset: `/mnt/backup50/pi05_stage1_fast_20260402/datasets/stage1_fast_real_v1`
- env: `/mnt/backup50/pi05_stage1_fast_20260402/venvs/stage1`
- repos: `/mnt/backup50/pi05_stage1_fast_20260402/repos`
- run dirs: `/mnt/backup50/pi05_stage1_fast_20260402/stage1_e2e/runs`

## Final State At Handoff

- GPU-side processes stopped
- Docker containers for Stage 1 probes stopped
- latest scripts prepared for resumed 7-camera probing
- all important model/data/progress still preserved on the 50 GB attached storage

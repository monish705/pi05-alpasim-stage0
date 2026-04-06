# PI0.5 AlpaSim Stage Progress

This repository contains the code, docs, and selected artifacts for a real PI0.5-to-AlpaSim transfer project.

## What is proven

- Stage 0 same-scene closed-loop transfer worked end-to-end.
- Stage 1 LoRA training ran to checkpoint `1000`.
- Stage 1 offline eval showed real turn geometry signal on held-out clips.
- One real Stage 1 closed-loop AlpaSim rollout completed on a scene-valid 4-camera runtime rig.

## What is not proven yet

- A real closed-loop Stage 1 rollout on a full 6/7-camera runtime scene.
- Strong final driving performance.
- A recovered copy of the lost trained checkpoint storage.

## Start Here

- [Docs index](docs/README.md)
- [Public artifacts](artifacts/public/README.md)
- [Stage 1 handoff](docs/pi05_stage1_handoff_20260403.md)
- [Stage 1 1k training run](docs/pi05_stage1_1k_training_run.md)
- [Stage 1 4-camera E2E run](docs/pi05_stage1_e2e_run_20260403.md)
- [Architecture dashboard](docs/pi05_alpasim_stage0_architecture_dashboard.html)

## Code

- `ops/pi05_alpasim_stage1/` contains the Stage 1 dataset, BEV, training, and manifest code.
- `alpasim_pi05_driver/` contains the external driver and runtime configs.

## Current blocker

The next important runtime test is finding a `clipgt-*` AlpaSim scene that actually exposes the full Stage 1 7-camera rig.

## Notes

- This repo is intentionally transparent about the lost checkpoint and the remaining runtime mismatch.
- Large raw run bundles are not part of the curated public surface. See [public artifacts](artifacts/public/README.md).

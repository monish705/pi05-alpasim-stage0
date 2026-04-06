# PI0.5-to-Driving Research Memo

## Summary
We built a real closed-loop transfer stack that adapts `pi0.5` into a custom driving policy inside NVIDIA AlpaSim. The current result is not a driving-performance claim. It is a research validation result: the fine-tuned policy loads, runs in the simulator loop, emits repeated action chunks, and completes same-scene rollouts with full logs, traces, metrics, and video.

The most important technical question now is not whether the stack exists. It does. The open question is whether this transfer path can become genuinely vision-grounded for driving, rather than staying dominated by route conditioning, bridge stabilization, and controller tracking.

## Thesis
Current AV policies are usually trained natively for driving. Embodied foundation models such as `pi0.5` may offer a broader prior over perception and action, but it is still unclear whether that prior transfers usefully to driving.

Our research thesis is:

> A broadly pretrained embodied VLA can be adapted into a useful driving policy substrate, and with the right training and evaluation setup may become more data-efficient than narrowly trained driving-native policies.

Stage 0 tests the first part of that thesis only:

> Can `pi0.5` be remapped into a real closed-loop driving stack and produce executable control in simulator?

## What We Built
- A Stage 0 local AV dataset pipeline using a tiny NVIDIA-gated subset.
- A shape-preserving `pi0.5` fine-tune path with LoRA on the JAX `openpi` stack.
- A custom AlpaSim external driver that restores the fine-tuned checkpoint and serves live inference.
- A kinematic bridge that converts active model outputs into feasible driving trajectories.
- A run-analysis stack with per-call traces, rollout metrics, simulator logs, and video artifacts.

## Exact Stage 0 Setup
- Base model: `pi0.5`
- Fine-tune mode: LoRA
- Base checkpoint: `gs://openpi-assets/checkpoints/pi05_base/params`
- Eval checkpoint: `/mnt/data/checkpoints/pi05_stage0_av/pi05_stage0_av/1000`
- Action tensor: `50 x 32`
- Active action dims: `delta_s`, `delta_yaw`, `target_speed`
- Dataset size: `5 episodes`, `545 total frames`, `10 Hz`
- Train steps: `2500`
- Batch size: `8`
- Token cap: `1024`
- Eval mode: AlpaSim `local_external_driver`
- Main eval scene: `clipgt-048b974e-1546-488a-b8f9-d32bff77f5aa`

## What We Proved
From the run logs and simulator traces, we can support these claims:

- The fine-tuned `pi0.5` checkpoint was actually restored by the custom external driver.
- AlpaSim actually connected to that driver over `localhost:6789`.
- The policy emitted repeated nontrivial action outputs across the rollout.
- The simulator consumed those outputs through repeated `PolicyEvent -> ControllerEvent -> PhysicsEvent -> StepEvent` cycles.
- The session completed successfully and produced full rollout artifacts.

This is a real closed-loop execution result, not just a static demo or an offline replay.

## Main Quantitative Result
Baseline same-scene rollout:

- `collision_any = 0.0`
- `offroad = 1.0`
- `wrong_lane = 1.0`
- `progress = 0.4772283417512127`
- `dist_traveled_m = 56.191953509330936`
- `plan_deviation = 0.46298382270814414`
- `n_policy_calls = 101`

This is enough to show nontrivial control. It is not enough to claim strong driving quality.

## Ablation Result
Official A/B/C/D camera ablation on the same scene:

- `A` all cameras live
- `B` front camera only
- `C` all cameras black
- `D` wrong-scene static override

Key outputs:

- `A progress_last = 0.4772`
- `B progress_last = 0.4382`
- `C progress_last = 0.5473`
- `D progress_last = 0.5667`

- `A target_speed_first_mean = 5.8232`
- `C target_speed_first_mean = 8.5896`

- `A accel_clamps_total = 559`
- `C accel_clamps_total = 1028`

This suggests:

- vision is affecting the policy
- real cameras reduce aggressiveness and acceleration-clamp pressure
- but the current stack is not yet visually grounded enough to support a strong route-following performance claim

## Main Bottlenecks
- Route conditioning appears too strong relative to visual grounding.
- The bridge is still rescuing many outputs.
- End-to-end latency is too high for a clean `10 Hz` loop.
- Offroad evaluation is still contaminated by spawn/eval alignment.
- Current results are same-scene only.

## Why This Still Matters
Even with those limitations, the work has clear value:

- It de-risks the core transfer pipeline.
- It shows that embodied VLA adaptation to driving is technically executable.
- It produces raw evidence rather than vague intuition.
- It identifies the next decisive bottleneck precisely enough to guide the next phase.

This is the right stage for research funding, compute support, and technical advisor validation.

## What We Need Next
The next milestone is not a larger narrative. It is a sharper experiment:

- fix spawn/offroad contamination
- repeat A/B/C/D under cleaner prompt control
- reduce route leakage
- run multiple trials per condition
- verify whether live vision becomes measurably beneficial beyond route-only behavior

That is the experiment that determines whether this becomes a stronger research paper and a more fundable technical direction.

## What We Are Asking For
We are not asking people to underwrite a finished AV product story.

The right ask at this stage is:

- compute credits
- a few months of research runway
- technical feedback
- introductions to researchers, autonomy engineers, and technically strong angels

## Links
- Dashboard: [pi05_alpasim_stage0_dashboard.html](C:\Users\brind\Documents\New project\docs\pi05_alpasim_stage0_dashboard.html)
- Full report: [pi05_alpasim_stage0_complete_report.html](C:\Users\brind\Documents\New project\docs\pi05_alpasim_stage0_complete_report.html)
- Full paper draft: [pi05_alpasim_stage0_full_paper.md](C:\Users\brind\Documents\New project\docs\pi05_alpasim_stage0_full_paper.md)

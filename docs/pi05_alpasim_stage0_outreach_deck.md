# PI0.5-to-Driving Outreach Deck

## Slide 1: Thesis
**Title**
Embodied VLA Transfer to Driving

**Core line**
We are testing whether a broadly pretrained embodied VLA can become a better substrate for driving adaptation than narrowly trained driving-native policies.

**What to say**
- This is a research-phase project, not a finished autonomy company.
- The immediate question is transfer feasibility and visual grounding.
- Stage 0 already proves the stack runs end-to-end in closed loop.

## Slide 2: What We Built
**Title**
Closed-Loop Stack Is Real

**Bullets**
- `pi0.5` fine-tuned with LoRA on a tiny NVIDIA AV subset
- Native action tensor preserved, driving semantics mapped into active dims
- Custom external AlpaSim driver
- Kinematic bridge with feasibility clamps
- Full logs, traces, metrics, and rollout videos

**What to say**
- This is not an offline notebook result.
- The model is actually in the simulator loop and controlling the car.

## Slide 3: Stage 0 Result
**Title**
Same-Scene Closed-Loop Execution Works

**Bullets**
- `101` policy calls during a completed rollout
- `collision_any = 0.0`
- `progress = 0.4772`
- `dist_traveled_m = 56.19`
- session completed with full event-cycle logs

**What to say**
- The point of Stage 0 was to validate real execution.
- That goal is met.
- The current quality is nontrivial, but not yet a strong driving result.

## Slide 4: What Failed and Why It Matters
**Title**
Main Bottleneck: Vision Grounding

**Bullets**
- all-black and wrong-scene ablations still perform too well
- route conditioning and bridge/controller effects are still too dominant
- real cameras reduce speed and clamp pressure, so vision is active
- but visual input is not yet strong enough to produce clearly better route-following behavior

**What to say**
- This is exactly the kind of intermediate result that matters.
- It tells us what the next experiment must solve.

## Slide 5: Why Support This Now
**Title**
High-Leverage Research Phase

**Bullets**
- core stack risk is already reduced
- next phase is clear and technically focused
- modest compute/support can produce a much more decisive result
- strongest immediate support types:
  - compute credits
  - research runway
  - technical advisors
  - early-stage research angels

**What to say**
- We are not claiming product-market fit.
- We are asking whether this research direction is important enough to back through the next milestone.

## Optional Slide 6: Exact Ask
**Title**
What We Need

**Bullets**
- 3-6 months of focused research runway
- GPU/compute support
- feedback from autonomy / embodied AI experts
- introductions to technically strong angels or research backers

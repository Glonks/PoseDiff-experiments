# PoseDiff-experiments

This project aims to understand some of the design decisions implemented in the paper
"PoseDiff: A Unified Diffusion Model Bridging Robot Pose Estimation and Video-To-Action Control"
and (maybe) provide support for them, and also explore some directions that we hypothesize can
improve performance.

## Model taxonomy
- Baseline: DDPM with 100 timesteps
- M1: DDPM with 1000 timesteps
- M2: DPM++ with 1000 timesteps
- M3: DPM++ (+ CFG) with 1000 timesteps
- M4: DDPM (+ CFG) with 1000 timesteps [Stretch]

## Things implemented
- DDPM sampler
- Conditional UNet
- CFG wrapper
- PoseDiff orchestration

## Things to do
- Implement:
  - DPM++ sampler [Aayush]
  - Dataset [Kushal]
  - Optical flow based feature extractor [Shubh]
  - Training machinery (Trainer)
  - Evaluation machinery
- Test:
  - Dataset
  - DDPM sampler
  - Conditional UNet
  - CFG wrapper
  - PoseDiff orchestration
  - DPM++ sampler
  - Optical flow based feature extractor
  - Training machinery
  - Evaluation machinery
- Figure out the compute
- Train:
  - Baseline
  - M1
  - M2
  - M3
  - (Maybe) M4
- Write the report
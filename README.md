# PoseDiff-experiments

This project aims to understand some of the design decisions implemented in the paper
"PoseDiff: A Unified Diffusion Model Bridging Robot Pose Estimation and Video-To-Action Control"
and (maybe) provide support for them, and also explore some directions that we hypothesize can
improve performance.

## Directions to explore
- D1: Increasing timesteps to make it easier to learn
- D2: Use CFG to reduce pose diversity
- D3: Using historical information (in some form) to make more informed decisions

## Model taxonomy
- Baseline: DDPM with 100 timesteps
- M1: DDPM with 1000 timesteps [D1]
- M2: DPM++ with 1000 timesteps [D1] [Recover some performance]
- M3: DPM++ (+ CFG) with 1000 timesteps [D2]
- M4: DDPM (+ CFG) with 1000 timesteps [D2] [Stretch]
- M5: DDPM with 100 timesteps and historical features [D3]
- M6: DDPM with 1000 timesteps and historical features [D3] [Stretch]

## Things implemented
- DDPM sampler
- Conditional UNet
- CFG wrapper
- PoseDiff orchestration

## Things to do
- Implement:
  - Have the dataset produce flattened keypoint data
  - DPM++ sampler [Aayush]
  - LogSNR scheduling
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
  - M5
  - (Maybe) M6
- Write the report
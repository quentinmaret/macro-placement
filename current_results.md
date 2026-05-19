# Experiment Results Log

## Current Safe Baseline

Date: 2026-05-19  
Branch/commit: TODO  
Command: TODO  

- Full IBM AVG proxy: 1.4938
- Full IBM overlaps: 0
- Full IBM validity: all VALID
- NG45 AVG proxy: 0.8418
- Notes: current safe fallback; spacing/logging patch committed.

## Spacing Calibration — 2026-05-19

Output directory:
`experiments/spacing_calibration_20260519_131220/`

### NG45 off-mode

- ariane133: 0.7669
- ariane136: 0.8192
- mempool_tile: 0.9610
- nvdla: 0.8203
- AVG: 0.8418
- all VALID / 0 overlaps

### Spacing-weight sweep

Weights tested:

- 0.005
- 0.01
- 0.02
- 0.05

Result: no selected NG45 placement changed. AVG remained 0.8418 for all weights.

### Repair run

Command/env:
`MPC_SPACING_WEIGHT=0.02 MPC_SPACING_REPAIR=1`

- AVG: 0.8417
- all VALID / 0 overlaps
- Interpretation: proxy-neutral, but no ORFS/WNS/TNS evidence yet.

### Decision

- Do not enable `MPC_SPACING_WEIGHT` by default.
- Keep `MPC_SPACING_REPAIR` opt-in.
- Next optimization focus: opt-in soft-macro recentering / cheap proxy improvement.

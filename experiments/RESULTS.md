# Macro Placement Experiment Results

## Current safe baseline

Branch: spacing-calibration  
Full IBM AVG proxy: 1.4938  
Overlaps: 0  
Validity: all VALID  

NG45 AVG proxy: 0.8418  
NG45 validity: all VALID / 0 overlaps  

Decision: this is the current fallback submission.

## Spacing calibration

Result:
- `MPC_SPACING_WEIGHT` at 0.005, 0.01, 0.02, 0.05 did not change selected NG45 placements.
- NG45 AVG stayed 0.8418.
- `MPC_SPACING_REPAIR=1` was proxy-neutral: NG45 AVG 0.8417.

Decision:
- Do not enable spacing weight by default.
- Keep repair opt-in only.

## Soft recenter experiment

Result:
- ibm01 default: 1.1355
- ibm01 with `MPC_SOFT_RECENTER=1`: 1.1355
- attempted to move 894 soft macros
- proxy worsened from 1.135459 to 1.163466, so the move was rejected
- NG45 unchanged at AVG 0.8418

Decision:
- Do not make default.
- Do not tune further.

## Extra candidates experiment

Detailed summary: `experiments/extra_candidates_20260519_162255/README.md`

Result:
- ibm01: 1.1355 -> 1.1355
- NG45 AVG: 0.8418 -> 0.8418
- IBM sample AVG: 1.451533 -> 1.451517
- runtime increased about 1.52x

Decision:
- Do not make default.
- Not worth full IBM run unless idle.

## Current hypothesis

The placer is generation-limited. It likely does not produce strong enough candidates. Next promising direction is to integrate an RL/RL-inspired policy as an optional candidate generator, then let true proxy choose whether it wins.

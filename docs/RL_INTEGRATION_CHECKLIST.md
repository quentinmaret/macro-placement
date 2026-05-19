# RL Integration Checklist

## Goal

Integrate RL or RL-inspired placement work only as an optional candidate generator inside the existing final tournament placer. RL should provide additional candidate placements for `FinalMacroPlacer` to judge; it should not replace the final placer, change the true proxy objective, or bypass the existing repair, legality, scoring, and winner-selection paths.

The current working hypothesis is that the placer is generation-limited. Small post-processing experiments did not materially improve proxy, so the practical role for RL is to generate stronger candidate placements and let the existing true proxy score decide whether any RL candidate wins.

Tier 1 proxy ranking remains the entry gate. Tier 2 feasibility and scoring depend on OpenROAD/ORFS results, where WNS and TNS feasibility matter before area, and final scoring weights WNS, TNS, and area in that priority order. RL candidates should therefore improve or preserve proxy without creating placements that are fragile for NG45/ORFS.

Default behavior must remain unchanged unless full validation proves an improvement.

## Required RL Policy Interface

The RL code must provide:

- A callable function or class that can be invoked from `submissions/final/placer.py`.
- A deterministic seed option.
- Inference-only mode.
- No training during final evaluation.
- No fragile new dependencies.
- No required GPU unless GPU availability is guaranteed in the evaluation environment.
- Output compatible with the `FinalMacroPlacer` placement format: a `torch.Tensor` shaped like `(benchmark.num_macros, 2)`.
- Hard macro positions that are valid or repairable by existing final placer legalization paths.
- Reasonable hard macro clearance for NG45/ORFS when possible; Tier 2 may snap coordinates and push macros apart to maintain approximately 12 um clearance.
- Fixed macro positions preserved or restorable by existing repair logic.
- Runtime acceptable for use as 1-3 extra tournament candidates.
- Clear policy name and configuration metadata when available.
- Reproducible output with a fixed seed.

## Integration Plan

Add an opt-in environment variable:

```bash
MPC_RL_CANDIDATES=1
```

When `MPC_RL_CANDIDATES` is unset:

- Existing candidate generation is unchanged.
- Existing deterministic behavior is unchanged.
- Existing validation results remain the baseline.

When `MPC_RL_CANDIDATES=1`:

- Generate 1-3 RL candidates.
- Treat each RL placement as another tournament candidate source.
- Use existing repair/legalization paths before validity and scoring.
- Score with the existing true proxy path used by current candidates.
- Allow an RL candidate to win only if it is valid and better than the current best candidate.
- Log the candidate source as `rl_policy`.
- Keep all RL behavior behind the env var until default-on criteria are met.

## Required Logging Fields

When `MPC_FINAL_LOG=1`, RL candidate records should include:

- `rl_candidate_enabled`
- `rl_seed`
- `rl_policy_name`
- `rl_policy_config` when available
- `candidate_proxy`
- `candidate_wl`
- `candidate_density`
- `candidate_congestion`
- `candidate_overlap_count`
- `valid`
- `became_winner`
- `runtime_seconds`

Use existing candidate logging conventions where possible. Logging must not affect candidate selection.

## Validation Gates

Start with compile/import checks:

```bash
uv run python -m py_compile submissions/final/placer.py
```

Run ibm01 baseline and RL comparison:

```bash
uv run evaluate submissions/final/placer.py -b ibm01
MPC_RL_CANDIDATES=1 uv run evaluate submissions/final/placer.py -b ibm01
```

Run NG45 baseline and RL comparison:

```bash
uv run evaluate submissions/final/placer.py --ng45
MPC_RL_CANDIDATES=1 uv run evaluate submissions/final/placer.py --ng45
```

Run IBM sample baseline and RL comparison:

```bash
uv run evaluate submissions/final/placer.py -b ibm01
MPC_RL_CANDIDATES=1 uv run evaluate submissions/final/placer.py -b ibm01

uv run evaluate submissions/final/placer.py -b ibm03
MPC_RL_CANDIDATES=1 uv run evaluate submissions/final/placer.py -b ibm03

uv run evaluate submissions/final/placer.py -b ibm06
MPC_RL_CANDIDATES=1 uv run evaluate submissions/final/placer.py -b ibm06

uv run evaluate submissions/final/placer.py -b ibm09
MPC_RL_CANDIDATES=1 uv run evaluate submissions/final/placer.py -b ibm09

uv run evaluate submissions/final/placer.py -b ibm12
MPC_RL_CANDIDATES=1 uv run evaluate submissions/final/placer.py -b ibm12

uv run evaluate submissions/final/placer.py -b ibm17
MPC_RL_CANDIDATES=1 uv run evaluate submissions/final/placer.py -b ibm17
```

Run full IBM only if the sample improves materially:

```bash
uv run evaluate submissions/final/placer.py --all
MPC_RL_CANDIDATES=1 uv run evaluate submissions/final/placer.py --all
```

For all runs, record proxy score, validity, overlap count, runtime, and whether any RL candidate became the winner.

For NG45 runs, also check for ORFS robustness risk where available: WNS/TNS direction, area direction, OpenROAD failures, and whether spacing push/snap would materially perturb the submitted placement.

## Default-On Criteria

Make RL default only if:

- Full IBM AVG proxy improves.
- All designs are `VALID`.
- Hard macro overlaps remain 0.
- Runtime is acceptable.
- No fragile dependency issue is introduced.
- NG45 does not regress badly.
- NG45/ORFS feasibility risk does not increase materially; WNS and TNS remain the primary Tier 2 concerns.
- Results reproduce with fixed seed.
- The default-off path remains unchanged.

## Rejection Criteria

Do not integrate, or do not make default, if any of these hold:

- RL produces invalid placements that existing repair cannot reliably recover.
- RL produces hard macro overlaps after repair/legalization.
- RL needs training during evaluation.
- RL requires unsupported dependencies.
- RL requires GPU when GPU is not guaranteed.
- RL improves one design but hurts the IBM sample average.
- RL causes unacceptable runtime growth.
- RL results cannot be reproduced with a fixed seed.
- RL requires benchmark-specific hardcoded placements or hidden-design assumptions.
- RL improves proxy while obviously worsening NG45 timing robustness or causing ORFS failures.

## Meeting Questions For RL Teammate

- What benchmarks has the policy run on?
- What proxy scores did it achieve?
- Were all outputs valid with 0 hard macro overlaps?
- What is inference runtime per benchmark?
- Does inference need GPU?
- Does inference need saved weights?
- Are saved weights small enough and stable enough for submission use?
- Is inference deterministic with a fixed seed?
- What exact file/function/class should the final placer call?
- What placement format does the policy return?
- What dependencies does it add?

## Final Deliverable Format

The first RL integration patch should be small and feature-flagged:

- Optional env var only: `MPC_RL_CANDIDATES=1`.
- No default behavior change.
- No proxy scoring changes.
- No evaluator changes.
- No wholesale final placer rewrite.
- Candidate logging included when `MPC_FINAL_LOG=1`.
- Validation results attached to the patch discussion.

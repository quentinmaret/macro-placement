# Extra Candidates Experiment

Branch: `extra-candidates-experiment`

## Files Changed

- `submissions/final/placer.py`

## Exact Candidates Added

All additions are gated behind `MPC_EXTRA_CANDIDATES=1`; unset behavior is intended to remain unchanged.

For designs with more than 320 hard macros:
- `extra:will_seed_7`

For designs with 320 or fewer hard macros:
- `extra:will_seed_409`

For designs with 220 or fewer hard macros:
- `extra_chain:pin_aware,analytical_stage1,legalize`
- `extra_chain:peripheral,macro_spread,legalize`
- `extra_chain:anchor,analytical_stage1,legalize,local_shift`

For designs with 221-280 hard macros:
- `extra_chain:pin_aware,analytical_stage1,legalize`
- `extra_chain:anchor,analytical_stage1,legalize,local_shift`

For designs with 281-320 hard macros:
- `extra_chain:anchor,analytical_stage1,legalize,local_shift`

The `local_shift` operator remains off by default and is only enabled for extra-candidate mode, with `iterations=2`, `require_legal=True`, and `shift_fraction=0.35`.

## Logging Added

When `MPC_FINAL_LOG=1`, candidate JSONL records now include:
- `extra_candidates_enabled`
- `candidate_source`
- `candidate_chain`
- `candidate_seed`
- `candidate_proxy`
- `candidate_wl`
- `candidate_density`
- `candidate_congestion`
- `candidate_overlap_count`

Logging was spot-checked on `ibm12`; the winner was `extra:will_seed_7` with exact proxy `1.6526631116867065`, 0 overlaps, and all candidate metadata fields populated.

## Validation Commands Run

```bash
python3 -m py_compile submissions/final/placer.py
uv run python -m py_compile submissions/final/placer.py

uv run evaluate submissions/final/placer.py -b ibm01 2>&1 | tee experiments/extra_candidates_20260519_162255/ibm01_default.txt
MPC_EXTRA_CANDIDATES=1 uv run evaluate submissions/final/placer.py -b ibm01 2>&1 | tee experiments/extra_candidates_20260519_162255/ibm01_extra_candidates.txt

uv run evaluate submissions/final/placer.py --ng45 2>&1 | tee experiments/extra_candidates_20260519_162255/ng45_default.txt
MPC_EXTRA_CANDIDATES=1 uv run evaluate submissions/final/placer.py --ng45 2>&1 | tee experiments/extra_candidates_20260519_162255/ng45_extra_candidates.txt

uv run evaluate submissions/final/placer.py -b ibm03 2>&1 | tee experiments/extra_candidates_20260519_162255/ibm03_default.txt
MPC_EXTRA_CANDIDATES=1 uv run evaluate submissions/final/placer.py -b ibm03 2>&1 | tee experiments/extra_candidates_20260519_162255/ibm03_extra_candidates.txt

for b in ibm06 ibm09 ibm12 ibm17; do
  uv run evaluate submissions/final/placer.py -b "$b" 2>&1 | tee "experiments/extra_candidates_20260519_162255/${b}_default.txt"
  MPC_EXTRA_CANDIDATES=1 uv run evaluate submissions/final/placer.py -b "$b" 2>&1 | tee "experiments/extra_candidates_20260519_162255/${b}_extra_candidates.txt"
done

MPC_EXTRA_CANDIDATES=1 MPC_FINAL_LOG=1 MPC_FINAL_LOG_DIR=experiments/extra_candidates_20260519_162255/logs_ibm12_extra \
  uv run evaluate submissions/final/placer.py -b ibm12 2>&1 | tee experiments/extra_candidates_20260519_162255/ibm12_extra_candidates_logging.txt
```

## IBM01 Delta

| Mode | Proxy | Runtime | Validity |
|---|---:|---:|---|
| default | 1.1355 | 95.11s | VALID, 0 overlaps |
| extra candidates | 1.1355 | 127.33s | VALID, 0 overlaps |

Delta: `+0.0000` proxy, `+32.22s`.

## NG45 Delta

| Mode | AVG Proxy | Runtime | Validity |
|---|---:|---:|---|
| default | 0.8418 | 505.60s | all VALID, 0 overlaps |
| extra candidates | 0.8418 | 646.52s | all VALID, 0 overlaps |

Delta: `+0.0000` AVG proxy, `+140.92s`.

Per-design NG45 proxies were unchanged:
- `ariane133`: 0.7669 -> 0.7669
- `ariane136`: 0.8192 -> 0.8192
- `mempool_tile`: 0.9610 -> 0.9610
- `nvdla`: 0.8203 -> 0.8203

## IBM Sample Delta

| Benchmark | Default | Extra | Delta | Runtime Default | Runtime Extra | Validity |
|---|---:|---:|---:|---:|---:|---|
| ibm01 | 1.1355 | 1.1355 | +0.0000 | 95.11s | 127.33s | VALID, 0 overlaps |
| ibm03 | 1.3636 | 1.3636 | +0.0000 | 54.37s | 70.77s | VALID, 0 overlaps |
| ibm06 | 1.6719 | 1.6719 | +0.0000 | 70.68s | 100.07s | VALID, 0 overlaps |
| ibm09 | 1.1361 | 1.1361 | +0.0000 | 100.63s | 147.36s | VALID, 0 overlaps |
| ibm12 | 1.6528 | 1.6527 | -0.0001 | 30.43s | 59.20s | VALID, 0 overlaps |
| ibm17 | 1.7493 | 1.7493 | +0.0000 | 70.62s | 137.57s | VALID, 0 overlaps |

Sample average:
- default: `1.451533`
- extra candidates: `1.451517`
- delta: `-0.000017`

Runtime impact:
- default sample total: `421.84s`
- extra sample total: `642.30s`
- ratio: `1.52x`

## Full IBM

Full IBM was not run. The limited sample was safe but not promising enough: only `ibm12` improved, and only by `0.0001` at printed precision, while runtime increased substantially.

## Recommendation

Keep opt-in or abandon. Do not make this default.

The experiment preserved validity and found one tiny IBM improvement, but the gain is much too small relative to the runtime cost. If time is tight, this is not worth spending a full IBM run on unless a future candidate set is more targeted.

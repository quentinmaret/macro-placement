# Model Descriptions

---

### ``core.py``

#### Description

Copyable legality-first baseline.

This file is the default starting point for new models. It starts from the
initial benchmark placement, legalizes hard macros with small moves, keeps soft
macros unchanged, and leaves a clear refinement hook for future work.

---

### ``rl_local_policy.py``

#### Description

RL-style local policy template.

This file keeps the same legality-first workflow as `core.py`, but changes the
refinement stage into a sequential policy over local macro moves. It is meant
to show what an RL-flavored model looks like in this repo before we build a
full training stack.

#### Notes

- Starts from the initial placement and legalizes first.
- Builds a hard-macro graph from benchmark nets.
- Visits movable hard macros one by one.
- Generates local legal actions for each macro.
- Scores those actions with an interpretable policy over simple features.
- Keeps the structure easy to replace with a learned policy later.

---

### ``initializer.py``

#### Description

Initializer registry, manual pipeline runner, and ensemble initializer template.

This file focuses on generating strong starting placements rather than running
an extended local search policy. It keeps the same copyable model structure as
the other templates, but now also exposes a practical initializer/operator
registry for short manually specified chains.

#### Notes

- Includes the raw benchmark placement as a safe baseline candidate.
- Keeps the original graph clustering / hierarchy initializer and registers it
  as `hierarchical`, with aliases `hierarchical_partition` and `graph_hierarchy`.
- Adds common fast macro-placement seeds and refinements: grid, random spread,
  pin-aware shifting, peripheral placement, spectral ordering, force smoothing,
  macro spreading, legalization, local swap, and local shift.
- Legalization still uses the shared `core.py` scaffold.
- Chain execution is deterministic under a provided seed.
- The current experiment script tests individual operators and manually chosen
  chains only. It does not search for an optimal strategy.

#### Initializer Registry

`initializer.py` defines:

- `InitializerOperator`: a small wrapper with `name`, `kind`, and `run(...)`.
- `INITIALIZER_REGISTRY`: a name-to-operator registry.
- `run_initializer_chain(problem, chain, seed=0, config=None, collect_metrics=True, plc=None, placement=None)`.

Operator kinds are:

- `seed`: creates a placement from scratch.
- `transform`: modifies an existing placement.
- `repair`: legality/bounds repair.
- `refine`: small-budget local improvement.

The first operator in a chain must be a `seed` unless an existing `placement=`
is provided. Unknown operator names fail with a clear list of available names.

Chain syntax accepts a comma-separated string or a list:

```python
run_initializer_chain(benchmark, "hierarchical,legalize", seed=0, plc=plc)
run_initializer_chain(benchmark, ["grid", "force_smooth", "legalize"], seed=0)
```

Supported examples:

```text
hierarchical
hierarchical,pin_aware_shift,legalize
hierarchical,spectral_order,legalize,local_swap
peripheral,legalize
```

Per-stage metadata includes runtime, proxy cost when a `PlacementCost` object is
available, normalized wirelength/HPWL proxy, overlap metrics, boundary
violations, legality, and legalization damage when it can be computed.

Config is a nested dictionary keyed by operator name:

```python
config = {
    "chain_budget_sec": 6.0,
    "stage_budget_sec": 4.0,
    "force_smooth": {"iterations": 30, "attraction": 0.12},
    "local_swap": {"iterations": 120},
    "placer": {"safety_gap": 0.05},
}
```

`run_initializer_chain(...)` records per-stage metadata when
`collect_metrics=True`: operator name, canonical operator name, kind, runtime,
macro count, proxy metrics when `plc` is available, overlap metrics, boundary
violations, legal status, and legalization damage when a `legalize` stage can
be compared against the previous proxy score. Metric collection is best-effort:
metric failures are saved in `metrics_error` and do not stop the chain.

The chain runner also supports soft budget protection. `chain_budget_sec` caps
the whole manual chain and `stage_budget_sec` is passed to budget-aware
operators. Budgets are checked between stages and inside the iterative
operators where practical. When a budget is exceeded after a placement exists,
the runner returns the current placement and sets `budget_stopped=True`.

#### Operators

| Name | Kind | Purpose | Key Config | Strengths | Weaknesses |
|------|------|---------|------------|-----------|------------|
| `hierarchical` | seed | Existing graph-clustering / recursive region initializer. | `placer.max_cluster_size`, `placer.min_cluster_size_to_split` | Connectivity-aware and usually structured. | Raw seed may still need `legalize`. |
| `hierarchical_partition` | seed alias | Alias for `hierarchical`. | Same as `hierarchical` | Keeps old hierarchy naming discoverable. | Same as `hierarchical`. |
| `graph_hierarchy` | seed alias | Alias for the original method name. | Same as `hierarchical` | Preserves existing behavior hooks. | Same as `hierarchical`. |
| `random_spread` | seed | Shuffled coarse-grid baseline with deterministic jitter. | `jitter_fraction` | Good reproducible baseline and stress test. | Not connectivity-aware. |
| `grid` | seed | Uniform lattice placement, larger macros first. | None currently | Stable deterministic baseline. | Can ignore IO/connectivity. |
| `pin_aware` | seed | Grid seed followed by IO/pin attraction. | `strength` | Useful when port locations are available. | Falls back to approximate net-neighbor targets when explicit ports are absent. |
| `peripheral` | seed | Boundary-biased placement for large, high-degree, or IO-heavy macros. | `boundary_fraction`, `boundary_count`, `interior_margin` | Often sensible for IO-dominated designs. | Side assignment is heuristic. |
| `pin_aware_shift` | transform | Nudge an existing placement toward connected IO sides. | `strength` | Cheap cleanup after hierarchy/grid seeds. | Can increase density near edges without `legalize`. |
| `spectral_order` | transform | Reassign existing slots by a Fiedler-vector macro ordering. | `max_macros`, `raise_on_skip` | Fast ordering heuristic on smaller cases. | Skips large macro counts by default; it is not a full spectral/QAP placer. |
| `force_smooth` | transform | Fixed-budget connectivity attraction plus overlap/boundary repulsion. | `iterations`, `attraction`, `repulsion`, `max_move` | Lightweight smoothing before legalization. | May still leave overlaps. |
| `analytical_smooth` | transform alias | Alias for `force_smooth`. | Same as `force_smooth` | Easier name for analytical-style experiments. | Same as `force_smooth`. |
| `macro_spread` | transform | Push overlapping/dense hard macros apart. | `iterations`, `strength`, `max_move` | Reduces obvious overlap and crowding. | Not a complete legalizer. |
| `legalize` | repair | Existing conservative `CorePlacer` legalization. | `placer.search_radii`, `placer.step_scale`, `placer.safety_gap` | Reuses known legality repair path. | Can damage wirelength if the seed is very tangled. |
| `local_swap` | refine | Small random greedy macro-position swaps. | `iterations`, `require_legal`, `use_proxy` | Cheap post-legalization refinement. | Uses graph surrogate by default; proxy scoring is slower and requires `plc`. |
| `local_shift` | refine | Greedy cardinal-direction local shifts. | `iterations`, `shift_fraction`, `require_legal`, `use_proxy` | Simple local cleanup. | Small move set; not a full optimizer. |

#### How To Use

Run it on one benchmark:

```bash
uv run evaluate submissions/models/initializer.py -b ibm01
```

Run it on all IBM benchmarks:

```bash
uv run evaluate submissions/models/initializer.py --all
```

What it does at runtime:

1. Builds an ensemble of initialization candidates.
2. Includes the original benchmark placement as a fallback candidate.
3. Builds a graph hierarchy candidate from hard-macro connectivity.
4. Legalizes every candidate with the shared `core.py` logic.
5. Scores the legal candidates and returns the best one.

The direct evaluator path preserves the existing ensemble-style submission
behavior. The registry/chain API is available for experiments and future
integration.

#### Experiment Script

`scripts/test_initializers.py` runs short manual chains on one benchmark. Use
the repo environment:

```bash
uv run python scripts/test_initializers.py --benchmark ibm01
uv run python scripts/test_initializers.py --benchmark ibm01 --chains "hierarchical,legalize" "hierarchical,macro_spread,legalize"
uv run python scripts/test_initializers.py --benchmark ibm01 --default-suite
uv run python scripts/test_initializers.py --benchmark ibm01 --output results/initializer_ibm01.jsonl
uv run python scripts/test_initializers.py --benchmark ibm01 --debug
```

Plain `python scripts/test_initializers.py ...` also works when the project
dependencies are installed in the active Python environment. The script accepts
IBM shorthand such as `ibm1` and normalizes it to `ibm01`.

If no `--chains` are provided, the default suite is:

```text
hierarchical,legalize
hierarchical,macro_spread,legalize
grid,legalize
peripheral,legalize
```

For high macro counts, that default suite shrinks automatically unless
`--force-full-suite` is passed. The script prints a compact table with chain,
legal status, proxy cost, overlap count, runtime, final operator, and budget
stop status. If `--output` is provided, each chain result is written as JSONL.
Proxy cost, density, and congestion require loading from benchmark source so a
`PlacementCost` object is available; `.pt` loads still report geometric
legality and overlap metrics.

#### Runtime-Aware Tournament Workflow

WillSeed is the current strong baseline. `submissions/will_seed/placer.py`
legalizes the initial hard-macro placement, extracts macro connectivity when a
`.plc` can be loaded, and runs a deterministic simulated-annealing-style
refinement with overlap rejection. It is the candidate to beat.

`submissions/final/placer.py` wraps WillSeed in a deterministic tournament.
For benchmarks with at most 320 hard macros, it scores several WillSeed starts,
the `EnsembleInitializerPlacer`, and a small macro-count-gated set of manual
initializer chains. Each candidate is repaired, checked for zero hard-macro
overlap, scored with the true proxy objective when a `PlacementCost` object is
available, and compared against the current best. For benchmarks with more
than 320 hard macros, the tournament only runs the proven WillSeed baseline.

`EnsembleInitializerPlacer` is still the direct evaluator-facing initializer
submission: it generates its built-in ensemble, legalizes candidates, and
selects the best with a graph-aware surrogate. Manual chains are different:
they are explicit operator strings such as `hierarchical,macro_spread,legalize`
used as cheap, diverse tournament candidates and experiments. A chain is worth
keeping only if its win rate or proxy improvement justifies its runtime.

The current final placer chain portfolio is intentionally macro-count-aware:

| Hard macros | Manual chains |
|------:|------|
| `n <= 220` | `hierarchical,macro_spread,legalize`; `hierarchical,legalize,local_swap`; `peripheral,legalize` |
| `221 <= n <= 280` | `hierarchical,macro_spread,legalize`; `hierarchical,legalize` |
| `281 <= n <= 320` | `hierarchical,legalize` |
| `n > 320` | none |

Operator budgets shrink with macro count:

| Hard macros | macro_spread | force_smooth | local_swap | legalizer search radii |
|------:|------:|------:|------:|------:|
| `n <= 220` | 18 | 18 | 80 | 150 |
| `221 <= n <= 280` | 10 | 8 | 30 | 100 |
| `n > 280` | 4 | 4 | 0 | 60 |

The final tournament can write candidate-level JSONL without spamming stdout:

```bash
uv run python scripts/test_tournament.py --benchmark ibm01 --output results/tournament_ibm01.jsonl --debug
FINAL_PLACER_LOG_PATH=results/tournament_ibm01.jsonl uv run evaluate submissions/final/placer.py -b ibm01
```

Each tournament JSONL row includes the benchmark, hard macro count, candidate
name, candidate type (`will_seed`, `initializer_ensemble`, `chain`, or
`fallback`), generation/repair/scoring/total runtime, validity, overlap count,
score, whether it became the current best, whether it was the final winner,
and any recorded error. Use these logs to spot slow candidates, chains that
never win, candidates that need too much repair, and operator stages that
consume budget without improving the tournament result.

This is a manual/runtime-aware workflow, not automated chain optimization.
Future work should build on the registry and metrics with automated chain
search, beam search over operator sequences, successive halving, and
marginal-value pruning.

#### How To Extend

To add another initializer/operator method:

1. Open `submissions/models/initializer.py`.
2. Implement a small helper method or top-level runner.
3. Register it with `register_initializer_operator(InitializerOperator(...))`.
4. Use kind `seed`, `transform`, `repair`, or `refine`.
5. Keep randomness flowing through the provided `rng`.

The old ensemble hook still exists for direct evaluator use:

- `benchmark_anchor`
- `graph_hierarchy`

Future work should add automated chain search / beam search / successive
halving over the registered operators. That search is intentionally not
implemented yet.

---

## Summary Table

| Model | Main Idea | Status | ibm01 | All IBM | Runtime Notes | Keep Going? |
|------|------|------|------|------|------|------|
| `core.py` | Minimal-displacement legalization baseline | Ready | Pending | Pending | Designed for debugging and extension | Yes |
| `rl_local_policy.py` | Sequential RL-style local action policy over legal moves | Ready | Pending | Pending | Educational RL bridge before full training | Yes |
| `initializer.py` | Registry-backed initializer suite plus hierarchy ensemble | Ready | Pending re-test | Pending | Manual chains now available through `scripts/test_initializers.py` | Yes |
| `submissions/final/placer.py` | Deterministic proxy-scored tournament over Will Seed and initializer candidates | Final candidate | 1.2226 | 1.5071 | Zero overlaps; 657.40s total IBM runtime; worst case 131.19s on ibm18 | Tune runtime/quality tradeoffs |

---

## Final Submission: `submissions/final/placer.py`

#### Description

The final submission is a deterministic tournament placer. For small and
medium IBM benchmarks, it generates legal candidates from:

- the exact `WillSeedPlacer(seed=42, refine_iters=3000)` baseline,
- several additional deterministic Will Seed starts,
- the initializer ensemble,
- selected initializer chains using `macro_spread`, `force_smooth`, `legalize`,
  and `local_swap`.

Each candidate is repaired for hard-macro overlap if needed, scored with the
true proxy objective when the benchmark `.plc` can be loaded, and the lowest
zero-overlap placement is returned. For benchmarks with more than 320 hard
macros, the placer takes a fast path and returns the proven Will Seed baseline;
this avoids multi-minute initializer legalization on large designs such as
`ibm10`, `ibm12`, `ibm14`, and `ibm17`.

This uses practical ideas from WireMask-BBO and VeoPlace without implementing
their full stacks: optimize in a candidate/search space, keep legal placements
as a hard constraint, use greedy legalization/repair, keep an elite-style
portfolio of diverse starts, and let the real proxy objective choose among
legal candidates.

#### Verification

Setup and tests:

```bash
git submodule update --init external/MacroPlacement
uv sync
uv sync --extra dev
uv run python -m pytest
```

`uv run pytest` resolved to an environment without `macro_place` importability
in this workspace; `uv run python -m pytest` is the reliable invocation after
installing the dev extra. Result: 5 passed.

Baseline measurements:

```bash
uv run evaluate submissions/will_seed/placer.py -b ibm01
uv run evaluate submissions/will_seed/placer.py --all
uv run evaluate submissions/models/initializer.py -b ibm01
uv run python scripts/test_initializers.py --benchmark ibm01 --chains "hierarchical,legalize" "hierarchical,macro_spread,legalize" --debug
```

Observed:

- Will Seed `ibm01`: 1.2920, 0 overlaps.
- Will Seed all IBM average: 1.5336, 0 overlaps, 37.50s total.
- Initializer direct `ibm01`: 1.2226, 0 overlaps, 15.94s in the latest quick check.
- Initializer chain quick check on `ibm01`: `hierarchical,legalize` at 1.4877
  and `hierarchical,macro_spread,legalize` at 1.4530. Direct initializer
  ensemble remains better on this benchmark.

Final verification:

```bash
uv run evaluate submissions/final/placer.py -b ibm01
uv run evaluate submissions/final/placer.py --all
uv run python scripts/test_tournament.py --benchmark ibm01 --debug --output results/tournament_ibm01.jsonl
```

Latest `ibm01` quick check after runtime-aware instrumentation: final placer
1.2226, 0 overlaps, 37.07s; tournament diagnostics winner
`initializer_ensemble` at 1.2226. Latest full IBM suite after the cleanup:
average 1.5071, zero overlaps, 657.40s total.

Final all-IBM results:

| Benchmark | Final Proxy | Will Seed Proxy | Overlaps | Runtime |
|------|------:|------:|------:|------:|
| ibm01 | 1.2226 | 1.2920 | 0 | 35.17s |
| ibm02 | 1.6409 | 1.6798 | 0 | 51.92s |
| ibm03 | 1.4003 | 1.4043 | 0 | 33.42s |
| ibm04 | 1.3891 | 1.4478 | 0 | 36.86s |
| ibm06 | 1.7198 | 1.7965 | 0 | 38.76s |
| ibm07 | 1.4949 | 1.5903 | 0 | 43.43s |
| ibm08 | 1.5087 | 1.5877 | 0 | 66.33s |
| ibm09 | 1.1361 | 1.1625 | 0 | 44.46s |
| ibm10 | 1.4104 | 1.4116 | 0 | 23.11s |
| ibm11 | 1.2547 | 1.2547 | 0 | 6.40s |
| ibm12 | 1.6528 | 1.6528 | 0 | 19.61s |
| ibm13 | 1.4113 | 1.4113 | 0 | 7.78s |
| ibm14 | 1.6515 | 1.6515 | 0 | 23.52s |
| ibm15 | 1.6379 | 1.6379 | 0 | 14.84s |
| ibm16 | 1.5484 | 1.5484 | 0 | 27.91s |
| ibm17 | 1.7493 | 1.7493 | 0 | 52.67s |
| ibm18 | 1.7921 | 1.7921 | 0 | 131.19s |
| AVG | 1.5071 | 1.5336 | 0 | 657.40s total |

Worst final proxy benchmarks are `ibm18`, `ibm17`, `ibm06`, `ibm12`, `ibm14`,
and `ibm02`. Worst runtime is `ibm18`; it remains well under the one-hour
per-benchmark requirement.

#### Remaining Risks

- The small/medium search cutoff is deliberately conservative and general, but
  it leaves larger benchmarks at the Will Seed baseline.
- `ibm18` is under the runtime limit but much slower than the other small cases;
  future work should add a time budget or cheaper pre-filter before initializer
  chains.
- The final placer relies on loading `.plc` files to score candidates with the
  true proxy. If a future benchmark lacks a loadable `.plc`, it falls back to a
  graph/geometric surrogate.
- `macro_place.loader` creates a temporary parser-compatible netlist copy when
  IBM data contains signed scientific notation like `5.68434e-16`; this keeps
  `external/MacroPlacement` clean while preserving the original evaluator and
  scoring logic.

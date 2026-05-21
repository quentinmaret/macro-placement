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
violations, legality, legalization damage when it can be computed, and
Stage-1 readiness metrics such as coarse bin density, density overflow energy,
narrow-channel count, movable-macro spread, and crowding energy.

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
| `anchor` | seed | Clone the benchmark anchor placement as a chain seed. | None currently | Lets Stage-1 transforms improve the already strong benchmark anchor path. | Can over-preserve the input if no transform follows. |
| `benchmark_anchor` | seed alias | Alias for `anchor`. | Same as `anchor` | Exposes the direct ensemble fallback through the registry. | Same as `anchor`. |
| `random_spread` | seed | Shuffled coarse-grid baseline with deterministic jitter. | `jitter_fraction` | Good reproducible baseline and stress test. | Not connectivity-aware. |
| `grid` | seed | Uniform lattice placement, larger macros first. | None currently | Stable deterministic baseline. | Can ignore IO/connectivity. |
| `pin_aware` | seed | Grid seed followed by IO/pin attraction. | `strength` | Useful when port locations are available. | Falls back to approximate net-neighbor targets when explicit ports are absent. |
| `peripheral` | seed | Boundary-biased placement for macros with refined periphery desirability. | `boundary_fraction`, `boundary_count`, `interior_margin`, `area_weight`, `io_weight`, `degree_weight`, `centrality_penalty`, `min_side_gap_fraction` | Backward-compatible hard seed with capacity-aware side assignment. | Still a hard seed; it can be too disruptive versus soft Stage-1 transforms. |
| `pin_aware_shift` | transform | Nudge an existing placement toward connected IO sides. | `strength` | Cheap cleanup after hierarchy/grid seeds. | Can increase density near edges without `legalize`. |
| `spectral_order` | transform | Reassign existing slots by a Fiedler-vector macro ordering. | `max_macros`, `raise_on_skip` | Fast ordering heuristic on smaller cases. | Skips large macro counts by default; it is not a full spectral/QAP placer. |
| `periphery_bias` | transform | Softly move selected macros toward capacity-aware boundary targets. | `strength`, `boundary_fraction`, `boundary_count`, `area_weight`, `io_weight`, `degree_weight`, `centrality_penalty`, `min_side_gap_fraction`, `max_move_fraction` | Keeps existing placement topology and leaves legalization late. | Hurts when applied after already tangled seeds; best as a light anchor-stage bias. |
| `force_smooth` | transform | Fixed-budget connectivity attraction plus overlap/boundary repulsion. | `iterations`, `attraction`, `repulsion`, `max_move` | Lightweight smoothing before legalization. | May still leave overlaps. |
| `analytical_smooth` | transform alias | Alias for `force_smooth`. | Same as `force_smooth` | Preserves earlier analytical-style experiments. | Same as `force_smooth`; use `analytical_stage1` for the physical Stage-1 loop. |
| `analytical_stage1` | transform | Continuous physical relaxation with graph attraction, overlap/crowding repulsion, density overflow, boundary cleanup, and optional periphery attraction. | `iterations`, `attraction`, `overlap_repulsion`, `density_repulsion`, `boundary_repulsion`, `periphery_attraction`, `spread_repulsion`, `max_move`, `bin_count` | Improves refinement-agnostic physical quality without legalizing early. | Needs a good seed and budget; hierarchy starts were often worse than anchor starts. |
| `analytical_physical` | transform alias | Alias for `analytical_stage1`. | Same as `analytical_stage1` | More descriptive name for Stage-1 experiments. | Same as `analytical_stage1`. |
| `physical_smooth` | transform alias | Alias for `analytical_stage1`. | Same as `analytical_stage1` | Short alias for physical smoothing experiments. | Same as `analytical_stage1`. |
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

Stage-1 is now treated as a soft physical initializer rather than an early
legalizer. The strongest path starts from the benchmark `anchor`, applies
`analytical_stage1` to reduce overlap energy, bin overflow, boundary pressure,
and crowding, optionally adds a very light `periphery_bias`, and only then
commits through `macro_spread` plus `legalize`. This keeps the placement
refinement-agnostic: the transforms improve physical/topological quality but
do not assume WillSeed-specific repair behavior.

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

The current final placer chain portfolio is intentionally macro-count-aware.
The winning Stage-1 idea is not "start every seed over"; it is to gently
improve the strong benchmark anchor before late legalization. Hierarchy
Stage-1 and pin-aware/periphery Stage-1 chains were tested on `ibm01`; they
were legal but worse than the old hierarchy chains, so the final portfolio
keeps only the anchor Stage-1 chains plus the strongest old comparisons.

| Hard macros | Manual chains |
|------:|------|
| `n <= 220` | `hierarchical,macro_spread,legalize`; `hierarchical,legalize,local_swap`; `anchor,analytical_stage1,legalize`; `anchor,periphery_bias,analytical_stage1,macro_spread,legalize` |
| `221 <= n <= 280` | `anchor,analytical_stage1,legalize`; `anchor,periphery_bias,analytical_stage1,macro_spread,legalize`; `hierarchical,macro_spread,legalize`; `hierarchical,legalize` |
| `281 <= n <= 320` | `anchor,analytical_stage1,legalize` |
| `n > 320` | none |

Operator budgets shrink with macro count:

| Hard macros | macro_spread | analytical_stage1 | force_smooth | local_swap | legalizer search radii |
|------:|------:|------:|------:|------:|------:|
| `n <= 220` | 18 | 16 | 18 | 80 | 150 |
| `221 <= n <= 280` | 12 | 16 | 8 | 30 | 100 |
| `281 <= n <= 320` | 4 | 4 | 4 | 0 | 60 |
| `n > 320` | none | none | none | none | fast WillSeed path |

Final Stage-1 tournament config is intentionally conservative:

- `periphery_bias`: `strength=0.10`, `boundary_fraction=0.08`,
  `max_move_fraction=0.018`, high IO weight, and high centrality penalty.
- `analytical_stage1`: `attraction=0.025`, `overlap_repulsion=0.80`,
  `density_repulsion=0.05`, `boundary_repulsion=0.08`,
  `spread_repulsion=0.10`, and `periphery_attraction=0.0`.

The final tournament can write candidate-level JSONL without spamming stdout:

```bash
uv run python scripts/test_tournament.py --benchmark ibm01 --output results/tournament_ibm01.jsonl --debug
FINAL_PLACER_LOG_PATH=results/tournament_ibm01.jsonl uv run evaluate submissions/final/placer.py -b ibm01
```

Each tournament JSONL row includes the benchmark, hard macro count, candidate
name, candidate type (`will_seed`, `initializer_ensemble`, `chain`, or
`fallback`), generation/repair/scoring/total runtime, validity, overlap count,
score, whether it became the current best, whether it was the final winner,
and any recorded error. Chain rows also include the initializer-chain metadata
and final stage readiness metrics when available. Use these logs to spot slow
candidates, chains that never win, candidates that need too much repair, and
operator stages that consume budget without improving the tournament result.

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

### ``ml_param_tournament.py``

#### Description

Adaptive branch-family parameter tuner. It is a per-instance optimizer over
existing initializer/operator chains, not offline ML training and not broad
arbitrary chain search.

The failed first version optimized hierarchy / analytical chains that were
structurally weak on IBM01: the best hierarchy primary preset was around 1.4763
while the current final tournament was around 1.1355. The useful lesson was not
that IBM01 needs a one-off config; it was that parameter tuning only helps after
the search starts in a competitive branch family.

The active version therefore keeps a small registry of branch families and runs
them in stages:

1. always run `anchor,legalize` as a safe legal baseline,
2. probe a few representative presets per enabled branch family,
3. choose the current benchmark's best branch family or top few families,
4. run a small directed parameter sweep around that basin,
5. return the best legal post-legalize proxy candidate, falling back to the
   baseline if the tuner is illegal or worse.

The default registry includes:

- `anchor_stage1_spread`:
  `anchor,periphery_bias,analytical_stage1,macro_spread,legalize`.
- `anchor_stage1`:
  `anchor,analytical_stage1,legalize`.
- a large-macro WillSeed fallback for benchmarks where the Stage-1 chain
  portfolio is unsuitable.

WillSeed seed sweeps and the initializer ensemble are still present, but only in
`ML_PARAM_MODE=explore` or behind `ML_PARAM_ENABLE_BROAD_FAMILIES=1`; they do
not dominate the default runtime.

IBM01 discovered a strong `anchor_stage1_spread` low-gap / wide-legalizer basin
called `LOW_GAP_WIDE_LEGALIZER`. That preset is kept as a reusable branch-family
preset, not as an IBM01 benchmark gate. Directed variants then adjust legalizer
gap/radius/step, Stage-1 overlap/density/attraction, macro-spread strength, and
periphery softness around the current winning basin.

Ranking is:

1. legal placement,
2. lower post-legalize `proxy_cost` when a `.plc` is available,
3. overlap / boundary / fallback surrogate only when proxy is unavailable,
4. runtime only as a tie-breaker or hard-budget guardrail.

#### Modes And Controls

```bash
ML_PARAM_MODE=adaptive uv run evaluate submissions/models/ml_param_tournament.py -b ibm01
ML_PARAM_FAST_MODE=1 uv run evaluate submissions/models/ml_param_tournament.py -b ibm01
ML_PARAM_MODE=explore uv run evaluate submissions/models/ml_param_tournament.py -b ibm01
```

Important controls:

- `ML_PARAM_TOTAL_BUDGET_SEC`, default 480.
- `ML_PARAM_BRANCH_PROBE_BUDGET_SEC`, default 60.
- `ML_PARAM_SWEEP_BUDGET_SEC`, default 180.
- `ML_PARAM_MAX_BRANCH_FAMILIES`, default 1.
- `ML_PARAM_MAX_VARIANTS`, default 10.
- `ML_PARAM_ENABLE_BROAD_FAMILIES`, default off.
- `ML_PARAM_ENABLE_LOCAL_REFINEMENT`, default off because IBM01 logs showed
  local refinement hurt the best candidate.
- `ML_PARAM_USE_PREFIX_REUSE`, parsed but not currently used; the active
  directed variants usually change prefix stages, so a prefix cache would add
  complexity before it saves much time.

Diagnostics are written as JSONL under `runs/ml_param_tournament/` by default,
or to `ML_PARAM_TOURNAMENT_LOG_PATH` when set. Each row includes benchmark,
mode, stage, branch family, candidate name, chain, config, parent, variant
family, runtime, final proxy, overlap count, boundary violations, legality,
selected/accepted state, per-stage metrics, errors, and prefix-reuse status.

#### How To Run

```bash
ML_PARAM_MODE=adaptive \
ML_PARAM_TOURNAMENT_LOG_PATH=runs/ml_param_tournament/adaptive_ibm01.jsonl \
uv run evaluate submissions/models/ml_param_tournament.py -b ibm01

ML_PARAM_FAST_MODE=1 \
ML_PARAM_TOURNAMENT_LOG_PATH=runs/ml_param_tournament/fast_ibm01.jsonl \
uv run evaluate submissions/models/ml_param_tournament.py -b ibm01
```

For final-tournament branch attribution:

```bash
FINAL_PLACER_BRANCH_DEBUG=1 \
FINAL_PLACER_LOG_PATH=runs/ml_param_tournament/final_branch_debug_ibm01.jsonl \
uv run evaluate submissions/final/placer.py -b ibm01
```

To add a future branch family, add a `BranchFamily` entry in
`build_branch_families()` with a chain/callable, probe presets, and an optional
directed variant builder. Keep the default probe count small; use
`ML_PARAM_MODE=explore` for wider experiments.

Known limitations:

- The tuner only searches parameters around registered branch families.
- Directed variants are hand-written, not learned.
- Some useful branches, especially WillSeed, expose only a few tunable knobs.
- If proxy metrics are unavailable, fallback scoring uses legality, overlap,
  boundary, density, and wirelength-like surrogate metrics.
- Prefix reuse is intentionally deferred until legalizer-only sweeps justify
  the added state management.

Archived failed first attempt:

- Old logs from the hierarchy-first experiment were moved to
  `runs/archive/ml_param_tournament_hierarchy_failed/`.
- That archive is useful only as a negative result; it is not the active
  optimization direction.

IBM01 reference before this simplification:

- `anchor,legalize`: about 1.2226, zero overlaps, about 15s.
- current final tournament: about 1.1355, zero overlaps, about 58s.
- broad recentered optimizer: 1.1156, zero overlaps, about 675s.
- winning branch family: `anchor_stage1_spread`.
- winning broad-run candidate:
  `LEGALIZER_WIDE_MUT2_FOCUS_LOWER_SAFETY_GAP`.
- local refinement hurt the selected candidate, so it is disabled by default.

Simplified tuner check on 2026-05-20:

- adaptive IBM01: 1.1125, zero overlaps, about 242s evaluator runtime.
- fast IBM01: 1.1125, zero overlaps, about 96s evaluator runtime.
- selected IBM01 candidate:
  `LOW_GAP_WIDE_LEGALIZER_LEGALIZER_GAP_ZERO`.
- adaptive IBM02 smoke test: 1.6366, zero overlaps, about 194s evaluator
  runtime, selected `anchor_stage1` with a Stage-1 density-relaxed variant.
- The IBM02 run also caught a legality-ordering bug: legal candidates now beat
  illegal candidates before proxy cost is compared.

---

## Summary Table

| Model | Main Idea | Status | ibm01 | All IBM | Runtime Notes | Keep Going? |
|------|------|------|------|------|------|------|
| `core.py` | Minimal-displacement legalization baseline | Ready | Pending | Pending | Designed for debugging and extension | Yes |
| `rl_local_policy.py` | Sequential RL-style local action policy over legal moves | Ready | Pending | Pending | Educational RL bridge before full training | Yes |
| `initializer.py` | Registry-backed initializer suite plus Stage-1 physical transforms | Ready | 1.1355 via final chain | 1.4938 via final tournament | Manual chains and readiness metrics available through `scripts/test_initializers.py` | Yes |
| `ml_param_tournament.py` | Adaptive branch-family tuner over proven initializer chains | Experimental | 1.1125 | Pending | Fast mode reaches same IBM01 result in about 96s; logs to `runs/ml_param_tournament/` | Yes |
| `submissions/final/placer.py` | Deterministic proxy-scored tournament with anchor Stage-1 candidates | Final candidate | 1.1355 | 1.4938 | Zero overlaps; 796.85s total IBM runtime; worst case 134.00s on ibm18 | Tune runtime/quality tradeoffs |

---

## Final Submission: `submissions/final/placer.py`

#### Description

The final submission is a deterministic tournament placer. For small and
medium IBM benchmarks, it generates legal candidates from:

- the exact `WillSeedPlacer(seed=42, refine_iters=3000)` baseline,
- several additional deterministic Will Seed starts,
- the initializer ensemble,
- selected initializer chains using `anchor`, `periphery_bias`,
  `analytical_stage1`, `macro_spread`, `legalize`, and `local_swap`.

Each candidate is repaired for hard-macro overlap if needed, scored with the
true proxy objective when the benchmark `.plc` can be loaded, and the lowest
zero-overlap placement is returned. For benchmarks with more than 320 hard
macros, the placer takes a fast path and returns the proven Will Seed baseline;
this avoids multi-minute initializer legalization on large designs such as
`ibm10`, `ibm12`, `ibm14`, and `ibm17`.

This uses practical ideas from WireMask-BBO and VeoPlace without implementing
their full stacks: optimize in a candidate/search space, improve soft physical
quality before commitment, keep legal placements as a hard constraint, use
greedy legalization/repair, keep an elite-style portfolio of diverse starts,
and let the real proxy objective choose among legal candidates.

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
- Stage-1 initializer smoke on `ibm01` after tuning:
  `hierarchical,analytical_stage1,macro_spread,legalize` reached 1.4518,
  slightly better than `hierarchical,macro_spread,legalize` at 1.4530.
  `hierarchical,periphery_bias,analytical_stage1,legalize` reached 1.5207,
  and `pin_aware,periphery_bias,analytical_stage1,legalize` reached 1.6388;
  these legal but losing starts were not kept in the final tournament.
- Anchor Stage-1 tournament smoke on `ibm01`: winner
  `chain:anchor,periphery_bias,analytical_stage1,macro_spread,legalize` at
  1.1355 with 0 overlaps.

Final verification:

```bash
uv run evaluate submissions/final/placer.py -b ibm01
uv run evaluate submissions/final/placer.py --all
uv run python scripts/test_tournament.py --benchmark ibm01 --debug --output results/tournament_ibm01.jsonl
```

For repeated long all-IBM checks, the equivalent project-venv entrypoint was
used to avoid repeated `uv` environment churn:

```bash
MPLCONFIGDIR=/private/tmp/macro-placement-matplotlib \
FINAL_PLACER_LOG_PATH=results/tournament_stage1_all_pruned.jsonl \
.venv/bin/evaluate submissions/final/placer.py --all
```

Latest `ibm01` quick check after runtime-aware instrumentation: final placer
1.1355, 0 overlaps, 90.94s in the final all-IBM run; tournament diagnostics
winner `chain:anchor,periphery_bias,analytical_stage1,macro_spread,legalize`.
Latest full IBM suite after Stage-1 pruning: average 1.4938, zero overlaps,
796.85s total.

Previous final all-IBM results:

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

Stage-1 all-IBM results:

| Benchmark | Previous Final Proxy | New Final Proxy | Delta | Winner Candidate | Overlaps | Runtime |
|------|------:|------:|------:|------|------:|------:|
| ibm01 | 1.2226 | 1.1355 | -0.0871 | `chain:anchor,periphery_bias,analytical_stage1,macro_spread,legalize` | 0 | 90.94s |
| ibm02 | 1.6409 | 1.5998 | -0.0411 | `chain:anchor,periphery_bias,analytical_stage1,macro_spread,legalize` | 0 | 102.20s |
| ibm03 | 1.4003 | 1.3636 | -0.0367 | `chain:anchor,analytical_stage1,legalize` | 0 | 35.99s |
| ibm04 | 1.3891 | 1.3858 | -0.0033 | `chain:anchor,analytical_stage1,legalize` | 0 | 44.20s |
| ibm06 | 1.7198 | 1.6719 | -0.0479 | `chain:anchor,periphery_bias,analytical_stage1,macro_spread,legalize` | 0 | 44.95s |
| ibm07 | 1.4949 | 1.4866 | -0.0083 | `chain:anchor,analytical_stage1,legalize` | 0 | 46.09s |
| ibm08 | 1.5087 | 1.5061 | -0.0026 | `chain:anchor,analytical_stage1,legalize` | 0 | 69.94s |
| ibm09 | 1.1361 | 1.1361 | -0.0000 | `initializer_ensemble` | 0 | 64.84s |
| ibm10 | 1.4104 | 1.4104 | -0.0000 | `will_seed_baseline` | 0 | 23.54s |
| ibm11 | 1.2547 | 1.2547 | -0.0000 | `will_seed_baseline` | 0 | 6.42s |
| ibm12 | 1.6528 | 1.6528 | -0.0000 | `will_seed_baseline` | 0 | 19.19s |
| ibm13 | 1.4113 | 1.4113 | -0.0000 | `will_seed_baseline` | 0 | 7.71s |
| ibm14 | 1.6515 | 1.6515 | -0.0000 | `will_seed_baseline` | 0 | 23.11s |
| ibm15 | 1.6379 | 1.6379 | -0.0000 | `will_seed_baseline` | 0 | 13.58s |
| ibm16 | 1.5484 | 1.5484 | -0.0000 | `will_seed_baseline` | 0 | 25.90s |
| ibm17 | 1.7493 | 1.7493 | -0.0000 | `will_seed_baseline` | 0 | 44.24s |
| ibm18 | 1.7921 | 1.7919 | -0.0002 | `chain:anchor,analytical_stage1,legalize` | 0 | 134.00s |
| AVG | 1.5071 | 1.4938 | -0.0134 | mixed | 0 | 796.85s total |

Quick sample before the final all-IBM run (`ibm01`, `ibm02`, `ibm04`,
`ibm06`, `ibm09`) moved from 1.4217 average to 1.3858 average. The only
neutral sample case was `ibm09`, where `initializer_ensemble` stayed slightly
ahead of the new Stage-1 candidates.

Worst final proxy benchmarks remain `ibm18`, `ibm17`, `ibm06`, `ibm12`,
`ibm14`, and `ibm02`. Worst runtime is still `ibm18`; it remains well under
the one-hour per-benchmark requirement.

#### Remaining Risks

- The small/medium search cutoff is deliberately conservative and general, but
  it leaves larger benchmarks at the Will Seed baseline.
- Stage-1 improves the final average but increases total IBM runtime from
  657.40s to 796.85s. The pruned portfolio removed non-winning hierarchy
  Stage-1 chains; future work should add automatic marginal-value pruning.
- `ibm18` is under the runtime limit but remains much slower than the other
  small/medium cases; future work should add a cheaper pre-filter before
  anchor analytical chains.
- The final placer relies on loading `.plc` files to score candidates with the
  true proxy. If a future benchmark lacks a loadable `.plc`, it falls back to a
  graph/geometric surrogate.
- `macro_place.loader` creates a temporary parser-compatible netlist copy when
  IBM data contains signed scientific notation like `5.68434e-16`; this keeps
  `external/MacroPlacement` clean while preserving the original evaluator and
  scoring logic.

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
    "force_smooth": {"iterations": 30, "attraction": 0.12},
    "local_swap": {"iterations": 120},
    "placer": {"safety_gap": 0.05},
}
```

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
| `spectral_order` | transform | Reassign existing slots by a Fiedler-vector macro ordering. | None currently | Fast ordering heuristic for 246-537 macro cases. | It is not a full spectral/QAP placer and can be neutral or harmful. |
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

`test_initializers.py` runs individual initializers and short manual chains on
one benchmark. Use the repo environment:

```bash
uv run python test_initializers.py --benchmark ibm01
uv run python test_initializers.py --benchmark ibm01 --seed 0
uv run python test_initializers.py --benchmark ibm01 --initializers hierarchical,grid,random_spread,peripheral
uv run python test_initializers.py --benchmark ibm01 --chains hierarchical,legalize "hierarchical,pin_aware_shift,legalize" "grid,force_smooth,legalize"
uv run python test_initializers.py --benchmark ibm01 --output results/initializer_results_ibm01.jsonl
```

Plain `python test_initializers.py ...` also works when the project
dependencies are installed in the active Python environment. The script accepts
IBM shorthand such as `ibm1` and normalizes it to `ibm01`.

If no `--initializers` or `--chains` are provided, the default suite is:

```text
hierarchical
grid,legalize
random_spread,legalize
peripheral,legalize
hierarchical,pin_aware_shift,legalize
hierarchical,spectral_order,legalize
hierarchical,force_smooth,legalize
hierarchical,macro_spread,legalize
hierarchical,legalize,local_swap
```

The script prints a summary table and writes JSONL by default, or CSV when the
output path ends in `.csv`. Proxy cost, density, and congestion require loading
from the benchmark source so a `PlacementCost` object is available; `.pt` loads
still report geometric legality and overlap metrics.

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
| `initializer.py` | Registry-backed initializer suite plus hierarchy ensemble | Ready | Pending re-test | Pending | Manual chains now available through `test_initializers.py` | Yes |
| `submissions/final/placer.py` | Deterministic proxy-scored tournament over Will Seed and initializer candidates | Final candidate | 1.2226 | 1.5072 | Zero overlaps; 958.27s total IBM runtime; worst case 246.50s on ibm18 | Tune runtime/quality tradeoffs |

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
uv run python test_initializers.py --benchmark ibm01
```

Observed:

- Will Seed `ibm01`: 1.2920, 0 overlaps.
- Will Seed all IBM average: 1.5336, 0 overlaps, 37.50s total.
- Initializer direct `ibm01`: 1.2226, 0 overlaps, 24.14s.
- Initializer chain test best on `ibm01`: `hierarchical,macro_spread,legalize`
  at 1.4310; direct initializer ensemble was better on this benchmark.

Final verification:

```bash
uv run evaluate submissions/final/placer.py -b ibm01
uv run evaluate submissions/final/placer.py --all
```

Final all-IBM results:

| Benchmark | Final Proxy | Will Seed Proxy | Overlaps | Runtime |
|------|------:|------:|------:|------:|
| ibm01 | 1.2226 | 1.2920 | 0 | 57.47s |
| ibm02 | 1.6409 | 1.6798 | 0 | 106.99s |
| ibm03 | 1.4003 | 1.4043 | 0 | 87.05s |
| ibm04 | 1.3891 | 1.4478 | 0 | 75.29s |
| ibm06 | 1.7198 | 1.7965 | 0 | 57.08s |
| ibm07 | 1.4949 | 1.5903 | 0 | 90.97s |
| ibm08 | 1.5087 | 1.5877 | 0 | 130.82s |
| ibm09 | 1.1361 | 1.1625 | 0 | 73.43s |
| ibm10 | 1.4116 | 1.4116 | 0 | 5.46s |
| ibm11 | 1.2547 | 1.2547 | 0 | 1.70s |
| ibm12 | 1.6528 | 1.6528 | 0 | 4.29s |
| ibm13 | 1.4113 | 1.4113 | 0 | 2.02s |
| ibm14 | 1.6515 | 1.6515 | 0 | 4.72s |
| ibm15 | 1.6379 | 1.6379 | 0 | 2.59s |
| ibm16 | 1.5484 | 1.5484 | 0 | 3.00s |
| ibm17 | 1.7493 | 1.7493 | 0 | 8.90s |
| ibm18 | 1.7921 | 1.7921 | 0 | 246.50s |
| AVG | 1.5072 | 1.5336 | 0 | 958.27s total |

Worst final proxy benchmarks are `ibm18`, `ibm17`, `ibm06`, `ibm12`, `ibm14`,
and `ibm02`. Worst runtime is `ibm18`; it is still well under the one-hour
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

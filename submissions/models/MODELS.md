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

# `submissions/core`

This folder is the Phase 0 API layer. It does not reimplement placement logic. It wraps the existing `macro_place` utilities into a small, consistent interface for:

- `evaluate(placement, benchmark, plc)`
- `validate(placement, benchmark)`
- `visualize(placement, benchmark)`
- running a placer across benchmarks
- logging results
- saving placements and images
- seed/config management

## Files

- `types.py`: shared dataclasses, placement format helpers, seed helper
- `eval.py`: unified evaluation / validation / visualization wrappers
- `runner.py`: experiment runner and artifact logger
- `run_interactive.py`: prompt-driven script for testing any placer
- `__init__.py`: convenient imports

## Standard placement format

Use a PyTorch tensor with shape `[num_macros, 2]`.

- column `0`: `x` center
- column `1`: `y` center
- dtype: usually `torch.float32`

If you have a list or NumPy array, `as_placement_tensor(...)` converts it to the standard format.

## Quick usage

```python
from macro_place.loader import load_benchmark_from_dir
from submissions.core import evaluate, validate, visualize

benchmark, plc = load_benchmark_from_dir(
    "external/MacroPlacement/Testcases/ICCAD04/ibm01"
)

placement = benchmark.macro_positions.clone()

is_valid, violations = validate(placement, benchmark)
report = evaluate(placement, benchmark, plc)
visualize(placement, benchmark, save_path="ibm01.png", plc=plc)
```

`report` includes:

- `proxy_cost`
- `wirelength_cost`
- `density_cost`
- `congestion_cost`
- `overlap_count`
- validation fields: `valid`, `violations`

## Running a placer across benchmarks

```python
from pathlib import Path

from submissions.core import RunConfig, run_placer
from submissions.examples.greedy_row_placer import GreedyRowPlacer

config = RunConfig(
    benchmark_root=Path("external/MacroPlacement/Testcases/ICCAD04"),
    benchmark_names=["ibm01", "ibm02"],
    output_dir=Path("submissions/core/outputs/demo"),
    seed=123,
)

results = run_placer(GreedyRowPlacer(), config=config)
```

This creates:

- `placements/*.pt`
- `visualizations/*.png`
- `results.json`
- `config.json`

## Interactive script

If you want to test a placer without writing any Python, run:

```bash
python submissions/core/run_interactive.py
```

The script prompts you for:

- placer file path
- benchmark root
- benchmark names
- output directory
- seed
- overlap checking
- whether to save visualizations
- whether to save placements
- results log filename
- config filename

### Example session

```text
Path to placer Python file: submissions/examples/greedy_row_placer.py
Benchmark root [external/MacroPlacement/Testcases/ICCAD04]:
benchmark_names: ibm01,ibm02
Output directory [submissions/core/outputs/interactive_run]: submissions/core/outputs/greedy_demo
Seed [42]: 123
Check overlaps during validation [y/n, default=y]:
Save visualization PNGs [y/n, default=y]:
Save placement tensors [y/n, default=y]:
Results log filename [results.json]:
Config filename [config.json]:
```

After the run finishes, the script prints a short summary and writes all artifacts into the chosen output directory.

Notes:

- Press Enter to accept a default value.
- For `benchmark_names`, leave it blank to run every benchmark directory under `benchmark_root`.
- The placer file should contain a class with a `place(self, benchmark)` method.

## Loading a submission from a file

```python
from pathlib import Path

from submissions.core import RunConfig, load_placer_from_file, run_placer

placer = load_placer_from_file(
    Path("submissions/examples/greedy_row_placer.py")
)

results = run_placer(
    placer,
    RunConfig(benchmark_names=["ibm01"])
)
```

The loader instantiates the first class in the file that has a `place(self, benchmark)` method.

## Reproducibility

Use `RunConfig(seed=...)` or call `set_seed(...)` directly.

`set_seed(...)` seeds:

- Python `random`
- `numpy`
- `torch`

## Notes

- The actual legality, visualization, and scoring logic comes from `macro_place.utils` and `macro_place.objective`.
- The runner expects benchmark directories that contain `netlist.pb.txt` and usually `initial.plc`.
- The saved placement tensors are easy to reload later with `torch.load(...)`.
